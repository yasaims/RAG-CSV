"""Ollama を用いた `ChatModel` 実装。

temperature / seed を明示的に固定する。プロトタイプ検証では既定値のまま
実行しており、tool calling の再現性を定量評価できないという課題が
残っていた（docs/stack-feasibility-report.md D-3）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import ollama
import structlog

from safety_qa.core.models import ToolCall
from safety_qa.core.prompts import build_system_prompt
from safety_qa.core.tools import TOOLS

logger = structlog.get_logger(__name__)


def _parse_arguments(raw: Any) -> dict[str, str]:
    """LLM のツール呼び出し引数を文字列辞書へ正規化する。

    JSON の `null` は「未指定」を表す。`str(None)` で文字列 "None" にすると
    実フィルタ値として扱われてしまう（例: severity="None" が enum 外エラーに
    なる、equipment="None" が LIKE '%None%' として実行される）ため、null と
    空文字は値ごと辞書から除外する。
    """
    parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return {str(k): str(v) for k, v in parsed.items() if v is not None and str(v).strip() != ""}


def _get(response: Any, key: str) -> Any:
    """`ollama.ChatResponse`（属性アクセス）とテスト用の dict の両方から値を取る。"""
    if isinstance(response, Mapping):
        return response.get(key)
    return getattr(response, key, None)


def _ns_to_ms(value: Any) -> float | None:
    return None if value is None else value / 1e6


def _round_or_none(value: float | None, ndigits: int = 1) -> float | None:
    return None if value is None else round(value, ndigits)


def _response_metrics(response: Any, num_ctx: int) -> dict[str, float | int | None]:
    """Ollama のレスポンスからトークン内訳を取り出す。

    `prompt_eval_count`/`eval_count`/`*_duration` は Ollama が既に返している
    値で、取得コストはゼロ。CPU 推論では「プロンプトが長すぎる」のか
    「生成が遅い」のかで打ち手が正反対になるため、`duration_ms`（wall-clock）
    だけでは区別できなかった問題に対応する。欠損時は None（フィールド自体は
    常に出力し、値の有無で判別できるようにする）。
    """
    prompt_tokens = _get(response, "prompt_eval_count")
    completion_tokens = _get(response, "eval_count")
    eval_duration = _get(response, "eval_duration")
    prompt_eval_ms = _ns_to_ms(_get(response, "prompt_eval_duration"))
    eval_ms = _ns_to_ms(eval_duration)
    load_ms = _ns_to_ms(_get(response, "load_duration"))

    prefill_ms = None
    if load_ms is not None or prompt_eval_ms is not None:
        # 非ストリーミング API のため厳密な TTFT ではなく近似値。モデル
        # ロード + プロンプト評価の合計を「最初のトークンが出るまでの目安」とする。
        prefill_ms = (load_ms or 0.0) + (prompt_eval_ms or 0.0)

    tokens_per_sec = (
        completion_tokens / (eval_duration / 1e9)
        if completion_tokens is not None and eval_duration
        else None
    )
    context_used_ratio = prompt_tokens / num_ctx if prompt_tokens is not None else None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_eval_ms": _round_or_none(prompt_eval_ms),
        "eval_ms": _round_or_none(eval_ms),
        "load_ms": _round_or_none(load_ms),
        "prefill_ms": _round_or_none(prefill_ms),
        "tokens_per_sec": _round_or_none(tokens_per_sec),
        "num_ctx": num_ctx,
        "context_used_ratio": (
            round(context_used_ratio, 3) if context_used_ratio is not None else None
        ),
    }


class OllamaChatModel:
    """`ChatModel` ポートの実装。プロセス分離された Ollama サーバーに接続する。"""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        num_thread: int,
        timeout_seconds: float,
        num_ctx: int,
        keep_alive: str,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._client = ollama.Client(host=host, timeout=timeout_seconds)
        self._model = model
        self._options = {"num_thread": num_thread, "temperature": 0, "seed": 0, "num_ctx": num_ctx}
        self._num_ctx = num_ctx
        # 既定（5分）でアイドル後にアンロードされると、次の質問がモデル再ロード
        # 待ちになる。"-1" 等を指定すると常駐させられる（SAFETY_QA_OLLAMA_KEEP_ALIVE）。
        self._keep_alive = keep_alive
        # システムプロンプトへ注入する「本日」の基準。常駐プロセス中に日付が
        # 変わるため、インスタンス生成時に固定せず呼び出しごとに取得する。
        self._today = today

    def _system_prompt(self) -> str:
        return build_system_prompt(self._today())

    def route(self, question: str) -> ToolCall | None:
        started_at = time.perf_counter()
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": question},
                ],
                tools=TOOLS,
                options=self._options,
                keep_alive=self._keep_alive,
            )
        except Exception as e:
            self._log_call_failed("route", started_at, e)
            raise
        self._log_call_completed("route", started_at, response)
        tool_calls = response["message"].get("tool_calls") or []
        if not tool_calls:
            return None
        # 複数呼び出しが返る場合があるが、1 質問 1 ツール呼び出しの設計のため
        # 先頭のみ採用する。
        fn = tool_calls[0]["function"]
        arguments = _parse_arguments(fn["arguments"])
        return ToolCall(name=fn["name"], arguments=arguments)

    def answer(self, prompt: str) -> str:
        # 2 パス目は tools を渡さない。ツール未呼び出し時に「呼び出すふりだけをする」
        # 不整合が観測されたため（レポート D-2）、能力として存在しないことを
        # メッセージ構成のレベルで一致させる。
        started_at = time.perf_counter()
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                options=self._options,
                keep_alive=self._keep_alive,
            )
        except Exception as e:
            self._log_call_failed("answer", started_at, e)
            raise
        self._log_call_completed("answer", started_at, response)
        return response["message"]["content"]

    def _log_call_completed(self, pass_name: str, started_at: float, response: Any) -> None:
        metrics = _response_metrics(response, self._num_ctx)
        logger.info(
            "llm.call.completed",
            pass_name=pass_name,
            model=self._model,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
            **metrics,
        )
        prompt_tokens = metrics["prompt_tokens"]
        if prompt_tokens is not None and prompt_tokens >= self._num_ctx:
            # Ollama は num_ctx を超えた分を黙って切り詰める。top_n 件の事故記録の
            # 一部が回答に届いていない可能性があることを示す（SPEC.md
            # 「障害を無言で握り潰さない」）。
            logger.warning(
                "llm.context_overflow",
                pass_name=pass_name,
                model=self._model,
                prompt_tokens=prompt_tokens,
                num_ctx=self._num_ctx,
            )

    def _log_call_failed(self, pass_name: str, started_at: float, exc: Exception) -> None:
        # core（rag_service.py）が AnswerUnavailableError に正規化する前の、
        # Ollama 側の生の失敗理由をここで残す。
        logger.warning(
            "llm.call.failed",
            pass_name=pass_name,
            model=self._model,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
            error_type=type(exc).__name__,
        )
