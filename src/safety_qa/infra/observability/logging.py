"""構造化ログ（stdout への JSON Lines）の設定。

`configure_logging()` を呼ぶだけで、structlog 経由のログ（api/infra）と
標準 `logging` 経由のログ（core・uvicorn・外部ライブラリ）の両方が同じ JSON 形式で
標準出力に出る。保存・転送・ローテーションは一切持たない。AWS へ載せる際は
実行基盤側（ECS の awslogs ドライバ等）が stdout を回収するだけで済み、
アプリのコード変更を要さない設計。

uvicorn の `--log-config` には依存しない。`uv run uvicorn ...` でも
`python -m safety_qa.infra`（ETL 単体実行）でも、このモジュールを import した
時点で同じログ設定になる。
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from safety_qa.infra.settings import Settings

_CONFIGURED = False

# structlog 既定の dict_tracebacks は例外フレームのローカル変数をそのまま出力する
# （show_locals=True）。core の例外フレームには質問文・回答文がローカル変数として
# 乗りうり、D4（ログに本文を出さない）に反するため明示的に無効化した変換器を使う。
_EXCEPTION_RENDERER = structlog.processors.ExceptionRenderer(
    structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
)

_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    _EXCEPTION_RENDERER,
]


def _make_service_context_processor(
    *, service_name: str, environment: str
) -> structlog.typing.Processor:
    """全ログ行に `service` / `env` を付与する processor を作る。

    複数環境のログを 1 つの CloudWatch ロググループに集約したときの絞り込みに使う想定。
    """

    def add_service_context(
        logger: object, method_name: str, event_dict: structlog.typing.EventDict
    ) -> structlog.typing.EventDict:
        event_dict.setdefault("service", service_name)
        event_dict.setdefault("env", environment)
        return event_dict

    return add_service_context


def configure_logging(settings: Settings) -> None:
    """ルートロガー・structlog を JSON Lines（stdout）で構成する。複数回呼んでも冪等。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    service_context = _make_service_context_processor(
        service_name=settings.service_name, environment=settings.environment
    )

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            service_context,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: structlog.typing.Processor
    if settings.log_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)

    formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain: core / uvicorn 等、標準 logging を直接使う呼び出し元にも
        # 同じ前処理（trace_id 付与・extra の昇格・タイムスタンプ等）を適用する。
        foreign_pre_chain=[*_SHARED_PROCESSORS, structlog.stdlib.ExtraAdder(), service_context],
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.log_level)

    # uvicorn.access は自前の http.request.completed と重複するため無効化する。
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    # uvicorn / uvicorn.error は自前ハンドラを外し、root の JSON ハンドラに束ねる。
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    _CONFIGURED = True
