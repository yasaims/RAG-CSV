"""評価ハーネスの CLI エントリポイント。

用法:
    uv run python -m evals.run retrieval               # Ollama 不要。検索精度のみ（毎 PR 想定）
    uv run python -m evals.run llm --repeat 3           # 要 Ollama（対象パス変更 PR / 手動実行想定）
    uv run python -m evals.run llm --update-baseline    # 実測値を暫定ベースラインとして書き出す

出力は `var/eval/<suite>.json`（機械可読）と `var/eval/<suite>.md`（Job Summary 用）。
絶対要件違反、またはベースラインからの低下（`--update-baseline` 指定時を除く）があれば
終了コード 1 を返す。依存の取得失敗・データセット不整合・ベースラインとの比較不成立・
ゲート対象ケース0件等で評価そのものが実行できなかった／判定できなかった場合は
「評価不能」として区別し、終了コード 2 を返す（品質低下と混同しない。
SPEC.md「障害を無言で握り潰さない」）。評価不能の場合も
理由付きのレポートを必ず書き出す（Job Summary に理由不明の赤だけが残ることを防ぐ）。
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import anyio
import duckdb

from evals.baseline import (
    Baseline,
    InvalidBaselineError,
    baseline_path,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)
from evals.conditions import MeasurementConditions, corpus_version
from evals.harness import RecordingChatModel
from evals.metrics import (
    arguments_match,
    extract_malformed_id_like_tokens,
    find_fabricated_ids,
    has_count_mismatch,
    has_false_zero_claim,
    has_unsupported_fabrication,
    recall_at_k,
    tool_matches,
)
from evals.report import CaseResult, InconclusiveReason, SuiteReport, write_report
from evals.schema import (
    EvalCase,
    dataset_version,
    load_dataset,
    validate_expected_degraded_reason,
    validate_must_hit_ids_exist,
)
from evals.suites import SUITES
from safety_qa.core.exceptions import (
    InvalidToolArgumentsError,
    InvalidToolCallError,
    QaServiceError,
)
from safety_qa.core.ports import ChatModel
from safety_qa.core.prompts import split_answer_basis
from safety_qa.core.rag_service import RagQaService, truncated_count
from safety_qa.core.tools import (
    AGGREGATE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    parse_aggregate_tool_arguments,
    parse_search_arguments,
)
from safety_qa.infra.container import SearchStack, build_chat_model, build_search_stack
from safety_qa.infra.duckdb_store.query_repository import DuckDbQueryRepository
from safety_qa.infra.duckdb_store.schema import ensure_queries_schema
from safety_qa.infra.search.excerpt_builder import BM25ExcerptBuilder
from safety_qa.infra.settings import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "var" / "eval"
DEFAULT_BASELINE_DIR = Path(__file__).resolve().parent / "baseline"

# routing_stability は同一質問への repeat 回の試行を比較する指標。repeat=1 では
# 比較対象が1件しかなく数学的に必ず1.0になり、ゲート指標として意味を持たない。
_MIN_REPEAT_FOR_STABILITY = 2

_EXIT_CODES = {"passed": 0, "failed": 1, "inconclusive": 2}


class EvaluationInconclusiveError(Exception):
    """依存の取得失敗・データセット不整合等で評価そのものが実行できなかった。

    品質低下（絶対要件違反・ベースライン低下）とは区別する。
    `reason` は `evals.report.InconclusiveReason` のうち評価実行前に判明しうる種別のみを持つ。
    """

    def __init__(self, message: str, *, reason: InconclusiveReason = "dependency") -> None:
        super().__init__(message)
        self.reason: InconclusiveReason = reason


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 1.0


def _git_commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _runner_name() -> str:
    return "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"


def _build_search_stack_or_raise(settings: Settings) -> SearchStack:
    """検索スタックの構築（埋め込みモデル取得等）を試み、失敗は評価不能として扱う。

    Ollama 到達性確認（`_probe_chat_model`）と同じ理由で例外の種類を問わず
    捕捉する（依存の到達性確認）。
    """
    try:
        return build_search_stack(settings)
    except Exception as e:  # noqa: BLE001 -- 依存の到達性確認のため例外の種類を問わない
        raise EvaluationInconclusiveError(
            f"検索スタックを構築できません（埋め込みモデル取得等）: {e}", reason="dependency"
        ) from e


def _load_cases_or_raise(*, today: date | None = None) -> list[EvalCase]:
    try:
        return load_dataset(today=today) if today is not None else load_dataset()
    except ValueError as e:
        raise EvaluationInconclusiveError(
            f"評価データセットの読み込みに失敗しました: {e}", reason="dataset"
        ) from e


def _validate_must_hit_ids_or_raise(cases: list[EvalCase], known_ids: set[str]) -> None:
    try:
        validate_must_hit_ids_exist(cases, known_ids)
    except ValueError as e:
        raise EvaluationInconclusiveError(str(e), reason="dataset") from e


def _validate_expected_degraded_reason_or_raise(
    cases: list[EvalCase], data_period: tuple[date, date], *, today: date
) -> None:
    try:
        validate_expected_degraded_reason(cases, data_period, today=today)
    except ValueError as e:
        raise EvaluationInconclusiveError(str(e), reason="dataset") from e


# ---------------------------------------------------------------------------
# スイート実行結果
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuiteRunResult:
    case_results: list[CaseResult]
    aggregate_metrics: dict[str, float]
    reference_metrics: dict[str, float]
    dataset_version: str
    # 集計指標の算出に使ったケース数。0 件なら `_mean([])` が 1.0 を返す都合で
    # 「満点」に化けるため、呼び出し側で「評価不能」に倒す（カバレッジ喪失と
    # 満点が同じ見た目になってはならない）。
    gated_case_count: int
    # llm スイートのみ意味を持つ（システムプロンプト注入・期待引数解決に使った
    # 評価基準日）。retrieval では None。`temperature=0`/`seed=0` と同じ「制御変数
    # の固定」として、実行日ではなくコーパス最終日を使う（`_run_llm_suite` 参照）。
    evaluation_date: date | None = None


# ---------------------------------------------------------------------------
# retrieval suite（L1: Ollama 不要）
# ---------------------------------------------------------------------------


def _run_retrieval_suite(settings: Settings, top_n: int) -> SuiteRunResult:
    search_stack = _build_search_stack_or_raise(settings)
    try:
        cases = [c for c in _load_cases_or_raise() if c.must_hit_ids]
        known_ids = {r.id for r in search_stack.accident_repository.list_all()}
        _validate_must_hit_ids_or_raise(cases, known_ids)

        case_results: list[CaseResult] = []
        recalls: list[float] = []
        recalls_bm25_only: list[float] = []
        recalls_vector_only: list[float] = []
        bm25_zero_hit_flags: list[bool] = []
        latencies: list[float] = []

        for case in cases:
            start = time.perf_counter()
            diagnostics = search_stack.retriever.search_with_diagnostics(case.question, top_n=top_n)
            latencies.append(time.perf_counter() - start)
            hit_ids = diagnostics.record_ids

            recall = recall_at_k(hit_ids, case.must_hit_ids)
            recalls.append(recall)
            # ハイブリッド（RRF 統合）の recall と並べることで、融合が実際に
            # 効いているのか片方の経路だけで足りているのかを毎 PR で見える
            # ようにする（他プロジェクトの RAG ログ仕様を検討した際の指摘:
            # 「ベクトルと BM25 のスコアを別々に残さないと、ハイブリッドの
            # 重み調整が勘になる」への対応）。
            bm25_only_ids = diagnostics.bm25_ids[:top_n]
            vector_only_ids = diagnostics.vector_ids[:top_n]
            recalls_bm25_only.append(recall_at_k(bm25_only_ids, case.must_hit_ids))
            recalls_vector_only.append(recall_at_k(vector_only_ids, case.must_hit_ids))
            bm25_zero_hit_flags.append(len(diagnostics.bm25_ids) == 0)

            case_results.append(
                CaseResult(
                    id=case.id,
                    category=case.category,
                    question=case.question,
                    metrics={"recall_at_k": recall},
                    # L1 は検索器単体の再現率のみを見る（絞り込みなしの生の再現率
                    # であり、体験上の再現率ではない）。個々のケースを
                    # 絶対要件としては落とさず、低下の検出はベースライン比較に委ねる。
                    detail={
                        "hit_ids": hit_ids,
                        "must_hit_ids": case.must_hit_ids,
                        "rationale": case.rationale,
                        # 経路別の内訳（per-hit）。本番ログには出さず（ADR 008 の
                        # フラット原則）、正解ラベルのあるここでのみ出す。
                        "retrieval": {
                            "hits": [asdict(h) for h in diagnostics.hits],
                            "bm25_ids": bm25_only_ids,
                            "vector_ids": vector_only_ids,
                        },
                    },
                )
            )

        aggregate_metrics = {"mean_recall_at_k": _mean(recalls)}
        reference_metrics = {
            "mean_latency_seconds": _mean(latencies),
            "mean_recall_at_k_bm25_only": _mean(recalls_bm25_only),
            "mean_recall_at_k_vector_only": _mean(recalls_vector_only),
            "bm25_zero_hit_rate": _mean([float(f) for f in bm25_zero_hit_flags]),
        }
        return SuiteRunResult(
            case_results=case_results,
            aggregate_metrics=aggregate_metrics,
            reference_metrics=reference_metrics,
            dataset_version=dataset_version(cases),
            gated_case_count=len(cases),
        )
    finally:
        search_stack.close()


# ---------------------------------------------------------------------------
# llm suite（L2 + L3: 要 Ollama）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Attempt:
    tool_name: str | None
    tool_arguments: dict[str, str]
    answer: str
    cited_ids: list[str]
    elapsed_seconds: float
    # 回答が2パス目のLLM生成を経ず固定文で縮退したか（"no_tool_call" |
    # "unmatched_filter" | "out_of_coverage" | None）。絶対要件B の判定に使う。
    degraded_reason: str | None


@dataclass
class _CaseOutcome:
    result: CaseResult
    tool_match_flags: list[bool] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    argument_match_flags: list[bool] = field(default_factory=list)


def _probe_chat_model(chat_model: ChatModel) -> None:
    """Ollama への到達性を事前確認する。

    ここで失敗した場合はスイート全体を「評価不能」として止める。確認せずに
    全ケースを流すと、Ollama 未起動のような単一の原因が多数の「絶対要件違反」
    として報告され、品質低下と誤読される。
    """
    try:
        chat_model.route("接続確認のためのテスト問い合わせです。")
    except Exception as e:  # noqa: BLE001 -- 到達性確認なので例外の種類を問わない
        raise EvaluationInconclusiveError(
            f"Ollama に到達できません: {e}", reason="dependency"
        ) from e


def _run_case_once(
    qa_service: RagQaService, chat_model: RecordingChatModel, question: str
) -> _Attempt:
    start = time.perf_counter()
    query = anyio.run(qa_service.ask, question)
    elapsed = time.perf_counter() - start
    routing = chat_model.last_routing
    # 回答文末尾の根拠フッターは LLM の生成物ではなく決定的に組み立てた事実。
    # 捏造検出・件数照合・偽ゼロ判定は「モデルが自由生成した部分」だけを
    # 対象にする。
    generated_answer, _basis_footer = split_answer_basis(query.answer)
    return _Attempt(
        tool_name=routing.tool_name if routing else None,
        tool_arguments=routing.tool_arguments if routing else {},
        answer=generated_answer,
        cited_ids=[c.record_id for c in query.citations],
        elapsed_seconds=elapsed,
        degraded_reason=query.degraded_reason,
    )


def _run_attempts(
    qa_service: RagQaService, chat_model: RecordingChatModel, question: str, repeat: int
) -> tuple[list[_Attempt], list[str], list[str]]:
    """1 ケースを repeat 回試行する。1 回の失敗で全体を止めず、種類を分けて記録する。

    `InvalidToolCallError`（不正なツール呼び出しの決定的な拒否）は捏造ではなく
    「安全側に倒れて失敗した」動作であり、絶対要件違反としては扱わない
    （`hard_declines` に記録するのみ）。それ以外の `QaServiceError`（Ollama の
    タイムアウト等）は `failures` に記録し、違反として報告する。
    """
    attempts: list[_Attempt] = []
    failures: list[str] = []
    hard_declines: list[str] = []
    for i in range(1, repeat + 1):
        prefix = f"[試行{i}/{repeat}]"
        try:
            attempts.append(_run_case_once(qa_service, chat_model, question))
        except InvalidToolCallError as e:
            hard_declines.append(f"{prefix} ツール呼び出しが不正なため拒否された: {e}")
        except QaServiceError as e:
            failures.append(f"{prefix} 回答生成に失敗: {e}")
    return attempts, failures, hard_declines


def _truncated_count_for_search(
    attempts: list[_Attempt], question: str, search_stack: SearchStack
) -> int | None:
    """検索の絞り込み条件に対する総ヒット数と実際に引用した件数の差の最大値
    （見落としの可視化）。

    `must_hit_ids` の最小集合ラベルだけでは打ち切りを測れない。実際に呼ばれた
    絞り込み条件があれば、その総ヒット数を実データから求め、実際に引用された
    件数（`a.cited_ids`）との差を取る。`core.rag_service.truncated_count` と
    同じ関数（`AnswerBasis.truncated_count` の単一情報源）を使う。絞り込みなし
    （自由記述のみ）の試行は対象外にする（全件が「総ヒット数」になり指標として
    意味を持たないため）。
    """
    truncations: list[int] = []
    for a in attempts:
        if a.tool_name != SEARCH_TOOL_NAME:
            continue
        try:
            _, record_filter, _ignored_axes = parse_search_arguments(
                a.tool_arguments, fallback_query=question
            )
        except InvalidToolArgumentsError:
            continue
        if record_filter.is_empty:
            continue
        total_matching = len(search_stack.accident_repository.find_ids(record_filter))
        truncations.append(truncated_count(total_matching, len(a.cited_ids)))
    return max(truncations) if truncations else None


def _evaluate_case(
    case: EvalCase,
    attempts: list[_Attempt],
    failures: list[str],
    hard_declines: list[str],
    search_stack: SearchStack,
    *,
    today: date,
) -> _CaseOutcome:
    violations: list[str] = list(failures)
    metrics: dict[str, float] = {}
    detail: dict[str, object] = {
        "expected_tool": case.expected_tool,
        "attempts": [
            {
                "tool_name": a.tool_name,
                "tool_arguments": a.tool_arguments,
                "cited_ids": a.cited_ids,
                "answer": a.answer,
                "elapsed_seconds": round(a.elapsed_seconds, 3),
                "degraded_reason": a.degraded_reason,
            }
            for a in attempts
        ],
        "failures": failures,
        "hard_declines": hard_declines,
    }
    outcome = _CaseOutcome(
        result=CaseResult(id=case.id, category=case.category, question=case.question, detail=detail)
    )
    if not attempts:
        outcome.result.metrics = metrics
        outcome.result.violations = violations
        return outcome

    tool_match_flags = [tool_matches(a.tool_name, case.expected_tool) for a in attempts]
    outcome.tool_match_flags = tool_match_flags

    for i, a in enumerate(attempts, start=1):
        prefix = f"[試行{i}/{len(attempts)}]"
        fabricated = find_fabricated_ids(a.answer, a.cited_ids)
        if fabricated:
            violations.append(
                f"{prefix} citations外の id を言及（捏造の疑い）: {sorted(fabricated)}"
            )

        if case.expected_tool is not None:
            # search / analysis / aggregate / temporal: 答えられるはずの質問（絶対要件4）。
            if a.tool_name is None:
                violations.append(f"{prefix} 対応できるはずの質問でツールを呼び出さなかった")
            elif a.tool_name != case.expected_tool:
                violations.append(
                    f"{prefix} 期待ツール={case.expected_tool!r} 実際={a.tool_name!r}"
                )

    if case.must_hit_ids:
        recalls = [recall_at_k(a.cited_ids, case.must_hit_ids) for a in attempts]
        outcome.recalls = recalls
        detail["must_hit_ids"] = case.must_hit_ids
        detail["hit_ids"] = attempts[-1].cited_ids
        metrics["mean_recall_at_k"] = _mean(recalls)
        for i, a in enumerate(attempts, start=1):
            prefix = f"[試行{i}/{len(attempts)}]"
            if a.degraded_reason is not None:
                # 絶対要件B: 該当記録がある（must_hit_ids が非空）のに、2パス目の
                # LLM生成を経ず固定文で縮退した。degraded_reason で原因を明示する
                # （空振り軸の誤判定・収録期間外の誤判定等）。
                violations.append(
                    f"{prefix} 実際は該当記録があるのに縮退固定文で回答した"
                    f"（degraded_reason={a.degraded_reason!r}、実引数={a.tool_arguments}）"
                )
            elif not a.cited_ids:
                violations.append(
                    f"{prefix} 実際は該当記録があるのに citations が空（偽ゼロの疑い）"
                )
            elif recalls[i - 1] == 0.0:
                # 絶対要件B: 縮退もせず citations も返したが、必須の記録を1件も
                # 引用できなかった（自信を持った誤答）。縮退（利用者が気づける
                # 失敗）より危険なため違反として扱う。判定は must_hit_ids（実装
                # から独立した人のラベル）のみを使う。recall < 1.0 まで要件にする
                # と、正しく動作している既存ケース（search-01/02・analysis-01 は
                # recall 0.50）まで初日から違反になるため、recall == 0（起きては
                # ならない事象）だけを対象にする。
                violations.append(
                    f"{prefix} 必須の記録を1件も引用していない（recall=0、実引数={a.tool_arguments}）"
                    f": 必須={case.must_hit_ids} 実際={sorted(a.cited_ids)}"
                )
        truncated = _truncated_count_for_search(attempts, case.question, search_stack)
        if truncated is not None:
            detail["truncated_count"] = truncated

    if case.category in ("aggregate", "temporal"):
        resolved_arguments = case.resolved_expected_arguments(today=today)
        record_filter, group_by = parse_aggregate_tool_arguments(resolved_arguments)
        aggregate_result = search_stack.accident_repository.aggregate(
            record_filter, group_by=group_by
        )
        expected_count = aggregate_result.count
        expected_counts = {expected_count} | {g.count for g in aggregate_result.groups}
        expected_population = sorted(aggregate_result.record_ids)
        detail["expected_count"] = expected_count
        detail["expected_population"] = expected_population
        detail["expected_arguments"] = resolved_arguments
        detail["expected_degraded_reason"] = case.expected_degraded_reason

        argument_flags = [
            ok and arguments_match(a.tool_arguments, resolved_arguments)
            for a, ok in zip(attempts, tool_match_flags, strict=True)
        ]
        outcome.argument_match_flags = argument_flags
        metrics["argument_match_rate"] = _mean([float(f) for f in argument_flags])

        for i, a in enumerate(attempts, start=1):
            prefix = f"[試行{i}/{len(attempts)}]"
            if has_count_mismatch(a.answer, expected_counts):
                violations.append(
                    f"{prefix} 回答文中の件数が実データ（{expected_count}件）と不一致"
                )
            if has_false_zero_claim(a.answer, expected_count):
                violations.append(
                    f"{prefix} 実際は{expected_count}件あるのに0件/該当なしと回答（偽ゼロ）"
                )

            # 絶対要件B: 実データの期待件数が非0（＝答えられるはずの質問）なのに、
            # 2パス目のLLM生成を経ず固定文で縮退した。真に収録期間外の質問
            # （expected_count == 0）で縮退した場合は正しい動作なので対象外にする
            # （ADR 009 が意図した安全機構そのものを違反にしないため）。
            if expected_count > 0 and a.degraded_reason is not None:
                violations.append(
                    f"{prefix} 実際は{expected_count}件あるのに縮退固定文で回答した"
                    f"（degraded_reason={a.degraded_reason!r}、実引数={a.tool_arguments}）"
                )

            # 絶対要件A: citations が「実際に実行した引数」の母集団と一致するか
            # （ADR 006 が宣言する citations の決定性そのものの検証。LLM が選んだ
            # 引数の正しさとは独立に、実装が常に成立させるべき不変条件）。
            if a.tool_name == AGGREGATE_TOOL_NAME:
                try:
                    actual_filter, actual_group_by = parse_aggregate_tool_arguments(
                        a.tool_arguments
                    )
                except InvalidToolArgumentsError:
                    actual_filter = None
                if actual_filter is not None:
                    actual_population = sorted(
                        search_stack.accident_repository.aggregate(
                            actual_filter, group_by=actual_group_by
                        ).record_ids
                    )
                    if sorted(a.cited_ids) != actual_population:
                        violations.append(
                            f"{prefix} citations が実行した引数（{a.tool_arguments}）の"
                            f"母集団と不一致（実装のバグの疑い）: "
                            f"期待={actual_population} 実際={sorted(a.cited_ids)}"
                        )

            # 絶対要件C: citations が期待引数（データセットのラベル）の母集団と
            # 一致するか（質問解釈の正しさ）。件数だけの照合では、絞り込み条件
            # そのものを取り違えた場合に件数が偶然一致して見逃すことがある
            # （temporal カテゴリの実測で確認: 「今年」「去年」を誤った年に解釈
            # しても、その誤った年の該当件数が正解件数と偶然同じであれば件数照合
            # だけでは検出できない）。citations（母集団の全 id）を実データと直接
            # 突合することで、件数の偶然の一致をすり抜けないようにする。
            if sorted(a.cited_ids) != expected_population:
                violations.append(
                    f"{prefix} citations が実データの母集団と不一致"
                    f"（件数は一致していても中身が異なる可能性）: "
                    f"期待引数={resolved_arguments} 実引数={a.tool_arguments} / "
                    f"期待={expected_population} 実際={sorted(a.cited_ids)}"
                )

            # ケースが縮退そのものを期待している場合（`expected_degraded_reason`。
            # 例: 収録期間を確実に外す質問）、実際に縮退したかを直接検証する。
            # 要件B は expected_count>0 の場合のみ判定するため、この種のケース
            # （expected_count==0）は要件B・要件C の一致だけでは「本当に縮退した
            # か」「たまたま同じ結果になる非縮退の回答が返っただけか」を区別
            # できない（temporal-01 の空回りと同じ構造）。
            if case.expected_degraded_reason is not None:
                if a.degraded_reason != case.expected_degraded_reason:
                    violations.append(
                        f"{prefix} 収録期間外の質問が期待どおり縮退しなかった"
                        f"（期待degraded_reason={case.expected_degraded_reason!r} "
                        f"実際={a.degraded_reason!r}）"
                    )

    if case.category == "unsupported":
        for i, a in enumerate(attempts, start=1):
            # ツール未呼び出し（対応不可の宣言）で件数・数値を提示していないかのみ検証する。
            # モデルがツールを呼んだ場合（例: 未対応の group_by を無視して全体件数だけを
            # 返す）は、実際に実行されたツールの結果に基づく回答であり、この関数が
            # 検出したい「無から数値を捏造する」失敗モードとは異なるため対象外とする
            # （どの軸を無視したかの妥当性は detail の attempts から人手で確認する）。
            if a.tool_name is None and has_unsupported_fabrication(a.answer):
                violations.append(
                    f"[試行{i}/{len(attempts)}] 対応不可のはずの質問で件数・数値を提示している（捏造の疑い）"
                )

    if case.expected_tool is not None:
        metrics["tool_match_rate"] = _mean([float(f) for f in tool_match_flags])

    outcome.result.metrics = metrics
    outcome.result.violations = violations
    return outcome


def _stability(attempts: list[dict[str, object]]) -> float:
    keys = {
        (a["tool_name"], tuple(sorted(a["tool_arguments"].items())))  # type: ignore[union-attr]
        for a in attempts
    }
    return 1.0 if len(keys) <= 1 else 0.0


def _run_llm_suite(settings: Settings, top_n: int, repeat: int) -> SuiteRunResult:
    search_stack = _build_search_stack_or_raise(settings)
    try:
        data_period = search_stack.accident_repository.date_range()
        if data_period is None:
            raise EvaluationInconclusiveError(
                "事故記録が0件のため評価基準日を決定できません", reason="dependency"
            )
        # 評価基準日はコーパス最終日に固定する（`temperature=0`/`seed=0` と同じ
        # 「制御変数の固定」）。実行日（wall clock）のままだと「今年」の期待値が
        # 実行日によって変わり、コーパスが尽きた年では検証が恒真化する
        # （temporal-01 で実測。ADR 007 引き継ぎ事項）。out_of_coverage 縮退の
        # カバレッジは expected_degraded_reason="out_of_coverage" のケースで
        # 別途検証する（`_validate_expected_degraded_reason_or_raise`）。
        _, today = data_period

        cases = _load_cases_or_raise(today=today)
        known_ids = {r.id for r in search_stack.accident_repository.list_all()}
        _validate_must_hit_ids_or_raise(cases, known_ids)
        _validate_expected_degraded_reason_or_raise(cases, data_period, today=today)

        # プロンプト注入用の today（`build_chat_model`）と `RagQaService`/データ
        # セットの today（下の `today=lambda: today`）を同じ値に揃える。ずれると
        # システムプロンプトの相対日付解釈と根拠フッター・期待値の基準日が
        # 食い違う（infra/container.py の docstring 参照）。
        real_chat_model = build_chat_model(settings, today=lambda: today)
        _probe_chat_model(real_chat_model)
        chat_model = RecordingChatModel(real_chat_model)

        # 評価実行の監査ログは本番（var/queries.duckdb）と分離する。同じ
        # DuckDbQueryRepository 実装をそのまま使い、パスだけを切り替える
        # （アプリケーションコードに評価専用の分岐を作らない）。
        eval_queries_db_path = settings.accidents_db_path.parent / "eval_queries.duckdb"
        queries_con = duckdb.connect(str(eval_queries_db_path))
        ensure_queries_schema(queries_con)
        query_repository = DuckDbQueryRepository(queries_con)
        try:
            qa_service = RagQaService(
                accident_repository=search_stack.accident_repository,
                retriever=search_stack.retriever,
                query_repository=query_repository,
                chat_model=chat_model,
                excerpt_builder=BM25ExcerptBuilder(search_stack.bm25_index),
                top_n=top_n,
                max_concurrent_answers=settings.max_concurrent_answers,
                # データセットの期待値（`resolved_expected_arguments`）と回答の
                # 根拠フッターが同じ基準日を使うよう、呼び出しごとの再取得ではなく
                # このスイート実行に固定した `today` を渡す。
                today=lambda: today,
            )

            outcomes: list[_CaseOutcome] = []
            all_latencies: list[float] = []
            all_malformed_ids: set[str] = set()

            for case in cases:
                attempts, failures, hard_declines = _run_attempts(
                    qa_service, chat_model, case.question, repeat
                )
                all_latencies.extend(a.elapsed_seconds for a in attempts)
                for a in attempts:
                    all_malformed_ids |= extract_malformed_id_like_tokens(a.answer)
                outcomes.append(
                    _evaluate_case(
                        case,
                        attempts,
                        failures,
                        hard_declines,
                        search_stack,
                        today=today,
                    )
                )
        finally:
            queries_con.close()

        all_tool_flags = [f for o in outcomes for f in o.tool_match_flags]
        all_recalls = [r for o in outcomes for r in o.recalls]
        all_argument_flags = [f for o in outcomes for f in o.argument_match_flags]
        stability = [
            _stability(o.result.detail["attempts"])  # type: ignore[arg-type]
            for o in outcomes
            if o.result.detail["attempts"]
        ]

        aggregate_metrics = {
            "tool_selection_match_rate": _mean([float(f) for f in all_tool_flags]),
            "citation_recall": _mean(all_recalls),
            "aggregate_arguments_match_rate": _mean([float(f) for f in all_argument_flags]),
        }
        reference_metrics = {
            "mean_latency_seconds": _mean(all_latencies),
            "max_latency_seconds": max(all_latencies) if all_latencies else 0.0,
            "malformed_id_like_count": float(len(all_malformed_ids)),
        }
        if repeat >= _MIN_REPEAT_FOR_STABILITY:
            aggregate_metrics["routing_stability"] = _mean(stability)
        else:
            reference_metrics["routing_stability"] = _mean(stability)

        return SuiteRunResult(
            case_results=[o.result for o in outcomes],
            aggregate_metrics=aggregate_metrics,
            reference_metrics=reference_metrics,
            dataset_version=dataset_version(cases),
            gated_case_count=len(cases),
            evaluation_date=today,
        )
    finally:
        search_stack.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("1 以上を指定してください")
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("suite", choices=tuple(SUITES))
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        help="llm スイートのみ。同一質問の試行回数（既定 1）。"
        f"{_MIN_REPEAT_FOR_STABILITY} 以上で routing_stability をベースライン比較対象にする",
    )
    parser.add_argument("--top-n", type=int, default=None, help="既定値は Settings.top_n")
    parser.add_argument(
        "--update-baseline", action="store_true", help="実測値を暫定ベースラインとして保存する"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    return parser.parse_args(argv)


def _inconclusive_report(
    suite: str,
    reason: InconclusiveReason,
    detail: str,
    *,
    conditions: MeasurementConditions | None = None,
    baseline_update_run: bool = False,
) -> SuiteReport:
    return SuiteReport(
        suite=suite,
        cases=[],
        aggregate_metrics={},
        reference_metrics={},
        conditions=conditions,
        comparison=None,
        baseline=None,
        inconclusive_reason=reason,
        inconclusive_detail=detail,
        baseline_update_run=baseline_update_run,
    )


def _finish(report: SuiteReport, output_dir: Path) -> int:
    json_path, md_path = write_report(report, output_dir)
    print(report.to_markdown())
    print(f"\n(詳細: {json_path} / {md_path})")
    return _EXIT_CODES[report.status]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings()
    top_n = args.top_n if args.top_n is not None else settings.top_n
    suite_spec = SUITES[args.suite]
    model = suite_spec.model(settings)
    runner = _runner_name()

    try:
        corpus_ver = corpus_version(settings.csv_path)
    except OSError as e:
        return _finish(
            _inconclusive_report(
                args.suite,
                "dependency",
                f"事故記録 CSV を読み込めません: {settings.csv_path} ({e})",
            ),
            args.output_dir,
        )

    try:
        if args.suite == "retrieval":
            run_result = _run_retrieval_suite(settings, top_n)
        else:
            # llm スイートの評価基準日（コーパス最終日）は `_run_llm_suite` 内で
            # 決める（`SuiteRunResult.evaluation_date`）。ここでは wall clock を
            # 持ち込まない（`temperature=0`/`seed=0` と同じ「制御変数の固定」）。
            run_result = _run_llm_suite(settings, top_n, args.repeat)
    except EvaluationInconclusiveError as e:
        return _finish(
            _inconclusive_report(args.suite, e.reason, str(e)),
            args.output_dir,
        )

    conditions = MeasurementConditions(
        model=model,
        runner=runner,
        corpus_version=corpus_ver,
        dataset_version=run_result.dataset_version,
        top_n=top_n,
        repeat=args.repeat if suite_spec.uses_repeat else None,
        evaluation_date=(
            run_result.evaluation_date.isoformat()
            if suite_spec.date_dependent and run_result.evaluation_date is not None
            else None
        ),
    )

    if run_result.gated_case_count == 0:
        return _finish(
            _inconclusive_report(
                args.suite,
                "no_cases",
                "ゲート対象のケースが0件のため、集計指標は算出していません。",
                conditions=conditions,
                baseline_update_run=args.update_baseline,
            ),
            args.output_dir,
        )

    b_path = baseline_path(args.baseline_dir, args.suite, runner)
    try:
        baseline = load_baseline(b_path)
    except InvalidBaselineError as e:
        return _finish(
            _inconclusive_report(
                args.suite, "dependency", f"ベースラインを読み込めません: {b_path} ({e})"
            ),
            args.output_dir,
        )

    comparison = compare_to_baseline(
        baseline,
        current_metrics=run_result.aggregate_metrics,
        gated_metrics=list(run_result.aggregate_metrics),
        conditions=conditions,
    )

    report = SuiteReport(
        suite=args.suite,
        cases=run_result.case_results,
        aggregate_metrics=run_result.aggregate_metrics,
        reference_metrics=run_result.reference_metrics,
        conditions=conditions,
        comparison=comparison,
        baseline=baseline,
        baseline_update_run=args.update_baseline,
    )
    exit_code = _finish(report, args.output_dir)

    if args.update_baseline:
        if report.violations:
            print("\n絶対要件違反があるため、ベースラインは更新していません。")
        else:
            new_baseline = Baseline(
                measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
                commit=_git_commit(),
                conditions=conditions,
                metrics=run_result.aggregate_metrics,
            )
            save_baseline(b_path, new_baseline)
            print(f"\nベースラインを更新しました: {b_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
