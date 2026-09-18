"""RAG パイプライン本体。

tool calling によるルーティング -> 検索/集計の実行 -> 回答生成、の 2 パス構成。
`citations` と `AnswerBasis`（解釈済み条件・母数・収録期間）はツール実行結果から
決定的に組み立て、モデルには生成させない。

ブロッキング呼び出し（LLM 推論・DB アクセス・埋め込み）はすべて
`anyio.to_thread.run_sync` でスレッドへ退避する。
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

import anyio

from safety_qa.core.exceptions import (
    AnswerUnavailableError,
    InvalidToolArgumentsError,
    InvalidToolCallError,
    QaServiceError,
    QueryNotFoundError,
)
from safety_qa.core.excerpt import build_excerpt
from safety_qa.core.filters import PARTIAL_MATCH_AXES, VALID_SEVERITIES, RecordFilter
from safety_qa.core.ids import new_query_id
from safety_qa.core.models import (
    AccidentRecord,
    AnswerBasis,
    Citation,
    Query,
    QueryRecord,
    ToolCall,
)
from safety_qa.core.ports import (
    AccidentRecordRepository,
    ChatModel,
    ExcerptBuilder,
    QueryRepository,
    Retriever,
)
from safety_qa.core.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    NO_TOOL_CALL_ANSWER,
    build_answer_basis_footer,
    out_of_coverage_answer,
    unmatched_filter_answer,
)
from safety_qa.core.tools import (
    AGGREGATE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    parse_aggregate_tool_arguments,
    parse_search_arguments,
)

logger = logging.getLogger(__name__)

# ログに出さない自由記述由来の引数。絞り込み軸はデータ側の語彙であり監査上の
# 価値が高いためそのまま出す。evals/metrics.py の引数一致判定も同じ境界を使う。
FREE_TEXT_ARG_KEYS = frozenset({"query"})


def _loggable_tool_arguments(arguments: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in arguments.items() if k not in FREE_TEXT_ARG_KEYS}


def truncated_count(matched_count: int, cited_count: int) -> int:
    """`matched_count` 件のうち回答に現れなかった件数（0 未満にはならない）。

    `evals/metrics.py` の同名関数と `AnswerBasis.truncated_count` の単一情報源。
    """
    return max(matched_count - cited_count, 0)


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """ツール実行結果。`answer` が設定されている場合は2パス目のLLM生成を行わず、
    その固定文をそのまま回答として使う（空振り軸・収録期間外の縮退）。

    `record_filter`/`group_by`/`matched_count`/`data_period` は `AnswerBasis` を
    組み立てるために `_ask` へ引き渡す。`degraded_reason` は `answer` が設定される
    2 つの縮退経路のどちらかを示す（`rag.query.completed` の集計用）。真の 0 件
    （`answer is None` のまま LLM に生成させる）は縮退として扱わないため None のまま。
    """

    context: str
    citations: list[Citation]
    record_filter: RecordFilter
    data_period: tuple[date, date]
    group_by: str | None = None
    matched_count: int = 0
    answer: str | None = None
    degraded_reason: str | None = None
    # aggregate は severity 未絞り込みの場合のみ、search は常に埋める（引用した
    # 記録の重大度別内訳。citations と同じ母集団から導出するため決定的）。
    # `build_answer_basis_footer` に渡すだけで LLM の生成には含めない。
    severity_breakdown: dict[str, int] | None = None
    # search で LLM が渡したが実行に使わなかった軸（`SEARCH_EXCLUDED_FILTER_KEYS`）。
    # 根拠フッターの開示にのみ使う（`core/tools.py` の `parse_search_arguments` 参照）。
    ignored_axes: tuple[tuple[str, str], ...] = ()


def _severity_breakdown(records: list[AccidentRecord]) -> dict[str, int]:
    """`records`（母集団全件）を重大度別に集計する。VALID_SEVERITIES の順で返す。

    citations と同じ母集団（`_fetch_ordered` 済みの全件）から Python 側で数える
    だけで、LLM にも SQL にも新たな呼び出しを追加しない。
    """
    counter = Counter(r.severity for r in records)
    # 既知の重大度は VALID_SEVERITIES の順で出す。ingest が検証済みの値のみを
    # 書き込む前提だが、想定外の値が紛れても黙って捨てず末尾に残す（防御的）。
    ordered = {s: counter.pop(s) for s in VALID_SEVERITIES if s in counter}
    ordered.update(counter)
    return ordered


def _context_line(record: AccidentRecord) -> str:
    return (
        f"- {record.id} ({record.date} {record.equipment} {record.location} "
        f"重大度:{record.severity}): {record.description} / "
        f"原因: {record.cause} / 対策: {record.countermeasure}"
    )


class _FallbackExcerptBuilder:
    """`RagQaService.excerpt_builder` 未指定時の既定実装。terms を使わず
    description 先頭にフォールバックする。
    """

    def build(self, record: AccidentRecord, query: str | None) -> tuple[str | None, str]:
        return build_excerpt(
            description=record.description,
            cause=record.cause,
            countermeasure=record.countermeasure,
            terms=[],
        )


class RagQaService:
    """`QaService` の本実装。ポート越しに検索・集計・LLM 推論を組み合わせる。"""

    def __init__(
        self,
        *,
        accident_repository: AccidentRecordRepository,
        retriever: Retriever,
        query_repository: QueryRepository,
        chat_model: ChatModel,
        excerpt_builder: ExcerptBuilder | None = None,
        top_n: int = 5,
        max_concurrent_answers: int = 1,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._accidents = accident_repository
        self._retriever = retriever
        self._queries = query_repository
        self._chat_model = chat_model
        # 未指定時は terms を使わず description 先頭にフォールバックする
        # （ヒット欄の推定には BM25 の語彙統計が要るため、`core` 単体では行えない）。
        # 実運用は `infra.search.excerpt_builder.BM25ExcerptBuilder` を注入する。
        self._excerpt_builder = excerpt_builder or _FallbackExcerptBuilder()
        self._top_n = top_n
        # ローカル LLM は 4 コア級の環境が前提（SPEC.md）。並行リクエストが相互に
        # 詰まらないよう同時実行数を制限する。
        self._limiter = anyio.CapacityLimiter(max_concurrent_answers)
        # 根拠フッターの「最終記録からの経過日数」の基準日。プロンプト注入と
        # 同じ基準（`infra/clock.py`）を呼び出し側から注入する。
        self._today = today

    async def ask(self, question: str) -> Query:
        async with self._limiter:
            return await self._ask(question)

    async def _ask(self, question: str) -> Query:
        query_id = new_query_id()
        created_at = datetime.now()
        tool_name: str | None = None
        tool_arguments: dict[str, str] = {}
        basis: AnswerBasis | None = None
        degraded_reason: str | None = None
        started_at = time.perf_counter()
        # どのフェーズで失敗したかを rag.query.failed に残すため、フェーズに
        # 入る直前に更新する（例外はそのフェーズの処理中にしか起きない）。
        current_phase = "route"
        route_ms: float | None = None
        tool_ms: float | None = None
        answer_ms: float | None = None

        logger.info(
            "rag.query.started", extra={"query_id": str(query_id), "question_length": len(question)}
        )

        try:
            try:
                route_started_at = time.perf_counter()
                tool_call = await anyio.to_thread.run_sync(self._chat_model.route, question)
            except Exception as e:  # noqa: BLE001 -- Ollama 側の failure mode を問わず回答不能エラーに正規化する
                raise AnswerUnavailableError(f"ルーティングに失敗しました: {e}") from e
            route_ms = round((time.perf_counter() - route_started_at) * 1000, 1)

            citations: list[Citation] = []

            if tool_call is None:
                # 対策5(core/prompts.py): LLM を一切呼ばず固定文を返し、捏造の
                # 発生経路そのものを塞ぐ。発火した事実は警告として残す。
                degraded_reason = "no_tool_call"
                logger.warning("rag.no_tool_call", extra={"query_id": str(query_id)})
                logger.info(
                    "rag.routing.completed",
                    extra={
                        "query_id": str(query_id),
                        "tool_name": None,
                        "duration_ms": route_ms,
                    },
                )
                answer_text = NO_TOOL_CALL_ANSWER
            else:
                tool_name = tool_call.name
                tool_arguments = tool_call.arguments
                logger.info(
                    "rag.routing.completed",
                    extra={
                        "query_id": str(query_id),
                        "tool_name": tool_name,
                        "filter_args": _loggable_tool_arguments(tool_arguments),
                        "query_length": len(tool_arguments.get("query", "")) or None,
                        "duration_ms": route_ms,
                    },
                )
                current_phase = "tool"
                try:
                    tool_started_at = time.perf_counter()
                    outcome = await anyio.to_thread.run_sync(
                        self._execute_tool, tool_call, question, query_id
                    )
                except InvalidToolCallError:
                    raise  # 決定的な失敗。temperature=0 のため再試行しても結果は変わらない。
                except AnswerUnavailableError:
                    # コーパス空。既に正規化済みなので下の汎用 except で
                    # 二重ラップしない。
                    raise
                except Exception as e:  # noqa: BLE001 -- 検索/集計に使う DB・埋め込み等の失敗を取りこぼさない
                    raise AnswerUnavailableError(f"ツール実行に失敗しました: {e}") from e
                tool_ms = round((time.perf_counter() - tool_started_at) * 1000, 1)
                citations = outcome.citations
                degraded_reason = outcome.degraded_reason
                logger.info(
                    "rag.tool.completed",
                    extra={
                        "query_id": str(query_id),
                        "tool_name": tool_name,
                        "citation_count": len(citations),
                        "duration_ms": tool_ms,
                    },
                )
                if outcome.answer is not None:
                    # 空振り軸・収録期間外の固定文。対策5と同じ理由で2パス目の
                    # LLM生成を行わない。
                    answer_text = outcome.answer
                else:
                    current_phase = "answer"
                    prompt = ANSWER_PROMPT_TEMPLATE.format(
                        question=question, context=outcome.context
                    )
                    try:
                        answer_started_at = time.perf_counter()
                        answer_text = await anyio.to_thread.run_sync(
                            self._chat_model.answer, prompt
                        )
                    except Exception as e:  # noqa: BLE001 -- 上と同様
                        raise AnswerUnavailableError(f"回答生成に失敗しました: {e}") from e
                    answer_ms = round((time.perf_counter() - answer_started_at) * 1000, 1)
                    logger.info(
                        "rag.answer.generated",
                        extra={
                            "query_id": str(query_id),
                            "prompt_length": len(prompt),
                            "answer_length": len(answer_text),
                            "duration_ms": answer_ms,
                        },
                    )

                # 解釈済み条件・母数・収録期間を LLM を通さず回答文へ付す。
                # 2パス目の文脈には母数を渡していない（自由記述の中で誤転写
                # されうるため）ので、フッターは常にここで連結する。
                footer = build_answer_basis_footer(
                    record_filter=outcome.record_filter,
                    group_by=outcome.group_by,
                    matched_count=outcome.matched_count,
                    cited_count=len(citations),
                    data_period=outcome.data_period,
                    today=self._today(),
                    query_id=query_id,
                    severity_breakdown=outcome.severity_breakdown,
                    ignored_axes=outcome.ignored_axes,
                )
                answer_text = f"{answer_text}\n\n{footer}"
                first, last = outcome.data_period
                basis = AnswerBasis(
                    tool=tool_name,
                    filters=outcome.record_filter.to_dict(),
                    group_by=outcome.group_by,
                    matched_count=outcome.matched_count,
                    truncated_count=truncated_count(outcome.matched_count, len(citations)),
                    first_record_date=first.isoformat(),
                    last_record_date=last.isoformat(),
                )
        except QaServiceError as e:
            logger.error(
                "rag.query.failed",
                extra={
                    "query_id": str(query_id),
                    "error_code": type(e).__name__,
                    "failed_phase": current_phase,
                    "route_ms": route_ms,
                    "tool_ms": tool_ms,
                    "answer_ms": answer_ms,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
                },
            )
            await self._save_audit(
                QueryRecord(
                    id=query_id,
                    question=question,
                    created_at=created_at,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    error_code=type(e).__name__,
                    error_detail=str(e),
                ),
                # 監査ログへの保存自体が失敗しても、元のエラーを覆い隠さず警告に留める。
                swallow_save_errors=True,
            )
            raise

        query = Query(
            id=query_id,
            question=question,
            answer=answer_text,
            citations=citations,
            basis=basis,
            degraded_reason=degraded_reason,
            created_at=created_at,
        )
        await self._save_audit(
            QueryRecord(
                id=query.id,
                question=query.question,
                created_at=query.created_at,
                answer=query.answer,
                citations=query.citations,
                basis=query.basis,
                tool_name=tool_name,
                tool_arguments=tool_arguments,
            ),
            # 成功結果を監査ログに記録できていないのに 201 を返さない。保存失敗は伝播させる。
            swallow_save_errors=False,
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
        logger.info(
            "rag.query.completed",
            extra={
                "query_id": str(query_id),
                "tool_name": tool_name,
                "citation_count": len(citations),
                "matched_count": basis.matched_count if basis is not None else 0,
                "truncated_count": basis.truncated_count if basis is not None else 0,
                "degraded_reason": degraded_reason,
                "route_ms": route_ms,
                "tool_ms": tool_ms,
                "answer_ms": answer_ms,
                # generate（2パス目のLLM生成）が全体に占める割合。CPU推論では
                # generate が大半を占めるはずで、それを数字で示す（フェーズ別の
                # 個別イベントだけでは 1 クエリの内訳を 1 行で集計できないため）。
                "generate_share": (
                    round(answer_ms / duration_ms, 3)
                    if answer_ms is not None and duration_ms > 0
                    else None
                ),
                "duration_ms": duration_ms,
            },
        )
        return query

    async def _save_audit(self, record: QueryRecord, *, swallow_save_errors: bool) -> None:
        try:
            await anyio.to_thread.run_sync(self._queries.save, record)
        except Exception:
            if not swallow_save_errors:
                raise
            logger.error("audit.save_failed", extra={"query_id": str(record.id)}, exc_info=True)

    def _execute_tool(self, tool_call: ToolCall, question: str, query_id: UUID) -> _ToolOutcome:
        # ETL 未実行・失敗等でコーパスが空のまま起動すると、すべての質問が 0 件に
        # なり「事故ゼロの工場」として答え続ける。ツール実行前に検知し、縮退
        # させず回答不能として扱う。
        data_period = self._accidents.date_range()
        if data_period is None:
            raise AnswerUnavailableError("事故記録が読み込まれていません")

        if tool_call.name == SEARCH_TOOL_NAME:
            query, record_filter, ignored_axes = parse_search_arguments(
                tool_call.arguments, fallback_query=question
            )
            return self._execute_search(
                query,
                record_filter,
                data_period=data_period,
                query_id=query_id,
                ignored_axes=ignored_axes,
            )
        if tool_call.name == AGGREGATE_TOOL_NAME:
            return self._execute_aggregate(
                tool_call.arguments, data_period=data_period, query_id=query_id
            )
        raise InvalidToolArgumentsError(f"未知のツール: {tool_call.name!r}")

    def _fetch_ordered(self, record_ids: list[str]) -> list[AccidentRecord]:
        """`get_many` は順不同で返るため、渡した id の順序に復元する。"""
        records_by_id = {r.id: r for r in self._accidents.get_many(record_ids)}
        return [records_by_id[rid] for rid in record_ids if rid in records_by_id]

    def _to_citation(self, record: AccidentRecord, query: str | None) -> Citation:
        excerpt_field, excerpt = self._excerpt_builder.build(record, query)
        return Citation(
            record_id=record.id,
            date=record.date,
            location=record.location,
            equipment=record.equipment,
            severity=record.severity,
            excerpt=excerpt,
            excerpt_field=excerpt_field,
        )

    def _unmatched_axes(self, record_filter: RecordFilter) -> list[str]:
        """指定された部分一致軸のうち、単体でも1件も一致しない軸を返す。

        複合地名を location/equipment に誤分割すると、片方の軸が実データに
        存在しない値になり AND 条件で静かに 0 件になる。「本当に0件」と区別する
        ため、0件時に限り軸ごとに `find_ids` を1回ずつ呼んで空振りを検知する。

        検知対象は `PARTIAL_MATCH_AXES`（LIKE 部分一致の軸）に限る。この軸だけ
        「1件も一致しない」と「値がデータに存在しない」が定義上同値になるため
        （`core/filters.py` 参照）。severity/shift 等の enum 軸・date_*/hour_*/
        experience_* 等の範囲軸まで対象にすると、有効な値で単に該当0件のケース
        （例: shift="夜勤"だが夜勤の記録が実際に0件）まで「値が存在しない」と
        誤検知してしまう（過検知。修正時に実測して発覚した）。
        最大でも `PARTIAL_MATCH_AXES` の軸数（3）だけ `find_ids` を追加で呼ぶ。
        既存の `find_ids` ポートを流用し、新しいポートメソッドや起動時の
        SELECT DISTINCT は追加しない。
        """
        return [
            axis
            for axis in record_filter.specified_axes()
            if axis in PARTIAL_MATCH_AXES and not self._accidents.find_ids(record_filter.only(axis))
        ]

    def _zero_hit_outcome(
        self,
        record_filter: RecordFilter,
        *,
        data_period: tuple[date, date],
        base_context: str,
        query_id: UUID,
        group_by: str | None = None,
        ignored_axes: tuple[tuple[str, str], ...] = (),
    ) -> _ToolOutcome:
        """0 件時の縮退経路を1箇所に集約する（search/aggregate 共用）。

        判定順は 空振り軸 -> 収録期間外 -> 真の 0件。収録期間は呼び出し側の
        根拠フッターが常に併記するため、優先順位により情報が失われることはない。
        `ignored_axes` は aggregate では常に空（search 限定の軸のため）。
        """
        unmatched = self._unmatched_axes(record_filter)
        if unmatched:
            logger.warning(
                "rag.unmatched_filter",
                extra={"query_id": str(query_id), "unmatched_axes": unmatched},
            )
            return _ToolOutcome(
                context="",
                citations=[],
                record_filter=record_filter,
                data_period=data_period,
                group_by=group_by,
                answer=unmatched_filter_answer(record_filter, unmatched),
                degraded_reason="unmatched_filter",
                ignored_axes=ignored_axes,
            )

        first, last = data_period
        out_of_coverage = (
            record_filter.date_from is not None and record_filter.date_from > last
        ) or (record_filter.date_to is not None and record_filter.date_to < first)
        if out_of_coverage:
            logger.warning("rag.out_of_coverage", extra={"query_id": str(query_id)})
            return _ToolOutcome(
                context="",
                citations=[],
                record_filter=record_filter,
                data_period=data_period,
                group_by=group_by,
                answer=out_of_coverage_answer(record_filter, data_period),
                degraded_reason="out_of_coverage",
                ignored_axes=ignored_axes,
            )

        return _ToolOutcome(
            context=base_context,
            citations=[],
            record_filter=record_filter,
            data_period=data_period,
            group_by=group_by,
            ignored_axes=ignored_axes,
        )

    def _execute_search(
        self,
        query: str,
        record_filter: RecordFilter,
        *,
        data_period: tuple[date, date],
        query_id: UUID,
        ignored_axes: tuple[tuple[str, str], ...] = (),
    ) -> _ToolOutcome:
        if ignored_axes:
            # プロンプト側で抑止しきれず、LLM が重大度の絞り込みを渡した事実を残す
            # （実装が静かに吸収すると、プロンプト劣化が観測できなくなるため）。
            logger.warning(
                "rag.search.ignored_filter_axes",
                extra={"query_id": str(query_id), "ignored_axes": [k for k, _ in ignored_axes]},
            )
        ids = self._retriever.hybrid_search(query, top_n=self._top_n, record_filter=record_filter)
        ordered = self._fetch_ordered(ids)
        if not ordered:
            if record_filter.is_empty:
                base_context = "(該当する記録が見つかりませんでした)"
            else:
                base_context = (
                    f"条件（{record_filter.describe()}）に合致する記録は見つかりませんでした。"
                )
            return self._zero_hit_outcome(
                record_filter,
                data_period=data_period,
                base_context=base_context,
                query_id=query_id,
                ignored_axes=ignored_axes,
            )
        # 根拠フッターの「参照範囲」に示す母数。`record_filter.is_empty` でも
        # `count()` は WHERE 句なしの全件件数を返すため分岐不要。
        matched_count = self._accidents.count(record_filter)
        context = "\n".join(_context_line(r) for r in ordered)
        return _ToolOutcome(
            context=context,
            citations=[self._to_citation(r, query) for r in ordered],
            record_filter=record_filter,
            data_period=data_period,
            matched_count=matched_count,
            ignored_axes=ignored_axes,
            # search は重大度で絞り込まないため、引用した記録の重大度が一目で
            # 分かるよう常に内訳を計算する（aggregate と異なり severity 指定時の
            # 分岐は無い。search には severity という絞り込み軸自体が無いため）。
            severity_breakdown=_severity_breakdown(ordered),
        )

    def _execute_aggregate(
        self, arguments: dict[str, str], *, data_period: tuple[date, date], query_id: UUID
    ) -> _ToolOutcome:
        record_filter, group_by = parse_aggregate_tool_arguments(arguments)
        result = self._accidents.aggregate(record_filter, group_by=group_by)
        if not result.record_ids:
            base_context = f"条件（{record_filter.describe()}）に合致する記録はありません（0件）。"
            return self._zero_hit_outcome(
                record_filter,
                data_period=data_period,
                base_context=base_context,
                query_id=query_id,
                group_by=group_by,
            )
        # 回答生成には件数と条件（・内訳）で十分（id 一覧は citations が決定的に運ぶため
        # プロンプトには含めない。母集団が大きいとプロンプト長が件数に比例して
        # しまうため）。citations には監査性のため母集団全件の
        # メタデータを含める。内訳の件数は SQL の GROUP BY 由来の値をそのまま
        # 文字列化するだけで、LLM に数値を生成させない。
        ordered = self._fetch_ordered(result.record_ids)
        context = f"条件（{record_filter.describe()}）に合致する件数: {result.count} 件"
        if result.groups:
            breakdown = "\n".join(f"- {g.key}: {g.count} 件" for g in result.groups)
            context += f"\n{group_by}別の内訳:\n{breakdown}"
        # group_by 未指定の集計は、件数の見出しだけでは重大度が混在しているか
        # 分からない（例:「溶接機に関する事故は全部で何件ですか」に負傷事故と
        # ヒヤリハットが無告知で混ざる）。citations と同じ母集団から重大度内訳を
        # 決定的に組み立て、根拠フッターに添える（`build_answer_basis_footer` 側で
        # 単一の重大度しか無い場合は冗長として省く）。
        severity_breakdown = _severity_breakdown(ordered) if group_by is None else None
        return _ToolOutcome(
            context=context,
            citations=[self._to_citation(r, None) for r in ordered],
            record_filter=record_filter,
            data_period=data_period,
            group_by=group_by,
            # aggregate の citations は母集団全件なので、母数は追加クエリ無しで
            # `AggregateResult.count` からそのまま得られる（`search` と異なり
            # `count()` を別途呼ばない）。
            matched_count=result.count,
            severity_breakdown=severity_breakdown,
        )

    async def get(self, query_id: UUID) -> Query:
        query = await anyio.to_thread.run_sync(self._queries.get, query_id)
        if query is None:
            raise QueryNotFoundError(query_id)
        return query

    async def list_queries(self, *, limit: int, before_id: UUID | None = None) -> list[QueryRecord]:
        return await anyio.to_thread.run_sync(
            functools.partial(self._queries.list_records, limit=limit, before_id=before_id)
        )
