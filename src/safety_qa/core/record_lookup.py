"""`RecordService` の本実装。

citation の id から元記録を引く経路（`get`）と、問い合わせが実際に対象とした
母集団を保存済みの `tool_arguments` から再構成してページングする経路
（`list_for_query`）を持つ。既存の `AccidentRecordRepository`/`Retriever`/
`QueryRepository` のみを使い、新しいポート実装は追加しない。
"""

from __future__ import annotations

from uuid import UUID

import anyio

from safety_qa.core.exceptions import (
    InvalidToolArgumentsError,
    QueryNotFoundError,
    RecordNotFoundError,
)
from safety_qa.core.models import AccidentRecord, QueryRecordPage
from safety_qa.core.ordering import restore_order
from safety_qa.core.ports import AccidentRecordRepository, QueryRepository, Retriever
from safety_qa.core.tools import (
    AGGREGATE_TOOL_NAME,
    SEARCH_TOOL_NAME,
    parse_aggregate_tool_arguments,
    parse_search_arguments,
)


class AccidentRecordLookupService:
    """`RecordService` の本実装。"""

    def __init__(
        self,
        *,
        accident_repository: AccidentRecordRepository,
        retriever: Retriever,
        query_repository: QueryRepository,
        max_page_size: int = 100,
    ) -> None:
        self._accidents = accident_repository
        self._retriever = retriever
        self._queries = query_repository
        self._max_page_size = max_page_size

    async def get(self, record_id: str) -> AccidentRecord:
        records = await anyio.to_thread.run_sync(self._accidents.get_many, [record_id])
        if not records:
            raise RecordNotFoundError(record_id)
        return records[0]

    async def list_for_query(self, query_id: UUID, *, limit: int, offset: int) -> QueryRecordPage:
        # `RagQaService._execute_tool` と同じ作法: 検索・DB アクセスを1回の
        # スレッド退避にまとめ、内部では素朴な同期呼び出しにする。
        return await anyio.to_thread.run_sync(self._list_for_query_sync, query_id, limit, offset)

    def _list_for_query_sync(self, query_id: UUID, limit: int, offset: int) -> QueryRecordPage:
        limit = min(limit, self._max_page_size)
        record = self._queries.get_record(query_id)
        if record is None:
            raise QueryNotFoundError(query_id)

        as_of = self._accidents.ingested_at()

        # ツール未呼び出し、または途中で失敗した問い合わせ（tool_name はあっても
        # basis が無い = 集計・回答生成のどこかで失敗した）は対象母集団が
        # 定まらないため空ページを返す。保存済みの生引数をそのまま再パースして
        # 未知の失敗を起こさないための防御でもある。
        if record.tool_name is None or record.basis is None:
            return QueryRecordPage(
                items=[],
                total=0,
                limit=limit,
                offset=offset,
                ranking="date",
                query_text=None,
                basis=None,
                as_of=as_of,
            )

        if record.tool_name == SEARCH_TOOL_NAME:
            query_text, record_filter, _ignored_axes = parse_search_arguments(
                record.tool_arguments, fallback_query=record.question
            )
            top_n = min(offset + limit, self._max_page_size)
            ids = self._retriever.hybrid_search(query_text, top_n, record_filter=record_filter)
            page_ids = ids[offset : offset + limit]
            items = restore_order(self._accidents.get_many(page_ids), page_ids)
            ranking = "relevance"
        elif record.tool_name == AGGREGATE_TOOL_NAME:
            record_filter, _group_by = parse_aggregate_tool_arguments(record.tool_arguments)
            items = self._accidents.list_records(record_filter, limit=limit, offset=offset)
            ranking = "date"
            query_text = None
        else:
            # basis が組み立てられた時点でどちらかのツール名のはずで、通常
            # 到達しない（`_execute_tool` と同じ防御的分岐）。
            raise InvalidToolArgumentsError(f"未知のツール: {record.tool_name!r}")

        return QueryRecordPage(
            items=items,
            total=record.basis.matched_count,
            limit=limit,
            offset=offset,
            ranking=ranking,
            query_text=query_text,
            basis=record.basis,
            as_of=as_of,
        )
