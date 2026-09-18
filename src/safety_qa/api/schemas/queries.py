"""`/v1/queries` のリクエスト/レスポンスモデル。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from safety_qa.api.schemas.records import RecordResponse
from safety_qa.core.models import Query, QueryRecord, QueryRecordPage


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    id: str = Field(description="参照した事故記録の id")
    date: str = Field(description="事故記録の発生日")
    location: str = Field(description="発生場所")
    equipment: str = Field(description="関連する設備")
    severity: str = Field(description="重大度")
    excerpt: str = Field(description="回答の根拠となった記述の抜粋")
    excerpt_field: str | None = Field(
        description="excerpt の抜粋元の欄（description/cause/countermeasure）。"
        "ヒット箇所を特定できなかった場合は null"
    )


class AnswerBasis(BaseModel):
    """回答の根拠情報。

    解釈済みの絞り込み条件・参照した件数と母数・記録の収録期間を機械可読な形で
    運ぶ。人間向けの要約は `answer` 本文末尾のフッターに決定的に付与される
    （どちらもモデルには生成させない。`core/rag_service.py` 参照）。ツール
    未呼び出しの問い合わせでは `null`。
    """

    tool: str = Field(description="実行したツール名")
    filters: dict[str, str] = Field(description="検証・正規化後の絞り込み条件（軸→値）")
    group_by: str | None = Field(description="集計の内訳軸（未指定は null）")
    matched_count: int = Field(description="絞り込み条件に合致した記録数（絞り込みなしは全記録数）")
    truncated_count: int = Field(description="参照されなかった件数（aggregate は常に0）")
    first_record_date: str = Field(description="収録記録の最古日")
    last_record_date: str = Field(description="収録記録の最新日")


class QueryResponse(BaseModel):
    id: UUID
    question: str
    answer: str
    citations: list[Citation]
    basis: AnswerBasis | None = None
    created_at: datetime

    @classmethod
    def from_core(cls, query: Query) -> QueryResponse:
        return cls(
            id=query.id,
            question=query.question,
            answer=query.answer,
            citations=[
                Citation(
                    id=c.record_id,
                    date=c.date,
                    location=c.location,
                    equipment=c.equipment,
                    severity=c.severity,
                    excerpt=c.excerpt,
                    excerpt_field=c.excerpt_field,
                )
                for c in query.citations
            ],
            basis=AnswerBasis(**asdict(query.basis)) if query.basis is not None else None,
            created_at=query.created_at,
        )


class QueryRecordsResponse(BaseModel):
    """`GET /v1/queries/{id}/records` のレスポンス。

    問い合わせが実際に対象とした母集団を、保存済みの引数から再構成して
    ページングした結果。`citations`（回答生成時点の凍結）とは別に、`as_of`
    時点の現在値を返す（両者は ETL の再実行により乖離しうる）。
    """

    items: list[RecordResponse]
    total: int = Field(description="対象母集団の総件数（basis.matched_count と一致する）")
    limit: int
    offset: int
    ranking: str = Field(description='items の並び順（"relevance" | "date"）')
    query_text: str | None = Field(
        description="search 由来の自由文クエリ（aggregate 由来は null）。"
        "絞り込みには使われず並び順にのみ影響する"
    )
    basis: AnswerBasis | None = Field(description="元の問い合わせの回答根拠情報")
    as_of: datetime | None = Field(description="items の取得時点（ETL 取り込み時刻、UTC）")

    @classmethod
    def from_core(cls, page: QueryRecordPage) -> QueryRecordsResponse:
        return cls(
            items=[RecordResponse.from_core(r) for r in page.items],
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            ranking=page.ranking,
            query_text=page.query_text,
            basis=AnswerBasis(**asdict(page.basis)) if page.basis is not None else None,
            as_of=page.as_of,
        )


class QuerySummary(BaseModel):
    """問い合わせ一覧 1 件分。`answer` 本文・`citations` は含まない（詳細は
    `GET /v1/queries/{id}` を別途参照する）。
    """

    id: UUID
    question: str
    created_at: datetime
    status: str = Field(description='"succeeded"（成功） または "failed"（失敗）')
    error_code: str | None = Field(description="失敗時のエラー種別。成功時は null")
    basis: AnswerBasis | None = Field(
        description="回答の根拠情報。同じ質問文の再実行を区別する手がかりになる。未呼び出し・失敗時は null"
    )

    @classmethod
    def from_core(cls, record: QueryRecord) -> QuerySummary:
        return cls(
            id=record.id,
            question=record.question,
            created_at=record.created_at,
            status="succeeded" if record.error_code is None else "failed",
            error_code=record.error_code,
            basis=AnswerBasis(**asdict(record.basis)) if record.basis is not None else None,
        )


class QueryListResponse(BaseModel):
    items: list[QuerySummary]
    next_before_id: UUID | None = Field(
        description="続きを取得する際に before_id に渡す値。これ以上遡れる問い合わせが無い場合は null"
    )

    @classmethod
    def from_core(cls, records: list[QueryRecord], *, limit: int) -> QueryListResponse:
        return cls(
            items=[QuerySummary.from_core(r) for r in records],
            next_before_id=records[-1].id if len(records) == limit else None,
        )
