"""`POST /v1/queries`, `GET /v1/queries/{id}`, `GET /v1/queries/{id}/records`。"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, Response, status

from safety_qa.api.dependencies import QaServiceDep, RecordServiceDep
from safety_qa.api.schemas.problem import ProblemDetails
from safety_qa.api.schemas.queries import (
    QueryListResponse,
    QueryRecordsResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/v1/queries", tags=["queries"])


def _problem_response(description: str) -> dict:
    return {"model": ProblemDetails, "description": description}


# 実際のメディアタイプ (application/problem+json) への書き換えは
# api/openapi.py の custom_openapi() で行う
# （FastAPI の `responses={..., "model": ...}` は常に application/json で
#   スキーマを生成するため）。
_ERROR_RESPONSES: dict[int | str, dict] = {
    400: _problem_response("リクエストが不正"),
    404: _problem_response("問い合わせが見つからない"),
    429: _problem_response("レート制限"),
    503: _problem_response("回答生成が利用不可"),
    500: _problem_response("内部エラー"),
}


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
async def create_query(
    body: QueryRequest, service: QaServiceDep, response: Response
) -> QueryResponse:
    """質問を受け付け、回答を生成して 201 Created で返す。"""
    query = await service.ask(body.question)
    response.headers["Location"] = f"/v1/queries/{query.id}"
    return QueryResponse.from_core(query)


@router.get(
    "/{query_id}",
    response_model=QueryResponse,
    responses=_ERROR_RESPONSES,
)
async def get_query(query_id: UUID, service: QaServiceDep) -> QueryResponse:
    """過去の問い合わせを id で再参照する。"""
    query = await service.get(query_id)
    return QueryResponse.from_core(query)


@router.get(
    "",
    response_model=QueryListResponse,
    responses=_ERROR_RESPONSES,
)
async def list_queries(
    service: QaServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    before_id: UUID | None = None,
) -> QueryListResponse:
    """呼び出し元がアクセスを許された問い合わせの一覧を新しい順に返す。

    成功・失敗の両方を含む。`status` が `"failed"` の項目は回答を保存していない
    ため、`GET /v1/queries/{id}` は 404 になる。
    """
    records = await service.list_queries(limit=limit, before_id=before_id)
    return QueryListResponse.from_core(records, limit=limit)


@router.get(
    "/{query_id}/records",
    response_model=QueryRecordsResponse,
    responses=_ERROR_RESPONSES,
)
async def get_query_records(
    query_id: UUID,
    service: RecordServiceDep,
    limit: int = Query(50, ge=1, description="1ページあたりの件数（実効上限は運用設定による）"),
    offset: int = Query(0, ge=0, description="スキップする件数"),
) -> QueryRecordsResponse:
    """その問い合わせが実際に対象とした母集団をページングして取得する。

    回答の citations（top_n 件、または集計の母集団）に含まれなかった残りの
    記録を取得する経路（issue #23）。`query`/絞り込み条件は元の問い合わせが
    保存した引数から再構成するため、回答生成時と同じ母集団になる。
    """
    page = await service.list_for_query(query_id, limit=limit, offset=offset)
    return QueryRecordsResponse.from_core(page)
