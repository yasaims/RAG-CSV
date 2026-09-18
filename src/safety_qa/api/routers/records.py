"""`GET /v1/records/{id}`。"""

from __future__ import annotations

from fastapi import APIRouter

from safety_qa.api.dependencies import RecordServiceDep
from safety_qa.api.schemas.problem import ProblemDetails
from safety_qa.api.schemas.records import RecordResponse

router = APIRouter(prefix="/v1/records", tags=["records"])


def _problem_response(description: str) -> dict:
    return {"model": ProblemDetails, "description": description}


_ERROR_RESPONSES: dict[int | str, dict] = {
    404: _problem_response("記録が見つからない"),
    500: _problem_response("内部エラー"),
}


@router.get(
    "/{record_id}",
    response_model=RecordResponse,
    responses=_ERROR_RESPONSES,
)
async def get_record(record_id: str, service: RecordServiceDep) -> RecordResponse:
    """citation の id から元記録を参照する。

    citation は回答生成時点の凍結スナップショットであり、この記録は現在の
    値を返す（両者が乖離しうることは `GET /v1/queries/{id}/records` の
    `as_of` で扱う）。
    """
    record = await service.get(record_id)
    return RecordResponse.from_core(record)
