"""core の例外 → RFC 9457 Problem Details への変換。

フレームワーク既定のバリデーションエラー（422）は素通しせず、ここで 400 に変換する。
"""

from __future__ import annotations

import structlog
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from safety_qa.api.schemas.problem import ErrorCode, ProblemDetails
from safety_qa.core.exceptions import (
    AnswerUnavailableError,
    InvalidToolCallError,
    QueryNotFoundError,
    RecordNotFoundError,
)
from safety_qa.infra.observability.context import get_trace_id, new_trace_id

logger = structlog.get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
_DOCS_BASE = "https://kaminashi-inc.github.io/eng-6001_yasaims/errors"


def _problem_response(
    *,
    request: Request,
    status_code: int,
    title: str,
    detail: str,
    code: ErrorCode,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    # 通常は TraceContextMiddleware が既にコンテキストへ trace_id をバインド済み。
    # 未設定の場合（ミドルウェア未適用のテスト等）のみここで新規発行する。
    trace_id = get_trace_id() or new_trace_id()
    problem = ProblemDetails(
        type=f"{_DOCS_BASE}/{code.value.lower().replace('_', '-')}",
        title=title,
        status=status_code,
        detail=detail,
        instance=request.url.path,
        code=code,
        trace_id=trace_id,
    )
    # ヘッダにも trace_id を積む。TraceContextMiddleware は Starlette の
    # ServerErrorMiddleware（常に最外層）より内側にいるため、想定外例外が
    # そこまで抜けた場合はミドルウェアの送出フックを経由しない。ここで
    # レスポンス自体に X-Trace-Id を持たせておくことで、その経路でもヘッダが失われない。
    response_headers = {**(headers or {}), "X-Trace-Id": trace_id}
    return JSONResponse(
        status_code=status_code,
        content=problem.model_dump(mode="json"),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=response_headers,
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI 既定の 422 を 400 + Problem Details に変換する。"""
    return _problem_response(
        request=request,
        status_code=status.HTTP_400_BAD_REQUEST,
        title="質問が不正です",
        detail=str(exc.errors()),
        code=ErrorCode.INVALID_REQUEST,
    )


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """FastAPI/Starlette 既定の HTTPException を Problem Details に変換する。

    未定義ルートへのアクセス等、core のどの例外にも対応しない 404 はここに来る。
    `QueryNotFoundError`/`RecordNotFoundError` は専用ハンドラが個別に扱うため、
    ここでの 404 を `QUERY_NOT_FOUND` にすると無関係なエラーコードを返すことになる。
    """
    code = ErrorCode.INTERNAL_ERROR
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        code = ErrorCode.NOT_FOUND
    return _problem_response(
        request=request,
        status_code=exc.status_code,
        title=exc.detail if isinstance(exc.detail, str) else "エラーが発生しました",
        detail=exc.detail if isinstance(exc.detail, str) else "",
        code=code,
    )


async def handle_query_not_found(request: Request, exc: QueryNotFoundError) -> JSONResponse:
    return _problem_response(
        request=request,
        status_code=status.HTTP_404_NOT_FOUND,
        title="問い合わせが見つかりません",
        detail=str(exc),
        code=ErrorCode.QUERY_NOT_FOUND,
    )


async def handle_record_not_found(request: Request, exc: RecordNotFoundError) -> JSONResponse:
    return _problem_response(
        request=request,
        status_code=status.HTTP_404_NOT_FOUND,
        title="記録が見つかりません",
        # citation の id は回答生成時点の凍結スナップショットのため、記録が
        # 後から削除された可能性にも触れる（システムの不具合と区別できないため）。
        detail=f"{exc}（存在しない id、または回答生成後に削除された可能性があります）",
        code=ErrorCode.RECORD_NOT_FOUND,
    )


async def handle_answer_unavailable(request: Request, exc: AnswerUnavailableError) -> JSONResponse:
    return _problem_response(
        request=request,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        title="回答を生成できません",
        detail=str(exc) or "検索または LLM 推論に失敗しました。",
        code=ErrorCode.ANSWER_UNAVAILABLE,
        headers={"Retry-After": "30"},
    )


async def handle_invalid_tool_call(request: Request, exc: InvalidToolCallError) -> JSONResponse:
    """LLM のツール呼び出しが決定的に不正。

    temperature=0 のため再試行しても同じ結果になる一時的でない失敗であり、
    `ANSWER_UNAVAILABLE`（503 + Retry-After）とは区別して Retry-After を付けない。
    """
    return _problem_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="回答を生成できません",
        detail=str(exc) or "ツール呼び出しの内容が不正です。",
        code=ErrorCode.INTERNAL_ERROR,
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """想定外の例外の最終防衛線。素の 500（text/plain）ではなく契約どおりの
    Problem Details（application/problem+json）を返す。

    ここに到達する例外は core 側のどの正規化にも当てはまらなかったバグの可能性が
    高い。原因追跡の唯一の手段になるため、`exc_info` 付きで必ずログに残す。
    """
    logger.error(
        "http.request.failed",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type=type(exc).__name__,
        exc_info=exc,
    )
    return _problem_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="内部エラーが発生しました",
        detail="内部エラーが発生しました。",
        code=ErrorCode.INTERNAL_ERROR,
    )


EXCEPTION_HANDLERS = {
    RequestValidationError: handle_validation_error,
    StarletteHTTPException: handle_http_exception,
    QueryNotFoundError: handle_query_not_found,
    RecordNotFoundError: handle_record_not_found,
    AnswerUnavailableError: handle_answer_unavailable,
    InvalidToolCallError: handle_invalid_tool_call,
    Exception: handle_unexpected_error,
}
