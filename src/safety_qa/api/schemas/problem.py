"""RFC 9457 Problem Details のレスポンスモデル。

標準メンバーに加え、機械可読な `code` とログ突合用の `trace_id` を持つ。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    QUERY_NOT_FOUND = "QUERY_NOT_FOUND"
    RECORD_NOT_FOUND = "RECORD_NOT_FOUND"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    ANSWER_UNAVAILABLE = "ANSWER_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ProblemDetails(BaseModel):
    type: str = Field(description="人間向けドキュメントを指す URI")
    title: str
    status: int
    detail: str
    instance: str
    code: ErrorCode
    trace_id: str = Field(description="構造化ログと突合するための追跡 ID")
