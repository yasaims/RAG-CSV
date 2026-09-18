"""FastAPI 既定の 422 を生成 OpenAPI から除去する。

FastAPI はリクエストボディを持つ operation に自動で `422
Validation Error` を載せるが、実際には 422 を返さず 400 +
Problem Details に変換している（api/errors.py）。素通しすると生成 spec が
実装と食い違うため、`app.openapi()` を差し替えて除去する。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

# エラー応答はすべて RFC 9457 Problem Details として返す。
# FastAPI の `responses={status: {"model": ...}}` は常に application/json で
# スキーマを生成するため、ここで application/problem+json に付け替える。
_PROBLEM_STATUS_CODES = {"400", "404", "429", "503", "500"}


def custom_openapi(app: FastAPI) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
    )

    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            responses = operation.get("responses", {})
            responses.pop("422", None)
            for status_code in _PROBLEM_STATUS_CODES:
                response = responses.get(status_code)
                if response and "application/json" in response.get("content", {}):
                    response["content"]["application/problem+json"] = response["content"].pop(
                        "application/json"
                    )

    app.openapi_schema = schema
    return app.openapi_schema
