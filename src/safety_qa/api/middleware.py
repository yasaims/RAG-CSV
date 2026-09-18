"""リクエストごとの `trace_id` 発行・応答ヘッダ付与・アクセスログ出力。

`BaseHTTPMiddleware` ではなく素の ASGI ミドルウェアで実装する。Starlette の
`ExceptionMiddleware`（個別に登録した例外ハンドラ）より外側に位置するため、
Problem Details に変換された応答にも `X-Trace-Id` を付与でき、所要時間も
正確に測れる。

ただし Starlette の `ServerErrorMiddleware`（`Exception` 全般の最終防衛線）は
常にこのミドルウェアより外側になる（Starlette の制約）。想定外例外がそこまで
抜けた場合、この層の `send_wrapper` を経由しないため `X-Trace-Id` を後付けできない。
そのため `X-Trace-Id` の付与は `api/errors.py` の `_problem_response` 側でも
行い、ここでは「まだ付いていない応答にだけ補完する」フォールバックとする。
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from safety_qa.infra.observability.context import bind_trace_id, clear_trace_id, extract_trace_id
from safety_qa.infra.settings import Settings

logger = structlog.get_logger(__name__)


class TraceContextMiddleware:
    """trace_id の発行/継承・`X-Trace-Id` 付与・`http.request.completed` の出力を行う。"""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self._app = app
        self._trust_inbound = settings.trust_inbound_trace_header

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", [])
        }
        trace_id = extract_trace_id(headers, trust_inbound=self._trust_inbound)
        bind_trace_id(trace_id)

        started_at = time.perf_counter()
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers_list = list(message.get("headers", []))
                # errors.py の _problem_response が既に X-Trace-Id を付けている場合は
                # 二重に付与しない（同じ trace_id なので実害はないが、応答を汚さない）。
                if not any(k.lower() == b"x-trace-id" for k, _ in headers_list):
                    headers_list.append((b"x-trace-id", trace_id.encode("latin1")))
                message = {**message, "headers": headers_list}
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            route: Any = scope.get("route")
            path = getattr(route, "path", scope.get("path", ""))
            logger.info(
                "http.request.completed",
                method=scope.get("method", ""),
                path=path,
                status_code=status_code,
                duration_ms=round(duration_ms, 1),
            )
            clear_trace_id()
