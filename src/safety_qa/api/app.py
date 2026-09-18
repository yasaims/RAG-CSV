"""FastAPI アプリケーションのエントリポイント。

起動: `uv run uvicorn safety_qa.api.app:app`
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import anyio
import structlog
from fastapi import FastAPI

from safety_qa.api.errors import EXCEPTION_HANDLERS
from safety_qa.api.middleware import TraceContextMiddleware
from safety_qa.api.openapi import custom_openapi
from safety_qa.api.routers.queries import router as queries_router
from safety_qa.api.routers.records import router as records_router
from safety_qa.infra.observability.logging import configure_logging
from safety_qa.infra.settings import load_settings

# `infra.settings` / `infra.observability` は torch / duckdb を引き込まない軽量な
# モジュールのため、ここ（モジュールトップ）で読んでよい。uvicorn は自身のログ設定
# （Config.__init__）の後にアプリを import するため、この呼び出しが最後に効き
# 標準出力への構造化ログが確立した状態でアプリ本体が組み立てられる。
_settings = load_settings()
configure_logging(_settings)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # infra.container の import はこの関数内に閉じる。モジュールトップに置くと
    # scripts/gen_openapi.py や core/api の軽量なユニットテストまで
    # torch / duckdb のロードを強いてしまう。
    from safety_qa.infra.container import build_container

    started_at = time.perf_counter()
    container = await anyio.to_thread.run_sync(build_container)
    app.state.container = container
    logger.info(
        "startup.completed", duration_ms=round((time.perf_counter() - started_at) * 1000, 1)
    )
    try:
        yield
    finally:
        container.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="安全管理 Q&A API",
        description="工場の事故・ヒヤリハット記録を参照して回答する Q&A API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(queries_router)
    app.include_router(records_router)

    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)

    app.add_middleware(TraceContextMiddleware, settings=_settings)

    app.openapi = partial(custom_openapi, app)  # type: ignore[method-assign]

    return app


app = create_app()
