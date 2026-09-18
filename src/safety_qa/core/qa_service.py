"""QA サービスのインターフェース。

API スキーマレイヤーはこの Protocol にのみ依存し、実装（`RagQaService` および
テスト用フェイク）の差し替えに影響を受けない。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from safety_qa.core.models import Query, QueryRecord


class QaService(Protocol):
    """自然言語の問い合わせを扱うバックエンドの契約。"""

    async def ask(self, question: str) -> Query:
        """質問を受け付け、回答を生成して問い合わせリソースを返す。"""
        ...

    async def get(self, query_id: UUID) -> Query:
        """過去の問い合わせを id で取得する。存在しない場合は QueryNotFoundError。"""
        ...

    async def list_queries(self, *, limit: int, before_id: UUID | None = None) -> list[QueryRecord]:
        """過去の問い合わせを新しい順に一覧する（成功・失敗の両方を含む）。"""
        ...
