"""記録参照サービスのインターフェース。

API スキーマレイヤーはこの Protocol にのみ依存し、実装
（`AccidentRecordLookupService` およびテスト用フェイク）の差し替えに影響を受けない。
`QaService`（`core/qa_service.py`）と同じ役割分担。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from safety_qa.core.models import AccidentRecord, QueryRecordPage


class RecordService(Protocol):
    """citation の id・問い合わせの id から事故記録を参照するバックエンドの契約。"""

    async def get(self, record_id: str) -> AccidentRecord:
        """id で事故記録を取得する。存在しない場合は RecordNotFoundError。"""
        ...

    async def list_for_query(self, query_id: UUID, *, limit: int, offset: int) -> QueryRecordPage:
        """問い合わせが実際に対象とした母集団をページングして返す。

        保存済みの `tool_arguments` から検索クエリ・絞り込み条件を再構成する。
        問い合わせが存在しない場合は QueryNotFoundError。ツール未呼び出し、
        または途中で失敗した問い合わせ（`AnswerBasis` が無い）は空ページを返す。
        """
        ...
