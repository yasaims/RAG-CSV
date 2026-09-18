"""DI: QaService / RecordService の解決。

実体は `app.state.container`（infra/container.py の composition root）が持つ。
テストは `app.dependency_overrides[get_qa_service]` 等でフェイクを注入する。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from safety_qa.core.qa_service import QaService
from safety_qa.core.record_service import RecordService


def get_qa_service(request: Request) -> QaService:
    return request.app.state.container.qa_service


def get_record_service(request: Request) -> RecordService:
    return request.app.state.container.record_service


QaServiceDep = Annotated[QaService, Depends(get_qa_service)]
RecordServiceDep = Annotated[RecordService, Depends(get_record_service)]
