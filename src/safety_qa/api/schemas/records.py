"""`/v1/records` のレスポンスモデル。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from safety_qa.core.models import AccidentRecord


class RecordResponse(BaseModel):
    """事故記録 1 件。

    `AccidentRecord` の全列ではない。`worker_id` は個人識別子そのものであり、
    フェーズ1は認証を持たない（SPEC.md 実行前提）ため含めない。`shift`/
    `worker_role`/`worker_experience_years` は集計（`aggregate_accident_records`）の
    `group_by` 軸であり、内訳の検算に使えるよう含める。
    """

    id: str = Field(description="事故記録の id")
    date: str = Field(description="発生日")
    time: str = Field(description="発生時刻")
    shift: str = Field(description="勤務帯（日勤/夜勤）")
    location: str = Field(description="発生場所")
    equipment: str = Field(description="関連する設備")
    severity: str = Field(description="重大度")
    worker_role: str = Field(description="作業者の職種")
    worker_experience_years: int = Field(description="作業者の経験年数")
    description: str = Field(description="事故の内容")
    cause: str = Field(description="原因（記載が無い記録では空文字）")
    countermeasure: str = Field(description="対策（記載が無い記録では空文字）")

    @classmethod
    def from_core(cls, record: AccidentRecord) -> RecordResponse:
        return cls(
            id=record.id,
            date=record.date,
            time=record.time,
            shift=record.shift,
            location=record.location,
            equipment=record.equipment,
            severity=record.severity,
            worker_role=record.worker_role,
            worker_experience_years=record.worker_experience_years,
            description=record.description,
            cause=record.cause,
            countermeasure=record.countermeasure,
        )
