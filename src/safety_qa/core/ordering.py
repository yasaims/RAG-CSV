"""`get_many` の順序復元。

`AccidentRecordRepository.get_many` は id 一覧に対応する記録を順不同で返す
契約のため、呼び出し側が渡した順序を復元する処理をここに置く。
`core/rag_service.py` は同じロジックを `_fetch_ordered` として持つが、
`core/record_lookup.py`（`GET /v1/queries/{id}/records` の search 由来経路）が
同じ復元を必要とするため、共有の純関数として切り出す。
"""

from __future__ import annotations

from safety_qa.core.models import AccidentRecord


def restore_order(records: list[AccidentRecord], record_ids: list[str]) -> list[AccidentRecord]:
    """`records` を `record_ids` と同じ順序に並べ替える。存在しない id は無視する。"""
    records_by_id = {r.id: r for r in records}
    return [records_by_id[rid] for rid in record_ids if rid in records_by_id]
