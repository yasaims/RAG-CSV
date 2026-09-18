"""集計の内訳軸（`group_by`）の語彙と検証。

絞り込み条件（`core.filters.RecordFilter`）とは別の概念のため独立させる。
SQL 式へのマッピングは infra 側（`infra/duckdb_store/accident_repository.py`）が担う。
"""

from __future__ import annotations

from safety_qa.core.exceptions import InvalidToolArgumentsError

# 件数の多い順に並べる軸（実務で「多い順トップN」を知りたい対象）。
CATEGORICAL_GROUP_BY: tuple[str, ...] = (
    "equipment",
    "location",
    "severity",
    "shift",
    "worker_role",
)
# 時系列として key 昇順に並べる軸（推移を時間順に読むもの）。
TIME_SERIES_GROUP_BY: tuple[str, ...] = ("month", "year")

VALID_GROUP_BY: tuple[str, ...] = (*CATEGORICAL_GROUP_BY, *TIME_SERIES_GROUP_BY)


def validate_group_by(value: str | None) -> str | None:
    """`group_by` 引数を検証する。未指定（None）は内訳なしとして許可する。"""
    if value is None:
        return None
    if value not in VALID_GROUP_BY:
        raise InvalidToolArgumentsError(
            f"未知の group_by: {value!r}（有効値: {list(VALID_GROUP_BY)}）"
        )
    return value
