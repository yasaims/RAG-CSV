"""DuckDB を用いた `AccidentRecordRepository` 実装。

生 SQL をこのファイルに閉じ、`core` 層には意味のある操作の単位だけを見せる。
"""

from __future__ import annotations

import threading
from datetime import date, datetime

import duckdb

from safety_qa.core.aggregation import TIME_SERIES_GROUP_BY
from safety_qa.core.filters import RecordFilter, severities_at_or_above
from safety_qa.core.models import AccidentRecord, AggregateGroup, AggregateResult, ScoredHit
from safety_qa.infra.duckdb_store.schema import RECORD_COLUMNS

# group_by の値（core.aggregation.VALID_GROUP_BY で検証済み）から SQL 式へのマッピング。
# 列名そのままの軸はそのまま列参照、month/year は date から派生する式にする。
_GROUP_BY_EXPRESSIONS: dict[str, str] = {
    "equipment": "equipment",
    "location": "location",
    "severity": "severity",
    "shift": "shift",
    "worker_role": "worker_role",
    "month": "strftime(date, '%Y-%m')",
    "year": "strftime(date, '%Y')",
}

# SELECT 列挙は RECORD_COLUMNS（AccidentRecord のフィールドと 1:1）を単一の
# 情報源とする。DDL・行変換・INSERT がそれぞれ独自に列を手書きして食い違う
# のを防ぐ（schema.py 参照）。
_COLUMNS = ", ".join(RECORD_COLUMNS)


def _row_to_record(row: tuple) -> AccidentRecord:
    values = dict(zip(RECORD_COLUMNS, row, strict=True))
    values["date"] = values["date"].isoformat()
    values["time"] = values["time"].isoformat(timespec="minutes")
    return AccidentRecord(**values)


def _escape_like(value: str) -> str:
    """LIKE パターンのメタ文字（% _ \\）をエスケープする。

    LLM が生成した文字列をそのまま部分一致の値として使うため、意図しない
    ワイルドカード展開（例: 値に `%` を含む設備名）を防ぐ。
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_where(record_filter: RecordFilter | None) -> tuple[str, list[object]]:
    """`RecordFilter` から WHERE 句と束縛パラメータを組み立てる（集計・検索で共用）。

    `record_filter` が None、またはいずれの軸も指定されていない場合は
    絞り込みなし（WHERE 句なし）として扱う。
    """
    conditions: list[str] = []
    params: list[object] = []
    if record_filter is None:
        return "", params
    if record_filter.severity is not None:
        conditions.append("severity = ?")
        params.append(record_filter.severity)
    if record_filter.date_from is not None:
        conditions.append("date >= ?")
        params.append(record_filter.date_from)
    if record_filter.date_to is not None:
        conditions.append("date <= ?")
        params.append(record_filter.date_to)
    if record_filter.equipment is not None:
        conditions.append("equipment LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(record_filter.equipment)}%")
    if record_filter.location is not None:
        conditions.append("location LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(record_filter.location)}%")
    if record_filter.shift is not None:
        conditions.append("shift = ?")
        params.append(record_filter.shift)
    if record_filter.severity_min is not None:
        allowed = severities_at_or_above(record_filter.severity_min)
        conditions.append(f"severity IN ({', '.join('?' for _ in allowed)})")
        params.extend(allowed)
    if record_filter.worker_role is not None:
        conditions.append("worker_role LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(record_filter.worker_role)}%")
    if record_filter.experience_min is not None:
        conditions.append("worker_experience_years >= ?")
        params.append(record_filter.experience_min)
    if record_filter.experience_max is not None:
        conditions.append("worker_experience_years <= ?")
        params.append(record_filter.experience_max)
    if record_filter.hour_from is not None and record_filter.hour_to is not None:
        if record_filter.hour_from <= record_filter.hour_to:
            conditions.append("hour(time) BETWEEN ? AND ?")
            params.extend([record_filter.hour_from, record_filter.hour_to])
        else:
            # 日をまたぐ時間帯（例: 22時〜翌5時）
            conditions.append("(hour(time) >= ? OR hour(time) <= ?)")
            params.extend([record_filter.hour_from, record_filter.hour_to])
    elif record_filter.hour_from is not None:
        conditions.append("hour(time) >= ?")
        params.append(record_filter.hour_from)
    elif record_filter.hour_to is not None:
        conditions.append("hour(time) <= ?")
        params.append(record_filter.hour_to)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


class DuckDbAccidentRepository:
    """`accidents.duckdb` へのアクセスをこのファイルに閉じる。

    DuckDB の単一コネクションを複数スレッドから叩くのは安全でないため、
    呼び出しをロックで直列化する。書き込みは起動時 ETL のみで、読み取りも
    数十〜数千件規模を想定するため直列化のコストは無視できる。
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._con = connection
        self._lock = threading.Lock()

    def get_many(self, record_ids: list[str]) -> list[AccidentRecord]:
        if not record_ids:
            return []
        with self._lock:
            rows = self._con.execute(
                f"SELECT {_COLUMNS} FROM accidents WHERE list_contains(?, id)", [record_ids]
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_all(self) -> list[AccidentRecord]:
        with self._lock:
            rows = self._con.execute(f"SELECT {_COLUMNS} FROM accidents ORDER BY id").fetchall()
        return [_row_to_record(row) for row in rows]

    def vector_search(
        self, query_vector: list[float], top_n: int, *, record_filter: RecordFilter | None = None
    ) -> list[ScoredHit]:
        dim = len(query_vector)
        where, where_params = _build_where(record_filter)
        with self._lock:
            rows = self._con.execute(
                f"""
                SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{dim}]) AS score
                FROM accidents
                {where}
                ORDER BY score DESC
                LIMIT ?
                """,
                [query_vector, *where_params, top_n],
            ).fetchall()
        return [ScoredHit(record_id=row[0], score=row[1]) for row in rows]

    def find_ids(self, record_filter: RecordFilter) -> list[str]:
        # record_filter は core.tools.parse_aggregate_arguments が構築時点で検証済み
        # （RecordFilter.__post_init__ が enum を、date.fromisoformat が日付形式を検証）。
        where, params = _build_where(record_filter)
        with self._lock:
            rows = self._con.execute(
                f"SELECT id FROM accidents {where} ORDER BY id", params
            ).fetchall()
        return [row[0] for row in rows]

    def list_records(
        self, record_filter: RecordFilter, *, limit: int, offset: int
    ) -> list[AccidentRecord]:
        # `date, id` の全順序で LIMIT/OFFSET する。offset ページング中に重複・
        # 欠落が起きないよう安定した順序が契約（core/ports.py 参照）。
        where, params = _build_where(record_filter)
        with self._lock:
            rows = self._con.execute(
                f"SELECT {_COLUMNS} FROM accidents {where} ORDER BY date, id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def aggregate(
        self, record_filter: RecordFilter, *, group_by: str | None = None
    ) -> AggregateResult:
        ids = self.find_ids(record_filter)
        groups = self._aggregate_groups(record_filter, group_by) if group_by is not None else []
        return AggregateResult(count=len(ids), record_ids=ids, groups=groups)

    def count(self, record_filter: RecordFilter) -> int:
        where, params = _build_where(record_filter)
        with self._lock:
            row = self._con.execute(f"SELECT count(*) FROM accidents {where}", params).fetchone()
        return row[0]

    def date_range(self) -> tuple[date, date] | None:
        with self._lock:
            row = self._con.execute("SELECT min(date), max(date) FROM accidents").fetchone()
        if row is None or row[0] is None:
            return None
        return row[0], row[1]

    def ingested_at(self) -> datetime | None:
        with self._lock:
            row = self._con.execute("SELECT value FROM meta WHERE key = 'ingested_at'").fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(row[0])

    def _aggregate_groups(self, record_filter: RecordFilter, group_by: str) -> list[AggregateGroup]:
        expr = _GROUP_BY_EXPRESSIONS[group_by]
        where, params = _build_where(record_filter)
        # 時系列（month/year）は key 昇順、それ以外は「多い順トップN」が知りたい
        # 実務ニーズに合わせて件数降順・key 昇順で並べる。
        order = "k ASC" if group_by in TIME_SERIES_GROUP_BY else "c DESC, k ASC"
        with self._lock:
            rows = self._con.execute(
                f"""
                SELECT {expr} AS k, count(*) AS c, list(id) AS ids
                FROM accidents
                {where}
                GROUP BY k
                ORDER BY {order}
                """,
                params,
            ).fetchall()
        return [
            AggregateGroup(key=key, count=count, record_ids=list(ids)) for key, count, ids in rows
        ]
