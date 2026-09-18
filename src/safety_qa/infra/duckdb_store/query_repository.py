"""DuckDB を用いた `QueryRepository` 実装。問い合わせの監査ログを兼ねる。

`accidents.duckdb`（再構築可能なキャッシュ）とは別ファイルに保存する。
CSV 更新に伴う ETL 再実行で監査ログを失わないようにするため。
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from uuid import UUID

import duckdb

from safety_qa.core.models import AnswerBasis, Citation, Query, QueryRecord


class DuckDbQueryRepository:
    """`queries.duckdb` へのアクセスをこのファイルに閉じる。"""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._con = connection
        self._lock = threading.Lock()

    def save(self, record: QueryRecord) -> None:
        citations_json = json.dumps([asdict(c) for c in record.citations], ensure_ascii=False)
        tool_arguments_json = json.dumps(record.tool_arguments, ensure_ascii=False)
        # 回答の根拠情報。ツール未呼び出し時は `basis` 自体が無いため NULL。
        basis_json = json.dumps(asdict(record.basis), ensure_ascii=False) if record.basis else None
        with self._lock:
            self._con.execute(
                """
                INSERT INTO queries
                    (id, question, answer, citations, tool_name, tool_arguments,
                     created_at, error_code, error_detail, answer_basis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(record.id),
                    record.question,
                    record.answer,
                    citations_json,
                    record.tool_name,
                    tool_arguments_json,
                    record.created_at,
                    record.error_code,
                    record.error_detail,
                    basis_json,
                ],
            )

    def get(self, query_id: UUID) -> Query | None:
        with self._lock:
            row = self._con.execute(
                "SELECT id, question, answer, citations, created_at, answer_basis FROM queries "
                "WHERE id = ? AND error_code IS NULL",
                [str(query_id)],
            ).fetchone()
        if row is None:
            return None
        id_, question, answer, citations_json, created_at, basis_json = row
        citations = [Citation(**d) for d in json.loads(citations_json)]
        # `answer_basis` 列を持たない旧行は NULL のまま復元する（解釈条件を
        # 遡って再現できないのは既知の限界）。
        basis = AnswerBasis(**json.loads(basis_json)) if basis_json else None
        return Query(
            id=UUID(id_),
            question=question,
            answer=answer,
            citations=citations,
            basis=basis,
            created_at=created_at,
        )

    def get_record(self, query_id: UUID) -> QueryRecord | None:
        with self._lock:
            row = self._con.execute(
                "SELECT id, question, answer, citations, tool_name, tool_arguments, "
                "created_at, error_code, error_detail, answer_basis FROM queries WHERE id = ?",
                [str(query_id)],
            ).fetchone()
        if row is None:
            return None
        return _row_to_query_record(row)

    def list_records(self, *, limit: int, before_id: UUID | None = None) -> list[QueryRecord]:
        columns = (
            "id, question, answer, citations, tool_name, tool_arguments, "
            "created_at, error_code, error_detail, answer_basis"
        )
        with self._lock:
            if before_id is not None:
                rows = self._con.execute(
                    f"SELECT {columns} FROM queries WHERE id < ? ORDER BY id DESC LIMIT ?",
                    [str(before_id), limit],
                ).fetchall()
            else:
                rows = self._con.execute(
                    f"SELECT {columns} FROM queries ORDER BY id DESC LIMIT ?",
                    [limit],
                ).fetchall()
        return [_row_to_query_record(row) for row in rows]


def _row_to_query_record(row: tuple) -> QueryRecord:
    (
        id_,
        question,
        answer,
        citations_json,
        tool_name,
        tool_arguments_json,
        created_at,
        error_code,
        error_detail,
        basis_json,
    ) = row
    citations = [Citation(**d) for d in json.loads(citations_json)] if citations_json else []
    tool_arguments = json.loads(tool_arguments_json) if tool_arguments_json else {}
    basis = AnswerBasis(**json.loads(basis_json)) if basis_json else None
    return QueryRecord(
        id=UUID(id_),
        question=question,
        created_at=created_at,
        answer=answer,
        citations=citations,
        basis=basis,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        error_code=error_code,
        error_detail=error_detail,
    )
