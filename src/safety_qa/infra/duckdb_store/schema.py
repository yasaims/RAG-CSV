"""DuckDB テーブル定義。

`accidents.duckdb`（CSV から再構築可能なキャッシュ）と `queries.duckdb`
（問い合わせの監査ログ）はファイルを分離するため、DDL も分けて定義する。
"""

from __future__ import annotations

from dataclasses import fields

import duckdb

from safety_qa.core.models import AccidentRecord

# accidents テーブルの列名・順序の単一情報源。DDL・SELECT列挙・行変換・INSERT の
# 列順をそれぞれ手書きすると、列を追加した際に一部だけ更新漏れが起きうる
# （AccidentRecord のフィールドと DB の列は 1:1 対応させる）。
RECORD_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(AccidentRecord))
# ドメインモデルには含まれない、埋め込み・差分検知用の付加列。
STORAGE_COLUMNS: tuple[str, ...] = (*RECORD_COLUMNS, "source_hash", "embedding")

_COLUMN_TYPES: dict[str, str] = {
    "id": "VARCHAR PRIMARY KEY",
    "date": "DATE",
    "time": "TIME",
    "shift": "VARCHAR",
    "location": "VARCHAR",
    "equipment": "VARCHAR",
    "severity": "VARCHAR",
    "worker_id": "VARCHAR",
    "worker_role": "VARCHAR",
    "worker_experience_years": "INTEGER",
    "description": "VARCHAR",
    "cause": "VARCHAR",
    "countermeasure": "VARCHAR",
    "source_hash": "VARCHAR",
}
# embedding は次元がランタイムの設定値なので _COLUMN_TYPES には含めず別扱いにする。
assert set(_COLUMN_TYPES) | {"embedding"} == set(STORAGE_COLUMNS)


def ensure_accidents_schema(con: duckdb.DuckDBPyConnection, embedding_dim: int) -> None:
    column_defs = ", ".join(f"{name} {_COLUMN_TYPES[name]}" for name in RECORD_COLUMNS)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS accidents (
            {column_defs},
            source_hash VARCHAR,
            embedding FLOAT[{embedding_dim}]
        )
    """)
    con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")


def ensure_queries_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id VARCHAR PRIMARY KEY,
            question VARCHAR,
            answer VARCHAR,
            citations VARCHAR,
            tool_name VARCHAR,
            tool_arguments VARCHAR,
            created_at TIMESTAMP,
            error_code VARCHAR,
            error_detail VARCHAR
        )
    """)
    # answer が NULL の行（失敗した問い合わせ）も監査ログとして記録する。
    # 既存の queries.duckdb には無い列なので、後方互換のため個別に追加する。
    con.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS error_code VARCHAR")
    con.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS error_detail VARCHAR")
    # 回答の根拠情報（解釈済み条件・母数・収録期間）。事故記録データが後から
    # 変わっても回答時点を再現できるよう、生成時点の値を JSON で凍結する
    # （`citations` と同じ理由）。既存行は解釈条件を復元できないため NULL。
    con.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS answer_basis VARCHAR")
