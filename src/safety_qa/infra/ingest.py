"""CSV -> DuckDB の起動時 ETL。

- CSV を正データとし、DuckDB 側は再構築可能なキャッシュとして扱う
- 正規化: cause / countermeasure の空欄を「記載なし」化
- エンリッチ: time から日勤/夜勤列を導出
- 埋め込み: description + cause + countermeasure を連結し Ruri v3 でベクトル化
- 差分埋め込み: 埋め込み対象テキストのハッシュが変わっていない行は埋め込みを
  再計算せず既存ベクトルを再利用する（メタデータ自体は毎回 UPSERT して最新化する）。
- 削除の反映: CSV に存在しない id は DELETE する（CSV を正とする原則の徹底）
- 冪等性: id を主キーに UPSERT。同じ CSV から複数回実行しても件数・埋め込みは変わらない
- モデル不一致検知: 埋め込みモデル名・次元を source_hash に含める。モデルや
  次元を変更すると全行のハッシュが変わり、差分埋め込みの仕組みがそのまま
  「変更された行を再埋め込みする」動作として全件再計算になる。
  使用したモデル名は meta テーブルに記録する（監査用途）
- DB への書き込み（削除・UPSERT・meta 更新）は 1 トランザクションにまとめる。
  埋め込み計算中の失敗で DB が一部だけ削除・更新された中途半端な状態に
  ならないようにするため
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dtime
from pathlib import Path

import duckdb
import structlog

from safety_qa.core.ports import Embedder
from safety_qa.core.search_text import build_search_text
from safety_qa.infra.duckdb_store.schema import STORAGE_COLUMNS, ensure_accidents_schema

logger = structlog.get_logger(__name__)

NIGHT_SHIFT_START = dtime(20, 0)
NIGHT_SHIFT_END = dtime(6, 0)
NOT_RECORDED = "記載なし"

_UPDATABLE_COLUMNS = tuple(c for c in STORAGE_COLUMNS if c != "id")


@dataclass
class IngestResult:
    row_count: int
    embedded_count: int
    reused_count: int
    deleted_count: int


def _fill_blank(value: str) -> str:
    return value.strip() if value.strip() else NOT_RECORDED


def _derive_shift(t: dtime) -> str:
    if t >= NIGHT_SHIFT_START or t < NIGHT_SHIFT_END:
        return "夜勤"
    return "日勤"


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _embedding_text(row: dict[str, str]) -> str:
    # BM25 コーパス（infra/search/bm25_index.py）と同じ定義（core/search_text.py）を
    # 使う。埋め込み・BM25 の双方が異なる文書空間を索引してしまうのを防ぐ。
    return build_search_text(
        description=row["description"],
        cause=_fill_blank(row["cause"]),
        countermeasure=_fill_blank(row["countermeasure"]),
    )


def _source_hash(text: str, *, model_name: str, dim: int) -> str:
    # モデル名・次元をハッシュに含める（指摘12対応）。埋め込みモデルを変更すると
    # 全行のハッシュが変わり、下の run_ingest の差分判定がそのまま全件を
    # 「再埋め込みが必要な行」として扱う。専用の不一致検知は不要になる。
    return hashlib.sha256(f"{model_name}|{dim}|{text}".encode()).hexdigest()


def _row_values(row: dict[str, str], *, source_hash: str, vector: list[float]) -> dict[str, object]:
    t = dtime.fromisoformat(row["time"])
    return {
        "id": row["id"],
        "date": row["date"],
        "time": row["time"],
        "shift": _derive_shift(t),
        "location": row["location"],
        "equipment": row["equipment"],
        "severity": row["severity"],
        "worker_id": row["worker_id"],
        "worker_role": row["worker_role"],
        "worker_experience_years": int(row["worker_experience_years"]),
        "description": row["description"],
        "cause": _fill_blank(row["cause"]),
        "countermeasure": _fill_blank(row["countermeasure"]),
        "source_hash": source_hash,
        "embedding": vector,
    }


def _upsert_row(
    con: duckdb.DuckDBPyConnection,
    row: dict[str, str],
    source_hash: str,
    vector: list[float],
    embedding_dim: int,
) -> None:
    # 列名・順序は STORAGE_COLUMNS（schema.py）を単一の情報源とする。列を
    # 追加した場合はここではなく schema.py と _row_values だけを更新すればよい。
    values = _row_values(row, source_hash=source_hash, vector=vector)
    columns_sql = ", ".join(STORAGE_COLUMNS)
    placeholders_sql = ", ".join(
        f"?::FLOAT[{embedding_dim}]" if column == "embedding" else "?" for column in STORAGE_COLUMNS
    )
    update_sql = ", ".join(f"{column} = excluded.{column}" for column in _UPDATABLE_COLUMNS)
    con.execute(
        f"""
        INSERT INTO accidents ({columns_sql})
        VALUES ({placeholders_sql})
        ON CONFLICT (id) DO UPDATE SET {update_sql}
        """,
        [values[column] for column in STORAGE_COLUMNS],
    )


def run_ingest(
    *,
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    embedder: Embedder,
    embedding_model_name: str,
    embedding_dim: int,
) -> IngestResult:
    started_at = time.perf_counter()
    ensure_accidents_schema(con, embedding_dim)

    # フェーズ1: 読み込み・差分判定・埋め込み計算。DB は変更しない（読み取りのみ）。
    rows = _read_rows(csv_path)
    csv_ids = {row["id"] for row in rows}

    existing_ids = {r[0] for r in con.execute("SELECT id FROM accidents").fetchall()}
    deleted_ids = existing_ids - csv_ids

    existing_by_id = {
        r[0]: (r[1], r[2])
        for r in con.execute("SELECT id, source_hash, embedding FROM accidents").fetchall()
    }

    to_embed: list[tuple[dict[str, str], str]] = []
    to_embed_texts: list[str] = []
    to_reuse: list[tuple[dict[str, str], str, list[float]]] = []
    for row in rows:
        text = _embedding_text(row)
        source_hash = _source_hash(text, model_name=embedding_model_name, dim=embedding_dim)
        previous = existing_by_id.get(row["id"])
        if previous is not None and previous[0] == source_hash:
            to_reuse.append((row, source_hash, previous[1]))
        else:
            to_embed.append((row, source_hash))
            to_embed_texts.append(text)

    # 埋め込み計算は時間のかかる処理（CPU/ネットワーク依存）なので、DB への
    # 書き込みトランザクションの外で行う。
    vectors = embedder.embed_documents(to_embed_texts) if to_embed_texts else []

    # フェーズ2: DB への書き込み。削除・UPSERT・meta 更新を 1 トランザクションに
    # まとめ、途中で失敗した場合に DB が中途半端な状態にならないようにする。
    con.begin()
    try:
        if deleted_ids:
            con.execute("DELETE FROM accidents WHERE list_contains(?, id)", [list(deleted_ids)])
        for (row, source_hash), vector in zip(to_embed, vectors, strict=True):
            _upsert_row(con, row, source_hash, vector, embedding_dim)
        for row, source_hash, vector in to_reuse:
            # メタデータ（location 修正等）は毎回反映するが、埋め込みは既存ベクトルを再利用する。
            _upsert_row(con, row, source_hash, vector, embedding_dim)
        con.execute(
            "INSERT OR REPLACE INTO meta VALUES ('embedding_model', ?)", [embedding_model_name]
        )
        # `GET /v1/queries/{id}/records` の `as_of`（citations との乖離検知用）。
        con.execute(
            "INSERT OR REPLACE INTO meta VALUES ('ingested_at', ?)",
            [datetime.now(UTC).isoformat()],
        )
    except Exception:
        con.rollback()
        raise
    else:
        con.commit()

    row_count = con.execute("SELECT count(*) FROM accidents").fetchone()[0]
    result = IngestResult(
        row_count=row_count,
        embedded_count=len(to_embed),
        reused_count=len(to_reuse),
        deleted_count=len(deleted_ids),
    )
    logger.info(
        "ingest.completed",
        row_count=result.row_count,
        embedded_count=result.embedded_count,
        reused_count=result.reused_count,
        deleted_count=result.deleted_count,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
    )
    return result
