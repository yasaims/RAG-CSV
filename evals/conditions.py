"""測定条件（`MeasurementConditions`）。ベースラインとの比較可否判定に使う。

指標の値を決めるものを条件として明示する。「スイートの実装特性（決定的かどうか）」
を比較可否の判断根拠にしない。BM25 + 全件スキャンのベクトル検索という現行実装は
SPEC.md が近似最近傍探索への移行を明記しており、実装特性への依存は移行時に
静かに無効化する。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# 全条件の一致をもって比較可能とする。スイートごとの取捨選択はしない
# （「この条件は無視してよい」という判断こそが実装特性への依存を持ち込む経路のため）。
ALL_CONDITION_KEYS: tuple[str, ...] = (
    "model",
    "runner",
    "corpus_version",
    "dataset_version",
    "top_n",
    "repeat",
    "evaluation_date",
)


@dataclass(frozen=True, slots=True)
class MeasurementConditions:
    """指標の値を決める測定条件。この一致をもって比較可能とする。"""

    model: str
    runner: str
    corpus_version: str
    dataset_version: str
    top_n: int
    # llm スイートのみ意味を持つ（同一質問の試行回数）。retrieval では None。
    repeat: int | None = None
    # llm スイートはシステムプロンプトへ「本日」を注入し、相対日付
    # （expected_arguments の解決）もこれを基準にするため、出力がこの日付に
    # 依存する。`temperature=0`/`seed=0` と同じ「制御変数の固定」として、
    # 実行日（wall clock）ではなく評価実行時点のコーパス最終日を使う
    # （`evals/run.py` の `_run_llm_suite`）。ISO 8601 の日付文字列（YYYY-MM-DD）。
    # retrieval は日付に依存しないため None。
    evaluation_date: str | None = None

    def mismatches(self, other: MeasurementConditions) -> list[str]:
        """`other`（現在の実行）との不一致キーを `"key: old -> new"` の形で列挙する。"""
        return [
            f"{key}: {old!r} -> {new!r}"
            for key in ALL_CONDITION_KEYS
            for old, new in [(getattr(self, key), getattr(other, key))]
            if old != new
        ]


def corpus_version(csv_path: Path) -> str:
    """検索対象データ（事故記録 CSV）の版識別子。

    `evals/schema.py` の `dataset_version`（評価質問集の版）とは別軸。事故記録が
    増減しても評価質問集は変わらないため、これを分けないと「記録追加による
    Recall 変動」が「品質低下」として比較され続け、ベースライン更新のたびに
    劣化が基準へ取り込まれる。
    """
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()[:12]
