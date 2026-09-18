"""検索対象テキストの定義。

BM25 インデックスとベクトル埋め込みの両方が「どの文字列を検索対象とするか」を
個別に定義すると、一方だけを変更した際に検索器ごとに異なる文書空間を索引して
しまう（RRF が異なる母集団の順位を統合することになる）。ここに定義を集約し、
`infra/ingest.py`（埋め込み用テキスト）と `infra/search/bm25_index.py`
（BM25 コーパス）の両方から呼ぶ。
"""

from __future__ import annotations


def build_search_text(*, description: str, cause: str, countermeasure: str) -> str:
    """事故記録の検索対象テキスト（description + cause + countermeasure）を組み立てる。"""
    return " ".join([description, cause, countermeasure])
