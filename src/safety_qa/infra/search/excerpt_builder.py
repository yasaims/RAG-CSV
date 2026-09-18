"""`ExcerptBuilder` ポートの実装。BM25 の語彙統計を使ってヒット欄を推定する。"""

from __future__ import annotations

from safety_qa.core.excerpt import build_excerpt
from safety_qa.core.models import AccidentRecord
from safety_qa.infra.search.bm25_index import BM25Index


class BM25ExcerptBuilder:
    def __init__(self, bm25_index: BM25Index) -> None:
        self._bm25_index = bm25_index

    def build(self, record: AccidentRecord, query: str | None) -> tuple[str | None, str]:
        terms = self._bm25_index.discriminative_terms(query) if query is not None else []
        return build_excerpt(
            description=record.description,
            cause=record.cause,
            countermeasure=record.countermeasure,
            terms=terms,
        )
