"""BM25（bm25s + fugashi）によるキーワード検索インデックス。

インメモリインデックスで、起動時に `accidents.duckdb` の全件から一度だけ構築する
（CSV 更新への追随は起動時 ETL と同じタイミングでプロセス再起動により行う）。
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

import bm25s
import numpy as np

from safety_qa.core.models import AccidentRecord, ScoredHit
from safety_qa.core.search_text import build_search_text
from safety_qa.infra.search.tokenize_ja import tokenize


@dataclass
class BM25Index:
    retriever: bm25s.BM25
    ids: list[str]
    # 語 -> その語を含む文書数。excerpt のヒット欄推定（discriminative_terms）専用
    # の付帯情報で、検索本体（search）は使わない。
    document_frequency: dict[str, int] = field(default_factory=dict)
    _index_by_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self._index_by_id = {record_id: idx for idx, record_id in enumerate(self.ids)}

    def search(
        self, query: str, top_n: int, *, allowed_ids: Collection[str] | None = None
    ) -> list[ScoredHit]:
        """クエリに対する上位 top_n 件をスコア降順で返す。

        bm25s.BM25.retrieve は常に k 件を返す（クエリと語彙が一切重ならない
        文書もスコア 0 で埋めて返す）ため、スコア 0 の文書は「一致していない」
        として除外する。`allowed_ids` を指定すると、対象外の文書のスコアを
        `weight_mask` で 0 にしてから検索する（絞り込み条件との AND）。
        """
        if not self.ids:
            return []
        weight_mask = None
        if allowed_ids is not None:
            weight_mask = np.zeros(len(self.ids), dtype=np.float32)
            for record_id in allowed_ids:
                idx = self._index_by_id.get(record_id)
                if idx is not None:
                    weight_mask[idx] = 1.0
            if not weight_mask.any():
                return []
        query_tokens = tokenize(query)
        doc_ids, scores = self.retriever.retrieve(
            [query_tokens],
            k=min(top_n, len(self.ids)),
            show_progress=False,
            weight_mask=weight_mask,
        )
        return [
            ScoredHit(record_id=self.ids[idx], score=float(score))
            for idx, score in zip(doc_ids[0], scores[0], strict=True)
            if score > 0.0
        ]

    def discriminative_terms(
        self, query: str, *, max_document_frequency_ratio: float = 0.3
    ) -> list[str]:
        """query をトークナイズし、コーパスの一定割合を超える文書に出現する
        高頻度語（「作業」「確認」等、事故種別を区別しない語）を除いた語を返す。

        excerpt のヒット欄推定専用。除外せずに使うと、質問文フォールバック
        （`query` 未指定時の質問文そのもの）でほぼ全レコードに当たる語が
        「ヒット理由」として誤って表示される。
        """
        if not self.ids:
            return []
        # 小規模コーパス（極端には1件）では割合換算の閾値が0になり、実在する
        # 語まで「高頻度」として全滅させてしまう。最低でも1件までは許容する。
        threshold = max(1, int(len(self.ids) * max_document_frequency_ratio))
        seen: set[str] = set()
        terms: list[str] = []
        for token in tokenize(query):
            if token in seen:
                continue
            seen.add(token)
            if self.document_frequency.get(token, 0) <= threshold:
                terms.append(token)
        return terms


def build_bm25_index(records: list[AccidentRecord]) -> BM25Index:
    ids = [r.id for r in records]
    retriever = bm25s.BM25()
    document_frequency: dict[str, int] = {}
    if ids:
        corpus_tokens = [
            tokenize(
                build_search_text(
                    description=r.description, cause=r.cause, countermeasure=r.countermeasure
                )
            )
            for r in records
        ]
        retriever.index(corpus_tokens)
        for tokens in corpus_tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
    return BM25Index(retriever=retriever, ids=ids, document_frequency=document_frequency)
