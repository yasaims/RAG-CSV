"""Reciprocal Rank Fusion（RRF）。

BM25 のスコアとコサイン類似度はスケールが異なり、単純な加重平均では正規化方法の
選択が恣意的になる。RRF は各手法の順位のみを使うため、チューニングすべき
パラメータが少なく実装・説明のコストが低い。純粋関数なので外部依存を持たず
core に置く。

`fuse_with_ranks` は統合後のスコアに加え、各入力リストでの順位も保持して返す。
これは診断用（どちらの検索経路が効いたかをログ・評価レポートで確認する）であり、
統合方法自体を順位ベースの RRF から加重和へ変える意図ではない。
"""

from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60  # Cormack et al. 2009 の既定値をそのまま採用


@dataclass(frozen=True, slots=True)
class FusedHit:
    """統合後の 1 件。`ranks` は `fuse_with_ranks` に渡した `result_lists` と
    同じ並びの順位（1 始まり）。その経路に出現しなかった場合は None。
    """

    record_id: str
    rrf_score: float
    ranks: tuple[int | None, ...]


def fuse_with_ranks(result_lists: list[list[str]], top_n: int, k: int = RRF_K) -> list[FusedHit]:
    """複数の検索結果（順位順の record_id リスト）を順位の逆数和で統合し、
    上位 top_n 件を `FusedHit`（統合スコア + 各経路での順位）として返す。
    """
    scores: dict[str, float] = {}
    ranks_by_id: dict[str, list[int | None]] = {}
    for list_index, record_ids in enumerate(result_lists):
        for rank, record_id in enumerate(record_ids, start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
            ranks_by_id.setdefault(record_id, [None] * len(result_lists))[list_index] = rank
    ranked_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        FusedHit(record_id=record_id, rrf_score=score, ranks=tuple(ranks_by_id[record_id]))
        for record_id, score in ranked_ids
    ]


def reciprocal_rank_fusion(result_lists: list[list[str]], top_n: int, k: int = RRF_K) -> list[str]:
    """複数の検索結果（順位順の record_id リスト）を順位の逆数和で統合し、
    上位 top_n 件の record_id を返す。
    """
    return [hit.record_id for hit in fuse_with_ranks(result_lists, top_n, k)]
