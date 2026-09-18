"""BM25 + ベクトル検索のハイブリッド（RRF 統合）。`Retriever` ポートの実装。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from safety_qa.core.filters import RecordFilter
from safety_qa.core.fusion import fuse_with_ranks
from safety_qa.core.ports import AccidentRecordRepository, Embedder
from safety_qa.infra.search.bm25_index import BM25Index

logger = structlog.get_logger(__name__)

# fuse_with_ranks に渡す result_lists の並び順（0番目=BM25, 1番目=ベクトル）。
# FusedHit.ranks から経路別の順位を取り出す際の添字に使う。
_BM25_LIST_INDEX = 0
_VECTOR_LIST_INDEX = 1


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """融合後の 1 件と、各経路での順位・スコア（診断用）。

    `bm25_rank`/`vector_rank` はその経路の候補に入らなかった場合 None。
    スコアの尺度は経路ごとに異なる（`core.models.ScoredHit` 参照）ため、
    2 経路間で大小を比較しない。
    """

    record_id: str
    rrf_rank: int
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    vector_rank: int | None
    vector_score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """`search_with_diagnostics` の戻り値。ログ・評価レポート向けの内訳を持つ。"""

    record_ids: list[str]
    """融合後の最終結果（`hybrid_search` の戻り値と同一）。"""
    hits: list[RetrievalHit]
    bm25_ids: list[str]
    """BM25 側の候補（融合前、スコア降順）。経路別 recall の算出に使う。"""
    vector_ids: list[str]
    """ベクトル側の候補（融合前、スコア降順）。経路別 recall の算出に使う。"""
    filtered_count: int | None
    """`record_filter` 適用後の母集団件数。絞り込みなしは None。"""
    embed_ms: float
    bm25_ms: float
    vector_ms: float
    filter_ms: float
    duration_ms: float


class HybridRetriever:
    """`Retriever` ポートの実装。BM25 とベクトル検索の順位を RRF で統合する。"""

    def __init__(
        self,
        *,
        accidents: AccidentRecordRepository,
        embedder: Embedder,
        bm25_index: BM25Index,
    ) -> None:
        self._accidents = accidents
        self._embedder = embedder
        self._bm25_index = bm25_index

    def hybrid_search(
        self, query: str, top_n: int, *, record_filter: RecordFilter | None = None
    ) -> list[str]:
        return self.search_with_diagnostics(query, top_n, record_filter=record_filter).record_ids

    def search_with_diagnostics(
        self, query: str, top_n: int, *, record_filter: RecordFilter | None = None
    ) -> RetrievalDiagnostics:
        """`hybrid_search` と同じ検索を行い、経路別のスコア・所要時間を付けて返す。

        本番の回答生成（`RagQaService`）は `hybrid_search`（id リストのみ）を使う。
        診断つきの戻り値は運用ログ（`retrieval.completed`）と評価ハーネス
        （`evals/run.py`）が「BM25・ベクトルのどちらの検索経路が効いたか」を
        切り分けるために使う。
        """
        started_at = time.perf_counter()

        # BM25・ベクトル検索の両方を同じ絞り込み条件に限定してから RRF で統合する。
        # 条件を満たす id をあらかじめ求め、BM25 側は weight_mask による限定、
        # ベクトル側は SQL の WHERE による限定として、同じ条件を異なる仕組みで適用する。
        allowed_ids: list[str] | None = None
        filtered_count: int | None = None
        filter_started_at = time.perf_counter()
        if record_filter is not None and not record_filter.is_empty:
            allowed_ids = self._accidents.find_ids(record_filter)
            filtered_count = len(allowed_ids)
        filter_ms = _elapsed_ms(filter_started_at)

        if allowed_ids is not None and not allowed_ids:
            diagnostics = RetrievalDiagnostics(
                record_ids=[],
                hits=[],
                bm25_ids=[],
                vector_ids=[],
                filtered_count=filtered_count,
                embed_ms=0.0,
                bm25_ms=0.0,
                vector_ms=0.0,
                filter_ms=filter_ms,
                duration_ms=_elapsed_ms(started_at),
            )
            self._log_completed(top_n, diagnostics)
            return diagnostics

        bm25_started_at = time.perf_counter()
        bm25_hits = self._bm25_index.search(query, top_n=top_n * 2, allowed_ids=allowed_ids)
        bm25_ms = _elapsed_ms(bm25_started_at)

        embed_started_at = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embed_ms = _elapsed_ms(embed_started_at)

        vector_started_at = time.perf_counter()
        vector_hits = self._accidents.vector_search(
            query_vector, top_n=top_n * 2, record_filter=record_filter
        )
        vector_ms = _elapsed_ms(vector_started_at)

        bm25_ids = [h.record_id for h in bm25_hits]
        vector_ids = [h.record_id for h in vector_hits]
        bm25_scores = {h.record_id: h.score for h in bm25_hits}
        vector_scores = {h.record_id: h.score for h in vector_hits}

        fused = fuse_with_ranks([bm25_ids, vector_ids], top_n=top_n)
        hits = [
            RetrievalHit(
                record_id=hit.record_id,
                rrf_rank=rank,
                rrf_score=hit.rrf_score,
                bm25_rank=hit.ranks[_BM25_LIST_INDEX],
                bm25_score=bm25_scores.get(hit.record_id),
                vector_rank=hit.ranks[_VECTOR_LIST_INDEX],
                vector_score=vector_scores.get(hit.record_id),
            )
            for rank, hit in enumerate(fused, start=1)
        ]

        diagnostics = RetrievalDiagnostics(
            record_ids=[h.record_id for h in hits],
            hits=hits,
            bm25_ids=bm25_ids,
            vector_ids=vector_ids,
            filtered_count=filtered_count,
            embed_ms=embed_ms,
            bm25_ms=bm25_ms,
            vector_ms=vector_ms,
            filter_ms=filter_ms,
            duration_ms=_elapsed_ms(started_at),
        )
        self._log_completed(top_n, diagnostics)
        return diagnostics

    def _log_completed(self, top_n: int, diagnostics: RetrievalDiagnostics) -> None:
        """`retrieval.completed` を出す。フラットなスカラーのみ（per-hit 詳細は
        評価レポート側にのみ出し、本番ログには出さない。ADR 008 のフラット原則）。
        """
        top1 = diagnostics.hits[0] if diagnostics.hits else None
        top1_source: str | None = None
        if top1 is not None:
            in_bm25 = top1.bm25_rank is not None
            in_vector = top1.vector_rank is not None
            top1_source = "both" if in_bm25 and in_vector else ("bm25" if in_bm25 else "vector")
        overlap_count = sum(
            1 for h in diagnostics.hits if h.bm25_rank is not None and h.vector_rank is not None
        )
        logger.info(
            "retrieval.completed",
            top_n=top_n,
            result_count=len(diagnostics.record_ids),
            filtered_count=diagnostics.filtered_count,
            bm25_candidate_count=len(diagnostics.bm25_ids),
            vector_candidate_count=len(diagnostics.vector_ids),
            overlap_count=overlap_count,
            bm25_zero_hit=len(diagnostics.bm25_ids) == 0,
            top1_source=top1_source,
            top1_bm25_score=top1.bm25_score if top1 else None,
            top1_vector_score=top1.vector_score if top1 else None,
            embed_ms=round(diagnostics.embed_ms, 1),
            bm25_ms=round(diagnostics.bm25_ms, 1),
            vector_ms=round(diagnostics.vector_ms, 1),
            filter_ms=round(diagnostics.filter_ms, 1),
            duration_ms=round(diagnostics.duration_ms, 1),
        )


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000
