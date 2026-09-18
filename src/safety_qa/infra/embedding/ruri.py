"""Ruri v3 埋め込みモデルのラッパ。

`cl-nagoya/ruri-v3-130m` は「1+3 prefix scheme」を採用しており、クエリと文書で
異なるプレフィックスを付与する必要がある。
"""

from __future__ import annotations

import threading

from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "検索クエリ: "
DOCUMENT_PREFIX = "検索文書: "


class RuriEmbedder:
    """`Embedder` ポートの実装。CPU 推論のみを前提とする（SPEC.md の GPU なし制約）。"""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")
        actual_dim = self._model.get_embedding_dimension()
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"想定次元と不一致: 期待={expected_dim} 実際={actual_dim}"
                "（DuckDB の embedding 列定義を見直す必要がある）"
            )
        # SPEC.md の 4 コア前提では並行実行しても総スループットは上がらず、
        # DuckDbAccidentRepository と同じ理由でモデル呼び出しを直列化する
        # （SentenceTransformer.encode を複数スレッドから同時に呼ぶのは想定外）。
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            vector = self._model.encode(
                QUERY_PREFIX + text, convert_to_numpy=True, show_progress_bar=False
            )
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        with self._lock:
            vectors = self._model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
