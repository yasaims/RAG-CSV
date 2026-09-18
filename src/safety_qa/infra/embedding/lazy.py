"""`Embedder` ポートの遅延ロード実装。

差分埋め込みで再埋め込みが 0 件のとき（CSV 変更なしでの再起動、
`python -m safety_qa.infra` での ETL 単体実行等）でも、埋め込みモデルの
実体（torch + SentenceTransformer）が起動の最初期に無条件でロードされて
いた。実際に `embed_query`/`embed_documents` が呼ばれるまでロードを遅延する。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from safety_qa.core.ports import Embedder


class LazyEmbedder:
    """`Embedder` ポートの遅延ロード版。初回呼び出し時にのみ `factory()` を実行する。"""

    def __init__(self, factory: Callable[[], Embedder]) -> None:
        self._factory = factory
        self._embedder: Embedder | None = None
        self._lock = threading.Lock()

    def _get(self) -> Embedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = self._factory()
        return self._embedder

    def embed_query(self, text: str) -> list[float]:
        return self._get().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get().embed_documents(texts)
