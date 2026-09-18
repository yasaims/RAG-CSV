"""組み立て（composition root）。

`core` はここでしか実体化されず、`api` 層はこのモジュールが返す `Container` を
通じてのみ実装（DuckDB / Ollama / 埋め込みモデル）に触れる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import duckdb

from safety_qa.core.ports import AccidentRecordRepository, ChatModel, Embedder
from safety_qa.core.qa_service import QaService
from safety_qa.core.rag_service import RagQaService
from safety_qa.core.record_lookup import AccidentRecordLookupService
from safety_qa.core.record_service import RecordService
from safety_qa.infra.clock import business_today
from safety_qa.infra.duckdb_store.accident_repository import DuckDbAccidentRepository
from safety_qa.infra.duckdb_store.query_repository import DuckDbQueryRepository
from safety_qa.infra.duckdb_store.schema import ensure_accidents_schema, ensure_queries_schema
from safety_qa.infra.embedding.lazy import LazyEmbedder
from safety_qa.infra.embedding.ruri import RuriEmbedder
from safety_qa.infra.ingest import IngestResult, run_ingest
from safety_qa.infra.llm.ollama_chat import OllamaChatModel
from safety_qa.infra.search.bm25_index import BM25Index, build_bm25_index
from safety_qa.infra.search.excerpt_builder import BM25ExcerptBuilder
from safety_qa.infra.search.hybrid import HybridRetriever
from safety_qa.infra.settings import Settings, load_settings


@dataclass
class SearchStack:
    """検索スタック（ETL・埋め込み・BM25・DuckDB）のみを組み立てた結果。

    `evals/` の検索精度評価が Ollama・監査ログ用 DB に触れずに
    `retriever`/`accident_repository` へ到達できるよう、`build_container` から
    切り出す。`Container` は `Retriever` を公開しておらず、また
    `queries.duckdb`（監査ログ）の作成という検索評価には不要な副作用を伴うため。
    """

    accident_repository: AccidentRecordRepository
    # `Retriever` ではなく `HybridRetriever` を公開する。`evals` が
    # `search_with_diagnostics`（経路別スコアの診断）へ型付きで到達できるようにする。
    retriever: HybridRetriever
    # excerpt のヒット欄推定（discriminative_terms）が語彙統計を再利用できるよう
    # 公開する。retriever 内部にも同じインスタンスを持つが private 属性のため。
    bm25_index: BM25Index
    ingest_result: IngestResult
    accidents_connection: duckdb.DuckDBPyConnection

    def close(self) -> None:
        self.accidents_connection.close()


@dataclass
class Container:
    qa_service: QaService
    record_service: RecordService
    ingest_result: IngestResult
    accidents_connection: duckdb.DuckDBPyConnection
    queries_connection: duckdb.DuckDBPyConnection

    def close(self) -> None:
        try:
            self.accidents_connection.close()
        finally:
            self.queries_connection.close()


def build_search_stack(
    settings: Settings | None = None, *, embedder: Embedder | None = None
) -> SearchStack:
    """起動時 ETL を実行し、検索スタック（DuckDB・BM25・ハイブリッド検索）を組み立てる。

    Ollama・監査ログ用 DB には触れない（`build_container` 側の責務）。
    `settings.skip_etl` が true の場合は ETL を実行せず、既存の DB をそのまま使う。
    """
    settings = settings or load_settings()
    settings.accidents_db_path.parent.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        if settings.preload_embedder:
            # 起動時に埋め込みモデルの誤設定（次元不一致等）を検知したい運用向け。
            embedder = RuriEmbedder(settings.embedding_model_name, settings.embedding_dim)
        else:
            # 差分埋め込みが 0 件の起動（CSV 変更なしの再起動、ETL 単体実行での
            # 冪等性確認）では torch/モデルの実体に一切触れない。
            embedder = LazyEmbedder(
                lambda: RuriEmbedder(settings.embedding_model_name, settings.embedding_dim)
            )

    accidents_con = duckdb.connect(str(settings.accidents_db_path))
    if settings.skip_etl:
        # 既存の DB をそのまま使う（開発時の再起動を速くする用途）。CSV との
        # 差分は反映されないため、CSV を更新した場合は ETL を実行し直すこと。
        ensure_accidents_schema(accidents_con, settings.embedding_dim)
        row_count = accidents_con.execute("SELECT count(*) FROM accidents").fetchone()[0]
        ingest_result = IngestResult(
            row_count=row_count, embedded_count=0, reused_count=0, deleted_count=0
        )
    else:
        ingest_result = run_ingest(
            con=accidents_con,
            csv_path=settings.csv_path,
            embedder=embedder,
            embedding_model_name=settings.embedding_model_name,
            embedding_dim=settings.embedding_dim,
        )
    accident_repository = DuckDbAccidentRepository(accidents_con)

    bm25_index = build_bm25_index(accident_repository.list_all())
    retriever = HybridRetriever(
        accidents=accident_repository, embedder=embedder, bm25_index=bm25_index
    )

    return SearchStack(
        accident_repository=accident_repository,
        retriever=retriever,
        bm25_index=bm25_index,
        ingest_result=ingest_result,
        accidents_connection=accidents_con,
    )


def build_chat_model(settings: Settings, *, today: Callable[[], date] | None = None) -> ChatModel:
    """Ollama 接続のみを組み立てる。

    `today` は相対日付の解釈基準（システムプロンプトへ注入する「本日」）。
    未指定時は本番と同じ基準（`business_today`）を使う。`evals/run.py` は
    評価基準日（コーパス最終日に固定）を注入するためにこの引数を使う
    （`temperature=0`/`seed=0` と同じ「制御変数の固定」。RagQaService に渡す
    `today` と実行のたびにずれないよう、呼び出し側で同じ値を両方へ渡すこと）。
    """
    return OllamaChatModel(
        host=settings.ollama_host,
        model=settings.ollama_model,
        num_thread=settings.ollama_num_thread,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
        today=today or (lambda: business_today(settings.business_timezone)),
    )


def build_container(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    chat_model: ChatModel | None = None,
) -> Container:
    """`build_search_stack`/`build_chat_model` を組み合わせ、RAG パイプライン一式を
    組み立てる（ブロッキング）。

    `embedder`/`chat_model` はテストで実際の重量級コンポーネント（torch モデル・
    Ollama 接続）をフェイクに差し替えるための注入ポイント。通常運用では省略し、
    設定値から実装を構築する。
    """
    settings = settings or load_settings()
    settings.queries_db_path.parent.mkdir(parents=True, exist_ok=True)

    search_stack = build_search_stack(settings, embedder=embedder)

    queries_con = duckdb.connect(str(settings.queries_db_path))
    ensure_queries_schema(queries_con)
    query_repository = DuckDbQueryRepository(queries_con)

    if chat_model is None:
        chat_model = build_chat_model(settings)

    qa_service = RagQaService(
        accident_repository=search_stack.accident_repository,
        retriever=search_stack.retriever,
        query_repository=query_repository,
        chat_model=chat_model,
        excerpt_builder=BM25ExcerptBuilder(search_stack.bm25_index),
        top_n=settings.top_n,
        max_concurrent_answers=settings.max_concurrent_answers,
        today=lambda: business_today(settings.business_timezone),
    )
    record_service = AccidentRecordLookupService(
        accident_repository=search_stack.accident_repository,
        retriever=search_stack.retriever,
        query_repository=query_repository,
        max_page_size=settings.max_page_size,
    )

    return Container(
        qa_service=qa_service,
        record_service=record_service,
        ingest_result=search_stack.ingest_result,
        accidents_connection=search_stack.accidents_connection,
        queries_connection=queries_con,
    )
