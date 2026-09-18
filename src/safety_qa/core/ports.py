"""バックエンドのポート（Protocol）。

実装（DuckDB / sentence-transformers / bm25s / fugashi / ollama）は
`safety_qa.infra` に閉じる。ここで定義するのは「意味のある操作の
単位」であり、生 SQL やモデル呼び出しの詳細を上位・他レイヤーに漏らさない。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol
from uuid import UUID

from safety_qa.core.filters import RecordFilter
from safety_qa.core.models import (
    AccidentRecord,
    AggregateResult,
    Query,
    QueryRecord,
    ScoredHit,
    ToolCall,
)


class AccidentRecordRepository(Protocol):
    """事故記録データへのアクセス。"""

    def get_many(self, record_ids: list[str]) -> list[AccidentRecord]:
        """id 一覧に対応する事故記録を返す（順不同）。存在しない id は無視する。"""
        ...

    def list_all(self) -> list[AccidentRecord]:
        """全件を返す。BM25 インデックス構築等、起動時の一括処理に使う。"""
        ...

    def vector_search(
        self, query_vector: list[float], top_n: int, *, record_filter: RecordFilter | None = None
    ) -> list[ScoredHit]:
        """埋め込みベクトルによるコサイン類似度検索。上位 top_n 件を
        類似度（`ScoredHit.score`）降順で返す。`record_filter` を指定すると条件に
        合致するレコードのみを対象にする。
        """
        ...

    def find_ids(self, record_filter: RecordFilter) -> list[str]:
        """条件に合致する record_id を id 昇順で返す。集計・ハイブリッド検索の
        絞り込み（BM25 側の対象限定）で共用する。
        """
        ...

    def list_records(
        self, record_filter: RecordFilter, *, limit: int, offset: int
    ) -> list[AccidentRecord]:
        """条件に合致する記録を発生日昇順（`date`, `id` の全順序）でページングして返す。

        `GET /v1/queries/{id}/records`（aggregate 由来）が使う。`find_ids` と違い
        id だけでなく記録本体を返し、`LIMIT`/`OFFSET` で絞ることで
        `find_ids`/`aggregate` のような全件転送を避ける。並び順は offset
        ページング中に重複・欠落が起きないよう安定した全順序でなければならない。
        """
        ...

    def aggregate(
        self, record_filter: RecordFilter, *, group_by: str | None = None
    ) -> AggregateResult:
        """条件で絞り込んで件数と該当 id 一覧を返す（定義済みの集計のみ）。

        `group_by` を指定すると、結果の `groups` に軸ごとの内訳が入る。
        """
        ...

    def count(self, record_filter: RecordFilter) -> int:
        """条件に合致する件数のみを返す（回答の根拠として示す母数）。

        `find_ids` は id 一覧全件を転送するため、母数だけが欲しい場合はこちらを
        使う（`infra` 側は `SELECT count(*)` で済む）。
        """
        ...

    def date_range(self) -> tuple[date, date] | None:
        """収録されている記録の日付範囲（最古日, 最新日）を返す。

        記録が 1 件も無い場合は `None`。呼び出し側でキャッシュせず毎回問い合わせる
        （増分 ETL 後も常に最新の範囲を返すため）。
        """
        ...

    def ingested_at(self) -> datetime | None:
        """直近の起動時 ETL が完了した時刻（UTC）を返す。ETL が一度も
        成功していない場合は `None`。

        `GET /v1/queries/{id}/records` の `as_of` に使う。`citations`
        （回答生成時点の凍結）と、このエンドポイントが返す現在値が ETL の
        再実行により乖離しうることを利用者が検知できるようにする。
        """
        ...


class Retriever(Protocol):
    """記述検索（BM25 + ベクトルのハイブリッド、RRF 統合）。"""

    def hybrid_search(
        self, query: str, top_n: int, *, record_filter: RecordFilter | None = None
    ) -> list[str]:
        """自然言語クエリに対するハイブリッド検索結果を、上位 top_n 件の
        record_id として順位順に返す。`record_filter` を指定すると、BM25・
        ベクトル検索の両方をその条件に合致するレコードのみに絞り込む。
        """
        ...


class QueryRepository(Protocol):
    """問い合わせ（質問 + 回答）の永続化。監査ログを兼ねる。

    成功・失敗を問わずすべての問い合わせを記録する。失敗した問い合わせも
    「どの質問がどう失敗したか」を監査できることが耐障害性の観点で必要なため。
    """

    def save(self, record: QueryRecord) -> None:
        """問い合わせを保存する。どのツールをどの引数で呼んだかも監査用に記録する。"""
        ...

    def get(self, query_id: UUID) -> Query | None:
        """id で成功した問い合わせを取得する。存在しない、または失敗した問い合わせは None。"""
        ...

    def get_record(self, query_id: UUID) -> QueryRecord | None:
        """id で問い合わせを取得する（成功・失敗を問わない）。存在しない場合は None。

        `get` と異なり `tool_name`/`tool_arguments`/`error_code` を含む監査ログの
        生データを返す。`GET /v1/queries/{id}/records` が、実際に実行された
        ツールの引数から対象母集団を再構成するために使う。
        """
        ...

    def list_records(self, *, limit: int, before_id: UUID | None = None) -> list[QueryRecord]:
        """問い合わせを新しい順（id 降順）に最大 limit 件返す。

        id は UUIDv7（`core/ids.py`）で生成時刻順に単調増加するため、
        `created_at`（ローカル時刻、TZ 非依存性なし）ではなくこちらをカーソルに
        する。`before_id` を指定すると、その id より古い問い合わせから返す
        （ページ間で監査ログへの書き込みが続いても取りこぼさないため）。
        `get` と異なり成功・失敗の両方を返す。
        """
        ...


class Embedder(Protocol):
    """自然言語テキストの埋め込みベクトル化。"""

    def embed_query(self, text: str) -> list[float]:
        """検索クエリを埋め込む（文書埋め込みとは異なる prefix を使うモデルがある）。"""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """事故記録本文をバッチで埋め込む（起動時 ETL 用）。"""
        ...


class ExcerptBuilder(Protocol):
    """回答の根拠として示す抜粋（ヒット箇所ベース）を組み立てる。"""

    def build(self, record: AccidentRecord, query: str | None) -> tuple[str | None, str]:
        """(ヒットした欄名 | None, 抜粋) を返す。

        `query` が None（aggregate 由来など自由文が無い経路）の場合や、
        ヒットした語がどの欄にも見つからない場合は、欄名 None と
        description の先頭を切り詰めた文字列を返す。
        """
        ...


class ChatModel(Protocol):
    """LLM 推論（Ollama 等）。"""

    def route(self, question: str) -> ToolCall | None:
        """システムプロンプト + 質問を渡し、ツール呼び出しの要否・内容を判定する。
        ツール未呼び出しの場合は None。
        """
        ...

    def answer(self, prompt: str) -> str:
        """ツールを渡さず、文脈に基づいて回答文を生成する（2 パス目）。"""
        ...
