"""バックエンドレイヤーのドメインモデル。

HTTP / Web フレームワークに依存しないプレーンな dataclass として定義する。
スキーマレイヤー（api/schemas）はこれらを Pydantic モデルへ変換する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Citation:
    """回答の根拠として参照した事故記録。

    安全管理担当者が API レスポンスだけで根拠を確認できるよう、id に加えて
    最小限のメタデータ（日付・場所・設備・重大度・抜粋）を持つ。
    """

    record_id: str
    date: str
    location: str
    equipment: str
    severity: str
    excerpt: str
    # ヒットした欄名（description/cause/countermeasure）。ヒット箇所を特定できな
    # かった場合（terms がどこにも当たらない等）は None。監査ログの旧行も
    # `Citation(**d)` で復元時に None のまま扱われる（推測値を混ぜない）。
    excerpt_field: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerBasis:
    """回答の根拠情報。

    解釈済みの絞り込み条件・参照した件数と母数・記録の収録期間を、LLM を通さず
    `RagQaService` がツール実行結果から組み立てる（`citations` と同じ位置付け）。
    事故記録データが後から変わると再計算できないため、生成時点の値を凍結する。
    ツール未呼び出し時は `None`。
    """

    tool: str
    filters: dict[str, str]
    group_by: str | None
    matched_count: int
    truncated_count: int
    first_record_date: str
    last_record_date: str


@dataclass(frozen=True, slots=True)
class Query:
    """1 回の問い合わせ（質問 + 回答）。

    `degraded_reason` は回答が2パス目のLLM生成を経ず固定文で縮退したかどうかを
    機械可読に示す（"no_tool_call" | "unmatched_filter" | "out_of_coverage" |
    None）。利用者が「真の0件」と「判断できない」を回答文の文字列に頼らず区別
    できるようにする（ADR 009 と同じ理由）。`basis` は `no_tool_call` の場合
    `None` のままだが、`degraded_reason` はその場合も設定される。
    永続化はしない（監査ログからは追跡できない。既知の制約）。
    """

    id: UUID
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    basis: AnswerBasis | None = None
    degraded_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """監査ログ 1 行分のデータ（`QueryRepository.save` の入力）。

    問い合わせが失敗した場合も記録できるよう、成功時の `Query` とは別に定義する。
    失敗時は `answer=None`・`error_code`/`error_detail` が入る。
    """

    id: UUID
    question: str
    created_at: datetime
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    basis: AnswerBasis | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, str] = field(default_factory=dict)
    error_code: str | None = None
    error_detail: str | None = None


@dataclass(frozen=True, slots=True)
class AccidentRecord:
    """事故・ヒヤリハット記録 1 件（SPEC.md のデータ定義に対応）。"""

    id: str
    date: str
    time: str
    shift: str
    location: str
    equipment: str
    severity: str
    worker_id: str
    worker_role: str
    worker_experience_years: int
    description: str
    cause: str
    countermeasure: str


@dataclass(frozen=True, slots=True)
class QueryRecordPage:
    """`GET /v1/queries/{id}/records` の戻り値。

    問い合わせが実際に対象とした母集団を、保存済みの `tool_arguments` から
    再構成してページングした結果。`citations`（回答生成時点の凍結）とは別に、
    `as_of`（ETL 取り込み時刻）時点の現在値を返す。両者は ETL の再実行により
    乖離しうる。

    `ranking` は `items` の並び順の意味（`"relevance"` | `"date"`）。search 由来は
    "relevance"（ハイブリッド検索の順位）、aggregate 由来は "date"（発生日昇順）。
    `query_text` は search 由来の自由文クエリ（aggregate 由来は None）で、
    絞り込みには使われず並び順にのみ影響する（`total` は常に
    `basis.matched_count` と一致する）。
    """

    items: list[AccidentRecord]
    total: int
    limit: int
    offset: int
    ranking: str
    query_text: str | None
    basis: AnswerBasis | None
    as_of: datetime | None


@dataclass(frozen=True, slots=True)
class AggregateGroup:
    """`group_by` 指定時の内訳 1 行分（例: 設備別の件数）。"""

    key: str
    count: int
    record_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """集計結果。母集団の全 id を保持し監査性を担保する。

    `group_by` を指定した場合、`groups` に内訳が入る（未指定時は空リスト）。
    `count` / `record_ids` は group_by の有無によらず母集団全体を表す。
    """

    count: int
    record_ids: list[str] = field(default_factory=list)
    groups: list[AggregateGroup] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """LLM が選択したツールと引数（1 パス目のルーティング結果）。"""

    name: str
    arguments: dict[str, str]


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """検索経路が返す 1 件（record_id と、その経路でのスコア）。

    スコアの意味と尺度は経路ごとに異なる（BM25 は非負の重み、ベクトル検索は
    コサイン類似度）。経路をまたいで大小を比較してはならない。順位の統合は
    スコアではなく順位だけを使う `core/fusion.py` の責務で、このスコアは
    どちらの経路が効いたかを診断するために持ち回る値。
    """

    record_id: str
    score: float
