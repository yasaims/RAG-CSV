"""バックエンドレイヤーが送出する例外。

HTTP status を一切知らない。スキーマレイヤー（api/errors.py）がこれらを
RFC 9457 Problem Details にマッピングする。
"""

from __future__ import annotations

from uuid import UUID


class QaServiceError(Exception):
    """QA サービスに関するエラーの基底クラス。"""


class QueryNotFoundError(QaServiceError):
    """指定された id の問い合わせが存在しない。"""

    def __init__(self, query_id: UUID) -> None:
        self.query_id = query_id
        super().__init__(f"query not found: {query_id}")


class RecordNotFoundError(QaServiceError):
    """指定された id の事故記録が存在しない。

    citation の id は回答生成時点の凍結スナップショットであり、記録が後から
    更新・削除された場合もこの例外になる（現在存在しないことと、そもそも
    実在しなかったことは区別できない）。
    """

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__(f"record not found: {record_id}")


class AnswerUnavailableError(QaServiceError):
    """LLM 推論・検索の失敗やタイムアウトなど、一時的な要因により回答を生成できない。

    再試行で回復しうる（503 + Retry-After）。
    """


class InvalidToolCallError(QaServiceError):
    """LLM が選択したツール呼び出しが決定的に不正。

    temperature=0 で固定しているため再試行しても同じ結果になる。一時的な失敗
    ではないため Retry-After は付けない（`AnswerUnavailableError` とは区別する）。
    """


class InvalidToolArgumentsError(InvalidToolCallError):
    """LLM が生成したツール引数が不正（未知の値・未知の引数）。"""
