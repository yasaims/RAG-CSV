"""ID 生成をここに隔離する。

`uuid6` は純 Python 実装で C 拡張のビルドが不要なため、Codespaces / Windows
どちらでも事故らない。Python 3.14 で標準ライブラリに `uuid.uuid7()` が入り次第、
この関数の中身だけ差し替える。
"""

from uuid import UUID

from uuid6 import uuid7


def new_query_id() -> UUID:
    """問い合わせリソースの id（UUIDv7）を生成する。"""
    return uuid7()
