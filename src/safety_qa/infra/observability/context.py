"""`trace_id` の発行・受信ヘッダからの継承・contextvars への出し入れ。

監査性の要である `trace_id`（Problem Details / 応答ヘッダ / 構造化ログ）を
このモジュールの 3 関数だけに閉じる。将来 AWS X-Ray / OpenTelemetry(ADOT) に
差し替える際の変更点をここ 1 箇所に限定する。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from uuid import uuid4

import structlog

# W3C traceparent の trace-id 部分（32桁16進）。全ゼロは仕様上「未設定」を表す無効値。
_TRACEPARENT_RE = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")
_INVALID_TRACE_ID = "0" * 32

# X-Amzn-Trace-Id（ALB / X-Ray）: `Root=1-<8桁hex>-<24桁hex>;Parent=...;Sampled=...`
_AMZN_ROOT_RE = re.compile(r"Root=(1-[0-9a-f]{8}-[0-9a-f]{24})")

# X-Request-Id 等の任意トークン。ログ注入（改行・制御文字）を避けるため、
# 妥当な長さの英数字・ハイフン・アンダースコアのみ許可する。
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def new_trace_id() -> str:
    """新規 trace_id を発行する。W3C traceparent の trace-id 部分と同じ 32 桁16進。"""
    return uuid4().hex


def extract_trace_id(headers: Mapping[str, str], *, trust_inbound: bool) -> str:
    """受信ヘッダから trace_id を継承する。信頼しない/該当なし/不正な場合は新規発行する。

    `headers` はキーを小文字化した辞書を想定する（ASGI の raw ヘッダに合わせる）。
    公開 API で受信ヘッダを無条件に信じるとログ ID の偽装・注入経路になるため、
    継承は `trust_inbound=True`（信頼できるプロキシ配下）のときだけ行う
    （既定 off。`SAFETY_QA_TRUST_INBOUND_TRACE_HEADER`）。
    """
    if not trust_inbound:
        return new_trace_id()

    traceparent = headers.get("traceparent")
    if traceparent:
        m = _TRACEPARENT_RE.match(traceparent)
        if m and m.group(1) != _INVALID_TRACE_ID:
            return m.group(1)

    amzn = headers.get("x-amzn-trace-id")
    if amzn:
        m = _AMZN_ROOT_RE.search(amzn)
        if m:
            return m.group(1)

    request_id = headers.get("x-request-id")
    if request_id and _OPAQUE_TOKEN_RE.match(request_id):
        return request_id

    return new_trace_id()


def bind_trace_id(trace_id: str) -> None:
    """以降このコンテキストで出力する全ログ行に `trace_id` を自動付与する。"""
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def get_trace_id() -> str | None:
    """現在のコンテキストにバインドされている `trace_id`（未設定なら None）。"""
    return structlog.contextvars.get_contextvars().get("trace_id")


def clear_trace_id() -> None:
    """コンテキストをクリアする（リクエスト完了時にミドルウェアが呼ぶ）。"""
    structlog.contextvars.clear_contextvars()
