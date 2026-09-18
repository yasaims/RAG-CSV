"""業務日付の基準。

`date.today()` はプロセスのローカルタイムゾーン依存で、UTC のコンテナでは
JST 09:00 未満の時間帯に「本日」が 1 日ずれ、「今年」の判定も年末年始でずれる。
基準は設定（`SAFETY_QA_BUSINESS_TIMEZONE`）で明示する。
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo


def business_today(timezone: str) -> date:
    """指定タイムゾーンでの「今日」を返す。"""
    return datetime.now(ZoneInfo(timezone)).date()
