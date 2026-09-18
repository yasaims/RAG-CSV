"""ヒット箇所ベースの抜粋生成。

`description` を機械的に先頭から切るのではなく、検索でヒットした語（terms）が
実際に含まれる欄（description/cause/countermeasure）を優先し、その語の周辺を
抜粋する。terms がどの欄にも当たらない場合は description の先頭にフォールバック
する（`excerpt_field=None` で「たまたま description を切っただけ」と区別する）。
"""

from __future__ import annotations

from collections.abc import Sequence

_FIELD_PRIORITY = ("description", "cause", "countermeasure")


def build_excerpt(
    *,
    description: str,
    cause: str,
    countermeasure: str,
    terms: Sequence[str],
    max_len: int = 80,
) -> tuple[str | None, str]:
    """(ヒットした欄名 | None, 抜粋) を返す。

    terms がどこにも当たらなければ (None, description の先頭 max_len 文字) を返す。
    """
    fields = {"description": description, "cause": cause, "countermeasure": countermeasure}
    for field_name in _FIELD_PRIORITY:
        text = fields[field_name]
        hit = _find_earliest_hit(text, terms)
        if hit is not None:
            start, length = hit
            return field_name, _window(text, start, length, max_len)
    return None, _window(description, 0, 0, max_len)


def _find_earliest_hit(text: str, terms: Sequence[str]) -> tuple[int, int] | None:
    earliest: tuple[int, int] | None = None
    for term in terms:
        if not term:
            continue
        index = text.find(term)
        if index != -1 and (earliest is None or index < earliest[0]):
            earliest = (index, len(term))
    return earliest


def _window(text: str, hit_start: int, hit_len: int, max_len: int) -> str:
    """ヒット位置ができるだけ窓の中央に来るよう、前後 max_len 文字に切り出す。"""
    if len(text) <= max_len:
        return text
    pad = max(0, (max_len - hit_len) // 2)
    start = max(0, hit_start - pad)
    end = min(len(text), start + max_len)
    start = max(0, end - max_len)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt = excerpt + "…"
    return excerpt
