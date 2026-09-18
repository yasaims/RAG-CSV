"""評価指標の純粋関数群。

LLM を一切使わない決定的な検証をここに集約する。id 捏造・数値捏造の検出は
`tests/integration/test_end_to_end.py` のプロトタイプ由来チェックをここへ
引き上げたもので、単体テスト（`tests/test_eval_metrics.py`）と評価ハーネスの
両方から共有する。
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from safety_qa.core.rag_service import FREE_TEXT_ARG_KEYS

# 回答文中の id 言及を広く拾う。正規の書式（ACC-YYYY-NNN）に完全一致するかどうかを
# 問わない。実測された「id と日付を混同した崩れた表記」（例: ACC-2024-001 と
# 2024-01-18 を混同した "ACC-2024-01-18"）は厳密な書式一致では検出できない。
# 抽出は広く行い、「citations に含まれるか」だけで絶対要件を判定する。
_ID_MENTION_PATTERN = re.compile(r"ACC-[\d-]+")

# 正規の記録 id 書式（参考指標: 表記ゆれの検出専用。絶対要件の判定には使わない）。
_STRICT_ID_PATTERN = re.compile(r"(?<![-\d])ACC-\d{4}-\d{3}(?![-\d])")

# 日本語の件数表記（例: "5件", "5 件"）。年・時刻・休業日数等での誤検出を避けるため
# 「件」を伴う数値だけを対象にする。
_COUNT_PATTERN = re.compile(r"(\d+)\s*件")

# 「該当なし」を主張する定型的な言い回し。`core/rag_service.py` の 0 件時の
# context 文字列（例:「条件に合致する記録はありません（0件）」「見つかりませんでした」）
# を LLM が言い換えて生成しうるため、数値表記だけでなく言い回しでも判定する。
# ヒューリスティックであり、将来の言い回し追加で拡張しうる既知の限界。
_NO_RESULT_PHRASES: tuple[str, ...] = (
    "見つかりませんでした",
    "見つかりません",
    "該当する記録はありません",
    "該当なし",
    "0件",
    "0 件",
)


def extract_id_mentions(answer_text: str) -> set[str]:
    """回答文中に出現する id 言及（広義。書式の厳密一致は問わない）を抽出する。"""
    return set(_ID_MENTION_PATTERN.findall(answer_text))


def extract_malformed_id_like_tokens(answer_text: str) -> set[str]:
    """id 言及のうち正規の書式（ACC-YYYY-NNN）に一致しない崩れた表記（参考指標）。"""
    return {
        tok for tok in extract_id_mentions(answer_text) if not _STRICT_ID_PATTERN.fullmatch(tok)
    }


def find_fabricated_ids(answer_text: str, cited_ids: Collection[str]) -> set[str]:
    """citations に含まれない id 言及を検出する（絶対要件1）。

    id の書式に完全一致するかどうかを問わない（`extract_id_mentions` 参照）。
    citations（`cited_ids`）はツール実行結果から決定的に採用される値であり、
    この関数が判定の基準に使うのは常にその集合だけである。
    """
    return extract_id_mentions(answer_text) - set(cited_ids)


def extract_mentioned_counts(answer_text: str) -> list[int]:
    """回答文中の「N件」表記をすべて抽出する。"""
    return [int(n) for n in _COUNT_PATTERN.findall(answer_text)]


def has_count_mismatch(answer_text: str, expected_counts: Collection[int]) -> bool:
    """回答文中の「N件」表記が、期待される件数の集合に含まれないものを

    1つでも含む場合に True（絶対要件2）。

    `expected_counts` には集計対象の総件数と、`group_by` 指定時の内訳件数を
    まとめて渡す（総件数 ∪ 各グループ件数）。内訳付き回答は複数の正しい件数を
    含みうるため、単一の期待値との完全一致にすると正常な回答で誤発火する。
    """
    mentioned = extract_mentioned_counts(answer_text)
    if not mentioned:
        return False
    allowed = set(expected_counts)
    return any(n not in allowed for n in mentioned)


def has_unsupported_fabrication(answer_text: str) -> bool:
    """ツール未呼び出し（対応不可）の回答文中に件数表記が出現していないかを検証する（絶対要件5）。"""
    return bool(extract_mentioned_counts(answer_text))


def claims_no_results(answer_text: str) -> bool:
    """回答が「該当する記録がない」旨を主張しているかを判定する。

    数値表記（0件）と定型的な言い回しの両方を見る。`_execute_search` の
    空フィルタ時 context「(該当する記録が見つかりませんでした)」等、
    数値を伴わない言い回しで生成されることがあるため。
    """
    if 0 in extract_mentioned_counts(answer_text):
        return True
    return any(phrase in answer_text for phrase in _NO_RESULT_PHRASES)


def has_false_zero_claim(answer_text: str, expected_count: int) -> bool:
    """実件数が1件以上あるのに「該当なし/0件」と主張している場合に True（絶対要件3）。

    `_unmatched_axes`（core/rag_service.py）は部分一致軸（equipment/location/
    worker_role）に限って空振りを検知しており、日付軸の解釈ずれによる偽ゼロは
    実装上塞がれていない生きた経路。この関数は文字列の一致ではなく
    「実件数と矛盾する主張をしているか」で判定する。
    """
    return expected_count > 0 and claims_no_results(answer_text)


def recall_at_k(hit_ids: Sequence[str], must_hit_ids: Sequence[str]) -> float:
    """`must_hit_ids` のうち `hit_ids`（top_n）に含まれる割合。"""
    if not must_hit_ids:
        return 1.0
    hit_set = set(hit_ids)
    matched = sum(1 for record_id in must_hit_ids if record_id in hit_set)
    return matched / len(must_hit_ids)


def tool_matches(actual_tool: str | None, expected_tool: str | None) -> bool:
    return actual_tool == expected_tool


def arguments_match(
    actual_arguments: Mapping[str, str],
    expected_arguments: Mapping[str, str],
    *,
    ignore_keys: Collection[str] = FREE_TEXT_ARG_KEYS,
) -> bool:
    """絞り込み軸の引数が期待値と一致するか（`ignore_keys` は比較対象から除く）。"""

    def _filtered(arguments: Mapping[str, str]) -> dict[str, str]:
        return {k: v for k, v in arguments.items() if k not in ignore_keys}

    return _filtered(actual_arguments) == _filtered(expected_arguments)
