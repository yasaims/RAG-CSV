"""Qwen3-4B の tool calling に渡すツール定義と引数検証。

`shift`（日勤/夜勤）引数は当初のプロトタイプ検証では未公開だった。ETL では
既に導出・保持していた列で、これを公開したところ「傾向」を問う分析系の質問で
ツール選択が安定した（docs/stack-feasibility-report.md D-3 対策1）。
ツール引数スキーマは実データが持つ全軸を漏れなく反映することを原則とする。

**例外**: `severity`/`severity_min` は `search_accident_records` には公開しない
（`SEARCH_EXCLUDED_FILTER_KEYS`）。理由は ADR 011 を参照。要約すると、enum 軸は
`core/rag_service.py` の空振り軸検知（部分一致軸のみが対象）の死角であり、
検索側で誤った重大度の絞り込みを実装が検知できない。かつ実データのエスカレー
ション連鎖（同一箇所でヒヤリハット→軽微→実接触、のように重大度が変化しながら
進行する事例群）を重大度で AND すると静かに分断してしまう。集計
（`aggregate_accident_records`）は件数を断定するため条件の厳密さが要り、対象外
にはしない。

絞り込み条件のキー一覧・enum 検証は `core.filters.RecordFilter` に
集約されている。ここでは JSON Schema の生成と、LLM が返す文字列引数から
`RecordFilter` への変換（日付のパース含む）のみを扱う。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from safety_qa.core.aggregation import VALID_GROUP_BY, validate_group_by
from safety_qa.core.exceptions import InvalidToolArgumentsError
from safety_qa.core.filters import (
    EQUIPMENT_EXAMPLES,
    FILTER_KEYS,
    LOCATION_EXAMPLES,
    VALID_SEVERITIES,
    VALID_SHIFTS,
    WORKER_ROLE_EXAMPLES,
    RecordFilter,
)

SEARCH_TOOL_NAME = "search_accident_records"
AGGREGATE_TOOL_NAME = "aggregate_accident_records"

_FILTER_PROPERTIES: dict[str, dict[str, object]] = {
    "severity": {
        "type": "string",
        "enum": list(VALID_SEVERITIES),
        "description": "重大度で絞り込む。指定しない場合は全件対象。",
    },
    "date_from": {
        "type": "string",
        "description": "この日付（YYYY-MM-DD）以降。指定しない場合は下限なし。",
    },
    "date_to": {
        "type": "string",
        "description": "この日付（YYYY-MM-DD）以前。指定しない場合は上限なし。",
    },
    "equipment": {
        "type": "string",
        "description": (
            "設備名で絞り込む（部分一致）。実データの設備名の例: "
            + "、".join(EQUIPMENT_EXAMPLES)
            + "。「〜エリア」「〜棟」等の場所を表す語は設備名ではないため、"
            "ここではなく location に渡すこと。"
        ),
    },
    "location": {
        "type": "string",
        "description": (
            "場所で絞り込む（部分一致）。実データの場所名は「棟＋エリア」の複合表記。"
            "実データの例: "
            + "、".join(LOCATION_EXAMPLES)
            + "。質問文中の場所表現（例:「第2製造棟 機械加工エリア」）は"
            "棟とエリアに分割せず、そのまま1つの値としてここに渡すこと。"
            "プレス機・フォークリフト・高所作業車 等の設備名は場所ではないため、"
            "ここではなく equipment に渡すこと。質問文に具体的な場所名が書かれて"
            "いない場合（「同じ場所で」「どこで」等、場所を問うだけの表現を含む）は"
            "指定しないこと。"
        ),
    },
    "shift": {
        "type": "string",
        "enum": list(VALID_SHIFTS),
        "description": "日勤/夜勤で絞り込む。指定しない場合は両方対象。",
    },
    "severity_min": {
        "type": "string",
        "enum": list(VALID_SEVERITIES),
        "description": (
            "指定した重大度以上で絞り込む（例: 中位を指定すると中位・重大が対象）。"
            "severity と同時に指定することはできない。指定しない場合は絞り込みなし。"
        ),
    },
    "worker_role": {
        "type": "string",
        "description": (
            "作業員の職種で絞り込む（部分一致）。実データの職種の例: "
            + "、".join(WORKER_ROLE_EXAMPLES)
        ),
    },
    "experience_min": {
        "type": "string",
        "description": "作業員の経験年数（年）の下限。指定しない場合は下限なし。",
    },
    "experience_max": {
        "type": "string",
        "description": "作業員の経験年数（年）の上限。指定しない場合は上限なし。",
    },
    "hour_from": {
        "type": "string",
        "description": (
            "発生時刻の時間帯（0〜23）の開始。hour_to より大きい値を指定すると、"
            "日をまたぐ時間帯（例: hour_from=22, hour_to=5 で22時〜翌5時）として扱う。"
        ),
    },
    "hour_to": {
        "type": "string",
        "description": "発生時刻の時間帯（0〜23）の終了。",
    },
}
# _FILTER_PROPERTIES のキーは RecordFilter のフィールドと 1:1 対応させる
assert tuple(_FILTER_PROPERTIES) == FILTER_KEYS

# search_accident_records には公開しない絞り込み軸（モジュール docstring の
# 「例外」参照）。aggregate_accident_records は _FILTER_PROPERTIES をそのまま使う
# ため対象外。
SEARCH_EXCLUDED_FILTER_KEYS: tuple[str, ...] = ("severity", "severity_min")

_SEARCH_FILTER_PROPERTIES: dict[str, dict[str, object]] = {
    key: value
    for key, value in _FILTER_PROPERTIES.items()
    if key not in SEARCH_EXCLUDED_FILTER_KEYS
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "事故・ヒヤリハット記録の自由記述（description/cause/countermeasure）を"
                "自然言語で検索する。設備名・現場名・状況の記述、傾向や特徴を問う質問など、"
                "記述内容に関する質問に使う。件数の集計には使わない。"
                "期間・設備・場所・勤務帯で絞り込みたい場合は対応する引数も指定する。"
                "重大度では絞り込めない（この引数はここには無い）。重大度を表す語が"
                "質問文にあっても query に含めるだけにすること。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ。質問文からキーワード・状況を抽出したもの。",
                    },
                    **_SEARCH_FILTER_PROPERTIES,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": AGGREGATE_TOOL_NAME,
            "description": (
                "事故・ヒヤリハット記録を条件で絞り込んで件数を集計する。"
                "「何件」「件数」等、数を尋ねる質問に使う。"
                "件数の集計のみ対応し、平均・合計・割合等の算出はできない。"
                "「設備別」「月別」等、軸ごとの内訳が知りたい場合は group_by も指定する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_FILTER_PROPERTIES,
                    "group_by": {
                        "type": "string",
                        "enum": list(VALID_GROUP_BY),
                        "description": (
                            "指定すると、この軸ごとの内訳を件数とともに返す"
                            "（例: equipment で設備別の件数、month で月別の推移）。"
                            "指定しない場合は合計件数のみ。"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]


def _normalize(value: str | None) -> str | None:
    """空文字・前後の空白のみの値を「未指定」として扱う。

    LLM が JSON の null を文字列化して渡す経路（infra/llm/ollama_chat.py）とは
    別に、ここでも空文字を防御的に「未指定」へ正規化する。
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_date(value: str | None, *, key: str) -> date | None:
    normalized = _normalize(value)
    if normalized is None:
        return None
    try:
        return date.fromisoformat(normalized)
    except ValueError as e:
        raise InvalidToolArgumentsError(
            f"{key} の形式が不正です（YYYY-MM-DD 形式で指定してください）: {value!r}"
        ) from e


def _parse_int(value: str | None, *, key: str) -> int | None:
    normalized = _normalize(value)
    if normalized is None:
        return None
    try:
        return int(normalized)
    except ValueError as e:
        raise InvalidToolArgumentsError(f"{key} は整数で指定してください: {value!r}") from e


def parse_aggregate_arguments(arguments: Mapping[str, str]) -> RecordFilter:
    """`aggregate_accident_records` / `search_accident_records` の絞り込み引数を検証する。

    実行される SQL は infra 側の固定クエリであり、ここでの検証は
    「LLM の誤生成をそのまま DB 層へ通さない」ための最終防衛線。
    未知の引数・不正な日付/整数形式は `InvalidToolArgumentsError` で拒否する。
    enum・範囲（severity/shift/severity_min/hour_*/experience_*）の検証は
    `RecordFilter.__post_init__` が行う。
    """
    unknown_keys = set(arguments) - set(FILTER_KEYS)
    if unknown_keys:
        raise InvalidToolArgumentsError(f"未知の引数: {sorted(unknown_keys)}")

    return RecordFilter(
        severity=_normalize(arguments.get("severity")),
        date_from=_parse_date(arguments.get("date_from"), key="date_from"),
        date_to=_parse_date(arguments.get("date_to"), key="date_to"),
        equipment=_normalize(arguments.get("equipment")),
        location=_normalize(arguments.get("location")),
        shift=_normalize(arguments.get("shift")),
        severity_min=_normalize(arguments.get("severity_min")),
        worker_role=_normalize(arguments.get("worker_role")),
        experience_min=_parse_int(arguments.get("experience_min"), key="experience_min"),
        experience_max=_parse_int(arguments.get("experience_max"), key="experience_max"),
        hour_from=_parse_int(arguments.get("hour_from"), key="hour_from"),
        hour_to=_parse_int(arguments.get("hour_to"), key="hour_to"),
    )


def parse_aggregate_tool_arguments(arguments: Mapping[str, str]) -> tuple[RecordFilter, str | None]:
    """`aggregate_accident_records` の引数を検証する。`group_by` を分離して返す。

    `group_by` は絞り込み軸（`RecordFilter`）とは別概念のため、
    `parse_aggregate_arguments` には渡さず個別に検証する。
    """
    group_by = validate_group_by(_normalize(arguments.get("group_by")))
    filter_arguments = {key: value for key, value in arguments.items() if key != "group_by"}
    return parse_aggregate_arguments(filter_arguments), group_by


def parse_search_arguments(
    arguments: Mapping[str, str], *, fallback_query: str
) -> tuple[str, RecordFilter, tuple[tuple[str, str], ...]]:
    """`search_accident_records` の引数を検証し、検索クエリと絞り込み条件に分ける。

    `query` が未指定・空文字の場合は `fallback_query`（質問文そのもの）を使う。
    `SEARCH_EXCLUDED_FILTER_KEYS`（severity/severity_min）は `RecordFilter` には
    載せず、3番目の戻り値（キーと値の組）として分離する。ツールスキーマには
    公開していないが、LLM が それでも渡した場合に静かに無視するのではなく、
    呼び出し側（根拠フッター・監査ログ）が開示できるようにするため。
    残りの引数は `parse_aggregate_arguments` と同じ検証を通す。
    """
    query = _normalize(arguments.get("query")) or fallback_query
    ignored_axes = tuple(
        (key, normalized)
        for key in SEARCH_EXCLUDED_FILTER_KEYS
        if (normalized := _normalize(arguments.get(key))) is not None
    )
    filter_arguments = {
        key: value
        for key, value in arguments.items()
        if key != "query" and key not in SEARCH_EXCLUDED_FILTER_KEYS
    }
    return query, parse_aggregate_arguments(filter_arguments), ignored_axes
