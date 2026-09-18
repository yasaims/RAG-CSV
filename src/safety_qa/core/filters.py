"""集計・検索で共有する絞り込み条件（core の値オブジェクト）。

`aggregate_accident_records`・`search_accident_records` の両方が `RecordFilter`
の全軸（`FILTER_KEYS` 参照）を受け付ける。軸の一覧・enum 検証をここに集約する
ことで、ツール引数スキーマ（core/tools.py）・DuckDB の WHERE 構築（infra）で
軸の一覧や検証ルールが個別に重複するのを防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date

from safety_qa.core.exceptions import InvalidToolArgumentsError

VALID_SEVERITIES: tuple[str, ...] = ("ヒヤリハット", "軽微", "中位", "重大")
VALID_SHIFTS: tuple[str, ...] = ("日勤", "夜勤")

# ツール説明文に列挙する実データの設備名（静的な例示。起動時の SELECT DISTINCT は
# 行わない。設備名は部分一致で絞り込むため、この一覧はあくまで LLM への手掛かり）。
EQUIPMENT_EXAMPLES: tuple[str, ...] = (
    "プレス機",
    "フォークリフト",
    "旋盤",
    "溶接機",
    "コンベア",
    "グラインダー",
    "切断機",
    "高所作業車",
)

# ツール説明文に列挙する実データの職種名（EQUIPMENT_EXAMPLES と同じ位置付けの静的な例示。
# 部分一致で絞り込むため enum 化はしない。職種の増減でコード変更が必要になるトレードオフを避ける）。
WORKER_ROLE_EXAMPLES: tuple[str, ...] = (
    "製造スタッフ",
    "フォークリフトオペレーター",
    "機械オペレーター",
    "品質管理スタッフ",
    "設備保全スタッフ",
    "パート・アルバイト",
    "溶接工",
    "派遣スタッフ",
)

# ツール説明文に列挙する実データの場所名（EQUIPMENT_EXAMPLES と同じ位置付けの静的な例示）。
# 実データの location は「第2製造棟 機械加工エリア」のような棟+エリアの複合表記であり、
# 単一トークンの例しか示さないと LLM が「棟」と「エリア」を location/equipment に
# 誤分割することが確認された。複合表記のまま例示することでこれを防ぐ。
LOCATION_EXAMPLES: tuple[str, ...] = (
    "第1製造棟 プレスエリア",
    "第1製造棟 組立エリア",
    "第2製造棟 機械加工エリア",
    "第2製造棟 溶接エリア",
    "倉庫・搬送エリア",
    "屋外（荷捌き場）",
)

# 部分一致で絞り込む軸のキー -> 実データの値の例。0件時に「どの軸が実データに
# 存在しない値か」を利用者に示す固定文（core/prompts.py の unmatched_filter_answer）
# で使う。キーは _matches が部分一致で扱う軸（equipment/location/worker_role）と一致する。
AXIS_VALUE_EXAMPLES: dict[str, tuple[str, ...]] = {
    "equipment": EQUIPMENT_EXAMPLES,
    "location": LOCATION_EXAMPLES,
    "worker_role": WORKER_ROLE_EXAMPLES,
}

# 0件時の空振り軸検知（core/rag_service.py の _unmatched_axes）の対象軸。
# 部分一致（LIKE）で絞り込む軸に限る。この軸では「1件も一致しない」と
# 「その値がデータに存在しない」が定義上同値になるため、固定文の主張が常に正しい。
# 一方 severity/shift 等の enum 軸は RecordFilter.__post_init__ で構築時に検証済みの
# 有効値であり、date_*/hour_*/experience_* 等の範囲軸は値の存在ではなく範囲条件のため、
# これらの軸が単体で0件なのは「本当に0件」であって「値が存在しない」ではない。
# AXIS_VALUE_EXAMPLES と対象軸をずらして手書きするとドリフトするため、そのキーを
# 単一情報源として導出する。
PARTIAL_MATCH_AXES: tuple[str, ...] = tuple(AXIS_VALUE_EXAMPLES)


def severities_at_or_above(severity: str) -> tuple[str, ...]:
    """`severity` 以上の重大度一覧を返す（`VALID_SEVERITIES` は既に昇順）。

    「中位以上」のような順序を伴う絞り込みのドメイン知識をここに閉じ、
    infra 側には IN 句に束縛する値の一覧としてのみ渡す。
    """
    index = VALID_SEVERITIES.index(severity)
    return VALID_SEVERITIES[index:]


@dataclass(frozen=True, slots=True)
class RecordFilter:
    """事故記録の絞り込み条件。集計・検索の両方の経路で共有する。

    構築できた時点で値は検証済み、という不変条件を持つ。不正な値
    （未知の severity/shift）は構築時に `InvalidToolArgumentsError` を送出する。
    日付形式の検証は呼び出し側（`core/tools.py` の `parse_aggregate_arguments`）
    が担い、ここでは受け取った `date` 型をそのまま保持する。
    """

    severity: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    equipment: str | None = None
    location: str | None = None
    shift: str | None = None
    severity_min: str | None = None
    worker_role: str | None = None
    experience_min: int | None = None
    experience_max: int | None = None
    hour_from: int | None = None
    hour_to: int | None = None

    def __post_init__(self) -> None:
        if self.severity is not None and self.severity not in VALID_SEVERITIES:
            raise InvalidToolArgumentsError(
                f"未知の severity: {self.severity!r}（有効値: {list(VALID_SEVERITIES)}）"
            )
        if self.shift is not None and self.shift not in VALID_SHIFTS:
            raise InvalidToolArgumentsError(
                f"未知の shift: {self.shift!r}（有効値: {list(VALID_SHIFTS)}）"
            )
        if self.severity_min is not None and self.severity_min not in VALID_SEVERITIES:
            raise InvalidToolArgumentsError(
                f"未知の severity_min: {self.severity_min!r}（有効値: {list(VALID_SEVERITIES)}）"
            )
        if self.severity is not None and self.severity_min is not None:
            # 両方指定すると「重大度=X かつ severity_min 以上」という矛盾しうる
            # 条件になり、AND すると静かに 0 件になりうる。
            raise InvalidToolArgumentsError("severity と severity_min は同時に指定できません")
        for key, value in (("hour_from", self.hour_from), ("hour_to", self.hour_to)):
            if value is not None and not (0 <= value <= 23):
                raise InvalidToolArgumentsError(f"{key} は 0〜23 の範囲で指定してください: {value}")
        for key, value in (
            ("experience_min", self.experience_min),
            ("experience_max", self.experience_max),
        ):
            if value is not None and value < 0:
                raise InvalidToolArgumentsError(f"{key} は 0 以上で指定してください: {value}")
        if (
            self.experience_min is not None
            and self.experience_max is not None
            and self.experience_min > self.experience_max
        ):
            raise InvalidToolArgumentsError(
                f"experience_min（{self.experience_min}）が"
                f"experience_max（{self.experience_max}）を超えています"
            )

    @property
    def is_empty(self) -> bool:
        """いずれの軸も指定されていない（絞り込みなし = 全件対象）。"""
        return all(getattr(self, f.name) is None for f in fields(self))

    def specified_axes(self) -> tuple[str, ...]:
        """値が指定されている軸のキー一覧（`FILTER_KEYS` の部分集合）。"""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)

    def only(self, key: str) -> RecordFilter:
        """指定した軸だけを残した条件を返す。

        0 件の原因が「どの軸か」を切り分けるために使う（`core/rag_service.py`
        の空振り軸検知）。severity/severity_min は同時指定できない制約
        （`__post_init__`）があるが、単一軸だけを残すため常に構築できる。
        """
        return RecordFilter(**{key: getattr(self, key)})

    def to_dict(self) -> dict[str, str]:
        """指定済みの軸を文字列表現の辞書として返す（API レスポンス `basis.filters` 用）。

        日付は ISO 形式、それ以外は `str()` で文字列化する。人間向けの
        `describe()` とは用途が異なるが、同じ `specified_axes()` から導出する。
        """
        result: dict[str, str] = {}
        for axis in self.specified_axes():
            value = getattr(self, axis)
            result[axis] = value.isoformat() if isinstance(value, date) else str(value)
        return result

    def describe(self) -> str:
        """人間が読める条件の要約。回答生成プロンプトや 0 件時の文言に使う。"""
        parts: list[str] = []
        if self.severity is not None:
            parts.append(f"重大度={self.severity}")
        if self.severity_min is not None:
            parts.append(f"重大度={self.severity_min}以上")
        if self.date_from is not None or self.date_to is not None:
            date_from = self.date_from.isoformat() if self.date_from else "指定なし"
            date_to = self.date_to.isoformat() if self.date_to else "指定なし"
            parts.append(f"期間={date_from}〜{date_to}")
        if self.equipment is not None:
            parts.append(f"設備={self.equipment}")
        if self.location is not None:
            parts.append(f"場所={self.location}")
        if self.shift is not None:
            parts.append(f"勤務帯={self.shift}")
        if self.worker_role is not None:
            parts.append(f"職種={self.worker_role}")
        if self.experience_min is not None or self.experience_max is not None:
            exp_min = self.experience_min if self.experience_min is not None else "指定なし"
            exp_max = self.experience_max if self.experience_max is not None else "指定なし"
            parts.append(f"経験年数={exp_min}〜{exp_max}")
        if self.hour_from is not None or self.hour_to is not None:
            hour_from = self.hour_from if self.hour_from is not None else "指定なし"
            hour_to = self.hour_to if self.hour_to is not None else "指定なし"
            parts.append(f"時間帯={hour_from}時〜{hour_to}時")
        return "、".join(parts) if parts else "条件なし（全件）"


# 絞り込み軸のキー一覧の単一情報源。ツール引数スキーマ・LLM 引数の検証で使う。
FILTER_KEYS: tuple[str, ...] = tuple(f.name for f in fields(RecordFilter))
