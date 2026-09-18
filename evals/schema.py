"""評価データセットのスキーマと YAML ローダ。

集計系の期待件数はここに書かない。期待引数から
`AccidentRecordRepository.aggregate()` を実行時に呼んで導出することで、
`AI/data/accidents.csv` の行数・内容が変わってもデータセットが腐らないようにする
（AGENTS.md「データサイズのスケールを想定する」）。

相対日付（`temporal` カテゴリ）も同じ理由で固定日付を書かない。実行日から導出する
（固定日付は CSV の変更ではなく時計の進行で腐るため）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from safety_qa.core.aggregation import VALID_GROUP_BY
from safety_qa.core.exceptions import InvalidToolArgumentsError
from safety_qa.core.tools import parse_aggregate_tool_arguments

DATASET_DIR = Path(__file__).parent / "dataset"

Category = Literal["search", "analysis", "aggregate", "temporal", "unsupported"]

_DATASET_FILES = (
    "search.yaml",
    "analysis.yaml",
    "aggregate.yaml",
    "temporal.yaml",
    "unsupported.yaml",
)

# `temporal` カテゴリの相対日付トークン -> 実行日からの導出関数。
# 固定日付ではなくトークン名をデータセットに書くことで、実行のたびに実行日
# 基準で期待条件を組み立てる（相対日付の解釈ずれの検証に使う）。
# `next_year_*` は「収録期間外に必ずなる期間」を作るためのトークン（ADR 007）。
# 評価基準日はコーパス最終日に固定するため（`evals/run.py`）、翌年は常に
# 収録期間の外側になる。
_RELATIVE_DATE_TOKENS: dict[str, Callable[[date], date]] = {
    "this_year_start": lambda today: date(today.year, 1, 1),
    "this_year_end": lambda today: date(today.year, 12, 31),
    "last_year_start": lambda today: date(today.year - 1, 1, 1),
    "last_year_end": lambda today: date(today.year - 1, 12, 31),
    "next_year_start": lambda today: date(today.year + 1, 1, 1),
    "next_year_end": lambda today: date(today.year + 1, 12, 31),
}


def resolve_relative_date(token: str, *, today: date) -> str:
    try:
        return _RELATIVE_DATE_TOKENS[token](today).isoformat()
    except KeyError:
        raise ValueError(
            f"未知の相対日付トークン: {token!r}（有効値: {list(_RELATIVE_DATE_TOKENS)}）"
        ) from None


class EvalCase(BaseModel):
    """1 件の評価ケース。

    - search / analysis: `must_hit_ids`（top_n に必ず含まれるべき最小集合）を検証する
    - aggregate: `expected_arguments` から実行時に導出した件数と回答文中の件数を照合する
    - temporal: `relative_arguments`（実行日基準）を `expected_arguments` に合成してから
      aggregate と同様に検証する。偽ゼロ（絶対要件3）の主な検証対象
    - unsupported: ツール未呼び出し・件数非提示を検証する。`unsupported_group_by` を
      指定すると、その軸が本当に対応外のままかを実行時に照合する
    - `expected_degraded_reason`: aggregate/temporal のうち、回答が2パス目の
      LLM生成を経ず固定文で縮退すること自体を期待するケースに設定する
      （"out_of_coverage" 等）。真に収録期間外の質問（期待件数0件）で正しく
      縮退できているかを検証する（絶対要件Bの除外対象を裏側から検証する）。
      未指定時は通常どおり非縮退を期待する
    """

    id: str
    category: Category
    question: str
    rationale: str
    must_hit_ids: list[str] = Field(default_factory=list)
    expected_tool: str | None = None
    expected_arguments: dict[str, str] = Field(default_factory=dict)
    relative_arguments: dict[str, str] = Field(default_factory=dict)
    unsupported_group_by: str | None = None
    expected_degraded_reason: str | None = None

    def resolved_expected_arguments(self, *, today: date) -> dict[str, str]:
        """`relative_arguments` を実行日基準で解決し `expected_arguments` に合成する。"""
        resolved = dict(self.expected_arguments)
        for key, token in self.relative_arguments.items():
            resolved[key] = resolve_relative_date(token, today=today)
        return resolved


def load_dataset(dataset_dir: Path = DATASET_DIR, *, today: date | None = None) -> list[EvalCase]:
    """`evals/dataset/*.yaml` を読み込み、全カテゴリのケースを 1 つのリストにまとめて返す。

    読み込み時点で以下を検証し、不正なデータセットはここで落とす
    （Ollama を起動する前に気づけるようにする）:
    - id の重複
    - aggregate / temporal の `expected_arguments`（相対日付解決後）が
      本番と同じ検証器（`parse_aggregate_tool_arguments`）を通ること
    - unsupported の `unsupported_group_by` が実際に `VALID_GROUP_BY` の外であること
      （軸が追加されたらデータセット側が先に落ちる）
    """
    today = today or date.today()
    cases: list[EvalCase] = []
    for filename in _DATASET_FILES:
        path = dataset_dir / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        cases.extend(EvalCase.model_validate(item) for item in raw)

    seen_ids: set[str] = set()
    for case in cases:
        if case.id in seen_ids:
            raise ValueError(f"評価ケース id が重複しています: {case.id!r}")
        seen_ids.add(case.id)

        if case.category in ("aggregate", "temporal"):
            resolved = case.resolved_expected_arguments(today=today)
            try:
                parse_aggregate_tool_arguments(resolved)
            except InvalidToolArgumentsError as e:
                raise ValueError(
                    f"評価ケース {case.id!r} の expected_arguments が不正です: {e}"
                ) from e

        if case.category == "unsupported" and case.unsupported_group_by is not None:
            if case.unsupported_group_by in VALID_GROUP_BY:
                raise ValueError(
                    f"評価ケース {case.id!r} の unsupported_group_by="
                    f"{case.unsupported_group_by!r} は現在対応済みの軸です"
                    "（VALID_GROUP_BY に含まれる）。対応済みになった軸なので"
                    "ケースの見直しが必要です。"
                )
    return cases


def validate_must_hit_ids_exist(cases: list[EvalCase], known_ids: Collection[str]) -> None:
    """`must_hit_ids` が全てコーパスに実在することを検証する（fail-fast）。

    起動時 ETL は CSV に存在しない id を DELETE する（`infra/ingest.py`）。
    CSV 更新でラベルが消えると「Recall 低下」に化けて品質低下と区別できなくなるため、
    指標を計算する前にここで検出する。
    """
    known = set(known_ids)
    missing = {
        case.id: absent
        for case in cases
        if (absent := [rid for rid in case.must_hit_ids if rid not in known])
    }
    if missing:
        details = "; ".join(f"{case_id}: {ids}" for case_id, ids in missing.items())
        raise ValueError(
            f"評価データセットの must_hit_ids がコーパスに実在しません（ラベルの腐敗の疑い）: {details}"
        )


def validate_expected_degraded_reason(
    cases: list[EvalCase], data_period: tuple[date, date], *, today: date
) -> None:
    """`expected_degraded_reason="out_of_coverage"` のケースが実際にコーパスの
    収録期間外を指しているかを検証する（fail-fast）。

    評価基準日をコーパス最終日に固定する運用（`evals/run.py`）では
    `next_year_start`/`next_year_end` は常に収録期間外になるはずだが、
    トークンの導出ロジックが変わった場合や `today` の決め方が変わった場合に、
    ラベルの前提が静かに崩れる（＝縮退を検証しているつもりで実は非縮退の
    ケースを検証してしまう）のを防ぐ。
    """
    first, last = data_period
    broken: list[str] = []
    for case in cases:
        if case.expected_degraded_reason != "out_of_coverage":
            continue
        resolved = case.resolved_expected_arguments(today=today)
        date_from = date.fromisoformat(resolved["date_from"]) if "date_from" in resolved else None
        date_to = date.fromisoformat(resolved["date_to"]) if "date_to" in resolved else None
        out_of_coverage = (date_from is not None and date_from > last) or (
            date_to is not None and date_to < first
        )
        if not out_of_coverage:
            broken.append(f"{case.id}: 解決済み期間={date_from}〜{date_to}")
    if broken:
        details = "; ".join(broken)
        raise ValueError(
            "評価データセットの expected_degraded_reason='out_of_coverage' のケースが、"
            f"実際にはコーパスの収録期間（{first}〜{last}）の内側を指しています"
            f"（ラベルの前提が崩れている疑い）: {details}"
        )


def dataset_version(cases: list[EvalCase]) -> str:
    """スイートが実際に評価するケース集合から版識別子を導出する。

    ベースラインとの比較可否判定（`evals/baseline.py`）に使う。対象ケースの
    内容が変わればこの値も変わり、古いベースラインとの比較が「比較不能」として
    明示される（沈黙のまま比較しない）。

    ファイル全体ではなくケース集合を対象にするのは、粒度を対象スイートに
    合わせるため。例えば retrieval スイートは `must_hit_ids` を持つケースしか
    使わないので、`unsupported.yaml` の変更はそのケース集合に影響せず、
    無関係な変更で比較不能（＝評価不能）にはならない。
    """
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda c: c.id):
        digest.update(case.model_dump_json().encode("utf-8"))
    return digest.hexdigest()[:12]
