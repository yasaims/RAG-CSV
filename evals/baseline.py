"""ベースライン（暫定合否基準）の読み書き・比較。

固定閾値を置かず、直近実測値からの低下のみを失敗条件にする
（AGENTS.md「評価基準を勝手に数値設定しないこと」）。測定条件
（`MeasurementConditions`）のいずれかが変わった比較は「比較不能」として明示し、
沈黙のスキップにも失敗にもしない。

ベースラインは `evals/baseline/<suite>.<runner>.json` に環境別で持つ
（`baseline_path`）。1 スイート 1 ファイルにすると local と CI のどちらか一方は
必ず比較不能になる。環境ごとに自分の基準と比較できるようにする。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from evals.conditions import MeasurementConditions


class InvalidBaselineError(ValueError):
    """ベースライン JSON の形式が `Baseline` と一致しない。"""


@dataclass(frozen=True, slots=True)
class Baseline:
    """暫定ベースライン 1 件分。実測条件を併記し、何の数値かを追跡可能にする。"""

    measured_at: str  # ISO 8601
    commit: str
    conditions: MeasurementConditions
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class Regression:
    metric: str
    baseline_value: float
    current_value: float


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """ベースラインとの比較結果。

    `status` が `"incomparable"` の場合、退行の有無は判定できない
    （低下でも向上でもなく「わからない」）ことを明示する。
    """

    status: Literal["no_baseline", "incomparable", "compared"]
    regressions: list[Regression] = field(default_factory=list)
    reason: str | None = None
    # ゲート対象に指定されたが、ベースライン側に無く比較できなかった指標名。
    # 黙って除外すると「比較成立と称して 0 個の指標しか検査していない」状態が
    # 緑として報告される。
    skipped_metrics: list[str] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return self.status == "compared" and bool(self.regressions)


def baseline_path(baseline_dir: Path, suite: str, runner: str) -> Path:
    return baseline_dir / f"{suite}.{runner}.json"


def load_baseline(path: Path) -> Baseline | None:
    """ベースライン JSON を読み込む。存在しない場合は None（未設定として扱う）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        conditions = MeasurementConditions(**data["conditions"])
        return Baseline(
            measured_at=data["measured_at"],
            commit=data["commit"],
            conditions=conditions,
            metrics=data["metrics"],
        )
    except (TypeError, KeyError, ValueError, json.JSONDecodeError) as e:
        raise InvalidBaselineError(f"{path} の形式が不正です: {e}") from e


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(baseline), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def compare_to_baseline(
    baseline: Baseline | None,
    *,
    current_metrics: Mapping[str, float],
    gated_metrics: list[str],
    conditions: MeasurementConditions,
) -> ComparisonResult:
    """`gated_metrics` に指定した指標のみ、ベースラインからの低下を検出する。

    測定条件（model・runner・corpus_version・dataset_version・top_n・repeat）の
    いずれかが異なる場合は比較を成立させず `"incomparable"` を返す（SPEC.md が
    モデルの設定切替・データ量の増加を要件にしているため、切替は起きる前提で
    設計する）。
    """
    if baseline is None:
        return ComparisonResult(status="no_baseline", reason="ベースライン未設定（初回実行）")

    mismatches = baseline.conditions.mismatches(conditions)
    if mismatches:
        return ComparisonResult(status="incomparable", reason="; ".join(mismatches))

    gated = sorted(set(gated_metrics))
    skipped = [
        name for name in gated if name not in baseline.metrics or name not in current_metrics
    ]
    if skipped:
        return ComparisonResult(
            status="incomparable",
            reason=f"ベースラインに無い指標のため比較できません: {skipped}",
            skipped_metrics=skipped,
        )

    regressions = [
        Regression(
            metric=name,
            baseline_value=baseline.metrics[name],
            current_value=current_metrics[name],
        )
        for name in gated
        if current_metrics[name] < baseline.metrics[name]
    ]
    return ComparisonResult(status="compared", regressions=regressions)
