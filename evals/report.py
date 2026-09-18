"""評価結果のレポート生成（JSON / Markdown）。

数値の集計だけでは安全管理担当者・上長に何が起きたかを説明できない。
**落ちたケース（質問文・期待した記録 id・実際に返った id）を必ず一覧で
出す**ことを優先し、集計指標はその補助に置く。

JSON は `var/eval/<suite>.json` に機械可読な形で出力し、Markdown は
`var/eval/<suite>.md` として GitHub Actions の Job Summary にそのまま流し込む。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from evals.baseline import Baseline, ComparisonResult
from evals.conditions import ALL_CONDITION_KEYS, MeasurementConditions


@dataclass
class CaseResult:
    """1 ケースの実行結果。"""

    id: str
    category: str
    question: str
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


# 「評価不能」（依存の取得失敗・データセット不整合・比較不成立・ゲート対象
# ケース0件）は品質低下ではない（SPEC.md「障害を無言で握り潰さない」）。
# 合否とは別軸で扱う。
SuiteStatus = Literal["passed", "failed", "inconclusive"]

# - dependency: Ollama 不達・HuggingFace 取得失敗等、評価そのものが実行できなかった
# - dataset: load_dataset / validate_must_hit_ids_exist の検証失敗（ラベルの腐敗等）
# - no_cases: ゲート対象ケースが 0 件（`_mean([])` が 1.0 を返すため「満点」に
#   化けるのを防ぐ。カバレッジ喪失と満点が同じ見た目になってはならない）
# - no_baseline / incomparable: `evals.baseline.ComparisonResult.status` と同義。
#   比較そのものは実行できたが、判定材料が無い／条件が一致しない
InconclusiveReason = Literal["dependency", "dataset", "no_cases", "no_baseline", "incomparable"]

_INCONCLUSIVE_REASON_LABELS: dict[InconclusiveReason, str] = {
    "dependency": "依存の取得に失敗しました（Ollama 未起動・埋め込みモデル取得失敗等）",
    "dataset": "評価データセットの検証に失敗しました",
    "no_cases": "ゲート対象のケースが 0 件でした",
    "no_baseline": "ベースライン未設定のため低下検出ができません",
    "incomparable": "ベースラインと実行条件が異なるため比較できません",
}


@dataclass
class SuiteReport:
    """1 スイート（retrieval / llm）分のレポート。"""

    suite: str
    cases: list[CaseResult]
    aggregate_metrics: dict[str, float]
    reference_metrics: dict[str, float]
    # 現在実行の測定条件。依存失敗等で評価そのものが実行できなかった場合は None。
    conditions: MeasurementConditions | None
    # 比較を試みなかった場合（依存失敗・データセット不整合・ゲート対象ケース0件）は
    # None。比較を試みた場合は必ず設定される（no_baseline / incomparable を含む）。
    comparison: ComparisonResult | None
    baseline: Baseline | None
    # 評価そのものが実行できなかった／比較が成立しなかった理由。None なら
    # `comparison` と `violations` から通常どおり合否を判定する。
    inconclusive_reason: InconclusiveReason | None = None
    inconclusive_detail: str | None = None
    # `--update-baseline` 実行時は「低下」「比較不成立」があっても合否には含めない
    # （そのかわり Markdown 上で明示し、絶対要件違反があれば別途ベースライン更新
    # 自体を止める）。
    baseline_update_run: bool = False

    @property
    def violations(self) -> list[tuple[str, str]]:
        """(ケース id, 違反内容) の一覧。"""
        return [(c.id, v) for c in self.cases for v in c.violations]

    @property
    def status(self) -> SuiteStatus:
        if self.inconclusive_reason is not None:
            return "inconclusive"
        if self.violations:
            return "failed"
        if self.comparison is None:
            # 構築時の不変条件: inconclusive_reason が None なら comparison は必ず設定される。
            return "passed"
        if self.comparison.status in ("no_baseline", "incomparable"):
            # 比較が成立していない = 退行の有無が「わからない」。緑にすると
            # 「沈黙のスキップにも失敗にもしない」がゲートに反映されない。
            return "passed" if self.baseline_update_run else "inconclusive"
        if self.comparison.has_regression:
            return "passed" if self.baseline_update_run else "failed"
        return "passed"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "status": self.status,
            "inconclusive_reason": self.inconclusive_reason,
            "inconclusive_detail": self.inconclusive_detail,
            "baseline_update_run": self.baseline_update_run,
            "aggregate_metrics": self.aggregate_metrics,
            "reference_metrics": self.reference_metrics,
            "conditions": asdict(self.conditions) if self.conditions else None,
            "baseline": asdict(self.baseline) if self.baseline else None,
            "comparison": (
                {
                    "status": self.comparison.status,
                    "reason": self.comparison.reason,
                    "regressions": [asdict(r) for r in self.comparison.regressions],
                    "skipped_metrics": self.comparison.skipped_metrics,
                }
                if self.comparison is not None
                else None
            ),
            "cases": [
                {
                    "id": c.id,
                    "category": c.category,
                    "question": c.question,
                    "ok": c.ok,
                    "metrics": c.metrics,
                    "violations": c.violations,
                    "detail": c.detail,
                }
                for c in self.cases
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2) + "\n"

    def to_markdown(self) -> str:
        lines: list[str] = [f"# 評価結果: {self.suite}", ""]
        status_marks = {"passed": "✅ 合格", "failed": "❌ 不合格", "inconclusive": "⚠️ 評価不能"}
        lines.append(status_marks[self.status])
        lines.append("")

        if self.status == "inconclusive":
            # 理由は inconclusive_reason（依存失敗・データセット不整合・ゲート対象
            # ケース0件）か、comparison.status（no_baseline / incomparable）のどちらか
            # 一方に必ず入る。ここで統一して見せないと、下部の比較不能メッセージまで
            # スクロールしないと理由が分からない。
            lines.append("## 評価不能の理由")
            lines.append("")
            if self.inconclusive_reason is not None:
                lines.append(f"- {_INCONCLUSIVE_REASON_LABELS[self.inconclusive_reason]}")
                if self.inconclusive_detail:
                    lines.append(f"- 詳細: {self.inconclusive_detail}")
            elif self.comparison is not None and self.comparison.status != "compared":
                lines.append(f"- {_INCONCLUSIVE_REASON_LABELS[self.comparison.status]}")
                if self.comparison.reason:
                    lines.append(f"- 詳細: {self.comparison.reason}")
            lines.append("")

        # 落ちたケースを主役にする（数値だけでは何が起きたか説明できない）。
        failing = [c for c in self.cases if not c.ok]
        if failing:
            lines.append("## 落ちた質問（原因を切り分けられる形で表示）")
            lines.append("")
            for c in failing:
                lines.append(f"### `{c.id}`: {c.question}")
                for v in c.violations:
                    lines.append(f"- {v}")
                must_hit = c.detail.get("must_hit_ids")
                hit = c.detail.get("hit_ids") or c.detail.get("cited_ids")
                if must_hit is not None or hit is not None:
                    lines.append(f"  - 期待した記録 id: {must_hit}")
                    lines.append(f"  - 実際に返った id: {hit}")
                lines.append("")

        # retrieval スイートは recall@k を絶対要件として個々のケースを落とさない
        # （低下の検出はベースライン比較に委ねる。上の「落ちた質問」には出ない）。
        # そのため取りこぼしのある質問は別枠で、BM25・ベクトルのどちらが
        # 外したかを切り分けられる形で出す。
        imperfect_recall = [
            c
            for c in self.cases
            if c.detail.get("retrieval") is not None and c.metrics.get("recall_at_k", 1.0) < 1.0
        ]
        if imperfect_recall:
            lines.append("## 検索経路の内訳（recall@k が 1.0 未満の質問）")
            lines.append("")
            for c in imperfect_recall:
                retrieval = c.detail["retrieval"]
                must_hit = c.detail.get("must_hit_ids") or []
                lines.append(
                    f"### `{c.id}`: {c.question} (recall_at_k={c.metrics['recall_at_k']:.2f})"
                )
                for record_id in must_hit:
                    in_bm25 = "○" if record_id in retrieval["bm25_ids"] else "×"
                    in_vector = "○" if record_id in retrieval["vector_ids"] else "×"
                    lines.append(f"  - {record_id}: BM25={in_bm25} / ベクトル={in_vector}")
                lines.append("")

        if self.cases and self.status != "inconclusive":
            # 評価不能時は「N問中M問が正しく答えられた」という合格見出しを出さない
            # （⚠️評価不能の直後に成功を示唆する見出しが出ると矛盾して見える）。
            lines.append(
                f"## 見出し指標: {len(self.cases) - len(failing)} / {len(self.cases)} 問が"
                "根拠付きで正しく答えられた"
            )
            lines.append("")

        if self.aggregate_metrics:
            lines.append("## 集計指標（ベースライン比較対象）")
            lines.append("")
            lines.append("| 指標 | 現在値 | ベースライン |")
            lines.append("|---|---|---|")
            baseline_metrics = self.baseline.metrics if self.baseline else {}
            for name, value in self.aggregate_metrics.items():
                base = baseline_metrics.get(name)
                base_str = f"{base:.3f}" if base is not None else "未設定"
                lines.append(f"| {name} | {value:.3f} | {base_str} |")
            lines.append("")

        if self.reference_metrics:
            lines.append("## 参考指標（合否には影響しない）")
            lines.append("")
            lines.append("| 指標 | 値 |")
            lines.append("|---|---|")
            for name, value in self.reference_metrics.items():
                lines.append(f"| {name} | {value:.3f} |")
            lines.append("")

        if self.conditions is not None:
            lines.append("## 比較条件")
            lines.append("")
            if self.baseline is not None:
                mismatches = set(self.baseline.conditions.mismatches(self.conditions))
                # ALL_CONDITION_KEYS を単一情報源にする（ハードコードした部分集合
                # だと新しいキー追加時にここだけ更新漏れが起き、不一致マークが
                # 出ないまま「比較条件」表から消える）。
                for field_name in ALL_CONDITION_KEYS:
                    current_value = getattr(self.conditions, field_name)
                    baseline_value = getattr(self.baseline.conditions, field_name)
                    mark = "（不一致）" if any(f"{field_name}:" in m for m in mismatches) else ""
                    lines.append(
                        f"- {field_name}: {baseline_value!r} -> {current_value!r} {mark}".rstrip()
                    )
            else:
                lines.append(
                    f"- 現在の実行条件のみ（ベースライン未設定）: {asdict(self.conditions)}"
                )
            lines.append("")

        if self.comparison is not None:
            if (
                self.comparison.status in ("no_baseline", "incomparable")
                and self.baseline_update_run
            ):
                # 通常はここが status=="inconclusive" になり、理由は冒頭の
                # 「## 評価不能の理由」に既に出ている。ここに来るのは
                # `--update-baseline` で比較不成立を合否対象から外した場合のみ
                # （status は "passed"）なので、その旨だけ補足する。
                lines.append(
                    f"> {_INCONCLUSIVE_REASON_LABELS[self.comparison.status]}"
                    f"（{self.comparison.reason}）。`--update-baseline` 指定のため、"
                    "比較不成立のまま合否には含めていません。"
                )
                lines.append("")
            elif self.comparison.regressions:
                lines.append("## ベースラインからの低下")
                lines.append("")
                for r in self.comparison.regressions:
                    lines.append(f"- `{r.metric}`: {r.baseline_value:.3f} -> {r.current_value:.3f}")
                if self.baseline_update_run:
                    lines.append("")
                    lines.append(
                        "> `--update-baseline` 指定のため、この低下は合否 (passed) に含めていません。"
                    )
                lines.append("")

        if self.cases:
            lines.append("## ケース別結果")
            lines.append("")
            lines.append("| id | category | question | 状態 | 指標 |")
            lines.append("|---|---|---|---|---|")
            for c in self.cases:
                if not c.ok:
                    mark = "❌"
                elif self.status == "inconclusive":
                    # 個々のケースに違反は無いが、スイート全体は評価不能
                    # （比較不成立等）。緑のチェックは「合格」を示唆してしまうため
                    # 中立の表示にする。
                    mark = "-"
                else:
                    mark = "✅"
                metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in c.metrics.items())
                lines.append(f"| {c.id} | {c.category} | {c.question} | {mark} | {metrics_str} |")
            lines.append("")

        return "\n".join(lines)


def write_report(report: SuiteReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report.suite}.json"
    md_path = output_dir / f"{report.suite}.md"
    json_path.write_text(report.to_json(), encoding="utf-8", newline="\n")
    md_path.write_text(report.to_markdown(), encoding="utf-8", newline="\n")
    return json_path, md_path
