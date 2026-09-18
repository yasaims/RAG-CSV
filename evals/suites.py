"""評価スイートの定義。

スイート名は CLI 引数の choices・モデル名の選択・ベースラインファイル名の
複数箇所に散らばりやすいので 1 箇所にまとめる。SPEC.md「回答品質の評価」節が
フェーズ2 で L4（LLM-as-Judge）を要件化しており、スイートは今後も増える前提
（AGENTS.md「データサイズとインフラのスケールまたは置換を想定する」）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from safety_qa.infra.settings import Settings


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    name: str
    # そのスイートの指標を左右するモデルは何か（retrieval: 埋め込みモデル /
    # llm: 生成モデル）。比較可能性の判定条件（MeasurementConditions.model）に使う。
    model: Callable[[Settings], str]
    # --repeat が指標に意味を持つか。retrieval は単発実行のみで repeat 概念を持たない。
    uses_repeat: bool
    # 実行年が指標に意味を持つか。
    date_dependent: bool


SUITES: dict[str, SuiteSpec] = {
    "retrieval": SuiteSpec(
        name="retrieval",
        model=lambda settings: settings.embedding_model_name,
        uses_repeat=False,
        date_dependent=False,
    ),
    "llm": SuiteSpec(
        name="llm",
        model=lambda settings: settings.ollama_model,
        uses_repeat=True,
        date_dependent=True,
    ),
}
