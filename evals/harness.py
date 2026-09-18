"""評価専用の補助実装。

本番の `RagQaService` はツール名・引数を呼び出し元に返さない（監査ログにのみ
記録する設計）。
評価ではツール選択の正誤を直接検証したいため、`ChatModel` を包んでルーティング
結果を記録するスパイをここに用意する。`core.ports.ChatModel` の Protocol を
満たすだけで、本番コードには一切変更を加えない。
"""

from __future__ import annotations

from dataclasses import dataclass

from safety_qa.core.models import ToolCall
from safety_qa.core.ports import ChatModel


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    tool_name: str | None
    tool_arguments: dict[str, str]


class RecordingChatModel:
    """`ChatModel` を包み、直近の `route()` 結果を記録するスパイ。"""

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner
        self.last_routing: RoutingRecord | None = None

    def route(self, question: str) -> ToolCall | None:
        tool_call = self._inner.route(question)
        self.last_routing = RoutingRecord(
            tool_name=tool_call.name if tool_call else None,
            tool_arguments=dict(tool_call.arguments) if tool_call else {},
        )
        return tool_call

    def answer(self, prompt: str) -> str:
        return self._inner.answer(prompt)
