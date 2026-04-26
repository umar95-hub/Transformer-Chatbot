"""Tool registry for PocketSidekick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


ToolHandler = Callable[[str], str]


@dataclass
class Tool:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self.register(
            Tool(
                name="echo",
                description="Echoes the input text for debugging.",
                handler=lambda text: text,
            )
        )
        self.register(
            Tool(
                name="word_count",
                description="Returns number of words in the input.",
                handler=lambda text: str(len(text.split())),
            )
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def call(self, name: str, payload: str) -> str:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools))
            raise ValueError(f"Unknown tool '{name}'. Available tools: {available}")
        return self._tools[name].handler(payload)
