"""PocketSidekick application orchestrator."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .memory import SqliteMemory
from .persona import PersonaMode
from .quantization import export_quantization_config
from .router import RequestRouter
from .tools import ToolRegistry


class PocketSidekickApp:
    def __init__(self, db_path: str = "pocketsidekick.db") -> None:
        self.router = RequestRouter()
        self.tools = ToolRegistry()
        self.memory = SqliteMemory(db_path=db_path)
        self.persona_mode = PersonaMode()

    def handle_message(self, message: str) -> str:
        self.memory.add("user", message)
        routed = self.router.route(message)

        if routed.intent == "tool":
            result = self.tools.call(routed.tool_name or "", routed.payload)
            response = f"Tool '{routed.tool_name}' => {result}"
        else:
            history = self.memory.recent(limit=3)
            response = (
                "I logged your message. "
                f"Recent memory size={len(history)}. "
                f"Persona prompt: {self.persona_mode.persona.system_prompt()}"
            )

        formatted = self.persona_mode.format_response(response)
        self.memory.add("assistant", formatted)
        return formatted

    def set_persona(self, name: str, tone: str, style_guide: str) -> Dict[str, Any]:
        persona = self.persona_mode.set_persona(name=name, tone=tone, style_guide=style_guide)
        return asdict(persona)

    def export_quantization(self, output_path: str = "artifacts/quantization.json") -> str:
        return export_quantization_config(output_path=output_path)


if __name__ == "__main__":
    app = PocketSidekickApp(db_path=str(Path("pocketsidekick.db")))
    print(app.handle_message("Hello sidekick"))
    print(app.handle_message("/tool:word_count this is a tiny tool call"))
    print(app.export_quantization())
