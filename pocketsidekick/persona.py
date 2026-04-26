"""Persona mode for response shaping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Persona:
    name: str = "PocketSidekick"
    tone: str = "helpful, concise, and practical"
    style_guide: str = "Give actionable next steps and keep responses compact."

    def system_prompt(self) -> str:
        return (
            f"You are {self.name}. Respond in a {self.tone} tone. "
            f"Style rule: {self.style_guide}"
        )


class PersonaMode:
    def __init__(self, persona: Persona | None = None) -> None:
        self._persona = persona or Persona()

    @property
    def persona(self) -> Persona:
        return self._persona

    def set_persona(self, name: str, tone: str, style_guide: str) -> Persona:
        self._persona = Persona(name=name, tone=tone, style_guide=style_guide)
        return self._persona

    def format_response(self, raw_response: str) -> str:
        return f"[{self._persona.name}] {raw_response.strip()}"
