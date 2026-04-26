"""Request router for PocketSidekick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RoutedRequest:
    intent: str
    tool_name: Optional[str]
    payload: str


class RequestRouter:
    """Tiny intent router.

    Conventions:
    - '/tool:<name> <payload>' invokes a tool.
    - everything else is chat.
    """

    TOOL_PREFIX = "/tool:"

    def route(self, message: str) -> RoutedRequest:
        stripped = message.strip()
        if stripped.startswith(self.TOOL_PREFIX):
            command = stripped[len(self.TOOL_PREFIX) :]
            if " " in command:
                tool_name, payload = command.split(" ", 1)
            else:
                tool_name, payload = command, ""
            return RoutedRequest(intent="tool", tool_name=tool_name.strip(), payload=payload)

        return RoutedRequest(intent="chat", tool_name=None, payload=message)
