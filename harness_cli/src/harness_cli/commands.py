"""Slash-command registry and input routing for the REPL (same shape as the Koog CLI's Command/route)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# A handler returns False to end the session.
Handler = Callable[[], bool]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Handler
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Route:
    kind: str  # "empty" | "command" | "unknown" | "query"
    command: Command | None = None
    text: str = ""


class CommandRegistry:
    def __init__(self, commands: list[Command]) -> None:
        self._by_name: dict[str, Command] = {}
        for cmd in commands:
            for key in (cmd.name, *cmd.aliases):
                self._by_name[key] = cmd
        self.commands = tuple(commands)

    def find(self, name: str) -> Command | None:
        return self._by_name.get(name.lower())


def route_input(line: str, registry: CommandRegistry) -> Route:
    text = line.strip()
    if not text:
        return Route("empty")
    if text.startswith("/"):
        cmd = registry.find(text[1:].split()[0]) if len(text) > 1 else None
        return Route("command", cmd) if cmd else Route("unknown", text=text)
    return Route("query", text=text)
