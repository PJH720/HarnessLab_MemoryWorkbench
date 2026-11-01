"""Slash-command registry and input routing for the REPL (same shape as the Koog CLI's Command/route)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Collection

from . import lego
from .docqa import NotesError, resolve_mode
from .presets import Preset, find_preset

# A handler returns False to end the session. Handlers of commands with takes_args=True get the rest of the line.
Handler = Callable[..., bool]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    takes_args: bool = False

    def run(self, args: str = "") -> bool:
        return self.handler(args) if self.takes_args else self.handler()


@dataclass(frozen=True)
class Route:
    kind: str  # "empty" | "command" | "unknown" | "query" | "preset"
    command: Command | None = None
    text: str = ""
    args: str = ""


class CommandRegistry:
    def __init__(self, commands: list[Command]) -> None:
        self._by_name: dict[str, Command] = {}
        for cmd in commands:
            for key in (cmd.name, *cmd.aliases):
                self._by_name[key] = cmd
        self.commands = tuple(commands)

    def find(self, name: str) -> Command | None:
        return self._by_name.get(name.lower())


def route_input(line: str, registry: CommandRegistry, presets: Collection[str] = ()) -> Route:
    """Route one REPL line. A line equal to a preset key (e.g. "2") runs that preset."""
    text = line.strip()
    if not text:
        return Route("empty")
    if text.startswith("/"):
        name, _, args = text[1:].partition(" ")
        cmd = registry.find(name) if name else None
        return Route("command", cmd, args=args.strip()) if cmd else Route("unknown", text=text)
    if text in presets:
        return Route("preset", text=text)
    return Route("query", text=text)


# --- argument parsers for /record and /compare (pure, so they are tested without engines) -----

RECORD_USAGE = '사용법: /record "<영어 증상 문장>" -> "<질병 이름>"   예) /record "itchy red rash on both arms" -> "psoriasis"'
COMPARE_USAGE = "사용법: /compare <하네스1> <하네스2> [프리셋 번호|영어 증상 문장]   또는 /compare 1|2|3|<증상> (기억 없음 vs 지금 블록)"
_ARROW = re.compile(r"\s*(?:->|→)\s*")
_QUOTED = re.compile(r"""^\s*(["'“])(?P<s>.+?)["'”]\s*(?:->|→)\s*(?P<d>.+?)\s*$""", re.S)
_QUOTES = "\"'“” "


class UsageError(ValueError):
    """Bad command arguments; the message is shown to the user as is."""


def parse_record(args: str) -> tuple[str, str]:
    """`"<symptoms>" -> "<disease>"` (quotes optional) → (symptoms, disease)."""
    m = _QUOTED.match(args)
    if m:
        symptoms, disease = m["s"], m["d"]
    else:
        parts = _ARROW.split(args.strip())
        if len(parts) != 2:
            raise UsageError(RECORD_USAGE)
        symptoms, disease = parts
    symptoms, disease = symptoms.strip(_QUOTES), disease.strip(_QUOTES)
    if not symptoms or not disease:
        raise UsageError(RECORD_USAGE)
    return symptoms, disease


@dataclass(frozen=True)
class CompareArgs:
    left: str
    right: str
    symptoms: str
    preset: Preset | None


def _block_name(token: str) -> str | None:
    try:
        return lego.resolve(token).name
    except lego.UnknownHarness:
        return None


def parse_compare(args: str, current: str) -> CompareArgs:
    """Two block names (aliases allowed) then a preset or text; otherwise legacy no_memory vs `current`.

    The two-block form needs both of the first two words to be block names, so symptom text that merely
    starts with an alias ("all my joints ache") keeps its legacy meaning. Without a case, preset 1 runs.
    """
    text = args.strip()
    if not text:
        raise UsageError(COMPARE_USAGE)
    words = text.split()
    first = _block_name(words[0])
    second = _block_name(words[1]) if len(words) > 1 else None
    if first and second:
        rest = " ".join(words[2:]) or "1"
        left, right = first, second
    elif words[0].lower() in lego.BLOCKS or (first and (len(words) == 1 or find_preset(" ".join(words[1:])))):
        # a canonical name never occurs in symptom text; "knn 2" is one block plus a preset, not symptoms
        raise UsageError(f"하네스를 두 개 지정하세요. {COMPARE_USAGE}")
    else:
        rest, left, right = text, "no_memory", current
    preset = find_preset(rest)
    return CompareArgs(left, right, preset.symptoms if preset else rest, preset)


STUDY_RECORD_USAGE = ('사용법 (study): /record "<개념 또는 질문>" -> "<답 또는 규칙>"   '
                      '예) /record "Welch t-test" -> "use it when the two groups have unequal variances"')
STUDY_COMPARE_USAGE = "사용법 (study): /compare <모드1> <모드2> [영어 질문]   (질문을 비우면 직전 질문)  모드: no_memory, knn_retrieval, lesson_cues"


def parse_study_compare(args: str, last_question: str = "") -> tuple[str, str, str]:
    """`<mode> <mode> [question]` → (left, right, question); the question defaults to the last one asked."""
    words = args.split()
    if len(words) < 2:
        raise UsageError(STUDY_COMPARE_USAGE)
    try:
        left, right = resolve_mode(words[0]).name, resolve_mode(words[1]).name
    except NotesError as exc:
        raise UsageError(f"{exc}\n{STUDY_COMPARE_USAGE}") from exc
    question = " ".join(words[2:]) or last_question
    if not question:
        raise UsageError(f"비교할 질문이 없습니다. {STUDY_COMPARE_USAGE}")
    return left, right, question
