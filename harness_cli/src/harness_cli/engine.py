"""Load the evaluated memory system and expose predictions together with their evidence.

The prompt, retrieval and confusion warnings all come from the harness code that was
scored in the midterm run; this module only reads them back for display.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .config import HARNESS_LAB, Settings, prepare_runtime_env

_SYMPTOM_SLOT = "\u0000SYMPTOMS\u0000"
_EXAMPLES_RE = re.compile(r"\*\*Similar solved examples:\*\*\s*(?P<body>.*?)\s*\*\*Known confusions", re.S)
_CONFUSIONS_RE = re.compile(r"\*\*Known confusions[^\n]*\n(?P<body>.*?)\s*\*\*Problem:\*\*", re.S)
_QA_RE = re.compile(r"Q: (?P<q>.*?)\nA: (?P<a>.*?)(?=\n\nQ: |\Z)", re.S)


@dataclass(frozen=True)
class Evidence:
    question: str
    label: str
    past_mistake: bool


@dataclass(frozen=True)
class Prediction:
    diagnosis: str
    reasoning: str
    warnings: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    prompt_chars: int
    seconds: float


@dataclass(frozen=True)
class MemoryStats:
    system: str
    examples: int
    labels: tuple[tuple[str, int], ...]
    mistakes: int
    confusions: tuple[tuple[str, str, int], ...]


def _import_harness() -> tuple[Callable, Callable, Callable, Callable]:
    if str(HARNESS_LAB) not in sys.path:
        sys.path.insert(0, str(HARNESS_LAB))
    from text_classification.data.loaders import _load_symptom2disease
    from text_classification.inner_loop import load_memory_system
    from text_classification.llm import LLM
    from text_classification.memory_system import extract_json_field

    return load_memory_system, LLM, _load_symptom2disease, extract_json_field


def _symptom_template(load_split: Callable) -> str:
    """Reuse the dataset's own diagnostician prompt so CLI input matches evaluation input."""
    example = load_split("train", limit=1)[0]
    raw = example["raw_question"]
    if example["input"].count(raw) != 1:
        raise RuntimeError("Could not locate the symptom slot in the dataset prompt template")
    return example["input"].replace(raw, _SYMPTOM_SLOT)


def parse_prompt(prompt: str, mistake_texts: set[str]) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
    """Read the selected examples and confusion warnings back out of a built prompt."""
    evidence: list[Evidence] = []
    if match := _EXAMPLES_RE.search(prompt):
        for qa in _QA_RE.finditer(match.group("body")):
            q = qa.group("q").strip()
            evidence.append(Evidence(q, qa.group("a").strip(), q in mistake_texts))
    warnings: tuple[str, ...] = ()
    if match := _CONFUSIONS_RE.search(prompt):
        lines = [ln.lstrip("- ").strip() for ln in match.group("body").splitlines()]
        warnings = tuple(ln for ln in lines if ln and ln != "none recorded")
    return tuple(evidence), warnings


class HarnessEngine:
    def __init__(self, settings: Settings, llm: Any = None) -> None:
        prepare_runtime_env()
        load_memory_system, LLM, load_split, self._extract = _import_harness()
        if not settings.memory_path.exists():
            raise FileNotFoundError(f"memory file not found: {settings.memory_path}")
        self.settings = settings
        self.llm = llm or LLM(model=settings.model, api_base=settings.api_base, temperature=0.0)
        self.system = load_memory_system(settings.system, self.llm)
        self.system.set_state(settings.memory_path.read_text())
        self._template = _symptom_template(load_split)

    def wrap(self, symptoms: str) -> str:
        return self._template.replace(_SYMPTOM_SLOT, symptoms.strip())

    def _mistake_texts(self) -> set[str]:
        ids = getattr(self.system, "error_ids", [])
        return {self.system._text(self.system.examples[i]) for i in ids if i < len(self.system.examples)}

    def predict(self, symptoms: str) -> Prediction:
        task_input = self.wrap(symptoms)
        prompt = self.system._build_prompt(task_input) if hasattr(self.system, "_build_prompt") else ""
        evidence, warnings = parse_prompt(prompt, self._mistake_texts())
        started = time.perf_counter()
        answer, info = self.system.predict(task_input)
        elapsed = time.perf_counter() - started
        reasoning = self._extract(info.get("full_response", ""), "reasoning")
        return Prediction(answer, reasoning, warnings, evidence, len(prompt), elapsed)

    def stats(self) -> MemoryStats:
        examples = getattr(self.system, "examples", [])
        labels = Counter(e["target"] for e in examples).most_common()
        confusions = getattr(self.system, "confusions", Counter())
        return MemoryStats(
            system=self.settings.system,
            examples=len(examples),
            labels=tuple(labels),
            mistakes=len(getattr(self.system, "error_ids", [])),
            confusions=tuple((t, p, c) for (t, p), c in confusions.most_common()),
        )
