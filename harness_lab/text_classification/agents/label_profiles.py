"""LLM-distilled per-label profiles + nearest-neighbour demonstrations."""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import extract_json_field
from .tfidf_cover import TfidfCover

PROFILE_PROMPT = """Below are example inputs that all belong to the category "{label}".
Write ONE concise line (max 40 words) listing the distinguishing features shared by these inputs
that would help tell this category apart from similar ones.

{examples}

Respond in JSON: {{"final_answer": "[the line]"}}"""

PREDICT_PROMPT = """Classify the problem below. First read the category profiles distilled from labelled data, then the most similar solved examples.

**Category profiles:**
{profiles}

**Similar solved examples:**
{examples_section}

**Problem:**
{input}

**Instructions:**
- Compare the problem to the profiles and examples; choose the best-matching category
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class LabelProfiles(TfidfCover):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm, k=5)
        self.profiles: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def _refresh(self) -> None:
        """Lazily (re)distill profiles for labels whose example set changed."""
        with self._lock:
            by_label: dict[str, list[str]] = {}
            for e in self.examples:
                by_label.setdefault(e["target"], []).append(self._text(e))
            for lab, texts in by_label.items():
                if lab in self.profiles and self._counts.get(lab) == len(texts):
                    continue
                joined = "\n".join(f"- {t[:400]}" for t in texts[:6])
                resp = self.call_llm(PROFILE_PROMPT.format(label=lab, examples=joined))
                self.profiles[lab] = extract_json_field(resp, "final_answer")
                self._counts[lab] = len(texts)

    def _build_prompt(self, input: str) -> str:
        if not self.examples:
            return super()._build_prompt(input)
        self._refresh()
        profiles = "\n".join(f"- {l}: {p}" for l, p in sorted(self.profiles.items()) if p)
        top = [i for _, i in self._rank(input)[: self.k]][::-1]
        return PREDICT_PROMPT.format(profiles=profiles, examples_section=self._fmt(top), input=input)

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples, "profiles": self.profiles, "counts": self._counts})

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = d.get("examples", [])
        self.profiles = d.get("profiles", {})
        self._counts = d.get("counts", {})
