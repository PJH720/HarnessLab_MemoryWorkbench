"""Base for memories that turn each real training mistake into a retrievable text item.

Subclasses define how a mistake becomes an item (_induce) and how items render (_render).
Counters (n_errors_seen, n_items, n_parse_failures) are saved so runs can be audited.
"""

import json
from typing import Any

from ..llm import LLMCallable
from ..memory_system import extract_json_field
from ._text_rank import rank_texts
from . import tfidf_cover

PREDICT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**{lessons_header}**
{lessons}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Check the lessons from past mistakes before answering
- Respond in JSON format
{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class ErrorMemory(tfidf_cover.TfidfCover):
    lessons_header = "Lessons from past mistakes on similar cases:"
    top_items = 3

    def __init__(self, llm: LLMCallable, k: int = 6):
        super().__init__(llm, k=k)
        self.items: list[dict[str, str]] = []
        self.n_errors_seen = 0
        self.n_parse_failures = 0

    def _induce(self, symptoms: str, wrong: str, right: str, reasoning: str) -> dict[str, str] | None:
        raise NotImplementedError

    def _render(self, item: dict[str, str]) -> str:
        raise NotImplementedError

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            if r.get("was_correct") is not False:
                continue
            self.n_errors_seen += 1
            response = (r.get("metadata") or {}).get("full_response", "")
            reasoning = extract_json_field(response, "reasoning")[:600]
            symptoms = r.get("raw_question", r["input"])
            item = self._induce(symptoms, str(r.get("prediction", "")), r["ground_truth"], reasoning)
            if item is None:
                self.n_parse_failures += 1
            else:
                self.items.append({"symptoms": symptoms, "wrong": str(r.get("prediction", "")),
                                   "right": r["ground_truth"], **item})
        super().learn_from_batch(batch_results)

    def _build_prompt(self, input: str) -> str:
        nearest = [i for _, i in self._rank(input)[: self.k]][::-1]
        ranked = rank_texts(input, [it["symptoms"] for it in self.items])
        lessons = [self._render(self.items[i]) for score, i in ranked[: self.top_items] if score > 0]
        return PREDICT_TEMPLATE.format(
            examples_section=self._fmt(nearest),
            lessons_header=self.lessons_header,
            lessons="\n".join(lessons) or "- none recorded",
            input=input,
        )

    def get_state(self) -> str:
        return json.dumps({
            "examples": self.examples,
            "items": self.items,
            "n_errors_seen": self.n_errors_seen,
            "n_items": len(self.items),
            "n_parse_failures": self.n_parse_failures,
        })

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = d.get("examples", [])
        self.items = d.get("items", [])
        self.n_errors_seen = d.get("n_errors_seen", 0)
        self.n_parse_failures = d.get("n_parse_failures", 0)
