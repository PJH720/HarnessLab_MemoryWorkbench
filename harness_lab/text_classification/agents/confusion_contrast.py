"""Error-driven contrastive retrieval: neighbours, hard past mistakes, and confusion warnings."""

import json
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from .tfidf_cover import TfidfCover

PREDICT_PROMPT = """Classify the problem below using the solved examples.

**Similar solved examples:**
{examples_section}

**Known confusions (a reader often mislabels these):**
{confusions}

**Problem:**
{input}

**Instructions:**
- Pay attention to the confusions above; check distinguishing details before answering
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


class ConfusionContrast(TfidfCover):
    def __init__(self, llm: LLMCallable):
        super().__init__(llm, k=6)
        self.confusions: Counter = Counter()  # (true, predicted) -> count
        self.error_ids: list[int] = []

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            if r.get("was_correct") is False:
                self.confusions[(r["ground_truth"], str(r.get("prediction", "")).strip().lower())] += 1
                self.error_ids.append(len(self.examples))
        super().learn_from_batch(batch_results)

    def _build_prompt(self, input: str) -> str:
        if not self.examples:
            return super()._build_prompt(input)
        ranked = self._rank(input)
        errs = set(self.error_ids)
        top = [i for _, i in ranked[: self.k]]
        hard = [i for _, i in ranked if i in errs][:3]
        chosen = list(dict.fromkeys(hard + top))[::-1]
        labels = {self.examples[i]["target"] for i in chosen}
        lines = [
            f"- '{t}' was mistaken for '{p}' ({c}x)"
            for (t, p), c in self.confusions.most_common()
            if t in labels or p in labels
        ][:6]
        return PREDICT_PROMPT.format(
            examples_section=self._fmt(chosen), confusions="\n".join(lines) or "- none recorded", input=input
        )

    def get_state(self) -> str:
        return json.dumps(
            {
                "examples": self.examples,
                "confusions": [[t, p, c] for (t, p), c in self.confusions.items()],
                "error_ids": self.error_ids,
            }
        )

    def set_state(self, state: str) -> None:
        d = json.loads(state)
        self.examples = d.get("examples", [])
        self.confusions = Counter({(t, p): c for t, p, c in d.get("confusions", [])})
        self.error_ids = d.get("error_ids", [])
