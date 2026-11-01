"""Retrieval few-shot: TF-IDF nearest neighbours plus one exemplar per remaining label."""

import json
import math
import re
import threading
from collections import Counter
from typing import Any

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field

PROMPT_TEMPLATE = """Solve the problem below based on the examples provided.

{examples_section}

**Problem:**
{input}

**Instructions:**
- Follow the patterns shown in the examples above
- Respond in JSON format

{{"reasoning": "[your reasoning]", "final_answer": "[your answer]"}}"""


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


class TfidfCover(MemorySystem):
    """Top-k similar examples (most similar placed last) + label-coverage exemplars."""

    def __init__(self, llm: LLMCallable, k: int = 8):
        super().__init__(llm)
        self.k = k
        self.examples: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _text(self, ex: dict[str, Any]) -> str:
        return ex.get("raw_question", ex["input"])

    def _vecs(self, query: str):
        docs = [Counter(tokens(self._text(e))) for e in self.examples]
        df = Counter(t for d in docs for t in d)
        n = len(docs) + 1
        idf = {t: math.log(n / (1 + c)) + 1 for t, c in df.items()}

        def vec(c):
            v = {t: (1 + math.log(f)) * idf.get(t, 0) for t, f in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {t: x / norm for t, x in v.items()}

        return [vec(d) for d in docs], vec(Counter(tokens(query)))

    def _rank(self, query: str) -> list[tuple[float, int]]:
        if not self.examples:
            return []
        dvecs, q = self._vecs(query)
        scored = [(sum(w * d.get(t, 0) for t, w in q.items()), i) for i, d in enumerate(dvecs)]
        return sorted(scored, reverse=True)

    def _select(self, query: str) -> list[int]:
        ranked = self._rank(query)
        top = [i for _, i in ranked[: self.k]]
        seen = {self.examples[i]["target"] for i in top}
        cover = []
        for _, i in ranked:
            lab = self.examples[i]["target"]
            if lab not in seen:
                seen.add(lab)
                cover.append(i)
        return cover[::-1] + top[::-1]  # coverage first, nearest last

    def _fmt(self, idxs: list[int]) -> str:
        return "\n\n".join(f"Q: {self._text(self.examples[i])}\nA: {self.examples[i]['target']}" for i in idxs)

    def _build_prompt(self, input: str) -> str:
        return PROMPT_TEMPLATE.format(examples_section=self._fmt(self._select(input)), input=input)

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        response = self.call_llm(self._build_prompt(input))
        return extract_json_field(response, "final_answer"), {
            "full_response": response,
            "num_examples": len(self.examples),
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            ex = {"input": r["input"], "target": r["ground_truth"]}
            if "raw_question" in r:
                ex["raw_question"] = r["raw_question"]
            self.examples.append(ex)

    def get_context_length(self) -> int:
        return len(self._fmt(self._select(self.examples[-1]["input"]))) if self.examples else 0

    def get_state(self) -> str:
        return json.dumps({"examples": self.examples})

    def set_state(self, state: str) -> None:
        self.examples = json.loads(state).get("examples", [])
