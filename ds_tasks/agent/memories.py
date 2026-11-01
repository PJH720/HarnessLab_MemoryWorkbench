"""Harness memories. Each exposes context(card) for injection and learn(...) after an episode.

Counters are part of the state so a run can be audited: every training failure must be
accounted for as a stored item, a rejected item, or a parse failure.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable

from .loop import Episode, run_episode

TOKEN_RE = re.compile(r"[a-z]{3,}")


def rank_texts(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """TF-IDF cosine ranking (same weighting as the text-classification harness)."""
    if not texts:
        return []
    docs = [Counter(TOKEN_RE.findall(t.lower())) for t in texts]
    df = Counter(t for d in docs for t in d)
    n = len(docs) + 1
    idf = {t: math.log(n / (1 + c)) + 1 for t, c in df.items()}

    def vec(c):
        v = {t: (1 + math.log(f)) * idf.get(t, 0) for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    q = vec(Counter(TOKEN_RE.findall(query.lower())))
    return sorted(((sum(w * vec(d).get(t, 0) for t, w in q.items()), i) for i, d in enumerate(docs)), reverse=True)


class NoMemory:
    name = "ds_base"

    def context(self, card: dict) -> tuple[str, list[int]]:
        return "", []

    def learn(self, card: dict, ep: Episode, correct: bool, llm, seed: int, scorer) -> None:
        return None

    def state(self) -> dict:
        return {}


class KnnSuccess:
    """Keeps (task, working code) only from successful episodes; shows the most similar ones."""

    name = "ds_knn"
    top_k = 2

    def __init__(self):
        self.items: list[dict] = []
        self.n_success_seen = 0

    def context(self, card):
        ranked = [(s, i) for s, i in rank_texts(card["question"], [it["question"] for it in self.items])[: self.top_k] if s > 0]
        blocks = [f"Similar task solved before:\n{self.items[i]['question'][-400:]}\nCode that worked:\n```python\n{self.items[i]['code']}\n```" for _, i in ranked]
        return ("Reference solutions from similar past tasks (adapt, do not copy numbers):\n\n" + "\n\n".join(blocks)) if blocks else "", [i for _, i in ranked]

    def learn(self, card, ep, correct, llm, seed, scorer):
        if correct and ep.last_code:
            self.n_success_seen += 1
            self.items.append({"card": card["id"], "question": card["question"], "code": ep.last_code})

    def state(self):
        return {"items": self.items, "n_success_seen": self.n_success_seen, "n_items": len(self.items)}


CUE_PROMPT = """A data-analysis attempt failed. Write ONE reusable lesson for future, different tasks.

Task: {question}
The analyst's code:
```python
{code}
```
Its output (tail): {output}
Its answer: {answer}
Expected answer: {expected}

Think about which step went wrong (data loading, join, missing values, filtering, the statistical test, the model setup, or answer formatting).
The lesson must be general: do not include the expected answer or any number from it.

Respond in JSON:
{{"title": "[short name]", "when": "[which kind of task this applies to]", "cue": "[one sentence instruction]"}}"""


def _json_field(text: str, key: str) -> str:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return str(json.loads(match.group(0)).get(key, "")).strip() if match else ""
    except json.JSONDecodeError:
        return ""


class CueVerified:
    """Distill each failure into a cue; store it only if re-running that task with the cue succeeds."""

    name = "ds_cue_verified"
    top_k = 3
    min_score = 0.1

    def __init__(self):
        self.items: list[dict] = []
        self.n_failures_seen = self.n_rejected = self.n_parse_failures = 0

    def _render(self, item):
        return f"- [{item['title']}] {item['when']}: {item['cue']}"

    def context(self, card):
        ranked = [(s, i) for s, i in rank_texts(card["question"], [it["question"] for it in self.items])[: self.top_k] if s >= self.min_score]
        lines = [self._render(self.items[i]) for _, i in ranked]
        return ("Verified lessons from past mistakes:\n" + "\n".join(lines)) if lines else "", [i for _, i in ranked]

    def learn(self, card, ep, correct, llm, seed, scorer):
        if correct:
            return
        self.n_failures_seen += 1
        last = ep.steps[-1] if ep.steps else {"stdout": "", "stderr": ""}
        prompt = CUE_PROMPT.format(question=card["question"], code=ep.last_code[-1500:] or "(no code)",
                                   output=(last["stdout"] + last["stderr"])[-600:], answer=ep.answer, expected=card["answer"])
        text, _ = llm([{"role": "user", "content": prompt}], seed)
        item = {k: _json_field(text, k) for k in ("title", "when", "cue")}
        if not item["cue"]:
            self.n_parse_failures += 1
            return
        retry = run_episode(card, llm, seed, memory_text="Verified lessons from past mistakes:\n" + self._render(item))
        if not scorer(card, retry.answer):
            self.n_rejected += 1
            return
        self.items.append({"card": card["id"], "question": card["question"], **item})

    def state(self):
        return {"items": self.items, "n_failures_seen": self.n_failures_seen, "n_items": len(self.items),
                "n_rejected": self.n_rejected, "n_parse_failures": self.n_parse_failures}


HARNESSES: dict[str, Callable] = {"ds_base": NoMemory, "ds_knn": KnnSuccess, "ds_cue_verified": CueVerified}
