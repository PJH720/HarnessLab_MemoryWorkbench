"""cue_bank with a verification gate, after diagnosing why plain cue_bank lost 12%p.

Diagnosis (logs/online-v2): plain cues were textbook rules that sometimes contradicted
this dataset's labels (a cue for an asthma case argued for pneumonia), named the wrong
disease (priming it), and were retrieved by loose word overlap.

Changes:
- the cue argues only for the correct disease and must not name the wrong one;
- a cue is stored only if re-running its own source case with the cue now yields the
  correct answer (verify-before-store, as ReasoningBank/Honest-Lying motivate);
- at most one cue per (correct, wrong) pair; retrieval needs a minimum similarity.
"""

import json
from typing import Any

from ..memory_system import extract_json_field
from . import _error_memory

CUE_PROMPT = """You are reviewing a mistake made by a symptom-to-disease classifier on this dataset.

Patient symptoms: {symptoms}
Correct answer: {right}
Classifier's reasoning at the time: {reasoning}

Write ONE reusable cue: which symptom pattern in cases like this points to "{right}".
Base it on the symptoms shown here, not on general textbook expectations.
Do not mention any other disease. Do not copy the patient's sentence.

Respond in JSON:
{{"title": "[short name]", "when": "[when this cue applies]", "cue": "[one sentence: the symptom pattern that points to {right}]"}}"""


def _same_label(prediction: str, target: str) -> bool:
    return prediction.strip().lower() == target.strip().lower()


class CueBankVerified(_error_memory.ErrorMemory):
    lessons_header = "Verified diagnostic cues from similar past cases:"
    min_score = 0.15

    def __init__(self, llm, k: int = 6):
        super().__init__(llm, k=k)
        self.n_rejected = 0
        self.n_duplicates = 0

    def _render(self, item: dict[str, str]) -> str:
        return f"- [{item['title']}] When: {item['when']} -> {item['cue']}"

    def _prompt_with(self, input: str, lessons: list[str]) -> str:
        nearest = [i for _, i in self._rank(input)[: self.k]][::-1]
        return _error_memory.PREDICT_TEMPLATE.format(
            examples_section=self._fmt(nearest), lessons_header=self.lessons_header,
            lessons="\n".join(lessons) or "- none recorded", input=input,
        )

    def _build_prompt(self, input: str) -> str:
        ranked = _error_memory.rank_texts(input, [it["symptoms"] for it in self.items])
        lessons = [self._render(self.items[i]) for score, i in ranked[: self.top_items] if score >= self.min_score]
        return self._prompt_with(input, lessons)

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        for r in batch_results:
            if r.get("was_correct") is not False:
                continue
            self.n_errors_seen += 1
            right, wrong = r["ground_truth"], str(r.get("prediction", ""))
            if any(it["right"] == right and it["wrong"].lower() == wrong.lower() for it in self.items):
                self.n_duplicates += 1
                continue
            symptoms = r.get("raw_question", r["input"])
            reasoning = extract_json_field((r.get("metadata") or {}).get("full_response", ""), "reasoning")[:600]
            response = self.call_llm(CUE_PROMPT.format(symptoms=symptoms, right=right, reasoning=reasoning))
            item = {k: extract_json_field(response, k).strip() for k in ("title", "when", "cue")}
            if not item["cue"]:
                self.n_parse_failures += 1
                continue
            check = self.call_llm(self._prompt_with(r["input"], [self._render(item)]))
            if not _same_label(extract_json_field(check, "final_answer"), right):
                self.n_rejected += 1
                continue
            self.items.append({"symptoms": symptoms, "wrong": wrong, "right": right, **item})
        _error_memory.tfidf_cover.TfidfCover.learn_from_batch(self, batch_results)

    def get_state(self) -> str:
        state = json.loads(super().get_state())
        state.update(n_rejected=self.n_rejected, n_duplicates=self.n_duplicates)
        return json.dumps(state)

    def set_state(self, state: str) -> None:
        super().set_state(state)
        d = json.loads(state)
        self.n_rejected = d.get("n_rejected", 0)
        self.n_duplicates = d.get("n_duplicates", 0)
