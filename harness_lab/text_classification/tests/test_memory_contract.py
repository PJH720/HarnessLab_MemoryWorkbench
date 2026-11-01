"""Memory contract: error-driven memories must actually store the model's real mistakes.

Regression for the midterm run, where offline training passed ground truth as the
prediction (was_correct=True) and confusion_contrast recorded nothing.
"""

from __future__ import annotations

import json
import unittest

from text_classification.agents.contrast_online import ContrastOnline
from text_classification.agents.cue_bank import CueBank
from text_classification.agents.knn_top6 import KnnTop6
from text_classification.agents.reflexion_notes import ReflexionNotes
from text_classification.inner_loop import run_inner_loop

ALWAYS_WRONG = json.dumps({"reasoning": "fever", "final_answer": "malaria", "title": "t", "when": "w", "cue": "c"})
EXAMPLES = [
    {"input": f"Patient Symptoms: case {i} with rash and itching", "target": t, "raw_question": f"case {i} with rash and itching"}
    for i, t in enumerate(["psoriasis", "impetigo", "fungal infection", "chicken pox"])
]


def check(pred: str, target: str, **_) -> bool:
    return pred.strip().lower() == target.strip().lower()


def train(cls, llm_response: str, mode: str):
    memory = cls(lambda _prompt: llm_response)
    run_inner_loop(memory, EXAMPLES, check, mode=mode, num_epochs=1, skip_train_eval=True)
    return memory, json.loads(memory.get_state())


class MemoryContractTests(unittest.TestCase):
    def test_online_records_every_real_mistake(self):
        _, contrast = train(ContrastOnline, ALWAYS_WRONG, "online")
        self.assertEqual(contrast["n_errors_seen"], 4)
        self.assertEqual(len(contrast["error_ids"]), 4)
        self.assertEqual(sum(c for _, _, c in contrast["confusions"]), 4)
        for cls in (CueBank, ReflexionNotes):
            _, state = train(cls, ALWAYS_WRONG, "online")
            self.assertEqual(state["n_errors_seen"], 4, cls.__name__)
            self.assertEqual(state["n_items"], 4, cls.__name__)
            self.assertEqual(len(state["items"]), 4, cls.__name__)

    def test_offline_mode_never_shows_mistakes(self):
        for cls in (ContrastOnline, CueBank, ReflexionNotes):
            _, state = train(cls, ALWAYS_WRONG, "offline")
            self.assertEqual(state["n_errors_seen"], 0, cls.__name__)
            self.assertEqual(state["n_items"], 0, cls.__name__)

    def test_knn_baseline_stores_examples_only(self):
        _, state = train(KnnTop6, ALWAYS_WRONG, "online")
        self.assertEqual(len(state["examples"]), 4)
        self.assertNotIn("items", state)

    def test_unparseable_lesson_counts_as_parse_failure(self):
        _, state = train(CueBank, "not json at all", "online")
        self.assertEqual(state["n_errors_seen"], 4)
        self.assertEqual(state["n_items"], 0)
        self.assertEqual(state["n_parse_failures"], 4)

    def test_state_round_trip_keeps_counts(self):
        memory, state = train(CueBank, ALWAYS_WRONG, "online")
        restored = CueBank(lambda _p: ALWAYS_WRONG)
        restored.set_state(memory.get_state())
        self.assertEqual(json.loads(restored.get_state()), state)

    def test_lessons_are_injected_at_prediction(self):
        memory, _ = train(CueBank, ALWAYS_WRONG, "online")
        prompt = memory._build_prompt("Patient Symptoms: rash and itching")
        self.assertIn("[t] When: w -> c", prompt)


if __name__ == "__main__":
    unittest.main()


class LoaderContractTests(unittest.TestCase):
    """load_memory_system returns the first MemorySystem subclass in a module (alphabetical),
    so a candidate that imports its base class by name can silently load the base instead."""

    def test_loader_picks_the_candidate_class(self):
        from text_classification.inner_loop import load_memory_system

        expected = {"knn_top6": KnnTop6, "contrast_online": ContrastOnline, "cue_bank": CueBank, "reflexion_notes": ReflexionNotes}
        for name, cls in expected.items():
            self.assertIs(type(load_memory_system(name, lambda _p: "{}")), cls, name)


def verified_stub(verify_passes: bool):
    """Cue prompts get a cue naming the correct label; a prompt carrying that cue answers it
    (if verify_passes); every other prompt answers 'malaria' so training is always wrong."""

    def llm(prompt: str) -> str:
        if "Write ONE reusable cue" in prompt:
            label = prompt.split('Correct answer: ')[1].split("\n")[0]
            return json.dumps({"title": "t", "when": "w", "cue": f"points to {label}"})
        if verify_passes and "points to " in prompt:
            label = prompt.split("points to ")[1].split("\n")[0].strip()
            return json.dumps({"reasoning": "cue", "final_answer": label})
        return json.dumps({"reasoning": "r", "final_answer": "malaria"})

    return llm


class VerifiedCueContractTests(unittest.TestCase):
    def _train(self, verify_passes: bool, examples=EXAMPLES):
        from text_classification.agents.cue_bank_verified import CueBankVerified

        memory = CueBankVerified(verified_stub(verify_passes))
        run_inner_loop(memory, examples, check, mode="online", num_epochs=1, skip_train_eval=True)
        return json.loads(memory.get_state())

    def test_cue_stored_only_when_it_fixes_its_source_case(self):
        state = self._train(verify_passes=True)
        self.assertEqual((state["n_errors_seen"], state["n_items"], state["n_rejected"]), (4, 4, 0))
        state = self._train(verify_passes=False)
        self.assertEqual((state["n_errors_seen"], state["n_items"], state["n_rejected"]), (4, 0, 4))

    def test_duplicate_pair_is_not_stored_twice(self):
        # no word overlap with stored cues, so the model repeats the same (psoriasis <- malaria) mistake
        dup = EXAMPLES + [{"input": "zzz qqq", "target": "psoriasis", "raw_question": "zzz qqq"}]
        state = self._train(verify_passes=True, examples=dup)
        self.assertEqual((state["n_errors_seen"], state["n_items"], state["n_duplicates"]), (5, 4, 1))

    def test_cue_never_names_the_wrong_disease_in_prompt(self):
        from text_classification.agents.cue_bank_verified import CUE_PROMPT

        self.assertNotIn("{wrong}", CUE_PROMPT)
