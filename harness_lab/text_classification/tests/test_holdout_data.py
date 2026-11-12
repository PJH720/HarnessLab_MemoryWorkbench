"""Confirmatory holdout: same prompt as Symptom2Disease, disjoint test texts."""

import unittest
from pathlib import Path

from text_classification.data import MCE_DATA_SPLITS, get_evaluator, load_dataset_splits_3way
from text_classification.data.loaders import load_mce_dataset

DATA = Path(__file__).resolve().parents[1] / "data"
HAVE_DATA = (DATA / "symptom_holdout" / "test.jsonl").exists() and (
    DATA / "symptom_diagnosis" / "test.jsonl"
).exists()


@unittest.skipUnless(HAVE_DATA, "holdout or vendored Symptom2Disease data not present")
class HoldoutDataTest(unittest.TestCase):
    def test_sizes_match_constants(self) -> None:
        for split, count in MCE_DATA_SPLITS["Symptom2DiseaseHoldout"].items():
            with self.subTest(split=split):
                self.assertEqual(len(load_mce_dataset("Symptom2DiseaseHoldout", split)), count)

    def test_train_and_val_are_the_vendored_splits(self) -> None:
        for split in ("train", "val"):
            with self.subTest(split=split):
                self.assertEqual(
                    load_mce_dataset("Symptom2DiseaseHoldout", split),
                    load_mce_dataset("Symptom2Disease", split),
                )

    def test_test_texts_are_disjoint_and_prompt_is_unchanged(self) -> None:
        holdout = load_mce_dataset("Symptom2DiseaseHoldout", "test")
        vendored = [
            ex for split in ("train", "val", "test") for ex in load_mce_dataset("Symptom2Disease", split)
        ]
        self.assertFalse({ex["raw_question"] for ex in holdout} & {ex["raw_question"] for ex in vendored})
        prefix = vendored[0]["input"].split("## Patient Symptoms")[0]
        self.assertTrue(all(ex["input"].startswith(prefix) for ex in holdout))

    def test_split_loader_and_evaluator(self) -> None:
        _, _, test, evaluator = load_dataset_splits_3way("Symptom2DiseaseHoldout", 60, 50, 572, 42)
        self.assertEqual(len(test), 572)
        self.assertIs(type(evaluator), type(get_evaluator("Symptom2Disease")))


if __name__ == "__main__":
    unittest.main()
