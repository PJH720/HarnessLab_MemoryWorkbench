"""The bundled prompt copy must match the dataset-derived one, so a standalone bundle sends the same prompt."""

import pytest

from harness_cli import engine


def test_bundled_template_matches_the_dataset_prompt_byte_for_byte():
    _, _, load_split, _ = engine._import_harness()
    try:
        load_split("train", limit=1)
    except FileNotFoundError:
        pytest.skip("dataset splits absent (standalone bundle)")
    assert engine.bundled_template() == engine._symptom_template(load_split)


def test_missing_dataset_falls_back_to_the_bundled_template():
    def no_dataset(split, limit=None):
        raise FileNotFoundError(split)

    template = engine._symptom_template(no_dataset)
    assert template == engine.bundled_template()
    assert len(engine.labels_from_template(template)) == 22


def test_other_load_errors_are_not_hidden():
    def broken(split, limit=None):
        raise ValueError("bad row")

    with pytest.raises(ValueError):
        engine._symptom_template(broken)
