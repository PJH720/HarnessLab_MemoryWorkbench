"""Demo presets and the logged numbers the demo screens quote.

Every number here is copied from a run log; the comment above each names the file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    contested: tuple[str, ...]  # the two labels this case sits between (both are in the 22-class set)
    symptoms: str


PRESETS: tuple[Preset, ...] = (
    Preset(
        "1",
        "Asthma vs Pneumonia (천식/폐렴 경계)",
        ("bronchial asthma", "pneumonia"),
        "I have been coughing continuously with shortness of breath and wheezing sounds when exhaling, "
        "accompanied by chest tightness.",
    ),
    Preset(
        "2",
        "Malaria vs Dengue (열대성 고열 질환)",
        ("malaria", "dengue"),
        "Experiencing sudden high fever, shaking chills, severe joint pain, and profuse sweating every two days "
        "after traveling.",
    ),
    Preset(
        "3",
        "Typhoid Fever (지속 고열과 위장 증상)",
        ("typhoid", "malaria"),
        "High fever that has been rising progressively over a week, severe abdominal pain, persistent headache, "
        "and extreme lethargy.",
    ),
)


def find_preset(key: str) -> Preset | None:
    return next((p for p in PRESETS if p.key == key.strip()), None)


@dataclass(frozen=True)
class Logged:
    system: str
    val_acc: float  # n=50
    test_acc: float  # n=212
    val_input_tokens: int


# harness_lab/text_classification/logs/midterm/Symptom2Disease/<system>/qwen3.8-flash-next-nvfp4/{val,test}.json
LOGGED = {
    "no_memory": Logged("no_memory", 0.64, 0.6840, 8_077),
    "confusion_contrast": Logged("confusion_contrast", 0.82, 0.8066, 28_745),
    "fewshot_all": Logged("fewshot_all", 0.82, 0.8491, 144_224),
}
VAL_N, TEST_N = 50, 212


def token_reduction_vs_fewshot_all() -> float:
    """Input-token saving of the harness against showing all 60 examples (val run)."""
    full, harness = LOGGED["fewshot_all"].val_input_tokens, LOGGED["confusion_contrast"].val_input_tokens
    return 1 - harness / full
