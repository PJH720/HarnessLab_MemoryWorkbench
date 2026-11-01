"""Runtime settings, resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HARNESS_LAB = PROJECT_ROOT / "harness_lab"
DEFAULT_MEMORY = (
    HARNESS_LAB
    / "text_classification/logs/midterm/Symptom2Disease/confusion_contrast"
    / "qwen3.8-flash-next-nvfp4/memory.json"
)


@dataclass(frozen=True)
class Settings:
    system: str
    memory_path: Path
    model: str
    api_base: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            system=os.environ.get("HARNESS_SYSTEM", "confusion_contrast"),
            memory_path=Path(os.environ.get("HARNESS_MEMORY", str(DEFAULT_MEMORY))).expanduser(),
            model=os.environ.get("HARNESS_MODEL", "openai/Inferact/Qwen3.8-Flash-Next-NVFP4"),
            api_base=os.environ.get("HARNESS_API_BASE", "https://model-endpoint.example.invalid/v1"),
        )


def prepare_runtime_env() -> None:
    """Match the evaluated run: reasoning off, and a placeholder key (litellm requires one; SGLang ignores it)."""
    os.environ.setdefault("TC_DISABLE_THINKING", "1")
    os.environ.setdefault("OPENAI_API_KEY", "onprem-placeholder")
