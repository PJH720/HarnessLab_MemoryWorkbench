"""Persistent user state under ~/.harness (HARNESS_HOME): a study profile and the /record scratchpad.

Hermes-style three tiers: the frozen research memories (read only, under harness_lab/.../logs),
this persistent working layer, and the per-session state held by the engines. Nothing here may
ever point into the frozen logs, and no API key is ever stored.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from . import config

PROFILE_KEYS = ("name", "target_exam", "topics", "style")
PROFILE_HELP = {
    "name": "how the CLI addresses you",
    "target_exam": "exam or course you are preparing for",
    "topics": "topics to focus on, comma separated",
    "style": "how answers should be explained (e.g. 'short bullet points with one example')",
}
MAX_PROFILE_CHARS = 200
TASKS = ("medical", "study")
# Prompts to the model are English only, so Korean/CJK text is refused before it reaches one.
_CJK = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힣豈-﫿]")


class HomeError(ValueError):
    """A refused write; the message is shown to the user as is."""


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


class HarnessHome:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or config.harness_home()).expanduser().resolve()
        if _inside(self.root, config.LOGS):
            raise HomeError(f"HARNESS_HOME may not point into the frozen research logs: {self.root}")

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.json"

    @property
    def scratchpad_path(self) -> Path:
        return self.root / "scratchpad.json"

    # --- profile ----------------------------------------------------------------------------
    def profile(self) -> dict[str, str]:
        data = self._read(self.profile_path)
        return {k: str(data[k]) for k in PROFILE_KEYS if data.get(k)}

    def set_profile(self, key: str, value: str) -> dict[str, str]:
        key = key.strip().lower()
        value = " ".join(value.split()).strip("'\"")
        if key not in PROFILE_KEYS:
            raise HomeError(f"unknown profile field '{key}'. fields: {', '.join(PROFILE_KEYS)}")
        if has_cjk(value):
            raise HomeError("프로필 값은 영어로 입력해 주세요 (모델 프롬프트는 영어만 씁니다).")
        if len(value) > MAX_PROFILE_CHARS:
            raise HomeError(f"profile values are at most {MAX_PROFILE_CHARS} characters")
        profile = {**self.profile(), key: value}
        if not value:
            profile.pop(key)
        self._write(self.profile_path, profile)
        return profile

    # --- scratchpad -------------------------------------------------------------------------
    def records(self, task: str) -> list[dict[str, str]]:
        rows = self._read(self.scratchpad_path).get(task, [])
        return [r for r in rows if isinstance(r, dict)]

    def save_records(self, task: str, rows: list[dict[str, str]]) -> None:
        if task not in TASKS:
            raise HomeError(f"unknown task: {task}")
        data = {**self._read(self.scratchpad_path), task: rows}
        self._write(self.scratchpad_path, data)

    # --- endpoint config (harness-cli config) -----------------------------------------------
    def save_config(self, api_base: str = "", model: str = "") -> dict[str, str]:
        """Merge api_base / model into config.json; an empty value leaves the field as it was. No keys."""
        api_base, model = api_base.strip().rstrip("/"), model.strip()
        if api_base and not api_base.startswith(("http://", "https://")):
            raise HomeError(f"--api-base must start with http:// or https:// (got '{api_base}')")
        if has_cjk(model) or " " in model:
            raise HomeError(f"invalid model id: '{model}'")
        data = {**config.user_config(), **{k: v for k, v in (("api_base", api_base), ("model", model)) if v}}
        self._write(self.root / "config.json", data)
        return data

    def clear_config(self) -> None:
        (self.root / "config.json").unlink(missing_ok=True)

    # --- io ---------------------------------------------------------------------------------
    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        """Atomic, so an interrupted save never leaves a truncated file behind."""
        if _inside(path, config.LOGS):
            raise HomeError("refusing to write into the frozen research logs")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
