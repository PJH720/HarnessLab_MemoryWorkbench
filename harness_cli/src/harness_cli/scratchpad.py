"""Session memory: examples added with /record, layered over the frozen memories (copy-on-write).

Modelled on the Hermes agent's memory tool: the stored memory is a frozen snapshot loaded at
session start, writes go to a separate layer, duplicates and oversized entries are refused with a
readable reason. Given a HarnessHome the layer persists to ~/.harness/scratchpad.json and is
reloaded next session; the frozen memory files under harness_lab/text_classification/logs/ are
research results and are never written.

Retrieval needs no refit: the TF-IDF systems recompute their index over `examples` on every query.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any

from .home import HarnessHome

MAX_RECORDS = 20
MIN_CHARS, MAX_CHARS = 15, 600
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


class ScratchpadError(ValueError):
    """A /record input that was refused; the message is shown to the user as is."""


@dataclass(frozen=True)
class Record:
    symptoms: str
    disease: str


@dataclass(frozen=True)
class PadRow:
    system: str
    frozen: int  # examples in the frozen memory file
    session: int  # examples added this session that this system retrieves from
    affected: bool  # False for systems with no example retrieval (no_memory)


def _clean(text: str) -> str:
    return " ".join(text.split())


class Scratchpad:
    """One session layer shared by every loaded engine, including engines loaded after a /record."""

    def __init__(self, home: HarnessHome | None = None) -> None:
        self._home = home
        self._records: tuple[Record, ...] = self._load()
        self._engines: dict[str, Any] = {}

    @property
    def persistent(self) -> bool:
        return self._home is not None

    def _load(self) -> tuple[Record, ...]:
        if self._home is None:
            return ()
        rows = self._home.records("medical")
        ok = (r for r in rows if isinstance(r.get("symptoms"), str) and isinstance(r.get("disease"), str))
        records = tuple(Record(_clean(r["symptoms"]), r["disease"]) for r in ok)
        return tuple(r for r in records if not _NON_ASCII.search(r.symptoms))[:MAX_RECORDS]

    def _save(self) -> None:
        if self._home is not None:
            self._home.save_records("medical", [{"symptoms": r.symptoms, "disease": r.disease} for r in self._records])

    @property
    def records(self) -> tuple[Record, ...]:
        return self._records

    def attach(self, engine: Any) -> None:
        """Register a freshly loaded engine and give it the session's additions so far."""
        self._engines[engine.settings.system] = engine
        self._apply(engine)

    def normalize_disease(self, disease: str, labels: tuple[str, ...]) -> str:
        key = _clean(disease).lower().strip("'\" ")
        if key in labels:
            return key
        close = difflib.get_close_matches(key, labels, n=1, cutoff=0.6)
        hint = f" 혹시 '{close[0]}'?" if close else ""
        raise ScratchpadError(f"'{disease.strip()}'은(는) 22개 질병 목록에 없습니다.{hint}")

    def validate(self, symptoms: str, disease: str, labels: tuple[str, ...]) -> Record:
        text = _clean(symptoms).strip("'\"")
        if _NON_ASCII.search(text):
            raise ScratchpadError("증상은 영어로 입력해 주세요 (모델 프롬프트는 영어만 씁니다).")
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            raise ScratchpadError(f"증상 문장은 {MIN_CHARS}~{MAX_CHARS}자여야 합니다 (현재 {len(text)}자).")
        if any(r.symptoms.lower() == text.lower() for r in self._records):
            raise ScratchpadError("같은 증상이 이미 세션 기억에 있습니다.")
        if len(self._records) >= MAX_RECORDS:
            raise ScratchpadError(f"세션 기억은 {MAX_RECORDS}건까지입니다. /memory reset 으로 비울 수 있습니다.")
        return Record(text, self.normalize_disease(disease, labels))

    def add(self, symptoms: str, disease: str, labels: tuple[str, ...]) -> Record:
        """Validate, then add one example to every loaded example-based system (and to ~/.harness when persistent)."""
        record = self.validate(symptoms, disease, labels)
        self._records = (*self._records, record)
        self._save()
        for engine in self._engines.values():
            self._apply(engine)
        return record

    def reset(self) -> int:
        """Drop the session layer: every loaded system goes back to its frozen state. Returns records removed."""
        removed = len(self._records)
        self._records = ()
        self._save()
        for engine in self._engines.values():
            self._apply(engine)
        return removed

    def status(self) -> tuple[PadRow, ...]:
        return tuple(
            PadRow(name, self._frozen_count(e), len(e.session_examples), e.uses_examples)
            for name, e in self._engines.items()
        )

    @staticmethod
    def _frozen_count(engine: Any) -> int:
        return len(json.loads(engine.frozen_state or "{}").get("examples", []))

    def _apply(self, engine: Any) -> None:
        """Rebuild the system from its frozen text, then append the session examples to a new list."""
        engine.system.set_state(engine.frozen_state)
        if not engine.uses_examples:
            engine.session_examples = ()
            return
        added = tuple(
            {"input": engine.wrap(r.symptoms), "raw_question": r.symptoms, "target": r.disease}
            for r in self._records
        )
        engine.system.examples = [*engine.system.examples, *added]
        engine.session_examples = added
