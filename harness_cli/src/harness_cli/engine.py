"""Load the evaluated memory systems and expose predictions together with their evidence.

The prompt, retrieval and injected notes all come from the harness code that was scored in the
research runs; this module only reads them back for display. Frozen memories are loaded from
disk once and never written back: session additions live in memory only (see scratchpad.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from . import config, lego
from .config import Settings, prepare_runtime_env
from .scratchpad import Scratchpad

REPLAY_PATH = Path(__file__).with_name("replay.json")

_SYMPTOM_SLOT = "\u0000SYMPTOMS\u0000"
BUNDLED_TEMPLATE_PATH = Path(__file__).with_name("symptom_prompt.txt")
_BUNDLED_SLOT = "{{SYMPTOMS}}"
_PROBLEM = "**Problem:**"
_HEADER_RE = re.compile(r"^\*\*(?P<title>[^*\n]+)\*\*[ \t]*$", re.M)
_QA_RE = re.compile(r"Q: (?P<q>.*?)\nA: (?P<a>[^\n]*)", re.S)
_LABELS_RE = re.compile(r"Possible diagnoses include: (?P<labels>[^\n]+?)\.?\n")
_EXAMPLES_TITLE = "Similar solved examples:"


@dataclass(frozen=True)
class Evidence:
    question: str
    label: str
    past_mistake: bool
    session: bool = False  # added during this session with /record


@dataclass(frozen=True)
class Sections:
    """What a built prompt injects besides the patient's symptoms."""

    evidence: tuple[Evidence, ...]
    notes: tuple[str, ...]  # confusion warnings, cues or reflection notes, one per line
    notes_title: str  # the prompt's own header for `notes`, e.g. "Known confusions (...)"


@dataclass(frozen=True)
class Prediction:
    diagnosis: str
    reasoning: str
    warnings: tuple[str, ...]  # confusion warnings (confusion_contrast, contrast_online)
    evidence: tuple[Evidence, ...]
    prompt_chars: int
    seconds: float
    system: str = ""
    source: str = "live"  # "live" | "replay" | "preview" (retrieval only, no model answer)
    response: str = ""  # raw model output, kept so `harness-cli record` can store it
    notes: tuple[str, ...] = ()  # every injected note line, whatever the system calls them
    notes_title: str = ""
    prompt_tokens: int | None = None  # as counted by the server; None when unknown (cache hit, preview)
    prompt_sha: str = ""
    prompt: str = field(default="", repr=False)  # the exact text sent (or, for a preview, that would be sent)


@dataclass(frozen=True)
class Pick:
    question: str
    label: str
    score: float | None  # TF-IDF cosine against the query; None when the system does not rank (fewshot_all)
    reason: str  # "top-k similar" | "past mistake" | "label coverage" | "all"


@dataclass(frozen=True)
class MemoryStats:
    system: str
    examples: int
    labels: tuple[tuple[str, int], ...]
    mistakes: int
    confusions: tuple[tuple[str, str, int], ...]
    items: int = 0  # cues / reflection notes stored by error-memory systems
    session_examples: int = 0


def _import_harness() -> tuple[Callable, Callable, Callable, Callable]:
    lab = str(config.HARNESS_LAB)
    if lab not in sys.path:
        sys.path.insert(0, lab)
    from text_classification.data.loaders import _load_symptom2disease
    from text_classification.inner_loop import load_memory_system
    from text_classification.llm import LLM
    from text_classification.memory_system import extract_json_field

    return load_memory_system, LLM, _load_symptom2disease, extract_json_field


def bundled_template() -> str:
    """Copy of the dataset prompt shipped in the wheel; tests pin it byte-for-byte to the derived one."""
    text = BUNDLED_TEMPLATE_PATH.read_text(encoding="utf-8")
    if text.count(_BUNDLED_SLOT) != 1:
        raise RuntimeError(f"{BUNDLED_TEMPLATE_PATH.name} must contain exactly one {_BUNDLED_SLOT}")
    return text.replace(_BUNDLED_SLOT, _SYMPTOM_SLOT)


def _symptom_template(load_split: Callable) -> str:
    """Reuse the dataset's own diagnostician prompt so CLI input matches evaluation input.

    A standalone bundle ships without the dataset splits; only then is the bundled copy used.
    """
    try:
        example = load_split("train", limit=1)[0]
    except FileNotFoundError:
        return bundled_template()
    raw = example["raw_question"]
    if example["input"].count(raw) != 1:
        raise RuntimeError("Could not locate the symptom slot in the dataset prompt template")
    return example["input"].replace(raw, _SYMPTOM_SLOT)


def labels_from_template(template: str) -> tuple[str, ...]:
    """The 22 Symptom2Disease classes, read from the 'Possible diagnoses include:' line of the prompt."""
    match = _LABELS_RE.search(template)
    if not match:
        raise RuntimeError("Could not find the label list in the dataset prompt template")
    return tuple(label.strip() for label in match.group("labels").split(","))


def prompt_sections(prompt: str, mistake_texts: set[str], session_texts: set[str] = frozenset()) -> Sections:
    """Read examples and notes back out of a prompt built by any of the eight systems.

    Layouts: knn_top6 / fewshot_all put Q/A pairs straight before **Problem:**; confusion_contrast and
    contrast_online head them with **Similar solved examples:** and add **Known confusions ...**;
    cue_bank / cue_bank_verified / reflexion_notes add one bold lessons header after the examples.
    """
    if _PROBLEM not in prompt:  # no_memory: the prompt is the bare question
        return Sections((), (), "")
    head = prompt.split(_PROBLEM, 1)[0]
    notes_title, notes_body = "", ""
    for match in _HEADER_RE.finditer(head):
        if match.group("title").strip() != _EXAMPLES_TITLE:
            notes_title = match.group("title").strip()
            notes_body = head[match.end():]
            head = head[: match.start()]
            break
    evidence = []
    for qa in _QA_RE.finditer(head):
        q = qa.group("q").strip()
        evidence.append(Evidence(q, qa.group("a").strip(), q in mistake_texts, q in session_texts))
    lines = (ln.strip() for ln in notes_body.splitlines())
    notes = tuple(ln[2:].strip() if ln.startswith("- ") else ln for ln in lines if ln and ln != "- none recorded")
    return Sections(tuple(evidence), notes, notes_title)


def parse_prompt(prompt: str, mistake_texts: set[str]) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
    """(examples, confusion warnings) — the v0.1 view, kept for the comparison screen."""
    sections = prompt_sections(prompt, mistake_texts)
    warnings = sections.notes if sections.notes_title.startswith("Known confusions") else ()
    return sections.evidence, warnings


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class OfflineLLM:
    """Stands in for the LLM in demo mode, so nothing can reach the network."""

    def __call__(self, prompt: str) -> str:
        raise ConnectionError("demo mode: LLM calls are disabled")


class HarnessEngine:
    def __init__(self, settings: Settings, llm: Any = None, offline: bool = False) -> None:
        prepare_runtime_env(offline=offline, api_base=settings.api_base)
        if offline and llm is None:
            llm = OfflineLLM()
        load_memory_system, LLM, load_split, self._extract = _import_harness()
        if not settings.memory_path.exists():
            raise FileNotFoundError(f"memory file not found: {settings.memory_path}")
        self.settings = settings
        self.llm = llm or LLM(model=settings.model, api_base=settings.api_base, temperature=0.0)
        self.system = load_memory_system(settings.system, self.llm)
        self.frozen_state = settings.memory_path.read_text()  # read once; never written back
        self.system.set_state(self.frozen_state)
        self.session_examples: tuple[dict[str, str], ...] = ()  # owned by the Scratchpad
        self._template = _symptom_template(load_split)
        self.labels = labels_from_template(self._template)
        self._lock = threading.Lock()

    # --- prompt -----------------------------------------------------------------------------
    def wrap(self, symptoms: str) -> str:
        return self._template.replace(_SYMPTOM_SLOT, symptoms.strip())

    @property
    def uses_examples(self) -> bool:
        """True when the system retrieves from an `examples` list that /record can extend."""
        return isinstance(getattr(self.system, "examples", None), list)

    def _mistake_texts(self) -> set[str]:
        examples = getattr(self.system, "examples", [])
        ids = getattr(self.system, "error_ids", [])
        texts = {self.system._text(examples[i]) for i in ids if i < len(examples)}
        return texts | {it["symptoms"] for it in getattr(self.system, "items", [])}

    def _session_texts(self) -> set[str]:
        return {ex["raw_question"] for ex in self.session_examples}

    def prompt_for(self, task_input: str) -> str:
        """The exact prompt the system sends for `task_input`."""
        system = self.system
        if hasattr(system, "_build_prompt"):
            return system._build_prompt(task_input)
        module = sys.modules[type(system).__module__]
        if hasattr(system, "_format_examples_section"):  # fewshot_all builds its prompt inside predict()
            base = next(sys.modules[c.__module__] for c in type(system).__mro__ if hasattr(
                sys.modules[c.__module__], "PROMPT_TEMPLATE"))
            section = system._format_examples_section(seed=base._seed_for_input(task_input))
            return base.PROMPT_TEMPLATE.format(examples_section=section, input=task_input)
        template = getattr(module, "PROMPT", "")
        return template.format(input=task_input) if template else ""

    def sections(self, prompt: str) -> Sections:
        return prompt_sections(prompt, self._mistake_texts(), self._session_texts())

    def fingerprint(self) -> str:
        """Identifies the memory a prompt is built from: frozen file plus session additions."""
        return sha(self.frozen_state + json.dumps(self.session_examples, sort_keys=True))

    @property
    def modified(self) -> bool:
        return bool(self.session_examples)

    # --- predictions ------------------------------------------------------------------------
    def _build(self, symptoms: str, response: str, seconds: float, source: str,
               prompt: str, tokens: int | None) -> Prediction:
        sec = self.sections(prompt)
        warnings = sec.notes if sec.notes_title.startswith("Known confusions") else ()
        return Prediction(
            self._extract(response, "final_answer") if response else "",
            self._extract(response, "reasoning") if response else "",
            warnings, sec.evidence, len(prompt), seconds, self.settings.system, source, response,
            sec.notes, sec.notes_title, tokens, sha(prompt), prompt,
        )

    def predict(self, symptoms: str) -> Prediction:
        task_input = self.wrap(symptoms)
        prompt = self.prompt_for(task_input)
        with self._lock:  # one call at a time per engine, so the token delta belongs to this call
            before = getattr(self.llm, "total_input_tokens", 0)
            started = time.perf_counter()
            answer, info = self.system.predict(task_input)
            elapsed = time.perf_counter() - started
            tokens = getattr(self.llm, "total_input_tokens", 0) - before
        sent = self.system.get_last_prompt_info().get("prompt_text")
        if sent is not None and sent != prompt:
            raise RuntimeError(f"{self.settings.system}: displayed prompt differs from the prompt sent")
        pred = self._build(symptoms, info.get("full_response", ""), elapsed, "live", prompt, tokens or None)
        return replace(pred, diagnosis=answer)

    def from_response(self, symptoms: str, response: str, seconds: float, prompt_tokens: int | None = None) -> Prediction:
        """Rebuild a prediction from a recorded model response; retrieval is local, so evidence is recomputed."""
        prompt = self.prompt_for(self.wrap(symptoms))
        return self._build(symptoms, response, seconds, "replay", prompt, prompt_tokens)

    def preview(self, symptoms: str) -> Prediction:
        """Retrieval only: what this memory would put in the prompt, with no model answer."""
        prompt = self.prompt_for(self.wrap(symptoms))
        return self._build(symptoms, "", 0.0, "preview", prompt, None)

    def picks(self, symptoms: str) -> tuple[Pick, ...]:
        """Why each injected example was chosen, with the system's own TF-IDF cosine score.

        Scores come from the system's `_rank` (the code that built the prompt). Only the top-k are
        chosen for similarity; the rest are past mistakes or one-per-label coverage picks.
        """
        task_input = self.wrap(symptoms)
        evidence = self.sections(self.prompt_for(task_input)).evidence
        rank = getattr(self.system, "_rank", None)
        if rank is None or not hasattr(self.system, "k"):
            return tuple(Pick(e.question, e.label, None, "all" if evidence else "") for e in evidence)
        ranked = rank(task_input)
        text = self.system._text
        scores = {text(self.system.examples[i]): s for s, i in reversed(ranked)}  # best score wins on duplicates
        top = {text(self.system.examples[i]) for _, i in ranked[: self.system.k]}

        def reason(e: Evidence) -> str:
            if e.question in top:
                return f"top-{self.system.k} similar"
            return "past mistake" if e.past_mistake else "label coverage"

        return tuple(Pick(e.question, e.label, scores.get(e.question), reason(e)) for e in evidence)

    def stats(self) -> MemoryStats:
        examples = getattr(self.system, "examples", [])
        labels = Counter(e["target"] for e in examples).most_common()
        confusions = getattr(self.system, "confusions", Counter())
        items = getattr(self.system, "items", [])
        return MemoryStats(
            system=self.settings.system,
            examples=len(examples),
            labels=tuple(labels),
            mistakes=len(getattr(self.system, "error_ids", [])) or len(items),
            confusions=tuple((t, p, c) for (t, p), c in confusions.most_common()),
            items=len(items),
            session_examples=len(self.session_examples),
        )


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if config.has_real_key() else {}


def list_models(api_base: str, timeout: float = 1.5) -> list[str] | None:
    """Model ids served at api_base, or None when it does not answer /models within `timeout` seconds."""
    request = urllib.request.Request(f"{api_base.rstrip('/')}/models", headers=_auth_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read() or b"{}").get("data", [])
    except Exception:
        return None
    return [m["id"] for m in data if isinstance(m, dict) and "id" in m]


def probe(api_base: str, timeout: float = 1.5) -> bool:
    """True when the model server answers /models within `timeout` seconds."""
    return list_models(api_base, timeout) is not None


def discover(settings: Settings, timeout: float = 1.5,
             lister: Callable[[str, float], list[str] | None] = list_models) -> Settings | None:
    """First reachable endpoint of config.api_base_candidates(), with a model it serves; None when all are down.

    Model choice: HARNESS_MODEL, then the model saved by `harness-cli config` for that same endpoint, then
    the scored Qwen on the on-prem server, gpt-4o-mini on api.openai.com, and otherwise (Ollama, vLLM,
    OpenRouter) the first model the endpoint lists, routed through litellm's openai/ prefix.
    """
    saved = config.user_config()
    for base in config.api_base_candidates():
        models = lister(base, timeout)
        if models is None:
            continue
        if "HARNESS_MODEL" in os.environ:
            return replace(settings, api_base=base)
        if saved.get("model") and saved.get("api_base", "").rstrip("/") == base:
            model = saved["model"]
            return replace(settings, api_base=base, model=model if model.startswith("openai/") else f"openai/{model}")
        if base == config.ONPREM_API_BASE or not models:
            return replace(settings, api_base=base)
        if base == config.OPENAI_API_BASE:
            return replace(settings, api_base=base, model=config.OPENAI_DEFAULT_MODEL)
        return replace(settings, api_base=base, model=f"openai/{models[0]}")
    return None


def run_with_deadline(fn: Callable[[], Any], seconds: float) -> Any:
    """Run fn in a daemon thread and give up after `seconds`.

    The harness LLM retries for minutes on a dead network, so a plain try/except would freeze
    the demo. A daemon thread can be abandoned without blocking interpreter exit.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # handed back to the caller below
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no answer within {seconds:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _norm(symptoms: str) -> str:
    return " ".join(symptoms.lower().split())


class ReplayCache:
    """Model responses recorded from the live server (see `harness-cli record`).

    Recordings exist only for the frozen memories. A memory changed by /record gets a fingerprinted
    key that nothing was recorded under, so a stale answer can never be replayed for it.
    """

    def __init__(self, path: Path = REPLAY_PATH) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Any]] = json.loads(path.read_text()) if path.exists() else {}
        self._lock = threading.Lock()

    @staticmethod
    def key(system: str, symptoms: str, fingerprint: str | None = None) -> str:
        return f"{system}@{fingerprint}::{_norm(symptoms)}" if fingerprint else f"{system}::{_norm(symptoms)}"

    def get(self, system: str, symptoms: str, fingerprint: str | None = None) -> dict[str, Any] | None:
        return self.entries.get(self.key(system, symptoms, fingerprint))

    def put(self, system: str, symptoms: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self.entries[self.key(system, symptoms)] = entry

    def save(self) -> None:
        """Write atomically, so an interrupted or concurrent record never leaves a truncated replay file."""
        with self._lock:
            text = json.dumps(self.entries, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(text)
            os.replace(tmp, self.path)


PREVIEW_NOTE = "세션 기억이 바뀌어 녹화 응답을 쓸 수 없습니다. 바뀐 검색 결과만 표시합니다 (진단이 바뀌는지는 실시간 서버에서만 확인 가능)."
OFFLINE_PREVIEW = "[OFFLINE RETRIEVAL PREVIEW: Exact Prompt Generated | LLM Unreachable]"
UNRECORDED_NOTE = (f"{config.NO_SERVER_MESSAGE}\n{OFFLINE_PREVIEW} 하네스가 고른 예시와 보낼 프롬프트 원문을 "
                   "그대로 보여 줍니다. 설정: harness-cli config --api-base <url> --model <model>")


@dataclass
class SafeEngine:
    """Live prediction with a time limit, falling back to recorded responses.

    Once a live call fails, the session stays in replay mode so later cases do not each
    wait out the deadline again.
    """

    engine: HarnessEngine
    cache: ReplayCache
    live: bool
    deadline: float = 20.0

    def predict(self, symptoms: str) -> tuple[Prediction | None, str | None]:
        """Return (prediction, note).

        prediction.source is "live", "replay", or "preview" (memory changed by /record and no live
        server: evidence only, empty diagnosis). None when nothing can answer.
        """
        note = None
        if self.live:
            try:
                return run_with_deadline(lambda: self.engine.predict(symptoms), self.deadline), None
            except Exception as exc:
                self.live = False
                note = f"서버 응답 실패({type(exc).__name__}) → 녹화 응답으로 전환"
        engine = self.engine
        fingerprint = engine.fingerprint() if engine.modified else None
        entry = self.cache.get(engine.settings.system, symptoms, fingerprint)
        if entry is not None:
            return engine.from_response(symptoms, entry["response"], entry["seconds"], entry.get("prompt_tokens")), note
        why = PREVIEW_NOTE if engine.modified else UNRECORDED_NOTE
        return engine.preview(symptoms), (note + " · " if note else "") + why


def with_system(settings: Settings, system: str, memory_path: Path) -> Settings:
    return replace(settings, system=system, memory_path=memory_path)


class EnginePool:
    """One SafeEngine per lego block, built on first use; every engine shares the session scratchpad."""

    def __init__(self, settings: Settings, cache: ReplayCache, live: bool, deadline: float = 20.0,
                 scratchpad: Scratchpad | None = None, llm_factory: Callable[[], Any] | None = None) -> None:
        self.settings = settings
        self.cache = cache
        self.live = live
        self.deadline = deadline
        self.scratchpad = scratchpad or Scratchpad()
        self._llm_factory = llm_factory
        self._engines: dict[str, SafeEngine] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> SafeEngine:
        block = lego.resolve(name)
        with self._lock:
            if block.name not in self._engines:
                # HARNESS_MEMORY still overrides the memory of the starting block (HARNESS_SYSTEM)
                path = self.settings.memory_path if block.name == self.settings.system else block.memory_path
                settings = with_system(self.settings, block.name, path)
                llm = self._llm_factory() if self._llm_factory else None
                engine = HarnessEngine(settings, llm=llm, offline=not self.live)
                self.scratchpad.attach(engine)
                self._engines[block.name] = SafeEngine(engine, self.cache, self.live, self.deadline)
            return self._engines[block.name]

    def loaded(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def predict(self, name: str, symptoms: str) -> tuple[Prediction | None, str | None]:
        safe = self.get(name)
        outcome = safe.predict(symptoms)
        if not safe.live and self.live:  # one dead call means the server is gone for every block
            self.live = False
            for other in self._engines.values():
                other.live = False
        return outcome
