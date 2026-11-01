"""Study Buddy task: questions over the user's own notes, with three swappable memory modes.

Modelled on Reference/koog-study-buddy-cli. Notes (.md / .txt) are split into heading-aware
chunks and ranked with the harness's own TF-IDF (agents/_text_rank.py), so retrieval here
weighs words exactly as knn_top6 does. Lessons taught with /record persist in ~/.harness and are
indexed with the notes immediately: the next question can retrieve them.

Every prompt sent to the model is English; Korean/CJK notes, questions and lessons are refused.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import config
from .home import HarnessHome, has_cjk

NOTE_SUFFIXES = (".md", ".markdown", ".txt")
MAX_CHUNK_CHARS = 900
MAX_NOTES_BYTES = 2_000_000
KNN_K = 4
CUE_K, CUE_NOTES_K = 5, 2
MAX_LESSONS = 50
LESSON_MIN, LESSON_MAX = 3, 600


@dataclass(frozen=True)
class StudyMode:
    name: str
    title: str  # short Korean label for badges
    summary: str  # one Korean line: what the mode puts into the prompt


MODES: dict[str, StudyMode] = {
    m.name: m
    for m in (
        StudyMode("no_memory", "기억 없음", "질문만 넣는다 (노트·규칙·프로필 없음, 기준선)"),
        StudyMode("knn_retrieval", "유사 단락", f"TF-IDF로 가장 비슷한 노트 단락 {KNN_K}개 (+ /record 로 가르친 항목)"),
        StudyMode("lesson_cues", "학습 규칙", f"/record 로 쌓은 규칙 중 비슷한 {CUE_K}개 + 노트 단락 {CUE_NOTES_K}개 (ReasoningBank 방식)"),
    )
}
MODE_ALIASES = {
    "none": "no_memory", "raw": "no_memory", "baseline": "no_memory",
    "knn": "knn_retrieval", "rag": "knn_retrieval", "retrieval": "knn_retrieval",
    "cues": "lesson_cues", "lessons": "lesson_cues", "rules": "lesson_cues",
}


class NotesError(ValueError):
    """A refused load, question or lesson; the message is shown to the user as is."""


def resolve_mode(name: str) -> StudyMode:
    key = name.strip().lower().replace("-", "_")
    key = MODE_ALIASES.get(key, key)
    if key not in MODES:
        raise NotesError(f"알 수 없는 학습 모드: '{name.strip()}'. 사용 가능: {', '.join(MODES)}")
    return MODES[key]


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str
    lesson: bool = False  # taught with /record rather than loaded from a file

    @property
    def searchable(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Lesson:
    concept: str
    answer: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class StudyAnswer:
    mode: str
    question: str
    answer: str
    prompt: str
    notes: tuple[Hit, ...]  # note chunks (and taught lessons, in knn_retrieval) placed in the prompt
    cues: tuple[Hit, ...]  # lessons placed as study rules (lesson_cues)
    source: str  # "live" | "preview"
    seconds: float
    prompt_tokens: int | None
    notes_chars: int  # size of everything loaded, to show how much context retrieval saved

    @property
    def context_saved(self) -> float | None:
        """Share of the loaded notes left out of the prompt; None when nothing is loaded."""
        if not self.notes_chars:
            return None
        injected = sum(len(h.chunk.text) for h in self.notes if not h.chunk.lesson)
        return 1 - injected / self.notes_chars


# --- loading ------------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    parts, current = [], ""
    for sentence in _SENTENCE.split(paragraph):
        while len(sentence) > limit:  # one very long sentence: hard cut
            parts.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + 1 + len(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return [*parts, current] if current else parts


def chunk_text(text: str, source: str, limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split notes at headings, then pack paragraphs into chunks of at most `limit` characters."""
    sections: list[tuple[str, list[str]]] = [(Path(source).stem, [])]
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for title, lines in sections:
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", "\n".join(lines))]
        pieces = [piece for p in paragraphs if p for piece in _split_long(p, limit)]
        current = ""
        for piece in pieces:
            if current and len(current) + 2 + len(piece) > limit:
                chunks.append(Chunk(source, title, current))
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append(Chunk(source, title, current))
    return chunks


def read_notes(path: Path) -> str:
    path = path.expanduser()
    if path.suffix.lower() not in NOTE_SUFFIXES:
        raise NotesError(f"{path.name}: .md, .markdown, .txt 파일만 불러올 수 있습니다 (PDF는 텍스트로 변환해 주세요).")
    if not path.is_file():
        raise NotesError(f"파일이 없습니다: {path}")
    if path.stat().st_size > MAX_NOTES_BYTES:
        raise NotesError(f"{path.name}: {MAX_NOTES_BYTES // 1_000_000}MB 이하 파일만 불러올 수 있습니다.")
    text = path.read_text(encoding="utf-8", errors="replace")
    if has_cjk(text):
        raise NotesError(f"{path.name}: 한글/한자가 들어 있습니다. 모델 프롬프트는 영어만 쓰므로 영어 노트만 불러올 수 있습니다.")
    if not text.strip():
        raise NotesError(f"{path.name}: 빈 파일입니다.")
    return text


# --- prompts ------------------------------------------------------------------------------------

_HEAD = "You are a study assistant helping a student review their own course notes."
_INSTRUCTIONS_GROUNDED = """**Instructions:**
- Answer in English, in at most 200 words.
- Base the answer on the notes and study rules above when they cover the question, and cite note numbers like [2].
- If they do not cover it, write "Not in your notes." and then give a brief general answer."""
_INSTRUCTIONS_RAW = """**Instructions:**
- Answer in English, in at most 200 words."""


def profile_line(profile: dict[str, str]) -> str:
    parts = [
        f"preparing for {profile['target_exam']}" if profile.get("target_exam") else "",
        f"focus topics: {profile['topics']}" if profile.get("topics") else "",
        f"preferred explanation style: {profile['style']}" if profile.get("style") else "",
    ]
    parts = [p for p in parts if p]
    return f"Learner profile: {'; '.join(parts)}." if parts else ""


def build_prompt(question: str, mode: str, notes: tuple[Hit, ...], cues: tuple[Hit, ...],
                 profile: dict[str, str]) -> str:
    if mode == "no_memory":
        return f"{_HEAD}\n\n**Question:**\n{question}\n\n{_INSTRUCTIONS_RAW}"
    blocks = [_HEAD]
    if line := profile_line(profile):
        blocks.append(line)
    if notes:
        body = "\n\n".join(f"[{i}] ({h.chunk.title}) {h.chunk.text}" for i, h in enumerate(notes, 1))
        blocks.append(f"**Course notes:**\n{body}")
    if mode == "lesson_cues":
        rules = "\n".join(f"- {h.chunk.title}: {h.chunk.text}" for h in cues) or "- none recorded"
        blocks.append(f"**Study rules (taught by the student):**\n{rules}")
    blocks.append(f"**Question:**\n{question}")
    blocks.append(_INSTRUCTIONS_GROUNDED)
    return "\n\n".join(blocks)


# --- retrieval ----------------------------------------------------------------------------------

Ranker = Callable[[str, list[str]], list[tuple[float, int]]]


def harness_ranker() -> Ranker:
    """The harness's TF-IDF cosine ranking, so study retrieval weighs words like knn_top6 does."""
    lab = str(config.HARNESS_LAB)
    if lab not in sys.path:
        sys.path.insert(0, lab)
    from text_classification.agents._text_rank import rank_texts

    return rank_texts


def _rank(ranker: Ranker, query: str, chunks: list[Chunk], k: int) -> tuple[Hit, ...]:
    ranked = ranker(query, [c.searchable for c in chunks])
    return tuple(Hit(chunks[i], s) for s, i in ranked[:k] if s > 0)


class StudyBuddy:
    def __init__(self, home: HarnessHome | None = None, ranker: Ranker | None = None) -> None:
        self._home = home
        self._ranker = ranker
        self.chunks: list[Chunk] = []
        self.sources: dict[str, int] = {}  # file -> characters loaded
        self.lessons: tuple[Lesson, ...] = self._load_lessons()

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            self._ranker = harness_ranker()
        return self._ranker

    @property
    def notes_chars(self) -> int:
        return sum(self.sources.values())

    # --- notes ------------------------------------------------------------------------------
    def load(self, path: Path) -> int:
        """Index a notes file (replacing an earlier load of the same file); returns its chunk count."""
        text = read_notes(path)
        source = str(path.expanduser().resolve())
        new = chunk_text(text, source)
        self.chunks = [c for c in self.chunks if c.source != source] + new
        self.sources[source] = sum(len(c.text) for c in new)
        return len(new)

    # --- lessons (persistent working memory) ------------------------------------------------
    def _load_lessons(self) -> tuple[Lesson, ...]:
        if self._home is None:
            return ()
        rows = self._home.records("study")
        lessons = (Lesson(r["concept"], r["answer"]) for r in rows
                   if isinstance(r.get("concept"), str) and isinstance(r.get("answer"), str))
        return tuple(x for x in lessons if not has_cjk(x.concept + x.answer))[:MAX_LESSONS]

    def _save_lessons(self) -> None:
        if self._home is not None:
            self._home.save_records("study", [{"concept": lesson.concept, "answer": lesson.answer} for lesson in self.lessons])

    def add_lesson(self, concept: str, answer: str) -> Lesson:
        concept, answer = " ".join(concept.split()), " ".join(answer.split())
        for text in (concept, answer):
            if has_cjk(text):
                raise NotesError("개념과 답은 영어로 입력해 주세요 (모델 프롬프트는 영어만 씁니다).")
            if not LESSON_MIN <= len(text) <= LESSON_MAX:
                raise NotesError(f"개념과 답은 각각 {LESSON_MIN}~{LESSON_MAX}자여야 합니다.")
        if any(lesson.concept.lower() == concept.lower() for lesson in self.lessons):
            raise NotesError("같은 개념이 이미 학습 기억에 있습니다. /memory reset 으로 비울 수 있습니다.")
        if len(self.lessons) >= MAX_LESSONS:
            raise NotesError(f"학습 기억은 {MAX_LESSONS}건까지입니다. /memory reset 으로 비울 수 있습니다.")
        lesson = Lesson(concept, answer)
        self.lessons = (*self.lessons, lesson)
        self._save_lessons()
        return lesson

    def reset_lessons(self) -> int:
        removed = len(self.lessons)
        self.lessons = ()
        self._save_lessons()
        return removed

    def lesson_chunks(self) -> list[Chunk]:
        return [Chunk("/record", lesson.concept, lesson.answer, lesson=True) for lesson in self.lessons]

    # --- asking -----------------------------------------------------------------------------
    def retrieve(self, question: str, mode: str) -> tuple[tuple[Hit, ...], tuple[Hit, ...]]:
        """(note hits, cue hits) the given mode puts into the prompt."""
        if mode == "no_memory":
            return (), ()
        if mode == "knn_retrieval":
            return _rank(self.ranker, question, self.chunks + self.lesson_chunks(), KNN_K), ()
        return (_rank(self.ranker, question, self.chunks, CUE_NOTES_K),
                _rank(self.ranker, question, self.lesson_chunks(), CUE_K))

    def prepare(self, question: str, mode: str, profile: dict[str, str]) -> StudyAnswer:
        """Everything but the model call: the exact prompt and what retrieval chose (source='preview')."""
        question = " ".join(question.split())
        if has_cjk(question):
            raise NotesError("질문은 영어로 입력해 주세요 (모델 프롬프트는 영어만 씁니다).")
        mode = resolve_mode(mode).name
        notes, cues = self.retrieve(question, mode)
        prompt = build_prompt(question, mode, notes, cues, profile)
        return StudyAnswer(mode, question, "", prompt, notes, cues, "preview", 0.0, None, self.notes_chars)

    def ask(self, question: str, mode: str, profile: dict[str, str],
            complete: Callable[[str], tuple[str, int | None]] | None = None) -> StudyAnswer:
        """Answer with `complete` (prompt -> (text, prompt tokens)); without it, return the preview."""
        prepared = self.prepare(question, mode, profile)
        if complete is None:
            return prepared
        started = time.perf_counter()
        text, tokens = complete(prepared.prompt)
        return StudyAnswer(prepared.mode, prepared.question, text.strip(), prepared.prompt, prepared.notes,
                           prepared.cues, "live", time.perf_counter() - started, tokens, prepared.notes_chars)


QUIZ_PROMPT = "Write three short review questions, each followed by its answer, about: {topic}"


def quiz_question(topic: str) -> str:
    return QUIZ_PROMPT.format(topic=topic.strip() or "the most important ideas in these notes")


# --- model call ---------------------------------------------------------------------------------

def chat_completion(api_base: str, model: str, prompt: str, timeout: float = 60.0,
                    api_key: str = "") -> tuple[str, int | None]:
    """One OpenAI-compatible chat call (SGLang, vLLM, Ollama, OpenRouter); returns (text, prompt tokens)."""
    body: dict = {
        "model": model.removeprefix("openai/"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    if not config.is_hosted(api_base):
        body["chat_template_kwargs"] = {"enable_thinking": False}  # same as the evaluated runs
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != config.PLACEHOLDER_KEY:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{api_base.rstrip('/')}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"].get("content") or ""
    return text, (data.get("usage") or {}).get("prompt_tokens")
