"""The eight memory systems as swappable blocks: frozen memory file plus the scored numbers behind it.

Every number below is copied once from a result file and named next to it; tests/test_lego.py
re-reads those files (when present) and fails if a constant drifts.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from . import config

ONLINE_V2 = "online-v2/Symptom2Disease"
MIDTERM = "midterm/Symptom2Disease"
ACCURACY_CSV = config.PROJECT_ROOT / "analysis/multiseed/out/accuracy_tokens.csv"


@dataclass(frozen=True)
class Run:
    seed: int
    split: str  # "val" (n=50) | "test" (n=212)
    n: int
    correct: int
    input_tokens: int  # summed over all n evaluation prompts of the split, not per query

    @property
    def accuracy(self) -> float:
        return self.correct / self.n


@dataclass(frozen=True)
class Scores:
    runs: tuple[Run, ...]
    source: str  # file the runs were copied from
    token_metric: str  # column / field the input_tokens came from

    def split(self, name: str) -> tuple[Run, ...]:
        return tuple(r for r in self.runs if r.split == name)

    def seeds(self, split: str = "test") -> tuple[int, ...]:
        return tuple(r.seed for r in self.split(split))

    def n(self, split: str = "test") -> int:
        return self.split(split)[0].n

    def mean_accuracy(self, split: str = "test") -> float:
        runs = self.split(split)
        return sum(r.accuracy for r in runs) / len(runs)

    def mean_tokens(self, split: str = "test") -> float:
        runs = self.split(split)
        return sum(r.input_tokens for r in runs) / len(runs)

    def seed_accuracy(self, seed: int, split: str = "test") -> float | None:
        return next((r.accuracy for r in self.split(split) if r.seed == seed), None)


def _online_v2(rows: tuple[tuple[int, str, int, int, int], ...]) -> Scores:
    return Scores(
        tuple(Run(*row) for row in rows),
        "analysis/multiseed/out/accuracy_tokens.csv",
        "cached_eval_input_tokens",
    )


# Copied from analysis/multiseed/out/accuracy_tokens.csv (seed, split, n, correct, cached_eval_input_tokens).
_CSV_ROWS: dict[str, tuple[tuple[int, str, int, int, int], ...]] = {
    "no_memory": (
        (42, "test", 212, 142, 56_814), (123, "test", 212, 142, 56_814), (456, "test", 212, 142, 56_814),
        (42, "val", 50, 35, 13_424), (123, "val", 50, 35, 13_424), (456, "val", 50, 35, 13_424),
    ),
    "fewshot_all": (
        (42, "test", 212, 182, 611_406), (123, "test", 212, 181, 603_562), (456, "test", 212, 175, 607_378),
        (42, "val", 50, 41, 144_224), (123, "val", 50, 41, 142_374), (456, "val", 50, 38, 143_274),
    ),
    "knn_top6": (
        (42, "test", 212, 175, 115_873), (123, "test", 212, 173, 115_764), (456, "test", 212, 180, 118_381),
        (42, "val", 50, 41, 27_395), (123, "val", 50, 38, 27_243), (456, "val", 50, 40, 27_894),
    ),
    "contrast_online": (
        (42, "test", 212, 170, 162_263), (123, "test", 212, 173, 158_582), (456, "test", 212, 176, 151_724),
        (42, "val", 50, 41, 38_679), (123, "val", 50, 37, 37_448), (456, "val", 50, 40, 35_479),
    ),
    "cue_bank": (
        (42, "test", 212, 172, 180_893), (123, "test", 212, 179, 179_937), (456, "test", 212, 172, 177_915),
        (42, "val", 50, 35, 42_495), (123, "val", 50, 39, 42_556), (456, "val", 50, 40, 41_936),
    ),
    "cue_bank_verified": (
        (42, "test", 212, 176, 158_511), (123, "test", 212, 177, 160_818), (456, "test", 212, 179, 166_942),
        (42, "val", 50, 37, 37_390), (123, "val", 50, 35, 38_022), (456, "val", 50, 40, 39_267),
    ),
    "reflexion_notes": (
        (42, "test", 212, 175, 174_276), (123, "test", 212, 172, 170_145), (456, "test", 212, 178, 175_120),
        (42, "val", 50, 39, 41_233), (123, "val", 50, 37, 40_085), (456, "val", 50, 39, 41_243),
    ),
}

# confusion_contrast was run once (seed 42, offline mode) in the midterm, so it has no 3-seed mean.
# Copied from logs/midterm/Symptom2Disease/confusion_contrast/qwen3.8-flash-next-nvfp4/{test,val}.json
# (correct, total, llm_input_tokens). llm_input_tokens is counted live, not recounted from the cache,
# so it is not the same metric as cached_eval_input_tokens above.
_MIDTERM_CONFUSION_CONTRAST = Scores(
    (Run(42, "test", 212, 171, 120_982), Run(42, "val", 50, 41, 28_745)),
    f"harness_lab/text_classification/logs/{MIDTERM}/confusion_contrast/{config.MODEL_DIR}/{{test,val}}.json",
    "llm_input_tokens",
)


@dataclass(frozen=True)
class Block:
    name: str
    title: str  # short Korean label for badges
    summary: str  # one Korean line: what the block puts into the prompt
    run: str  # ONLINE_V2 or MIDTERM
    scores: Scores
    uses_examples: bool  # /record changes this block's retrieved examples
    notes_kind: str = ""  # what the block injects besides examples ("" | "confusions" | "cues" | "notes")

    @property
    def memory_path(self) -> Path:
        """The copy bundled in the package (data/memories), so any clone or wheel finds it."""
        return config.bundled_memory(self.name)

    @property
    def log_path(self) -> Path:
        """Where the research run wrote this memory; gitignored, so only present on the machine that ran it."""
        return config.LOGS / self.run / self.name / config.MODEL_DIR / "memory.json"

    @property
    def multi_seed(self) -> bool:
        return len(self.scores.seeds()) > 1


BLOCKS: dict[str, Block] = {
    b.name: b
    for b in (
        Block("no_memory", "기억 없음", "증상만 넣는다 (예시·단서 없음)", ONLINE_V2,
              _online_v2(_CSV_ROWS["no_memory"]), uses_examples=False),
        Block("fewshot_all", "예시 전부", "학습 예시 60개를 30,000자 상한까지 모두 넣는다", ONLINE_V2,
              _online_v2(_CSV_ROWS["fewshot_all"]), uses_examples=True),
        Block("knn_top6", "유사 예시 6개", "TF-IDF로 가장 비슷한 예시 6개만 골라 넣는다", ONLINE_V2,
              _online_v2(_CSV_ROWS["knn_top6"]), uses_examples=True),
        Block("contrast_online", "오답 대조", "비슷한 예시 + 학습 중 틀린 예시 + 헷갈린 질병 쌍 경고", ONLINE_V2,
              _online_v2(_CSV_ROWS["contrast_online"]), uses_examples=True, notes_kind="confusions"),
        Block("cue_bank", "회피 단서", "비슷한 예시 + 오답에서 뽑은 구분 단서 (ReasoningBank 방식)", ONLINE_V2,
              _online_v2(_CSV_ROWS["cue_bank"]), uses_examples=True, notes_kind="cues"),
        Block("cue_bank_verified", "검증된 단서", "비슷한 예시 + 자기 검증을 통과한 단서만", ONLINE_V2,
              _online_v2(_CSV_ROWS["cue_bank_verified"]), uses_examples=True, notes_kind="cues"),
        Block("reflexion_notes", "반성 노트", "비슷한 예시 + 오답마다 쓴 자유 서술 반성 (Reflexion 방식)", ONLINE_V2,
              _online_v2(_CSV_ROWS["reflexion_notes"]), uses_examples=True, notes_kind="notes"),
        Block("confusion_contrast", "중간과제 하네스", "중간과제에서 AI가 만든 후보: 비슷한 예시 + 헷갈린 쌍 경고", MIDTERM,
              _MIDTERM_CONFUSION_CONTRAST, uses_examples=True, notes_kind="confusions"),
    )
}

ALIASES: dict[str, str] = {
    "none": "no_memory", "nomem": "no_memory", "raw": "no_memory", "baseline": "no_memory",
    "fewshot": "fewshot_all", "all": "fewshot_all",
    "knn": "knn_top6", "top6": "knn_top6",
    "contrast": "contrast_online",
    "cue": "cue_bank", "cues": "cue_bank", "reasoningbank": "cue_bank",
    "verified": "cue_bank_verified", "cue_verified": "cue_bank_verified",
    "reflexion": "reflexion_notes", "notes": "reflexion_notes",
    "cc": "confusion_contrast", "midterm": "confusion_contrast",
}


class UnknownHarness(ValueError):
    pass


def names() -> tuple[str, ...]:
    return tuple(BLOCKS)


def resolve(name: str) -> Block:
    """Look a block up by name or alias (case, spaces and '-' ignored); raise UnknownHarness with a hint."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key in BLOCKS:
        return BLOCKS[key]
    close = difflib.get_close_matches(key, [*BLOCKS, *ALIASES], n=1)
    hint = f" 혹시 '{ALIASES.get(close[0], close[0])}'?" if close else ""
    raise UnknownHarness(f"알 수 없는 하네스: '{name.strip()}'.{hint} 사용 가능: {', '.join(BLOCKS)}")


def token_reduction(name: str, against: str = "fewshot_all", split: str = "test") -> float:
    """1 - mean input tokens of `name` / mean of `against` (online-v2 blocks only, same metric)."""
    block, base = resolve(name), resolve(against)
    if block.scores.token_metric != base.scores.token_metric:
        raise ValueError(f"{block.name} and {base.name} count tokens differently; not comparable")
    return 1 - block.scores.mean_tokens(split) / base.scores.mean_tokens(split)


def pareto_frontier(split: str = "test") -> tuple[str, ...]:
    """Blocks no other block beats on both mean accuracy and mean input tokens (same token metric only).

    confusion_contrast counts tokens differently (live, single run), so it is left out.
    """
    metric = BLOCKS["knn_top6"].scores.token_metric
    pool = [b for b in BLOCKS.values() if b.scores.token_metric == metric]

    def dominated(b: Block) -> bool:
        acc, tok = b.scores.mean_accuracy(split), b.scores.mean_tokens(split)
        return any(
            o.scores.mean_accuracy(split) >= acc and o.scores.mean_tokens(split) <= tok
            and (o.scores.mean_accuracy(split) > acc or o.scores.mean_tokens(split) < tok)
            for o in pool if o is not b
        )

    frontier = sorted((b for b in pool if not dominated(b)), key=lambda b: b.scores.mean_tokens(split))
    return tuple(b.name for b in frontier)
