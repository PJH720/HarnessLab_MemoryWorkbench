"""/study: a four-step walk through the research results, one side-by-side run per step.

Every number shown here comes from a result file named next to it; tests/test_study.py re-reads
those files (when present) and fails if a constant drifts. Accuracies and token counts are taken
from lego.BLOCKS, which has its own drift test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import lego

TALK_URL = "https://harness-midterm-talk.vercel.app"
MCNEMAR_CSV = "analysis/multiseed/out/mcnemar_by_seed.csv"
POOLED_CSV = "analysis/multiseed/out/pooled.csv"
H1_SOURCES = ("analysis/multiseed/out/coverage.json", "analysis/multiseed/out/SUMMARY.md")
REFERENCE = "knn_top6"  # analysis/multiseed/stats.py REFERENCE: every paired test is against this block


@dataclass(frozen=True)
class Paired:
    """One paired comparison against knn_top6: questions only knn_top6 got right vs only `system` got right."""

    system: str
    split: str
    seed: int | None  # None: pooled over seeds 42/123/456
    only_ref: int
    only_sys: int
    p: float
    p_name: str  # which column of the source file `p` is
    source: str


# analysis/multiseed/out/pooled.csv (split=test, n=636 = 212 x 3 seeds), column mh_p
POOLED_NO_MEMORY = Paired("no_memory", "test", None, 130, 28, 4.870493385070401e-16, "Mantel-Haenszel", POOLED_CSV)
POOLED_FEWSHOT_ALL = Paired("fewshot_all", "test", None, 23, 33, 0.18144920772141646, "Mantel-Haenszel", POOLED_CSV)
# analysis/multiseed/out/mcnemar_by_seed.csv (sys=cue_bank, seed=42), column p_exact
CUE_BANK_VAL = Paired("cue_bank", "val", 42, 7, 1, 0.0703125, "exact McNemar", MCNEMAR_CSV)
CUE_BANK_TEST = Paired("cue_bank", "test", 42, 13, 10, 0.6776394844055176, "exact McNemar", MCNEMAR_CSV)
# analysis/multiseed/out/coverage.json "decision" (37/37 runs analysed); SUMMARY.md line "Decision per prereg ..."
H1_DECISION = "NOT SUPPORTED"

STEP_COMPARES: tuple[tuple[str, str, str], ...] = (
    ("no_memory", "knn_top6", "1"),
    ("knn_top6", "fewshot_all", "2"),
    ("knn_top6", "cue_bank", "3"),
)

Ask = Callable[[str], str]
Compare = Callable[[str, str, str], None]


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}%p"


def _mean(name: str) -> float:
    return lego.BLOCKS[name].scores.mean_accuracy("test")


def _seed42(name: str, split: str) -> float:
    return lego.BLOCKS[name].scores.seed_accuracy(42, split)


def _p(p: float) -> str:
    return "p < 1e-15" if p < 1e-15 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def intro() -> Text:
    return Text.assemble(
        ("레고 하네스 학습 가이드 (4단계)\n\n", "bold"),
        ("모델(Qwen3.8)은 고정하고, 프롬프트에 무엇을 넣는지(하네스)만 갈아 끼워 봅니다.\n", ""),
        ("1. 기억 없음의 한계  2. 예시 선별  3. 단서 주입  4. 자유 실험\n\n", "dim"),
        ("각 단계의 수치는 test 212문항 평가 기록에서 가져오고, 화면의 한 건은 예시로만 봅니다.\n", "dim"),
        (f"발표 자료: {TALK_URL}", "dim"),
    )


def step1() -> Text:
    nm, knn, pair = _mean("no_memory"), _mean("knn_top6"), POOLED_NO_MEMORY
    return Text.assemble(
        ("1단계 · 기억 없음의 한계\n\n", "bold"),
        ("no_memory는 증상만 넣고, knn_top6는 비슷한 학습 예시 6개를 함께 넣습니다.\n\n", ""),
        ("test 정확도 (3-seed 평균, 각 n=212)  ", "dim"),
        (f"no_memory {_pct(nm)}  vs  knn_top6 {_pct(knn)}  ({_pp(knn - nm)})\n", "bold"),
        ("한쪽만 맞힌 문항 (3 seed 합산, 636문항)  ", "dim"),
        (f"knn_top6만 {pair.only_ref} : no_memory만 {pair.only_sys} · {pair.p_name} {_p(pair.p)}\n", "bold"),
        (f"출처: {pair.source}, analysis/multiseed/out/accuracy_tokens.csv\n\n", "dim"),
        ("아래 프리셋 한 건에서는 둘 다 맞을 수도 있습니다. 차이는 문항 수백 개에서 드러납니다.", "dim"),
    )


def step2() -> Text:
    few, knn, pair = _mean("fewshot_all"), _mean("knn_top6"), POOLED_FEWSHOT_ALL
    saving = lego.token_reduction("knn_top6", "fewshot_all")
    return Text.assemble(
        ("2단계 · 예시 선별\n\n", "bold"),
        ("fewshot_all은 학습 예시 60개를 모두 넣고, knn_top6는 비슷한 6개만 고릅니다.\n\n", ""),
        ("test 정확도 (3-seed 평균)  ", "dim"),
        (f"fewshot_all {_pct(few)}  vs  knn_top6 {_pct(knn)}\n", "bold"),
        ("한쪽만 맞힌 문항 (636문항)  ", "dim"),
        (f"knn_top6만 {pair.only_ref} : fewshot_all만 {pair.only_sys} · {pair.p_name} {_p(pair.p)}"
         " → 두 시스템은 통계적으로 구분되지 않습니다 (같다는 뜻은 아님)\n", "bold"),
        ("평가 입력 토큰  ", "dim"),
        (f"knn_top6가 fewshot_all보다 {saving:.1%} 적음 (약 1/5, test 3-seed 평균)\n", "bold"),
        (f"출처: {pair.source}, analysis/multiseed/out/accuracy_tokens.csv (cached_eval_input_tokens)\n\n", "dim"),
        ("아래 비교에서 프롬프트 길이 차이를 확인해 보세요.", "dim"),
    )


def step3() -> Text:
    val, test = CUE_BANK_VAL, CUE_BANK_TEST
    d_val = _seed42("cue_bank", "val") - _seed42("knn_top6", "val")
    d_test = _seed42("cue_bank", "test") - _seed42("knn_top6", "test")
    return Text.assemble(
        ("3단계 · 단서 주입\n\n", "bold"),
        ("cue_bank는 비슷한 예시에 더해, 학습 중 오답에서 뽑은 구분 단서를 넣습니다.\n", ""),
        ("단서가 모델을 특정 답으로 끌어당긴다(닻 효과)는 설명은 가설일 뿐이며, 아래 기록으로 확인되지 않았습니다.\n\n", ""),
        ("seed 42 val (n=50)  ", "dim"),
        (f"cue_bank − knn_top6 = {_pp(d_val)} · 한쪽만 맞힘 {val.only_ref}:{val.only_sys} · "
         f"{val.p_name} {_p(val.p)} → 유의하지 않음\n", "bold"),
        ("seed 42 test (n=212, 같은 기억)  ", "dim"),
        (f"cue_bank − knn_top6 = {_pp(d_test)} · {test.only_ref}:{test.only_sys} · {test.p_name} {_p(test.p)}\n", "bold"),
        ("사전 등록 가설 H1  ", "dim"),
        (f"{H1_DECISION} (지지되지 않음)\n", "bold"),
        (f"출처: {val.source}, {', '.join(H1_SOURCES)}\n\n", "dim"),
        ("val 50문항의 −12%p는 test 212문항에서 재현되지 않았습니다. 작은 표본의 차이를 결론으로 읽지 마세요.", "dim"),
    )


def step4(live: bool) -> Text:
    tail = (
        "서버가 연결되어 있으면 새 예시가 진단을 바꾸는지까지 볼 수 있습니다."
        if live else "지금은 오프라인이라, 세션 기억을 바꾸면 검색 결과(고른 예시)만 보이고 진단은 비어 있습니다."
    )
    return Text.assemble(
        ("4단계 · 자유 실험\n\n", "bold"),
        ("직접 예시를 세션 기억에 넣고, knn_top6가 그 예시를 찾아오는지 확인해 보세요.\n\n", ""),
        ('  /record "<영어 증상>" -> "<질병>"', "bold"), ("   세션 기억에 예시 추가 (원본 기억 파일은 그대로)\n", "dim"),
        ("  /compare no_memory knn_top6 <같은 증상>", "bold"), ("   좌우 비교\n", "dim"),
        ("  /memory status · /memory reset", "bold"), ("   세션 기억 확인 · 원래대로\n", "dim"),
        ("  /harness", "bold"), ("   블록 8개 목록 · /harness <이름> 으로 교체\n\n", "dim"),
        (tail + "\n", "dim"),
        (f"발표 자료: {TALK_URL}", "dim"),
    )


def _wants_quit(ask: Ask, prompt: str) -> bool:
    try:
        return ask(prompt).strip().lower() in ("q", "quit", "exit")
    except (EOFError, KeyboardInterrupt):
        return True


def run(console: Console, compare: Compare, ask: Ask, live: bool, session_records: int = 0) -> int:
    """Run the tutorial; returns the number of steps shown. Enter advances, q (or EOF) quits."""
    console.print(Panel(intro(), border_style="#64748b", padding=(1, 2)))
    if session_records and not live:
        console.print(Text(
            f"세션 기억에 예시 {session_records}건이 있어, 오프라인에서는 예시 기반 블록이 진단 없이 검색 결과만 보입니다."
            " 원래 기록대로 보려면 /memory reset 후 다시 /study.", style="yellow"))
    steps = (step1(), step2(), step3())
    for i, (body, (left, right, preset)) in enumerate(zip(steps, STEP_COMPARES), start=1):
        if _wants_quit(ask, "Enter 다음 · q 종료"):
            return i - 1
        console.print(Panel(body, border_style="#64748b", padding=(1, 2)))
        compare(left, right, preset)
    if _wants_quit(ask, "Enter 다음 · q 종료"):
        return len(steps)
    console.print(Panel(step4(live), border_style="#64748b", padding=(1, 2)))
    return len(steps) + 1
