"""Rich renderers for v0.2.1 screens: offline prompt preview, /inspect, /explain, the study task, /profile."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config, lego
from .docqa import MODES, StudyAnswer, StudyMode
from .engine import OFFLINE_PREVIEW, Pick
from .home import PROFILE_HELP, PROFILE_KEYS
from .ui import ACCENT

PROMPT_HEAD, PROMPT_TAIL = 1_400, 700
FEWSHOT_CAP_CHARS = 30_000  # harness_lab/text_classification/agents/fewshot_memory.py MAX_CHARS
SERVER_CONTEXT_TOKENS = 262_144  # context length the on-prem SGLang server is launched with


def est_tokens(chars: int) -> int:
    return chars // 4


def prompt_panel(prompt: str, title: str = OFFLINE_PREVIEW, full: bool = False) -> Panel:
    """The exact prompt text; long prompts are cut in the middle unless `full`."""
    body = prompt
    if not full and len(prompt) > PROMPT_HEAD + PROMPT_TAIL:
        cut = len(prompt) - PROMPT_HEAD - PROMPT_TAIL
        body = f"{prompt[:PROMPT_HEAD]}\n\n… ({cut:,}자 생략 · 전체는 /inspect context) …\n\n{prompt[-PROMPT_TAIL:]}"
    caption = f"{len(prompt):,}자 · ≈{est_tokens(len(prompt)):,}토큰 (글자수/4 추정)"
    return Panel(Text(body), title=f"[bold]{title}", subtitle=f"[dim]{caption}", border_style="#a3a3a3", padding=(0, 1))


# --- /inspect context ---------------------------------------------------------------------------

def picks_table(picks: tuple[Pick, ...]) -> Table:
    table = Table(title="프롬프트에 넣은 예시와 고른 이유 (문제 바로 앞 = 마지막 행)", title_justify="left", expand=True)
    table.add_column("#", width=3, justify="right")
    table.add_column("이유", width=16)
    table.add_column("TF-IDF 유사도", width=13, justify="right")
    table.add_column("예시 증상", ratio=4, overflow="fold")
    table.add_column("정답", ratio=1)
    for i, p in enumerate(picks, 1):
        score = "—" if p.score is None else f"{p.score:.3f}"
        style = "bold" if p.reason.startswith("top-") else "dim"
        table.add_row(str(i), Text(p.reason, style=style), score, p.question[:150], p.label)
    return table


def inspect_medical(console: Console, system: str, symptoms: str, prompt: str, picks: tuple[Pick, ...]) -> None:
    console.print(Text.assemble(("inspect ", "dim"), (system, f"bold {ACCENT}"), ("  입력: ", "dim"), (symptoms, "italic")))
    if picks:
        console.print(picks_table(picks))
        console.print(Text("유사도는 이 블록이 프롬프트를 만들 때 쓴 _rank()의 값입니다. top-k 밖의 예시는 유사도가 아니라 "
                           "'과거 오답'이나 '라벨마다 하나씩(coverage)' 규칙으로 들어갔습니다.", style="dim"))
    else:
        console.print(Text("이 블록은 예시를 넣지 않습니다 (질문만 보냄).", style="dim"))
    console.print(prompt_panel(prompt, title="RAW PROMPT (모델에 보내는 그대로)", full=True))


def inspect_study(console: Console, ans: StudyAnswer) -> None:
    console.print(Text.assemble(("inspect ", "dim"), (ans.mode, f"bold {ACCENT}"), ("  질문: ", "dim"), (ans.question, "italic")))
    rows = [*(("note", h) for h in ans.notes), *(("rule", h) for h in ans.cues)]
    if rows:
        table = Table(title="프롬프트에 넣은 노트 단락·규칙", title_justify="left", expand=True)
        table.add_column("종류", width=6)
        table.add_column("TF-IDF 유사도", width=13, justify="right")
        table.add_column("제목", ratio=1, overflow="fold")
        table.add_column("내용", ratio=4, overflow="fold")
        for kind, h in rows:
            kind = "taught" if h.chunk.lesson and kind == "note" else kind
            table.add_row(kind, f"{h.score:.3f}", h.chunk.title, h.chunk.text[:200])
        console.print(table)
    console.print(prompt_panel(ans.prompt, title="RAW PROMPT (모델에 보내는 그대로)", full=True))


# --- /explain fewshot ---------------------------------------------------------------------------

def _per_query(name: str) -> float:
    s = lego.resolve(name).scores
    return s.mean_tokens("test") / s.n("test")


def explain_fewshot(console: Console) -> None:
    few, knn = lego.resolve("fewshot_all").scores, lego.resolve("knn_top6").scores
    few_q, knn_q, raw_q = _per_query("fewshot_all"), _per_query("knn_top6"), _per_query("no_memory")
    per_example = (few_q - raw_q) / 60
    capacity = int((SERVER_CONTEXT_TOKENS - raw_q) / per_example)
    saved = lego.token_reduction("knn_top6")
    frontier = lego.pareto_frontier("test")
    lines = [
        Text.assemble(("fewshot_all", "bold"), f": 학습 예시 60개를 질문마다 전부 넣습니다. "
                      f"test 3-seed 평균 {few.mean_accuracy():.1%}, 질문당 입력 약 {few_q:,.0f}토큰."),
        Text.assemble(("knn_top6", "bold"), f": TF-IDF로 가장 비슷한 6개만 넣습니다. "
                      f"test 3-seed 평균 {knn.mean_accuracy():.1%}, 질문당 약 {knn_q:,.0f}토큰 (입력 토큰 {saved:.1%} 절감)."),
        Text(),
        Text("왜 fewshot_all은 장난감 기준선인가", style="bold"),
        Text(f"• 프롬프트가 기억 크기에 비례해 커집니다. 예시 1개당 약 {per_example:.0f}토큰 "
             f"(= (fewshot {few_q:,.0f} − 기억 없음 {raw_q:,.0f}) / 60)."),
        Text(f"• 선형 외삽하면 예시 약 {capacity:,}개에서 서버 컨텍스트 {SERVER_CONTEXT_TOKENS:,}토큰을 넘습니다. "
             f"실제 하네스 코드는 그 전에 {FEWSHOT_CAP_CHARS:,}자에서 예시를 잘라냅니다 (fewshot_memory.py MAX_CHARS)."),
        Text("• 질문마다 같은 60개를 다시 읽으므로 비용·지연이 기억 크기에 비례합니다. "
             "knn_top6는 기억이 커져도 프롬프트는 예시 6개로 일정하고, 검색(로컬 TF-IDF) 비용만 늘어납니다."),
        Text(),
        Text("통계적으로 보면", style="bold"),
        Text("• 3-seed 사전 등록 분석(analysis/multiseed)에서 fewshot_all과 knn_top6의 정확도 차이는 유의하지 않았습니다. "
             "같은 조건 반복 실행에서도 50문항 중 4문항의 정오가 바뀌므로, 1~2%p 차이는 잡음 범위로 읽어야 합니다."),
        Text(f"• 파레토 경계 (test 평균 정확도 × 입력 토큰, 같은 토큰 집계 블록만): {' → '.join(frontier)}. "
             "어느 블록도 이들보다 정확도가 높으면서 토큰이 적지 않습니다."),
        Text("• 이 수치는 Qwen3.8-Flash-Next / Symptom2Disease에서 잰 값입니다. 다른 모델·과제로 일반화하지 마세요.", style="dim"),
    ]
    console.print(Panel(Group(*lines), title="[bold]/explain fewshot", border_style=ACCENT, padding=(1, 2)))


EXPLAIN_TOPICS = {"fewshot": explain_fewshot}


# --- study task ---------------------------------------------------------------------------------

def modes_table(current: str) -> Table:
    table = Table(title="학습 모드 (study task)", title_justify="left")
    table.add_column("", width=2)
    table.add_column("모드")
    table.add_column("넣는 것")
    for m in MODES.values():
        table.add_row("▶" if m.name == current else "", Text(m.name, style="bold"), m.summary)
    return table


def mode_switched(console: Console, mode: StudyMode) -> None:
    console.print(Text.assemble(("[학습 모드] ", "dim"), (f" {mode.name} ", "bold white on #0f766e"),
                                (f" {mode.title} · {mode.summary}", "dim")))


def _size(ans: StudyAnswer) -> str:
    chars = len(ans.prompt)
    if ans.prompt_tokens:
        return f"{chars:,}자 · {ans.prompt_tokens:,}토큰 (서버 집계)"
    return f"{chars:,}자 · ≈{est_tokens(chars):,}토큰 (글자수/4 추정)"


def _saved(ans: StudyAnswer) -> str:
    saved = ans.context_saved
    return "" if saved is None or ans.mode == "no_memory" else f" · 노트 전체 대비 컨텍스트 {saved:.0%} 절감"


def study_answer(console: Console, ans: StudyAnswer, model: str = "") -> None:
    if ans.source == "live":
        console.print(Panel(Text(ans.answer or "(빈 응답)"), title=f"[bold]Answer · {ans.mode}", border_style=ACCENT, padding=(1, 2)))
    if ans.notes or ans.cues:
        table = Table(title="검색된 기억", title_justify="left", expand=True)
        table.add_column("종류", width=6)
        table.add_column("유사도", width=7, justify="right")
        table.add_column("제목", ratio=1, overflow="fold")
        table.add_column("앞부분", ratio=4, overflow="fold")
        for h in ans.notes:
            table.add_row("taught" if h.chunk.lesson else "note", f"{h.score:.3f}", h.chunk.title, h.chunk.text[:110])
        for h in ans.cues:
            table.add_row("rule", f"{h.score:.3f}", h.chunk.title, h.chunk.text[:110])
        console.print(table)
    elif ans.mode != "no_memory":
        console.print(Text("검색된 노트·규칙이 없습니다 (/load 로 노트를, /record 로 규칙을 넣으세요).", style="dim"))
    if ans.source == "preview":
        console.print(prompt_panel(ans.prompt))
    detail = f"프롬프트 {_size(ans)}{_saved(ans)}"
    if ans.source == "live":
        detail += f" · {ans.seconds:.1f}초" + (f" · {model}" if model else "")
    else:
        detail += " · 모델 호출 없음 (미리보기)"
    console.print(Text(detail, style="dim"))


def study_comparison(console: Console, left: StudyAnswer, right: StudyAnswer) -> None:
    table = Table(expand=True, show_lines=True)
    for ans in (left, right):
        table.add_column(f"[bold white on #0f766e] {ans.mode} [/] [dim]{MODES[ans.mode].title}[/]", ratio=1)

    def cell(ans: StudyAnswer) -> Group:
        lines: list = [
            Text.assemble(("프롬프트 ", "dim"), (_size(ans), "bold")),
            Text.assemble(("노트 단락 ", "dim"), (f"{len(ans.notes)}개", "bold"), (" · 규칙 ", "dim"), (f"{len(ans.cues)}개", "bold")),
        ]
        if ans.context_saved is not None and ans.mode != "no_memory":
            lines.append(Text(f"노트 전체 대비 컨텍스트 {ans.context_saved:.0%} 절감", style="dim"))
        lines.append(Text())
        if ans.source == "live":
            text = ans.answer if len(ans.answer) <= 600 else ans.answer[:600] + "…"
            lines.append(Text(text or "(빈 응답)"))
        else:
            lines.append(Text(" PREVIEW ", style="bold black on #a3a3a3"))
            lines.append(Text("모델 호출 없음: 프롬프트 차이만 비교합니다.", style="dim"))
        return Group(*lines)

    table.add_row(cell(left), cell(right))
    console.print(table)


def study_status(console: Console, notes: dict[str, int], chunks: int, lessons: int, persistent: bool) -> None:
    table = Table(title="study task 기억", title_justify="left")
    table.add_column("계층")
    table.add_column("내용", justify="right")
    table.add_column("위치", overflow="fold")
    table.add_row("불러온 노트 (세션)", f"{len(notes)}개 파일 · {chunks}단락 · {sum(notes.values()):,}자", ", ".join(notes) or "—")
    where = str(config.harness_home() / "scratchpad.json") if persistent else "메모리 (저장 안 함)"
    table.add_row("가르친 규칙 (/record)", f"{lessons}건", where)
    console.print(table)


# --- /profile -----------------------------------------------------------------------------------

def profile_table(profile: dict[str, str], path: str) -> Table:
    table = Table(title=f"학습자 프로필 ({path})", title_justify="left")
    table.add_column("필드")
    table.add_column("값", overflow="fold")
    table.add_column("설명", style="dim")
    for key in PROFILE_KEYS:
        table.add_row(key, profile.get(key, "—"), PROFILE_HELP[key])
    return table
