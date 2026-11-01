"""Rich renderers: banner and status bar, prediction boxes, side-by-side comparison, memory info."""

from __future__ import annotations

from urllib.parse import urlparse

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import lego
from .config import Settings
from .engine import MemoryStats, Prediction
from .presets import PRESETS, Preset
from .scratchpad import PadRow, Record
from .study import TALK_URL

ACCENT = "#c2410c"
REPLAY_BADGE = " DEMO MODE: CACHED REPLAY "

# ANSI Shadow glyphs, one list of six rows per character.
_GLYPHS = {
    "H": ["██╗  ██╗", "██║  ██║", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "-": ["      ", "      ", "█████╗", "╚════╝", "      ", "      "],
    "C": [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    "L": ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
}
TITLE = "HARNESS-CLI"


def ascii_title(word: str = TITLE) -> list[str]:
    return ["".join(_GLYPHS[ch][row] for ch in word) for row in range(6)]


def _host(api_base: str) -> str:
    return (urlparse(api_base).hostname or api_base).split(".")[0]


def badges(settings: Settings) -> Text:
    saving = lego.token_reduction("knn_top6", "fewshot_all")
    items = [
        ("ON-PREM QWEN3.8", "bold white on #1e3a8a"),
        ("SGLANG FP4", "bold white on #065f46"),
        (_host(settings.api_base).upper(), "bold white on #374151"),
        (f"knn_top6: -{saving:.1%} INPUT TOKENS vs fewshot_all (test)", f"bold white on {ACCENT}"),
    ]
    text = Text()
    for label, style in items:
        text.append(f" {label} ", style=style)
        text.append(" ")
    return text


# --- lego blocks ---------------------------------------------------------------------------------

BLOCK_COLORS = {
    "no_memory": "#6b7280",
    "fewshot_all": "#1d4ed8",
    "knn_top6": "#047857",
    "contrast_online": "#b45309",
    "cue_bank": "#7c3aed",
    "cue_bank_verified": "#0e7490",
    "reflexion_notes": "#be185d",
    "confusion_contrast": ACCENT,
}


def accuracy_line(block: lego.Block) -> str:
    """Headline test accuracy with its split, n and number of runs, as recorded."""
    s = block.scores
    if block.multi_seed:
        return f"test {s.mean_accuracy('test'):.1%} · {len(s.seeds())}-seed 평균 · n={s.n('test')}"
    return f"test {s.mean_accuracy('test'):.1%} · seed 42 단일 실행 · n={s.n('test')} (val {s.mean_accuracy('val'):.1%}, n={s.n('val')})"


def tokens_line(block: lego.Block) -> str:
    s = block.scores
    per_q = s.mean_tokens("test") / s.n("test")
    if block.multi_seed:
        return f"문항당 입력 ≈{per_q:,.0f} 토큰 (test, {s.token_metric})"
    return f"문항당 입력 ≈{per_q:,.0f} 토큰 ({s.token_metric}: 다른 블록과 비교 불가)"


def memory_line(block: lego.Block) -> str:
    seed42 = block.scores.seed_accuracy(42, "test")
    if block.multi_seed and seed42 is not None:
        return f"불러온 기억: seed 42 memory.json (seed 42 test {seed42:.1%})"
    return "불러온 기억: seed 42 memory.json"


def lego_badge(block: lego.Block, current: bool = False) -> Table:
    """A coloured brick (studs on top) with the block's name, next to its recorded numbers."""
    color = BLOCK_COLORS.get(block.name, "#374151")
    width = max(len(block.name), 18) + 2
    studs = "".join(" ▄▄ " for _ in range(width // 4)).ljust(width)
    brick = Text.assemble(
        (studs + "\n", color),
        (f"{block.name:^{width}}\n", f"bold white on {color}"),
        (f"{block.title:^{width - len(block.title)}}", f"white on {color}"),
    )
    info = Text.assemble(
        ("● 지금 끼운 블록  " if current else "", f"bold {color}"),
        (block.summary + "\n", ""),
        (accuracy_line(block) + "\n", "bold"),
        (tokens_line(block) + "\n", "dim"),
        (memory_line(block), "dim"),
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column()
    grid.add_row(brick, info)
    return grid


def blocks_table(current: str) -> Table:
    table = Table(title="레고 하네스 블록 8개", title_justify="left", expand=True)
    table.add_column("", width=2)
    table.add_column("이름", no_wrap=True)
    table.add_column("넣는 것", ratio=3, overflow="fold")
    table.add_column("test 정확도", justify="right", no_wrap=True)
    table.add_column("기록", no_wrap=True)
    table.add_column("문항당 입력 토큰", justify="right", no_wrap=True)
    for block in lego.BLOCKS.values():
        s = block.scores
        runs = f"{len(s.seeds())}-seed 평균" if block.multi_seed else "seed 42 단일"
        per_q = f"≈{s.mean_tokens('test') / s.n('test'):,.0f}" + ("" if block.multi_seed else " *")
        mark = Text("●", style=BLOCK_COLORS.get(block.name, "")) if block.name == current else Text("")
        table.add_row(mark, Text(block.name, style=f"bold {BLOCK_COLORS.get(block.name, '')}"), block.summary,
                      f"{s.mean_accuracy('test'):.1%}", runs, per_q)
    table.caption = (
        "test n=212 · 출처 analysis/multiseed/out/accuracy_tokens.csv · "
        "* confusion_contrast는 중간과제 단일 실행 (출처 logs/midterm/.../{test,val}.json), 토큰 지표(llm_input_tokens)가 달라 직접 비교 불가"
    )
    return table


def switched(console: Console, block: lego.Block) -> None:
    console.print(lego_badge(block, current=True))


# --- session memory ------------------------------------------------------------------------------

def memory_table(rows: tuple[PadRow, ...]) -> Table:
    table = Table(title="기억 상태 (원본 동결 + 세션 추가)", title_justify="left", expand=True)
    table.add_column("블록", no_wrap=True)
    table.add_column("원본 예시 (동결)", justify="right")
    table.add_column("세션 추가", justify="right")
    table.add_column("합계", justify="right")
    table.add_column("검색에 반영", no_wrap=True)
    for row in rows:
        reflected = "예" if row.affected else "아니오 (예시를 쓰지 않음)"
        style = "bold" if row.session else ""
        table.add_row(row.system, str(row.frozen), str(row.session), str(row.frozen + row.session), reflected, style=style)
    table.caption = "원본 기억 파일은 읽기만 합니다. 세션 추가분은 이 터미널을 닫거나 /memory reset 하면 사라집니다."
    return table


def session_records(records: tuple[Record, ...]) -> Table:
    table = Table(title=f"세션에 추가한 예시 {len(records)}건", title_justify="left", expand=True)
    table.add_column("#", width=3, justify="right")
    table.add_column("증상", ratio=4, overflow="fold")
    table.add_column("질병", ratio=1)
    for i, r in enumerate(records, start=1):
        table.add_row(str(i), r.symptoms, r.disease)
    return table


def status_bar(stats: MemoryStats, live: bool) -> Table:
    bar = Table.grid(padding=(0, 4))
    bar.add_column(no_wrap=True)
    bar.add_column(no_wrap=True)
    mode = Text(" LIVE ", style="bold black on #22c55e") if live else Text(REPLAY_BADGE, style="bold black on yellow")
    bar.add_row(
        Text.assemble(("system ", "dim"), (stats.system, "bold")),
        Text.assemble(("서버 ", "dim"), mode),
    )
    examples = Text.assemble(("기억된 예시 ", "dim"), (f"{stats.examples}개", "bold"), (f" · 질병 {len(stats.labels)}종", "dim"))
    if stats.session_examples:
        examples.append(f" (세션 추가 {stats.session_examples}개 포함)", style="yellow")
    mistakes = Text.assemble(("기록된 오답 ", "dim"), (f"{stats.mistakes}건", "bold"))
    if stats.system == "confusion_contrast" and not stats.mistakes:
        mistakes.append(" (오프라인 학습 경로 결함)", style="dim")
    bar.add_row(examples, mistakes)
    return bar


def banner(console: Console, settings: Settings, stats: MemoryStats | None = None, live: bool = True,
           block: lego.Block | None = None) -> None:
    rows = ascii_title()
    if console.width >= len(rows[0]) + 6:
        console.print(Text("\n".join(rows), style=f"bold {ACCENT}"))
    else:  # narrow terminal or large projector font: the art would wrap and garble
        console.print(Text(TITLE, style=f"bold {ACCENT}"))
    body: list = [
        Text("서강대학교 빅데이터종합설계(캡스톤디자인) 중간발표 라이브 데모 · v0.2 레고 하네스", style="bold"),
        Text("모델은 그대로, 프롬프트에 넣는 기억(하네스)만 갈아 끼운다", style="dim"),
        Text(f"발표 자료: {TALK_URL}", style="dim"),
        Text(),
        badges(settings),
    ]
    if block is not None:
        body += [Text(), lego_badge(block, current=True)]
    if stats is not None:
        body += [Text(), status_bar(stats, live)]
    body += [
        Text(),
        Text.assemble(
            ("증상을 영어로 입력하거나 ", "dim"), ("1", ACCENT), (" ", "dim"), ("2", ACCENT), (" ", "dim"), ("3", ACCENT),
            (" 프리셋 · ", "dim"), ("/harness", ACCENT), (" 블록 교체 · ", "dim"),
            ("/compare no_memory knn_top6 1", ACCENT), (" 좌우 비교 · ", "dim"), ("/study", ACCENT), (" · ", "dim"),
            ("/help", ACCENT), (" · ", "dim"), ("/exit", ACCENT),
        ),
    ]
    console.print(Panel(Group(*body), border_style=ACCENT, padding=(1, 2)))


def presets_menu(console: Console) -> None:
    table = Table(title="데모 프리셋", title_justify="left", show_header=False, expand=True)
    table.add_column(width=3)
    table.add_column(ratio=2)
    table.add_column(ratio=5, overflow="fold")
    for p in PRESETS:
        table.add_row(Text(f"[{p.key}]", style=f"bold {ACCENT}"), p.title, Text(p.symptoms, style="dim"))
    console.print(table)
    console.print(Text(
        "숫자만 입력하면 지금 끼운 블록으로 진단 · /compare N 은 기억 없음과 비교 · "
        "/compare <블록1> <블록2> N 은 두 블록 비교 · /harness 로 블록 교체", style="dim"))


def preset_header(console: Console, preset: Preset) -> None:
    body = Text.assemble(
        (f"[{preset.key}] {preset.title}\n", "bold"),
        ("경계 질병: ", "dim"), (" vs ".join(preset.contested), "bold"),
        ("\n\n", ""), (preset.symptoms, "italic"),
    )
    console.print(Panel(body, title="[bold]입력 증상", border_style="#64748b", padding=(0, 2)))


def notice(console: Console, note: str | None) -> None:
    if note:
        console.print(Text(note, style="yellow"))


def prompt_size(pred: Prediction) -> str:
    """Server-counted tokens when known; otherwise a labelled chars/4 estimate."""
    if pred.prompt_tokens:
        when = " (녹화 당시 서버 집계)" if pred.source == "replay" else " (서버 집계)"
        return f"{pred.prompt_chars:,}자 · {pred.prompt_tokens:,}토큰{when}"
    return f"{pred.prompt_chars:,}자 · ≈{pred.prompt_chars // 4:,}토큰 (글자수/4 추정)"


def latency(pred: Prediction) -> str:
    if pred.source == "preview":
        return "모델 호출 없음 (미리보기)"
    if pred.source == "replay":
        return f"{pred.seconds:.1f}초 (녹화 당시)"
    if pred.prompt_tokens is None:  # litellm answered from its local response cache
        return f"{pred.seconds:.1f}초 (로컬 응답 캐시 적중 · 측정값 아님)"
    return f"{pred.seconds:.1f}초"


def _diagnosis(pred: Prediction) -> str:
    if pred.source == "preview":
        return "(미리보기: 진단 없음)"
    return pred.diagnosis or "(답 추출 실패)"


def _source_line(pred: Prediction) -> Text:
    detail = f"프롬프트 {prompt_size(pred)} · {latency(pred)}"
    if pred.source == "replay":
        return Text.assemble((REPLAY_BADGE, "bold black on yellow"), (f"  녹화된 실제 서버 응답 · {detail}", "dim"))
    if pred.source == "preview":
        return Text.assemble((" PREVIEW ", "bold black on #a3a3a3"), (f"  {detail}", "dim"))
    return Text(detail, style="dim")


def prediction(console: Console, pred: Prediction, show_evidence: bool = True) -> None:
    head = Text.assemble(("진단  ", "dim"), (_diagnosis(pred), f"bold {ACCENT}"), (f"   [{pred.system}]", "dim"))
    parts: list = [head]
    if pred.reasoning:
        parts.append(Text(pred.reasoning, style="dim"))
    console.print(Panel(Group(*parts), title="[bold]Diagnosis", border_style=ACCENT, padding=(1, 2)))

    if pred.warnings:
        warn = Text("\n".join(f"• {w}" for w in pred.warnings))
        console.print(Panel(warn, title="[bold yellow][오답 대조 경고]", border_style="yellow", padding=(0, 2)))
    elif show_evidence and pred.notes:
        notes = Text("\n".join(f"• {n}" for n in pred.notes))
        console.print(Panel(notes, title=f"[bold]프롬프트에 넣은 메모: {pred.notes_title}", border_style="#64748b", padding=(0, 2)))

    if show_evidence and pred.evidence:
        console.print(evidence_table(pred))
        legend = "⚠ = 학습 중 모델이 틀렸던 예시 (오답 대조용으로 다시 보여 줌)"
        if any(ev.session for ev in pred.evidence):
            legend += " · ★ = /record 로 이번 세션에 넣은 예시"
        console.print(Text(legend, style="dim"))

    console.print(_source_line(pred))


def _mark(ev) -> Text:
    if ev.session:
        return Text("★", style="bold green")
    return Text("⚠", style="yellow") if ev.past_mistake else Text("·", style="dim")


def evidence_table(pred: Prediction, width: int = 160) -> Table:
    table = Table(title="하네스가 고른 기억", title_justify="left", show_lines=False, expand=True)
    table.add_column("", width=2)
    table.add_column("예시 증상", ratio=4, overflow="fold")
    table.add_column("정답", ratio=1)
    for ev in pred.evidence:
        table.add_row(_mark(ev), ev.question[:width] + ("…" if len(ev.question) > width else ""), ev.label)
    return table


TOP_EVIDENCE = 3


def nearest_evidence(pred: Prediction, n: int = TOP_EVIDENCE) -> list:
    """Every agent puts its nearest example last, right before the problem; show those first."""
    return list(reversed(pred.evidence))[:n]


def _compare_cell(pred: Prediction | None, note: str | None) -> Group:
    if pred is None:
        return Group(Text("응답 없음", style="bold red"), Text(note or "", style="dim"))
    lines: list = [
        Text.assemble(("진단  ", "dim"), (_diagnosis(pred), f"bold {ACCENT}")),
        Text(),
        Text.assemble(("프롬프트 ", "dim"), (prompt_size(pred), "bold")),
        Text.assemble(("응답     ", "dim"), (latency(pred), "bold")),
        Text.assemble(("고른 예시 ", "dim"), (f"{len(pred.evidence)}개", "bold"),
                      (f" · {pred.notes_title} {len(pred.notes)}줄" if pred.notes else "", "dim")),
    ]
    if pred.source == "replay":
        lines.append(Text(REPLAY_BADGE, style="bold black on yellow"))
    elif pred.source == "preview":
        lines.append(Text(" PREVIEW ", style="bold black on #a3a3a3"))
    if note:
        lines.append(Text(note, style="yellow"))
    if pred.reasoning:
        reason = pred.reasoning if len(pred.reasoning) <= 220 else pred.reasoning[:220] + "…"
        lines += [Text(), Text(reason, style="dim")]
    if pred.evidence:
        closest = nearest_evidence(pred)
        lines += [Text(), Text(f"문제 바로 앞 예시 {len(closest)}개 (정답 라벨 · 증상 앞부분)", style="dim")]
        for ev in closest:
            lines.append(Text.assemble(_mark(ev), (f" {ev.label}", "bold"), (f"  {ev.question[:40]}…", "dim")))
    return Group(*lines)


Outcome = tuple[Prediction | None, str | None]


def _header(name: str) -> str:
    block = lego.resolve(name)
    color = BLOCK_COLORS.get(block.name, ACCENT)
    return f"[bold white on {color}] {block.name} [/] [dim]{block.title}[/]\n[dim]{accuracy_line(block)}[/]"


def comparison(console: Console, left: tuple[str, Outcome], right: tuple[str, Outcome]) -> None:
    """Two blocks side by side on the same symptoms, then the scored numbers behind them."""
    table = Table(expand=True, show_lines=True, border_style="#64748b")
    table.add_column(_header(left[0]), ratio=1)
    table.add_column(_header(right[0]), ratio=1)
    table.add_row(_compare_cell(*left[1]), _compare_cell(*right[1]))
    console.print(table)
    console.print(logged_summary((left[0], right[0])))


def logged_summary(names: tuple[str, ...] = ("no_memory", "knn_top6")) -> Table:
    """The scored numbers behind a comparison, so a single live case is not read as the result."""
    table = Table(title="기록된 평가 결과", title_justify="left", expand=True)
    table.add_column("system")
    table.add_column("test 정확도 (n=212)", justify="right")
    table.add_column("기록", no_wrap=True)
    table.add_column("seed 42 test", justify="right")
    table.add_column("문항당 입력 토큰", justify="right")
    shown = list(dict.fromkeys(lego.resolve(n).name for n in names))
    if "fewshot_all" not in shown:
        shown.append("fewshot_all")  # reference point: all 60 examples
    for name in shown:
        block = lego.BLOCKS[name]
        s = block.scores
        runs = f"{len(s.seeds())}-seed 평균" if block.multi_seed else "단일 실행"
        per_q = f"≈{s.mean_tokens('test') / s.n('test'):,.0f}" + ("" if block.multi_seed else " *")
        table.add_row(f"{name} ({block.title})", f"{s.mean_accuracy('test'):.1%}", runs,
                      f"{s.seed_accuracy(42, 'test'):.1%}", per_q)
    table.caption = (
        "시연 한 건은 예시일 뿐이고, 시스템 간 차이의 근거는 위 평가 결과입니다. "
        "출처: analysis/multiseed/out/accuracy_tokens.csv"
        + (" · * 중간과제 단일 실행, 토큰 지표 다름" if any(not lego.BLOCKS[n].multi_seed for n in shown) else "")
    )
    return table


def info(console: Console, stats: MemoryStats, settings: Settings) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row(Text("system", style="dim"), Text(stats.system, style="bold"))
    summary.add_row(Text("memory", style="dim"), Text(str(settings.memory_path), overflow="fold"))
    summary.add_row(Text("model", style="dim"), Text(f"{settings.model} @ {settings.api_base}"))
    summary.add_row(Text("기억된 예시", style="dim"), Text(f"{stats.examples}개 · 질병 {len(stats.labels)}종"))
    summary.add_row(Text("학습 중 오답", style="dim"), Text(f"{stats.mistakes}건"))
    console.print(Panel(summary, title="[bold]Memory", border_style=ACCENT, padding=(1, 2)))

    if stats.confusions:
        table = Table(title="자주 헷갈린 쌍 (정답 → 오답)", title_justify="left")
        table.add_column("정답")
        table.add_column("모델의 오답")
        table.add_column("횟수", justify="right")
        for true, pred, count in stats.confusions[:10]:
            table.add_row(true, pred or "(빈 답)", str(count))
        console.print(table)

    labels = Table(title="질병별 예시 수", title_justify="left")
    labels.add_column("질병")
    labels.add_column("예시", justify="right")
    for label, count in stats.labels:
        labels.add_row(label, str(count))
    console.print(labels)
