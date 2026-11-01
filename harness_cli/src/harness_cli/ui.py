"""Rich renderers: banner, prediction boxes, memory info."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .engine import MemoryStats, Prediction

ACCENT = "#c2410c"


def banner(console: Console, settings: Settings) -> None:
    title = Text("harness-cli", style=f"bold {ACCENT}")
    body = Text.assemble(
        ("모델은 그대로, 하네스만 AI가 고쳤다\n", "bold"),
        ("system ", "dim"), (settings.system, "bold"), ("   model ", "dim"), (settings.model.split("/")[-1], "bold"),
        ("\n증상을 영어로 입력하세요. ", "dim"), ("/help", ACCENT), (" 명령어 · ", "dim"), ("/exit", ACCENT), (" 종료", "dim"),
    )
    console.print(Panel(body, title=title, border_style=ACCENT, padding=(1, 2)))


def prediction(console: Console, pred: Prediction, show_evidence: bool = True) -> None:
    head = Text.assemble(("진단  ", "dim"), (pred.diagnosis or "(답 추출 실패)", f"bold {ACCENT}"))
    parts: list = [head]
    if pred.reasoning:
        parts.append(Text(pred.reasoning, style="dim"))
    console.print(Panel(Group(*parts), title="[bold]Diagnosis", border_style=ACCENT, padding=(1, 2)))

    if pred.warnings:
        warn = Text("\n".join(f"• {w}" for w in pred.warnings))
        console.print(Panel(warn, title="[bold yellow][오답 대조 경고]", border_style="yellow", padding=(0, 2)))

    if show_evidence and pred.evidence:
        table = Table(title="하네스가 고른 기억", title_justify="left", show_lines=False, expand=True)
        table.add_column("", width=2)
        table.add_column("예시 증상", ratio=4, overflow="fold")
        table.add_column("정답", ratio=1)
        for ev in pred.evidence:
            mark = Text("⚠", style="yellow") if ev.past_mistake else Text("·", style="dim")
            table.add_row(mark, ev.question[:160] + ("…" if len(ev.question) > 160 else ""), ev.label)
        console.print(table)
        console.print(Text("⚠ = 학습 중 모델이 틀렸던 예시 (오답 대조용으로 다시 보여 줌)", style="dim"))

    console.print(Text(f"프롬프트 {pred.prompt_chars:,}자 · {pred.seconds:.1f}초", style="dim"))


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
