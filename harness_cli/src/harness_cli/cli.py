"""harness-cli entry point: REPL by default, plus `predict` and `info` subcommands."""

from __future__ import annotations

import json
from dataclasses import asdict

import typer
from rich.console import Console
from rich.prompt import Prompt

from . import ui
from .commands import Command, CommandRegistry, route_input
from .config import Settings
from .engine import HarnessEngine

app = typer.Typer(add_completion=False, help="Meta-Harness memory system on the on-prem Qwen, from your terminal.")
console = Console()


def _engine() -> HarnessEngine:
    settings = Settings.from_env()
    try:
        with console.status("하네스와 기억을 불러오는 중…"):
            return HarnessEngine(settings)
    except Exception as exc:  # surface setup errors as one readable line
        console.print(f"[red]시작 실패:[/] {exc}")
        raise typer.Exit(1) from exc


def _predict(engine: HarnessEngine, symptoms: str):
    try:
        with console.status(f"{engine.settings.model.split('/')[-1]}에 묻는 중…"):
            return engine.predict(symptoms)
    except Exception as exc:
        console.print(f"[red]예측 실패:[/] {exc}\n[dim]엔드포인트: {engine.settings.api_base}[/]")
        return None


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """인자 없이 실행하면 대화형 모드로 들어갑니다."""
    if ctx.invoked_subcommand is None:
        chat()


@app.command()
def predict(
    symptoms: str = typer.Argument(..., help="환자 증상 (영어 문장)"),
    json_out: bool = typer.Option(False, "--json", help="결과를 JSON으로 출력"),
    evidence: bool = typer.Option(True, "--evidence/--no-evidence", help="하네스가 고른 예시 표시"),
) -> None:
    """증상 한 건을 진단합니다."""
    engine = _engine()
    pred = _predict(engine, symptoms)
    if pred is None:
        raise typer.Exit(1)
    if json_out:
        typer.echo(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
    else:
        ui.prediction(console, pred, show_evidence=evidence)


@app.command()
def info() -> None:
    """기억 상태(예시 수, 오답 통계)를 보여 줍니다."""
    engine = _engine()
    ui.info(console, engine.stats(), engine.settings)


@app.command()
def chat() -> None:
    """대화형 모드 (REPL)."""
    engine = _engine()
    ui.banner(console, engine.settings)
    last = {"pred": None}

    def show_help() -> bool:
        for cmd in registry.commands:
            names = ", ".join(f"/{n}" for n in (cmd.name, *cmd.aliases))
            console.print(f"  [bold]{names}[/]  {cmd.help}")
        return True

    def show_info() -> bool:
        ui.info(console, engine.stats(), engine.settings)
        return True

    def show_why() -> bool:
        if last["pred"] is None:
            console.print("[dim]아직 진단한 증상이 없습니다.[/]")
        else:
            ui.prediction(console, last["pred"], show_evidence=True)
        return True

    def clear() -> bool:
        console.clear()
        ui.banner(console, engine.settings)
        return True

    registry = CommandRegistry(
        [
            Command("help", "명령어 목록", show_help, ("h",)),
            Command("info", "기억 상태와 오답 통계", show_info),
            Command("why", "직전 진단의 근거(고른 예시) 다시 보기", show_why),
            Command("clear", "화면 지우기", clear),
            Command("exit", "종료", lambda: False, ("quit", "q")),
        ]
    )

    while True:
        try:
            line = Prompt.ask(f"[bold {ui.ACCENT}]증상 >[/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        route = route_input(line, registry)
        if route.kind == "empty":
            continue
        if route.kind == "unknown":
            console.print(f"[dim]알 수 없는 명령어: {route.text} — /help 를 입력해 보세요.[/]")
            continue
        if route.kind == "command":
            if not route.command.handler():
                break
            continue
        pred = _predict(engine, route.text)
        if pred is not None:
            last["pred"] = pred
            ui.prediction(console, pred, show_evidence=False)
            console.print("[dim]근거 예시: /why[/]")
    console.print("[dim]종료합니다.[/]")
