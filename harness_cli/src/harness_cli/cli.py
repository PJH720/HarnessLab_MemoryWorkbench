"""harness-cli entry point: REPL by default, plus `predict`, `demo`, `compare`, `info` and `record`."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from . import config, docqa, lego, panels, study, ui
from .commands import (STUDY_RECORD_USAGE, Command, CommandRegistry, UsageError, parse_compare, parse_record,
                       parse_study_compare, route_input)
from .config import Settings
from .docqa import NotesError, StudyAnswer, StudyBuddy
from .engine import (OFFLINE_PREVIEW, EnginePool, Prediction, ReplayCache, SafeEngine, discover, list_models,
                     run_with_deadline)
from .home import HarnessHome, HomeError, has_cjk
from .presets import PRESETS, Preset, find_preset
from .scratchpad import Scratchpad, ScratchpadError

app = typer.Typer(add_completion=False, help="Modular lego-harness CLI: swap memory systems, diagnose, and study your own notes.")
console = Console()
OPTIONS = {"demo": False, "deadline": 60.0}

Outcome = tuple[Prediction | None, str | None]


@dataclass
class Session:
    """Engines for every lego block (built on first use) and the block currently plugged in."""

    pool: EnginePool
    current: str
    home: HarnessHome | None = None  # None in --demo: nothing persisted is read or written
    buddy: StudyBuddy = field(default_factory=StudyBuddy)
    task: str = "medical"  # "medical" (Symptom2Disease benchmark) | "study" (your notes)
    mode: str = "knn_retrieval"  # study-task memory mode
    last: dict[str, str] = field(default_factory=dict)  # task -> last query, for /inspect

    @property
    def profile(self) -> dict[str, str]:
        return self.home.profile() if self.home else {}

    @property
    def settings(self) -> Settings:
        return self.pool.settings

    @property
    def cache(self) -> ReplayCache:
        return self.pool.cache

    @property
    def scratchpad(self) -> Scratchpad:
        return self.pool.scratchpad

    @property
    def live(self) -> bool:
        return self.pool.live

    @property
    def harness(self) -> SafeEngine:
        return self.pool.get(self.current)

    def get(self, name: str) -> SafeEngine:
        """Engine for a block name or alias; raises lego.UnknownHarness."""
        return self.pool.get(name)

    def switch(self, name: str) -> lego.Block:
        """Plug another block in; it is loaded now so a bad memory file fails here, not mid-demo."""
        block = lego.resolve(name)
        self.pool.get(block.name)
        self.current = block.name
        return block

    def predict(self, name: str, symptoms: str) -> Outcome:
        return self.pool.predict(name, symptoms)

    def raw(self) -> SafeEngine:
        return self.pool.get("no_memory")


def _home() -> HarnessHome | None:
    """~/.harness, unless --demo (the demo must replay the frozen memories, untouched by saved records)."""
    if OPTIONS["demo"]:
        return None
    try:
        return HarnessHome()
    except HomeError as exc:
        console.print(f"[yellow]{exc} → 이번 세션은 저장하지 않습니다.[/]")
        return None


def _session() -> Session:
    settings = Settings.from_env()
    home = _home()
    try:
        with console.status("모델 서버를 찾고 하네스와 기억을 불러오는 중…"):
            found = None if OPTIONS["demo"] else discover(settings)
            live = found is not None
            settings = found or settings
            pool = EnginePool(settings, ReplayCache(), live, OPTIONS["deadline"], scratchpad=Scratchpad(home))
            session = Session(pool, lego.resolve(settings.system).name, home, StudyBuddy(home))
            session.harness  # load the starting block now
    except Exception as exc:  # surface setup errors as one readable line
        console.print(f"[red]시작 실패:[/] {exc}")
        raise typer.Exit(1) from exc
    if not live and not OPTIONS["demo"]:
        tried = ", ".join(config.api_base_candidates())
        console.print(f"[yellow]{config.NO_SERVER_MESSAGE}[/]")
        console.print(f"[dim]시도한 주소: {tried}. 설정: harness-cli config --api-base <url> --model <model>. "
                      "그동안 새 입력은 프롬프트 미리보기(검색된 예시 + 프롬프트 원문), 프리셋 1·2·3은 녹화 응답으로 보여 줍니다.[/]")
    elif live:
        console.print(f"[dim]모델 서버: {settings.api_base} · {settings.model}[/]")
    if live and not settings.scored_model:
        console.print(f"[yellow]지금 모델은 {settings.model} 입니다. 블록 배지의 정확도·토큰은 "
                      f"{config.ONPREM_MODEL.split('/')[-1]}에서 잰 값이라 이 모델의 성능이 아닙니다.[/]")
    if session.scratchpad.records:
        console.print(f"[dim]~/.harness 에서 세션 기억 {len(session.scratchpad.records)}건을 불러왔습니다 (/memory status).[/]")
    return session


def _banner(session: Session) -> None:
    engine = session.harness.engine
    ui.banner(console, engine.settings, engine.stats(), session.live, lego.resolve(session.current))


def _ask(session: Session, symptoms: str) -> Outcome:
    if has_cjk(symptoms):
        return None, "증상은 영어로 입력해 주세요 (모델 프롬프트는 영어만 씁니다)."
    session.last["medical"] = symptoms
    model = session.harness.engine.settings.model.split("/")[-1]
    with console.status(f"{model}에 묻는 중…" if session.live else "녹화된 응답을 불러오는 중…"):
        return session.predict(session.current, symptoms)


def _show(outcome: Outcome, show_evidence: bool) -> Prediction | None:
    pred, note = outcome
    ui.notice(console, note)
    if pred is not None:
        preview = pred.source == "preview"
        ui.prediction(console, pred, show_evidence=show_evidence or preview)
        if preview and pred.prompt:
            console.print(panels.prompt_panel(pred.prompt))
    return pred


def _in_parallel(*jobs: Callable[[], Outcome]) -> list[Outcome]:
    """Both systems answer at once, so the audience waits for one call, not two."""
    results: list[Outcome] = [(None, "응답 없음")] * len(jobs)

    def run(i: int, job: Callable[[], Outcome]) -> None:
        results[i] = job()

    threads = [threading.Thread(target=run, args=(i, job), daemon=True) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _compare_blocks(session: Session, left: str, right: str, symptoms: str, preset: Preset | None = None) -> None:
    """Run two blocks on the same symptoms at once and show them side by side."""
    left, right = lego.resolve(left).name, lego.resolve(right).name
    if preset:
        ui.preset_header(console, preset)
    with console.status(f"{left}와 {right}가 동시에 답하는 중…"):
        for name in (left, right):  # load first, so a missing memory file fails before the race
            session.get(name)
        a, b = _in_parallel(lambda: session.predict(left, symptoms), lambda: session.predict(right, symptoms))
    ui.comparison(console, (left, a), (right, b))
    if left == right:
        console.print("[dim]같은 블록끼리 비교했습니다. /compare <블록1> <블록2> N 으로 다른 블록을 고르세요.[/]")


def _compare(session: Session, arg: str) -> None:
    if session.task == "study":
        _study_compare(session, arg)
        return
    try:
        parsed = parse_compare(arg, session.current)
    except UsageError as exc:
        console.print(f"[dim]{exc}[/]")
        return
    try:
        _compare_blocks(session, parsed.left, parsed.right, parsed.symptoms, parsed.preset)
    except Exception as exc:  # a block that fails to load is reported, not raised into the REPL
        console.print(f"[red]비교 실패:[/] {exc}")


def _switch(session: Session, arg: str) -> None:
    if session.task == "study":
        _study_mode(session, arg)
        return
    if not arg.strip():
        console.print(ui.blocks_table(session.current))
        console.print("[dim]/harness <이름> 으로 교체 (별칭: knn, fewshot, cue, verified, reflexion, contrast, cc, none)[/]")
        return
    try:
        with console.status("블록을 불러오는 중…"):
            block = session.switch(arg)
    except Exception as exc:  # unknown name (with a hint) or an unreadable memory file
        console.print(f"[red]{exc}[/]")
        return
    console.print(f"[dim]하네스 교체 완료 → 이후 입력은 {block.name}로 진단합니다.[/]")
    ui.switched(console, block)


def _memory(session: Session, arg: str) -> None:
    action = arg.strip().lower() or "status"
    if session.task == "study":
        _study_memory(session, action)
        return
    if action == "reset":
        removed = session.scratchpad.reset()
        console.print(f"세션 기억 {removed}건을 지우고 원본(동결) 기억으로 되돌렸습니다." if removed
                      else "[dim]세션에 추가한 예시가 없습니다. 이미 원본 상태입니다.[/]")
        return
    if action != "status":
        console.print("[dim]사용법: /memory status | /memory reset[/]")
        return
    with console.status("블록 8개의 기억을 불러오는 중…"):
        for name in lego.names():
            session.get(name)
    rows = {row.system: row for row in session.scratchpad.status()}
    console.print(ui.memory_table(tuple(rows[n] for n in lego.names() if n in rows)))
    if session.scratchpad.records:
        console.print(ui.session_records(session.scratchpad.records))
    where = f"{session.home.scratchpad_path} (다음 세션에도 유지)" if session.scratchpad.persistent else "메모리 (--demo: 저장 안 함)"
    console.print(f"[dim]세션 기억 저장 위치: {where} · 동결된 logs/ 파일은 읽기 전용[/]")


def _record(session: Session, arg: str) -> None:
    if session.task == "study":
        _study_record(session, arg)
        return
    try:
        symptoms, disease = parse_record(arg)
        record = session.scratchpad.add(symptoms, disease, session.harness.engine.labels)
    except (UsageError, ScratchpadError) as exc:
        console.print(f"[yellow]{exc}[/]")
        return
    total = len(session.scratchpad.records)
    console.print(f"세션 기억에 추가: [bold]{record.disease}[/] ← {record.symptoms}  [dim](세션 {total}건, 원본 파일은 그대로)[/]")
    console.print(f'[dim]같은 증상으로 확인: /compare no_memory knn_top6 {record.symptoms}[/]')
    if not session.live:
        console.print("[dim]오프라인에서는 바뀐 검색 결과(★ 표시 예시)만 보이고 진단은 비어 있습니다.[/]")


# --- study task (your own notes) ------------------------------------------------------------------

def _complete(session: Session) -> Callable[[str], tuple[str, int | None]] | None:
    """The live model call for study questions, or None when no server answered at startup."""
    if not session.live:
        return None
    s, deadline = session.settings, OPTIONS["deadline"]
    key = os.environ.get("OPENAI_API_KEY", "")

    def complete(prompt: str) -> tuple[str, int | None]:
        return run_with_deadline(lambda: docqa.chat_completion(s.api_base, s.model, prompt, deadline, key), deadline + 1)

    return complete


def _study_answer(session: Session, question: str, mode: str) -> tuple[StudyAnswer, str | None]:
    """Live answer, or the offline preview (exact prompt, retrieved notes) when the model is unreachable."""
    complete = _complete(session)
    if complete is not None:
        try:
            return session.buddy.ask(question, mode, session.profile, complete), None
        except Exception as exc:  # fall back to the preview; the note says why
            note = f"모델 호출 실패({type(exc).__name__}) → {OFFLINE_PREVIEW}"
            return session.buddy.prepare(question, mode, session.profile), note
    return session.buddy.prepare(question, mode, session.profile), OFFLINE_PREVIEW


def _study_ask(session: Session, question: str) -> None:
    try:
        with console.status("노트를 검색하고 답하는 중…" if session.live else "노트를 검색하는 중…"):
            ans, note = _study_answer(session, question, session.mode)
    except NotesError as exc:
        console.print(f"[yellow]{exc}[/]")
        return
    session.last["study"] = ans.question
    ui.notice(console, note)
    if not session.buddy.chunks and ans.mode != "no_memory":
        console.print("[dim]불러온 노트가 없습니다. /load <파일.md|파일.txt> 로 강의 노트를 넣으세요.[/]")
    panels.study_answer(console, ans, session.settings.model.split("/")[-1] if session.live else "")


def _study_compare(session: Session, arg: str) -> None:
    try:
        left, right, question = parse_study_compare(arg, session.last.get("study", ""))
        with console.status(f"{left}와 {right}가 같은 질문에 답하는 중…"):
            (a, note_a), (b, note_b) = _in_parallel(lambda: _study_answer(session, question, left),
                                                    lambda: _study_answer(session, question, right))
    except (UsageError, NotesError) as exc:
        console.print(f"[dim]{exc}[/]")
        return
    session.last["study"] = a.question
    ui.notice(console, note_a or note_b)
    panels.study_comparison(console, a, b)


def _study_mode(session: Session, arg: str) -> None:
    if not arg.strip():
        console.print(panels.modes_table(session.mode))
        console.print("[dim]/harness <모드> 로 교체 (별칭: none, knn, rag, cues, lessons)[/]")
        return
    try:
        mode = docqa.resolve_mode(arg)
    except NotesError as exc:
        console.print(f"[red]{exc}[/]")
        return
    session.mode = mode.name
    panels.mode_switched(console, mode)


def _study_record(session: Session, arg: str) -> None:
    try:
        concept, answer = parse_record(arg)
        lesson = session.buddy.add_lesson(concept, answer)
    except UsageError:
        console.print(f"[yellow]{STUDY_RECORD_USAGE}[/]")
        return
    except (NotesError, HomeError) as exc:
        console.print(f"[yellow]{exc}[/]")
        return
    where = "~/.harness 에 저장" if session.home else "메모리에만 (--demo)"
    console.print(f"학습 기억에 추가: [bold]{lesson.concept}[/] → {lesson.answer}  "
                  f"[dim]({len(session.buddy.lessons)}건, {where}, 다음 질문부터 바로 검색됩니다)[/]")


def _study_memory(session: Session, action: str) -> None:
    if action == "reset":
        removed = session.buddy.reset_lessons()
        console.print(f"가르친 규칙 {removed}건을 지웠습니다. 불러온 노트는 그대로입니다." if removed
                      else "[dim]가르친 규칙이 없습니다.[/]")
        return
    if action != "status":
        console.print("[dim]사용법: /memory status | /memory reset[/]")
        return
    buddy = session.buddy
    panels.study_status(console, buddy.sources, len(buddy.chunks), len(buddy.lessons), session.home is not None)


def _load(session: Session, arg: str) -> None:
    path = arg.strip().strip("'\"")
    if not path:
        console.print("[dim]사용법: /load <파일.md|파일.txt>   (영어 노트, 2MB 이하)[/]")
        return
    try:
        count = session.buddy.load(Path(path))
    except NotesError as exc:
        console.print(f"[yellow]{exc}[/]")
        return
    session.task = "study"
    total = session.buddy.notes_chars
    console.print(f"[green]불러옴:[/] {Path(path).name} → {count}단락 (노트 전체 {total:,}자, "
                  f"{len(session.buddy.sources)}개 파일). study task로 전환했습니다. 모드: {session.mode}")


def _task(session: Session, arg: str) -> None:
    name = arg.strip().lower()
    if name not in ("medical", "study"):
        console.print(f"[dim]지금 task: {session.task}.  /task medical (22개 질병 벤치마크) | /task study (내 노트 질의응답)[/]")
        return
    session.task = name
    if name == "study":
        console.print(f"[dim]study task: 영어 질문을 입력하세요. 모드 {session.mode} · 노트 {len(session.buddy.chunks)}단락 · "
                      f"규칙 {len(session.buddy.lessons)}건. /load 로 노트 추가.[/]")
    else:
        console.print(f"[dim]medical task: 영어 증상 문장 또는 프리셋 1·2·3. 블록 {session.current}.[/]")


def _inspect(session: Session, arg: str) -> None:
    query = session.last.get(session.task, "")
    if arg.strip() and arg.strip().lower() not in ("context", "prompt"):
        query = arg.strip()
    if not query:
        console.print("[dim]아직 질문이 없습니다. 질문한 뒤 /inspect context 를 입력하세요.[/]")
        return
    try:
        if session.task == "study":
            panels.inspect_study(console, session.buddy.prepare(query, session.mode, session.profile))
            return
        engine = session.harness.engine
        prompt = engine.prompt_for(engine.wrap(query))
        panels.inspect_medical(console, session.current, query, prompt, engine.picks(query))
    except NotesError as exc:
        console.print(f"[yellow]{exc}[/]")


def _explain(session: Session, arg: str) -> None:
    topic = arg.strip().lower() or "fewshot"
    show = panels.EXPLAIN_TOPICS.get(topic)
    if show is None:
        console.print(f"[dim]주제: {', '.join(panels.EXPLAIN_TOPICS)}   예) /explain fewshot[/]")
        return
    show(console)


def _profile(session: Session, arg: str) -> None:
    if session.home is None:
        console.print("[dim]--demo 모드에서는 프로필을 읽거나 저장하지 않습니다.[/]")
        return
    words = arg.split(maxsplit=2)
    if words and words[0].lower() == "set":
        if len(words) < 2:
            console.print("[dim]사용법: /profile set <필드> <값>   (값을 비우면 지움)[/]")
            return
        try:
            session.home.set_profile(words[1], words[2] if len(words) > 2 else "")
        except HomeError as exc:
            console.print(f"[yellow]{exc}[/]")
            return
    elif words:
        console.print("[dim]사용법: /profile | /profile set <필드> <값>[/]")
        return
    console.print(panels.profile_table(session.home.profile(), str(session.home.profile_path)))
    console.print("[dim]프로필은 study task의 knn_retrieval·lesson_cues 프롬프트에 한 줄로 들어갑니다 (no_memory 제외).[/]")


def _quiz(session: Session, arg: str) -> None:
    if has_cjk(arg):
        console.print("[yellow]주제는 영어로 입력해 주세요.[/]")
        return
    session.task = "study"
    _study_ask(session, docqa.quiz_question(arg))


def _study(session: Session, ask: Callable[[str], str] | None = None) -> int:
    def compare(left: str, right: str, key: str) -> None:
        preset = find_preset(key)
        _compare_blocks(session, left, right, preset.symptoms, preset)

    ask = ask or (lambda prompt: Prompt.ask(f"[dim]{prompt}[/]", default="", show_default=False))
    return study.run(console, compare, ask, session.live, len(session.scratchpad.records))


def _run_preset(session: Session, key: str) -> Prediction | None:
    preset = find_preset(key)
    if preset is None:
        ui.presets_menu(console)
        return None
    ui.preset_header(console, preset)
    return _show(_ask(session, preset.symptoms), show_evidence=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    demo: bool = typer.Option(False, "--demo", "--mock", help="서버에 접속하지 않고 녹화된 응답으로 시연"),
    deadline: float = typer.Option(60.0, "--timeout", help="실시간 응답을 기다리는 최대 초 (넘으면 녹화 응답)"),
) -> None:
    """인자 없이 실행하면 대화형 모드로 들어갑니다."""
    OPTIONS.update(demo=demo, deadline=deadline)
    if ctx.invoked_subcommand is None:
        chat()


@app.command()
def predict(
    symptoms: str = typer.Argument(..., help="환자 증상 (영어 문장)"),
    json_out: bool = typer.Option(False, "--json", help="결과를 JSON으로 출력"),
    evidence: bool = typer.Option(True, "--evidence/--no-evidence", help="하네스가 고른 예시 표시"),
) -> None:
    """증상 한 건을 진단합니다 (서버가 없고 녹화도 없으면 프롬프트 미리보기, 종료 코드 2)."""
    session = _session()
    pred, note = _ask(session, symptoms)
    if pred is None:
        ui.notice(console, note)
        raise typer.Exit(1)
    if json_out:
        typer.echo(json.dumps(asdict(pred), ensure_ascii=False, indent=2))
    else:
        _show((pred, note), show_evidence=evidence)
    if pred.source == "preview":  # retrieval and prompt only: scripts can tell there is no diagnosis
        raise typer.Exit(2)


@app.command("demo")
def demo_cmd(key: str = typer.Argument("", help="프리셋 번호 1·2·3 (비우면 대화형 모드)")) -> None:
    """발표용 프리셋 증상으로 바로 진단합니다."""
    if not key:
        chat()
        return
    session = _session()
    _banner(session)
    _run_preset(session, key)


@app.command()
def compare(
    args: list[str] = typer.Argument(..., help="[블록1 블록2] 프리셋 번호 1·2·3 또는 영어 증상 문장"),
) -> None:
    """두 블록을 같은 증상으로 좌우 비교합니다 (블록을 생략하면 기억 없음 vs 시작 블록)."""
    _compare(_session(), " ".join(args))


@app.command("study")
def study_cmd() -> None:
    """4단계 학습 가이드 (Enter 다음, q 종료)."""
    _study(_session())


@app.command()
def harness() -> None:
    """레고 블록 8개와 기록된 정확도·토큰을 보여 줍니다."""
    console.print(ui.blocks_table(""))


@app.command("config")
def config_cmd(
    api_base: str = typer.Option("", "--api-base", help="OpenAI 호환 주소 (예: http://localhost:11434/v1)"),
    model: str = typer.Option("", "--model", help="그 주소에서 쓸 모델 (예: llama3.2:3b, gpt-4o-mini)"),
    reset: bool = typer.Option(False, "--reset", help="저장한 설정 삭제"),
    check: bool = typer.Option(True, "--check/--no-check", help="후보 주소마다 /models 응답 확인"),
) -> None:
    """모델 엔드포인트를 ~/.harness/config.json 에 저장하고 연결을 확인합니다. API 키는 저장하지 않습니다."""

    home = HarnessHome()
    try:
        if reset:
            home.clear_config()
            console.print(f"[green]삭제:[/] {config.user_config_path()}")
        elif api_base or model:
            saved = home.save_config(api_base, model)
            console.print(f"[green]저장:[/] {config.user_config_path()} → {saved}")
    except HomeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    table = Table(title="엔드포인트 후보 (위에서부터 시도)", title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("주소")
    table.add_column("상태")

    for i, base in enumerate(config.api_base_candidates(), 1):
        models = list_models(base) if check else None
        state = "확인 안 함" if not check else ("응답 없음" if models is None else f"응답 · 모델 {len(models)}개")
        table.add_row(str(i), base, state)
    console.print(table)
    key = "설정됨" if config.has_real_key() else "없음 (OpenAI/OpenRouter는 필요, Ollama/vLLM은 보통 불필요)"
    console.print(f"[dim]OPENAI_API_KEY: {key} · 키는 파일에 저장하지 않고 환경 변수 또는 .env 파일(또는 ~/.harness/.env)에서 자동 로드됩니다.[/]")
    console.print("[dim]설정 예시: cp .env.example .env (자세한 설정 방법은 .env.example 참고)[/]")
    if check:
        found = discover(Settings.from_env())
        console.print(f"[green]사용할 서버:[/] {found.api_base} · {found.model}" if found else f"[yellow]{config.NO_SERVER_MESSAGE}[/]")


@app.command()
def info() -> None:
    """기억 상태(예시 수, 오답 통계)를 보여 줍니다."""
    session = _session()
    ui.info(console, session.harness.engine.stats(), session.harness.engine.settings)


@app.command(hidden=True)
def record(
    workers: int = typer.Option(1, "--workers", min=1, max=6, help="동시 호출 수 (지연 시간을 정직하게 남기려면 1)"),
    only: str = typer.Option("", "--only", help="쉼표로 구분한 하네스 이름만 녹화 (비우면 8개 전부)"),
) -> None:
    """프리셋 3건 × 레고 블록 8개의 실제 서버 응답을 녹화해 오프라인 재생에 씁니다."""
    # A private response cache, so every cell is a real round trip with server-counted prompt tokens.
    os.environ["TEXT_CLASSIFICATION_LLM_CACHE_DIR"] = tempfile.mkdtemp(prefix="harness-cli-record-")
    session = _session()
    if not session.live:
        console.print("[red]서버에 연결되지 않아 녹화할 수 없습니다.[/]")
        raise typer.Exit(1)
    blocks = [lego.resolve(n) for n in only.split(",") if n.strip()] or list(lego.BLOCKS.values())
    cells = [(block, preset) for block in blocks for preset in PRESETS]
    for block in blocks:  # load sequentially; the pool lock would serialise this anyway
        session.get(block.name)
    model = session.harness.engine.settings.model

    def run(cell: tuple[lego.Block, Preset]) -> str:
        block, preset = cell
        engine = session.get(block.name).engine
        try:
            pred = engine.predict(preset.symptoms)
        except Exception as exc:  # leave the cell unrecorded; never fill it in
            return f"{block.name:>20} [{preset.key}] 실패: {type(exc).__name__}: {exc}"
        if not pred.response:
            return f"{block.name:>20} [{preset.key}] 실패: 빈 응답"
        session.cache.put(block.name, preset.symptoms, {
            "response": pred.response,
            "seconds": round(pred.seconds, 2),
            "prompt_chars": pred.prompt_chars,
            "prompt_tokens": pred.prompt_tokens,
            "prompt_sha": pred.prompt_sha,
            "memory": f"harness_cli/data/memories/{block.memory_path.name}",
            "model": model,
            "workers": workers,
            "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        session.cache.save()
        return f"{block.name:>20} [{preset.key}] → {pred.diagnosis} ({pred.seconds:.1f}s, {pred.prompt_tokens} tok)"

    with console.status(f"{len(cells)}건 녹화 중…"), ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(run, cells):
            console.print(line)
    console.print(f"[green]저장:[/] {session.cache.path}")


@app.command()
def chat() -> None:
    """대화형 모드 (REPL)."""
    session = _session()
    _banner(session)
    ui.presets_menu(console)
    last: dict[str, Prediction | None] = {"pred": None}

    def show_help() -> bool:
        for cmd in registry.commands:
            names = ", ".join(f"/{n}" for n in (cmd.name, *cmd.aliases))
            console.print(f"  [bold]{names}[/]  {cmd.help}")
        console.print("  [bold]1, 2, 3[/]  프리셋 증상을 지금 끼운 블록으로 진단 (medical task)")
        console.print(f"  [dim]medical 블록: {', '.join(lego.names())}[/]")
        console.print(f"  [dim]study 모드: {', '.join(docqa.MODES)}[/]")
        return True

    def show_info() -> bool:
        engine = session.harness.engine
        ui.info(console, engine.stats(), engine.settings)
        return True

    def show_why() -> bool:
        if last["pred"] is None:
            console.print("[dim]아직 진단한 증상이 없습니다.[/]")
        else:
            ui.prediction(console, last["pred"], show_evidence=True)
        return True

    def demo(args: str) -> bool:
        last["pred"] = _run_preset(session, args) or last["pred"]
        return True

    def with_args(fn: Callable[[Session, str], None]) -> Callable[[str], bool]:
        def handler(args: str) -> bool:
            fn(session, args)
            return True
        return handler

    def run_study() -> bool:
        _study(session)
        return True

    def clear() -> bool:
        console.clear()
        _banner(session)
        return True

    registry = CommandRegistry(
        [
            Command("help", "명령어 목록", show_help, ("h",)),
            Command("task", "작업 전환 (/task medical | /task study)", with_args(_task), takes_args=True),
            Command("load", "강의 노트 불러오기 (/load notes.md) → study task", with_args(_load), takes_args=True),
            Command("harness", "블록 교체 (/harness knn_top6 · study: /harness lesson_cues). 이름 없이 입력하면 목록",
                    with_args(_switch), ("switch",), True),
            Command("compare", "두 블록 좌우 비교 (/compare no_memory knn_top6 1). /compare 2 는 기억 없음 vs 지금 블록",
                    with_args(_compare), ("vs",), True),
            Command("record", '기억에 추가 (medical: "<증상>" -> "<질병>" · study: "<개념>" -> "<규칙>"), ~/.harness 에 저장',
                    with_args(_record), takes_args=True),
            Command("memory", "기억 상태 표 (/memory status) · 추가분 지우기 (/memory reset)", with_args(_memory), takes_args=True),
            Command("inspect", "직전 질문의 RAW 프롬프트와 고른 예시·유사도 (/inspect context)", with_args(_inspect), takes_args=True),
            Command("explain", "개념 설명 (/explain fewshot)", with_args(_explain), takes_args=True),
            Command("profile", "학습자 프로필 보기·설정 (/profile set style short bullet points)", with_args(_profile), takes_args=True),
            Command("quiz", "노트로 복습 문제 만들기 (/quiz hypothesis testing)", with_args(_quiz), takes_args=True),
            Command("study", "4단계 학습 가이드 (Enter 다음, q 종료)", run_study),
            Command("demo", "프리셋 진단 (/demo 1). 번호 없이 입력하면 목록", demo, takes_args=True),
            Command("info", "지금 블록의 기억 상태와 오답 통계", show_info),
            Command("why", "직전 진단의 근거(고른 예시) 다시 보기", show_why),
            Command("clear", "화면 지우기", clear),
            Command("exit", "종료", lambda: False, ("quit", "q")),
        ]
    )
    preset_keys = tuple(p.key for p in PRESETS)

    while True:
        try:
            label = f"study:{session.mode}" if session.task == "study" else session.current
            line = Prompt.ask(f"[bold {ui.ACCENT}]{label} >[/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        route = route_input(line, registry, preset_keys)
        if route.kind == "empty":
            continue
        if route.kind == "unknown":
            console.print(f"[dim]알 수 없는 명령어: {route.text} — /help 를 입력해 보세요.[/]")
            continue
        if route.kind == "command":
            if not route.command.run(route.args):
                break
            continue
        if session.task == "study":
            _study_ask(session, route.text)
            continue
        if route.kind == "preset":
            last["pred"] = _run_preset(session, route.text) or last["pred"]
            continue
        pred = _show(_ask(session, route.text), show_evidence=False)
        if pred is not None:
            last["pred"] = pred
            console.print("[dim]근거 예시: /why · RAW 프롬프트: /inspect context[/]")
    console.print("[dim]종료합니다.[/]")
