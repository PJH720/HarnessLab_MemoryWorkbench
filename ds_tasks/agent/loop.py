"""ReAct-style episode: the model writes one Python block per turn, sees the output, then answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .sandbox import make_workdir, run_python

SYSTEM = """You are a careful data analyst. Solve the task by writing Python.
Installed: pandas, numpy, scipy, scikit-learn, lightgbm.
Each turn, reply with exactly one ```python code block. It runs in a fresh Python process, in a folder that contains the input files, and you will see its stdout and stderr. Variables do not persist between turns, so reload data each time and print what you need.
When you are sure of the answer, reply with a single line and no code:
FINAL ANSWER: <answer>
The answer is a number, a short string, or a JSON list."""

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.S)

LLMFn = Callable[[list[dict], int], tuple[str, dict]]


@dataclass
class Episode:
    card_id: str
    answer: object = None
    finished: bool = False
    steps: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    first_prompt_chars: int = 0

    @property
    def n_exec_errors(self) -> int:
        return sum(1 for s in self.steps if s["returncode"] != 0)

    @property
    def last_code(self) -> str:
        ok = [s["code"] for s in self.steps if s["returncode"] == 0]
        return ok[-1] if ok else (self.steps[-1]["code"] if self.steps else "")


def parse_answer(text: str):
    raw = text.strip().splitlines()[0].strip().strip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip("\"'")


def run_episode(card: dict, llm: LLMFn, seed: int, memory_text: str = "", max_steps: int = 8) -> Episode:
    workdir = make_workdir(card["inputs"])
    task = f"Task:\n{card['question']}\n\nInput files: {', '.join(card['inputs'])}"
    user = f"{memory_text}\n\n{task}" if memory_text else task
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ep = Episode(card["id"], first_prompt_chars=len(SYSTEM) + len(user))
    for _ in range(max_steps):
        reply, usage = llm(messages, seed)
        ep.llm_calls += 1
        ep.tokens_in += usage.get("in", 0)
        ep.tokens_out += usage.get("out", 0)
        messages.append({"role": "assistant", "content": reply})
        code = CODE_RE.search(reply)
        final = FINAL_RE.search(reply)
        if final and not code:
            ep.answer, ep.finished = parse_answer(final.group(1)), True
            break
        if not code:
            messages.append({"role": "user", "content": "Reply with one ```python block, or the FINAL ANSWER line."})
            continue
        result = run_python(code.group(1), workdir)
        ep.steps.append({"code": code.group(1), "stdout": result.stdout, "stderr": result.stderr,
                         "returncode": result.returncode, "timed_out": result.timed_out})
        messages.append({"role": "user", "content": f"Output (exit code {result.returncode}):\n"
                         f"{result.stdout or '(no stdout)'}\n{('STDERR:' + chr(10) + result.stderr) if result.stderr else ''}"})
    return ep
