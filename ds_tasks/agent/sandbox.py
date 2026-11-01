"""Run agent code in a fresh Python process inside a throwaway folder holding only the inputs.

Limits: 30 s wall clock, minimal environment, working directory = the episode folder.
This isolates the agent from the project tree; it is not a security boundary.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import reference as ref

OUTPUT_LIMIT = 2000


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def make_workdir(inputs: list[str]) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="ds_agent_"))
    for rel in inputs:
        target = workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((ref.DATA_DIR / rel).resolve())
    return workdir


def run_python(code: str, workdir: Path, timeout: int = 30) -> ExecResult:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2"}
    try:
        done = subprocess.run([sys.executable, "-I", "-c", code], cwd=workdir, env=env,
                              capture_output=True, text=True, timeout=timeout)
        return ExecResult(done.stdout[-OUTPUT_LIMIT:], done.stderr[-OUTPUT_LIMIT:], done.returncode, False)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        out = out.decode() if isinstance(out, bytes) else out
        return ExecResult(out[-OUTPUT_LIMIT:], f"TIMEOUT after {timeout}s", -1, True)
