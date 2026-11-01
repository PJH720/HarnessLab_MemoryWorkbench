"""Build dist/harness-cli-v<version>-standalone.zip from the checkout.

The bundle holds the harness_cli wheel (which carries the 8 frozen memories in harness_cli/data/memories),
an allowlisted copy of the harness_lab/text_classification runtime code, a run script and a README.
Nothing under harness_lab/text_classification/logs/ is read or written.

Usage (from the project root, after `cd harness_cli && uv build`):
    uv run --project harness_cli python scripts/build_standalone.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = ROOT / "harness_cli"
TC = ROOT / "harness_lab/text_classification"
OUT_DIR = ROOT / "dist"

# Runtime code only: no tests, scripts, config.yaml, claude_wrapper.py, caches, dataset splits or other logs.
TC_FILES = ("__init__.py", "inner_loop.py", "llm.py", "memory_system.py")
TC_DATA_FILES = ("__init__.py", "api.py", "constants.py", "evaluators.py", "loaders.py")

RUN_SH = """#!/usr/bin/env bash
# Starts harness-cli in demo mode against the files in this folder. Extra arguments go to harness-cli,
# e.g. ./run.sh compare no_memory knn_top6 1
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export HARNESS_DATA_ROOT="$DIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec uv run --no-project --with "$DIR/{wheel}" harness-cli --demo "$@"
"""

README = """harness-cli v{version} (standalone bundle)

Contents
  {wheel}       the CLI package (replay.json, the prompt template and the 8 frozen memories are inside it)
  replay.json   reference copy of the replay cache inside the wheel (sha256 {replay_sha})
  harness_lab/text_classification/   runtime code of the evaluated memory systems (read-only copy)
  run.sh        starts the CLI in --demo mode

Run
  ./run.sh                              interactive REPL
  ./run.sh compare no_memory knn_top6 1
  ./run.sh study

Requirements
  - uv (https://docs.astral.sh/uv/). The first run downloads the Python dependencies from PyPI
    unless they are already in the uv cache.
  - --demo needs no GPU server: answers come from replay.json, recorded from real server calls for
    presets 1-3 on every block. Any other input shows the exact prompt and the retrieved examples
    (offline retrieval preview) with no diagnosis.
  - Live answers need an OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, OpenAI). Run without
    --demo after pointing the CLI at one, e.g.
      HARNESS_DATA_ROOT="$PWD" uv run --no-project --with ./{wheel} harness-cli config --api-base http://localhost:11434/v1
      HARNESS_DATA_ROOT="$PWD" uv run --no-project --with ./{wheel} harness-cli

Data
  The dataset splits are not included; the diagnostician prompt comes from the copy bundled in the
  wheel, which the test suite pins to the dataset prompt byte for byte. The bundled memories contain
  Symptom2Disease training examples written by the evaluated runs (seed 42; confusion_contrast is the
  midterm run). Accuracy figures shown by the CLI come from analysis/multiseed/out/accuracy_tokens.csv
  in the source repository.

Talk: https://harness-midterm-talk.vercel.app
"""


def _version() -> str:
    sys.path.insert(0, str(CLI_DIR / "src"))
    from harness_cli import __version__

    return __version__


def _check_wheel_memories(wheel: Path) -> None:
    """The wheel must carry every block's memory, byte-identical to the package copy the tests pin."""
    from harness_cli import lego

    names = lego.names()
    if len(names) != 8:
        raise SystemExit(f"expected 8 lego blocks, found {len(names)}")
    with zipfile.ZipFile(wheel) as whl:
        for name in names:
            member = f"harness_cli/data/memories/{name}.json"
            try:
                data = whl.read(member)
            except KeyError:
                raise SystemExit(f"{wheel.name} lacks {member}; rebuild the wheel") from None
            if data != lego.resolve(name).memory_path.read_bytes():
                raise SystemExit(f"{member} differs from the package copy; rebuild the wheel")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(stage: Path, wheel: Path, version: str) -> None:
    shutil.copyfile(wheel, stage / wheel.name)
    replay = CLI_DIR / "src/harness_cli/replay.json"
    with zipfile.ZipFile(wheel) as whl:
        if whl.read("harness_cli/replay.json") != replay.read_bytes():
            raise SystemExit("replay.json differs from the copy in the wheel; rebuild the wheel first")
    shutil.copyfile(replay, stage / "replay.json")

    tc_out = stage / "harness_lab/text_classification"
    (tc_out / "data").mkdir(parents=True)
    (tc_out / "agents").mkdir()
    for name in TC_FILES:
        shutil.copyfile(TC / name, tc_out / name)
    for name in TC_DATA_FILES:
        shutil.copyfile(TC / "data" / name, tc_out / "data" / name)
    for src in sorted((TC / "agents").glob("*.py")):
        shutil.copyfile(src, tc_out / "agents" / src.name)

    _check_wheel_memories(wheel)

    (stage / "run.sh").write_text(RUN_SH.format(wheel=wheel.name))
    (stage / "README.txt").write_text(
        README.format(version=version, wheel=wheel.name, replay_sha=_sha256(replay)[:16])
    )


def _zip(stage: Path, out: Path, top: str) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(f"{top}/{path.relative_to(stage).as_posix()}", date_time=(2026, 11, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.name == "run.sh" else 0o644
            info.external_attr = (0o100000 | mode) << 16
            zf.writestr(info, path.read_bytes())


def main() -> None:
    version = _version()
    wheel = CLI_DIR / f"dist/harness_cli-{version}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit(f"{wheel} not found; run `uv build` in harness_cli first")
    top = f"harness-cli-v{version}"
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{top}-standalone.zip"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        _stage(stage, wheel, version)
        _zip(stage, out, top)
    print(f"{out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
