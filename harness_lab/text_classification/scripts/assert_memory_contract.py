"""Audit a finished run: do error-driven memories hold one item per real training mistake?

Counts wrong training steps in each system's log.jsonl and compares them with the
counters and items saved in memory.json. Raises AssertionError on any mismatch or on
an error-driven memory that stored nothing.

Usage: uv run python scripts/assert_memory_contract.py logs/online-v2
"""

import json
import sys
from pathlib import Path

ERROR_DRIVEN = {"contrast_online", "cue_bank", "reflexion_notes", "cue_bank_verified"}


def training_errors(log_path: Path) -> tuple[int, int]:
    steps = [json.loads(line) for line in log_path.read_text().splitlines() if '"type": "step"' in line]
    return sum(1 for s in steps if s.get("ok") is False), len(steps)


def audit(run_dir: Path) -> list[str]:
    failures, rows = [], []
    for mem_path in sorted(run_dir.glob("*/*/*/memory.json")):
        system = mem_path.parent.parent.name
        errors, steps = training_errors(mem_path.parent / "log.jsonl")
        state = json.loads(mem_path.read_text())
        if system == "contrast_online":
            stored, parse_fail = len(state["error_ids"]), 0
        elif system in ERROR_DRIVEN:
            stored = len(state["items"])
            # verified memories also drop cues that failed verification or duplicate a pair
            parse_fail = state["n_parse_failures"] + state.get("n_rejected", 0) + state.get("n_duplicates", 0)
        else:
            stored, parse_fail = None, None
        rows.append((system, steps, errors, stored, parse_fail))
        if system not in ERROR_DRIVEN:
            continue
        if steps == 0:
            failures.append(f"{system}: no training steps logged (offline mode?)")
        if errors == 0 or stored == 0:
            failures.append(f"{system}: errors={errors}, stored={stored} — mechanism never fired")
        if state.get("n_errors_seen") != errors:
            failures.append(f"{system}: n_errors_seen={state.get('n_errors_seen')} != logged errors {errors}")
        if stored + parse_fail != errors:
            failures.append(f"{system}: stored {stored} + parse failures {parse_fail} != errors {errors}")
    print(f"{'system':<20}{'train steps':>12}{'wrong':>8}{'stored':>8}{'dropped':>12}")
    for system, steps, errors, stored, pf in rows:
        print(f"{system:<20}{steps:>12}{errors:>8}{'-' if stored is None else stored:>8}{'-' if pf is None else pf:>12}")
    return failures


if __name__ == "__main__":
    problems = audit(Path(sys.argv[1]))
    assert not problems, "memory contract violated:\n" + "\n".join(problems)
    print("memory contract OK")
