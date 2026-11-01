"""Run harnesses on the task cards with 2-fold cross-fitting and log every episode.

Memory learns online on one fold (train) and is evaluated frozen on the other (eval);
folds then swap, so each card is evaluated exactly once per harness and seed (paired design).

Usage: uv run python -m agent.run --harness ds_base ds_knn ds_cue_verified --seed 0 --out runs/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from scoring import score

from .llm import ChatLLM
from .loop import Episode, run_episode
from .memories import HARNESSES

CARDS_DIR = Path(__file__).resolve().parent.parent / "cards"


def load_cards() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(CARDS_DIR.glob("*.json"))]


def fold_of(card_id: str) -> int:
    return int(hashlib.sha256(card_id.encode()).hexdigest(), 16) % 2


def row(harness, seed, fold, phase, card, ep: Episode, correct, memory_ids, memory_chars) -> dict:
    return {"harness": harness, "seed": seed, "fold": fold, "phase": phase, "card": card["id"],
            "tags": "|".join(card["tags"]), "correct": bool(correct), "answer": json.dumps(ep.answer, default=str),
            "expected": json.dumps(card["answer"], default=str), "finished": ep.finished, "steps": len(ep.steps),
            "exec_errors": ep.n_exec_errors, "timeouts": sum(s["timed_out"] for s in ep.steps),
            "llm_calls": ep.llm_calls, "tokens_in": ep.tokens_in, "tokens_out": ep.tokens_out,
            "first_prompt_chars": ep.first_prompt_chars, "memory_ids": json.dumps(memory_ids), "memory_chars": memory_chars}


def run_harness(name: str, seed: int, cards: list[dict], llm: ChatLLM, out: Path, workers: int) -> list[dict]:
    rows = []
    for eval_fold in (0, 1):
        memory = HARNESSES[name]()
        train = [c for c in cards if fold_of(c["id"]) != eval_fold]
        evaluate = [c for c in cards if fold_of(c["id"]) == eval_fold]
        if name != "ds_base":  # online learning on the training fold
            for card in train:
                text, ids = memory.context(card)
                ep = run_episode(card, llm, seed, memory_text=text)
                ok = score(card, ep.answer)
                rows.append(row(name, seed, eval_fold, "train", card, ep, ok, ids, len(text)))
                memory.learn(card, ep, ok, llm, seed, score)

        def evaluate_one(card):
            text, ids = memory.context(card)
            ep = run_episode(card, llm, seed, memory_text=text)
            return row(name, seed, eval_fold, "eval", card, ep, score(card, ep.answer), ids, len(text))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows += list(pool.map(evaluate_one, evaluate))
        (out / f"memory_{name}_seed{seed}_fold{eval_fold}.json").write_text(json.dumps(memory.state(), ensure_ascii=False, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", nargs="+", default=list(HARNESSES))
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--cards", nargs="*", help="limit to these card ids")
    parser.add_argument("--out", default="runs/pilot")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = [c for c in load_cards() if not args.cards or c["id"] in args.cards]
    llm = ChatLLM()
    all_rows = []
    for seed in args.seed:
        for name in args.harness:
            started = time.time()
            rows = run_harness(name, seed, cards, llm, out, args.workers)
            all_rows += rows
            ev = [r for r in rows if r["phase"] == "eval"]
            print(f"{name} seed={seed}: eval {sum(r['correct'] for r in ev)}/{len(ev)} in {time.time() - started:.0f}s", flush=True)
            pd.DataFrame(all_rows).to_parquet(out / "trajectories.parquet", index=False)


if __name__ == "__main__":
    main()
