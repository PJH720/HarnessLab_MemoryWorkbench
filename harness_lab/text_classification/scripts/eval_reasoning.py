"""Reasoning ablation on frozen memories: same prompts, reasoning switched on.

Each memory saved by the zero-think online run (logs/online-v2) is loaded unchanged.
The exact prompt it builds for every validation question is captured (a stub LLM
intercepts the call) and sent to a reasoning-capable backend:

  --provider sglang     same Qwen3.8 model with enable_thinking on/off  (causal arm: only reasoning differs)
  --provider openai     an OpenAI reasoning model via the Responses API, at a chosen reasoning effort
  --provider anthropic  a Claude model with extended thinking on (adaptive, summarized) or off
                        (between_tools); on Claude Sonnet 5.5 the depth is set by effort, since
                        budget_tokens and temperature are rejected by the API

Outputs (logs/reasoning-on/<run_tag>/<system>/): predictions.jsonl, summary.json,
plus <run_tag>/comparison.csv (accuracy and exact McNemar vs the zero-think run) and
<run_tag>/tracked.jsonl (the ten questions where zero-think memories failed, with the
reasoning text and a heuristic flag for whether the model considered and rejected the
cue-suggested disease). Responses are cached on disk, keyed by backend and prompt.

Examples:
  # Claude, thinking on and off on the same model (needs ANTHROPIC_API_KEY)
  uv run --with anthropic python scripts/eval_reasoning.py --provider anthropic --model claude-sonnet-5-5 --thinking on
  uv run --with anthropic python scripts/eval_reasoning.py --provider anthropic --model claude-sonnet-5-5 --thinking off
  # Qwen arm on the on-prem server
  uv run python scripts/eval_reasoning.py --provider sglang --thinking on
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import binomtest
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from text_classification.data.api import load_dataset_splits_3way  # noqa: E402
from text_classification.inner_loop import load_memory_system  # noqa: E402
from text_classification.memory_system import extract_json_field  # noqa: E402

SYSTEMS = ["no_memory", "knn_top6", "cue_bank", "cue_bank_verified"]
TRACKED = {0: "cue_bank", 4: "cue_bank", 13: "cue_bank", 25: "cue_bank", 26: "cue_bank", 30: "cue_bank",
           47: "cue_bank", 29: "cue_bank_verified", 36: "cue_bank_verified", 48: "cue_bank_verified"}
MODEL_LEAF = "qwen3.8-flash-next-nvfp4"
CACHE = Path(os.environ.get("REASONING_CACHE", Path.home() / ".cache/reasoning-ablation"))
PER_QUESTION_ZERO_THINK = HERE.parent.parent / "analysis/out/per_question.csv"


def capture_prompt(memory_name: str, state: str, task_input: str) -> str:
    """Return the exact prompt the frozen memory sends for this input."""
    seen: list[str] = []

    def stub(prompt: str) -> str:
        seen.append(prompt)
        return '{"reasoning": "", "final_answer": ""}'

    memory = load_memory_system(memory_name, stub)
    memory.set_state(state)
    memory.predict(task_input)
    if len(seen) != 1:
        raise RuntimeError(f"{memory_name}: expected one LLM call at prediction, saw {len(seen)}")
    return seen[0]


class Backend:
    def __init__(self, args):
        self.args = args
        if args.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(max_retries=8)  # reads ANTHROPIC_API_KEY; retries 429/5xx
            self.tag = f"{args.model}-thinking-{args.thinking}-effort-{args.effort}"
        elif args.provider == "sglang":
            from openai import OpenAI

            self.client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "onprem-placeholder"))
            self.tag = f"{args.model.split('/')[-1].lower()}-thinking-{args.thinking}"
        else:
            from openai import OpenAI

            self.client = OpenAI()  # reads OPENAI_API_KEY
            self.tag = f"{args.model}-effort-{args.effort}"
        CACHE.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str) -> dict:
        key = hashlib.sha256(json.dumps([self.args.provider, self.tag, self.args.max_tokens, prompt]).encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        call = {"sglang": self._sglang, "openai": self._openai, "anthropic": self._anthropic}[self.args.provider]
        result = call(prompt)
        path.write_text(json.dumps(result, ensure_ascii=False))
        return result

    def _anthropic(self, prompt: str) -> dict:
        # thinking off on Sonnet 5.5 is "between_tools" (no tools here, so no thinking at all);
        # "disabled" and budget_tokens both return 400 on this model
        thinking = {"type": "adaptive", "display": "summarized"} if self.args.thinking == "on" else {"type": "between_tools"}
        resp = self.client.messages.create(
            model=self.args.model, max_tokens=self.args.max_tokens, thinking=thinking,
            output_config={"effort": self.args.effort}, messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            # no server-side fallback: a different model answering would contaminate the ablation
            return {"content": "", "reasoning": "", "reasoning_tokens": 0, "stop_reason": "refusal",
                    "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        text = "".join(b.text for b in resp.content if b.type == "text")
        thought = "\n".join(b.thinking for b in resp.content if b.type == "thinking")
        # the API reports thinking inside output_tokens; it is not broken out separately
        return {"content": text, "reasoning": thought, "reasoning_tokens": 0, "stop_reason": resp.stop_reason,
                "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

    def _sglang(self, prompt: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.args.model, messages=[{"role": "user", "content": prompt}], temperature=0.0,
            max_tokens=self.args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.args.thinking == "on"}},
        )
        msg = resp.choices[0].message
        usage = resp.usage.model_dump() if resp.usage else {}
        return {"content": msg.content or "", "reasoning": getattr(msg, "reasoning_content", None) or "",
                "reasoning_tokens": usage.get("reasoning_tokens") or 0,
                "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}

    def _openai(self, prompt: str) -> dict:
        from openai import BadRequestError

        request = {"model": self.args.model, "input": prompt, "max_output_tokens": self.args.max_tokens}
        try:
            resp = self.client.responses.create(**request, reasoning={"effort": self.args.effort, "summary": "auto"})
        except BadRequestError:
            # some reasoning models do not offer reasoning summaries; keep the effort setting
            resp = self.client.responses.create(**request, reasoning={"effort": self.args.effort})
        summary = " ".join(s.text for item in resp.output if item.type == "reasoning" for s in (item.summary or []))
        usage = resp.usage
        details = getattr(usage, "output_tokens_details", None)
        return {"content": resp.output_text or "", "reasoning": summary,
                "reasoning_tokens": getattr(details, "reasoning_tokens", 0) if details else 0,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def mentions(text: str, disease: str) -> bool:
    return bool(disease) and disease.lower() in text.lower()


def zero_think_predictions() -> dict[tuple[str, int], tuple[str, bool]]:
    import pandas as pd

    if not PER_QUESTION_ZERO_THINK.exists():
        return {}
    df = pd.read_csv(PER_QUESTION_ZERO_THINK)
    return {(r.system, int(r.q)): (str(r.pred), bool(r.correct)) for r in df.itertuples()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["sglang", "openai", "anthropic"], required=True)
    p.add_argument("--model", default="Inferact/Qwen3.8-Flash-Next-NVFP4")
    p.add_argument("--base-url", default="https://model-endpoint.example.invalid/v1")
    p.add_argument("--thinking", choices=["on", "off"], default="on")
    p.add_argument("--effort", choices=["minimal", "low", "medium", "high"], default="high")
    # anthropic "between_tools" (thinking off) is accepted only at effort high or below, so the two
    # Claude arms share effort=high and differ only in the thinking setting
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--systems", nargs="+", default=SYSTEMS)
    p.add_argument("--run-dir", default=str(HERE / "logs/online-v2/Symptom2Disease"))
    p.add_argument("--limit", type=int, help="first N validation questions only (smoke test)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    backend = Backend(args)
    out = HERE / "logs/reasoning-on" / backend.tag
    _, val, _, evaluator = load_dataset_splits_3way("Symptom2Disease", 60, 50, 212, 42)
    questions = list(range(len(val)))[: args.limit] if args.limit else list(range(len(val)))
    zero = zero_think_predictions()
    comparison, tracked = [], []

    for system in args.systems:
        state = (Path(args.run_dir) / system / MODEL_LEAF / "memory.json").read_text()
        prompts = {q: capture_prompt(system, state, val[q]["input"]) for q in questions}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = dict(zip(questions, tqdm(pool.map(lambda q: backend(prompts[q]), questions),
                                               total=len(questions), desc=system)))
        rows = []
        for q in questions:
            r = results[q]
            pred = extract_json_field(r["content"], "final_answer") or r["content"].strip()
            raw = evaluator(pred, val[q]["target"])
            correct = raw if isinstance(raw, bool) else bool(raw["was_correct"])
            rows.append({"q": q, "target": val[q]["target"], "pred": pred, "correct": correct,
                         "prompt_chars": len(prompts[q]), **{k: r[k] for k in ("reasoning_tokens", "input_tokens", "output_tokens")},
                         "stop_reason": r.get("stop_reason", ""), "reasoning": r["reasoning"],
                         "answer_text": r["content"]})
            if TRACKED.get(q) == system and (system, q) in zero:
                zt_pred, zt_ok = zero[(system, q)]
                tracked.append({
                    "q": q, "system": system, "target": val[q]["target"],
                    "zero_think_pred": zt_pred, "zero_think_correct": zt_ok,
                    "reasoning_pred": pred, "reasoning_correct": correct,
                    # heuristic: thinking or the answer's own reasoning field names the disease the cue
                    # pushed toward, yet the answer is correct; confirmed by reading the text, not by this flag
                    "considered_and_rejected_cue_disease": (not zt_ok) and correct
                    and mentions(r["reasoning"] + "\n" + r["content"], zt_pred),
                    "reasoning": r["reasoning"], "answer_text": r["content"],
                })
        sdir = out / system
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
        acc = sum(x["correct"] for x in rows) / len(rows)
        summary = {"system": system, "backend": backend.tag, "n": len(rows), "accuracy": acc,
                   "correct": sum(x["correct"] for x in rows),
                   **{k: sum(x[k] for x in rows) for k in ("reasoning_tokens", "input_tokens", "output_tokens")}}
        (sdir / "summary.json").write_text(json.dumps(summary, indent=2))

        if zero:
            zt = [zero[(system, q)][1] for q in questions if (system, q) in zero]
            rs = [x["correct"] for x in rows if (system, x["q"]) in zero]
            only_rs = sum(1 for a, b in zip(zt, rs) if b and not a)
            only_zt = sum(1 for a, b in zip(zt, rs) if a and not b)
            n = only_rs + only_zt
            comparison.append({"system": system, "n": len(rs), "zero_think_acc": sum(zt) / len(zt),
                               "reasoning_acc": sum(rs) / len(rs), "only_reasoning_correct": only_rs,
                               "only_zero_think_correct": only_zt,
                               "mcnemar_p": binomtest(only_rs, n, 0.5).pvalue if n else 1.0})
        print(f"{system}: {summary['correct']}/{summary['n']} ({acc:.1%})", flush=True)

    if comparison:
        import pandas as pd

        pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
        print(pd.DataFrame(comparison).round(4).to_string(index=False))
    (out / "tracked.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in tracked) + "\n")


if __name__ == "__main__":
    main()
