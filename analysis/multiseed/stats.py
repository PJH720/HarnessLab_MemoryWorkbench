"""Pre-registered multi-seed analysis. Replay is local and cache-only by construction.

Run ``stats.py step`` repeatedly to advance the bounded DAG. A watch step is due
every 300 seconds; an early call does not consume an iteration. All writes stay
under this directory. No experiment runner or real LLM provider is imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "harness_lab/text_classification"
OUT = HERE / "out"
DEEP = LAB / "logs/deepen-20261101"
BASE = LAB / "logs/online-v2/Symptom2Disease"
LEAF = "qwen3.8-flash-next-nvfp4"
SYSTEMS = [
    "no_memory",
    "fewshot_all",
    "knn_top6",
    "contrast_online",
    "cue_bank",
    "reflexion_notes",
    "cue_bank_verified",
]
REFERENCE = "knn_top6"
SEEDS = [42, 123, 456]
ERROR_MEMORIES = ["cue_bank", "reflexion_notes", "cue_bank_verified"]
FAMILY = {
    s: ("none" if s == "no_memory" else "generated" if s in ERROR_MEMORIES else "raw")
    for s in SYSTEMS
}
MODEL = "openai/Inferact/Qwen3.8-Flash-Next-NVFP4"
ENDPOINT = "https://model-endpoint.example.invalid/v1"  # A cache-key string only; never opened.
RUN_COMMAND = [
    "uv",
    "run",
    "--project",
    "harness_lab/text_classification",
    "--with",
    "scipy",
    "--with",
    "statsmodels",
    "--with",
    "pandas",
    "--with",
    "pytest",
    "python",
]
ROW_COLUMNS = [
    "seed",
    "split",
    "sys",
    "qid",
    "gold",
    "pred",
    "correct",
    "input_tokens",
    "item_id",
    "row_status",
    "source",
    "cache_source",
]
_PREFIX = "_multiseed_cache_harness"


def write_json(path: Path, value: Any) -> None:
    path.resolve().relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_function(path: Path, name: str, namespace: dict) -> Callable:
    """Reuse an inspected function without running its module-level imports."""
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def safe_modules() -> dict:
    """Load the real memory classes with a provider-free LLM type shim.

    The distinct package name prevents importing text_classification.llm (which
    imports API clients). The memory implementations and ranker are unchanged.
    Python bytecode writes are disabled by the launch environment and here.
    """
    sys.dont_write_bytecode = True
    if _PREFIX not in sys.modules:
        for name, folder in [(_PREFIX, LAB), (_PREFIX + ".agents", LAB / "agents")]:
            module = types.ModuleType(name)
            module.__path__ = [str(folder)]
            sys.modules[name] = module
        shim = types.ModuleType(_PREFIX + ".llm")
        shim.LLMCallable = Callable
        sys.modules[shim.__name__] = shim
    return {s: importlib.import_module(f"{_PREFIX}.agents.{s}") for s in SYSTEMS}


def evaluator_functions() -> tuple[Callable, Callable]:
    namespace = {"json": json, "re": __import__("re")}
    extract = pure_function(LAB / "memory_system.py", "extract_json_field", namespace)
    pure_function(LAB / "data/evaluators.py", "extract_final_answer", namespace)
    evaluate = pure_function(
        LAB / "data/evaluators.py", "eval_symptom2disease", namespace
    )
    return extract, evaluate


def examples(seed: int, split: str) -> list[dict]:
    """Same fixed MCE split and seeded permutation as data/api.py:116-134."""
    path = LAB / "data/symptom_diagnosis" / f"{split}.jsonl"
    raw = [json.loads(s) for s in path.read_text().splitlines()]
    random.Random(seed).shuffle(raw)
    raw = raw[: 50 if split == "val" else 212]
    ns = {"_load_jsonl": lambda *_: raw, "_get_mce_split_path": lambda *_: path}
    build = pure_function(LAB / "data/loaders.py", "_load_symptom2disease", ns)
    return build(split)


class MissingCache(LookupError):
    pass


class CacheOnly:
    """Same request hash as llm.py:164-178, without a miss callback or provider."""

    def __init__(self, cache_dirs: list[Path]):
        self.cache_dirs = cache_dirs
        self.last_payload: dict = {}
        self.last_source = ""
        self.request_keys: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.last_payload = {}
        self.last_source = ""
        payload = {
            "version": 1,
            "model": MODEL,
            "api_base": ENDPOINT,
            "system_prompt": None,
            "prompt": prompt,
            "kwargs": {
                "temperature": 0.0,
                "max_tokens": 4096,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.request_keys.append(key)
        for folder in self.cache_dirs:
            p = folder / (key + ".json")
            if p.is_file():
                d = json.loads(p.read_text())
                if not isinstance(d.get("content"), str):
                    raise ValueError(f"Invalid cache payload: {p}")
                self.last_payload, self.last_source = d, str(p)
                return d["content"]
        raise MissingCache(key)


def memory_for(system: str, state: str, client: CacheOnly):
    modules = safe_modules()
    names = {
        "no_memory": "NoMemory",
        "fewshot_all": "FewShotAll",
        "knn_top6": "KnnTop6",
        "contrast_online": "ContrastOnline",
        "cue_bank": "CueBank",
        "reflexion_notes": "ReflexionNotes",
        "cue_bank_verified": "CueBankVerified",
    }
    memory = getattr(modules[system], names[system])(client)
    memory.set_state(state)
    return memory


def discover() -> list[dict]:
    runs = []
    cache = Path.home() / ".cache/text-classification/litellm-online"
    for seed in SEEDS:
        for system in SYSTEMS:
            for split in ["val", "test"]:
                directory = (
                    BASE / system / LEAF
                    if seed == 42 and split == "val"
                    else DEEP / "test42/Symptom2Disease" / system / LEAF
                    if seed == 42
                    else DEEP / "seeds/Symptom2Disease" / system / f"{LEAF}_seed{seed}"
                )
                memory = (
                    BASE / system / LEAF / "memory.json"
                    if seed == 42
                    else directory / "memory.json"
                )
                runs.append(
                    {
                        "seed": seed,
                        "split": split,
                        "sys": system,
                        "result": directory / f"{split}.json",
                        "memory": memory,
                        "cache_dirs": [cache],
                        "block": "baseline"
                        if seed == 42 and split == "val"
                        else "E1"
                        if seed == 42
                        else "E2",
                    }
                )
    for rep in [1, 2]:
        runs.append(
            {
                "seed": 42,
                "split": "val",
                "sys": "no_memory",
                "rep": rep,
                "result": DEEP / f"noise/rep{rep}/val.json",
                "memory": None,
                "cache_dirs": [DEEP / f"noise/cache-rep{rep}"],
                "block": "E3",
            }
        )
    return runs


def snapshot() -> dict:
    paths = {r["result"] for r in discover()}
    paths.update(r["memory"] for r in discover() if r["memory"] is not None)
    found = {}
    for p in sorted(paths):
        if p.is_file():
            try:
                st = p.stat()
                found[str(p.relative_to(ROOT))] = {
                    "mtime_ns": st.st_mtime_ns,
                    "bytes": st.st_size,
                    "sha256": digest(p),
                }
            except OSError:
                continue
    return found


def failure_features(
    table: pd.DataFrame, memories: dict, dataset: list[dict]
) -> pd.DataFrame:
    """Reuse the exact feature definition, including legacy right/wrong name sets."""
    ranker = importlib.import_module(_PREFIX + ".agents._text_rank").rank_texts
    fn = pure_function(
        ROOT / "analysis/harness_stats.py",
        "failure_features",
        {"pd": pd, "ERROR_MEMORIES": ERROR_MEMORIES, "rank_texts": ranker},
    )
    return fn(table, memories, dataset)


def load_run(spec: dict) -> tuple[list[dict], list[dict], dict]:
    """Discover aggregate/embedded schemas; otherwise reconstruct from disk cache."""
    p = spec["result"]
    meta = {
        k: v for k, v in spec.items() if k not in ["result", "memory", "cache_dirs"]
    }
    meta.update(
        path=str(p.relative_to(ROOT)),
        status="absent",
        schema=None,
        rows=0,
        missing_cache=0,
        feature_missing=0,
    )
    if not p.is_file():
        return [], [], meta
    try:
        d = json.loads(p.read_text())
        if not isinstance(d, dict):
            raise ValueError("result must be an object")
        meta["schema"] = sorted(d)
        for key in [
            "accuracy",
            "correct",
            "total",
            "llm_input_tokens",
            "llm_output_tokens",
            "model",
            "seed",
        ]:
            if key in d:
                meta["logged_" + key] = d[key]
        if "seed" in d and int(d["seed"]) != spec["seed"]:
            raise ValueError("result seed disagrees with run path")
        if "memory" in d and Path(d["memory"]).stem != spec["sys"]:
            raise ValueError("result system disagrees with run path")
        dataset = examples(spec["seed"], spec["split"])
        if int(d.get("total", len(dataset))) != len(dataset):
            raise ValueError("result sample count disagrees with split")
        state = "{}"
        if spec["memory"] is not None:
            if not spec["memory"].is_file() and spec["sys"] != "no_memory":
                meta["status"] = "missing_memory"
                return [], [], meta
            if spec["memory"].is_file():
                state = spec["memory"].read_text()
        client = CacheOnly(spec["cache_dirs"])
        memory = memory_for(spec["sys"], state, client)
        _, evaluate = evaluator_functions()
        embedded = d.get("predictions")
        if embedded is not None and (
            not isinstance(embedded, list) or len(embedded) != len(dataset)
        ):
            raise ValueError("unrecognized embedded prediction schema")
        output = []
        for qid, ex in enumerate(dataset):
            row = dict(
                seed=spec["seed"],
                split=spec["split"],
                sys=spec["sys"],
                qid=qid,
                gold=ex["target"],
                pred=None,
                correct=None,
                input_tokens=None,
                item_id=hashlib.sha256(ex["raw_question"].encode()).hexdigest(),
                row_status="missing_cache",
                source=meta["path"],
                cache_source="",
            )
            try:
                if embedded is None:
                    pred, _ = memory.predict(ex["input"])
                    row["cache_source"] = client.last_source
                    row["input_tokens"] = client.last_payload.get("input_tokens")
                else:
                    item = embedded[qid]
                    if not isinstance(item, dict) or not any(
                        k in item for k in ["pred", "prediction"]
                    ):
                        raise ValueError(
                            "embedded predictions require pred or prediction"
                        )
                    gold = item.get("gold", item.get("target", ex["target"]))
                    if gold != ex["target"]:
                        raise ValueError(
                            "embedded gold disagrees with seeded dataset order"
                        )
                    pred = item.get("pred", item.get("prediction"))
                    row["input_tokens"] = item.get("input_tokens")
                row.update(
                    pred=pred,
                    correct=bool(evaluate(pred, ex["target"])),
                    row_status="ok",
                )
            except MissingCache:
                meta["missing_cache"] += 1
            output.append(row)
        good = [r for r in output if r["row_status"] == "ok"]
        meta["rows"] = len(good)
        meta["recomputed_correct"] = sum(r["correct"] for r in good)
        meta["status"] = "complete" if len(good) == len(dataset) else "missing_cache"
        if meta["status"] == "complete" and (
            meta["recomputed_correct"] != int(d["correct"])
            or abs(meta["recomputed_correct"] / len(dataset) - float(d["accuracy"]))
            > 1e-9
        ):
            meta["status"] = "aggregate_mismatch"
        if meta["status"] != "complete":
            for row in output:
                if row["row_status"] == "ok":
                    row["row_status"] = "unvalidated_run"
        features = []
        if spec["sys"] in ERROR_MEMORIES:
            table = pd.DataFrame(
                [
                    {
                        "q": r["qid"],
                        "system": r["sys"],
                        "pred": r["pred"] or "",
                        "correct": r["correct"] if r["correct"] is not None else False,
                    }
                    for r in output
                ]
            )
            try:
                feature_table = failure_features(table, {spec["sys"]: memory}, dataset)
                for feat, row in zip(feature_table.to_dict("records"), output):
                    feat.update(
                        seed=spec["seed"],
                        split=spec["split"],
                        qid=row["qid"],
                        feature_status="ok"
                        if row["row_status"] == "ok"
                        else "missing_outcome",
                    )
                    if row["row_status"] != "ok":
                        feat["failed"] = None
                    features.append(feat)
            except (AttributeError, KeyError, ValueError) as exc:
                meta["feature_missing"] = len(dataset)
                meta["feature_error"] = str(exc)
        return output, features, meta
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        meta.update(status="invalid_or_in_progress", error=str(exc))
        return [], [], meta


def load_run_dir(
    directory: Path,
    split: str,
    *,
    seed: int | None = None,
    system: str | None = None,
    cache_dirs: list[Path] | None = None,
):
    """Public loader for any result directory; infer metadata from its JSON."""
    directory = Path(directory)
    result = directory / f"{split}.json"
    data = json.loads(result.read_text())
    seed = int(data["seed"]) if seed is None else seed
    system = Path(data["memory"]).stem if system is None else system
    if system not in SYSTEMS or split not in ["val", "test"]:
        raise ValueError("Unknown system or split")
    memory = directory / "memory.json"
    if not memory.exists() and seed == 42:
        memory = BASE / system / LEAF / "memory.json"
    return load_run(
        dict(
            seed=seed,
            split=split,
            sys=system,
            result=result,
            memory=memory,
            cache_dirs=cache_dirs
            or [Path.home() / ".cache/text-classification/litellm-online"],
            block="custom",
        )
    )


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    all_rows, all_features, noise_rows, runs = [], [], [], []
    before = snapshot()
    for spec in discover():
        output, features, meta = load_run(spec)
        runs.append(meta)
        if spec["block"] == "E3":
            noise_rows.extend(dict(row, rep=spec["rep"]) for row in output)
        else:
            all_rows.extend(output)
            all_features.extend(features)
    after = snapshot()
    expected = [r for r in runs if r["block"] != "baseline"]
    coverage = {
        "present_files": after,
        "snapshot_stable": before == after,
        "runs": runs,
        "expected_result_files": 37,
        "present_result_files": sum((ROOT / r["path"]).is_file() for r in expected),
        "analysable_runs": sum(r["status"] == "complete" for r in expected),
        "complete": all(r["status"] == "complete" for r in expected)
        and before == after,
        "missing_feature_rows": sum(r["feature_missing"] for r in runs),
        "sampling_note": "Seeds permute the same pre-existing val/test sets; they do not create disjoint patient samples (data/api.py:116-134). Clustering follows the registered (seed,qid) unit; residual cross-seed item dependence is not removed.",
        "token_note": "Per-question input_tokens come from cache payloads, not divided aggregate totals. Logged totals omit cache hits (llm.py:255-266); do not interpret them as monetary cost.",
        "source": "report/analysis_20261101_2040_preregistration.md",
        "cache_only": True,
    }
    return (
        pd.DataFrame(all_rows, columns=ROW_COLUMNS),
        pd.DataFrame(all_features),
        coverage,
        pd.DataFrame(noise_rows),
    )


def exact_mcnemar(b: int, c: int) -> float:
    if b < 0 or c < 0 or int(b) != b or int(c) != c:
        raise ValueError("discordance counts must be nonnegative integers")
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def matched_mh(strata: list[tuple[int, int]]) -> dict:
    """Seed-stratified matched-pair MH score, conditional on discordances.

    In seed s, E[b_s | d_s]=d_s/2, Var[b_s | d_s]=d_s/4.
    Thus X2=(sum(b_s-c_s))**2/sum(b_s+c_s), without continuity correction.
    This tests marginal symmetry, unlike CMH association on paired correctness
    cross-tabs. Its exact conditional counterpart is the pooled binomial test.
    """
    for b, c in strata:
        exact_mcnemar(b, c)
    b, c = sum(s[0] for s in strata), sum(s[1] for s in strata)
    stat = (b - c) ** 2 / (b + c) if b + c else 0.0
    return {
        "only_ref": b,
        "only_sys": c,
        "mh_statistic": stat,
        "mh_p": float(chi2.sf(stat, 1)) if b + c else 1.0,
        "pooled_exact_p": exact_mcnemar(b, c),
    }


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Keep the registered family size, treating missing tests as p=1 internally."""
    values = [1.0 if p is None else float(p) for p in pvalues]
    order = np.argsort(values, kind="stable")
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, idx in enumerate(order):
        previous = max(previous, (len(values) - rank) * values[idx])
        if pvalues[idx] is not None:
            adjusted[idx] = min(1.0, previous)
    return adjusted


def paired_bootstrap(
    strata: list[np.ndarray], draws: int = 10_000, seed: int = 0
) -> dict:
    """Stratified paired percentile interval; effects are system minus knn_top6."""
    if not strata or any(len(s) == 0 for s in strata):
        raise ValueError("bootstrap needs nonempty paired strata")
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    size = sum(len(s) for s in strata)
    for stratum in strata:
        values = np.asarray(stratum, dtype=float)
        indexes = rng.integers(0, len(values), size=(draws, len(values)))
        totals += values[indexes].sum(axis=1)
    effects = totals / size
    return {
        "diff_vs_ref": float(sum(np.sum(s) for s in strata) / size),
        "ci_low": float(np.quantile(effects, 0.025)),
        "ci_high": float(np.quantile(effects, 0.975)),
        "draws": draws,
        "rng_seed": seed,
    }


def usable(table: pd.DataFrame) -> pd.DataFrame:
    return table[table.row_status == "ok"].copy()


def paired(table: pd.DataFrame, system: str) -> pd.DataFrame:
    ref = table[table.sys == REFERENCE]
    other = table[table.sys == system]
    pair = ref.merge(
        other,
        on=["seed", "split", "qid", "gold", "item_id"],
        suffixes=("_ref", "_sys"),
        validate="one_to_one",
    )
    return pair.sort_values(["seed", "qid"])


def parity(table: pd.DataFrame) -> dict:
    """All 21 ordered pairs, exact equality, with no tolerance or p-value tuning."""
    ref = pd.read_csv(ROOT / "analysis/out/mcnemar.csv")
    selected = usable(table)
    selected = selected[(selected.seed == 42) & (selected.split == "val")]
    wide = selected.pivot(index="qid", columns="sys", values="correct").reindex(
        columns=SYSTEMS
    )
    if wide.shape != (50, 7) or wide.isna().any().any():
        return {
            "passed": False,
            "diff": ["seed-42 val replay is not complete (50 x 7)"],
        }
    wide = wide.astype(bool)
    # Reuse the original exact McNemar and column order, not a parity-specific statistic.
    fn = pure_function(
        ROOT / "analysis/harness_stats.py",
        "mcnemar_table",
        {"pd": pd, "itertools": itertools, "binomtest": binomtest},
    )
    actual = fn(wide)
    actual.to_csv(OUT / "parity_mcnemar.csv", index=False)
    diffs = []
    for idx in ref.index:
        for col in ref.columns:
            if actual.loc[idx, col] != ref.loc[idx, col]:
                diffs.append(
                    {
                        "row": int(idx) + 2,
                        "column": col,
                        "expected": str(ref.loc[idx, col]),
                        "actual": str(actual.loc[idx, col]),
                    }
                )
    reference_rows = pd.read_csv(ROOT / "analysis/out/per_question.csv")
    for system in SYSTEMS:
        expected = reference_rows[reference_rows.system == system].sort_values("q")
        got = selected[selected.sys == system].sort_values("qid")
        if (
            got.pred.tolist() != expected.pred.tolist()
            or got.gold.tolist() != expected.target.tolist()
        ):
            diffs.append(
                {"system": system, "reason": "per-question prediction or gold differs"}
            )
    return {
        "passed": not diffs,
        "comparisons": len(actual),
        "question_rows": len(selected),
        "exact_numeric_equality": not diffs,
        "byte_identical_csv": digest(OUT / "parity_mcnemar.csv")
        == digest(ROOT / "analysis/out/mcnemar.csv"),
        "reference_sha256": digest(ROOT / "analysis/out/mcnemar.csv"),
        "diff": diffs,
    }


def contrast_tables(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = usable(table)
    by_seed, pooled, boots = [], [], []
    for split in ["val", "test"]:
        other_systems = [s for s in SYSTEMS if s != REFERENCE]
        for seed in SEEDS:
            block = valid[(valid.seed == seed) & (valid.split == split)]
            block_rows = []
            for system in other_systems:
                pairs = paired(block, system)
                n = len(pairs)
                if not n:
                    block_rows.append(
                        dict(seed=seed, split=split, sys=system, n=0, p_exact=None)
                    )
                    continue
                a = pairs.correct_ref.to_numpy(dtype=bool)
                b = pairs.correct_sys.to_numpy(dtype=bool)
                x, y = int(np.sum(a & ~b)), int(np.sum(~a & b))
                block_rows.append(
                    dict(
                        seed=seed,
                        split=split,
                        sys=system,
                        n=n,
                        acc_ref=float(a.mean()),
                        acc_sys=float(b.mean()),
                        only_ref=x,
                        only_sys=y,
                        p_exact=exact_mcnemar(x, y),
                    )
                )
            adjusted = holm([r["p_exact"] for r in block_rows])
            by_seed.extend(dict(r, p_holm=p) for r, p in zip(block_rows, adjusted))
        pooled_rows = []
        for system in other_systems:
            pairs = paired(valid[valid.split == split], system)
            groups = [group for _, group in pairs.groupby("seed", sort=True)]
            if not groups:
                pooled_rows.append(
                    dict(
                        split=split,
                        sys=system,
                        n=0,
                        n_seeds=0,
                        mh_p=None,
                        pooled_exact_p=None,
                    )
                )
                continue
            strata, effects = [], []
            for group in groups:
                a, b = (
                    group.correct_ref.to_numpy(dtype=bool),
                    group.correct_sys.to_numpy(dtype=bool),
                )
                strata.append((int(np.sum(a & ~b)), int(np.sum(~a & b))))
                effects.append(b.astype(float) - a.astype(float))
            result = dict(
                split=split,
                sys=system,
                n=len(pairs),
                n_seeds=len(groups),
                seeds=",".join(str(s) for s in sorted(pairs.seed.unique())),
                status="complete" if len(groups) == 3 else "partial",
                **matched_mh(strata),
            )
            pooled_rows.append(result)
            boots.append(
                dict(
                    split=split,
                    sys=system,
                    n=len(pairs),
                    n_seeds=len(groups),
                    status=result["status"],
                    **paired_bootstrap(effects),
                )
            )
        for key in ["mh_p", "pooled_exact_p"]:
            adjusted = holm([r[key] for r in pooled_rows])
            for row, p in zip(pooled_rows, adjusted):
                row[key + "_holm"] = p
        pooled.extend(pooled_rows)
    return pd.DataFrame(by_seed), pd.DataFrame(pooled), pd.DataFrame(boots)


def clustered_logit(
    table: pd.DataFrame, features: pd.DataFrame, hypothesis: str
) -> str:
    blocks = []
    for split in ["test", "val"]:
        if hypothesis == "H2":
            data = usable(table)
            data = data[data.split == split].copy()
            if data.empty:
                blocks.append(f"{split}: INCOMPLETE (no analysable rows)")
                continue
            x = pd.DataFrame(
                {
                    "raw": data.sys.map(FAMILY).eq("raw").astype(float),
                    "generated": data.sys.map(FAMILY).eq("generated").astype(float),
                },
                index=data.index,
            )
            y = data.correct.astype(float)
        else:
            if features.empty:
                blocks.append(f"{split}: INCOMPLETE (no cached feature/outcome pairs)")
                continue
            data = features[
                (features.split == split) & (features.feature_status == "ok")
            ].copy()
            if data.empty:
                blocks.append(f"{split}: INCOMPLETE (no usable features)")
                continue
            x = data[["max_similarity", "target_in_cues", "n_other_diseases"]].astype(
                float
            )
            x["prompt_kchars"] = data.prompt_chars / 1000
            x = pd.concat(
                [x, pd.get_dummies(data.system, drop_first=True, dtype=float)], axis=1
            )
            y = data.failed.astype(float)
        groups = pd.factorize(pd.MultiIndex.from_frame(data[["seed", "qid"]]))[0]
        try:
            x = sm.add_constant(x, has_constant="add")
            with warnings.catch_warnings(record=True) as caught:
                fit = sm.Logit(y, x).fit(
                    disp=False, cov_type="cluster", cov_kwds={"groups": groups}
                )
            block = [
                f"{hypothesis} {split}; rows={len(data)}; clusters={len(set(groups))}; cluster=(seed,qid)",
                "Available-data descriptive fit; complete coverage is reported separately.",
                str(fit.summary()),
            ]
            if hypothesis == "H2":
                contrast = np.zeros(len(x.columns))
                contrast[x.columns.get_loc("raw")] = 1
                contrast[x.columns.get_loc("generated")] = -1
                block += [
                    "H2 raw-minus-generated log-odds contrast:",
                    str(fit.t_test(contrast)),
                ]
            block.extend("WARNING: " + str(w.message) for w in caught)
            if not fit.mle_retvals.get("converged", False):
                block.append(
                    "INCOMPLETE: optimization did not converge; no inferential conclusion."
                )
            blocks.append("\n".join(block))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
            blocks.append(f"{hypothesis} {split}: INCOMPLETE (fit failed: {exc})")
    return "\n\n".join(blocks) + "\n"


def noise_table(table: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    valid = usable(table)
    baseline = valid[
        (valid.seed == 42) & (valid.split == "val") & (valid.sys == "no_memory")
    ]
    sources = {"online_v2": baseline}
    for rep in [1, 2]:
        sources[f"rep{rep}"] = (
            noise[(noise.rep == rep) & (noise.row_status == "ok")]
            if not noise.empty
            else pd.DataFrame()
        )
    output = []
    for a, b in [("rep1", "rep2"), ("online_v2", "rep1"), ("online_v2", "rep2")]:
        left, right = sources[a], sources[b]
        row = dict(a=a, b=b, n=0, status="incomplete")
        if not left.empty and not right.empty:
            pairs = left.merge(
                right,
                on=["qid", "gold", "item_id"],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            n = len(pairs)
            changed = int((pairs.correct_a != pairs.correct_b).sum())
            pred_changed = int(
                (
                    pairs.pred_a.str.strip().str.lower()
                    != pairs.pred_b.str.strip().str.lower()
                ).sum()
            )
            row.update(
                n=n,
                status="complete" if n == 50 else "incomplete",
                correctness_flips=changed,
                correctness_flip_rate=changed / n if n else None,
                raw_prediction_flips=pred_changed,
                raw_prediction_flip_rate=pred_changed / n if n else None,
            )
        output.append(row)
    return pd.DataFrame(output)


def decide_h1(
    by_seed: pd.DataFrame, pooled: pd.DataFrame, complete: bool
) -> tuple[str, str]:
    if not complete:
        return "INCOMPLETE", "Required run coverage or cache replay is incomplete."
    h1 = pooled[(pooled.split == "test") & (pooled.sys == "cue_bank")]
    seeds = by_seed[(by_seed.split == "test") & (by_seed.sys == "cue_bank")]
    if len(h1) != 1 or len(seeds) != 3 or int(h1.iloc[0].n_seeds) != 3:
        return "INCOMPLETE", "Primary test comparison does not contain all three seeds."
    row = h1.iloc[0]
    deltas = seeds.acc_ref - seeds.acc_sys
    mixed = bool((deltas > 0).any() and (deltas < 0).any())
    support = row.mh_p < 0.05 and int((deltas >= 0).sum()) >= 2
    reject = row.mh_p >= 0.05 or mixed
    if support and reject:
        return (
            "INCOMPLETE",
            "Section 4 rule conflict: support (>=2 nonnegative seeds) and rejection (mixed directions) both apply. No post-hoc precedence chosen.",
        )
    if support:
        return (
            "SUPPORTED",
            "Primary matched-MH p<.05 and at least two seed directions are nonnegative.",
        )
    return (
        "NOT SUPPORTED",
        "Primary p>=.05 or seed directions fail the registered support rule; the earlier consistent trend may reflect sample noise.",
    )


def analyse() -> dict:
    table, features, coverage, noise = load_all()
    by_seed, pooled, boot = contrast_tables(table)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("per_question", table),
        ("mcnemar_by_seed", by_seed),
        ("pooled", pooled),
        ("bootstrap", boot),
        ("noise", noise_table(table, noise)),
        ("features", features),
    ]:
        frame.to_csv(OUT / (name + ".csv"), index=False)
    (OUT / "logit_h2.txt").write_text(clustered_logit(table, features, "H2"))
    (OUT / "logit_h3.txt").write_text(clustered_logit(table, features, "H3"))
    valid = usable(table)
    accuracy_rows = []
    for (seed, split, system), group in valid.groupby(["seed", "split", "sys"]):
        tokens = group.input_tokens.dropna()
        accuracy_rows.append(
            dict(
                seed=seed,
                split=split,
                sys=system,
                n=len(group),
                correct=int(group.correct.sum()),
                accuracy=float(group.correct.mean()),
                cached_eval_input_tokens=int(tokens.sum())
                if len(tokens) == len(group)
                else None,
            )
        )
    accuracy = pd.DataFrame(accuracy_rows)
    accuracy.to_csv(OUT / "accuracy_tokens.csv", index=False)
    coverage["decision"], coverage["decision_reason"] = decide_h1(
        by_seed, pooled, coverage["complete"]
    )
    coverage["methods"] = {
        "mh": "Matched-pair seed-stratified score; sum(b-c)^2/sum(b+c); chi-square df=1; no continuity correction",
        "exact": "Two-sided exact pooled-discordance binomial; H1 unadjusted",
        "holm": "Six registered comparisons per split; by-seed exact, pooled MH, and pooled exact adjusted separately",
        "bootstrap": "10,000 paired draws; resample questions within seed; RNG seed=0; system-minus-knn effect",
        "h2": "correct ~ intercept + raw family + generated family, none baseline; raw-minus-generated contrast",
        "h3": "Original features and system dummies; failure outcome; cluster=(seed,qid)",
    }
    write_json(OUT / "coverage.json", coverage)
    # Keep a coherent preview even if the iteration cap falls immediately after
    # this engine pass. G6 still performs the explicit summary DAG transition.
    summary()
    return coverage


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    frame = frame.reindex(columns=columns)

    def cell(value):
        if pd.isna(value):
            return "missing"
        if isinstance(value, (float, np.floating)):
            return f"{value:.8g}"
        return str(value).replace("|", "/").replace("\n", " ")

    return (
        "| "
        + " | ".join(columns)
        + " |\n|"
        + "|".join("---" for _ in columns)
        + "|\n"
        + "".join(
            "| " + " | ".join(cell(v) for v in row) + " |\n"
            for row in frame.itertuples(index=False, name=None)
        )
    )


def summary() -> None:
    coverage = json.loads((OUT / "coverage.json").read_text())
    by_seed, pooled = (
        pd.read_csv(OUT / "mcnemar_by_seed.csv"),
        pd.read_csv(OUT / "pooled.csv"),
    )
    h1_seeds = by_seed[by_seed.sys == "cue_bank"]
    run_table = pd.DataFrame(coverage["runs"])
    status = (
        run_table.groupby(["block", "status"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    notes = pd.DataFrame(
        [
            {
                "item": "Primary analysis",
                "detail": "test; seed42 standalone exact; all-seed matched-MH and exact pooled-discordance; H1 is unadjusted, Holm is supplementary",
            },
            {
                "item": "Incomplete comparisons",
                "detail": "Missing caches/results are not errors or imputed answers; partial p-values do not decide H1.",
            },
            {"item": "Repeated items", "detail": coverage["sampling_note"]},
            {"item": "Token accounting", "detail": coverage["token_note"]},
            {
                "item": "Regression",
                "detail": "H2/H3 are cluster-robust by (seed,qid); fits and failure warnings are in logit_h2.txt/logit_h3.txt.",
            },
            {
                "item": "Direction-rule ambiguity",
                "detail": "If section 4 support and rejection conditions overlap, decision remains INCOMPLETE pending clarification.",
            },
            {"item": "Decision reason", "detail": coverage["decision_reason"]},
        ]
    )
    legacy_h1 = h1_seeds[(h1_seeds.seed == 42) & (h1_seeds.split == "val")]
    if not legacy_h1.empty and legacy_h1.iloc[0].p_exact >= 0.05:
        notes.loc[len(notes)] = {
            "item": "Auxiliary seed42 validation H1",
            "detail": f"Exact McNemar p={legacy_h1.iloc[0].p_exact:.8g}, n={int(legacy_h1.iloc[0].n)}: no significant difference was found. This does not decide the primary test hypothesis.",
        }
    blocks = [
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "expected_new_results": 37,
                        "present": coverage["present_result_files"],
                        "analysable": coverage["analysable_runs"],
                        "complete": coverage["complete"],
                    }
                ]
            ),
            ["expected_new_results", "present", "analysable", "complete"],
        ),
        markdown_table(status, ["block", "status", "runs"]),
        markdown_table(
            pd.read_csv(OUT / "accuracy_tokens.csv"),
            [
                "seed",
                "split",
                "sys",
                "n",
                "correct",
                "accuracy",
                "cached_eval_input_tokens",
            ],
        ),
        markdown_table(
            h1_seeds,
            [
                "seed",
                "split",
                "n",
                "acc_ref",
                "acc_sys",
                "only_ref",
                "only_sys",
                "p_exact",
                "p_holm",
            ],
        ),
        markdown_table(
            pooled,
            [
                "split",
                "sys",
                "status",
                "n",
                "n_seeds",
                "mh_p",
                "pooled_exact_p",
                "mh_p_holm",
                "pooled_exact_p_holm",
            ],
        ),
        markdown_table(
            pd.read_csv(OUT / "bootstrap.csv"),
            ["split", "sys", "n_seeds", "diff_vs_ref", "ci_low", "ci_high", "draws"],
        ),
        markdown_table(
            pd.read_csv(OUT / "noise.csv"),
            ["a", "b", "n", "status", "correctness_flips", "correctness_flip_rate"],
        ),
        markdown_table(notes, ["item", "detail"]),
    ]
    (OUT / "SUMMARY.md").write_text(
        "\n".join(blocks)
        + f"\nDecision per prereg section 4: H1 {coverage['decision']}.\n"
    )


def initialize_graph() -> dict:
    path = HERE / "graph.json"
    if path.is_file():
        return json.loads(path.read_text())
    graph = {
        "nodes": [
            {"id": f"G{i}", "status": "todo", "outputs": [], "notes": ""}
            for i in range(1, 7)
        ],
        "iteration": 0,
        "parity_attempts": 0,
        "watch_checks": 0,
        "next_watch_at": None,
        "snapshots": [],
        "prereg_sha256": digest(
            ROOT / "report/analysis_20261101_2040_preregistration.md"
        ),
    }
    write_json(path, graph)
    return graph


def needs_refresh(current: dict, previous: dict | None, coverage: dict) -> bool:
    """Retry an unstable read even if its ending snapshot has since stopped moving."""
    return current != previous or not coverage.get("snapshot_stable", True)


def execution_limits(graph: dict) -> tuple[int, int]:
    """Operational budgets only; never use p-values to choose these limits."""
    configured = graph.get("limits", {})
    values = (
        configured.get("max_iterations", 20),
        configured.get("max_watch_checks", 12),
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Execution limits must be positive integers")
    return values


def step() -> None:
    graph = initialize_graph()
    max_iterations, max_checks = execution_limits(graph)
    if graph["iteration"] >= max_iterations:
        print(
            f"STOP: {max_iterations} iterations reached; consult coverage.json for incompleteness."
        )
        return
    nodes = {n["id"]: n for n in graph["nodes"]}
    if nodes["G2"]["status"] == "blocked":
        print("STOP: parity gate blocked; fix loader only before retrying.")
        return
    if any(n["status"] == "blocked" for n in nodes.values()):
        print("STOP: a required gate is blocked; see graph.json.")
        return
    if all(n["status"] == "done" for n in nodes.values()):
        print("STOP: DAG finished; coverage.json records whether coverage is complete.")
        return
    node = next((n for n in graph["nodes"] if n["status"] == "todo"), None)
    if node is None:
        print("STOP: no runnable node.")
        return
    name = node["id"]
    if name == "G5" and graph["next_watch_at"] and time.time() < graph["next_watch_at"]:
        print(
            f"WATCH_NOT_DUE: {max(0, graph['next_watch_at'] - time.time()):.0f} seconds remaining"
        )
        return
    node["status"] = "running"
    write_json(HERE / "graph.json", graph)
    note = ""
    if name == "G1":
        table, features, coverage, _ = load_all()
        table.to_csv(OUT / "loader_rows.csv", index=False)
        features.to_csv(OUT / "loader_features.csv", index=False)
        write_json(OUT / "loader_coverage.json", coverage)
        node["outputs"] = [
            "out/loader_rows.csv",
            "out/loader_features.csv",
            "out/loader_coverage.json",
        ]
        note = "Discovered aggregate result schema; cache-only replay and original features implemented."
    elif name == "G2":
        graph["parity_attempts"] += 1
        table, _, _, _ = load_all()
        result = parity(table)
        write_json(OUT / "parity.json", result)
        node["outputs"] = ["out/parity.json", "out/parity_mcnemar.csv"]
        node["status"] = "done" if result["passed"] else "blocked"
        note = (
            "PASS: exact equality for 21 McNemar comparisons and 350 question rows."
            if result["passed"]
            else "FAIL: loader diagnosis required; all downstream nodes are gated."
        )
    elif name == "G3":
        if not json.loads((OUT / "parity.json").read_text())["passed"]:
            raise RuntimeError("Parity must pass before tests/engine")
        env = dict(
            os.environ,
            UV_CACHE_DIR=str(HERE / ".uv-cache"),
            UV_OFFLINE="1",
            PYTHONDONTWRITEBYTECODE="1",
            UV_NO_SYNC="1",
        )
        command = RUN_COMMAND + [
            "-m",
            "pytest",
            "analysis/multiseed/test_stats.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(HERE / ".pytest-tmp"),
        ]
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True
        )
        (OUT / "pytest.txt").write_text(
            "Command: " + " ".join(command) + "\n" + result.stdout + result.stderr
        )
        node["outputs"] = ["out/pytest.txt"]
        if result.returncode:
            node["status"] = "blocked"
            note = "Tests failed; engine remains gated."
        else:
            note = next(
                (s for s in result.stdout.splitlines()[::-1] if "passed" in s),
                "Tests passed.",
            )
    elif name == "G4":
        if nodes["G2"]["status"] != "done" or nodes["G3"]["status"] != "done":
            raise RuntimeError("Parity and tests are required")
        coverage = analyse()
        graph["last_analysis_snapshot"] = coverage["present_files"]
        graph["snapshots"].append(
            {
                "iteration": graph["iteration"] + 1,
                "time": time.time(),
                "files": coverage["present_files"],
                "complete": coverage["complete"],
            }
        )
        node["outputs"] = [
            "out/" + n
            for n in [
                "per_question.csv",
                "mcnemar_by_seed.csv",
                "pooled.csv",
                "bootstrap.csv",
                "logit_h2.txt",
                "logit_h3.txt",
                "noise.csv",
                "coverage.json",
                "accuracy_tokens.csv",
            ]
        ]
        note = f"Analysed {coverage['analysable_runs']}/37 new runs; H1 {coverage['decision']}."
    elif name == "G5":
        graph["watch_checks"] += 1
        current = snapshot()
        last_coverage = json.loads((OUT / "coverage.json").read_text())
        changed = needs_refresh(
            current, graph.get("last_analysis_snapshot"), last_coverage
        )
        graph["watch_snapshot"] = current
        graph["next_watch_at"] = time.time() + 300
        if changed:
            nodes["G4"]["status"] = "todo"
            nodes["G6"]["status"] = "todo"
        elif nodes["G6"]["status"] == "done" and graph["watch_checks"] < max_checks:
            # Keep watching without needlessly regenerating an identical summary.
            node["status"] = "todo"
        node["outputs"] = ["graph.json snapshots"]
        note = f"Watch check {graph['watch_checks']}/{max_checks}; files changed={changed}."
    elif name == "G6":
        summary()
        node["outputs"] = ["out/SUMMARY.md"]
        coverage = json.loads((OUT / "coverage.json").read_text())
        note = f"Summary written: H1 {coverage['decision']}; complete coverage={coverage['complete']}."
        if (
            not coverage["complete"]
            and graph["watch_checks"] < max_checks
            and graph["iteration"] + 1 < max_iterations
        ):
            nodes["G5"]["status"] = "todo"
    if node["status"] == "running":
        node["status"] = "done"
    node["notes"] = note
    graph["iteration"] += 1
    write_json(HERE / "graph.json", graph)
    with (HERE / "loop_log.md").open("a") as f:
        next_node = next(
            (n["id"] for n in graph["nodes"] if n["status"] == "todo"), "stop"
        )
        f.write(f"iter {graph['iteration']} | {name} | {note} | {next_node}\n")
    print(f"iter {graph['iteration']} | {name} | {note} | next {next_node}")


def watch_loop() -> None:
    """Advance the DAG; wait locally without checking inputs before five minutes."""
    while True:
        graph = initialize_graph()
        max_iterations, max_checks = execution_limits(graph)
        if graph["iteration"] >= max_iterations or any(
            n["status"] == "blocked" for n in graph["nodes"]
        ):
            break
        todo = [n for n in graph["nodes"] if n["status"] == "todo"]
        if not todo:
            break
        if todo[0]["id"] == "G5" and graph["next_watch_at"]:
            delay = graph["next_watch_at"] - time.time()
            if delay > 0:
                time.sleep(min(30, delay))
                continue
        step()
        sys.stdout.flush()
    graph = initialize_graph()
    max_iterations, max_checks = execution_limits(graph)
    coverage = (
        json.loads((OUT / "coverage.json").read_text())
        if (OUT / "coverage.json").is_file()
        else {}
    )
    graph["stop_reason"] = (
        "complete coverage and G6 summary"
        if coverage.get("complete")
        else f"{max_iterations}-iteration cap; INCOMPLETE"
        if graph["iteration"] >= max_iterations
        else f"{max_checks}-watch-check cap or blocked gate; INCOMPLETE"
    )
    write_json(HERE / "graph.json", graph)
    print(graph["stop_reason"], flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["step", "status", "watch"])
    args = parser.parse_args()
    if args.command == "step":
        OUT.mkdir(parents=True, exist_ok=True)
        step()
    elif args.command == "watch":
        OUT.mkdir(parents=True, exist_ok=True)
        watch_loop()
    else:
        print(json.dumps(initialize_graph(), indent=2))
