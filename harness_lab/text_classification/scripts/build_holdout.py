"""Build the confirmatory Symptom2Disease holdout from the public source dataset.

The vendored splits (train 200, val 50, test 212) were drawn from
gretelai/symptom_to_diagnosis (train 853, test 212). This script checks that
provenance by exact match, then keeps every unused source row that is not an
exact or near duplicate (TF-IDF cosine >= THRESHOLD, the knn_top6 weighting) of
any vendored row or of an earlier kept row. No model is called.

  usage: PYTHONPATH=harness_lab python harness_lab/text_classification/scripts/build_holdout.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from text_classification.agents.tfidf_cover import tokens

THRESHOLD = 0.9  # fixed before the source rows were inspected
DATA = Path(__file__).resolve().parents[1] / "data"
VENDORED = DATA / "symptom_diagnosis"
HOLDOUT = DATA / "symptom_holdout"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def idf_table(texts: list[str]) -> dict[str, float]:
    df = Counter(t for text in texts for t in set(tokens(text)))
    n = len(texts) + 1
    return {t: math.log(n / (1 + c)) + 1 for t, c in df.items()}


def vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens(text))
    v = {t: (1 + math.log(f)) * idf.get(t, 0.0) for t, f in counts.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(w * large.get(t, 0.0) for t, w in small.items())


def main() -> int:
    vendored = {s: read_jsonl(VENDORED / f"{s}.jsonl") for s in ("train", "val", "test")}
    source_train = read_jsonl(HOLDOUT / "raw" / "train.jsonl")
    source_test = read_jsonl(HOLDOUT / "raw" / "test.jsonl")
    labels = sorted({r["answer"] for rows in vendored.values() for r in rows})

    # Provenance: vendored test == source test; vendored train/val come from source train.
    src_train_q = {r["input_text"] for r in source_train}
    provenance = {
        "test_equals_source_test": sorted(r["question"] for r in vendored["test"])
        == sorted(r["input_text"] for r in source_test),
        "train_subset_of_source_train": all(r["question"] in src_train_q for r in vendored["train"]),
        "val_subset_of_source_train": all(r["question"] in src_train_q for r in vendored["val"]),
    }
    if not all(provenance.values()):
        print(json.dumps({"status": "STOP", "provenance": provenance}, indent=2))
        return 1

    used = [r["question"] for rows in vendored.values() for r in rows]
    used_set = set(used)
    candidates = [{"question": r["input_text"], "answer": r["output_text"]} for r in source_train]
    idf = idf_table(used + [c["question"] for c in candidates])
    reference = [vector(q, idf) for q in used]

    reasons: Counter = Counter()
    kept: list[dict] = []
    kept_vecs: list[dict[str, float]] = []
    max_cos_to_vendored: list[float] = []
    for cand in candidates:
        q = cand["question"]
        if q in used_set:
            reasons["exact_vendored"] += 1
            continue
        if cand["answer"] not in labels:
            reasons["label_outside_22"] += 1
            continue
        v = vector(q, idf)
        best_ref = max(cosine(v, r) for r in reference)
        if best_ref >= THRESHOLD:
            reasons["near_duplicate_vendored"] += 1
            continue
        if any(q == k["question"] for k in kept):
            reasons["exact_within_holdout"] += 1
            continue
        if kept_vecs and max(cosine(v, k) for k in kept_vecs) >= THRESHOLD:
            reasons["near_duplicate_within_holdout"] += 1
            continue
        kept.append(cand)
        kept_vecs.append(v)
        max_cos_to_vendored.append(best_ref)

    out = HOLDOUT / "test.jsonl"
    with open(out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "OK",
        "threshold": THRESHOLD,
        "provenance": provenance,
        "source_rows": {"train": len(source_train), "test": len(source_test)},
        "vendored_rows": {s: len(rows) for s, rows in vendored.items()},
        "excluded": dict(reasons),
        "n_holdout": len(kept),
        "labels": len({r["answer"] for r in kept}),
        "label_counts": dict(sorted(Counter(r["answer"] for r in kept).items())),
        "max_cosine_to_vendored": round(max(max_cos_to_vendored), 4) if kept else None,
        "sha256": {
            "raw/train.jsonl": sha256(HOLDOUT / "raw" / "train.jsonl"),
            "raw/test.jsonl": sha256(HOLDOUT / "raw" / "test.jsonl"),
            "test.jsonl": sha256(out),
        },
    }
    (HOLDOUT / "DEDUP_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("provenance", "excluded", "n_holdout", "labels", "max_cosine_to_vendored")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
