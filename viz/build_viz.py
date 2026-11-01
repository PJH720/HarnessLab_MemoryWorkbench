"""Fill viz/template.html with measured results from the Meta-Harness run.

Every number comes from the run's val.json/test.json files; structure.json holds
the hand-checked description of each harness's five stages.

Usage: uv run python viz/build_viz.py [run_name]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "harness_lab/text_classification/logs"
DATASET = "Symptom2Disease"
MODEL_LEAF = "qwen3.8-flash-next-nvfp4"
REFERENCE = "fewshot_all"
BASELINES = ("no_memory", "fewshot_all")
METRIC_KEYS = (
    "accuracy",
    "correct",
    "total",
    "llm_calls",
    "llm_input_tokens",
    "llm_output_tokens",
    "memory_context_chars",
    "runtime_seconds",
)


def load_split(system_dir: Path, split: str) -> dict | None:
    path = system_dir / MODEL_LEAF / f"{split}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {k: data.get(k) for k in METRIC_KEYS}


def collect_systems(run_name: str, structure: list[dict]) -> list[dict]:
    labels = {s["id"]: s["label"] for s in structure}
    systems = []
    for system_dir in sorted((RUN_DIR / run_name / DATASET).iterdir()):
        val = load_split(system_dir, "val")
        if val is None:
            continue
        sid = system_dir.name
        systems.append(
            {
                "id": sid,
                "label": labels.get(sid, sid),
                "kind": "baseline" if sid in BASELINES else "candidate",
                "val": val,
                "test": load_split(system_dir, "test"),
            }
        )
    systems.sort(key=lambda s: (s["kind"] != "baseline", s["id"] != "no_memory"))
    return systems


def build_payload(run_name: str) -> dict:
    structure = json.loads((ROOT / "viz/structure.json").read_text())
    systems = collect_systems(run_name, structure)
    candidates = [s for s in systems if s["kind"] == "candidate"]
    ref = next((s for s in systems if s["id"] == REFERENCE), None)
    best = max(candidates, key=lambda s: s["val"]["accuracy"], default=None)

    kpis = []
    if ref:
        kpis.append({"k": "기준 (fewshot_all)", "v": f"{ref['val']['accuracy'] * 100:.1f}%", "s": "validation 정확도"})
    if best:
        delta = (best["val"]["accuracy"] - ref["val"]["accuracy"]) * 100 if ref else 0.0
        kpis.append({"k": "AI 제안 최고", "v": f"{best['val']['accuracy'] * 100:.1f}%", "s": f"{best['label']} · {delta:+.1f}%p"})
        if ref and ref["val"]["llm_input_tokens"] and best["val"]["llm_input_tokens"] is not None:
            change = (best["val"]["llm_input_tokens"] / ref["val"]["llm_input_tokens"] - 1) * 100
            kpis.append({"k": "입력 토큰", "v": f"{change:+.0f}%", "s": f"{best['val']['llm_input_tokens']:,} vs 기준 {ref['val']['llm_input_tokens']:,}"})
    kpis.append({"k": "AI가 쓴 후보", "v": str(len(candidates)), "s": "proposer 1회 반복"})

    return {
        "subtitle": "Meta-Harness 공식 오픈소스를 온프레미스 Qwen 인프라에 이식하여, 하네스 자기개선 루프 1회를 실측·검증했다.",
        "model_short": "Qwen3.8-Flash-Next",
        "meta": [
            ["task model", "Qwen3.8-Flash-Next-NVFP4 (on-prem, 동결)"],
            ["proposer", "Claude Code · sonnet"],
            ["dataset", "Symptom2Disease · train 60 / val 50"],
            ["run", run_name],
        ],
        "reference": REFERENCE,
        "best": best["id"] if best else None,
        "structure": structure,
        "systems": systems,
        "kpis": kpis,
        "metrics_lead": "같은 모델·같은 validation 50문항. 막대 옆 수치는 fewshot_all 대비 차이.",
        "note": "단일 시드(42), 1회 반복, 단일 데이터셋 결과다. 후보 선택은 validation만 사용했고, test(212문항)는 baseline 2개와 validation 최고 후보만 1회 평가했다.",
        "footer": "Code: stanford-iris-lab/meta-harness (MIT) · arXiv 2603.28052 · 수치는 실행 로그(val.json)에서 자동 생성",
    }


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "midterm"
    payload = build_payload(run_name)
    template = (ROOT / "viz/template.html").read_text()
    html = template.replace("/*DATA*/null", json.dumps(payload, ensure_ascii=False))
    out = ROOT / "viz/harness_viz.html"
    out.write_text(html)
    (ROOT / "viz/viz_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out} with {len(payload['systems'])} systems, best={payload['best']}")


if __name__ == "__main__":
    main()
