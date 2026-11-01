# Project101: Memory-Augmented Agent Evaluation

[한국어 문서](README.ko.md)

## Research presentation

The research presentation is deployed as an HTML slide deck on Vercel:
**[View the presentation in your browser](https://harness-midterm-talk.vercel.app)**.
The deployed presentation can be viewed without cloning or running this repository.

A research workspace for comparing memory strategies in language-model agents.
It combines symptom-to-disease text classification, a terminal application,
deterministic data-science task cards, and paired statistical analysis.

This is a research prototype. Disease labels are benchmark predictions, not
clinical diagnoses. The GitHub distribution is a privacy-reviewed source snapshot:
private datasets, trained memories, reports, credentials, inference caches, and
the original development history are intentionally excluded.

## Components

| Directory | Responsibility |
|---|---|
| `harness_lab/text_classification/` | Memory implementations, online learning/evaluation, benchmark orchestration, and contract tests |
| `harness_cli/` | Typer/Rich terminal interface with interactive and single-question modes |
| `ds_tasks/` | Data-science reference pipeline, deterministic task cards, ReAct loop, and answer scoring |
| `analysis/harness_stats.py` | Original single-seed replay, feature extraction, and statistical diagnosis |
| `analysis/multiseed/` | Cache-only multi-seed analysis, exact paired tests, bootstrap, clustered regressions, power planning, and privacy checks |

Memory strategies include `no_memory`, `fewshot_all`, `knn_top6`,
`contrast_online`, `reflexion_notes`, `cue_bank`, and `cue_bank_verified`.
The experiments distinguish retrieved examples from generated diagnostic cues
and notes. Generated memories have explicit accounting for accepted entries,
rejected candidates, and parse failures.

## Requirements and setup

- Python 3.11 or newer and `uv`.
- Git; GitHub CLI is only needed for repository administration.
- A separately provisioned model endpoint for live inference.
- Approved local datasets and frozen memories for data-dependent evaluation.

From the repository root, install only the component you need:

```sh
uv sync --project harness_cli --group dev
uv sync --project harness_lab/text_classification
uv sync --project ds_tasks --group dev
```

Dependency installation may access package registries. It does not launch model
inference. Do not run the experiment commands until your own endpoint, data,
access permissions, and spending limits are configured.

## Terminal application

```sh
uv run --project harness_cli harness-cli --help
```

Configure the CLI through environment variables in your own shell or an ignored
local environment file. Never commit real keys or endpoint details.

| Variable | Meaning |
|---|---|
| `HARNESS_SYSTEM` | Memory implementation name; the CLI defaults to `confusion_contrast` |
| `HARNESS_MEMORY` | Path to a separately provisioned frozen `memory.json` |
| `HARNESS_MODEL` | LiteLLM-compatible model identifier |
| `HARNESS_API_BASE` | Your own OpenAI-compatible endpoint URL |
| `OPENAI_API_KEY` | Provider credential if required by your endpoint |

The source export replaces internal endpoint defaults with a non-resolving
`.invalid` placeholder. For example, set `HARNESS_API_BASE` to your approved
endpoint and `HARNESS_MEMORY` to your local memory file before using:

```sh
# These commands perform model inference once configured.
uv run --project harness_cli harness-cli
uv run --project harness_cli harness-cli predict "Fever and chills" --json
uv run --project harness_cli harness-cli info
```

Interactive commands are `/help`, `/info`, `/why`, `/clear`, and `/exit`.
The CLI loads the existing classification harness; it is not a standalone bundle
of trained memories or datasets. A clean clone cannot predict until those inputs
are supplied. `--help` does not require a model call.

## Classification and data-science workflows

The classification inner loop evaluates a memory system on a named dataset;
the benchmark and meta-harness orchestrate candidate evaluation and selection.
Selection uses validation data; test evaluation should follow a fixed selection
and analysis plan. Internal `config.yaml` and raw JSONL datasets are excluded from
the source export. Recreate an approved local configuration before running sweeps.

Data provenance for the vendored classification tasks is documented in
[`data/README.md`](harness_lab/text_classification/data/README.md). The
Symptom2Disease files were obtained through the MCE artifact distribution.
No raw patient-text records are included in this export. This benchmark is not
evidence of clinical validity or a representative prospective patient study.

`ds_tasks/` includes task cards and a deterministic reference pipeline for churn
and Census-derived analysis. Raw source data and cloned reference repositories
are excluded. Set `CHURN_DATA_DIR` to an approved local data directory. Live
agent runs also require `DS_AGENT_BASE_URL`, `DS_AGENT_MODEL`, and the endpoint's
credential. The runner uses two-fold cross-fitting: memory is learned on one
fold, evaluated on the other, and the folds are then swapped.

## Cache-only statistical analysis

The multi-seed engine never calls a model. It loads existing run files and cached
responses through the replay path; a missing response is marked missing rather
than generated. It includes:

- Per-seed/split accuracies and exact paired McNemar tests.
- Seed-stratified matched-pair MH and pooled-discordance exact binomial tests.
- Holm adjustment for the six comparisons against `knn_top6`.
- Paired bootstrap within seed, 10,000 draws with random seed 0.
- Family and cue-feature logistic regressions with registered cluster units.
- Repeat-noise measurements and explicit input-coverage reporting.

The replay workflow requires the local frozen memories, result files, response
caches, original reference outputs, and preregistration. They are not bundled.
Internal endpoint strings used in historical cache hashes are redacted in the
export; exact historical cache replay requires restoring the original cache-key
configuration privately. Do not interpret a clean-clone replay failure as a new
experimental observation.

The independent power calculator can be run without experimental outcomes:

```sh
mkdir -p analysis/multiseed/out
uv run --project harness_lab/text_classification --with scipy \
  python analysis/multiseed/power_plan.py
```

It writes local planning outputs under `analysis/multiseed/out/`. Sample sizes
refer to independent paired questions, not repeated evaluations of the same text.

## Verified historical snapshot

The following is the saved local analysis snapshot from 2026-11-02 02:32 KST,
not a claim that a fresh clone reproduces the excluded private artifacts.

| Comparison | Correct predictions | Exact McNemar p | Holm-adjusted p | Interpretation |
|---|---|---:|---:|---|
| Seed42 test: knn_top6 vs no_memory | 175/212 vs 142/212 | 0.000005550521 | 0.000033303125 | Significant secondary comparison; +15.57 percentage points |
| Seed42 test: knn_top6 vs cue_bank | 175/212 vs 172/212 | 0.6776394844 | 1.0 | No significant difference found |
| Seed123 test: knn_top6 vs cue_bank | 173/212 vs 179/212 | 0.2101135254 | 0.8404541016 | No significant difference found; opposite direction |
| Seed42 validation: knn_top6 vs cue_bank | 41/50 vs 35/50 | 0.0703125 | 0.421875 | No significant difference found, n=50 |

The primary three-seed hypothesis remained **INCOMPLETE**: 29 of 37 required new
result files were analysable. Eight seed456 result files were missing. The
significant no-memory comparison does not confirm the primary generated-cue
hypothesis or prove negative priming. All seeds reuse the same test texts;
636 evaluations are not 636 independent questions. Non-significance does not
establish equivalence.

Saved verification: 26 analysis/power tests passed; 21 reference comparisons and
350 question rows matched exactly, including a byte-identical reference McNemar
CSV. The preregistration and 14 original baseline result/memory hashes were
unchanged. Detailed local evidence remains in the excluded
`analysis/multiseed/out/validation.json` and related CSV files.

Only operational limits changed (20 to 60 DAG transitions; 12 to 21 file checks).
No scientific settings were tuned after observing significance. A prospective
scenario with a 5 percentage-point effect and 20% discordance requires 658
independent paired questions for 80% power at two-sided alpha 0.05. This is an
assumption-based plan, not a significance guarantee.

## Testing

Statistical and privacy tests that exclude the private reference-replay fixture:

```sh
uv run --project harness_lab/text_classification \
  --with scipy --with statsmodels --with pandas --with pytest \
  python -m pytest analysis/multiseed/test_stats.py \
  analysis/multiseed/test_power_plan.py analysis/multiseed/test_privacy_guard.py \
  -k 'not reference_replay_matches_original_features_and_all_pairs' \
  -q -p no:cacheprovider --basetemp analysis/multiseed/.pytest-tmp
```

The omitted replay/parity test requires private fixtures and cannot pass in an
export without them. CLI tests use stubs; tests requiring a frozen memory are skipped if
it is absent. Classification data tests and data-science reference tests also
require the omitted datasets. Check prerequisites before selecting a full suite.

## Privacy and publication

The repository ignores reports, presentations, credentials, raw datasets,
memories, logs, caches, and agent session state. The original development history
is not part of the GitHub snapshot. Internal endpoint defaults in exported source
are replaced with non-resolving placeholders. No personal commit identity is
used for the source-export commit.

Offline guards are provided in `analysis/multiseed/hooks/`. To enable them in a
new clone:

```sh
git config --local core.hooksPath analysis/multiseed/hooks
```

See [publication privacy](analysis/multiseed/docs/publication_privacy.md).
Hooks can be bypassed and pattern scans cannot identify every type of private
information. Keep manual review and GitHub private visibility as separate controls.

## Attribution and licensing

The adapted Meta-Harness material retains its upstream MIT notice in
[`harness_lab/LICENSE-meta-harness`](harness_lab/LICENSE-meta-harness).
Dataset and other upstream materials have their own terms. No new blanket
license or redistribution permission for excluded datasets is asserted here.
