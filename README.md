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

`harness_cli` v0.2.2 is a terminal front-end for the evaluated memory systems,
with a second task for question answering over your own English notes.
The model stays fixed; only the memory injected into the prompt (the harness)
changes. Eight memory "blocks" share one interface: the seven online-v2 systems
(`no_memory`, `fewshot_all`, `knn_top6`, `contrast_online`, `cue_bank`,
`cue_bank_verified`, `reflexion_notes`) and the midterm candidate
`confusion_contrast`. The background and the full results are in the
[research presentation](https://harness-midterm-talk.vercel.app).

```sh
uv run --project harness_cli harness-cli --help
uv run --project harness_cli harness-cli --demo          # replayed answers, no model server
uv run --project harness_cli harness-cli --demo compare no_memory knn_top6 1
```

Configure the CLI through environment variables in your own shell or an ignored
local environment file. Never commit real keys or endpoint details.

| Variable | Meaning |
|---|---|
| `HARNESS_SYSTEM` | Starting block name or alias; the CLI defaults to `confusion_contrast` |
| `HARNESS_MEMORY` | Overrides the bundled memory of the starting block only |
| `HARNESS_DATA_ROOT` | Folder that holds `harness_lab/` (agent code); defaults to the checkout |
| `HARNESS_MODEL` | LiteLLM-compatible model identifier; without it, a non-default endpoint uses the first model its `/models` lists |
| `HARNESS_API_BASE` | Your own OpenAI-compatible endpoint URL (tried first) |
| `OPENAI_BASE_URL` | Second endpoint to try, e.g. vLLM, OpenRouter, or Ollama at `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | Provider credential if required by your endpoint (sent to `/models` and chat calls, never stored) |
| `HARNESS_HOME` | Folder for the persistent profile and scratchpad; defaults to `~/.harness` |

At startup the CLI probes `HARNESS_API_BASE`, `OPENAI_BASE_URL`, the built-in default and a local
Ollama in that order and uses the first that answers `/models`. When the live model is not the one
the results were measured on, the CLI says so: the logged accuracies describe that model only.

The source export replaces internal endpoint defaults with a non-resolving
`.invalid` placeholder. On a new machine, point the CLI at an endpoint once; the
setting is saved in `~/.harness/config.json` (never the key):

```sh
uv run --project harness_cli harness-cli config --api-base http://localhost:11434/v1 --model llama3.2:3b
OPENAI_API_KEY=... uv run --project harness_cli harness-cli config   # api.openai.com, gpt-4o-mini
```

Then:

```sh
# These commands perform model inference once configured.
uv run --project harness_cli harness-cli
uv run --project harness_cli harness-cli predict "Fever and chills" --json
uv run --project harness_cli harness-cli info
```

Interactive commands:

| Command | Effect |
|---|---|
| `/harness <name>` (alias `/switch`) | Swap the active block; without a name, list all eight |
| `/compare <block1> <block2> [1-3 or text]` | Run two blocks side by side: diagnosis, prompt size and tokens, latency, top retrieved examples, and the logged test accuracy of each |
| `/record "<symptoms>" -> "<disease>"` | Add an example to the session memory (English, one of the 22 labels) |
| `/memory status`, `/memory reset` | Frozen versus session examples per block; drop the session examples |
| `/study` | Four-step guide: no memory, example selection, injected cues, free experiment |
| `/task medical`, `/task study` | Switch between the 22-disease benchmark and question answering over your notes |
| `/load <notes.md\|notes.txt>` | Chunk English notes by heading and paragraph, index them, and switch to the study task |
| `/harness no_memory\|knn_retrieval\|lesson_cues` | In the study task: question only; top-4 note chunks by TF-IDF; up to 5 taught rules plus 2 chunks |
| `/record "<concept>" -> "<rule>"` | In the study task: teach a rule; it is retrievable from the next question on |
| `/inspect context` | Raw prompt of the last query, each injected example with its TF-IDF score and why it was chosen (top-k, past mistake, label coverage) |
| `/explain fewshot` | Why `fewshot_all` does not scale, with the logged accuracy, per-query tokens and the computed accuracy-token Pareto frontier |
| `/profile`, `/profile set <field> <value>` | Learner profile (`target_exam`, `topics`, `style`, `name`), added as one line to grounded study prompts |
| `/quiz [topic]` | Ask for three review questions with answers from the loaded notes |
| `/help`, `/info`, `/why`, `/clear`, `/exit` | As in v0.1 |

Memory has three tiers. The frozen `memory.json` files are only read; the CLI refuses any
`HARNESS_HOME` inside the frozen logs. `/record` additions and taught study rules are kept in
`~/.harness/scratchpad.json` and reloaded in the next session, until `/memory reset`;
`~/.harness/profile.json` holds the learner profile. Loaded notes live only in the session.
`--demo` neither reads nor writes `~/.harness`, so replayed presets always use the frozen memory. A recorded example enters the retrieval of every block
that uses examples; in `cue_bank`, `cue_bank_verified` and `reflexion_notes` it
changes only the similar-examples section, because writing new cues or notes
needs a model call.

`--demo` (or an unreachable server) answers from `replay.json`, which holds real
server responses for presets 1-3 on all eight blocks. Any other input, and any input after
`/record` changed the memory, gets an offline retrieval preview: the exact prompt the active
block would send, the retrieved examples and a token estimate, with no diagnosis
(`predict` exits with code 2). Study questions without a server get the same preview.
In the recorded presets both `no_memory` and `knn_top6` answer correctly, so a
single case does not show the difference between systems; the CLI shows the
logged results next to each comparison. The eight frozen memories ship inside the
package (`harness_cli/src/harness_cli/data/memories/`, byte-identical to the research
logs, which a test checks when the logs are present), so a fresh clone runs without
`logs/` or `HARNESS_DATA_ROOT`.

The figures the CLI displays come from `analysis/multiseed/out/accuracy_tokens.csv`
(test split, n=212 per seed, seeds 42/123/456):

| Block | Test accuracy (n=212) | Note |
|---|---|---|
| `no_memory` | 67.0% | 3-seed mean; baseline |
| `knn_top6` | 83.0% | 3-seed mean; 80.8% fewer input tokens than `fewshot_all` |
| `fewshot_all` | 84.6% | 3-seed mean; not distinguishable from `knn_top6` (pooled Mantel-Haenszel p = 0.18, before Holm) |
| `confusion_contrast` | 80.7% | Seed 42 single run (validation 82.0%, n=50) |

The seed 42, 50-item validation gap between `knn_top6` and `cue_bank` (41/50 vs 35/50,
p = 0.070) was not replicated on the 212-item test split (175/212 vs 172/212,
p = 0.68); see the verified snapshot below.

### Standalone bundle

```sh
cd harness_cli && uv build && cd ..
uv run --project harness_cli python scripts/build_standalone.py
# -> dist/harness-cli-v0.2.2-standalone.zip
```

The zip holds the wheel (with the eight frozen memories inside), the runtime code of
`harness_lab/text_classification`, and `run.sh`, which starts
`harness-cli --demo` through `uv run --with <wheel>`. It needs `uv` and, on first
run, PyPI access for the dependencies; it does not need the model server. The
dataset splits are not included: the prompt template comes from a copy inside
the wheel that the tests pin to the dataset prompt. The bundled memories contain
Symptom2Disease training examples written by the evaluated runs.

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
