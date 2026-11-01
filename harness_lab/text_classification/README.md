# Text classification harness

This is a privacy-reviewed source export of the adapted Meta-Harness classification
harness. See the root [English README](../../README.md) or [Korean README](../../README.ko.md).

Memory implementations live in `agents/`; `inner_loop.py` evaluates one system,
`benchmark.py` orchestrates sweeps, and `meta_harness.py` manages candidate selection.

Raw JSONL data, internal `config.yaml`, trained memories, logs, and response caches
are excluded. Data-dependent commands require approved local replacements. Dataset
provenance is documented in [data/README.md](data/README.md). Provider-backed commands
can incur costs and must be configured explicitly. No model server is contacted by
the source-export process. The upstream MIT notice is retained one directory above.
