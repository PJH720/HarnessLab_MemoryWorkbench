# Offline multi-seed statistical analysis

See the root [English README](../../README.md) and [Korean README](../../README.ko.md)
for setup, source-export restrictions, verified historical findings, and test commands.

`stats.py` provides cache-only replay, paired tests, bootstrap, regression, coverage,
and a bounded DAG watcher. Cache misses never invoke an LLM. Full replay needs the
excluded local datasets, memories, result files, reference outputs, preregistration,
and original cache-key endpoint configuration. The exported endpoint string is
a non-resolving placeholder. Statistical unit tests use synthetic inputs.

`power_plan.py` calculates prospective exact-McNemar power without reading observed
experiment outcomes. Create `out/` before invoking the script in a new clone.

`privacy_guard.py` and `hooks/` provide offline index/history publication checks.
Generated outputs and audit files remain ignored. Hooks are opt-in per checkout:

```sh
git config --local core.hooksPath analysis/multiseed/hooks
```

The original workspace and its development history are not bundled.
