# Publication privacy

This source snapshot was exported without the original development history.
Private reports, presentations, datasets, credentials, memories, logs, caches,
and agent state are excluded by the root ignore policy. Internal endpoint defaults
in exported source use non-resolving `.invalid` placeholders.

The original workspace remains local and retains its history and files. Its
privacy blockers were not bypassed to publish this export. No original remote
was configured or pushed. Copyright notices for upstream code are preserved.

To enable the provided offline hooks in this clone:

```sh
git config --local core.hooksPath analysis/multiseed/hooks
```

The commit hook checks the full index; the push hook also checks all locally
reachable commits and their identities. Protected credential paths are rejected
before opening their contents. Other blobs are checked for recognized credentials,
private addresses, personal identifiers, and unexpected binaries. Matching values
are never printed. Missing scanner/runtime dependencies block the operation.

Run the scanner from the repository root with:

```sh
uv run --project harness_lab/text_classification python analysis/multiseed/privacy_guard.py --mode push
```

Keep the GitHub repository private. Hooks are local, can be bypassed, and are not
a comprehensive personal-information detector. Review every publication manually.
