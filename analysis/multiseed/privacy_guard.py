"""Offline Git publication guard. Never prints matching content or reads secrets."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "api-key": re.compile(rb"\b(?:sk|ghp|gho|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{20,}"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private-key": re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
    "personal-home-path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private-ip": re.compile(rb"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    "internal-host": re.compile(rb"\b(?:[\w-]+\.tail[\w-]+\.ts\.net|spark-[a-z0-9]+)\b", re.I),
    "student-identifier": re.compile(rb"\b2018\d{4}\b"),
    "personal-name": re.compile(rb"Jaehyun[ _-]*Park", re.I),
    "email-address": re.compile(rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "literal-credential": re.compile(
        rb"\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\b"
        rb"\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]", re.I,
    ),
}
# Templates such as .env.example are not opened as secrets; their content is still scanned.
SECRET_PATH = re.compile(
    r"(?:^|/)(?:\.env(?:\.(?!(?:example|sample|template)$).*)?|\.credentials.*|\.netrc|auth\.(?:json|toml))$"
    r"|(?:credential|secret)|(?:^|/)(?:\.aws|\.ssh|session-env|shell-snapshots)/",
    re.I,
)
# GitHub noreply addresses are designed to be public; any other address in an identity blocks.
NOREPLY_IDENTITY = re.compile(rb"[\w.+\[\]-]+@users\.noreply\.github\.com|\bnoreply@github\.com")
# Binaries cannot be pattern-scanned, so each one passes only by its exact reviewed blob id.
REVIEWED_BINARIES = "analysis/multiseed/reviewed_binaries.txt"
ZERO_OID = "0" * 40
MAX_BLOB_BYTES = 5_000_000


def git(*args, data=None):
    result = subprocess.run(
        ["git", *args], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode:
        raise RuntimeError("Git inspection failed; publication blocked.")
    return result.stdout


def content_findings(content):
    if len(content) > MAX_BLOB_BYTES:
        return ["unreviewed-large-file"]
    if b"\x00" in content:
        return ["unreviewed-binary"]
    return [label for label, pattern in PATTERNS.items() if pattern.search(content)]


def identity_findings(identity):
    return content_findings(NOREPLY_IDENTITY.sub(b"", identity))


def reviewed_binaries(root):
    """Lines of '<blob id> <path>'; a changed byte changes the id and blocks again."""
    path = root / REVIEWED_BINARIES
    if not path.exists():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            oid, name = line.split(maxsplit=1)
            entries.add((name, oid))
    return entries


def pushed_tips(lines):
    """Local commit ids from pre-push stdin; deleted refs publish nothing."""
    tips = []
    for line in lines:
        fields = line.split()
        if len(fields) == 4 and fields[1] != ZERO_OID:
            tips.append(fields[1])
    return tips


def inspect(mode, tips=None):
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    allowed_binaries = reviewed_binaries(root)
    refs = ["INDEX"]
    if mode == "push":
        # Everything reachable from a pushed ref becomes public, not only the new commits.
        # Without pushed refs (a manual run), check every local commit.
        scope = tips if tips else ["--all"]
        refs += git("rev-list", *scope).decode().splitlines()
    findings = []
    seen = set()
    checked = 0
    for ref in refs:
        if ref != "INDEX":
            # Commit identities are published too; do not record their values.
            identity = git("show", "-s", "--format=%an <%ae>%n%cn <%ce>", ref)
            if identity_findings(identity):
                findings.append({"path": "<commit metadata>",
                                 "reason": "personal-commit-identity", "revision": ref})
        if ref == "INDEX":
            records = git("ls-files", "--stage", "-z").split(b"\x00")
        else:
            records = git("ls-tree", "-rz", ref).split(b"\x00")
        entries = []
        for record in filter(None, records):
            meta, raw_path = record.split(b"\t", 1)
            fields = meta.split()
            filemode, oid = fields[0], fields[1 if ref == "INDEX" else 2]
            path = raw_path.decode("utf-8", "surrogateescape")
            entries.append((path, oid.decode(), filemode.decode()))
        names = b"\x00".join(p.encode("utf-8", "surrogateescape") for p, _, _ in entries)
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-z", "--stdin"],
            input=names + (b"\x00" if names else b""),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("Ignore-rule inspection failed; publication blocked.")
        ignored = set(result.stdout.decode("utf-8", "surrogateescape").split("\x00"))
        for path, oid, filemode in entries:
            identity = (path, oid)
            if identity in seen:
                continue
            seen.add(identity)
            reasons = []
            if SECRET_PATH.search(path) or path in ignored:
                # Do not open credential files or private artifacts, even for auditing.
                reasons = ["protected-path"]
            elif filemode not in ("100644", "100755"):
                reasons = ["unreviewed-symlink-or-submodule"]
            elif int(git("cat-file", "-s", oid)) > MAX_BLOB_BYTES:
                reasons = ["unreviewed-large-file"]
            else:
                checked += 1
                reasons = content_findings(git("cat-file", "blob", oid))
                if reasons == ["unreviewed-binary"] and (path, oid) in allowed_binaries:
                    reasons = []
            for reason in reasons:
                findings.append({"path": path, "reason": reason, "revision": ref})
    return {"mode": mode, "status": "BLOCKED" if findings else "PASS",
            "revisions_checked": len(refs), "blobs_scanned": checked,
            "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("commit", "push"), default="push")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        tips = None
        if args.mode == "push" and not sys.stdin.isatty():
            tips = pushed_tips(sys.stdin.read().splitlines())
        report = inspect(args.mode, tips)
        if args.output:
            allowed = Path("analysis/multiseed/out").resolve()
            target = args.output.resolve()
            if not target.is_relative_to(allowed):
                raise RuntimeError("Audit output must stay in the ignored local output folder.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Privacy guard: {report['status']}; {len(report['findings'])} findings; "
              f"{report['revisions_checked']} index/revisions checked.")
        # Values, excerpts, and personal filenames are never printed by hooks.
        return 1 if report["findings"] else 0
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Privacy guard: BLOCKED. {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
