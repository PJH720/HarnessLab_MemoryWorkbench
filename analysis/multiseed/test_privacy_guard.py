"""Synthetic-only checks: no credentials or network access."""

from privacy_guard import (
    REVIEWED_BINARIES, SECRET_PATH, ZERO_OID, content_findings, identity_findings,
    pushed_tips, reviewed_binaries,
)


def test_sensitive_content_is_detected():
    samples = {
        "api-key": b"sk" + b"-" + b"a" * 32,
        "aws-access-key": b"AKIA" + b"A" * 16,
        "private-key": b"-----BEGIN " + b"PRIVATE KEY-----",
        "personal-home-path": b"/Users/" + b"example/project/",
        "private-ip": b"192.168." + b"1.10",
        "internal-host": b"node.tail" + b"example.ts.net",
        "student-identifier": b"2018" + b"0123",
        "personal-name": b"Jaehyun" + b"Park",
        "email-address": b"person" + b"@" + b"example.invalid",
        "literal-credential": b"API_KEY" + b" = '" + b"x" * 24 + b"'",
    }
    for reason, content in samples.items():
        assert reason in content_findings(content)


def test_safe_source_and_unknown_binary():
    assert content_findings(b"API_KEY = os.environ.get('API_KEY')") == []
    assert content_findings(b"binary\x00payload") == ["unreviewed-binary"]
    assert content_findings(b"x" * 5_000_001) == ["unreviewed-large-file"]


def test_secret_paths_are_blocked_before_reading():
    for path in [".credentials.json", "app/.env.prod", ".aws/config", "private/secret.txt"]:
        assert SECRET_PATH.search(path)


def test_env_templates_are_scanned_not_skipped():
    for path in [".env.example", "harness_cli/.env.sample", "app/.env.template"]:
        assert not SECRET_PATH.search(path)
    for path in [".env", ".env.local", "app/.env.example.bak"]:
        assert SECRET_PATH.search(path)


def test_noreply_identity_passes_but_personal_identity_blocks():
    at = b"@"
    bot = (b"dependabot[bot] <49699333+dependabot[bot]" + at + b"users.noreply.github.com>\n"
           b"GitHub <noreply" + at + b"github.com>")
    assert identity_findings(bot) == []
    personal = b"Some" + b"one <person" + b"@" + b"example.invalid>"
    assert "email-address" in identity_findings(personal)
    named = b"Jaehyun" + b" Park <1+someone" + at + b"users.noreply.github.com>"
    assert "personal-name" in identity_findings(named)


def test_reviewed_binaries_match_exact_blob_id(tmp_path):
    manifest = tmp_path / REVIEWED_BINARIES
    manifest.parent.mkdir(parents=True)
    manifest.write_text("# reviewed\n" + "a" * 40 + " analysis/figures/fig.png\n")
    assert reviewed_binaries(tmp_path) == {("analysis/figures/fig.png", "a" * 40)}
    assert reviewed_binaries(tmp_path / "missing") == set()


def test_pushed_tips_skip_deleted_refs():
    tip = "b" * 40
    lines = [f"refs/heads/main {tip} refs/heads/main {'c' * 40}",
             f"(delete) {ZERO_OID} refs/heads/old {'d' * 40}", ""]
    assert pushed_tips(lines) == [tip]


def test_delete_only_push_publishes_no_commits():
    assert pushed_tips([f"(delete) {ZERO_OID} refs/heads/old {'d' * 40}"]) == []
