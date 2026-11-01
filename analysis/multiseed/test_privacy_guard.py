"""Synthetic-only checks: no credentials or network access."""

from privacy_guard import SECRET_PATH, content_findings


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
