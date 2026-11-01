"""Every test gets its own HARNESS_HOME, so nothing ever writes to the real ~/.harness."""

import pytest


@pytest.fixture(autouse=True)
def isolated_harness_home(tmp_path, monkeypatch):
    home = tmp_path / "harness-home"
    monkeypatch.setenv("HARNESS_HOME", str(home))
    return home
