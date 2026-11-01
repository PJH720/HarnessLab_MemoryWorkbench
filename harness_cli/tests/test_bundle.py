"""Portability: the frozen memories ship inside the package, and endpoints come from env or ~/.harness/config.json."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from harness_cli import cli, config, lego
from harness_cli.config import Settings
from harness_cli.engine import UNRECORDED_NOTE, HarnessEngine, SafeEngine, ReplayCache, discover
from harness_cli.home import HarnessHome, HomeError


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("block", list(lego.BLOCKS.values()), ids=lambda b: b.name)
def test_every_block_has_a_bundled_memory_inside_the_package(block):
    path = block.memory_path
    assert path.is_file() and path.parent == config.MEMORY_DIR
    assert path.is_relative_to(config.MEMORY_DIR.parent.parent), "resolved from the package, not the checkout"
    json.loads(path.read_text())


@pytest.mark.parametrize("block", list(lego.BLOCKS.values()), ids=lambda b: b.name)
def test_bundled_memory_is_byte_identical_to_the_research_log(block):
    if not block.log_path.exists():
        pytest.skip("frozen logs not present (fresh clone)")
    assert _sha(block.memory_path) == _sha(block.log_path)


def test_default_settings_need_no_logs_or_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("HARNESS_MEMORY", raising=False)
    monkeypatch.setattr(config, "LOGS", tmp_path / "missing-logs")
    settings = Settings.from_env()
    assert settings.memory_path == config.DEFAULT_MEMORY and settings.memory_path.is_file()
    engine = HarnessEngine(settings, offline=True)
    assert engine.stats().examples > 0


def test_no_server_preview_says_how_to_connect():
    engine = HarnessEngine(Settings.from_env(), offline=True)
    pred, note = SafeEngine(engine, ReplayCache(), live=False).predict("my knees ache and swell every morning")
    assert pred.source == "preview" and pred.evidence and pred.prompt
    assert config.NO_SERVER_MESSAGE in note and "OPENAI_BASE_URL" in note
    assert "녹화된 응답이 없습니다" not in UNRECORDED_NOTE


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("HARNESS_API_BASE", "OPENAI_BASE_URL", "HARNESS_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_saved_config_is_tried_after_env_and_before_defaults(clean_env):
    HarnessHome().save_config("http://localhost:8080/v1/", "my-model")
    assert config.user_config() == {"api_base": "http://localhost:8080/v1", "model": "my-model"}
    clean_env.setenv("OPENAI_BASE_URL", "http://env-box:9000/v1")
    assert config.api_base_candidates()[:3] == (
        "http://env-box:9000/v1", "http://localhost:8080/v1", config.ONPREM_API_BASE)


def test_saved_model_applies_only_to_its_own_endpoint(clean_env):
    HarnessHome().save_config("http://localhost:8080/v1", "my-model")
    up = {"http://localhost:8080/v1": ["other"], config.OLLAMA_API_BASE: ["llama3.2:3b"]}
    found = discover(Settings.from_env(), lister=lambda b, t: up.get(b))
    assert (found.api_base, found.model) == ("http://localhost:8080/v1", "openai/my-model")
    only_ollama = discover(Settings.from_env(), lister=lambda b, t: up.get(b) if b == config.OLLAMA_API_BASE else None)
    assert only_ollama.model == "openai/llama3.2:3b"


def test_real_openai_key_adds_openai_with_gpt_4o_mini(clean_env):
    assert config.OPENAI_API_BASE not in config.api_base_candidates()
    clean_env.setenv("OPENAI_API_KEY", "test-key-not-real")
    assert config.api_base_candidates()[-1] == config.OPENAI_API_BASE
    found = discover(Settings.from_env(), lister=lambda b, t: ["gpt-4-0613"] if b == config.OPENAI_API_BASE else None)
    assert found.model == "openai/gpt-4o-mini"


def test_config_never_stores_a_key_and_validates_input(isolated_harness_home):
    with pytest.raises(HomeError):
        HarnessHome().save_config("localhost:11434", "")
    HarnessHome().save_config("http://localhost:11434/v1", "qwen2.5:7b")
    text = (isolated_harness_home / "config.json").read_text()
    assert "key" not in text.lower()
    HarnessHome().clear_config()
    assert config.user_config() == {}


def test_config_command_saves_and_lists_candidates(clean_env, isolated_harness_home):
    result = CliRunner().invoke(cli.app, ["config", "--api-base", "http://localhost:11434/v1", "--model", "llama3.2:3b",
                                          "--no-check"])
    assert result.exit_code == 0, result.output
    assert json.loads((isolated_harness_home / "config.json").read_text())["model"] == "llama3.2:3b"
    assert "http://localhost:11434/v1" in result.output and "저장하지 않고" in result.output


def test_settings_are_resolved_without_any_absolute_checkout_path():
    assert "/Users/" not in str(config.MEMORY_DIR.relative_to(config.MEMORY_DIR.parents[2]))
    assert config.MEMORY_DIR == config.Path(config.__file__).resolve().parent / "data/memories"
