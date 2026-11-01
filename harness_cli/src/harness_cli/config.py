"""Runtime settings, resolved from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# A standalone bundle points this at its own copy of harness_lab (agent code, dataset splits, frozen memories).
DATA_ROOT = Path(os.environ.get("HARNESS_DATA_ROOT", str(PROJECT_ROOT))).expanduser().resolve()
HARNESS_LAB = DATA_ROOT / "harness_lab"
LOGS = HARNESS_LAB / "text_classification/logs"  # frozen research data: read only, absent from a fresh clone
MODEL_DIR = "qwen3.8-flash-next-nvfp4"
# Byte-identical copies of the eight frozen memories (tests/test_bundle.py pins them to the logs when present),
# shipped inside the package so a clone or a wheel runs without the gitignored logs/.
MEMORY_DIR = Path(__file__).resolve().parent / "data/memories"


def bundled_memory(system: str) -> Path:
    return MEMORY_DIR / f"{system}.json"


DEFAULT_MEMORY = bundled_memory("confusion_contrast")
NO_MEMORY_PATH = bundled_memory("no_memory")

ONPREM_API_BASE = "https://model-endpoint.example.invalid/v1"  # the SGLang server every scored run used
ONPREM_MODEL = "openai/Inferact/Qwen3.8-Flash-Next-NVFP4"
OLLAMA_API_BASE = "http://localhost:11434/v1"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "openai/gpt-4o-mini"
PLACEHOLDER_KEY = "onprem-placeholder"
NO_SERVER_MESSAGE = ("[서버 연결 불가] LLM 엔드포인트에 접속할 수 없습니다. "
                     "OPENAI_API_KEY 또는 OPENAI_BASE_URL(예: Ollama localhost:11434/v1)을 설정해 주세요.")
# Hosted APIs reject SGLang's chat_template_kwargs; self-hosted servers (SGLang, vLLM, Ollama) accept or ignore it.
HOSTED_HOSTS = ("api.openai.com", "openrouter.ai")


def load_env_file() -> None:
    """Load variables from .env if present (cwd, project root, or ~/.harness/.env) without overwriting existing env."""
    home_dir = Path(os.environ.get("HARNESS_HOME", "~/.harness")).expanduser()
    candidates = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        home_dir / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass


load_env_file()


def harness_home() -> Path:
    """Where the persistent profile and scratchpad live (~/.harness unless HARNESS_HOME says otherwise)."""
    return Path(os.environ.get("HARNESS_HOME", "~/.harness")).expanduser().resolve()


def user_config_path() -> Path:
    return harness_home() / "config.json"


def user_config() -> dict[str, str]:
    """{api_base, model} saved by `harness-cli config`; never holds a key."""
    try:
        data = json.loads(user_config_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: str(data[k]) for k in ("api_base", "model") if isinstance(data, dict) and data.get(k)}


def has_real_key() -> bool:
    load_env_file()
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key) and key != PLACEHOLDER_KEY


def api_base_candidates() -> tuple[str, ...]:
    """Endpoints to try in order: HARNESS_API_BASE, OPENAI_BASE_URL, ~/.harness/config.json, the on-prem server,
    a local Ollama, then api.openai.com when a real OPENAI_API_KEY is set."""
    load_env_file()
    found = [os.environ.get("HARNESS_API_BASE"), os.environ.get("OPENAI_BASE_URL"), user_config().get("api_base"),
             ONPREM_API_BASE, OLLAMA_API_BASE, OPENAI_API_BASE if has_real_key() else None]
    return tuple(dict.fromkeys(b.rstrip("/") for b in found if b))


def is_hosted(api_base: str) -> bool:
    host = urlparse(api_base).hostname or ""
    return any(host == h or host.endswith("." + h) for h in HOSTED_HOSTS)


@dataclass(frozen=True)
class Settings:
    system: str
    memory_path: Path
    model: str
    api_base: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            system=os.environ.get("HARNESS_SYSTEM", "confusion_contrast"),
            memory_path=Path(os.environ.get("HARNESS_MEMORY", str(DEFAULT_MEMORY))).expanduser(),
            model=os.environ.get("HARNESS_MODEL", ONPREM_MODEL),
            api_base=api_base_candidates()[0],
        )

    @property
    def scored_model(self) -> bool:
        """True when this is the model the badge numbers were measured on."""
        return self.model == ONPREM_MODEL


def prepare_runtime_env(offline: bool = False, api_base: str = ONPREM_API_BASE) -> None:
    """Match the evaluated run: reasoning off, and a placeholder key (litellm requires one; SGLang ignores it).

    A real OPENAI_API_KEY is kept (setdefault). Reasoning-off is only requested from self-hosted servers.
    With offline=True nothing may touch the network: the dataset comes from the local HF cache and
    litellm uses its bundled cost map instead of fetching one.
    """
    if not is_hosted(api_base):
        os.environ.setdefault("TC_DISABLE_THINKING", "1")
    os.environ.setdefault("OPENAI_API_KEY", PLACEHOLDER_KEY)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
