"""Chat client for the on-prem Qwen (SGLang, OpenAI-compatible) with a disk cache.

Every response is cached by (model, messages, seed, sampling), so a finished run can be
replayed for analysis without calling the server.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("DS_AGENT_BASE_URL", "https://model-endpoint.example.invalid/v1")
DEFAULT_MODEL = os.environ.get("DS_AGENT_MODEL", "Inferact/Qwen3.8-Flash-Next-NVFP4")
CACHE_DIR = Path(os.environ.get("DS_AGENT_CACHE", Path.home() / ".cache/ds_agent"))


class ChatLLM:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.6, max_tokens: int = 1500, cache_dir: Path = CACHE_DIR):
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "onprem-placeholder"))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, messages: list[dict], seed: int) -> tuple[str, dict]:
        key = hashlib.sha256(json.dumps(
            [self.model, messages, seed, self.temperature, self.max_tokens], ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            return cached["text"], {**cached["usage"], "cached": True}
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, seed=seed,
            max_tokens=self.max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""
        usage = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
        path.write_text(json.dumps({"text": text, "usage": usage}, ensure_ascii=False))
        return text, {**usage, "cached": False}
