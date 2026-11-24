"""Settings, all from environment variables so the CLI and the MCP server agree.

Every endpoint is OpenAI-compatible, so any local server works (llama.cpp,
vLLM, Ollama, LM Studio) as long as it serves /v1/embeddings and
/v1/chat/completions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(_env("DOCSRAG_DB", "docsrag.sqlite")).expanduser())
    embed_url: str = field(default_factory=lambda: _env("DOCSRAG_EMBED_URL", "http://localhost:8081/v1"))
    embed_model: str = field(default_factory=lambda: _env("DOCSRAG_EMBED_MODEL", "bge-m3"))
    embed_api_key: str = field(default_factory=lambda: _env("DOCSRAG_EMBED_API_KEY"))
    llm_url: str = field(default_factory=lambda: _env("DOCSRAG_LLM_URL", "http://localhost:8080/v1"))
    llm_model: str = field(default_factory=lambda: _env("DOCSRAG_LLM_MODEL", "local"))
    llm_api_key: str = field(default_factory=lambda: _env("DOCSRAG_LLM_API_KEY"))
    # "off" sends chat_template_kwargs.enable_thinking=false, which llama.cpp and
    # vLLM pass to Qwen-style chat templates to skip the reasoning phase. Leave
    # it at "default" for servers that don't understand it.
    llm_thinking: str = field(default_factory=lambda: _env("DOCSRAG_LLM_THINKING", "default").lower())
    llm_max_tokens: int = field(default_factory=lambda: int(_env("DOCSRAG_LLM_MAX_TOKENS", "800")))
    # Shown to the agent in the MCP server's instructions, so it knows which
    # connector to use for what ("my homelab runbooks", "the Hermes docs").
    corpus_description: str = field(default_factory=lambda: _env("DOCSRAG_DESCRIPTION", "a folder of markdown docs"))
    server_name: str = field(default_factory=lambda: _env("DOCSRAG_NAME", "docsrag"))
    request_timeout: float = field(default_factory=lambda: float(_env("DOCSRAG_TIMEOUT", "180")))


def load() -> Settings:
    return Settings()
