"""Minimal OpenAI-compatible clients (stdlib only) for embeddings and chat.

``DOCSRAG_EMBED_URL=hash://`` swaps in a deterministic hashing embedder. It
exists so the tests and CI run without a model server; its vectors are a bag of
words, not semantics, so never use it for real search.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings


class ClientError(RuntimeError):
    pass


def post_json(url: str, payload: dict[str, Any], *, api_key: str = "", timeout: float = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ClientError(f"{url} answered HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"{url} unreachable: {exc.reason}") from exc


# ----------------------------------------------------------------------------- embeddings

_WORD = re.compile(r"[a-z0-9]+")
HASH_DIM = 256


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * HASH_DIM
    for word in _WORD.findall(text.lower()):
        digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
        vec[int.from_bytes(digest, "little") % HASH_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    """Batches texts to /v1/embeddings. Counts tokens when the server reports them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fake = settings.embed_url.startswith("hash://")
        self.tokens = 0

    @property
    def model(self) -> str:
        return "hash-bow-256" if self.fake else self.settings.embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.fake:
            return [_hash_embed(t) for t in texts]
        url = self.settings.embed_url.rstrip("/") + "/embeddings"
        reply = post_json(url, {"model": self.settings.embed_model, "input": texts},
                          api_key=self.settings.embed_api_key, timeout=self.settings.request_timeout)
        rows = sorted(reply.get("data") or [], key=lambda row: row.get("index", 0))
        if len(rows) != len(texts):
            raise ClientError(f"asked for {len(texts)} embeddings, got {len(rows)}")
        self.tokens += int((reply.get("usage") or {}).get("prompt_tokens") or 0)
        return [row["embedding"] for row in rows]

    def embed_batched(self, texts: list[str], *, max_items: int = 32, max_chars: int = 24000) -> list[list[float]]:
        """Embed in batches bounded by count and characters, so one request never
        exceeds what a single-slot embedding server will take."""
        out: list[list[float]] = []
        batch: list[str] = []
        size = 0
        for text in texts:
            if batch and (len(batch) >= max_items or size + len(text) > max_chars):
                out.extend(self.embed(batch))
                batch, size = [], 0
            batch.append(text)
            size += len(text)
        if batch:
            out.extend(self.embed(batch))
        return out


# ----------------------------------------------------------------------------- chat


def chat(settings: Settings, messages: list[dict[str, str]], *, max_tokens: int = 800,
         temperature: float = 0.2) -> dict[str, Any]:
    """One chat completion. Returns content, usage, wall time, and server timings
    (llama.cpp-style servers report prompt and decode speed; others don't)."""
    url = settings.llm_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {"model": settings.llm_model, "messages": messages, "max_tokens": max_tokens,
                               "temperature": temperature}
    if settings.llm_thinking == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.perf_counter()
    reply = post_json(url, payload, api_key=settings.llm_api_key, timeout=settings.request_timeout)
    wall = time.perf_counter() - started
    choice = (reply.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "content": (message.get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "usage": reply.get("usage") or {},
        "timings": reply.get("timings") or {},
        "wall_s": wall,
    }
