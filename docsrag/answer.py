"""Retrieval-augmented answers with numbered citations, from a local chat model."""

from __future__ import annotations

import re
from typing import Any

from .clients import chat
from .config import Settings
from .search import Searcher

NOT_FOUND = "The docs don't say."

SYSTEM_PROMPT = f"""You answer questions using only the numbered excerpts you are given.
Rules:
- Use only facts stated in the excerpts. Do not add outside knowledge.
- Cite each claim with the number of the excerpt it came from, in square brackets, like [2].
- If the excerpts do not answer the question, reply with exactly: {NOT_FOUND}
- Be brief: a few sentences or a short list. Quote commands, paths and values exactly."""

_CITE = re.compile(r"\[(\d{1,2})\]")


def build_context(hits) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit.path}, lines {hit.start_line}-{hit.end_line} ({hit.label})\n{hit.text}")
    return "\n\n".join(blocks)


def ask(settings: Settings, searcher: Searcher, question: str, *, k: int = 6, mode: str = "hybrid",
        max_tokens: int | None = None) -> dict[str, Any]:
    hits = searcher.search(question, k=k, mode=mode)
    retrieval_ms = searcher.last_timings.get("total_ms", 0.0)
    if not hits:
        return {"answer": NOT_FOUND, "sources": [], "cited": [], "retrieval_ms": retrieval_ms}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{build_context(hits)}\n\nQuestion: {question}"},
    ]
    reply = chat(settings, messages, max_tokens=max_tokens or settings.llm_max_tokens)
    answer = reply["content"]
    # A reasoning model can spend the whole token budget thinking and return no
    # answer at all. Say so plainly instead of returning an empty string.
    empty = not answer
    if empty:
        answer = (f"(no answer: the model stopped after {reply['usage'].get('completion_tokens', '?')} tokens "
                  f"without replying, finish_reason={reply['finish_reason']})")
    cited_numbers = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(hits)})
    sources = [{"n": n, **hit.as_dict(snippet_chars=0)} for n, hit in enumerate(hits, start=1)]
    for source in sources:
        source.pop("text", None)
    return {
        "answer": answer,
        "not_found": (not empty) and answer.strip().startswith(NOT_FOUND.rstrip(".")),
        "empty": empty,
        "sources": sources,
        "cited": [sources[n - 1] for n in cited_numbers],
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_wall_s": round(reply["wall_s"], 2),
        "finish_reason": reply["finish_reason"],
        "usage": reply["usage"],
        "timings": reply["timings"],
    }
