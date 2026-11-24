"""MCP server (stdio) exposing one docs index to an agent.

Run one server per corpus. ``DOCSRAG_NAME`` and ``DOCSRAG_DESCRIPTION`` tell the
agent which corpus this is, so two connectors ("labdocs", "hermesdocs") can sit
side by side in the same agent without confusing it.

Tools:
  search_docs     ranked sections with path, line range and a snippet
  read_doc        exact lines from an indexed file
  ask_docs        a short answer from the local chat model, with citations
  recent_changes  git commits that touched indexed docs
  index_info      what is indexed, when, and with which embedding model

Everything is read-only. read_doc and recent_changes only reach files that are
in the index.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from . import access, answer, config, store
from .clients import ClientError, Embedder
from .search import MODES, Searcher

settings = config.load()
_conn = store.connect(settings.db_path)
_embedder = Embedder(settings)
_searcher = Searcher(_conn, _embedder)

server = MCPServer(
    name=settings.server_name,
    instructions=(
        f"Search and read {settings.corpus_description}. Start with search_docs. Use read_doc to get the "
        "full section before relying on a snippet. ask_docs returns a short answer with numbered citations "
        "from a local model; check the cited lines when the answer matters."),
)


def _error(exc: Exception) -> dict[str, Any]:
    return {"error": str(exc)}


@server.tool()
def search_docs(query: str, k: int = 5, mode: str = "hybrid") -> dict[str, Any]:
    """Find the sections that best match a question or keywords.

    mode: hybrid (default, best overall), dense (meaning), bm25 (exact words,
    flags, error strings). Returns up to k sections with path, line range and a
    text snippet.
    """
    if mode not in MODES:
        return {"error": f"mode must be one of {list(MODES)}"}
    try:
        hits = _searcher.search(query, k=k, mode=mode)
    except ClientError as exc:
        return _error(exc)
    return {"query": query, "mode": mode, "results": [h.as_dict() for h in hits],
            "took_ms": round(_searcher.last_timings.get("total_ms", 0.0), 1)}


@server.tool()
def read_doc(path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
    """Read lines from an indexed document. Use the path and line range that
    search_docs returned. end_line 0 means to the end (at most 400 lines)."""
    try:
        return access.read_lines(_conn, path, start_line, end_line)
    except access.AccessError as exc:
        return _error(exc)


@server.tool()
def ask_docs(question: str, k: int = 6) -> dict[str, Any]:
    """Answer a question from the docs with the local chat model. The answer
    cites excerpts as [n]; `cited` lists the files and lines behind them. If the
    docs don't cover it, the answer says so instead of guessing."""
    try:
        result = answer.ask(settings, _searcher, question, k=k)
    except ClientError as exc:
        return _error(exc)
    return {key: result.get(key) for key in ("answer", "not_found", "cited", "sources", "retrieval_ms", "llm_wall_s")}


@server.tool()
def recent_changes(days: int = 14, limit: int = 30) -> dict[str, Any]:
    """Commits from the last `days` days that touched indexed docs (needs the
    docs folder to be a git repository). Useful for "what changed recently"."""
    try:
        return access.recent_changes(_conn, days=days, limit=limit)
    except access.AccessError as exc:
        return _error(exc)


@server.tool()
def index_info() -> dict[str, Any]:
    """Which corpus this server searches, how big it is, and when it was indexed."""
    meta = store.get_meta(_conn)
    return {"name": settings.server_name, "description": settings.corpus_description,
            "documents": int(meta.get("n_docs", 0)), "chunks": int(meta.get("n_chunks", 0)),
            "embed_model": meta.get("embed_model"), "indexed_at": meta.get("indexed_at")}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
