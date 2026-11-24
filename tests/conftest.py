"""Shared fixtures. Every test runs offline: DOCSRAG_EMBED_URL=hash:// swaps in a
deterministic bag-of-words embedder, so no model server is needed."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from docsrag import config, store
from docsrag.clients import Embedder
from docsrag.search import Searcher

DOCS = {
    "runbooks/backup.md": """\
        # Backups

        Nightly snapshots run at 02:00 and keep 14 days.

        ## Restore a snapshot

        Stop the service first, then run `restore-snapshot --latest` and start it again.

        ## Offsite copy

        A weekly copy is pushed to cold storage with rclone.
        """,
    "runbooks/gpu.md": """\
        ---
        title: GPU inference host
        ---

        # Ignored because front matter has a title

        ## Embedding server

        The embedding model runs on card 2 with a context of 8192 tokens.
        If a long input fails with "input is too large to process", raise the physical batch size.

        ```bash
        # not a heading inside a fence
        llama-server --embedding --pooling cls
        ```
        """,
    "notes/network.md": """\
        # Network

        Public services come in through an outbound tunnel. No inbound ports are open.
        """,
}


def write_corpus(root: Path) -> Path:
    for rel, body in DOCS.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    (root / "secret.txt").write_text("not markdown, never indexed\n", encoding="utf-8")
    return root


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCSRAG_EMBED_URL", "hash://")
    monkeypatch.setenv("DOCSRAG_DB", str(tmp_path / "index.sqlite"))
    corpus = write_corpus(tmp_path / "corpus")
    settings = config.load()
    conn = store.connect(settings.db_path)
    embedder = Embedder(settings)
    report = store.build_index(conn, corpus, embedder)
    return {"tmp": tmp_path, "corpus": corpus, "settings": settings, "conn": conn, "embedder": embedder,
            "searcher": Searcher(conn, embedder), "report": report}


@pytest.fixture()
def git_corpus(env):
    corpus = env["corpus"]
    run = lambda *args: subprocess.run(["git", "-C", str(corpus), *args], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    run("add", "-A")
    run("commit", "-q", "-m", "initial docs")
    return env
