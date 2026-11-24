"""SQLite store: documents, chunks, float32 vectors and an FTS5 index.

One file holds everything, so an index is easy to copy, inspect with the
sqlite3 shell, or delete and rebuild. Vectors are loaded into one numpy matrix
for search; at a few thousand chunks an exact scan takes milliseconds, so
there is no approximate index to tune.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .chunk import Chunk, split_markdown
from .clients import Embedder

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    n_chars INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, heading, text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, heading, text) VALUES (new.id, new.title, new.heading, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, heading, text)
    VALUES ('delete', old.id, old.title, old.heading, old.text);
END;
"""

DOC_SUFFIXES = (".md", ".mdx", ".markdown")
DEFAULT_EXCLUDES = (".git/*", "*/.git/*", "node_modules/*", "*/node_modules/*", ".venv/*", "*/.venv/*")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}


def set_meta(conn: sqlite3.Connection, **values: object) -> None:
    conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     [(key, str(value)) for key, value in values.items()])


@dataclass
class IndexReport:
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    embed_tokens: int = 0
    embed_seconds: float = 0.0
    total_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def iter_docs(root: Path, excludes: Iterable[str] = ()) -> list[Path]:
    patterns = tuple(DEFAULT_EXCLUDES) + tuple(excludes)
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            continue
        found.append(path)
    return found


def build_index(conn: sqlite3.Connection, root: Path, embedder: Embedder, *, excludes: Iterable[str] = (),
                max_chars: int = 1500, progress: Optional[Callable[[str], None]] = None) -> IndexReport:
    """Index every markdown file under ``root``. Unchanged files (same sha256)
    are skipped, changed files are re-chunked and re-embedded, and files that
    disappeared are dropped."""
    started = time.perf_counter()
    root = root.resolve()
    meta = get_meta(conn)
    if meta.get("embed_model") and meta["embed_model"] != embedder.model:
        raise RuntimeError(f"index was built with {meta['embed_model']!r}, not {embedder.model!r}; "
                           "delete the database to rebuild with a different embedding model")
    report = IndexReport()
    seen_paths = set()
    tokens_before = embedder.tokens

    for path in iter_docs(root, excludes):
        rel = path.relative_to(root).as_posix()
        seen_paths.add(rel)
        report.files_seen += 1
        raw = path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        row = conn.execute("SELECT id, sha256 FROM docs WHERE path = ?", (rel,)).fetchone()
        if row and row["sha256"] == digest:
            report.files_unchanged += 1
            continue
        chunks = split_markdown(raw, rel, max_chars=max_chars)
        t0 = time.perf_counter()
        vectors = embedder.embed_batched([c.embed_text() for c in chunks])
        report.embed_seconds += time.perf_counter() - t0
        with conn:
            if row:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (row["id"],))
                conn.execute("DELETE FROM docs WHERE id = ?", (row["id"],))
            title = chunks[0].title if chunks else path.stem
            doc_id = conn.execute("INSERT INTO docs(path, sha256, title, n_chars) VALUES (?, ?, ?, ?)",
                                  (rel, digest, title, len(raw))).lastrowid
            conn.executemany(
                "INSERT INTO chunks(doc_id, ord, title, heading, text, start_line, end_line, vec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(doc_id, i, c.title, c.heading, c.text, c.start_line, c.end_line,
                  np.asarray(v, dtype=np.float32).tobytes()) for i, (c, v) in enumerate(zip(chunks, vectors))])
        report.files_indexed += 1
        report.chunks_written += len(chunks)
        if progress:
            progress(f"{rel}: {len(chunks)} chunks")

    with conn:
        for row in conn.execute("SELECT id, path FROM docs").fetchall():
            if row["path"] not in seen_paths:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (row["id"],))
                conn.execute("DELETE FROM docs WHERE id = ?", (row["id"],))
                report.files_removed += 1
        counts = conn.execute("SELECT (SELECT COUNT(*) FROM docs), (SELECT COUNT(*) FROM chunks)").fetchone()
        dim_row = conn.execute("SELECT vec FROM chunks LIMIT 1").fetchone()
        set_meta(conn, root=str(root), embed_model=embedder.model,
                 dim=(len(dim_row["vec"]) // 4) if dim_row else 0,
                 n_docs=counts[0], n_chunks=counts[1], excludes=json.dumps(list(excludes)),
                 indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 generation=str(time.time_ns()))
    report.embed_tokens = embedder.tokens - tokens_before
    report.total_seconds = time.perf_counter() - started
    return report


def chunk_rows(conn: sqlite3.Connection, ids: Iterable[int]) -> dict[int, sqlite3.Row]:
    ids = list(ids)
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT c.id, c.title, c.heading, c.text, c.start_line, c.end_line, d.path "
        f"FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id IN ({marks})", ids).fetchall()
    return {row["id"]: row for row in rows}


__all__ = ["Chunk", "IndexReport", "build_index", "chunk_rows", "connect", "get_meta", "iter_docs"]
