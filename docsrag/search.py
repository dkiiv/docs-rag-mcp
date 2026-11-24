"""Search: dense (embedding cosine), BM25 (SQLite FTS5) and hybrid (RRF).

Hybrid is the default. Dense search finds paraphrases ("how do I restart the
model server" vs "start / stop"), BM25 finds exact tokens the embedding blurs
(error strings, flag names, port numbers). Reciprocal rank fusion combines the
two rankings without having to calibrate their scores against each other.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .clients import Embedder
from .store import chunk_rows, get_meta

MODES = ("hybrid", "dense", "bm25")
RRF_K = 60
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_STOP = frozenset("""a an and are as at be by can do does for from has have how i if in into is it its of on or
should so that the this to was were what when where which who why will with you your""".split())


@dataclass
class Hit:
    chunk_id: int
    path: str
    title: str
    heading: str
    start_line: int
    end_line: int
    text: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title

    def as_dict(self, snippet_chars: Optional[int] = 600) -> dict[str, object]:
        text = self.text if snippet_chars is None or len(self.text) <= snippet_chars \
            else self.text[:snippet_chars].rstrip() + " ..."
        return {"path": self.path, "section": self.label, "lines": f"{self.start_line}-{self.end_line}",
                "score": round(self.score, 4), "text": text}


def fts_query(text: str) -> str:
    """Free text to a safe FTS5 query: quoted terms OR'd together, stopwords dropped."""
    terms = [t for t in _TOKEN.findall(text) if t.lower() not in _STOP]
    if not terms:
        terms = _TOKEN.findall(text)
    seen: list[str] = []
    for term in terms:
        if term.lower() not in (s.lower() for s in seen):
            seen.append(term)
    return " OR ".join(f'"{term}"' for term in seen[:32])


def rrf(rankings: list[list[int]], *, k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class Searcher:
    """Holds the vector matrix in memory and reloads it when the index changes."""

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder) -> None:
        self.conn = conn
        self.embedder = embedder
        self._lock = threading.Lock()
        self._generation = None
        self._ids = np.zeros(0, dtype=np.int64)
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self.last_timings: dict[str, float] = {}

    def _load(self) -> None:
        generation = get_meta(self.conn).get("generation")
        with self._lock:
            if generation == self._generation and self._matrix.size:
                return
            rows = self.conn.execute("SELECT id, vec FROM chunks ORDER BY id").fetchall()
            if rows:
                self._ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
                self._matrix = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
                norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                self._matrix = self._matrix / np.where(norms == 0, 1, norms)
            else:
                self._ids = np.zeros(0, dtype=np.int64)
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._generation = generation

    def dense_ids(self, query: str, depth: int) -> list[int]:
        self._load()
        if not self._matrix.size:
            return []
        t0 = time.perf_counter()
        q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        self.last_timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        q /= np.linalg.norm(q) or 1.0
        scores = self._matrix @ q
        depth = min(depth, len(scores))
        top = np.argpartition(-scores, depth - 1)[:depth]
        top = top[np.argsort(-scores[top])]
        return [int(self._ids[i]) for i in top]

    def bm25_ids(self, query: str, depth: int) -> list[int]:
        match = fts_query(query)
        if not match:
            return []
        # Weights: title 3, heading 2, body 1. A term in a heading is a strong signal.
        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts, 3.0, 2.0, 1.0) LIMIT ?", (match, depth)).fetchall()
        return [int(r[0]) for r in rows]

    def search(self, query: str, *, k: int = 5, mode: str = "hybrid", depth: int = 50) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = max(1, min(int(k), 50))
        self.last_timings = {}
        t0 = time.perf_counter()
        ranks: dict[str, list[int]] = {}
        if mode in ("hybrid", "dense"):
            ranks["dense"] = self.dense_ids(query, max(depth, k))
        if mode in ("hybrid", "bm25"):
            ranks["bm25"] = self.bm25_ids(query, max(depth, k))
        if mode == "hybrid":
            scored = rrf([ranks["dense"], ranks["bm25"]])
        else:
            only = ranks[mode]
            scored = {cid: 1.0 / (RRF_K + r) for r, cid in enumerate(only, start=1)}
        best = sorted(scored, key=lambda cid: -scored[cid])[:k]
        rows = chunk_rows(self.conn, best)
        hits = []
        for cid in best:
            row = rows.get(cid)
            if row is None:
                continue
            hit_ranks = {name: ids.index(cid) + 1 for name, ids in ranks.items() if cid in ids}
            hits.append(Hit(chunk_id=cid, path=row["path"], title=row["title"], heading=row["heading"],
                            start_line=row["start_line"], end_line=row["end_line"], text=row["text"],
                            score=scored[cid], ranks=hit_ranks))
        self.last_timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return hits
