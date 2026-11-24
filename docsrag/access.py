"""Read-only access to the indexed corpus: file sections and git history.

Both functions only ever touch files that are inside the indexed root AND in
the index, so an MCP client can't use them to read anything else on the host.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .store import get_meta

MAX_READ_LINES = 400


class AccessError(ValueError):
    pass


def corpus_root(conn: sqlite3.Connection) -> Path:
    root = get_meta(conn).get("root")
    if not root:
        raise AccessError("the index is empty; run `docsrag index <folder>` first")
    return Path(root)


def resolve_indexed(conn: sqlite3.Connection, rel_path: str) -> Path:
    """The absolute path of an indexed document, or AccessError."""
    root = corpus_root(conn).resolve()
    rel = str(rel_path or "").strip().lstrip("/")
    if not rel or "\x00" in rel:
        raise AccessError("path is required")
    candidate = (root / rel).resolve()
    if root not in candidate.parents:
        raise AccessError(f"{rel_path!r} is outside the indexed folder")
    known = conn.execute("SELECT 1 FROM docs WHERE path = ?", (candidate.relative_to(root).as_posix(),)).fetchone()
    if not known:
        raise AccessError(f"{rel_path!r} is not an indexed document (use the paths search returns)")
    return candidate


def read_lines(conn: sqlite3.Connection, rel_path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
    path = resolve_indexed(conn, rel_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line or 1))
    end = int(end_line or 0)
    if end <= 0 or end > len(lines):
        end = len(lines)
    truncated = False
    if end - start + 1 > MAX_READ_LINES:
        end, truncated = start + MAX_READ_LINES - 1, True
    return {"path": rel_path, "lines": f"{start}-{end}", "total_lines": len(lines), "truncated": truncated,
            "text": "\n".join(lines[start - 1:end])}


def recent_changes(conn: sqlite3.Connection, days: int = 14, limit: int = 30) -> dict[str, Any]:
    """Commits that touched indexed docs in the last ``days`` days (git log, read-only)."""
    root = corpus_root(conn)
    days = max(1, min(int(days), 3650))
    limit = max(1, min(int(limit), 200))
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"git": False, "commits": [], "note": "the indexed folder is not in a git repository"}
    marker = "\x1e"
    out = subprocess.run(
        ["git", "-C", str(root), "log", f"--since={days} days ago", f"-n{limit}", "--name-only",
         f"--format={marker}%h%x1f%cs%x1f%s", "--", "."],
        capture_output=True, text=True, timeout=20).stdout
    indexed = {row[0] for row in conn.execute("SELECT path FROM docs")}
    root_rel = Path(root).resolve().relative_to(Path(top).resolve()).as_posix()
    prefix = "" if root_rel == "." else root_rel + "/"
    commits = []
    for block in out.split(marker):
        if not block.strip():
            continue
        head, *files = block.strip("\n").split("\n")
        sha, date, subject = (head.split("\x1f") + ["", "", ""])[:3]
        docs = [f[len(prefix):] for f in files if f.startswith(prefix) and f[len(prefix):] in indexed]
        if docs:
            commits.append({"commit": sha, "date": date, "subject": subject, "docs": docs})
    return {"git": True, "days": days, "commits": commits}
