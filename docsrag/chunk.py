"""Split markdown into heading-scoped chunks.

A chunk is one section (the text under a heading, down to the next heading of
any level), split further on blank lines when it is longer than ``max_chars``.
Each chunk keeps its heading path ("Page title > Section > Subsection") and its
line range, so a search hit can point at the exact place in the file.

Headings inside fenced code blocks are not headings. YAML front matter is
skipped, and its ``title:`` is used as the page title when there is one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$")


@dataclass(frozen=True)
class Chunk:
    path: str
    title: str
    heading: str  # "Section > Subsection", may be empty
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int

    @property
    def label(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title

    def embed_text(self) -> str:
        """What gets embedded: the heading path gives a short section context."""
        return f"{self.label}\n\n{self.text}"


def _front_matter(lines: list[str]) -> tuple[int, str]:
    """(first body line index, title from front matter or "")."""
    if not lines or lines[0].strip() != "---":
        return 0, ""
    title = ""
    for index in range(1, min(len(lines), 200)):
        if lines[index].strip() == "---":
            return index + 1, title
        match = _TITLE.match(lines[index])
        if match and not title:
            title = match.group(1)
    return 0, ""


def _split_long(body: list[tuple[int, str]], max_chars: int) -> list[list[tuple[int, str]]]:
    """Split (line_no, text) rows into pieces under max_chars, on blank lines
    outside code fences when possible, else on any line."""
    pieces: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    in_fence = False
    for row in body:
        line = row[1]
        if _FENCE.match(line):
            in_fence = not in_fence
        current.append(row)
        size += len(line) + 1
        at_break = (not line.strip() and not in_fence) or size >= max_chars * 1.5
        if size >= max_chars and at_break:
            pieces.append(current)
            current, size = [], 0
    if current:
        if pieces and sum(len(r[1]) + 1 for r in current) < max_chars // 4:
            pieces[-1].extend(current)  # don't leave a tiny tail on its own
        else:
            pieces.append(current)
    return pieces


def split_markdown(text: str, path: str, *, max_chars: int = 1500) -> list[Chunk]:
    lines = text.splitlines()
    start, title = _front_matter(lines)

    # Pass 1: sections. stack holds (level, heading text).
    sections: list[tuple[list[str], list[tuple[int, str]]]] = []
    stack: list[tuple[int, str]] = []
    body: list[tuple[int, str]] = []
    in_fence = False

    def flush() -> None:
        if any(row[1].strip() for row in body):
            sections.append(([h for _, h in stack], list(body)))
        body.clear()

    for index in range(start, len(lines)):
        line = lines[index]
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            if level == 1 and not title:
                title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            continue
        body.append((index + 1, line))
    flush()

    if not title:
        title = PurePosixPath(path).stem.replace("-", " ").replace("_", " ")

    chunks: list[Chunk] = []
    for headings, rows in sections:
        # Drop the page title from the heading path: it is already the title.
        if headings and headings[0] == title:
            headings = headings[1:]
        heading = " > ".join(headings)
        for piece in _split_long(rows, max_chars):
            # Trim blank lines at both ends so line ranges point at text.
            while piece and not piece[0][1].strip():
                piece = piece[1:]
            while piece and not piece[-1][1].strip():
                piece = piece[:-1]
            if not piece:
                continue
            chunk_text = "\n".join(row[1] for row in piece).strip()
            chunks.append(Chunk(path=path, title=title, heading=heading, text=chunk_text,
                                start_line=piece[0][0], end_line=piece[-1][0]))
    return chunks
