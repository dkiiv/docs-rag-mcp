"""Generate a retrieval eval set from an index: one question per sampled section.

The local chat model reads one randomly sampled section and writes a question
that the section answers. The section's file and line range become the expected
result. Sampling is seeded, and at most one section is taken per file, so the
set spreads across the corpus.

Caveat (also in the README): generated questions tend to reuse the section's
words, which flatters keyword search. The prompt asks for paraphrase, and the
results should be read with that in mind.

    python eval/make_questions.py --db index.sqlite --n 40 --seed 7 --out eval/q.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsrag import config  # noqa: E402
from docsrag.clients import ClientError, chat  # noqa: E402

PROMPT = """Below is one section from a documentation set.

Write ONE question that a user of this documentation might ask, which this section answers.
Rules:
- The question must make sense on its own, without seeing the section.
- Use your own words. Do not copy distinctive phrases, headings, or file names from the section.
- Ask about one specific fact, setting, command, or procedure.
- At most 25 words.
Then give the short answer, taken from the section.

Reply with JSON only: {{"question": "...", "answer": "..."}}

Section ({path}, {label}):
---
{text}
---"""


def _code_ratio(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    fenced, in_fence = 0, False
    for line in lines:
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            fenced += 1
        elif in_fence or line.lstrip().startswith("|"):
            fenced += 1
    return fenced / len(lines)


def _parse(content: str) -> dict | None:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and str(data.get("question", "")).strip():
        return {"question": str(data["question"]).strip(), "answer": str(data.get("answer", "")).strip()}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-code", type=float, default=0.5, help="skip sections that are mostly code/tables")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT c.id, c.title, c.heading, c.text, c.start_line, c.end_line, d.path "
                        "FROM chunks c JOIN docs d ON d.id = c.doc_id ORDER BY c.id").fetchall()
    by_doc: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        if len(row["text"]) >= args.min_chars and _code_ratio(row["text"]) <= args.max_code:
            by_doc.setdefault(row["path"], []).append(row)

    rng = random.Random(args.seed)
    paths = sorted(by_doc)
    rng.shuffle(paths)
    settings = config.load()
    written = tried = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for path in paths:
            if written >= args.n:
                break
            row = rng.choice(by_doc[path])
            tried += 1
            label = f"{row['title']} > {row['heading']}" if row["heading"] else row["title"]
            try:
                reply = chat(settings, [{"role": "user", "content": PROMPT.format(
                    path=path, label=label, text=row["text"][:3000])}], max_tokens=1000, temperature=0.3)
            except ClientError as exc:
                print(f"skip {path}: {exc}", file=sys.stderr)
                continue
            parsed = _parse(reply["content"])
            if not parsed:
                print(f"skip {path}: unparseable reply", file=sys.stderr)
                continue
            written += 1
            record = {"id": f"q{written:03d}", **parsed, "path": path, "start_line": row["start_line"],
                      "end_line": row["end_line"], "section": label}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{record['id']} {path}: {parsed['question']}", file=sys.stderr)
    print(f"tried {tried} sections, wrote {written} questions, skipped {tried - written}; {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
