"""Command line: index, search, ask, stats, changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import access, answer, config, store
from .clients import Embedder
from .search import MODES, Searcher


def _open(settings: config.Settings):
    conn = store.connect(settings.db_path)
    embedder = Embedder(settings)
    return conn, embedder, Searcher(conn, embedder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docsrag", description="Semantic search and cited answers over markdown docs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index (or refresh) a folder of markdown")
    p_index.add_argument("root", type=Path)
    p_index.add_argument("--exclude", action="append", default=[], help="glob relative to root; repeatable")
    p_index.add_argument("--max-chars", type=int, default=1500)
    p_index.add_argument("-v", "--verbose", action="store_true")

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--mode", choices=MODES, default="hybrid")
    p_search.add_argument("--json", action="store_true")

    p_ask = sub.add_parser("ask", help="answer a question from the docs, with citations")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=6)
    p_ask.add_argument("--mode", choices=MODES, default="hybrid")
    p_ask.add_argument("--json", action="store_true")

    sub.add_parser("stats", help="what is in the index")

    p_changes = sub.add_parser("changes", help="recent git commits that touched indexed docs")
    p_changes.add_argument("--days", type=int, default=14)

    args = parser.parse_args(argv)
    settings = config.load()
    conn, embedder, searcher = _open(settings)

    if args.cmd == "index":
        report = store.build_index(conn, args.root, embedder, excludes=args.exclude, max_chars=args.max_chars,
                                   progress=(lambda line: print(line, file=sys.stderr)) if args.verbose else None)
        print(json.dumps(report.as_dict(), indent=2))
    elif args.cmd == "search":
        hits = searcher.search(args.query, k=args.k, mode=args.mode)
        if args.json:
            print(json.dumps([h.as_dict() for h in hits], indent=2))
        else:
            for n, hit in enumerate(hits, start=1):
                print(f"{n}. {hit.path}:{hit.start_line}-{hit.end_line}  {hit.label}  ({hit.score:.4f} {hit.ranks})")
            print(f"[{searcher.last_timings.get('total_ms', 0):.1f} ms]", file=sys.stderr)
    elif args.cmd == "ask":
        result = answer.ask(settings, searcher, args.question, k=args.k, mode=args.mode)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result["answer"])
            print()
            for source in result["sources"]:
                mark = "*" if source in result["cited"] else " "
                print(f" {mark}[{source['n']}] {source['path']}:{source['lines']}  {source['section']}")
    elif args.cmd == "stats":
        meta = store.get_meta(conn)
        meta.pop("generation", None)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "changes":
        print(json.dumps(access.recent_changes(conn, days=args.days), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
