"""Measure retrieval quality and latency, and optionally answer quality.

Retrieval, per mode (bm25, dense, hybrid):
  file hit@k   the expected file is among the top k results
  MRR@10       mean of 1/rank of the first result from the expected file
  section hit@5  a top-5 result overlaps the expected line range
  latency      wall time of one search call, including the query embedding

Answers (--ask), hybrid retrieval with k=6:
  cited the source   the answer cites an excerpt from the expected file
  said not found     the model answered "The docs don't say."
  negatives          questions the docs don't answer; the right reply is "not found"
  speed              wall time per answer, and the server's own decode/prefill
                     tokens per second when it reports them (llama.cpp style)

    python eval/run_eval.py --db index.sqlite --questions eval/q.jsonl [--ask] [--negatives eval/neg.txt]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docsrag import answer, config, store  # noqa: E402
from docsrag.clients import Embedder  # noqa: E402
from docsrag.search import MODES, Searcher  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def retrieval(searcher: Searcher, questions: list[dict], depth: int = 10) -> dict:
    results = {}
    for mode in ("bm25", "dense", "hybrid"):
        hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
        rr, section5, latencies, per_q = 0.0, 0, [], []
        for q in questions:
            t0 = time.perf_counter()
            hits = searcher.search(q["question"], k=depth, mode=mode)
            latencies.append((time.perf_counter() - t0) * 1000)
            rank = next((i for i, h in enumerate(hits, start=1) if h.path == q["path"]), None)
            if rank:
                rr += 1.0 / rank
                for k in hits_at:
                    hits_at[k] += rank <= k
            overlap = any(h.path == q["path"] and h.start_line <= q["end_line"] and h.end_line >= q["start_line"]
                          for h in hits[:5])
            section5 += overlap
            per_q.append({"id": q["id"], "rank": rank, "section_hit5": overlap})
        n = len(questions)
        results[mode] = {
            "n": n,
            **{f"file_hit@{k}": round(v / n, 3) for k, v in hits_at.items()},
            "mrr@10": round(rr / n, 3),
            "section_hit@5": round(section5 / n, 3),
            "latency_ms_p50": round(percentile(latencies, 0.5), 1),
            "latency_ms_p95": round(percentile(latencies, 0.95), 1),
            "per_question": per_q,
        }
    return results


def answers(settings, searcher: Searcher, questions: list[dict], negatives: list[str]) -> dict:
    rows = []
    for q in questions:
        result = answer.ask(settings, searcher, q["question"], k=6)
        timings = result.get("timings") or {}
        rows.append({
            "id": q["id"], "cited_source": any(c["path"] == q["path"] for c in result["cited"]),
            "not_found": result["not_found"], "empty": result.get("empty", False),
            "any_citation": bool(result["cited"]),
            "wall_s": result.get("llm_wall_s", 0.0), "retrieval_ms": result.get("retrieval_ms", 0.0),
            "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            "decode_tps": timings.get("predicted_per_second"), "prefill_tps": timings.get("prompt_per_second"),
            "answer": result["answer"],
        })
        print(f"{q['id']} cited={rows[-1]['cited_source']} nf={rows[-1]['not_found']} {rows[-1]['wall_s']}s",
              file=sys.stderr)
    neg_rows = []
    for text in negatives:
        result = answer.ask(settings, searcher, text, k=6)
        neg_rows.append({"question": text, "not_found": result["not_found"], "answer": result["answer"]})
        print(f"neg not_found={result['not_found']}: {text}", file=sys.stderr)
    n = len(rows) or 1
    walls = [r["wall_s"] for r in rows]
    decode = [r["decode_tps"] for r in rows if r["decode_tps"]]
    prefill = [r["prefill_tps"] for r in rows if r["prefill_tps"]]
    return {
        "n": len(rows),
        "cited_source": round(sum(r["cited_source"] for r in rows) / n, 3),
        "any_citation": round(sum(r["any_citation"] for r in rows) / n, 3),
        "said_not_found": round(sum(r["not_found"] for r in rows) / n, 3),
        "empty_answers": sum(r["empty"] for r in rows),
        "wall_s_p50": round(percentile(walls, 0.5), 2), "wall_s_p95": round(percentile(walls, 0.95), 2),
        "decode_tps_median": round(statistics.median(decode), 1) if decode else None,
        "prefill_tps_median": round(statistics.median(prefill), 1) if prefill else None,
        "prompt_tokens_median": statistics.median([r["prompt_tokens"] for r in rows if r["prompt_tokens"]] or [0]),
        "negatives_n": len(neg_rows),
        "negatives_not_found": round(sum(r["not_found"] for r in neg_rows) / (len(neg_rows) or 1), 3),
        "rows": rows, "negative_rows": neg_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--negatives", type=Path)
    parser.add_argument("--ask", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import os
    os.environ["DOCSRAG_DB"] = str(args.db)
    settings = config.load()
    conn = store.connect(settings.db_path)
    searcher = Searcher(conn, Embedder(settings))
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    negatives = [line.strip() for line in args.negatives.read_text().splitlines()
                 if line.strip() and not line.startswith("#")] if args.negatives else []

    searcher.search("warm up", k=1)  # first call loads the matrix; don't time it
    meta = store.get_meta(conn)
    report = {"corpus": {"docs": int(meta["n_docs"]), "chunks": int(meta["n_chunks"]),
                         "embed_model": meta["embed_model"]},
              "llm_model": settings.llm_model, "llm_thinking": settings.llm_thinking,
              "llm_max_tokens": settings.llm_max_tokens,
              "date": time.strftime("%Y-%m-%d", time.gmtime()),
              "retrieval": retrieval(searcher, questions)}
    if args.ask:
        report["answers"] = answers(settings, searcher, questions, negatives)

    print("| mode | file hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | section hit@5 | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for mode in ("bm25", "dense", "hybrid"):
        r = report["retrieval"][mode]
        print(f"| {mode} | {r['file_hit@1']} | {r['file_hit@3']} | {r['file_hit@5']} | {r['file_hit@10']} | "
              f"{r['mrr@10']} | {r['section_hit@5']} | {r['latency_ms_p50']} | {r['latency_ms_p95']} |")
    if args.ask:
        a = report["answers"]
        print(f"\nanswers n={a['n']}: cited source {a['cited_source']}, any citation {a['any_citation']}, "
              f"said not found {a['said_not_found']}, empty {a['empty_answers']}, wall p50 {a['wall_s_p50']} s / p95 {a['wall_s_p95']} s, "
              f"decode {a['decode_tps_median']} tok/s, prefill {a['prefill_tps_median']} tok/s, "
              f"prompt tokens median {a['prompt_tokens_median']}; negatives {a['negatives_n']}: "
              f"not found {a['negatives_not_found']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
