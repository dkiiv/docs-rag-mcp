# docs-rag-mcp

Semantic search and cited answers over a folder of markdown, served to AI
agents over MCP. It runs on local models: bge-m3 for embeddings and any
OpenAI-compatible chat model for answers. Nothing leaves your network.

I use it to give my Hermes Agent two connectors: one for my homelab runbooks
and one for the Hermes Agent docs. The agent searches first, reads the exact
lines, and only then answers.

## How it works

- **Chunking.** Each markdown file is split at its headings. A long section is
  split again at blank lines, never inside a code block. Every chunk keeps its
  heading path ("Page > Section > Subsection") and its line range, so a result
  points at an exact place in a file.
- **Index.** One SQLite file holds documents, chunks, float32 vectors and an
  FTS5 full-text index. Re-indexing skips files whose sha256 hasn't changed,
  re-embeds changed files, and drops deleted ones.
- **Search.** Three modes:
  - `dense`: cosine similarity on bge-m3 vectors. Finds paraphrases.
  - `bm25`: SQLite FTS5. Finds exact tokens such as flag names, error
    strings and port numbers.
  - `hybrid` (default): both rankings fused with reciprocal rank fusion, so
    neither score needs calibrating against the other.

  Vectors are scanned exactly with numpy. At ten thousand chunks that takes
  milliseconds, so there is no approximate index to tune.
- **Answers.** `ask` retrieves six sections, numbers them, and asks the chat
  model to answer only from them, cite each claim as `[n]`, and reply "The docs
  don't say." when they don't. The reply lists which files and lines the
  citations point to.

## MCP tools

Run one server per corpus. All tools are read-only.

| Tool | What it does |
|---|---|
| `search_docs(query, k, mode)` | Ranked sections: path, line range, heading path, snippet. |
| `read_doc(path, start_line, end_line)` | Exact lines from an indexed file, at most 400 per call. |
| `ask_docs(question, k)` | A short answer from the local model, with numbered citations. |
| `recent_changes(days, limit)` | Git commits that touched indexed docs. |
| `index_info()` | Which corpus this is, its size, and when it was indexed. |

`read_doc` and `recent_changes` only reach files that are in the index. A path
outside the indexed folder, or a file there that wasn't indexed, gets an error.

## Measured results

Measured 2026-11-24 on my homelab:

- The embedding model is bge-m3 (FP16 GGUF) on llama.cpp, sharing one V100 with
  the chat model.
- The chat model is a Mixture-of-Experts model on three V100s, behind an
  OpenAI-compatible server.
- Search runs on CPU in a separate container.

### Corpora

| Corpus | Files | Chunks | Tokens embedded | Embedding time |
|---|---|---|---|---|
| Hermes Agent docs, `v0.22.5` tag (`website/docs`) | 455 | 10,709 | 2,714,261 | 172 s (about 15,700 tokens/s) |
| My homelab runbooks (private) | 31 | 334 | 117,040 | 8.0 s (about 14,700 tokens/s) |

Re-indexing the homelab runbooks with nothing changed takes 0.08 s.

### Retrieval

The questions were made with `eval/make_questions.py`. The local chat model
read one randomly chosen section per file (seed 7) and wrote a question that
the section answers. A result counts as a hit when it comes from the same file.
"Section hit@5" is stricter: a top-5 result has to overlap the section's line
range. Latency is one search call, including embedding the query.

Hermes docs, 40 questions:

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Section hit@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.825 | 0.85 | 0.90 | 0.925 | 0.848 | 0.80 | 7.0 ms | 12.7 ms |
| dense | 0.75 | 0.80 | 0.80 | 0.80 | 0.775 | 0.70 | 13.4 ms | 18.6 ms |
| hybrid | 0.80 | 0.85 | 0.90 | 0.95 | 0.833 | 0.825 | 21.3 ms | 30.9 ms |

Homelab runbooks, 23 questions:

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Section hit@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.652 | 0.783 | 0.87 | 0.87 | 0.732 | 0.739 | 1.2 ms | 1.5 ms |
| dense | 0.783 | 0.87 | 0.87 | 0.913 | 0.824 | 0.783 | 11.4 ms | 15.5 ms |
| hybrid | 0.783 | 0.87 | 0.913 | 0.957 | 0.837 | 0.783 | 12.4 ms | 14.4 ms |

Hybrid is the most reliable mode at depth. It ties or beats both other modes
at hit@5 and hit@10 on both corpora. It doesn't win hit@1: BM25 is ahead on the
Hermes docs, and dense ties it on the runbooks.

### Answers

`ask` uses hybrid retrieval with k=6. I ran it twice: once with the model's
reasoning on (the default), and once with it off (`DOCSRAG_LLM_THINKING=off`).
"Cited the source" means the answer cites a section from the file the question
was written from. There are also 6 questions per corpus that the docs don't
answer, checked with grep. For those, the right reply is "The docs don't say."

| | Hermes, reasoning on | Hermes, reasoning off | Runbooks, reasoning on | Runbooks, reasoning off |
|---|---|---|---|---|
| Cited the source | 35/40 | 35/40 | 20/23 | 20/23 |
| Said "not found" on an answerable question | 2 | 2 | 1 | 0 |
| Empty answer (ran out of tokens while reasoning) | 1 | 0 | 0 | 0 |
| Unanswerable questions answered "not found" | 6/6 | 6/6 | 6/6 | 6/6 |
| Answer time, p50 / p95 | 3.6 s / 5.5 s | 2.6 s / 3.6 s | 3.8 s / 5.5 s | 3.1 s / 4.1 s |
| Tokens generated, median (reasoning included) | 181 | 51 | 166 | 59 |
| Decode speed reported by the server, median | 106 tok/s | 76 tok/s | 105 tok/s | 77 tok/s |
| Prompt size, median | 1,622 tokens | 1,586 tokens | 2,433 tokens | 2,397 tokens |

With reasoning off, answers came back about a second faster at the same citation
rate, and none came back empty. My Hermes connectors run with it off. The
server reported a lower decode speed with reasoning off, but it generated about
a third as many tokens, so answers still finished sooner.

### Caveats

- **The questions are machine-written.** The model was told to paraphrase, but
  generated questions still tend to reuse the section's words, which favors
  BM25. That probably explains BM25's hit@1 on the Hermes docs. The generator
  also skips a section when the reply isn't valid JSON. The Hermes set was made
  with a 400-token reply budget, and many replies ran out of room while the
  model was still reasoning, so those sections were skipped. The script now
  allows 1,000 tokens, which the runbook set used.
- **One right file.** A result from another page that says the same thing
  counts as a miss. In one Hermes miss, the answer gave the right command
  (`hermes setup --portal`) but cited a different page.
- **Small sets.** One question is 2.5 points on the Hermes set and 4.3 on the
  runbooks.
- **Grounding, not correctness.** "Cited the source" checks where the answer
  came from. Nobody graded the answers by hand.
- The runbook questions stay private because they describe my network. The
  Hermes questions and negatives are in `eval/`, so those numbers can be
  reproduced.

## Install

```bash
uv tool install git+https://github.com/dkiiv/docs-rag-mcp   # gives `docsrag` and `docsrag-mcp`
# or, from a clone:
uv venv && uv pip install -e '.[test]'
```

Python 3.11 or newer. Runtime dependencies: `numpy` and `mcp`.

## Use

```bash
export DOCSRAG_DB=~/.local/share/docsrag/mydocs.sqlite
export DOCSRAG_EMBED_URL=http://localhost:8081/v1     # serves bge-m3 at /v1/embeddings
export DOCSRAG_LLM_URL=http://localhost:8080/v1       # any OpenAI-compatible chat server
export DOCSRAG_LLM_MODEL=my-model

docsrag index ~/notes --exclude 'drafts/*'       # run again any time; unchanged files are skipped
docsrag search "how do I restart the embedding server"
docsrag search '"input is too large to process"' --mode bm25
docsrag ask "which GPU does the embedding model run on?"
docsrag changes --days 7
```

For llama.cpp, serve bge-m3 with `--embedding --pooling cls` and a physical
batch at least as large as the context (`--ubatch-size 8192`). Otherwise inputs
longer than the batch fail with HTTP 500.

## Connect it to Hermes Agent

In `~/.hermes/config.yaml`, one entry per corpus:

```yaml
mcp_servers:
  labdocs:
    command: /home/me/.local/bin/docsrag-mcp
    env:
      DOCSRAG_NAME: labdocs
      DOCSRAG_DESCRIPTION: my homelab runbooks (hosts, services, recovery steps)
      DOCSRAG_DB: /home/me/.local/share/docsrag/labdocs.sqlite
      DOCSRAG_EMBED_URL: http://localhost:8081/v1
      DOCSRAG_LLM_URL: http://localhost:8080/v1
      DOCSRAG_LLM_MODEL: my-model
      DOCSRAG_LLM_API_KEY: ${MY_LLM_KEY}      # resolved by Hermes from its .env
      DOCSRAG_LLM_THINKING: "off"
  hermesdocs:
    command: /home/me/.local/bin/docsrag-mcp
    env:
      DOCSRAG_NAME: hermesdocs
      DOCSRAG_DESCRIPTION: the Hermes Agent documentation
      DOCSRAG_DB: /home/me/.local/share/docsrag/hermesdocs.sqlite
      DOCSRAG_EMBED_URL: http://localhost:8081/v1
      DOCSRAG_LLM_URL: http://localhost:8080/v1
      DOCSRAG_LLM_MODEL: my-model
      DOCSRAG_LLM_THINKING: "off"
```

`hermes mcp test labdocs` should list five tools. Hermes prefixes them with
the server name (`mcp__labdocs__search_docs` on the build I run). Any other MCP
client that can start a stdio server works the same way.

Checked end to end on 2026-11-24: asked through `hermes chat` which GPU the
embedding server uses, Hermes (on the local model) called `search_docs` twice
and `ask_docs` once, and answered with the right card, its VRAM use, and the
file and line ranges it came from. The tool calls took 0.04 s, 0.02 s and
3.9 s.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `DOCSRAG_DB` | `docsrag.sqlite` | The index file. |
| `DOCSRAG_EMBED_URL` | `http://localhost:8081/v1` | OpenAI-compatible embeddings endpoint. `hash://` is the offline test embedder. |
| `DOCSRAG_EMBED_MODEL` | `bge-m3` | Model name sent to the embeddings endpoint. |
| `DOCSRAG_EMBED_API_KEY` | none | Bearer token, if the server wants one. |
| `DOCSRAG_LLM_URL` | `http://localhost:8080/v1` | OpenAI-compatible chat endpoint, used by `ask`. |
| `DOCSRAG_LLM_MODEL` | `local` | Model name sent to the chat endpoint. |
| `DOCSRAG_LLM_API_KEY` | none | Bearer token for the chat endpoint. |
| `DOCSRAG_LLM_THINKING` | `default` | `off` sends `chat_template_kwargs.enable_thinking=false` (llama.cpp and vLLM with Qwen-style templates). |
| `DOCSRAG_LLM_MAX_TOKENS` | `800` | Answer budget, reasoning included. |
| `DOCSRAG_NAME`, `DOCSRAG_DESCRIPTION` | `docsrag` | What the MCP server tells the agent about its corpus. |
| `DOCSRAG_TIMEOUT` | `180` | HTTP timeout in seconds. |

## What it touches

- **Network:** only the embeddings and chat endpoints you configure.
- **Files:** reads the folder you index and writes the one SQLite file.
- **Processes:** `recent_changes` runs `git log` in the indexed folder.
- It stores no credentials. API keys come from the environment.

## Reproduce the Hermes docs numbers

```bash
git clone --depth 1 --branch v0.22.5 https://github.com/NousResearch/hermes-agent
export DOCSRAG_DB=hermes-v0.22.5.sqlite   # plus the EMBED/LLM variables above
docsrag index hermes-agent/website/docs
python eval/run_eval.py --db hermes-v0.22.5.sqlite \
  --questions eval/hermes-docs-v0.22.5.questions.jsonl \
  --negatives eval/hermes-docs.negatives.txt --ask
```

Retrieval numbers should match exactly with bge-m3. Answer numbers depend on
your chat model. `eval/make_questions.py` makes a question set for your own
docs.

## Tests

```bash
python -m pytest -q
```

32 tests, all offline. A hashing embedder stands in for the model server, so CI
needs no GPU. They cover the chunker, all three search modes, incremental
re-indexing, FTS query escaping, path traversal in `read_doc`, citation
parsing, and the real MCP server over stdio with the MCP client.

## Limits

- Markdown only (`.md`, `.mdx`, `.markdown`). MDX components are indexed as
  text.
- One embedding model per index. Switching models means deleting the index file
  and rebuilding.
- The whole vector matrix sits in memory: about 4 KB per chunk with bge-m3.
  That is fine at tens of thousands of chunks. Past that, use a vector database.

## License

MIT. See `LICENSE`.
