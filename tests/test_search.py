import pytest

from docsrag import store
from docsrag.search import fts_query, rrf


def test_index_report(env):
    report = env["report"]
    assert report.files_seen == 3 and report.files_indexed == 3
    assert report.chunks_written >= 5
    meta = store.get_meta(env["conn"])
    assert meta["embed_model"] == "hash-bow-256" and meta["n_docs"] == "3"


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_each_mode_finds_the_right_section(env, mode):
    hits = env["searcher"].search("restore a snapshot", k=3, mode=mode)
    assert hits[0].path == "runbooks/backup.md"
    assert "Restore a snapshot" in hits[0].label


def test_exact_error_string_via_bm25(env):
    hits = env["searcher"].search('input is too large to process', k=1, mode="bm25")
    assert hits[0].path == "runbooks/gpu.md"
    assert hits[0].title == "GPU inference host"


def test_hybrid_reports_component_ranks(env):
    hit = env["searcher"].search("outbound tunnel inbound ports", k=1)[0]
    assert hit.path == "notes/network.md"
    assert set(hit.ranks) == {"dense", "bm25"}


def test_fts_query_is_injection_safe(env):
    query = fts_query('restore" OR NOT * NEAR( -- ;DROP')
    assert query.count('"') % 2 == 0
    env["searcher"].search('restore" OR NOT * NEAR( -- ;DROP', mode="bm25")  # must not raise


def test_rrf_prefers_items_ranked_well_by_both():
    scores = rrf([[1, 2, 3], [3, 1, 4]])
    assert max(scores, key=lambda cid: scores[cid]) == 1
    assert scores[3] > scores[2]


def test_reindex_skips_unchanged_updates_changed_and_drops_deleted(env):
    corpus, conn, embedder = env["corpus"], env["conn"], env["embedder"]
    again = store.build_index(conn, corpus, embedder)
    assert again.files_unchanged == 3 and again.files_indexed == 0

    (corpus / "notes/network.md").write_text("# Network\n\nNow there is a VPN called wiregate.\n")
    (corpus / "runbooks/backup.md").unlink()
    changed = store.build_index(conn, corpus, embedder)
    assert changed.files_indexed == 1 and changed.files_removed == 1
    assert env["searcher"].search("wiregate", k=1, mode="bm25")[0].path == "notes/network.md"
    assert all(h.path != "runbooks/backup.md" for h in env["searcher"].search("snapshot", k=5))


def test_changing_embedding_model_requires_rebuild(env, monkeypatch):
    env["embedder"].fake = False
    with pytest.raises(RuntimeError, match="rebuild"):
        store.build_index(env["conn"], env["corpus"], env["embedder"])


def test_excludes(env):
    paths = [p.relative_to(env["corpus"]).as_posix() for p in store.iter_docs(env["corpus"], ["notes/*"])]
    assert paths == ["runbooks/backup.md", "runbooks/gpu.md"]
