import pytest

from docsrag import access, answer


def test_read_indexed_lines(env):
    hit = env["searcher"].search("restore a snapshot", k=1)[0]
    out = access.read_lines(env["conn"], hit.path, hit.start_line, hit.end_line)
    assert "restore-snapshot --latest" in out["text"]


@pytest.mark.parametrize("bad", ["../../etc/passwd", "/etc/passwd", "runbooks/../../index.sqlite",
                                 "secret.txt", "runbooks/missing.md", "", "runbooks/\x00gpu.md"])
def test_read_refuses_anything_not_indexed(env, bad):
    with pytest.raises(access.AccessError):
        access.read_lines(env["conn"], bad)


def test_read_is_capped(env, monkeypatch):
    monkeypatch.setattr(access, "MAX_READ_LINES", 2)
    out = access.read_lines(env["conn"], "runbooks/gpu.md")
    assert out["truncated"] is True and out["lines"] == "1-2"


def test_recent_changes_lists_only_indexed_docs(git_corpus):
    result = access.recent_changes(git_corpus["conn"], days=1)
    assert result["git"] is True
    docs = sorted(d for c in result["commits"] for d in c["docs"])
    assert docs == ["notes/network.md", "runbooks/backup.md", "runbooks/gpu.md"]
    assert "secret.txt" not in str(result)


def test_recent_changes_without_git(env):
    assert access.recent_changes(env["conn"])["git"] is False


def test_ask_parses_citations_and_not_found(env, monkeypatch):
    replies = iter(["Stop the service, then run `restore-snapshot --latest` [1].", answer.NOT_FOUND])

    def fake_chat(settings, messages, **kwargs):
        assert "[1] runbooks/backup.md" in messages[1]["content"]
        return {"content": next(replies), "finish_reason": "stop", "usage": {}, "timings": {}, "wall_s": 0.01}

    monkeypatch.setattr(answer, "chat", fake_chat)
    first = answer.ask(env["settings"], env["searcher"], "how do I restore a snapshot", k=3)
    assert first["not_found"] is False
    assert [c["path"] for c in first["cited"]] == ["runbooks/backup.md"]
    second = answer.ask(env["settings"], env["searcher"], "how do I restore a snapshot", k=3)
    assert second["not_found"] is True and second["cited"] == []


def test_empty_reply_is_reported_not_hidden(env, monkeypatch):
    monkeypatch.setattr(answer, "chat", lambda *a, **k: {"content": "", "finish_reason": "length",
                                                        "usage": {"completion_tokens": 800}, "timings": {},
                                                        "wall_s": 0.01})
    result = answer.ask(env["settings"], env["searcher"], "how do I restore a snapshot")
    assert result["empty"] is True and result["not_found"] is False
    assert "finish_reason=length" in result["answer"]


def test_thinking_off_is_sent_to_the_server(monkeypatch):
    from docsrag import clients, config

    sent = {}

    def fake_post(url, payload, **kwargs):
        sent.update(payload)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(clients, "post_json", fake_post)
    monkeypatch.setenv("DOCSRAG_LLM_THINKING", "off")
    clients.chat(config.load(), [{"role": "user", "content": "hi"}])
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    sent.clear()
    monkeypatch.setenv("DOCSRAG_LLM_THINKING", "default")
    clients.chat(config.load(), [{"role": "user", "content": "hi"}])
    assert "chat_template_kwargs" not in sent
