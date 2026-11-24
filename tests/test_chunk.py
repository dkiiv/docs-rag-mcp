from docsrag.chunk import split_markdown


def test_sections_carry_heading_path_and_lines():
    text = "# Title\n\nintro line\n\n## A\n\nalpha body\n\n### A1\n\ndeep body\n\n## B\n\nbeta body\n"
    chunks = split_markdown(text, "x.md")
    labels = [c.label for c in chunks]
    assert labels == ["Title", "Title > A", "Title > A > A1", "Title > B"]
    a1 = chunks[2]
    assert a1.text == "deep body"
    assert text.splitlines()[a1.start_line - 1] == "deep body"


def test_front_matter_title_wins_and_is_not_indexed():
    text = "---\ntitle: \"Real title\"\nslug: x\n---\n\n# Other\n\nbody\n"
    chunks = split_markdown(text, "x.md")
    assert chunks[0].title == "Real title"
    assert all("slug:" not in c.text for c in chunks)


def test_hash_inside_code_fence_is_not_a_heading():
    text = "# T\n\n## Run\n\n```bash\n# comment, not a heading\necho hi\n```\n"
    chunks = split_markdown(text, "x.md")
    assert len(chunks) == 1
    assert "# comment, not a heading" in chunks[0].text
    assert chunks[0].heading == "Run"


def test_long_sections_split_on_paragraphs_with_correct_ranges():
    paragraphs = [f"paragraph {i} " + "word " * 60 for i in range(12)]
    text = "# T\n\n" + "\n\n".join(paragraphs) + "\n"
    chunks = split_markdown(text, "x.md", max_chars=800)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 * 1.5 + 400 for c in chunks)
    lines = text.splitlines()
    for chunk in chunks:
        assert lines[chunk.start_line - 1].strip() == chunk.text.splitlines()[0].strip()
        assert lines[chunk.end_line - 1].strip() == chunk.text.splitlines()[-1].strip()
    joined = " ".join(c.text for c in chunks)
    assert all(f"paragraph {i} " in joined for i in range(12))


def test_heading_only_sections_produce_no_empty_chunks():
    chunks = split_markdown("# T\n\n## Empty\n\n## Full\n\nbody\n", "x.md")
    assert [c.heading for c in chunks] == ["Full"]


def test_title_falls_back_to_filename():
    chunks = split_markdown("just text\n", "docs/my-runbook_notes.md")
    assert chunks[0].title == "my runbook notes"
