from askdocs.corpus import MAX_CHARS


def test_every_article_and_annex_is_present(chunks):
    articles = {c.article for c in chunks if c.article.startswith("art_")}
    annexes = {c.article for c in chunks if c.article.startswith("anx_")}
    assert len(articles) == 113
    assert len(annexes) == 13


def test_chunk_ids_are_unique_and_sizes_bounded(chunks):
    assert len({c.id for c in chunks}) == len(chunks)
    assert max(len(c.text) for c in chunks) <= MAX_CHARS


def test_long_paragraph_is_split_by_point(chunks):
    refs = {c.ref for c in chunks}
    assert "Article 5(1)(a)" in refs
    assert "Article 5(1)(h)" in refs


def test_each_definition_is_its_own_chunk(chunks):
    first = next(c for c in chunks if c.id == "art_3.1")
    assert first.ref == "Article 3(1)"
    assert "AI system" in first.text
    assert "(2)" not in first.text


def test_point_chunk_keeps_the_sentence_that_introduces_it(chunks):
    chunk = next(c for c in chunks if c.id == "art_5.1.a")
    assert chunk.text.startswith("The following AI practices shall be prohibited")


def test_every_chunk_links_to_its_article(chunks):
    assert all(c.url.endswith(f"#{c.article}") for c in chunks)
