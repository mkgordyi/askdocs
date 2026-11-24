from askdocs.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("What is the Risk of AI?") == ["risk", "ai"]


def test_document_with_the_query_terms_scores_highest():
    bm25 = BM25(["fines for prohibited practices", "human oversight measures", "data governance"])
    scores = bm25.scores("prohibited practices fines")
    assert max(scores, key=scores.get) == 0


def test_rare_term_weighs_more_than_common_term():
    bm25 = BM25(["system sandbox", "system risk", "system data"])
    scores = bm25.scores("system sandbox")
    assert scores[0] > scores[1]


def test_unknown_terms_match_nothing():
    assert BM25(["human oversight"]).scores("zebra") == {}
