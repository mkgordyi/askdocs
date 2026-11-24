from askdocs.generate import NOT_FOUND, ExtractiveGenerator, LLMGenerator
from tests.conftest import make_hit


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.seen_user = ""

    def complete(self, system: str, user: str):
        self.seen_user = user
        return self.reply, 120, 30


HITS = [make_hit("Article 5(1)(a)"), make_hit("Article 99(3)")]


def test_answer_with_valid_citation_is_accepted():
    result = LLMGenerator(FakeClient("Social scoring is prohibited [1].")).generate("q", HITS)
    assert result.text == "Social scoring is prohibited [1]."
    assert [h.chunk.ref for h in result.cited] == ["Article 5(1)(a)"]
    assert (result.tokens_in, result.tokens_out) == (120, 30)


def test_model_saying_not_found_gives_no_answer():
    result = LLMGenerator(FakeClient(NOT_FOUND)).generate("q", HITS)
    assert result.text is None
    assert result.reason == "not_in_passages"


def test_answer_without_citation_is_rejected():
    result = LLMGenerator(FakeClient("Social scoring is prohibited.")).generate("q", HITS)
    assert result.text is None
    assert result.reason == "uncited"


def test_answer_citing_a_passage_that_was_not_given_is_rejected():
    result = LLMGenerator(FakeClient("It is prohibited [1] and fined [7].")).generate("q", HITS)
    assert result.text is None
    assert result.reason == "uncited"


def test_passages_are_sent_numbered_with_their_reference():
    client = FakeClient("ok [2]")
    LLMGenerator(client).generate("What is fined?", HITS)
    assert "[1] Article 5(1)(a):" in client.seen_user
    assert "[2] Article 99(3):" in client.seen_user
    assert "Question: What is fined?" in client.seen_user


def test_extractive_quotes_the_best_passage_and_lists_close_matches():
    hits = [
        make_hit("Article 5(1)(a)", text="Rule one. Rule two.", similarity=0.80),
        make_hit("Article 5(1)(b)", similarity=0.77),
        make_hit("Article 99(3)", similarity=0.60),
    ]
    result = ExtractiveGenerator().generate("q", hits)
    assert result.text == "Rule one. Rule two."
    assert [h.chunk.ref for h in result.cited] == ["Article 5(1)(a)", "Article 5(1)(b)"]
