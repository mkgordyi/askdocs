import json
import logging
from dataclasses import replace

from askdocs.config import Settings
from askdocs.generate import ExtractiveGenerator
from askdocs.service import AskService
from tests.conftest import make_hit


class FakeIndex:
    def __init__(self, hits):
        self.hits = hits
        self.chunks = [h.chunk for h in hits]

    def search(self, question, k=5, mode="hybrid"):
        return self.hits


SETTINGS = replace(Settings(), min_confidence=0.6)


def service(similarity: float) -> AskService:
    return AskService(FakeIndex([make_hit(similarity=similarity)]), ExtractiveGenerator(), SETTINGS)


def test_question_far_from_the_documents_is_refused_without_sources():
    answer = service(0.59).ask("What is the capital of France?")
    assert answer.answered is False
    assert answer.answer is None
    assert answer.sources == []
    assert answer.reason == "low_confidence"


def test_question_close_to_the_documents_is_answered_with_a_source():
    answer = service(0.60).ask("What is prohibited?")
    assert answer.answered is True
    assert [h.chunk.ref for h in answer.sources] == ["Article 5(1)(a)"]


def test_empty_search_result_is_refused():
    answer = AskService(FakeIndex([]), ExtractiveGenerator(), SETTINGS).ask("anything")
    assert answer.answered is False


def test_log_line_has_metrics_but_not_the_question(caplog):
    with caplog.at_level(logging.INFO, logger="askdocs"):
        service(0.9).ask("a very private question")
    record = json.loads(caplog.records[-1].message)
    assert record["event"] == "ask"
    assert record["answered"] is True
    assert record["question_chars"] == len("a very private question")
    assert "private" not in caplog.records[-1].message
    assert {"latency_ms", "confidence", "tokens_in", "cost_usd"} <= record.keys()
