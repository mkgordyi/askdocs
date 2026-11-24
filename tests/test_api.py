"""End-to-end through HTTP with the real index. Needs `python -m askdocs.index` first."""

import pytest
from fastapi.testclient import TestClient

from askdocs.api import app
from askdocs.config import settings

pytestmark = pytest.mark.skipif(
    not (settings.index_dir / "embeddings.npy").exists(), reason="index is not built"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_index_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 800


def test_in_scope_question_is_answered_with_a_source_link(client):
    body = client.post("/ask", json={"question": "Is social scoring of citizens by AI allowed?"}).json()
    assert body["answered"] is True
    assert body["sources"][0]["ref"].startswith("Article 5")
    assert body["sources"][0]["url"].startswith("https://eur-lex.europa.eu/")


def test_out_of_scope_question_is_refused(client):
    body = client.post("/ask", json={"question": "How do I bake sourdough bread?"}).json()
    assert body["answered"] is False
    assert body["answer"] is None
    assert body["sources"] == []


def test_too_short_question_is_rejected(client):
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "AskDocs" in response.text
