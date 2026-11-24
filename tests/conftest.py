import pytest

from askdocs.config import settings
from askdocs.corpus import Chunk, parse
from askdocs.index import Hit


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    return parse(settings.source_file.read_text(encoding="utf-8"))


def make_hit(ref: str = "Article 5(1)(a)", text: str = "Some rule.", similarity: float = 0.8) -> Hit:
    chunk = Chunk(id=ref, article="art_5", ref=ref, title="Prohibited AI practices", text=text, url="https://example.test")
    return Hit(chunk=chunk, score=1.0, similarity=similarity)
