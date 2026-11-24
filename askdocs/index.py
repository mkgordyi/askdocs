"""Search index: keyword search (BM25) and meaning search (embeddings), fused together.

Build once with `python -m askdocs.index`. The service only loads the result.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .config import settings
from .corpus import Chunk, parse

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
CANDIDATES = 50


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float       # fused rank score, used for ordering
    similarity: float  # cosine similarity of meaning, used for the refusal rule


class Embedder:
    """Local embedding model. No API calls, no key."""

    def __init__(self, model_name: str = settings.embedding_model, cache_dir: Path = settings.model_cache):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts, batch_size=32)), dtype=np.float32))

    def query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed([QUERY_PREFIX + text])), dtype=np.float32))[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25([c.search_text() for c in chunks])

    def search(self, question: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        similarities = self.vectors @ self.embedder.query(question)
        dense_order = np.argsort(-similarities)[:CANDIDATES].tolist()
        keyword_scores = self.bm25.scores(question)
        keyword_order = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:CANDIDATES]

        if mode == "dense":
            ranked = [(i, float(similarities[i])) for i in dense_order]
        elif mode == "bm25":
            ranked = [(i, keyword_scores[i]) for i in keyword_order]
        else:  # reciprocal rank fusion: a chunk is good if either search ranks it high
            fused: dict[int, float] = {}
            for order in (dense_order, keyword_order):
                for rank, i in enumerate(order):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [Hit(self.chunks[i], float(score), float(similarities[i])) for i, score in ranked[:k]]


def build(index_dir: Path = settings.index_dir) -> dict:
    """Parse the source, embed every chunk and write the index files."""
    source = settings.source_file.read_bytes()
    chunks = parse(source.decode("utf-8"))
    vectors = Embedder().passages([c.search_text() for c in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", vectors)
    meta = {
        "chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(index_dir: Path = settings.index_dir) -> Index:
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    vectors = np.load(index_dir / "embeddings.npy")
    if len(chunks) != len(vectors):
        raise RuntimeError("Index files do not match. Rebuild with `python -m askdocs.index`.")
    return Index(chunks, vectors, Embedder())


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
