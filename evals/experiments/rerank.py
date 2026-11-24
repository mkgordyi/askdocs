"""Experiment, not part of the service: does a reranking model help?

    python -m evals.experiments.rerank

Takes the top 20 hybrid results, re-scores each (question, passage) pair with a
small cross-encoder and keeps the best 5. Result on the control questions is in
the README. It finds the right article more often, but costs about 1.7 s per
question on one CPU, so it is not used in the service yet.
"""

import json
import time
from pathlib import Path

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

from askdocs.config import settings
from askdocs.index import load

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    index = load()
    reranker = TextCrossEncoder(MODEL, cache_dir=str(settings.model_cache))
    rows = [json.loads(line) for line in open(Path(__file__).parent.parent / "golden.jsonl", encoding="utf-8")]
    answerable = [r for r in rows if r["expected"]]

    hit1 = hit5 = rr = 0.0
    started = time.perf_counter()
    for row in answerable:
        hits = index.search(row["question"], k=20)
        scores = list(reranker.rerank(row["question"], [h.chunk.search_text() for h in hits]))
        articles = [hits[i].chunk.article for i in np.argsort(-np.array(scores))[:5]]
        rank = next((i for i, a in enumerate(articles, 1) if a in row["expected"]), None)
        hit1 += rank == 1
        hit5 += rank is not None
        rr += 1 / rank if rank else 0
    n = len(answerable)
    per_question = (time.perf_counter() - started) / n * 1000
    print(f"hybrid + rerank: hit@1 {hit1 / n:.2f}  hit@5 {hit5 / n:.2f}  MRR {rr / n:.2f}  {per_question:.0f} ms per question")


if __name__ == "__main__":
    main()
