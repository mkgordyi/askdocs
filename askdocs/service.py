"""The whole answer flow in one place: search, decide, answer, log."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

from .config import Settings, settings as default_settings
from .generate import Generator
from .index import Hit, Index

log = logging.getLogger("askdocs")


@dataclass(frozen=True)
class Answer:
    answered: bool
    answer: str | None
    sources: list[Hit]
    mode: str
    confidence: float
    reason: str | None = None
    retrieved: list[Hit] = field(default_factory=list)


class AskService:
    def __init__(self, index: Index, generator: Generator, settings: Settings = default_settings):
        self.index = index
        self.generator = generator
        self.settings = settings

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        hits = self.index.search(question, k=self.settings.top_k)
        confidence = max((h.similarity for h in hits), default=0.0)

        tokens_in = tokens_out = 0
        if confidence < self.settings.min_confidence:
            # Nothing in the documents is close enough to the question: do not guess.
            answer = Answer(False, None, [], self.generator.mode, confidence, "low_confidence", hits)
        else:
            generated = self.generator.generate(question, hits)
            tokens_in, tokens_out = generated.tokens_in, generated.tokens_out
            answer = Answer(
                generated.text is not None,
                generated.text,
                generated.cited,
                self.generator.mode,
                confidence,
                generated.reason,
                hits,
            )
        self._log(question, answer, started, tokens_in, tokens_out)
        return answer

    def _log(self, question: str, answer: Answer, started: float, tokens_in: int, tokens_out: int) -> None:
        cost = (tokens_in * self.settings.price_in_per_1m + tokens_out * self.settings.price_out_per_1m) / 1e6
        log.info(
            json.dumps(
                {
                    "event": "ask",
                    # The question itself is not logged: only its size and a short fingerprint.
                    "question_chars": len(question),
                    "question_id": hashlib.sha256(question.encode()).hexdigest()[:12],
                    "answered": answer.answered,
                    "reason": answer.reason,
                    "mode": answer.mode,
                    "confidence": round(answer.confidence, 4),
                    "sources": [h.chunk.ref for h in answer.sources],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": round(cost, 6),
                }
            )
        )
