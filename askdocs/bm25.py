"""Small BM25 keyword search. No dependencies, so it is easy to read and to test."""

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or shall that the "
    "this to under what when which who whom with must can may any its their".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings.setdefault(term, []).append((doc_id, freq))
        self.n = len(documents)
        self.avg_len = sum(self.doc_len) / max(self.n, 1)

    def scores(self, query: str) -> dict[int, float]:
        """Score of every document that shares at least one term with the query."""
        result: dict[int, float] = {}
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (self.n - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, freq in postings:
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                result[doc_id] = result.get(doc_id, 0.0) + idf * freq * (self.k1 + 1) / norm
        return result
