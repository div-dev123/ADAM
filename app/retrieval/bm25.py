"""Self-contained Okapi BM25 implementation for lexical retrieval in ADAM."""

import math
import re
from collections import Counter, defaultdict
from typing import Optional, Sequence


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    if not text:
        return []
    return [term.lower() for term in re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text)]


class BM25Index:
    """In-memory Okapi BM25 index over a collection of text documents."""

    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_lengths: list[int] = []
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}

        if self.doc_count > 0:
            self._build_index(documents)
            self.avg_doc_length = (
                sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 1.0
            )
            self._compute_idfs()
        else:
            self.avg_doc_length = 0.0

    def _build_index(self, documents: Sequence[str]) -> None:
        for doc in documents:
            tokens = tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)

            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] += 1

    def _compute_idfs(self) -> None:
        """Compute Robertson-Spärck Jones IDF with smoothing for all terms."""
        n = self.doc_count
        for term, df in self.doc_freqs.items():
            # Standard smoothed Okapi IDF
            idf_val = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, idf_val)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a specific document against a query."""
        if doc_idx < 0 or doc_idx >= self.doc_count:
            return 0.0

        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0

        doc_tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        avg_dl = max(self.avg_doc_length, 1.0)
        len_norm = 1.0 - self.b + self.b * (doc_len / avg_dl)

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf or term not in self.idf:
                continue

            tf = doc_tf[term]
            idf = self.idf[term]
            term_score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
            score += term_score

        return score

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[float, int]]:
        """Search the corpus and return list of (score, doc_idx) ranked descending."""
        if self.doc_count == 0:
            return []

        scores: list[tuple[float, int]] = []
        for idx in range(self.doc_count):
            s = self.score(query, idx)
            scores.append((s, idx))

        scores.sort(key=lambda item: item[0], reverse=True)

        if top_k is not None and top_k > 0:
            return scores[:top_k]
        return scores
