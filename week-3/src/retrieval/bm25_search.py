"""BM25 sparse keyword retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List


class BM25Index:
    """
    Lightweight BM25 implementation (Okapi BM25).

    No external dependency — pure Python so the project stays simple.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict] = []         # original chunk dicts
        self.token_freqs: List[Counter] = []  # per-doc token counts
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.n_docs: int = 0

    # ── tokeniser (simple whitespace + lowering) ─────────────
    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    # ── indexing ─────────────────────────────────────────────
    def index(self, chunks: List[Dict]) -> None:
        """Build the BM25 index from a list of chunk dicts."""
        self.corpus = chunks
        self.n_docs = len(chunks)

        self.token_freqs = []
        self.doc_lengths = []

        for chunk in chunks:
            tokens = self._tokenise(chunk["text"])
            self.token_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))

        self.avg_dl = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 1.0
        )
        self._compute_idf()

    def _compute_idf(self) -> None:
        """Inverse Document Frequency for every term in the corpus."""
        df: Counter = Counter()
        for freq in self.token_freqs:
            for token in freq:
                df[token] += 1

        self.idf = {}
        for token, doc_freq in df.items():
            self.idf[token] = math.log(
                (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )

    # ── scoring ──────────────────────────────────────────────
    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_lengths[doc_idx]
        freq = self.token_freqs[doc_idx]

        for qt in query_tokens:
            if qt not in self.idf:
                continue
            tf = freq.get(qt, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * dl / self.avg_dl
            )
            score += self.idf[qt] * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return the top-k chunks ranked by BM25 score.

        Each result dict has the same keys as the input chunk
        plus a ``bm25_score`` field.
        """
        query_tokens = self._tokenise(query)
        scored = []

        for idx in range(self.n_docs):
            s = self._score_doc(query_tokens, idx)
            if s > 0:
                scored.append((idx, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict] = []
        for idx, bm25_score in scored[:top_k]:
            result = {**self.corpus[idx], "bm25_score": bm25_score}
            results.append(result)

        return results
