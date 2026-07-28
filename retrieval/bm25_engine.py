import re
from typing import Any
from rank_bm25 import BM25Okapi


class BM25Engine:
    """Fast rank-bm25 in-memory keyword search indexer."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def index_chunks(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        corpus_tokens = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = list(self.bm25.get_scores(tokens))
        max_s = max(scores, default=0.0)

        # Fallback for micro-corpora (N <= 2) where rank_bm25 IDF evaluates to 0.0
        if max_s <= 0.0:
            scores = [
                float(sum(1 for t in tokens if t in set(self._tokenize(c["content"]))))
                for c in self.chunks
            ]
            max_s = max(scores, default=0.0)

        max_score = max_s if max_s > 0 else 1.0

        scored_chunks = [
            {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": round(float(score / max_score), 4)
            }
            for chunk, score in zip(self.chunks, scores)
            if score > 0
        ]

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
