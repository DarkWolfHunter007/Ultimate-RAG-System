from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any

class BM25Engine:
    """Fast rank-bm25 in-memory keyword search indexer."""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return

        corpus_tokens = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair scores with chunks
        scored_chunks = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        for i, score in enumerate(scores):
            if score > 0:
                chunk = self.chunks[i]
                norm_score = round(float(score / max_score), 4)
                scored_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                    "score": norm_score
                })

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
