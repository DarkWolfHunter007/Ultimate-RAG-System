from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RerankerEngine:
    """Neural cross-encoder reranker bridge."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)
            except Exception as e:
                logger.warning(f"Could not load neural CrossEncoder: {e}. Using light lexical scoring fallback.")
                self._model = False

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        self._load_model()
        if self._model:
            try:
                pairs = [[query, c["content"]] for c in candidates]
                scores = self._model.predict(pairs)
                for i, score in enumerate(scores):
                    candidates[i]["rerank_score"] = float(score)
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return candidates[:top_n]
            except Exception as e:
                logger.warning(f"Reranking execution error: {e}")

        # Lightweight Fallback: boost chunks containing key query words
        words = set(query.lower().split())
        for c in candidates:
            c_words = set(c["content"].lower().split())
            match_ratio = len(words & c_words) / max(1, len(words))
            c["rerank_score"] = c.get("score", 0.5) + (0.2 * match_ratio)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]
