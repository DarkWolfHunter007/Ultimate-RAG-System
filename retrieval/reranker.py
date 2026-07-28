import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level lazy cache for the CrossEncoder model (None = not loaded yet, False = load failed)
_rerank_model = None


def rerank(query: str, candidates: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    """Neural cross-encoder reranker. Falls back to truncation if model unavailable."""
    global _rerank_model
    if not candidates:
        return []

    if _rerank_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _rerank_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)
        except Exception as e:
            logger.warning(f"Could not load neural CrossEncoder: {e}. Using truncation fallback.")
            _rerank_model = False

    if _rerank_model:
        try:
            pairs = [[query, c["content"]] for c in candidates]
            scores = _rerank_model.predict(pairs)
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates[:top_n]
        except Exception as e:
            logger.warning(f"Reranking execution error: {e}")

    return candidates[:top_n]
