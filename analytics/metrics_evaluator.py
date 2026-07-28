from typing import Any


def evaluate_metrics(query: str, response: str, contexts: list[dict[str, Any]]) -> dict[str, float]:
    """Calculates quantitative RAG evaluation metrics (Faithfulness, Precision, Recall)."""
    if not contexts or not response:
        return {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    response_words = set(response.lower().split())
    query_words = set(query.lower().split())
    all_context_text = " ".join([c.get("content", "") for c in contexts]).lower()
    all_context_words = set(all_context_text.split())

    grounded_count = sum(1 for word in response_words if len(word) > 3 and word in all_context_words)
    total_eval_words = sum(1 for word in response_words if len(word) > 3)
    faithfulness = round(grounded_count / max(1, total_eval_words), 2)

    relevant_chunks = sum(1 for c in contexts if len(query_words & set(c.get("content", "").lower().split())) > 0)
    context_precision = round(relevant_chunks / max(1, len(contexts)), 2)

    context_query_hits = sum(1 for q_word in query_words if q_word in all_context_words)
    context_recall = round(context_query_hits / max(1, len(query_words)), 2)

    return {
        "faithfulness": min(1.0, max(0.0, faithfulness)),
        "context_precision": min(1.0, max(0.0, context_precision)),
        "context_recall": min(1.0, max(0.0, context_recall))
    }
