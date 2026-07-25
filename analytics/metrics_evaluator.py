from typing import List, Dict, Any

class MetricsEvaluator:
    """Calculates quantitative RAG evaluation metrics (Faithfulness, Precision, Recall)."""

    @staticmethod
    def evaluate(query: str, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, float]:
        if not contexts or not response:
            return {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

        response_words = set(response.lower().split())
        query_words = set(query.lower().split())

        # 1. Faithfulness Score: Proportion of response key terms grounded in contexts
        all_context_text = " ".join([c["content"] for c in contexts]).lower()
        all_context_words = set(all_context_text.split())
        
        grounded_count = sum(1 for word in response_words if len(word) > 3 and word in all_context_words)
        total_eval_words = sum(1 for word in response_words if len(word) > 3)
        faithfulness = round(grounded_count / max(1, total_eval_words), 2)

        # 2. Context Precision @ N: Relevance of top-retrieved contexts to query
        relevant_chunks = 0
        for c in contexts:
            c_words = set(c["content"].lower().split())
            if len(query_words & c_words) > 0:
                relevant_chunks += 1
        context_precision = round(relevant_chunks / max(1, len(contexts)), 2)

        # 3. Context Recall: Coverage of query terms across retrieved contexts
        context_query_hits = sum(1 for q_word in query_words if q_word in all_context_words)
        context_recall = round(context_query_hits / max(1, len(query_words)), 2)

        return {
            "faithfulness": min(1.0, max(0.0, faithfulness)),
            "context_precision": min(1.0, max(0.0, context_precision)),
            "context_recall": min(1.0, max(0.0, context_recall))
        }
