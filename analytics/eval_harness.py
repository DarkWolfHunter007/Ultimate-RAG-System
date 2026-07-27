import os
import json
import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_EVAL_DATASET = [
    {
        "query": "What is the primary function of the semantic chunker?",
        "expected_keywords": ["sentence", "heading", "parent-child", "chunker"]
    },
    {
        "query": "How are DOCX tables preserved during ingestion?",
        "expected_keywords": ["table", "markdown", "caption"]
    },
    {
        "query": "What baseline is used for PDF heading detection?",
        "expected_keywords": ["modal", "font size", "baseline"]
    }
]

def run_retrieval_eval(vector_indexer, bm25_engine, eval_cases: List[Dict[str, Any]] = None, top_k: int = 5) -> Dict[str, float]:
    """Lightweight RAG retrieval evaluation harness measuring Hit Rate @ K and MRR."""
    if eval_cases is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                eval_cases = json.load(f)
        else:
            eval_cases = DEFAULT_EVAL_DATASET

    hits = 0
    reciprocal_ranks = []

    for item in eval_cases:
        query = item["query"]
        expected_kw = [kw.lower() for kw in item.get("expected_keywords", [])]

        vec_res = vector_indexer.search_vector(query_text=query, top_k=top_k * 2)
        sparse_res = bm25_engine.search(query=query, top_k=top_k * 2)

        # Combine results
        combined = vec_res + sparse_res
        hit_found = False

        for rank, c in enumerate(combined[:top_k], 1):
            meta = c.get("metadata", {}) or {}
            # Evaluate parent content (small-to-big) or child content
            resolved_text = (meta.get("parent_content") or c.get("content", "")).lower()
            parent_id = meta.get("parent_id", c.get("chunk_id", ""))
            
            target_pid = item.get("expected_parent_id")
            if target_pid and parent_id == target_pid:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                hit_found = True
                break
            elif expected_kw and any(kw in resolved_text for kw in expected_kw):
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                hit_found = True
                break

        if not hit_found:
            reciprocal_ranks.append(0.0)

    total = max(1, len(eval_cases))
    hit_rate = round(hits / total, 4)
    mrr = round(sum(reciprocal_ranks) / total, 4)

    results = {
        "total_test_cases": len(eval_cases),
        "hit_rate_at_k": hit_rate,
        "mrr": mrr
    }

    print(f"\n[RAG EVALUATION HARNESS BENCHMARK]")
    print(f"- Test Cases: {results['total_test_cases']}")
    print(f"- Hit Rate @ {top_k}: {results['hit_rate_at_k'] * 100:.1f}%")
    print(f"- MRR @ {top_k}: {results['mrr']:.4f}\n")

    return results

if __name__ == "__main__":
    from ingestion.vector_indexer import VectorIndexer
    from retrieval.bm25_engine import BM25Engine

    vi = VectorIndexer()
    bm = BM25Engine()
    bm.index_chunks(vi.get_all_chunks())
    run_retrieval_eval(vi, bm)
