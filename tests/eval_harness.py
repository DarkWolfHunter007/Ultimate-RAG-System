import os
import sys

# Ensure project root is in sys.path when script is executed directly from inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import asyncio
import logging
from typing import Any, Optional

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


def run_retrieval_eval(vector_indexer, bm25_engine, eval_cases: Optional[list[dict[str, Any]]] = None, top_k: int = 5) -> dict[str, float]:
    """Retrieval evaluation harness measuring Hit Rate @ K and MRR."""
    if eval_cases is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                eval_cases = json.load(f)
        else:
            eval_cases = DEFAULT_EVAL_DATASET

    if not eval_cases:
        print("\n[NOTICE] No evaluation test cases available.")
        return {"total_test_cases": 0, "hit_rate_at_k": 0.0, "mrr": 0.0}

    hits = 0
    reciprocal_ranks = []

    for item in eval_cases:
        query = item["query"]
        expected_kw = [kw.lower() for kw in item.get("expected_keywords", [])]

        vec_res = vector_indexer.search_vector(query_text=query, top_k=top_k * 2)
        sparse_res = bm25_engine.search(query=query, top_k=top_k * 2)

        combined = vec_res + sparse_res
        hit_found = False

        for rank, c in enumerate(combined[:top_k], 1):
            meta = c.get("metadata", {}) or {}
            resolved_text = (meta.get("parent_content") or c.get("content", "")).lower()
            parent_id = meta.get("parent_id", c.get("chunk_id", ""))

            target_pid = item.get("expected_parent_id")
            if (target_pid and parent_id == target_pid) or (expected_kw and any(kw in resolved_text for kw in expected_kw)):
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

    print("\n==========================================")
    print("[RAG RETRIEVAL EVALUATION BENCHMARK]")
    print(f"- Test Cases Evaluated: {results['total_test_cases']}")
    print(f"- Hit Rate @ {top_k}: {results['hit_rate_at_k'] * 100:.1f}%")
    print(f"- MRR @ {top_k}: {results['mrr']:.4f}")
    print("==========================================\n")

    return results


if __name__ == "__main__":
    from ingestion.vector_indexer import VectorIndexer
    from retrieval.bm25_engine import BM25Engine

    vi = VectorIndexer()
    bm = BM25Engine()

    if vi.count() == 0:
        print("\n==========================================")
        print("[NOTICE] ChromaDB Vector Database is empty (0 chunks indexed).")
        print("1. To run unit tests for code correctness: run 'python -m unittest discover -s tests'")
        print("2. To benchmark retrieval accuracy: upload your PDFs/DOCX documents via the UI first.")
        print("==========================================\n")
    else:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")
        if not os.path.exists(dataset_path):
            print("\n[INFO] Generating custom synthetic QA benchmark dataset from your uploaded documents...")
            try:
                from tests.dataset_generator import generate_synthetic_dataset
                asyncio.run(generate_synthetic_dataset(num_samples=5, output_file=dataset_path))
            except Exception as e:
                logger.warning(f"Auto-dataset generation note: {e}")

        bm.index_chunks(vi.get_all_chunks())
        run_retrieval_eval(vi, bm)
