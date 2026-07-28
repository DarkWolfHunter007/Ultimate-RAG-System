import os
import sys

# Ensure project root is in sys.path when script is executed directly from inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import asyncio
import logging
from typing import Any
from core.router import ModelRouter
from ingestion.vector_indexer import VectorIndexer

logger = logging.getLogger(__name__)


async def generate_synthetic_dataset(num_samples: int = 5, output_file: str = "./data/eval_dataset.json") -> list[dict[str, Any]]:
    """
    Generates synthetic Q&A evaluation dataset from document chunks (Prajwal BM method).
    Uses ModelRouter to generate ground-truth questions and keywords per chunk.
    """
    indexer = VectorIndexer()
    all_chunks = indexer.get_all_chunks()
    if not all_chunks:
        logger.warning("No document chunks found in vector indexer to generate synthetic dataset.")
        return []

    # Sample chunks across corpus
    step = max(1, len(all_chunks) // min(num_samples, len(all_chunks)))
    sampled_chunks = all_chunks[::step][:num_samples]

    router = ModelRouter()
    dataset = []

    for chunk in sampled_chunks:
        content = chunk.get("content", "")
        chunk_id = chunk.get("chunk_id", "")
        parent_id = chunk.get("metadata", {}).get("parent_id", chunk_id)

        prompt = (
            f"Read the following document text and generate 1 specific question that can ONLY be answered by this text, "
            f"along with 3 distinct keyword terms found in the text.\n\n"
            f"Document Text:\n{content[:1000]}\n\n"
            f"Output format JSON (return ONLY valid JSON):\n"
            f'{{"query": "your question here", "expected_keywords": ["term1", "term2", "term3"]}}'
        )

        try:
            resp = await router.generate_completion(prompt=prompt, temperature=0.3)
            raw = resp.get("content", "").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                item = json.loads(raw[start:end+1])
                item["expected_parent_id"] = parent_id
                item["chunk_id"] = chunk_id
                dataset.append(item)
        except Exception as e:
            logger.warning(f"Synthetic generation failed for chunk {chunk_id}: {e}")

    if dataset:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"[SUCCESS] Generated {len(dataset)} synthetic eval test case(s) -> saved to '{output_file}'")

    return dataset


if __name__ == "__main__":
    asyncio.run(generate_synthetic_dataset())
