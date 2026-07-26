from typing import List, Dict, Any
import numpy as np

class HybridFusion:
    """Combines Dense Vector and Sparse BM25 results using Reciprocal Rank Fusion (RRF) and MMR diversity."""

    @staticmethod
    def fuse_rrf(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion with alpha weighting.
        alpha = 1.0 -> Dense only
        alpha = 0.0 -> Sparse only
        """
        fused_scores: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks
        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            dense_rrf = alpha * (1.0 / (k + rank + 1))
            if cid not in fused_scores:
                fused_scores[cid] = {
                    "chunk_id": cid,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": dense_rrf,
                    "dense_rank": rank + 1,
                    "sparse_rank": None
                }
            else:
                fused_scores[cid]["rrf_score"] += dense_rrf
                fused_scores[cid]["dense_rank"] = rank + 1

        # Process Sparse Ranks
        for rank, item in enumerate(sparse_results):
            cid = item["chunk_id"]
            sparse_rrf = (1.0 - alpha) * (1.0 / (k + rank + 1))
            if cid not in fused_scores:
                fused_scores[cid] = {
                    "chunk_id": cid,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "rrf_score": sparse_rrf,
                    "dense_rank": None,
                    "sparse_rank": rank + 1
                }
            else:
                fused_scores[cid]["rrf_score"] += sparse_rrf
                fused_scores[cid]["sparse_rank"] = rank + 1

        fused_list = list(fused_scores.values())
        # Normalize RRF scores to 0..1 range
        if fused_list:
            max_s = max(x["rrf_score"] for x in fused_list)
            for x in fused_list:
                x["score"] = round(float(x["rrf_score"] / max_s), 4) if max_s > 0 else 0.0

        fused_list.sort(key=lambda x: x["score"], reverse=True)
        return fused_list

    @staticmethod
    def apply_mmr(
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance selection for context diversity."""
        if not candidates or len(candidates) <= top_n:
            return candidates[:top_n]

        # Precompute token sets for all candidates to avoid re-splitting strings
        for c in candidates:
            c["_words"] = set(c["content"].lower().split())

        selected = [candidates[0]]
        unselected = candidates[1:]

        def jaccard_sim(set1: set, set2: set) -> float:
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        while len(selected) < top_n and unselected:
            best_mmr = -1e9
            best_idx = 0

            for i, cand in enumerate(unselected):
                relevance = cand["score"]
                max_redundancy = max(jaccard_sim(cand["_words"], sel["_words"]) for sel in selected)
                mmr_score = (mmr_lambda * relevance) - ((1.0 - mmr_lambda) * max_redundancy)
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            selected.append(unselected.pop(best_idx))

        # Cleanup temporary word set keys
        for c in candidates:
            c.pop("_words", None)

        return selected
