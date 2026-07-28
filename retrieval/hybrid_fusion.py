from typing import Any


def fuse_rrf(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    alpha: float = 0.5,
    k: int = 60
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) with hybrid score weighting.
    alpha = 1.0 -> Dense (vector) primary
    alpha = 0.0 -> Sparse (BM25) primary
    """
    fused_scores: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        dense_rrf = alpha * (1.0 / (k + rank + 1))
        # Boost by direct vector similarity score if available
        raw_score_boost = float(item.get("score", 0.5)) * 0.1
        total_rrf = dense_rrf + raw_score_boost

        if cid not in fused_scores:
            fused_scores[cid] = {
                "chunk_id": cid,
                "content": item["content"],
                "metadata": item["metadata"],
                "rrf_score": total_rrf,
                "dense_rank": rank + 1,
                "sparse_rank": None
            }
        else:
            fused_scores[cid]["rrf_score"] += total_rrf
            fused_scores[cid]["dense_rank"] = rank + 1

    for rank, item in enumerate(sparse_results):
        cid = item["chunk_id"]
        sparse_rrf = (1.0 - alpha) * (1.0 / (k + rank + 1))
        raw_score_boost = float(item.get("score", 0.5)) * 0.05
        total_rrf = sparse_rrf + raw_score_boost

        if cid not in fused_scores:
            fused_scores[cid] = {
                "chunk_id": cid,
                "content": item["content"],
                "metadata": item["metadata"],
                "rrf_score": total_rrf,
                "dense_rank": None,
                "sparse_rank": rank + 1
            }
        else:
            fused_scores[cid]["rrf_score"] += total_rrf
            fused_scores[cid]["sparse_rank"] = rank + 1

    fused_list = list(fused_scores.values())
    if fused_list:
        max_s = max(x["rrf_score"] for x in fused_list)
        for x in fused_list:
            x["score"] = round(float(x["rrf_score"] / max_s), 4) if max_s > 0 else 0.0

    fused_list.sort(key=lambda x: x["score"], reverse=True)
    return fused_list


def _semantic_overlap(text1: str, text2: str) -> float:
    """Combines word token overlap and character 3-gram subword overlap for robust similarity."""
    t1, t2 = text1.lower(), text2.lower()
    words1, words2 = set(t1.split()), set(t2.split())
    if not words1 or not words2:
        return 0.0

    word_sim = len(words1 & words2) / len(words1 | words2)

    grams1 = set(t1[i:i+3] for i in range(len(t1) - 2))
    grams2 = set(t2[i:i+3] for i in range(len(t2) - 2))
    gram_sim = len(grams1 & grams2) / len(grams1 | grams2) if (grams1 and grams2) else 0.0

    return 0.5 * word_sim + 0.5 * gram_sim


def apply_mmr(
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    mmr_lambda: float = 0.5
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance selection balancing query relevance and context diversity."""
    if not candidates or len(candidates) <= top_n:
        return candidates[:top_n]

    selected = [candidates[0]]
    unselected = candidates[1:]

    while len(selected) < top_n and unselected:
        best_mmr = -1e9
        best_idx = 0
        for i, cand in enumerate(unselected):
            relevance = cand["score"]
            max_redundancy = max(_semantic_overlap(cand["content"], sel["content"]) for sel in selected)
            mmr_score = (mmr_lambda * relevance) - ((1.0 - mmr_lambda) * max_redundancy)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        selected.append(unselected.pop(best_idx))

    return selected
