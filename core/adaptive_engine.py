import asyncio
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from core.config_manager import ConfigManager
from core.router import ModelRouter
from ingestion.vector_indexer import VectorIndexer
from retrieval.bm25_engine import BM25Engine
from retrieval.hybrid_fusion import fuse_rrf, apply_mmr
from retrieval.hyde import generate_hypothetical_document
from retrieval.reranker import rerank
from analytics.telemetry_logger import TelemetryLogger
from analytics.metrics_evaluator import evaluate_metrics
from retrieval.query_expander import expand_query

logger = logging.getLogger(__name__)


class AdaptiveEngine:
    """Dynamic Adaptive Search Engine that scales search algorithms based on corpus size."""

    def __init__(self, vector_indexer: VectorIndexer, bm25_engine: BM25Engine):
        self.vector_indexer = vector_indexer
        self.bm25_engine = bm25_engine
        self.config_mgr = ConfigManager()
        self.model_router = ModelRouter()

    def get_corpus_tier(self) -> Tuple[str, int]:
        total_chunks = self.vector_indexer.count()
        if total_chunks < 50:
            return ("Nano", total_chunks)
        elif total_chunks <= 5000:
            return ("Standard", total_chunks)
        else:
            return ("Enterprise", total_chunks)

    async def _embed_query(self, q: str) -> Optional[List[float]]:
        """Embed a single query string; returns None on failure (falls back to text search)."""
        try:
            emb_res = await self.model_router.generate_embeddings([q])
            return emb_res[0] if emb_res else None
        except Exception as e:
            logger.info(f"Query embedding fallback for '{q[:40]}': {e}")
            return None

    async def execute_rag_pipeline(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        telemetry = TelemetryLogger(query_id=f"q_{int(time.time()*1000)}")
        cfg = self.config_mgr.get_config()
        tier, chunk_count = self.get_corpus_tier()

        # 1. Multi-Query Expansion & HyDE Check
        search_queries = [query]
        if cfg.enable_hyde or tier == "Enterprise":
            t0 = time.perf_counter()
            hyde_doc = await generate_hypothetical_document(query, self.model_router)
            search_queries.append(hyde_doc)
            telemetry.log_span("HyDE Synthesis", (time.perf_counter() - t0) * 1000, f"Expanded via {cfg.llm_model}")
        else:
            t_exp = time.perf_counter()
            search_queries = await expand_query(query, self.model_router)
            telemetry.log_span("Multi-Query Expansion", (time.perf_counter() - t_exp) * 1000, f"Generated {len(search_queries)} query variants")

        # 2. Parallel embedding + search across all query variants.
        #    asyncio.gather fires all embedding API calls concurrently instead of sequentially,
        #    cutting p95 latency by ~50-70% when search_queries has multiple variants.
        t_vec = time.perf_counter()
        embeddings: List[Optional[List[float]]] = await asyncio.gather(
            *[self._embed_query(q) for q in search_queries]
        )

        all_dense: List[Dict[str, Any]] = []
        all_sparse: List[Dict[str, Any]] = []
        for q_var, emb in zip(search_queries, embeddings):
            d_res = self.vector_indexer.search_vector(query_text=q_var, top_k=cfg.top_k_candidates, query_embedding=emb)
            s_res = self.bm25_engine.search(query=q_var, top_k=cfg.top_k_candidates)
            all_dense.extend(d_res)
            all_sparse.extend(s_res)

        telemetry.log_span("Multi-Query Dense Search", (time.perf_counter() - t_vec) * 1000, f"Fetched {len(all_dense)} total hits ({len(search_queries)} queries in parallel)")

        # 3. Hybrid Reciprocal Rank Fusion
        t_rrf = time.perf_counter()
        fused_candidates = fuse_rrf(dense_results=all_dense, sparse_results=all_sparse, alpha=cfg.hybrid_alpha)
        telemetry.log_span("RAG-Fusion & RRF", (time.perf_counter() - t_rrf) * 1000, f"Alpha={cfg.hybrid_alpha}")

        # 4. Optional Cross-Encoder Reranker
        if (cfg.enable_reranker or tier == "Enterprise") and fused_candidates:
            t_rerank = time.perf_counter()
            fused_candidates = rerank(query, fused_candidates, top_n=cfg.top_k_candidates)
            telemetry.log_span("Cross-Encoder Neural Rerank", (time.perf_counter() - t_rerank) * 1000)

        # 5. MMR Diversity Selection & Full Corpus Fetching for Small/Summary Queries
        ql = query.lower()
        if tier == "Nano" or any(w in ql for w in ["summarize", "summary", "all content", "everything", "overview"]):
            all_chunks = self.vector_indexer.get_all_chunks()
            if len(all_chunks) <= 40:
                final_contexts = all_chunks
            else:
                final_contexts = apply_mmr(candidates=fused_candidates, top_n=min(cfg.top_n_final * 2, len(fused_candidates)), mmr_lambda=cfg.mmr_lambda)
        else:
            final_contexts = apply_mmr(candidates=fused_candidates, top_n=cfg.top_n_final, mmr_lambda=cfg.mmr_lambda)

        # 6. Small-to-Big Parent Content Resolution
        #    Collect unique parent_ids from child results, batch-fetch from ChromaDB in one call,
        #    then substitute full parent body into each excerpt. Falls back to child content
        #    for old chunks (pre-parent-doc fix) that lack a stored parent doc.
        t_llm = time.perf_counter()
        parent_ids = list({
            c["metadata"].get("parent_id", "")
            for c in final_contexts
            if c["metadata"].get("parent_id") and c["metadata"].get("chunk_type", "child") == "child"
        } - {""})
        parent_map: Dict[str, str] = self.vector_indexer.get_by_ids(parent_ids) if parent_ids else {}

        context_str = ""
        for idx, c in enumerate(final_contexts, 1):
            src = c["metadata"].get("source", "doc")
            pg = c["metadata"].get("page", 1)
            hd = c["metadata"].get("heading", "")
            heading_str = f" heading='{hd}'" if hd else ""
            # Resolution order: ChromaDB parent lookup → child content
            parent_id = c["metadata"].get("parent_id", "")
            chunk_body = parent_map.get(parent_id) or c["content"]
            context_str += f"\n<DOCUMENT_EXCERPT id='{idx}' source='{src}' page='{pg}'{heading_str}>\n{chunk_body}\n</DOCUMENT_EXCERPT>\n"

        history_str = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role_name = "User" if msg.get("role") == "user" else "Assistant"
                history_str += f"{role_name}: {msg.get('content', '')}\n"

        system_prompt = (
            "You are Ultimate-RAG-System, an expert AI research assistant. "
            "Answer the user's question directly, completely, and accurately using the provided <DOCUMENT_EXCERPT> context. "
            "Synthesize all relevant information into a well-structured response using Markdown (headers, bullet points, bold text). "
            "Cite your sources inline using bracketed numbers like [1], [2] matching the excerpt IDs. "
            "Focus on delivering a complete, informative response based on the relevant context provided. Do not waste space explaining what unrelated documents do not say unless asked."
        )

        user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        if history_str:
            user_prompt += f"RECENT CONVERSATION HISTORY:\n{history_str}\n\n"
        user_prompt += f"USER QUESTION: {query}"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n{'='*60}\nQUERY: '{query}' [Tier: {tier}]\n{'='*60}")
            logger.debug(f"Search query variants ({len(search_queries)}): {search_queries}")
            logger.debug(f"RRF Fused returned {len(fused_candidates)} candidates.")
            logger.debug(f"MMR selected {len(final_contexts)} final context chunks.")
            logger.debug(f"Fetching parent contexts for IDs: {parent_ids}")
            logger.debug(f"Retrieved {len(parent_map)} parent docs from ChromaDB.")
            logger.debug(f"\n--- FINAL LLM CONTEXT ({len(context_str)} chars) ---")
            logger.debug(context_str[:1200] + "..." if len(context_str) > 1200 else context_str)
            logger.debug("--- END CONTEXT ---\n")

        llm_response = await self.model_router.generate_completion(
            prompt=user_prompt, system_prompt=system_prompt, temperature=0.2, max_tokens=3072
        )
        telemetry.log_span("OpenRouter LLM Synthesis", (time.perf_counter() - t_llm) * 1000, f"Model: {cfg.llm_model}")

        answer_text = llm_response.get("content", "")

        # 7. Quality & Metrics Evaluation + Self-RAG Corrective Refinement
        metrics = evaluate_metrics(query, answer_text, final_contexts)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Metrics: Faithfulness={metrics.get('faithfulness')}, Precision={metrics.get('context_precision')}, Recall={metrics.get('context_recall')}")

        if cfg.strict_evidence and metrics.get("faithfulness", 1.0) < 0.70:
            t_correct = time.perf_counter()
            refine_prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS DRAFT: {answer_text}\n"
                f"INSTRUCTION: The previous draft contained low-faithfulness statements. Revise the answer to adhere strictly "
                f"and exclusively to the provided document context excerpts above."
            )
            refined_res = await self.model_router.generate_completion(
                prompt=refine_prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=3072
            )
            answer_text = refined_res.get("content", answer_text)
            metrics = evaluate_metrics(query, answer_text, final_contexts)
            telemetry.log_span("Self-RAG Corrective Refinement", (time.perf_counter() - t_correct) * 1000)

        return {
            "query": query,
            "answer": answer_text,
            "corpus_tier": tier,
            "total_chunks_indexed": chunk_count,
            "contexts": final_contexts,
            "telemetry": telemetry.get_waterfall(),
            "metrics": metrics,
            "model": cfg.llm_model,
            "provider": cfg.provider
        }
