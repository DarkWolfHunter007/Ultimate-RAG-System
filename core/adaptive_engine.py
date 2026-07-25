import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from core.config_manager import ConfigManager
from core.router import ModelRouter
from ingestion.vector_indexer import VectorIndexer
from retrieval.bm25_engine import BM25Engine
from retrieval.hybrid_fusion import HybridFusion
from retrieval.hyde import HyDEGenerator
from retrieval.reranker import RerankerEngine
from analytics.telemetry_logger import TelemetryLogger
from analytics.metrics_evaluator import MetricsEvaluator

logger = logging.getLogger(__name__)

class AdaptiveEngine:
    """Dynamic Adaptive Search Engine that scales search algorithms based on corpus size."""

    def __init__(self, vector_indexer: VectorIndexer, bm25_engine: BM25Engine):
        self.vector_indexer = vector_indexer
        self.bm25_engine = bm25_engine
        self.config_mgr = ConfigManager.get_instance()
        self.model_router = ModelRouter()
        self.hyde_gen = HyDEGenerator(self.model_router)
        self.reranker = RerankerEngine()

    def get_corpus_tier(self) -> Tuple[str, int]:
        total_chunks = self.vector_indexer.count()
        if total_chunks < 50:
            return ("Nano", total_chunks)
        elif total_chunks <= 5000:
            return ("Standard", total_chunks)
        else:
            return ("Enterprise", total_chunks)

    async def execute_rag_pipeline(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        telemetry = TelemetryLogger(query_id=f"q_{int(time.time()*1000)}")
        cfg = self.config_mgr.get_config()
        tier, chunk_count = self.get_corpus_tier()

        search_query = query
        # 1. HyDE Check
        if cfg.enable_hyde or tier == "Enterprise":
            t0 = time.perf_counter()
            search_query = await self.hyde_gen.generate_hypothetical_document(query)
            telemetry.log_span("HyDE Synthesis", (time.perf_counter() - t0) * 1000, f"Query expanded via {cfg.llm_model}")

        # 2. Vector & BM25 Search
        t_vec = time.perf_counter()
        query_embeddings = None
        try:
            emb_res = await self.model_router.generate_embeddings([search_query])
            if emb_res:
                query_embeddings = emb_res[0]
        except Exception as e:
            logger.info(f"Query embedding fallback to default: {e}")

        dense_results = self.vector_indexer.search_vector(
            query_text=search_query,
            top_k=cfg.top_k_candidates,
            query_embedding=query_embeddings
        )
        telemetry.log_span("Dense Vector Search", (time.perf_counter() - t_vec) * 1000, f"Retrieved {len(dense_results)} candidates")

        t_bm25 = time.perf_counter()
        sparse_results = self.bm25_engine.search(query=search_query, top_k=cfg.top_k_candidates)
        telemetry.log_span("BM25 Keyword Search", (time.perf_counter() - t_bm25) * 1000, f"Retrieved {len(sparse_results)} candidates")

        # 3. Hybrid Reciprocal Rank Fusion
        t_rrf = time.perf_counter()
        fused_candidates = HybridFusion.fuse_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            alpha=cfg.hybrid_alpha
        )
        telemetry.log_span("Hybrid RRF Fusion", (time.perf_counter() - t_rrf) * 1000, f"Alpha={cfg.hybrid_alpha}")

        # 4. Optional Cross-Encoder Reranker
        if (cfg.enable_reranker or tier == "Enterprise") and fused_candidates:
            t_rerank = time.perf_counter()
            fused_candidates = self.reranker.rerank(query, fused_candidates, top_n=cfg.top_k_candidates)
            telemetry.log_span("Cross-Encoder Neural Rerank", (time.perf_counter() - t_rerank) * 1000)

        # 5. MMR Diversity Selection
        final_contexts = HybridFusion.apply_mmr(
            candidates=fused_candidates,
            top_n=cfg.top_n_final,
            mmr_lambda=cfg.mmr_lambda
        )

        # 6. LLM Context Synthesis & Conversation History Formatting
        t_llm = time.perf_counter()
        context_str = ""
        for idx, c in enumerate(final_contexts, 1):
            src = c["metadata"].get("source", "doc")
            pg = c["metadata"].get("page", 1)
            context_str += f"\n--- EXCERPT [{idx}] (Source: {src}, Page {pg}) ---\n{c['content']}\n"

        history_str = ""
        if chat_history:
            for msg in chat_history[-6:]:  # last 6 turns
                role_name = "User" if msg.get("role") == "user" else "Assistant"
                history_str += f"{role_name}: {msg.get('content', '')}\n"

        system_prompt = (
            "You are Ultimate-RAG-System, an enterprise AI assistant engaged in an interactive document chat. "
            "Answer the user question based strictly on the provided document context and conversation history. "
            "Use GitHub-flavored markdown for clear formatting (bold text, bullet points, headers, formatted code blocks). "
            "Cite relevant document excerpts using bracketed numbers like [1], [2] inline where applicable. "
            "If evidence is missing or context is insufficient, state clearly what is missing."
        )
        
        user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        if history_str:
            user_prompt += f"RECENT CONVERSATION HISTORY:\n{history_str}\n\n"
        user_prompt += f"USER QUESTION: {query}"

        llm_response = await self.model_router.generate_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )
        telemetry.log_span("OpenRouter LLM Synthesis", (time.perf_counter() - t_llm) * 1000, f"Model: {cfg.llm_model}")

        answer_text = llm_response.get("content", "")

        # 7. Quality & Metrics Evaluation
        metrics = MetricsEvaluator.evaluate(query, answer_text, final_contexts)

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
