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

from retrieval.query_expander import QueryExpander

class AdaptiveEngine:
    """Dynamic Adaptive Search Engine that scales search algorithms based on corpus size."""

    def __init__(self, vector_indexer: VectorIndexer, bm25_engine: BM25Engine):
        self.vector_indexer = vector_indexer
        self.bm25_engine = bm25_engine
        self.config_mgr = ConfigManager.get_instance()
        self.model_router = ModelRouter()
        self.hyde_gen = HyDEGenerator(self.model_router)
        self.query_expander = QueryExpander(self.model_router)
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

        # 1. Multi-Query Expansion & HyDE Check
        search_queries = [query]
        if cfg.enable_hyde or tier == "Enterprise":
            t0 = time.perf_counter()
            hyde_doc = await self.hyde_gen.generate_hypothetical_document(query)
            search_queries.append(hyde_doc)
            telemetry.log_span("HyDE Synthesis", (time.perf_counter() - t0) * 1000, f"Expanded via {cfg.llm_model}")
        else:
            t_exp = time.perf_counter()
            search_queries = await self.query_expander.expand_query(query)
            telemetry.log_span("Multi-Query Expansion", (time.perf_counter() - t_exp) * 1000, f"Generated {len(search_queries)} query variants")

        # 2. Multi-Query Vector & BM25 Search
        t_vec = time.perf_counter()
        all_dense = []
        all_sparse = []

        for q_var in search_queries:
            query_embeddings = None
            try:
                emb_res = await self.model_router.generate_embeddings([q_var])
                if emb_res:
                    query_embeddings = emb_res[0]
            except Exception as e:
                logger.info(f"Query embedding fallback: {e}")

            d_res = self.vector_indexer.search_vector(query_text=q_var, top_k=cfg.top_k_candidates, query_embedding=query_embeddings)
            s_res = self.bm25_engine.search(query=q_var, top_k=cfg.top_k_candidates)
            all_dense.extend(d_res)
            all_sparse.extend(s_res)

        telemetry.log_span("Multi-Query Dense Search", (time.perf_counter() - t_vec) * 1000, f"Fetched {len(all_dense)} total hits")

        # 3. Hybrid Reciprocal Rank Fusion
        t_rrf = time.perf_counter()
        fused_candidates = HybridFusion.fuse_rrf(dense_results=all_dense, sparse_results=all_sparse, alpha=cfg.hybrid_alpha)
        telemetry.log_span("RAG-Fusion & RRF", (time.perf_counter() - t_rrf) * 1000, f"Alpha={cfg.hybrid_alpha}")

        # 4. Optional Cross-Encoder Reranker
        if (cfg.enable_reranker or tier == "Enterprise") and fused_candidates:
            t_rerank = time.perf_counter()
            fused_candidates = self.reranker.rerank(query, fused_candidates, top_n=cfg.top_k_candidates)
            telemetry.log_span("Cross-Encoder Neural Rerank", (time.perf_counter() - t_rerank) * 1000)

        # 5. MMR Diversity Selection & Full Corpus Fetching for Small/Summary Queries
        if tier == "Nano" or any(w in query.lower() for w in ["summarize", "summary", "all content", "everything", "overview"]):
            all_chunks = self.vector_indexer.get_all_chunks()
            if len(all_chunks) <= 40:
                final_contexts = all_chunks
            else:
                final_contexts = HybridFusion.apply_mmr(candidates=fused_candidates, top_n=min(cfg.top_n_final * 2, len(fused_candidates)), mmr_lambda=cfg.mmr_lambda)
        else:
            final_contexts = HybridFusion.apply_mmr(candidates=fused_candidates, top_n=cfg.top_n_final, mmr_lambda=cfg.mmr_lambda)

        # 6. Parent-Child Content Resolution & XML Assembly
        t_llm = time.perf_counter()
        context_str = ""
        for idx, c in enumerate(final_contexts, 1):
            src = c["metadata"].get("source", "doc")
            pg = c["metadata"].get("page", 1)
            hd = c["metadata"].get("heading", "")
            heading_str = f" heading='{hd}'" if hd else ""
            # Small-to-Big / Parent-Child: Use full parent content for LLM synthesis if available
            chunk_body = c.get("metadata", {}).get("parent_content") or c["content"]
            context_str += f"\n<DOCUMENT_EXCERPT id='{idx}' source='{src}' page='{pg}'{heading_str}>\n{chunk_body}\n</DOCUMENT_EXCERPT>\n"

        history_str = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role_name = "User" if msg.get("role") == "user" else "Assistant"
                history_str += f"{role_name}: {msg.get('content', '')}\n"

        system_prompt = (
            "You are Ultimate-RAG-System, an enterprise AI assistant engaged in an interactive document chat. "
            "Answer the user question thoroughly based on the provided structured <DOCUMENT_EXCERPT> tags and conversation history. "
            "Use GitHub-flavored markdown for clear formatting (bold text, bullet points, headers, formatted code blocks). "
            "Cite relevant document excerpts using bracketed numbers like [1], [2] inline where applicable. "
            "If evidence is missing or context is insufficient, state clearly what is missing."
        )
        
        user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        if history_str:
            user_prompt += f"RECENT CONVERSATION HISTORY:\n{history_str}\n\n"
        user_prompt += f"USER QUESTION: {query}"

        llm_response = await self.model_router.generate_completion(prompt=user_prompt, system_prompt=system_prompt, temperature=0.2)
        telemetry.log_span("OpenRouter LLM Synthesis", (time.perf_counter() - t_llm) * 1000, f"Model: {cfg.llm_model}")

        answer_text = llm_response.get("content", "")

        # 7. Quality & Metrics Evaluation + Self-RAG Corrective Refinement
        metrics = MetricsEvaluator.evaluate(query, answer_text, final_contexts)
        
        if cfg.strict_evidence and metrics.get("faithfulness", 1.0) < 0.70:
            t_correct = time.perf_counter()
            refine_prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS DRAFT: {answer_text}\n"
                f"INSTRUCTION: The previous draft contained low-faithfulness statements. Revise the answer to adhere strictly "
                f"and exclusively to the provided document context excerpts above."
            )
            refined_res = await self.model_router.generate_completion(prompt=refine_prompt, system_prompt=system_prompt, temperature=0.1)
            answer_text = refined_res.get("content", answer_text)
            metrics = MetricsEvaluator.evaluate(query, answer_text, final_contexts)
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
