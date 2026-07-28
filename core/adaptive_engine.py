import time
import asyncio
import logging
from typing import Any, Optional, AsyncGenerator
from core.config_manager import ConfigManager
from core.router import ModelRouter
from ingestion.vector_indexer import VectorIndexer
from retrieval.bm25_engine import BM25Engine
from retrieval.hybrid_fusion import fuse_rrf, apply_mmr
from retrieval.reranker import rerank
from retrieval.hyde import generate_hypothetical_document
from retrieval.query_expander import expand_query
from analytics.telemetry_logger import TelemetryLogger
from analytics.metrics_evaluator import evaluate_metrics

logger = logging.getLogger(__name__)


class AdaptiveEngine:
    """
    Adaptive RAG Orchestrator.
    Manages multi-query expansion, hybrid RRF fusion, neural reranking, MMR diversity,
    small-to-big parent resolution, self-RAG corrective evaluation, and real-time streaming.
    """

    def __init__(self, vector_indexer: VectorIndexer, bm25_engine: BM25Engine):
        self.config_mgr = ConfigManager()
        self.vector_indexer = vector_indexer
        self.bm25_engine = bm25_engine
        self.model_router = ModelRouter()

    def get_corpus_tier(self) -> tuple[str, int]:
        total_chunks = self.vector_indexer.count()
        if total_chunks < 50:
            return "Nano", total_chunks
        elif total_chunks <= 5000:
            return "Standard", total_chunks
        else:
            return "Enterprise", total_chunks

    async def _embed_query(self, q: str) -> Optional[list[float]]:
        try:
            emb_res = await self.model_router.generate_embeddings([q])
            return emb_res[0] if (emb_res and len(emb_res) > 0) else None
        except Exception as e:
            logger.warning(f"Failed to generate query embedding for '{q[:30]}...': {e}")
            return None

    async def _prepare_rag_context(self, query: str, chat_history: Optional[list[dict[str, Any]]] = None) -> tuple[list[dict[str, Any]], str, TelemetryLogger, str, int]:
        telemetry = TelemetryLogger()
        cfg = self.config_mgr.get_config()
        tier, total_chunks = self.get_corpus_tier()

        t0 = time.perf_counter()
        query_variants = [query]

        if cfg.enable_hyde:
            hypo_doc = await generate_hypothetical_document(query, self.model_router)
            query_variants = [hypo_doc]
            telemetry.log_span("HyDE Generation", (time.perf_counter() - t0) * 1000, f"Hypothetical Doc: {hypo_doc[:60]}...")
        else:
            expanded = await expand_query(query, self.model_router)
            query_variants = list(set([query] + expanded))
            telemetry.log_span("Multi-Query Expansion", (time.perf_counter() - t0) * 1000, f"Generated {len(query_variants)} variants")

        t_search = time.perf_counter()
        embeddings: list[Optional[list[float]]] = await asyncio.gather(
            *[self._embed_query(q) for q in query_variants]
        )

        all_dense: list[dict[str, Any]] = []
        all_sparse: list[dict[str, Any]] = []

        for q, q_emb in zip(query_variants, embeddings):
            dense = self.vector_indexer.search_vector(query_text=q, top_k=cfg.top_k_candidates, query_embedding=q_emb)
            sparse = self.bm25_engine.search(query=q, top_k=cfg.top_k_candidates)
            all_dense.extend(dense)
            all_sparse.extend(sparse)

        telemetry.log_span("Hybrid Vector + BM25 Search", (time.perf_counter() - t_search) * 1000, f"Aggregated {len(all_dense)} vector & {len(all_sparse)} BM25 hits across variants")

        t_fusion = time.perf_counter()
        fused_candidates = fuse_rrf(all_dense, all_sparse, alpha=cfg.hybrid_alpha, k=60)
        telemetry.log_span("RRF Hybrid Fusion", (time.perf_counter() - t_fusion) * 1000, f"Fused into {len(fused_candidates)} candidates (alpha={cfg.hybrid_alpha})")

        t_rerank = time.perf_counter()
        if cfg.enable_reranker and fused_candidates:
            fused_candidates = rerank(query, fused_candidates, top_n=cfg.top_k_candidates)
            telemetry.log_span("Cross-Encoder Neural Rerank", (time.perf_counter() - t_rerank) * 1000)

        if tier == "Nano" or any(w in query.lower() for w in ["summarize", "summary", "all content", "everything", "overview"]):
            all_chunks = self.vector_indexer.get_all_chunks()
            if len(all_chunks) <= 40:
                final_contexts = all_chunks
            else:
                final_contexts = apply_mmr(fused_candidates, top_n=cfg.top_n_final, mmr_lambda=cfg.mmr_lambda)
        else:
            final_contexts = apply_mmr(fused_candidates, top_n=cfg.top_n_final, mmr_lambda=cfg.mmr_lambda)

        parent_ids = list({c["metadata"].get("parent_id") for c in final_contexts if c.get("metadata", {}).get("parent_id")})
        parent_map: dict[str, str] = self.vector_indexer.get_by_ids(parent_ids) if parent_ids else {}

        context_blocks = []
        for i, c in enumerate(final_contexts, 1):
            pid = c.get("metadata", {}).get("parent_id")
            resolved_content = parent_map.get(pid, c["content"]) if pid else c["content"]
            meta = c.get("metadata", {})
            src = meta.get("source", "doc")
            pg = meta.get("page", 1)
            context_blocks.append(f"[Document {i} | Source: {src} (Page {pg})]\n{resolved_content}")

        formatted_context = "\n\n---\n\n".join(context_blocks)
        return final_contexts, formatted_context, telemetry, tier, total_chunks

    async def execute_rag_pipeline(self, query: str, chat_history: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        cfg = self.config_mgr.get_config()
        final_contexts, formatted_context, telemetry, tier, total_chunks = await self._prepare_rag_context(query, chat_history)

        t_llm = time.perf_counter()
        system_instruction = (
            "You are an expert AI Assistant answering user questions strictly using the provided Context Documents.\n"
            "Rules:\n"
            "1. Base your answer ONLY on the information contained in the Context Documents.\n"
            "2. If the answer cannot be determined from the Context Documents, state clearly: 'I cannot find relevant information in the uploaded documents to answer this question.'\n"
            "3. Be concise, direct, and structured."
        )

        history_prompt = ""
        if chat_history and len(chat_history) > 0:
            history_lines = []
            for m in chat_history[-6:]:
                role = "User" if m.get("role") == "user" else "Assistant"
                content = m.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_prompt = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"{history_prompt}Context Documents:\n{formatted_context}\n\nUser Question: {query}\n\nAnswer:"

        llm_response = await self.model_router.generate_completion(
            prompt=prompt,
            system_prompt=system_instruction,
            temperature=0.2
        )
        telemetry.log_span("OpenRouter LLM Synthesis", (time.perf_counter() - t_llm) * 1000, f"Model: {cfg.llm_model}")

        answer_text = llm_response.get("content", "") or ""
        metrics = evaluate_metrics(query, answer_text, final_contexts)

        if cfg.strict_evidence and metrics["faithfulness"] < 0.70 and answer_text:
            logger.warning(f"Self-RAG Corrective Triggered! Faithfulness score low ({metrics['faithfulness']}). Re-synthesizing with strict grounding...")
            t_refine = time.perf_counter()
            strict_system = (
                "STRICT GROUNDING MODE: You are an ultra-precise Q&A model.\n"
                "Your previous answer had low faithfulness to the source text.\n"
                "Answer the user's question USING ONLY EXACT FACTS explicitly stated in the Context Documents below.\n"
                "Do NOT extrapolate or assume anything. Quote or closely rephrase source sentences."
            )
            refined_resp = await self.model_router.generate_completion(
                prompt=prompt,
                system_prompt=strict_system,
                temperature=0.0
            )
            telemetry.log_span("Self-RAG Refinement Pass", (time.perf_counter() - t_refine) * 1000)
            if refined_resp.get("content"):
                answer_text = refined_resp["content"]
                metrics = evaluate_metrics(query, answer_text, final_contexts)

        waterfall = telemetry.get_waterfall()

        return {
            "query": query,
            "answer": answer_text,
            "contexts": final_contexts,
            "metrics": metrics,
            "telemetry": waterfall,
            "model": cfg.llm_model,
            "provider": cfg.provider,
            "corpus_tier": tier,
            "total_chunks_searched": total_chunks
        }

    async def execute_rag_stream(self, query: str, chat_history: Optional[list[dict[str, Any]]] = None) -> AsyncGenerator[str, None]:
        """Streams completion tokens word-by-word via Server-Sent Events (SSE)."""
        cfg = self.config_mgr.get_config()
        final_contexts, formatted_context, telemetry, tier, total_chunks = await self._prepare_rag_context(query, chat_history)

        system_instruction = (
            "You are an expert AI Assistant answering user questions strictly using the provided Context Documents.\n"
            "Rules:\n"
            "1. Base your answer ONLY on the information contained in the Context Documents.\n"
            "2. If the answer cannot be determined from the Context Documents, state clearly: 'I cannot find relevant information in the uploaded documents to answer this question.'\n"
            "3. Be concise, direct, and structured."
        )

        history_prompt = ""
        if chat_history and len(chat_history) > 0:
            history_lines = [f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}" for m in chat_history[-6:]]
            history_prompt = "Recent Conversation History:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"{history_prompt}Context Documents:\n{formatted_context}\n\nUser Question: {query}\n\nAnswer:"

        header_payload = {
            "type": "header",
            "contexts": final_contexts,
            "model": cfg.llm_model,
            "provider": cfg.provider,
            "telemetry": telemetry.get_waterfall()
        }
        yield f"data: {json.dumps(header_payload)}\n\n"

        async for token in self.model_router.stream_completion(prompt=prompt, system_prompt=system_instruction, temperature=0.2):
            token_payload = {"type": "token", "content": token}
            yield f"data: {json.dumps(token_payload)}\n\n"

        yield "data: [DONE]\n\n"
