import logging
from typing import List
from core.router import ModelRouter

logger = logging.getLogger(__name__)

class QueryExpander:
    """Multi-query generator that expands user queries into alternative search perspectives for RAG-Fusion."""

    def __init__(self, model_router: ModelRouter):
        self.router = model_router

    async def expand_query(self, query: str) -> List[str]:
        prompt = (
            f"You are an AI search optimizer. Generate 3 alternative versions or rephrasings of the following query "
            f"to help retrieve all relevant document sections. Return ONLY the 3 queries, one per line, with no extra text or numbering.\n\n"
            f"Original Query: {query}"
        )
        try:
            resp = await self.router.generate_completion(
                prompt=prompt,
                system_prompt="Generate 3 search query variations, one per line.",
                temperature=0.4
            )
            raw = resp.get("content", "").strip()
            lines = [l.strip().lstrip("123456789.-* ") for l in raw.split("\n") if l.strip()]
            queries = [query] + [l for l in lines if len(l) > 3][:3]
            return list(set(queries))
        except Exception as e:
            logger.warning(f"Query expansion fallback: {e}")
            return [query]
