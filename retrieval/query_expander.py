import logging
from typing import List, Optional
from core.router import ModelRouter

logger = logging.getLogger(__name__)

async def expand_query(query: str, router: Optional[ModelRouter] = None) -> List[str]:
    """Generates alternative rephrasings of a user query for RAG-Fusion."""
    if router is None:
        router = ModelRouter()

    prompt = (
        f"You are an AI search optimizer. Generate 3 alternative versions or rephrasings of the following query "
        f"to help retrieve all relevant document sections. Return ONLY the 3 queries, one per line, with no extra text or numbering.\n\n"
        f"Original Query: {query}"
    )
    try:
        resp = await router.generate_completion(
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

class QueryExpander:
    """Backwards-compatible wrapper around expand_query."""
    def __init__(self, model_router: Optional[ModelRouter] = None):
        self.router = model_router or ModelRouter()

    async def expand_query(self, query: str) -> List[str]:
        return await expand_query(query, self.router)
