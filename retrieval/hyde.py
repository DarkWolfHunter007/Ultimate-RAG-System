import logging
from typing import Optional
from core.router import ModelRouter

logger = logging.getLogger(__name__)


async def generate_hypothetical_document(query: str, router: Optional[ModelRouter] = None) -> str:
    """Generates a synthetic authoritative passage for HyDE-based retrieval."""
    router = router or ModelRouter()
    prompt = (
        f"Write a brief, authoritative passage that answers the following question:\n"
        f"Question: {query}\n"
        f"Passage:"
    )
    try:
        res = await router.generate_completion(prompt=prompt, temperature=0.5, max_tokens=250)
        return res.get("content", query)
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
        return query
