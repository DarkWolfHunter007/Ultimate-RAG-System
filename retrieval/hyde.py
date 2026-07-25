from typing import Optional
import logging
from core.router import ModelRouter

logger = logging.getLogger(__name__)

class HyDEGenerator:
    """Generates synthetic draft responses to perform Hypothetical Document Embedding search."""

    def __init__(self, model_router: Optional[ModelRouter] = None):
        self.router = model_router or ModelRouter()

    async def generate_hypothetical_document(self, query: str) -> str:
        prompt = (
            f"Write a brief, authoritative passage that answers the following question:\n"
            f"Question: {query}\n"
            f"Passage:"
        )
        try:
            res = await self.router.generate_completion(prompt=prompt, temperature=0.5, max_tokens=250)
            return res.get("content", query)
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
            return query
