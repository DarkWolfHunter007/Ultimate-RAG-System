import asyncio
import traceback
from core.adaptive_engine import AdaptiveEngine
from ingestion.vector_indexer import VectorIndexer
from retrieval.bm25_engine import BM25Engine

async def main():
    try:
        indexer = VectorIndexer()
        bm25 = BM25Engine()
        engine = AdaptiveEngine(indexer, bm25)
        res = await engine.execute_rag_pipeline("What is this system?")
        print("SUCCESS:", res)
    except Exception as e:
        print("EXCEPTION OCCURRED:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
