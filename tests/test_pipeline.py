import os
import sys

# Ensure project root is in sys.path when script is executed directly from inside tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import tempfile
from ingestion.multi_parser import parse_file
from ingestion.semantic_chunker import SemanticChunker
from retrieval.bm25_engine import BM25Engine
from retrieval.hybrid_fusion import fuse_rrf, apply_mmr


class TestRAGPipeline(unittest.TestCase):
    """Zero-dependency lightweight test suite for RAG pipeline modules."""

    def test_text_parsing(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
            tmp.write("This is a sample document for RAG ingestion testing.")
            tmp_path = tmp.name

        try:
            pages = parse_file(tmp_path, "sample.txt")
            self.assertEqual(len(pages), 1)
            self.assertIn("sample document", pages[0]["content"])
        finally:
            os.remove(tmp_path)

    def test_semantic_chunking(self):
        chunker = SemanticChunker(chunk_size=50, child_size=30)
        pages = [{
            "content": "# Test Heading\n\nThis is paragraph one explaining vector indexing. This is paragraph two explaining sparse search.",
            "metadata": {"source": "test.md", "page": 1, "type": "text"}
        }]
        chunks = chunker.chunk_documents(pages)
        self.assertGreater(len(chunks), 0)

    def test_bm25_search(self):
        bm = BM25Engine()
        chunks = [
            {"chunk_id": "c1", "content": "PyMuPDF parses PDF documents efficiently."},
            {"chunk_id": "c2", "content": "ChromaDB stores vector embeddings for cosine similarity search."},
            {"chunk_id": "c3", "content": "FastAPI serves REST API endpoints for user queries."}
        ]
        bm.index_chunks(chunks)
        results = bm.search("PyMuPDF PDF")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_hybrid_fusion_rrf(self):
        dense = [{"chunk_id": "c1", "content": "a", "metadata": {}, "score": 0.9}]
        sparse = [{"chunk_id": "c1", "content": "a", "metadata": {}, "score": 0.8}]
        fused = fuse_rrf(dense, sparse)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
