import os
import logging
from typing import Any, Optional
import chromadb

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Manages ChromaDB persistent vector database indexing and retrieval."""

    def __init__(self, db_dir: str = "./data/chroma_db", collection_name: str = "ultimate_rag_corpus"):
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def recreate_collection(self):
        """Hard drop and recreate collection to reset embedding dimension constraints."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def clear(self, hard_reset: bool = True):
        """Clears collection. hard_reset=True drops collection to allow dimension changes."""
        if hard_reset:
            self.recreate_collection()
        else:
            try:
                if self.collection.count() > 0:
                    all_ids = self.collection.get(include=[])["ids"]
                    if all_ids:
                        self.collection.delete(ids=all_ids)
            except Exception as e:
                logger.warning(f"Error during collection delete(ids): {e}. Re-creating collection handle.")
                self.recreate_collection()

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: Optional[list[list[float]]] = None):
        if not chunks:
            return

        if logger.isEnabledFor(logging.DEBUG):
            emb_info = f"with {len(embeddings)} pre-computed embeddings" if embeddings else "using default ChromaDB embeddings"
            logger.debug(f"[ChromaDB] Indexing {len(chunks)} chunk(s) {emb_info} into collection '{self.collection_name}'...")

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "source": str(c["metadata"].get("source", "")),
                "page": int(c["metadata"].get("page", 1)),
                "chunk_index": int(c["metadata"].get("chunk_index", 0)),
                "heading": str(c["metadata"].get("heading", "")),
                "parent_id": str(c["metadata"].get("parent_id", c["chunk_id"])),
                "chunk_type": str(c["metadata"].get("chunk_type", "child")),
            }
            for c in chunks
        ]

        kwargs = {"embeddings": embeddings} if (embeddings and len(embeddings) == len(chunks)) else {}
        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, **kwargs)
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Embedding dimension mismatch during add_chunks ({e}). Recreating collection with new dimension and retrying...")
                self.recreate_collection()
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas, **kwargs)
            else:
                raise e

    def search_vector(self, query_text: str, top_k: int = 20, query_embedding: Optional[list[float]] = None) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        if logger.isEnabledFor(logging.DEBUG):
            emb_status = f"embedding vector (dim={len(query_embedding)})" if query_embedding else "text query"
            logger.debug(f"[ChromaDB Vector Search] Query: '{query_text}' | TopK: {top_k} | Search mode: {emb_status}")

        results = None
        try:
            if query_embedding:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.count()),
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, self.count()),
                    include=["documents", "metadatas", "distances"]
                )
        except Exception as e:
            err_str = str(e)
            if "dimension" in err_str.lower():
                logger.warning(
                    f"Vector search dimension mismatch ({err_str}). "
                    "Corpus was indexed with a different embedding dimension. "
                    "Please click 'Clear Vector & BM25 Corpus Index' in Settings and re-upload your documents to re-index. "
                    "Falling back to BM25 sparse search for this query."
                )
                return []
            else:
                raise e

        output = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for i in range(len(ids)):
                similarity = 1.0 - max(0.0, float(dists[i]))
                if similarity >= 0.15:
                    output.append({
                        "chunk_id": ids[i],
                        "content": docs[i],
                        "metadata": metas[i],
                        "score": round(similarity, 4)
                    })

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[ChromaDB Vector Search] Returned {len(output)} vector hit(s). Top score: {output[0]['score'] if output else 'N/A'}")

        return [r for r in output if r["metadata"].get("chunk_type", "child") != "parent"]

    def delete_by_source(self, source_name: str):
        """Deletes all chunks associated with a specific file source."""
        for src in {source_name, os.path.basename(source_name)}:
            try:
                self.collection.delete(where={"source": src})
            except Exception:
                pass

    def get_all_chunks(self) -> list[dict[str, Any]]:
        count = self.count()
        if count == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"], limit=count)
        return [
            {
                "chunk_id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i]
            }
            for i in range(len(results["ids"]))
            if results["metadatas"][i].get("chunk_type", "child") != "parent"
        ]

    def get_by_ids(self, ids: list[str]) -> dict[str, str]:
        """Fetch chunk contents by ID — used for small-to-big parent content resolution."""
        if not ids:
            return {}
        try:
            results = self.collection.get(ids=list(ids), include=["documents"])
            return dict(zip(results["ids"], results["documents"]))
        except Exception as e:
            logger.warning(f"get_by_ids failed: {e}")
            return {}
