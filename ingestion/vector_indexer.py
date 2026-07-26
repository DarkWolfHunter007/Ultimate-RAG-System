import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os
import logging

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

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        if not chunks:
            return
        
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "source": str(c["metadata"].get("source", "")),
                "page": int(c["metadata"].get("page", 1)),
                "chunk_index": int(c["metadata"].get("chunk_index", 0)),
                "heading": str(c["metadata"].get("heading", ""))
            }
            for c in chunks
        ]

        try:
            if embeddings and len(embeddings) == len(chunks):
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Embedding dimension mismatch during add_chunks ({e}). Clearing collection and retrying...")
                self.clear()
                if embeddings and len(embeddings) == len(chunks):
                    self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                else:
                    self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            else:
                raise e

    def search_vector(self, query_text: str, top_k: int = 20, query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []

        results = None
        if query_embedding:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, self.count()),
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                err_str = str(e)
                if "dimension" in err_str.lower():
                    logger.warning(f"Query embedding dimension mismatch ({err_str}). Falling back to text query search.")
                    try:
                        results = self.collection.query(
                            query_texts=[query_text],
                            n_results=min(top_k, self.count()),
                            include=["documents", "metadatas", "distances"]
                        )
                    except Exception as fallback_err:
                        raise RuntimeError(
                            "Embedding dimension mismatch: The active embedding model's dimensions do not match the indexed corpus. "
                            "Please click '🗑️ Clear Vector & BM25 Corpus Index' in Settings and re-upload your documents to re-index them."
                        ) from fallback_err
                else:
                    raise e
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.count()),
                include=["documents", "metadatas", "distances"]
            )

        output = []
        if results and results.get("ids"):
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for i in range(len(ids)):
                # Convert cosine distance to similarity score
                similarity = 1.0 - max(0.0, float(dists[i]))
                output.append({
                    "chunk_id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "score": round(similarity, 4)
                })

        return output

    def delete_by_source(self, source_name: str):
        """Deletes all chunks associated with a specific file source."""
        try:
            self.collection.delete(where={"source": source_name})
        except Exception:
            pass
        try:
            self.collection.delete(where={"source": os.path.basename(source_name)})
        except Exception as e:
            logger.warning(f"ChromaDB delete by source '{source_name}' failed: {e}")

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        count = self.count()
        if count == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"], limit=count)
        output = []
        for i in range(len(results["ids"])):
            output.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        return output
