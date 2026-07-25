from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Dict, Any, Optional

from core.config_manager import ConfigManager
from core.adaptive_engine import AdaptiveEngine
from ingestion.multi_parser import MultiParser
from ingestion.semantic_chunker import SemanticChunker
from ingestion.vector_indexer import VectorIndexer
from retrieval.bm25_engine import BM25Engine

app = FastAPI(title="Ultimate-RAG-System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from core.config_manager import ConfigManager, CustomModel
from core.chat_manager import ChatManager

# Initialize Core Services
vector_indexer = VectorIndexer()
bm25_engine = BM25Engine()
adaptive_engine = AdaptiveEngine(vector_indexer, bm25_engine)
config_mgr = ConfigManager.get_instance()
chat_mgr = ChatManager.get_instance()

# Temp upload directory
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

import time
import httpx

class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None

class ConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    hybrid_alpha: Optional[float] = None
    mmr_lambda: Optional[float] = None
    top_k_candidates: Optional[int] = None
    top_n_final: Optional[int] = None
    enable_reranker: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_graph_rag: Optional[bool] = None
    strict_evidence: Optional[bool] = None

class AddModelRequest(BaseModel):
    id: str
    name: str
    provider: str = "openrouter"
    type: str = "llm"

class PingModelRequest(BaseModel):
    provider: str = "openrouter"
    model: str
    type: Optional[str] = "llm"  # "llm" or "embedding"
    openrouter_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None

class ResetConfigRequest(BaseModel):
    clear_credentials: bool = False
    clear_custom_models: bool = False

@app.get("/api/config")
async def get_config():
    return config_mgr.get_config().model_dump()

@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    updated = config_mgr.update_config(req.model_dump())
    return updated.model_dump()

@app.post("/api/config/reset")
async def reset_config(req: ResetConfigRequest):
    updated = config_mgr.reset_config(clear_credentials=req.clear_credentials, clear_custom_models=req.clear_custom_models)
    return updated.model_dump()

@app.post("/api/models")
async def add_custom_model(req: AddModelRequest):
    model = CustomModel(id=req.id.strip(), name=req.name.strip(), provider=req.provider, type=req.type)
    updated = config_mgr.add_model(model)
    return updated.model_dump()

@app.delete("/api/models/{model_id:path}")
async def remove_custom_model(model_id: str):
    updated = config_mgr.remove_model(model_id)
    return updated.model_dump()

@app.post("/api/models/ping")
async def ping_model(req: PingModelRequest):
    start_time = time.time()
    cfg = config_mgr.get_config()
    provider = req.provider or cfg.provider
    key = req.openrouter_api_key or cfg.openrouter_api_key
    ollama_url = req.ollama_base_url or cfg.ollama_base_url

    is_embedding = (req.type == "embedding") or ("embed" in req.model.lower())

    if provider == "openrouter":
        if not key.strip():
            return {
                "status": "error",
                "latency_ms": 0,
                "message": "Missing OpenRouter API Key. Please configure your API key first."
            }
        
        headers = {
            "Authorization": f"Bearer {key.strip()}",
            "HTTP-Referer": "https://github.com/Ultimate-RAG-System",
            "X-Title": "Ultimate RAG System",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if is_embedding:
                    endpoint = f"{cfg.openrouter_base_url}/embeddings"
                    payload = {
                        "model": req.model,
                        "input": ["ping"]
                    }
                else:
                    endpoint = f"{cfg.openrouter_base_url}/chat/completions"
                    payload = {
                        "model": req.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    }

                resp = await client.post(endpoint, headers=headers, json=payload)
                elapsed_ms = round((time.time() - start_time) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    dimension = None
                    if is_embedding:
                        emb = data.get("data", [{}])[0].get("embedding", [])
                        dimension = len(emb) if isinstance(emb, list) else None

                    msg = f"Success ({elapsed_ms}ms)! Embedding model '{req.model}' is valid (Dimension: {dimension}d)." if is_embedding and dimension else f"Success ({elapsed_ms}ms)! Model '{req.model}' is valid and responding."

                    return {
                        "status": "ok",
                        "latency_ms": elapsed_ms,
                        "dimension": dimension,
                        "message": msg
                    }
                elif resp.status_code == 401:
                    return {
                        "status": "error",
                        "latency_ms": elapsed_ms,
                        "message": "Unauthorized (401): Invalid OpenRouter API Key."
                    }
                elif resp.status_code == 404:
                    return {
                        "status": "error",
                        "latency_ms": elapsed_ms,
                        "message": f"Not Found (404): Model '{req.model}' was not found on OpenRouter."
                    }
                else:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:120])
                    return {
                        "status": "error",
                        "latency_ms": elapsed_ms,
                        "message": f"Error [{resp.status_code}]: {err_msg}"
                    }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "latency_ms": elapsed_ms,
                "message": f"Connection Error: {str(e)[:150]}"
            }

    elif provider == "ollama":
        if is_embedding:
            target_url = f"{ollama_url.rstrip('/')}/api/embeddings"
            model_short = req.model.split("/")[-1]
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(target_url, json={"model": model_short, "prompt": "ping"})
                    elapsed_ms = round((time.time() - start_time) * 1000)
                    if resp.status_code == 200:
                        emb = resp.json().get("embedding", [])
                        dimension = len(emb) if isinstance(emb, list) else None
                        return {
                            "status": "ok",
                            "latency_ms": elapsed_ms,
                            "dimension": dimension,
                            "message": f"Success ({elapsed_ms}ms)! Ollama embedding '{model_short}' active (Dimension: {dimension}d)."
                        }
                    else:
                        return {
                            "status": "error",
                            "latency_ms": elapsed_ms,
                            "message": f"Ollama Error [{resp.status_code}]: {resp.text[:120]}"
                        }
            except Exception as e:
                elapsed_ms = round((time.time() - start_time) * 1000)
                return {
                    "status": "error",
                    "latency_ms": elapsed_ms,
                    "message": f"Ollama Connection Failed ({ollama_url}): Ensure model '{model_short}' is installed."
                }
        else:
            target_url = f"{ollama_url.rstrip('/')}/api/tags"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(target_url)
                    elapsed_ms = round((time.time() - start_time) * 1000)
                    if resp.status_code == 200:
                        models_data = resp.json().get("models", [])
                        model_names = [m.get("name", "") for m in models_data]
                        model_short = req.model.split("/")[-1]
                        found = any(model_short in name for name in model_names)
                        if found or len(models_data) > 0:
                            return {
                                "status": "ok",
                                "latency_ms": elapsed_ms,
                                "message": f"Success ({elapsed_ms}ms)! Ollama host reachable." + (f" Model '{model_short}' is installed." if found else " Server online.")
                            }
                        else:
                            return {
                                "status": "warning",
                                "latency_ms": elapsed_ms,
                                "message": f"Ollama reachable, but model '{model_short}' was not found in `ollama list`."
                            }
                    else:
                        return {
                            "status": "error",
                            "latency_ms": elapsed_ms,
                            "message": f"Ollama Error [{resp.status_code}]: {resp.text[:120]}"
                        }
            except Exception as e:
                elapsed_ms = round((time.time() - start_time) * 1000)
                return {
                    "status": "error",
                    "latency_ms": elapsed_ms,
                    "message": f"Ollama Connection Failed ({ollama_url}): Ensure Ollama is running."
                }

    return {
        "status": "error",
        "latency_ms": 0,
        "message": f"Unknown provider: {provider}"
    }


@app.get("/api/stats")
async def get_stats():
    count = vector_indexer.count()
    tier, _ = adaptive_engine.get_corpus_tier()
    return {
        "total_chunks": count,
        "corpus_tier": tier
    }

# Persistent documents directory
DOCUMENTS_DIR = "./data/documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

@app.get("/api/documents")
async def list_documents():
    if not os.path.exists(DOCUMENTS_DIR):
        return []
    
    all_chunks = vector_indexer.get_all_chunks()
    chunk_counts = {}
    for c in all_chunks:
        src = c.get("metadata", {}).get("source", "")
        if src:
            chunk_counts[src] = chunk_counts.get(src, 0) + 1

    docs = []
    for fname in os.listdir(DOCUMENTS_DIR):
        fpath = os.path.join(DOCUMENTS_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            docs.append({
                "filename": fname,
                "size_bytes": stat.st_size,
                "chunk_count": chunk_counts.get(fname, 0),
                "modified_at": stat.st_mtime
            })
    docs.sort(key=lambda x: x["modified_at"], reverse=True)
    return docs

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    fpath = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
        except Exception as e:
            logger.warning(f"Could not remove document file '{filename}': {e}")
    
    # Delete chunks from ChromaDB & reindex BM25
    vector_indexer.delete_by_source(filename)
    all_remaining = vector_indexer.get_all_chunks()
    bm25_engine.index_chunks(all_remaining)
    
    tier, total = adaptive_engine.get_corpus_tier()
    return {
        "message": f"Successfully deleted '{filename}' and purged its vector chunks.",
        "total_chunks": total,
        "corpus_tier": tier
    }

@app.post("/api/clear")
async def clear_index():
    vector_indexer.clear()
    bm25_engine.index_chunks([])
    if os.path.exists(DOCUMENTS_DIR):
        for fname in os.listdir(DOCUMENTS_DIR):
            fpath = os.path.join(DOCUMENTS_DIR, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
    return {"message": "Corpus index and uploaded files successfully cleared."}

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    all_chunks = []
    chunker = SemanticChunker(chunk_size=350, chunk_overlap=40)

    for file in files:
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        parsed_pages = MultiParser.parse_file(file_path, file.filename)
        chunks = chunker.chunk_documents(parsed_pages)
        all_chunks.extend(chunks)

    if all_chunks:
        # Generate embeddings using active embedding model
        embeddings = None
        try:
            chunk_texts = [c["content"] for c in all_chunks]
            embeddings = await adaptive_engine.model_router.generate_embeddings(chunk_texts)
        except Exception as e:
            logger.warning(f"Could not generate custom embeddings during upload: {e}")

        # Index in ChromaDB and BM25
        vector_indexer.add_chunks(all_chunks, embeddings=embeddings)

        # Update BM25 with full chunk set
        all_existing = vector_indexer.get_all_chunks()
        bm25_engine.index_chunks(all_existing)

    tier, total = adaptive_engine.get_corpus_tier()
    return {
        "message": f"Successfully copied & indexed {len(all_chunks)} chunks across {len(files)} files.",
        "total_chunks": total,
        "corpus_tier": tier
    }

class CreateChatRequest(BaseModel):
    title: Optional[str] = None

@app.get("/api/chats")
async def list_chats():
    return chat_mgr.list_chats()

@app.post("/api/chats")
async def create_chat(req: Optional[CreateChatRequest] = None):
    title = req.title if req else None
    return chat_mgr.create_chat(title=title)

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = chat_mgr.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return chat

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    success = chat_mgr.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"message": "Chat session deleted successfully"}

@app.post("/api/query")
async def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chat_id = req.chat_id
    chat_history = None
    if chat_id:
        chat = chat_mgr.get_chat(chat_id)
        if chat:
            chat_history = chat.get("messages", [])

    try:
        res = await adaptive_engine.execute_rag_pipeline(req.query, chat_history=chat_history)
        
        if chat_id:
            # Store user message
            chat_mgr.add_message(chat_id, {
                "id": f"msg_u_{int(time.time()*1000)}",
                "role": "user",
                "content": req.query
            })
            # Store assistant response
            chat_mgr.add_message(chat_id, {
                "id": f"msg_a_{int(time.time()*1000)}",
                "role": "assistant",
                "content": res.get("answer", ""),
                "contexts": res.get("contexts", []),
                "telemetry": res.get("telemetry", {}),
                "metrics": res.get("metrics", {}),
                "model": res.get("model", ""),
                "provider": res.get("provider", "")
            })

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static Files serving
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Ultimate RAG System Dashboard</h1>")
