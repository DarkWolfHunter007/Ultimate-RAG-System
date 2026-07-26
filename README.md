# 🚀 Ultimate-RAG-System

> **Next-Generation, Self-Adapting RAG Platform with Live Control Center, Multi-Turn Chat, and Microsecond Telemetry Waterfall**

---

## 👤 Author & Lead Architect

**Allen MT Maliyil** ([@DarkWolfHunter007](https://github.com/DarkWolfHunter007))
- **GitHub Profile**: [github.com/DarkWolfHunter007](https://github.com/DarkWolfHunter007)
- **Official Repository**: [github.com/DarkWolfHunter007/Ultimate-RAG-System](https://github.com/DarkWolfHunter007/Ultimate-RAG-System)

---

## 📑 Overview

**Ultimate-RAG-System** is an enterprise-grade Retrieval-Augmented Generation (RAG) platform created by **Allen MT Maliyil**. It automatically evaluates document corpus volume and dynamically adapts its indexing and retrieval algorithms in real time.

Featuring an intuitive glassmorphic **Control Center & Multi-Turn Chat Workspace**, operators can tune retrieval parameters on the fly ($\alpha$ hybrid weights, $\lambda$ MMR diversity, candidate pool depth, cross-encoder rerankers), manage custom cloud and local LLM/embedding models, inspect real-time quality metrics (*Faithfulness*, *Context Precision*, *Context Recall*), and audit microsecond-level latency waterfalls.

---

## 🏛️ System Architecture

```mermaid
graph TD
    User["👤 User Query / Web UI"] --> CC["🎛️ Control Center & Settings Bar"]
    CC --> Router["🔀 Adaptive Algorithm Engine & Router"]
    
    subgraph Corpus Adaptor
        Router -->|Corpus < 50 Chunks| Direct["⚡ Direct Vector / Sparse Hybrid (Nano)"]
        Router -->|50 - 5k Chunks| MidTier["🔍 Hybrid Dense+BM25 + MMR (Standard)"]
        Router -->|Corpus > 5k Chunks| DeepTier["🌳 Hierarchical Vector + HyDE + Reranker (Enterprise)"]
    end
    
    Direct & MidTier & DeepTier --> DiagnosticEngine["📊 Diagnostic Telemetry Suite"]
    DiagnosticEngine --> OpenRouter["🌐 OpenRouter / Ollama LLM Gateway"]
    OpenRouter --> AnswerEngine["💬 Citation & Excerpt Synthesizer"]
    AnswerEngine --> User
```

---

## ⚡ Key Features

1. **Multi-Turn Interactive Document Chat**:
   - Continuous chat sessions preserved across messages with context history integration.
   - Session management (Create, Switch, and Delete conversations).
   - Rich GitHub Markdown rendering with interactive citation badges `[1]`, `[2]`.

2. **Persistent Document File Storage & Chunk Purging**:
   - Uploaded files (`PDF`, `Markdown`, `TXT`) are stored securely in `./data/documents/`.
   - Single-click per-document deletion that purges physical files and clears ChromaDB vector chunks and BM25 indices.

3. **Model Registry & Ping Diagnostics**:
   - Register custom LLM and Embedding models (OpenRouter & Local Ollama).
   - Real-time endpoint ping test measuring latency ($ms$) and exact vector output dimensions ($d$).

4. **Corpus-Aware Auto-Scaling**:
   - **Nano Tier** ($< 50$ Chunks): In-memory vector + BM25 keyword search ($< 150\text{ ms}$ target).
   - **Standard Tier** ($50 - 5,000$ Chunks): Dense vector (ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion (RRF) and Maximal Marginal Relevance (MMR).
   - **Enterprise Tier** ($> 5,000$ Chunks): Hierarchical vector store + HyDE + Neural Cross-Encoder Reranking.

5. **Live Control Center & Quality Telemetry**:
   - Dynamic sliders for Hybrid Weight ($\alpha$), MMR Diversity ($\lambda$), candidate pool depth, and algorithm feature toggles.
   - Real-time quality scores (*Faithfulness*, *Context Precision*, *Context Recall*) and microsecond latency waterfall spans.

---

## 🛠️ Project Structure

```
Ultimate-RAG-System/
├── core/
│   ├── adaptive_engine.py      # Dynamic corpus scaling & algorithm routing
│   ├── chat_manager.py         # Multi-session conversation management & persistence
│   ├── config_manager.py       # Live settings state, model registry & JSON storage
│   └── router.py               # OpenRouter / Ollama API gateway with fallback matrix
├── ingestion/
│   ├── multi_parser.py         # PyMuPDF, Markdown & TXT parser
│   ├── semantic_chunker.py     # Heading & paragraph-aware semantic chunker
│   └── vector_indexer.py       # ChromaDB persistent vector database manager
├── retrieval/
│   ├── bm25_engine.py          # Fast rank-bm25 in-memory indexer
│   ├── hybrid_fusion.py        # Reciprocal Rank Fusion (RRF) & MMR diversity engine
│   ├── hyde.py                 # Hypothetical document embedding generator
│   └── reranker.py             # Neural cross-encoder bridge
├── analytics/
│   ├── metrics_evaluator.py    # Faithfulness, Precision, Recall calculators
│   └── telemetry_logger.py     # Microsecond latency waterfall recorder
├── ui/
│   ├── server.py               # FastAPI ASGI web server & API endpoints
│   └── static/                 # Glassmorphic UI dashboard (HTML, CSS, JS)
├── main.py                     # Root entry point
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/DarkWolfHunter007/Ultimate-RAG-System.git
cd Ultimate-RAG-System
pip install -r requirements.txt
```

### 3. Running the Server
Launch the ASGI web server:
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📜 License & Credit

Created and maintained by **Allen MT Maliyil** ([@DarkWolfHunter007](https://github.com/DarkWolfHunter007)). Distributed under the MIT License.
