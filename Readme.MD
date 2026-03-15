# RAG Technical Assistant — Autonomous Driving Research

**Retrieval-Augmented Generation pipeline for semantic Q&A over autonomous driving research papers — arXiv ingestion, ChromaDB vector store, sentence-transformers embeddings, Ollama (local) / Mistral-7B (HF Spaces), FastAPI + Gradio**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-yellow)](https://gradio.app/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue)](https://mlflow.org/)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-orange)](https://huggingface.co/spaces/danielamissah/rag-technical-assistant)

---

## Live Demo

**🤗 Hugging Face Spaces:** [huggingface.co/spaces/danielamissah/rag-technical-assistant](https://huggingface.co/spaces/danielamissah/rag-technical-assistant)

Ask questions about autonomous driving research — the system retrieves relevant papers and generates cited, technically precise answers in real time.

---

## Overview

A production-grade Retrieval-Augmented Generation (RAG) pipeline that enables semantic question answering over a corpus of 389 arXiv research papers on autonomous driving and automotive AI. Engineers can ask natural language questions and receive answers grounded in peer-reviewed literature, with citations and direct links to source PDFs.

The pipeline is fully modular: arXiv ingestion → text chunking → embedding → vector store → semantic retrieval → LLM generation → FastAPI endpoint → Gradio UI.

---

## Architecture

```
arXiv API (5 queries × 100 papers)
        │
        ▼
┌─────────────────────────┐
│   Ingestion              │  fetch_arxiv.py — Atom XML parsing, deduplication
│   + Chunking             │  chunk_documents.py — 800-word overlapping windows
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Embeddings             │  sentence-transformers/all-MiniLM-L6-v2
│   + Vector Store         │  ChromaDB (persistent, cosine similarity)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Retrieval              │  Top-k semantic search (k=5, threshold=0.3)
│   + Context Building     │  Formatted with title, authors, relevance score
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   LLM Generation         │  Local: Ollama (llama3.2:3b)
│                          │  Deployed: Mistral-7B (HF Inference API)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Serving                │  FastAPI REST API (/ask, /ask/batch, /stats)
│   + UI                   │  Gradio interface (HF Spaces)
└─────────────────────────┘
```

---

## Knowledge Base

**389 arXiv papers** crawled across 5 search queries:

| Query                                             | Papers |
| ------------------------------------------------- | ------ |
| autonomous driving sensor fusion                  | ~100   |
| lidar camera perception autonomous vehicle        | ~100   |
| deep learning object detection autonomous driving | ~100   |
| transformer autonomous driving                    | ~100   |
| anomaly detection automotive sensor               | ~100   |

Papers deduplicated → **389 unique papers** → **chunked into overlapping 800-word segments** →  **embedded and stored in ChromaDB** .

---

## Evaluation Results

Evaluated across 8 technical questions on autonomous driving topics:

| Metric                  | Run 1 (183 docs) | Run 2 (389 docs)                |
| ----------------------- | ---------------- | ------------------------------- |
| Documents indexed       | 183              | **389**                   |
| Avg top retrieval score | 0.610            | **0.622**                 |
| Avg min retrieval score | 0.517            | **0.556**                 |
| Zero result rate        | 0.0%             | **0.0%**                  |
| Answer quality          | Refused most     | **Substantive answers**✅ |
| Avg latency             | 4.1s             | 10.5s                           |

All 8 evaluation questions answered with substantive, cited responses after corpus expansion and prompt tuning. Both runs tracked in MLflow with full parameter and metric logging.

---

## Sample Q&A

**Q: What are the main challenges in LiDAR and camera sensor fusion for autonomous driving?**

> The main challenges in LiDAR and camera sensor fusion for autonomous driving are:
>
> 1. **Robustness to environmental conditions** — LiDAR and camera sensors have different failure modes in rain, fog, and direct sunlight (Chen et al., 2023 —  *Lift-Attend-Splat* )
> 2. **Spatial calibration** — aligning point clouds with image pixels requires precise extrinsic calibration...
> 3. **Computational cost** — real-time fusion of dense LiDAR point clouds with high-resolution images is computationally intensive...
>
> *Sources: [2312.14919](https://arxiv.org/abs/2312.14919) · [2310.06008](https://arxiv.org/abs/2310.06008) · [2212.07155](https://arxiv.org/abs/2212.07155)*

---

## API Endpoints

| Method | Endpoint       | Description                 |
| ------ | -------------- | --------------------------- |
| GET    | `/health`    | Health check + agent status |
| POST   | `/ask`       | Ask a single question       |
| POST   | `/ask/batch` | Ask multiple questions      |
| GET    | `/stats`     | Vector store + model info   |
| GET    | `/docs`      | Swagger UI                  |

### Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does LiDAR-camera fusion work in autonomous vehicles?"}'
```

```json
{
  "question": "How does LiDAR-camera fusion work...",
  "answer": "LiDAR-camera fusion combines...",
  "sources": [
    {
      "title": "Lift-Attend-Splat: Bird's-eye-view camera-lidar fusion using transformers",
      "authors": "James Gunn, Zygmunt Lenyk, Anuj Sharma",
      "published": "2023-12-22",
      "arxiv_id": "2312.14919v3",
      "score": 0.6526
    }
  ],
  "retrieval_scores": [0.6526, 0.6274, 0.610, 0.5949, 0.588],
  "latency_ms": 10450,
  "chunks_used": 5
}
```

---

## Project Structure

```
rag-technical-assistant/
├── app.py                              ← Gradio UI (HF Spaces entry point)
├── configs/config.yaml                 ← All settings
├── src/
│   ├── ingestion/
│   │   ├── fetch_arxiv.py              ← arXiv API crawler
│   │   └── chunk_documents.py          ← Overlapping text chunker
│   ├── embeddings/
│   │   └── build_vectorstore.py        ← Sentence-transformers + ChromaDB
│   ├── retrieval/
│   │   └── retriever.py                ← Semantic search + context builder
│   ├── agent/
│   │   └── rag_agent.py                ← RAG logic (Ollama / HF Inference API)
│   ├── serving/
│   │   └── app.py                      ← FastAPI REST endpoints
│   └── evaluation/
│       └── evaluate_rag.py             ← Retrieval + latency metrics + MLflow
├── data/
│   └── processed/
│       ├── chunks.json                 ← Chunked paper text
│       └── vectorstore/                ← ChromaDB persistent store (committed)
├── Makefile
└── requirements.txt
```

---

## Setup

### Local (with Ollama)

```bash
git clone https://github.com/danielamissah/rag-technical-assistant.git
cd rag-technical-assistant
pip install -r requirements.txt

# Install and start Ollama
brew install ollama
ollama pull llama3.2:3b
ollama serve   # keep running in separate terminal
```

### Running the Pipeline

```bash
# Vector store already built — skip to serving
# Or rebuild from scratch:
make pipeline   # fetch → chunk → embed (~10 min)

# Start API server
make serve      # http://localhost:8000/docs

# Start Gradio UI locally
python app.py   # http://localhost:7860

# Evaluate + MLflow tracking
make mlflow     # http://localhost:5001  (start first)
make evaluate
```

### HF Spaces Deployment

```bash
# Push to HF Spaces
git remote add hf https://huggingface.co/spaces/danielamissah/rag-technical-assistant
git push hf main
```

Set `HF_TOKEN` as a Space secret — the agent auto-detects it and switches from Ollama to the HF Inference API.

---

## Key Design Decisions

**Why RAG over fine-tuning?**
Fine-tuning requires labelled Q&A pairs and compute. RAG gives grounded, up-to-date answers from any document corpus without retraining — and every answer is traceable to a source.

**Why ChromaDB?**
Persistent, file-based, no server required. The 6.6MB vector store commits directly to the repo, so HF Spaces loads it instantly without re-embedding on startup.

**Why sentence-transformers/all-MiniLM-L6-v2?**
Fast, lightweight (80MB), excellent retrieval performance on technical text, runs on CPU. No GPU required locally or on HF Spaces free tier.

**Why dual LLM support (Ollama / HF)?**
Ollama for local development — no API costs, fast iteration. HF Inference API for deployment — free tier, no server management, publicly accessible.

**Why 800-word chunks with 100-word overlap?**
arXiv abstracts are dense. 800 words captures enough context for the LLM to reason about, while overlap prevents important information being split across chunk boundaries.

---

## Technologies

sentence-transformers · ChromaDB · Ollama · Mistral-7B · FastAPI · Gradio · MLflow · arXiv API · HuggingFace Spaces · Python

---

## MLflow Tracking

All evaluation runs tracked at `http://localhost:5001`:

* Parameters: embedding model, LLM model, top_k, chunk_size, n_test_questions
* Metrics: avg_top_score, avg_min_score, zero_result_rate, avg_latency_ms, min/max latency
