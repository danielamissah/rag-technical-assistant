"""
app.py — FastAPI REST API for RAG Technical Assistant
=====================================================

This module exposes the RAG agent as a production-ready REST API using
FastAPI. It provides endpoints for single and batch question answering,
health checks, and vector store statistics.

Why FastAPI?
- Automatic OpenAPI/Swagger docs at /docs
- Pydantic validation for request/response schemas
- Async support for high-throughput serving
- Type hints throughout for IDE support and documentation

Endpoints:
    GET  /health      — Liveness check (is the API running?)
    POST /ask         — Answer a single question
    POST /ask/batch   — Answer multiple questions
    GET  /stats       — Vector store and model information

The RAG agent is loaded once at startup (via lifespan context) and
shared across all requests — avoids expensive reloading per request.

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.agent.rag_agent import RAGAgent


# ── Application lifecycle ──────────────────────────────────────────────────

# Global agent instance — loaded once at startup, reused for all requests
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.

    Startup: Load the RAG agent (connects to ChromaDB, loads embedding model)
    Shutdown: Log shutdown message (ChromaDB handles its own cleanup)

    Using lifespan instead of @app.on_event("startup") is the modern
    FastAPI approach (on_event is deprecated in newer versions).
    """
    global agent
    logger.info("Starting RAG Technical Assistant API...")
    logger.info("Loading RAG agent (embedding model + vector store)...")

    # This takes a few seconds — loads SentenceTransformer + ChromaDB
    agent = RAGAgent()

    logger.success("RAG agent ready — API accepting requests")
    yield  # application runs here

    logger.info("Shutting down RAG Technical Assistant API")


# Initialise FastAPI app with metadata for Swagger docs
app = FastAPI(
    title       = "RAG Technical Assistant",
    description = (
        "Semantic search and Q&A over autonomous driving research papers. "
        "Powered by arXiv papers, ChromaDB retrieval, and Claude generation."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS middleware — allows browser-based frontends (e.g. Streamlit, React)
# to call this API without cross-origin errors
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # restrict to specific origins in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """
    Request body for single question endpoint.

    Attributes:
        question: Natural language question about autonomous driving.
        top_k:    Override default number of retrieved chunks (optional).
    """
    question: str
    top_k:    Optional[int] = None


class BatchRequest(BaseModel):
    """
    Request body for batch question endpoint.

    Attributes:
        questions: List of questions to answer sequentially.
        top_k:     Override default number of retrieved chunks (optional).
    """
    questions: list[str]
    top_k:     Optional[int] = None


class AnswerResponse(BaseModel):
    """
    Response schema for a single answered question.

    Attributes:
        question:          The original question.
        answer:            Generated answer with inline citations.
        sources:           List of source paper metadata dicts.
        retrieval_scores:  Cosine similarity scores for each retrieved chunk.
        latency_ms:        End-to-end latency in milliseconds.
        chunks_used:       Number of paper chunks passed to the LLM.
    """
    question:         str
    answer:           str
    sources:          list
    retrieval_scores: list
    latency_ms:       int
    chunks_used:      int


# ── API endpoints ──────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    """
    Liveness check endpoint.

    Returns the API status and whether the agent has been loaded.
    Use this to verify the service is running before sending questions.
    """
    return {
        "status":       "healthy",
        "agent_loaded": agent is not None,
        "timestamp":    time.time(),
    }


@app.post("/ask", response_model=AnswerResponse, summary="Ask a question")
def ask(request: QuestionRequest):
    """
    Answer a single question using the RAG pipeline.

    Retrieves relevant paper chunks from the vector store and generates
    a grounded answer with citations using Claude.

    Returns a structured response including the answer, source papers,
    retrieval scores, and end-to-end latency.
    """
    # Guard: agent must be loaded (should always be true after startup)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded yet")

    # Validate: reject empty questions before calling the expensive pipeline
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = agent.ask(request.question, request.top_k)
    return result


@app.post("/ask/batch", summary="Answer multiple questions")
def ask_batch(request: BatchRequest):
    """
    Answer multiple questions in a single request.

    Questions are processed sequentially. Useful for evaluation runs
    or pre-computing answers for a set of known queries.

    Returns a list of result dicts and the total count.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded yet")

    if not request.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")

    results = agent.ask_batch(request.questions)
    return {
        "results": results,
        "total":   len(results),
    }


@app.get("/stats", summary="Vector store statistics")
def stats():
    """
    Return information about the loaded vector store and models.

    Useful for verifying the pipeline was built correctly — shows how
    many documents are indexed and which models are active.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not loaded yet")

    return {
        "documents_indexed": agent.retriever.collection.count(),
        "embedding_dim":     agent.retriever.model.get_sentence_embedding_dimension(),
        "llm_model":         agent.model,
    }
