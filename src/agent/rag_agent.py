"""
RAG agent — retrieves relevant paper chunks and generates answers
using a local Ollama LLM (free, no API key required).

Usage:
    from src.agent.rag_agent import RAGAgent
    agent = RAGAgent()
    response = agent.ask("What are the main challenges in LiDAR-camera fusion?")
    print(response["answer"])
"""

import time
import yaml
import requests
from loguru import logger
from src.retrieval.retriever import Retriever


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class RAGAgent:
    def __init__(self, config_path="configs/config.yaml"):
        cfg = load_config(config_path)

        self.retriever  = Retriever(config_path)
        self.model      = cfg["llm"]["model"]
        self.max_tokens = cfg["llm"]["max_tokens"]
        self.system     = cfg["llm"]["system_prompt"]
        self.ollama_url = cfg["llm"].get("ollama_url", "http://localhost:11434")

        try:
            resp   = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama running — available models: {models}")
            if not any(self.model in m for m in models):
                logger.warning(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}. Run: ollama serve")

        logger.info(f"RAG Agent ready — model={self.model}")

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self.model,
            "prompt": f"{self.system}\n\n{prompt}",
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": 0.1,
            }
        }
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def ask(self, question: str, top_k: int = None) -> dict:
        t0      = time.time()
        chunks  = self.retriever.retrieve(question, top_k)
        context = self.retriever.format_context(chunks)

        if not chunks:
            return {
                "question": question,
                "answer":   "No relevant papers found in the knowledge base for this question.",
                "sources":  [], "retrieval_scores": [],
                "latency_ms": int((time.time() - t0) * 1000),
                "chunks_used": 0, "model": self.model,
            }

        prompt = (
            f"Using the following research paper excerpts, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"Provide a clear, technically precise answer with citations."
        )

        answer  = self._call_ollama(prompt)
        latency = int((time.time() - t0) * 1000)

        sources = [{
            "title": c["title"], "authors": c["authors"],
            "published": c["published"], "arxiv_id": c["arxiv_id"],
            "pdf_url": c["pdf_url"], "score": c["score"],
        } for c in chunks]

        logger.info(f"Q: {question[:60]}... | chunks={len(chunks)} | latency={latency}ms")

        return {
            "question": question, "answer": answer, "sources": sources,
            "retrieval_scores": [c["score"] for c in chunks],
            "latency_ms": latency, "model": self.model, "chunks_used": len(chunks),
        }

    def ask_batch(self, questions: list) -> list:
        return [self.ask(q) for q in questions]
