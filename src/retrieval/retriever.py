"""
retriever.py — Semantic Retrieval from Vector Store
====================================================

This module handles the retrieval half of Retrieval-Augmented Generation.
Given a natural language query, it finds the most semantically relevant
paper chunks from the ChromaDB vector store.

How retrieval works:
    1. The query is embedded using the same model used to index the documents
       (critically: must use the SAME model, otherwise embeddings are in
        incompatible vector spaces)
    2. ChromaDB performs approximate nearest-neighbour (ANN) search using
       the HNSW algorithm — finds the k closest vectors to the query vector
    3. Results are ranked by cosine similarity score (higher = more relevant)
    4. A score threshold filters out low-quality matches

Cosine similarity vs distance:
    ChromaDB returns cosine DISTANCE (0=identical, 2=opposite).
    We convert to similarity: similarity = 1 - distance
    Score range: 0.0 (unrelated) to 1.0 (identical)

Pipeline position:
    fetch_arxiv.py → chunk_documents.py → build_vectorstore.py
    → retriever.py → rag_agent.py

Usage:
    from src.retrieval.retriever import Retriever
    retriever = Retriever()
    chunks = retriever.retrieve("How does sensor fusion work?")
    context = retriever.format_context(chunks)
"""

import yaml
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to config YAML relative to project root.

    Returns:
        Parsed config dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


class Retriever:
    """
    Semantic retriever that queries a ChromaDB vector store.

    Embeds incoming queries using sentence-transformers and retrieves
    the most relevant document chunks ranked by cosine similarity.

    Attributes:
        top_k (int):      Default number of results to return.
        threshold (float): Minimum similarity score to include a result.
        model:            SentenceTransformer instance for query embedding.
        collection:       ChromaDB collection containing indexed chunks.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialise the retriever by loading the vector store and embedding model.

        Both the model and the ChromaDB collection are loaded once at startup
        and reused for all queries — avoids expensive reloading per request.

        Args:
            config_path: Path to config YAML relative to project root.
        """
        cfg = load_config(config_path)

        # Retrieval hyperparameters from config
        self.top_k     = cfg["retrieval"]["top_k"]
        self.threshold = cfg["retrieval"]["score_threshold"]

        store_path = Path(cfg["embeddings"]["vector_store_path"])
        collection = cfg["embeddings"]["collection_name"]
        model_name = cfg["embeddings"]["model"]

        # Load embedding model — MUST be the same model used in build_vectorstore.py
        # Using a different model would produce incompatible vector spaces
        self.model = SentenceTransformer(model_name)

        # Connect to the persistent ChromaDB store
        client = chromadb.PersistentClient(
            path=str(store_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get the existing collection — raises if not built yet
        self.collection = client.get_collection(collection)
        logger.info(
            f"Retriever ready — {self.collection.count()} documents indexed "
            f"| model={model_name}"
        )

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve the top-k most semantically similar chunks for a query.

        Process:
            1. Embed the query into a 384-dim vector
            2. Query ChromaDB for nearest neighbours (cosine distance)
            3. Convert distances to similarity scores
            4. Filter by score threshold
            5. Sort by descending similarity

        Args:
            query:  Natural language question or search string.
            top_k:  Number of results to return. Defaults to config value.

        Returns:
            List of chunk dicts sorted by relevance score, each containing:
                text, score, title, authors, published, arxiv_id, pdf_url
        """
        k = top_k or self.top_k

        # Embed the query — same model as indexing, returns shape (1, 384)
        # .tolist() converts numpy array to Python list for ChromaDB
        emb = self.model.encode([query]).tolist()

        # Query ChromaDB — returns distances, documents, and metadata
        # n_results: how many nearest neighbours to retrieve
        results = self.collection.query(
            query_embeddings = emb,
            n_results        = k,
            include          = ["documents", "metadatas", "distances"]
        )

        # Unpack results — ChromaDB wraps in outer list for batch queries
        # results["documents"][0] = list of chunk texts for our single query
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Convert cosine distance to similarity score
            # ChromaDB cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            score = 1 - dist

            # Skip low-relevance chunks below the configured threshold
            if score >= self.threshold:
                chunks.append({
                    "text":      doc,
                    "score":     round(score, 4),
                    "title":     meta.get("title", ""),
                    "authors":   meta.get("authors", ""),
                    "published": meta.get("published", ""),
                    "arxiv_id":  meta.get("arxiv_id", ""),
                    "pdf_url":   meta.get("pdf_url", ""),
                })

        # Sort by descending similarity (highest relevance first)
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks

    def format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a structured context string for the LLM.

        The formatted context includes:
        - Numbered citation markers [1], [2], ... for the LLM to reference
        - Paper title, date, and author information
        - Relevance score (shows the LLM how confident the retrieval was)
        - The chunk text itself
        - A direct arXiv URL for each source

        Args:
            chunks: List of chunk dicts from retrieve().

        Returns:
            Multi-section string formatted for LLM consumption.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {chunk['title']} ({chunk['published']})\n"
                f"Authors: {chunk['authors']}\n"
                f"Relevance: {chunk['score']:.2%}\n\n"
                f"{chunk['text']}\n"
                f"Source: https://arxiv.org/abs/{chunk['arxiv_id']}"
            )

        # Separate each source with a horizontal rule for LLM clarity
        return "\n\n---\n\n".join(context_parts)
