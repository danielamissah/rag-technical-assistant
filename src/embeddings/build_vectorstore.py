"""
build_vectorstore.py — Embedding Generation and Vector Store Construction
=========================================================================

This module is the third stage of the RAG pipeline. It takes the chunked
documents and converts each chunk into a dense vector embedding, then
stores all embeddings in a ChromaDB vector database for fast similarity
search at query time.

How embeddings work:
- A sentence-transformer model maps each text chunk to a fixed-size
  vector (384 dimensions for all-MiniLM-L6-v2)
- Semantically similar texts produce vectors that are close together
  in vector space (measured by cosine similarity)
- At query time, the query is embedded and the nearest chunk vectors
  are retrieved — these are the most semantically relevant chunks

Why ChromaDB?
- Persistent local vector store (no cloud required)
- Fast approximate nearest-neighbour search via HNSW index
- Stores document text and metadata alongside vectors
- Easy to rebuild from scratch when new papers are added

Why all-MiniLM-L6-v2?
- Small (80MB) and fast — good for local development
- Strong retrieval performance on technical text
- 384-dimensional embeddings — good balance of accuracy and speed

Pipeline position:
    fetch_arxiv.py → chunk_documents.py → build_vectorstore.py → rag_agent.py

Input:  data/processed/chunks.json
Output: data/processed/vectorstore/  (ChromaDB persistent directory)

Usage:
    python src/embeddings/build_vectorstore.py
"""

import json
import argparse
from pathlib import Path

import yaml
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


def build_vectorstore(config_path: str = "configs/config.yaml"):
    """
    Build and persist a ChromaDB vector store from chunked documents.

    Steps:
        1. Load chunks from JSON
        2. Load the sentence-transformer embedding model
        3. Embed all chunks in batches (to avoid memory issues)
        4. Create a ChromaDB collection with HNSW cosine index
        5. Insert all embeddings with metadata

    Args:
        config_path: Path to config YAML.

    Returns:
        The populated ChromaDB collection object.
    """
    cfg         = load_config(config_path)
    chunks_path = Path(cfg["chunking"]["processed_path"])
    store_path  = Path(cfg["embeddings"]["vector_store_path"])
    model_name  = cfg["embeddings"]["model"]
    collection  = cfg["embeddings"]["collection_name"]
    store_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load chunks ────────────────────────────────────────────────
    with open(chunks_path) as f:
        data = json.load(f)
    chunks = data["chunks"]
    logger.info(f"Loaded {len(chunks)} chunks for embedding")

    # ── Step 2: Load embedding model ───────────────────────────────────────
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Step 3: Embed in batches ───────────────────────────────────────────
    # Processing in batches avoids loading all embeddings into memory at once
    # and provides progress visibility for large corpora
    batch_size = 64
    texts      = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # encode() returns a numpy array of shape (batch_size, embedding_dim)
        embs = model.encode(batch, show_progress_bar=False)

        # Convert to Python lists for JSON/ChromaDB compatibility
        embeddings.extend(embs.tolist())

        # Log progress every 500 chunks
        if i % 500 == 0:
            logger.info(f"  Embedded {i}/{len(texts)} chunks...")

    logger.success(f"Generated {len(embeddings)} embeddings "
                   f"(dim={len(embeddings[0])})")

    # ── Step 4: Create ChromaDB collection ────────────────────────────────
    # PersistentClient saves to disk — survives restarts
    client = chromadb.PersistentClient(
        path=str(store_path),
        settings=Settings(anonymized_telemetry=False)  # disable telemetry
    )

    # Delete existing collection to rebuild from scratch
    # This ensures the store is always in sync with the latest chunks
    try:
        client.delete_collection(collection)
        logger.info(f"Deleted existing collection '{collection}' for rebuild")
    except Exception:
        pass  # collection didn't exist yet

    # Create collection with cosine similarity metric
    # Cosine similarity is better than L2 for text embeddings
    col = client.create_collection(
        name     = collection,
        metadata = {"hnsw:space": "cosine"}  # use cosine distance for HNSW index
    )

    # ── Step 5: Insert embeddings with metadata ────────────────────────────
    # ChromaDB stores: embedding vectors, raw document text, and metadata
    # Metadata is used to display citations in the RAG response
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embs   = embeddings[i:i + batch_size]

        col.add(
            # Unique ID for each chunk — required by ChromaDB
            ids        = [c["chunk_id"] for c in batch_chunks],
            # The actual embedding vectors
            embeddings = batch_embs,
            # Raw text stored alongside embeddings for retrieval
            documents  = [c["text"] for c in batch_chunks],
            # Metadata for citation and filtering
            metadatas  = [{
                "arxiv_id":  c["arxiv_id"],
                "title":     c["title"],
                # Join authors list into a string for ChromaDB compatibility
                "authors":   ", ".join(c["authors"]),
                "published": c["published"],
                "pdf_url":   c["pdf_url"],
                "chunk_idx": c["chunk_idx"],
            } for c in batch_chunks]
        )

    final_count = col.count()
    logger.success(f"Vector store ready: {final_count} documents in '{collection}'")
    logger.info(f"Persisted to: {store_path}")
    return col


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_vectorstore(args.config)
