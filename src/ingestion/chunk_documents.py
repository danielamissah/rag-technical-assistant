"""
chunk_documents.py — Text Chunking for RAG Pipeline
====================================================

This module is the second stage of the RAG pipeline. It takes raw paper
metadata (title + abstract) and splits it into overlapping text chunks
suitable for embedding and retrieval.

Why chunk documents?
- Embedding models have a maximum token limit (typically 256-512 tokens)
- Shorter, focused chunks improve retrieval precision — a chunk about
  "LiDAR point cloud processing" will rank higher for relevant queries
  than a chunk that mixes multiple topics
- Overlapping chunks (using chunk_overlap) prevent context loss at
  chunk boundaries

Chunking strategy:
- Unit: words (not tokens) for simplicity and language-independence
- Window: 800 words per chunk, 100 word overlap
- Each chunk includes paper title for context even in the middle of a doc

Pipeline position:
    fetch_arxiv.py → chunk_documents.py → build_vectorstore.py → rag_agent.py

Input:  data/raw/papers.json
Output: data/processed/chunks.json

Usage:
    python src/ingestion/chunk_documents.py
"""

import json
import argparse
from pathlib import Path

import yaml
from loguru import logger


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


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a text string into overlapping word-based chunks.

    This is a sliding window approach:
    - Window slides by (chunk_size - overlap) words each step
    - Overlap ensures that sentences near chunk boundaries are
      covered in at least two chunks, preventing retrieval gaps

    Example with chunk_size=5, overlap=2, text="a b c d e f g h":
        chunk 1: "a b c d e"
        chunk 2: "d e f g h"   (starts 3 words later: 5-2=3)

    Args:
        text:       Input text string to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of words to repeat between adjacent chunks.

    Returns:
        List of text chunk strings.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        # Take up to chunk_size words from current position
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Stop if we've reached the end of the text
        if end == len(words):
            break

        # Advance the window by (chunk_size - overlap) positions
        # so the next chunk shares `overlap` words with this one
        start += chunk_size - overlap

    return chunks


def build_chunks(papers: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Build enriched chunk dicts from a list of paper metadata dicts.

    For each paper, we concatenate the title and abstract into a single
    text, then split it into overlapping chunks. Each chunk is stored
    with its source paper's metadata so the RAG agent can cite sources.

    Args:
        papers:     List of paper dicts from fetch_arxiv.py.
        chunk_size: Words per chunk.
        overlap:    Overlap words between adjacent chunks.

    Returns:
        List of chunk dicts, each containing:
            chunk_id, arxiv_id, title, authors, published,
            pdf_url, categories, query, text, chunk_idx, total_chunks
    """
    chunks = []

    for paper in papers:
        # Combine title and abstract into a single searchable text
        # Prepending "Title:" helps the embedding model understand structure
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        text_chunks = chunk_text(full_text, chunk_size, overlap)

        for i, chunk in enumerate(text_chunks):
            chunks.append({
                # Unique ID combining arxiv ID and chunk index
                "chunk_id":     f"{paper['arxiv_id']}_chunk_{i}",
                "arxiv_id":     paper["arxiv_id"],
                "title":        paper["title"],
                # Limit to first 3 authors for metadata brevity
                "authors":      paper["authors"][:3],
                "published":    paper["published"],
                "pdf_url":      paper.get("pdf_url", ""),
                "categories":   paper.get("categories", []),
                "query":        paper.get("query", ""),
                "text":         chunk,
                # Track position within the original document
                "chunk_idx":    i,
                "total_chunks": len(text_chunks),
            })

    return chunks


def run_chunking(config_path: str = "configs/config.yaml") -> list[dict]:
    """
    Run the full chunking pipeline.

    Loads raw papers, builds chunks with metadata, and saves to JSON.

    Args:
        config_path: Path to config YAML.

    Returns:
        List of chunk dicts ready for embedding.
    """
    cfg        = load_config(config_path)
    raw_path   = Path(cfg["arxiv"]["raw_path"])
    out_path   = Path(cfg["chunking"]["processed_path"])
    chunk_size = cfg["chunking"]["chunk_size"]
    overlap    = cfg["chunking"]["chunk_overlap"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw papers from ingestion step
    with open(raw_path) as f:
        data = json.load(f)

    papers = data["papers"]
    logger.info(f"Chunking {len(papers)} papers "
                f"(chunk_size={chunk_size}, overlap={overlap})")

    chunks = build_chunks(papers, chunk_size, overlap)

    logger.success(f"Created {len(chunks)} chunks from {len(papers)} papers "
                   f"(avg {len(chunks)/len(papers):.1f} chunks/paper)")

    # Save chunks with pipeline provenance metadata
    with open(out_path, "w") as f:
        json.dump({
            "total_papers": len(papers),
            "total_chunks": len(chunks),
            "chunk_size":   chunk_size,
            "overlap":      overlap,
            "chunks":       chunks,
        }, f, indent=2)

    logger.success(f"Saved to {out_path}")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents for RAG pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_chunking(args.config)
