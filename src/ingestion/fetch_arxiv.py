"""
fetch_arxiv.py — arXiv Paper Ingestion for RAG Technical Assistant
==================================================================

This module is the first stage of the RAG pipeline. It crawls the arXiv
academic preprint server via its public Atom/XML API to collect research
papers on autonomous driving and related topics.

Why arXiv?
- Free, no authentication required
- Covers the latest ML/AI research before journal publication
- Papers are structured (title, abstract, authors, categories)
- Directly relevant to BMW's autonomous driving use case

Pipeline position:
    fetch_arxiv.py → chunk_documents.py → build_vectorstore.py → rag_agent.py

Output:
    data/raw/papers.json — list of paper metadata dicts

Usage:
    python src/ingestion/fetch_arxiv.py
    python src/ingestion/fetch_arxiv.py --query "sensor fusion lidar" --max 100
"""

import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import requests
import yaml
from loguru import logger


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Load YAML configuration file.

    Args:
        path: Path to the config file relative to project root.

    Returns:
        Parsed config dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_arxiv(query: str, max_results: int, start: int = 0) -> list[dict]:
    """
    Fetch papers from the arXiv Atom API for a given search query.

    The arXiv API returns results as an Atom XML feed. This function
    parses the XML and extracts structured metadata for each paper.

    Rate limit note: arXiv requests a maximum of 1 request per 3 seconds.
    The caller should sleep between queries (handled in run_ingestion).

    Args:
        query:       Search string. Searches all fields (title, abstract, etc.)
        max_results: Maximum number of results to return per query.
        start:       Offset for pagination (default 0 = first page).

    Returns:
        List of paper dicts, each containing:
            arxiv_id, title, abstract, authors, published,
            categories, pdf_url, query, source
    """
    # arXiv Atom API endpoint
    url = "http://export.arxiv.org/api/query"

    # Build query parameters
    # search_query uses field prefixes: all: searches all fields
    params = {
        "search_query": f"all:{query}",
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "relevance",      # most relevant first
        "sortOrder":    "descending",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  # raise on 4xx/5xx HTTP errors

    # Define XML namespaces used in the Atom feed
    # arXiv extends the standard Atom namespace with its own
    ns = {
        "atom":   "http://www.w3.org/2005/Atom",
        "arxiv":  "http://arxiv.org/schemas/atom",
    }

    # Parse the Atom XML response
    root    = ET.fromstring(response.text)
    entries = root.findall("atom:entry", ns)

    papers = []
    for entry in entries:
        # Extract arXiv ID from the full URL (e.g. "http://arxiv.org/abs/2301.12345v1")
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]

        # Clean title and abstract — API returns them with newlines
        title    = entry.find("atom:title",   ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

        # Published date — truncate to YYYY-MM-DD
        published = entry.find("atom:published", ns).text[:10]

        # Collect all author names
        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
        ]

        # Paper subject categories (e.g. cs.LG, cs.CV, cs.RO)
        categories = [
            c.get("term")
            for c in entry.findall("atom:category", ns)
        ]

        # Find the PDF link (arXiv provides multiple link types: abs, pdf, html)
        pdf_link = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href")
                break

        papers.append({
            "arxiv_id":   arxiv_id,
            "title":      title,
            "abstract":   abstract,
            "authors":    authors,
            "published":  published,
            "categories": categories,
            "pdf_url":    pdf_link,
            "query":      query,     # track which query produced this paper
            "source":     "arxiv",
        })

    return papers


def deduplicate(papers: list[dict]) -> list[dict]:
    """
    Remove duplicate papers by arxiv_id.

    When multiple queries return the same paper (e.g. a paper on
    "LiDAR-camera fusion" matches both "sensor fusion" and "lidar camera"
    queries), we keep only the first occurrence.

    Args:
        papers: List of paper dicts, potentially with duplicates.

    Returns:
        Deduplicated list preserving original order.
    """
    seen   = set()    # track seen arxiv IDs
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique


def run_ingestion(
    config_path:    str  = "configs/config.yaml",
    extra_query:    str  = None,
    max_override:   int  = None,
) -> list[dict]:
    """
    Run the full arXiv ingestion pipeline.

    Iterates over all configured search queries, fetches papers for each,
    deduplicates across queries, and saves the result to JSON.

    Args:
        config_path:  Path to config YAML.
        extra_query:  Optional additional query to prepend to the list.
        max_override: Override max_results_per_query from config.

    Returns:
        List of unique paper dicts.
    """
    cfg      = load_config(config_path)
    queries  = cfg["arxiv"]["search_queries"]
    max_per  = max_override or cfg["arxiv"]["max_results_per_query"]
    out_path = Path(cfg["arxiv"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepend any extra query provided via CLI
    if extra_query:
        queries = [extra_query] + queries

    all_papers = []
    for query in queries:
        logger.info(f"Fetching: '{query}' (max={max_per})")
        try:
            papers = fetch_arxiv(query, max_results=max_per)
            all_papers.extend(papers)
            logger.success(f"  Fetched {len(papers)} papers")

            # arXiv API rate limit: max 1 request per 3 seconds
            time.sleep(3)
        except Exception as e:
            logger.error(f"  Failed for query '{query}': {e}")

    # Remove papers that appeared in multiple query results
    all_papers = deduplicate(all_papers)
    logger.info(f"Total unique papers after deduplication: {len(all_papers)}")

    # Save with fetch timestamp for reproducibility
    with open(out_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "total":      len(all_papers),
            "papers":     all_papers,
        }, f, indent=2)

    logger.success(f"Saved {len(all_papers)} papers to {out_path}")
    return all_papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch arXiv papers for RAG pipeline")
    parser.add_argument("--query",  default=None, help="Extra search query to add")
    parser.add_argument("--max",    type=int, default=None, help="Override max results per query")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    run_ingestion(args.config, args.query, args.max)
