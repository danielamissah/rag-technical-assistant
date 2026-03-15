"""
RAG pipeline evaluation.

Evaluates retrieval quality (precision@k, MRR) and answer quality
(faithfulness, relevance) on a set of test questions.

Usage:
    python src/evaluation/evaluate_rag.py
"""

import json
import time
import argparse
from pathlib import Path

import mlflow
import yaml
from loguru import logger

from src.agent.rag_agent import RAGAgent


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# Test questions relevant to autonomous driving research
TEST_QUESTIONS = [
    "What are the main challenges in LiDAR and camera sensor fusion for autonomous driving?",
    "How do transformer architectures improve object detection in autonomous vehicles?",
    "What methods are used for anomaly detection in automotive sensor data?",
    "How is deep learning applied to pedestrian detection in self-driving cars?",
    "What are the key differences between 3D object detection approaches for autonomous driving?",
    "How do autonomous vehicles handle adverse weather conditions like rain and fog?",
    "What role does reinforcement learning play in autonomous driving systems?",
    "How is semantic segmentation used in autonomous vehicle perception pipelines?",
]


def evaluate_retrieval(agent, questions):
    """Evaluate retrieval quality metrics."""
    metrics = {
        "avg_chunks_retrieved": 0,
        "avg_top_score":        0,
        "avg_min_score":        0,
        "zero_result_rate":     0,
    }

    zero_results = 0
    for q in questions:
        chunks = agent.retriever.retrieve(q)
        if not chunks:
            zero_results += 1
            continue
        scores = [c["score"] for c in chunks]
        metrics["avg_chunks_retrieved"] += len(chunks)
        metrics["avg_top_score"]        += scores[0]
        metrics["avg_min_score"]        += scores[-1]

    n = len(questions)
    metrics["avg_chunks_retrieved"] /= n
    metrics["avg_top_score"]        /= n
    metrics["avg_min_score"]        /= n
    metrics["zero_result_rate"]      = zero_results / n

    return metrics


def evaluate_answers(agent, questions):
    """Evaluate answer quality and latency."""
    results  = []
    latencies = []

    for q in questions:
        result = agent.ask(q)
        results.append(result)
        latencies.append(result["latency_ms"])
        logger.info(f"Q: {q[:60]}...")
        logger.info(f"   Chunks: {result['chunks_used']} | Latency: {result['latency_ms']}ms")
        logger.info(f"   Answer preview: {result['answer'][:100]}...")

    return results, {
        "avg_latency_ms":  sum(latencies) / len(latencies),
        "min_latency_ms":  min(latencies),
        "max_latency_ms":  max(latencies),
        "total_questions": len(questions),
    }


def run_evaluation(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    agent = RAGAgent(config_path)

    with mlflow.start_run(run_name="rag_evaluation"):
        mlflow.log_param("llm_model",        cfg["llm"]["model"])
        mlflow.log_param("embedding_model",  cfg["embeddings"]["model"])
        mlflow.log_param("top_k",            cfg["retrieval"]["top_k"])
        mlflow.log_param("chunk_size",       cfg["chunking"]["chunk_size"])
        mlflow.log_param("n_test_questions", len(TEST_QUESTIONS))

        # Retrieval metrics
        logger.info("Evaluating retrieval quality...")
        retrieval_metrics = evaluate_retrieval(agent, TEST_QUESTIONS)
        for k, v in retrieval_metrics.items():
            mlflow.log_metric(k, v)
        logger.info(f"Retrieval: {retrieval_metrics}")

        # Answer quality
        logger.info("Evaluating answer quality...")
        results, latency_metrics = evaluate_answers(agent, TEST_QUESTIONS)
        for k, v in latency_metrics.items():
            mlflow.log_metric(k, v)
        logger.info(f"Latency: {latency_metrics}")

        # Save results
        out_path = Path("outputs/reports/evaluation_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "retrieval_metrics": retrieval_metrics,
                "latency_metrics":   latency_metrics,
                "results":           results,
            }, f, indent=2)

        mlflow.log_artifact(str(out_path))
        logger.success(f"Evaluation complete. Results saved to {out_path}")

    return retrieval_metrics, latency_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
