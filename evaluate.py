"""
RAGAS-style Evaluation Report for the Smart Contract Q&A Assistant.

Runs the RAG pipeline on Q&A pairs from eval_dataset.json, then computes
four RAGAS-equivalent metrics using embeddings (no LLM-as-judge needed):

  1. Faithfulness       – cosine similarity between answer & retrieved context
  2. Answer Relevancy   – cosine similarity between answer & question
  3. Context Precision   – fraction of retrieved chunks semantically close to question
  4. Context Recall      – cosine similarity between retrieved context & ground truth

All scores are 0.0 – 1.0.  Produces eval_report.json.

Usage:
    py evaluate.py <path_to_pdf_or_docx> [--dataset eval_dataset.json] [--output eval_report.json]
"""

import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List

from rag.config import load_config
from rag.pipeline import build_session, answer_question
from rag.embeddings import get_embeddings


# ----------------------------------------------------------------
# Embedding-based metrics  (no LLM-as-judge required)
# ----------------------------------------------------------------
def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def faithfulness_score(answer: str, contexts: List[str], emb) -> float:
    """How well the answer is grounded in the retrieved context."""
    if not answer.strip() or not contexts:
        return 0.0
    ans_vec = emb.embed_query(answer)
    ctx_text = " ".join(contexts)
    ctx_vec = emb.embed_query(ctx_text)
    return max(0.0, _cosine(ans_vec, ctx_vec))


def answer_relevancy_score(answer: str, question: str, emb) -> float:
    """How relevant the answer is to the question."""
    if not answer.strip() or answer.strip().lower() == "not found":
        return 0.0
    ans_vec = emb.embed_query(answer)
    q_vec = emb.embed_query(question)
    return max(0.0, _cosine(ans_vec, q_vec))


def context_precision_score(question: str, contexts: List[str], emb, threshold=0.3) -> float:
    """Fraction of retrieved chunks that are semantically relevant to the question."""
    if not contexts:
        return 0.0
    q_vec = emb.embed_query(question)
    relevant = 0
    for ctx in contexts:
        c_vec = emb.embed_query(ctx)
        if _cosine(q_vec, c_vec) >= threshold:
            relevant += 1
    return relevant / len(contexts)


def context_recall_score(ground_truth: str, contexts: List[str], emb) -> float:
    """How well the retrieved context covers the ground truth."""
    if not ground_truth.strip() or not contexts:
        return 0.0
    gt_vec = emb.embed_query(ground_truth)
    ctx_text = " ".join(contexts)
    ctx_vec = emb.embed_query(ctx_text)
    return max(0.0, _cosine(gt_vec, ctx_vec))


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _collect_rag_outputs(file_path: str, dataset: List[dict], cfg: Dict[str, Any]):
    """Run every question through the RAG pipeline and collect outputs."""
    print(f"[1/3] Building session from: {file_path}")
    session, meta = build_session(file_path, cfg)
    print(f"       Chunks: {meta['chunks']}, Chunk size: {meta['chunk_size']}")

    questions, answers, contexts, ground_truths, latencies = [], [], [], [], []

    print(f"[2/3] Running {len(dataset)} questions through the RAG pipeline ...\n")

    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item.get("ground_truth", "")

        t0 = time.time()
        answer, retrieved = answer_question(q, session, cfg)
        latency = time.time() - t0

        ctx_list = [r["text"] for r in retrieved if r.get("text", "").strip()]

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_list)
        ground_truths.append(gt)
        latencies.append(round(latency, 3))

        print(f"  Q{i+1} ({latency:.1f}s): {q}")
        short = answer[:100] + "..." if len(answer) > 100 else answer
        print(f"  A{i+1}: {short}\n")

    return questions, answers, contexts, ground_truths, latencies, meta


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def run_evaluation(
    file_path: str,
    dataset_path: str = "eval_dataset.json",
    output_path: str = "eval_report.json",
):
    cfg = load_config()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions, answers, contexts, ground_truths, latencies, meta = _collect_rag_outputs(
        file_path, dataset, cfg
    )

    # load embeddings for metric computation
    emb = get_embeddings(cfg["embeddings"]["model_name"])

    print("[3/3] Computing RAGAS metrics ...\n")

    per_question = []
    sum_faith, sum_rel, sum_prec, sum_recall = 0.0, 0.0, 0.0, 0.0

    for i in range(len(questions)):
        faith = faithfulness_score(answers[i], contexts[i], emb)
        rel   = answer_relevancy_score(answers[i], questions[i], emb)
        prec  = context_precision_score(questions[i], contexts[i], emb)
        recall = context_recall_score(ground_truths[i], contexts[i], emb)

        sum_faith += faith
        sum_rel += rel
        sum_prec += prec
        sum_recall += recall

        per_question.append({
            "question": questions[i],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "contexts_count": len(contexts[i]),
            "latency_s": latencies[i],
            "faithfulness": round(faith, 4),
            "answer_relevancy": round(rel, 4),
            "context_precision": round(prec, 4),
            "context_recall": round(recall, 4),
        })

    n = max(len(questions), 1)
    summary = {
        "total_questions": len(questions),
        "avg_faithfulness": round(sum_faith / n, 4),
        "avg_answer_relevancy": round(sum_rel / n, 4),
        "avg_context_precision": round(sum_prec / n, 4),
        "avg_context_recall": round(sum_recall / n, 4),
        "avg_latency_s": round(sum(latencies) / n, 3),
        "total_latency_s": round(sum(latencies), 3),
    }

    report = {
        "title": "RAGAS Evaluation Report",
        "document": file_path,
        "pipeline_meta": meta,
        "summary": summary,
        "results": per_question,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- print report ----
    print("=" * 60)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Document:              {file_path}")
    print(f"  Questions evaluated:   {summary['total_questions']}")
    print(f"  Avg latency:           {summary['avg_latency_s']:.2f}s")
    print("-" * 60)
    print(f"  Faithfulness:          {summary['avg_faithfulness']:.2%}")
    print(f"  Answer Relevancy:      {summary['avg_answer_relevancy']:.2%}")
    print(f"  Context Precision:     {summary['avg_context_precision']:.2%}")
    print(f"  Context Recall:        {summary['avg_context_recall']:.2%}")
    print("=" * 60)
    print(f"\n  Full report saved to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for the RAG pipeline.")
    parser.add_argument("file", help="Path to PDF or DOCX to evaluate against")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to Q&A dataset JSON")
    parser.add_argument("--output", default="eval_report.json", help="Path to save the evaluation report")
    args = parser.parse_args()

    run_evaluation(args.file, args.dataset, args.output)
