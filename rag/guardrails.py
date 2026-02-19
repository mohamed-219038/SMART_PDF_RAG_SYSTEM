from typing import List, Tuple
import numpy as np


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def check_relevance(
    question: str,
    docs,
    embeddings,
    threshold: float = 0.10,
) -> Tuple[bool, float]:
    """
    Compute average cosine similarity between the question embedding
    and the retrieved chunk embeddings.

    Returns (is_relevant, avg_score).
    If avg_score < threshold the retrieval is considered too weak
    and the system should return a "not relevant" fallback.
    """
    if not docs:
        return False, 0.0

    q_emb = embeddings.embed_query(question)

    scores = []
    for d in docs:
        d_emb = embeddings.embed_query(d.page_content)
        scores.append(_cosine_similarity(q_emb, d_emb))

    avg = float(np.mean(scores))
    return avg >= threshold, avg
