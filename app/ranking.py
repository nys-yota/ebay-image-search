from __future__ import annotations

from typing import List, Tuple

import numpy as np


def rank_by_similarity(query_vec: np.ndarray, candidate_vecs: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """Return (candidate_index, cosine_similarity) pairs, best first.

    Vectors need not be normalized; zero vectors score 0.
    """
    if len(candidate_vecs) == 0:
        return []
    q = query_vec / (np.linalg.norm(query_vec) or 1.0)
    norms = np.linalg.norm(candidate_vecs, axis=1)
    norms[norms == 0] = 1.0
    scores = (candidate_vecs / norms[:, None]) @ q
    order = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in order]
