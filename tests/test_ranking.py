import numpy as np

from app.ranking import rank_by_similarity


def test_orders_by_cosine_similarity():
    q = np.array([1.0, 0.0])
    cands = np.array([[0.0, 1.0], [1.0, 0.1], [-1.0, 0.0]])
    ranked = rank_by_similarity(q, cands, top_k=3)
    assert [i for i, _ in ranked] == [1, 0, 2]
    assert ranked[0][1] > ranked[1][1] > ranked[2][1]


def test_top_k_truncates_and_ignores_scale():
    q = np.array([2.0, 0.0])
    cands = np.array([[100.0, 0.0], [0.0, 0.5], [1.0, 1.0]])
    ranked = rank_by_similarity(q, cands, top_k=2)
    assert [i for i, _ in ranked] == [0, 2]
    assert abs(ranked[0][1] - 1.0) < 1e-6


def test_empty_and_zero_vectors():
    assert rank_by_similarity(np.array([1.0, 0.0]), np.zeros((0, 2)), 5) == []
    ranked = rank_by_similarity(np.array([1.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), 2)
    assert ranked[0][0] == 1 and ranked[1][1] == 0.0
