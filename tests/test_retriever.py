"""Retriever tests. They use the real index (data/index/), so they are skipped if it has not been built."""
import pytest

from retriever import INDEX_DIR, Retriever, recall_at_k

needs_index = pytest.mark.skipif(not (INDEX_DIR / "govt.faiss").exists(), reason="index not built")


@pytest.fixture(scope="module")
def retriever():
    return Retriever.load()


@needs_index
@pytest.mark.parametrize("k", [1, 4, 10])
def test_returns_exactly_k_passages(retriever, k):
    results = retriever.search("How do I know a government website is secure?", k)
    assert len(results) == k
    assert len({p["id"] for p in results}) == k  # no duplicates
    assert all(p["text"] for p in results)


@needs_index
def test_results_are_sorted_best_first(retriever):
    scores = [p["score"] for p in retriever.search("child support order", 10)]
    assert scores == sorted(scores, reverse=True)


@needs_index
def test_top4_is_prefix_of_top10(retriever):
    """Top-k filtering only cuts the list: k=4 must be the first 4 of k=10."""
    q = "What should I do if my garbage collection is missed?"
    assert [p["id"] for p in retriever.search(q, 4)] == [p["id"] for p in retriever.search(q, 10)][:4]


@needs_index
def test_rerank_returns_k_passages_from_the_candidates(retriever):
    q = "how do I recycle paint?"
    candidates = {p["id"] for p in retriever.search(q, 20)}
    reranked = retriever.search(q, 4, rerank_candidates=20)
    assert len(reranked) == 4
    assert {p["id"] for p in reranked} <= candidates  # reranking only re-orders, never adds passages
    scores = [p["score"] for p in reranked]
    assert scores == sorted(scores, reverse=True)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "x"]) == 0.5
    assert recall_at_k(["a"], ["a"]) == 1.0
    assert recall_at_k(["b"], ["a"]) == 0.0
    assert recall_at_k(["a"], []) is None  # undefined without gold passages
