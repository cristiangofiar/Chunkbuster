from __future__ import annotations

import math

import pytest

from chunkbuster.core.ranking import RankedItem, Ranking, reciprocal_rank_fusion


def test_ranking_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        Ranking(
            (
                RankedItem("same", "first", 1.0),
                RankedItem("same", "second", 0.5),
            )
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_ranked_item_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        RankedItem("item", object(), score)


def test_rrf_combines_by_identity_and_preserves_deterministic_order() -> None:
    dense = Ranking(
        (
            RankedItem("a", "A", 0.9),
            RankedItem("b", "B", 0.8),
        ),
        origins=frozenset({"dense"}),
    )
    sparse = Ranking(
        (
            RankedItem("b", "B", 12.0),
            RankedItem("c", "C", 8.0),
        ),
        origins=frozenset({"sparse"}),
    )

    fused = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        top_k=3,
        k=10,
    )

    assert fused.ids == ("b", "a", "c")
    assert tuple(item.rank for item in fused) == (1, 2, 3)
    assert fused[0].score == pytest.approx(1 / 12 + 1 / 11)
    assert fused[1].score == pytest.approx(1 / 11)
    assert fused[2].score == pytest.approx(1 / 12)
    assert fused.origins == frozenset({"dense", "sparse"})
    assert all(item.provenance[-1] == "rrf" for item in fused)
