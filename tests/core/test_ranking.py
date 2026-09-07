from __future__ import annotations

import math

import pytest

from chunkbuster.core.ranking import (
    RankedItem,
    Ranking,
    fuse_rankings,
    reciprocal_rank_fusion,
)


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


def test_ranking_to_text_is_minimal_ordered_and_customizable() -> None:
    ranking = Ranking(
        (
            RankedItem("a", "Alpha", 0.9, metadata={"source": "dense"}),
            RankedItem("b", "Beta", 0.7),
        )
    )

    assert ranking.to_text() == "- a: Alpha\n- b: Beta"
    assert Ranking().to_text() == ""
    assert (
        ranking.to_text(
            lambda candidate: (
                f"{candidate.rank}|{candidate.score}|{dict(candidate.metadata)}"
            )
        )
        == "1|0.9|{'source': 'dense'}\n2|0.7|{}"
    )


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


def test_shared_weighted_score_fusions() -> None:
    dense = Ranking(
        (
            RankedItem("a", "A", 3),
            RankedItem("b", "B", 2),
            RankedItem("c", "C", 1),
        ),
        origins=frozenset({"dense"}),
    )
    sparse = Ranking(
        (RankedItem("b", "B", 4), RankedItem("c", "C", 3)),
        origins=frozenset({"sparse"}),
    )

    summed = fuse_rankings(
        {"dense": dense, "sparse": sparse},
        method="weighted_sum",
        normalization="min_max",
        top_k=3,
    )
    consensus = fuse_rankings(
        {"dense": dense, "sparse": sparse},
        method="comb_mnz",
        normalization="min_max",
        top_k=3,
    )
    weighted_rrf = fuse_rankings(
        {"dense": dense, "sparse": sparse},
        method="weighted_rrf",
        weights={"dense": 1, "sparse": 2},
        top_k=3,
        k=10,
    )

    assert summed.ids == ("b", "a", "c")
    assert consensus[0].id == "b"
    assert consensus[0].score == pytest.approx(summed[0].score * 2)
    assert weighted_rrf[0].id == "b"

    sparse_single = Ranking((RankedItem("c", "C", 4),))
    with_zero_baseline = fuse_rankings(
        {"dense": dense, "sparse": sparse_single},
        method="weighted_sum",
        normalization="min_max",
        normalize_missing_as_zero=True,
        top_k=3,
    )
    assert with_zero_baseline[0].id in {"a", "c"}
    assert next(item for item in with_zero_baseline if item.id == "c").score == 1
