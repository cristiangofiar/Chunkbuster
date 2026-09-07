from __future__ import annotations

import pytest

from chunkbuster.core.normalization import normalize_scores


def test_builtin_score_normalizations() -> None:
    assert normalize_scores((2, 4, 6), "none") == (2, 4, 6)
    assert normalize_scores((2, 4, 6), "min_max") == (0, 0.5, 1)
    z_scores = normalize_scores((1, 2, 3), "z_score")
    assert sum(z_scores) == pytest.approx(0)
    assert z_scores[0] == pytest.approx(-z_scores[2])
    assert normalize_scores((3, 3), "min_max") == (0, 0)
    assert normalize_scores((3, 3), "z_score") == (0, 0)
