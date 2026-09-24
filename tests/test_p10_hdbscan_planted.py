"""
`tools/run/p10_hdbscan_planted.py` — known answer for HDBSCAN on duplicates planted in noise.

Checks the plant (group sizes, exact copies, order shuffled) and the scorer's
four outcomes on hand-made labels; one tiny end-to-end run needs `hdbscan`.
"""
import numpy as np
import pytest

from tools.run.p10_hdbscan_planted import plant, score

pytestmark = pytest.mark.deps


def test_plant_exact_copies():
    X, g, sizes = plant(np.random.default_rng(0), n_bg=10, d=8, groups={2: 3, 4: 1})
    assert X.shape == (10 + 6 + 4, 8) and sizes == [2, 2, 2, 4]
    assert (g == -1).sum() == 10
    for i, c in enumerate(sizes):
        rows = X[g == i]
        assert len(rows) == c and np.all(rows == rows[0])


def test_score_outcomes():
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3, -1, -1])
    labels = np.array([5, 5, -1, -1, 6, 6, 7, 8, 6, -1])
    r = score(labels, g, [2, 2, 2, 2])
    assert r["groups"][2] == {"whole": 1, "noise": 1, "merged": 1, "split": 1}
    assert r["background"] == [2, 1]
    assert sorted(r["groups_per_cluster"]) == [1, 1, 1, 1]


def test_end_to_end_small():
    pytest.importorskip("hdbscan")
    from tools.run.p10_hdbscan_planted import run
    r = run(2, n_bg=40, d=64, groups={3: 3})
    assert r["by_copies"]["3"]["n"] == 6
    assert r["by_copies"]["3"]["noise"] == 0
