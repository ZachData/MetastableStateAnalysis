"""
`tools/run/p10_f13_centres.py` — F13, the centre scan, with A9 and F15, as `design-10.md` "F13" fixed it.
"""
import numpy as np
import pytest

from tools.run import p10_f13_centres as f

# Tier: the module imports the label source lazily, but the run path needs scikit-learn (as E1's tests).
pytestmark = pytest.mark.deps


def _line(x):
    x = np.asarray(x, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def _sets(D, delta):
    O = np.arange(D.shape[0])[None, :]
    R = f.ranks_of(O)
    S = f.strong_sets(f.nearest_pred(D, R), [delta])[0, 0]
    Od = f.ordinary_sets(D, O, [delta])[0, 0]
    return set(np.flatnonzero(S)), set(np.flatnonzero(Od))


def test_hand_built_sequence_separates_the_two_rules():
    # 0.25 is within δ of 0; 0.5 is within δ of 0.25 (a token) but not of 0 (the only centre)
    S, O = _sets(_line([0.0, 0.25, 0.5]), 0.3)
    assert S == {0} and O == {0, 2}


def test_strong_is_a_subset_of_ordinary_on_random_clouds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        U = rng.normal(size=(50, 6))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        D = f.geodesic(U)
        O = f.orders(0, "x", 50, n_perm=5)
        R = f.ranks_of(O)
        S = f.strong_sets(f.nearest_pred(D, R), f.DELTAS)
        Od = f.ordinary_sets(D, O, f.DELTAS)
        assert (S <= Od).all()


def test_orders_hold_position_0_first_and_are_permutations():
    O = f.orders(512, "wiki_paragraph", 30, n_perm=10)
    assert (O[:, 0] == 0).all() and (O[0] == np.arange(30)).all()
    assert all(sorted(r) == list(range(30)) for r in O)


def test_exact_null_matches_the_draws_and_iid_circle_has_no_gap():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 300)
    D = np.abs(np.angle(np.exp(1j * (th[:, None] - th[None, :]))))
    delta = 0.3
    kept = np.arange(1, 300)
    O = f.orders(7, "circle", 300, n_perm=400)
    S = f.strong_sets(f.nearest_pred(D, f.ranks_of(O)), [delta])[0][:, kept].sum(1)
    ex = f.exact_null_strong(D, kept, delta)
    assert abs(S[1:].mean() - ex) < 3 * S[1:].std(ddof=1) / np.sqrt(400) + 1e-9
    # i.i.d. points: the stored order is one more random order, so its gap is small
    assert abs(np.log((S[0] + 0.5) / (ex + 0.5))) < 0.5


def test_log_rho_is_zero_under_uniform_causal_attention():
    n = 12
    A = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]
    assert np.allclose(f.log_rho(np.stack([A, A])), 0.0)


def test_binned_diff_recovers_a_planted_effect_and_ignores_position():
    n = 160
    kept = np.arange(1, n)
    bins = f.bins_of(kept)
    rng = np.random.default_rng(2)
    A = np.sort(rng.choice(kept, 30, replace=False))
    B = np.setdiff1d(kept, A)
    step_pos = np.zeros(n)
    for k, b in enumerate(bins):
        step_pos[b] = 3.0 * k                                  # a position-only effect, constant per bin
    v = step_pos.copy()
    assert abs(f.binned_diff(v, A, B, bins)) < 1e-12
    v[A] += 1.0
    assert abs(f.binned_diff(v, A, B, bins) - 1.0) < 1e-12
    assert f.binned_diff(v, A[:2], B, bins) is None


def test_coverage_of_two_planted_clusters_by_their_first_members():
    rng = np.random.default_rng(3)
    c1, c2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    U = np.vstack([c1 + 0.01 * rng.normal(size=(10, 3)), c2 + 0.01 * rng.normal(size=(10, 3))])
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    D = f.geodesic(U)
    kmask = np.ones(20, dtype=bool)
    Cm = np.zeros((2, 20), dtype=bool)
    Cm[0, [0, 10]] = True                                      # one member of each cluster
    Cm[1, [0, 1]] = True                                       # two members of one cluster
    cov, per = f.coverage(Cm, D <= 0.2, kmask)
    assert cov[0] == 1.0 and abs(cov[1] - 8 / 18) < 1e-12
    assert per[0] == 9.0


def test_t2_positions_are_first_occurrences_off_the_kept_set():
    toks = ["a", "b", "a", "\n", "c", "\n"]
    assert f.t2_positions(toks, np.array([1, 4])) == [3]


def test_sign_label_and_isolated_leans():
    assert f.sign_label([1, 2, 3, 4, 5, 6, 7], "up", "down") == "up"
    assert f.sign_label([1, 2, 3, 4, 5, 6, -7], "up", "down") == "leans up"
    assert f.sign_label([1, 2, None, None], "up", "down") == "too few"
    labs = f.mark_isolated({0: "mixed", 2: "leans up", 4: "mixed"})
    assert labs[2] == "leans up (isolated)"


def test_null_bar_allows_chance_misses_only():
    assert 2 <= f.null_bar(216) <= 5
    assert f.null_bar(0) == 0
