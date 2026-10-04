"""Unit 3's scale spectrum and its multi-scale synthetic (`p1d_cluster_ensemble/scale_spectrum.py`)."""

import numpy as np
import pytest

from p1d_cluster_ensemble import scale_spectrum as ss
from p1d_cluster_ensemble.methods import LayerData
from p1d_cluster_ensemble.merge_tree import labels_at_delta

pytestmark = pytest.mark.deps


class TestVmf:
    @pytest.mark.parametrize("d,kappa", [(3, 0.7), (10, 5.0), (64, 40.0), (1024, 800.0)])
    def test_mean_cosine_matches_scipy_where_it_is_finite(self, d, kappa):
        from scipy.special import ive
        assert ss.mean_cosine(kappa, d) == pytest.approx(ive(d / 2, kappa) / ive(d / 2 - 1, kappa), rel=1e-10)

    def test_mean_cosine_finite_where_scipy_underflows(self):
        assert 0 < ss.mean_cosine(1e-3, 1024) < 1e-5

    @pytest.mark.parametrize("rho", [0.3, 0.77, 0.92])
    def test_kappa_inverts(self, rho):
        assert ss.mean_cosine(ss.kappa_for(rho, 1024), 1024) == pytest.approx(rho, abs=1e-9)

    def test_draws_have_the_mean_cosine(self):
        rng = np.random.default_rng(0)
        mu = np.zeros(50)
        mu[0] = 1.0
        X = ss.vmf(mu, ss.kappa_for(0.8, 50), 4000, rng)
        assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
        assert (X @ mu).mean() == pytest.approx(0.8, abs=0.01)


class TestSynthetic:
    def test_planted_spreads_before_the_opening(self):
        s = ss.synthetic(0, open_t=0.0)
        assert s["Y"].shape == (399, ss.DIM)  # T1 drops position 0
        assert (s["fine"] >= 0).sum() + (s["fine"] < 0).sum() == 399
        assert set(np.unique(s["coarse"])) == {-1, 0, 1, 2}
        assert np.all((s["fine"] // 3 == s["coarse"]) | (s["fine"] < 0))
        c = ss.construction_distances(s["Y"], s["fine"], "raw")
        assert c["within_sub"] == pytest.approx(ss.D_FINE, abs=0.01)
        assert c["between_sub"] == pytest.approx(ss.D_COARSE, abs=0.02)

    def test_bad_spreads_refuse(self):
        with pytest.raises(ValueError):
            ss.synthetic(0, d_fine=0.4, d_coarse=0.3, open_t=0.0)


def _rows(spec):
    """(k_sub, stability, p) triples to rows."""
    return [{"r": float(i), "k_sub": k, "stability": s, "p": p, "informative": True}
            for i, (k, s, p) in enumerate(spec)]


_A = np.repeat([0, 1], 50)


def _moved(m):
    """_A with its first m tokens moved to cluster 1."""
    lab = _A.copy()
    lab[:m] = 1
    return lab


def _pl(rows, labels=None, **kw):
    return ss.robust_plateaus(rows, labels if labels is not None else [_A] * len(rows), **kw)


class TestRobustPlateaus:
    def test_run_of_three_found_two_not(self):
        rows = _rows([(3, .9, .01)] * 3 + [(0, None, 1.0)] + [(4, .9, .01)] * 2)
        assert [(p["start"], p["end"], p["k_sub"]) for p in _pl(rows)] == [(0, 2, 3)]

    def test_a_count_change_alone_does_not_split_a_run(self):
        rows = _rows([(3, .9, .01)] * 2 + [(4, .9, .01)] * 3)
        assert [(p["start"], p["end"], p["k_lo"], p["k_hi"]) for p in _pl(rows)] == [(0, 4, 3, 4)]

    def test_each_condition_breaks_it(self):
        base = [(3, .9, .01)] * 5
        for bad in [(3, .74, .01), (3, .9, .06), (1, .9, .01), (3, None, .01)]:
            rows = _rows(base[:2] + [bad] + base[3:])
            assert _pl(rows) == []

    def test_uninformative_b_breaks_it_only_with_b(self):
        rows = _rows([(3, .9, .01)] * 3)
        rows[1]["informative"] = False
        assert _pl(rows) == []
        assert len(_pl(rows, use_p=False)) == 1

    def test_without_b_ignores_p(self):
        rows = _rows([(3, .9, .5)] * 3)
        assert _pl(rows) == []
        assert len(_pl(rows, use_p=False)) == 1

    def test_run_at_the_end(self):
        rows = _rows([(0, None, 1.0)] + [(2, .8, .02)] * 3)
        assert [(p["start"], p["end"]) for p in _pl(rows)] == [(1, 3)]

    def test_drift_is_anchored_to_the_first_cut(self):
        from sklearn.metrics import adjusted_rand_score as ari
        labs = [_moved(m) for m in range(11)]
        assert all(ari(x, y) >= ss.CONT_ARI for x, y in zip(labs, labs[1:]))  # each step small
        assert ari(labs[0], labs[-1]) < ss.CONT_ARI  # the ends apart
        out = _pl(_rows([(2, .9, .01)] * 11), labs)
        assert len(out) >= 2 and out[0]["start"] == 0  # one run over all 11 would drift
        assert all(q["start"] == p["end"] + 1 for p, q in zip(out, out[1:]))
        assert all(ari(labs[p["start"]], labs[g]) >= ss.CONT_ARI
                   for p in out for g in range(p["start"], p["end"] + 1))


class TestOpening:
    def test_extent_and_label(self):
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((60, 20))
        fine = np.repeat(np.arange(3), 20)
        Y = np.vstack([Y[fine == c] for c in range(3)])  # tight groups by construction below
        Y += 4 * np.repeat(np.eye(20)[:3], 20, axis=0)
        perm = rng.permutation(60)
        Y, fine = Y[perm], fine[perm]
        Y[:5] = Y[0] + 0.01 * rng.standard_normal((5, 20))  # a tight opening of 5
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        J = ss.opening_extent(Y, fine)
        assert J >= 5
        lab = ss.planted_labels({"Y": Y, "fine": fine, "coarse": fine})
        assert np.all(lab["fine"][:J] == 3) and np.array_equal(lab["fine"][J:], fine[J:])
        assert np.array_equal(lab["coarse"], fine)

    def test_no_opening_without_the_flow(self):
        s = ss.synthetic(0, open_t=0.0)
        assert ss.opening_extent(s["Y"], s["fine"]) == 0


def test_clopper_pearson_lower():
    assert ss.clopper_pearson_lower(0, 10) == 0.0
    assert ss.clopper_pearson_lower(10, 10) == pytest.approx(0.05 ** 0.1)
    assert 0.55 < ss.clopper_pearson_lower(9, 10) < 0.65


def _two_blobs(n=40, d=30, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    c = np.eye(d)[:2]
    X = np.vstack([c[i] + spread * rng.standard_normal((n // 2, d)) for i in range(2)])
    return X / np.linalg.norm(X, axis=1, keepdims=True), np.repeat([0, 1], n // 2)


class TestStability:
    def test_separated_blobs_are_stable_and_singletons_have_none(self):
        X, _ = _two_blobs()
        data = LayerData.from_normed(X)
        Z = ss._tree(data)
        deltas = [1e-6, 0.5]
        labels = [labels_at_delta(Z, data.n, dl) for dl in deltas]
        st = ss.hennig_stability(data, labels, deltas, np.random.default_rng(0), n_sub=10)
        assert st[0] is None
        assert st[1] == pytest.approx(1.0)

    def test_a_lone_survivor_does_not_count(self):
        # a, b at cosine distance 0.01; c at 0.04 from a, 0.089 from b. At delta 0.05
        # the cut is {a, b}, {c}; the subsample {a, c} joins a to c (Jaccard 1/2 under the
        # old rule), but has one survivor of {a, b} and is not counted.
        th = np.array([0.0, np.arccos(0.99), -np.arccos(0.96)])
        X = np.c_[np.cos(th), np.sin(th)]
        data = LayerData.from_normed(X)
        lab = labels_at_delta(ss._tree(data), 3, 0.05)
        assert lab[0] == lab[1] != lab[2]
        st = ss.hennig_stability(data, [lab], [0.05], np.random.default_rng(0), n_sub=30, min_size=2)
        assert st == [pytest.approx(1.0)]

    def test_rank_p(self):
        assert ss.rank_p_higher(5, np.array([1, 2, 5, 6])) == pytest.approx(3 / 5)
        assert ss.rank_p_higher(9, np.zeros(49)) == pytest.approx(1 / 50)


class TestSpectrum:
    def test_rows_and_found(self):
        X, lab = _two_blobs()
        spec = ss.spectrum(X, "raw", 0, n_sub=5, n_draws=5)
        assert set(spec["rows"]) == set(ss.ARMS)
        rows = spec["rows"]["main"]
        assert len(rows) == ss.GRID_N == len(spec["_labels"])
        ks = [r["k"] for r in rows]
        assert ks == sorted(ks, reverse=True)
        assert all(r["delta"] == pytest.approx(r["r"] * spec["median"]) for r in rows)
        assert all(r["p"] == 1.0 for r in rows if r["stability"] is None)
        pl = [{"start": g, "end": g} for g, r in enumerate(rows) if r["k_sub"] == 2]
        assert pl
        out = ss.planted_ari(spec, pl, {"coarse": lab})
        assert ss.found_scales(out, ("coarse",)) == {"coarse": True}

    def test_each_draw_has_its_own_tree_median_and_subsamples(self):
        # (b)'s null: draw i is cut at r x its own median on its own tree, with
        # subsamples from the stream [seed, _SUB, i] (design "The re-run").
        from p1d_cluster_ensemble.gaussian_null import frame_vectors, gaussian_draw, span_coordinates
        X, _ = _two_blobs()
        spec = ss.spectrum(X, "raw", 3, n_sub=4, n_draws=2)
        Zs = span_coordinates(frame_vectors(X, "raw")[0])
        rng = np.random.default_rng([3, ss._NULL])
        for i in range(2):
            G = LayerData.from_normed(gaussian_draw(Zs, rng))
            dg = [r * ss._median_distance(G) for r in ss.relative_grid()]
            lg = [labels_at_delta(ss._tree(G), G.n, d) for d in dg]
            want = ss.hennig_stability(G, lg, dg, np.random.default_rng([3, ss._SUB, i]), n_sub=4)
            got = spec["_null"]["main"][1][i]
            assert np.array_equal(np.isnan(got), [v is None for v in want])
            assert np.allclose(got[~np.isnan(got)], [v for v in want if v is not None])


class TestOnAKnownClusteredCloud:
    """LESSONS 6 (2026-10-04): a tail is computed on a known clustered cloud first.
    On the synthetic without the opening (seed 0, seen): stability and count find both
    planted scales; the count's tail (the first check's (b), ``p_count``) rejects the
    coarse one, where the Gaussian has as many pieces or more; option 1's stability tail
    (Blocked 15) keeps both. 20 draws, so the smallest rank p (1/21) is below ALPHA: at
    10 (this test before the re-run) no plateau could pass any tail."""

    @pytest.fixture(scope="class")
    def read(self):
        assert 1 / 21 <= ss.ALPHA < 1 / 11
        s = ss.synthetic(0, open_t=0.0)
        spec = ss.spectrum(s["Y"], "centred", 0, n_sub=10, n_draws=20)
        return spec, {"coarse": s["coarse"], "fine": s["fine"]}

    def test_count_tail_rejects_planted_scales_that_stability_finds(self, read):
        spec, planted = read
        rows = spec["rows"]["main"]
        labels = spec["_labels"]
        without = ss.planted_ari(spec, ss.robust_plateaus(rows, labels, use_p=False), planted)
        assert ss.found_scales(without) == {"coarse": True, "fine": True}
        count = ss.planted_ari(spec, ss.robust_plateaus(rows, labels, p_key="p_count"), planted)
        assert ss.found_scales(count)["coarse"] is False

    def test_stability_tail_keeps_them(self, read):
        spec, planted = read
        pl = ss.planted_ari(spec, ss.robust_plateaus(spec["rows"]["main"], spec["_labels"]), planted)
        assert ss.found_scales(pl) == {"coarse": True, "fine": True}


class TestArmRows:
    grid = np.array([0.1, 0.2])
    labels = [np.arange(6), np.array([0, 0, 0, 0, 1, 2])]
    null_st = np.array([[np.nan, np.nan], [np.nan, 0.95], [np.nan, 0.5]])
    null_k = np.array([[0, 0], [0, 1], [0, 1]])

    def test_empty_draw_scores_zero_and_empty_cut_has_p_one(self):
        rows = ss.arm_rows(self.grid, [0.01, 0.02], self.labels, [None, 0.9],
                           self.null_k, self.null_st, 4)
        assert rows[0]["p"] == 1.0 and rows[0]["z"] is None
        assert rows[1]["k_sub"] == 1
        assert rows[1]["p"] == pytest.approx(2 / 4)  # refs 0, 0.95, 0.5
        assert rows[1]["null_n_empty"] == 1
        assert rows[1]["p_count"] == pytest.approx(3 / 4)  # refs 0, 1, 1

    def test_informative_needs_a_tenth_of_the_draws(self):
        lab = [np.array([0, 0, 0, 0, 1, 2])]
        for n_full, want in [(4, False), (5, True)]:
            st = np.full((50, 1), np.nan)
            st[:n_full] = 0.5
            row = ss.arm_rows(np.array([0.1]), [0.01], lab, [0.9], np.zeros((50, 1), int), st, 4)[0]
            assert row["informative"] is want

    def test_a_row_without_the_flag_refuses(self):
        rows = _rows([(3, .9, .01)] * 3)
        del rows[1]["informative"]
        with pytest.raises(KeyError):
            _pl(rows)

    def test_matched_count(self):
        rows = [{"k_sub": 0, "stability": None}, {"k_sub": 3, "stability": 0.9}]
        ks = np.array([[0, 0], [0, 3], [0, 3]])
        m = ss.matched_count(rows, ks, self.null_st, 3)
        assert (m["cloud_min"], m["n_draws_with_k"], m["draw_max"], m["n_draws_at_or_above"]) == (0.9, 2, 0.95, 1)
