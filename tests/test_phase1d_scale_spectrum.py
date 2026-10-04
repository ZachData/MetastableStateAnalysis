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
    return [{"r": float(i), "k_sub": k, "stability": s, "p": p} for i, (k, s, p) in enumerate(spec)]


class TestRobustPlateaus:
    def test_run_of_three_found_two_not(self):
        rows = _rows([(3, .9, .01)] * 3 + [(0, None, 1.0)] + [(4, .9, .01)] * 2)
        out = ss.robust_plateaus(rows)
        assert [(p["start"], p["end"], p["k_sub"]) for p in out] == [(0, 2, 3)]

    def test_count_change_splits_a_run(self):
        rows = _rows([(3, .9, .01)] * 2 + [(4, .9, .01)] * 3)
        assert [(p["start"], p["end"]) for p in ss.robust_plateaus(rows)] == [(2, 4)]

    def test_each_condition_breaks_it(self):
        base = [(3, .9, .01)] * 5
        for bad in [(3, .74, .01), (3, .9, .06), (1, .9, .01), (3, None, .01)]:
            rows = _rows(base[:2] + [bad] + base[3:])
            assert ss.robust_plateaus(rows) == []

    def test_without_b_ignores_p(self):
        rows = _rows([(3, .9, .5)] * 3)
        assert ss.robust_plateaus(rows) == []
        assert len(ss.robust_plateaus(rows, use_p=False)) == 1

    def test_run_at_the_end(self):
        rows = _rows([(0, None, 1.0)] + [(2, .8, .02)] * 3)
        assert [(p["start"], p["end"]) for p in ss.robust_plateaus(rows)] == [(1, 3)]


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

    def test_rank_p(self):
        assert ss.rank_p_higher(5, np.array([1, 2, 5, 6])) == pytest.approx(3 / 5)
        assert ss.rank_p_higher(9, np.zeros(49)) == pytest.approx(1 / 50)


class TestSpectrum:
    def test_rows_and_found(self):
        X, lab = _two_blobs()
        spec = ss.spectrum(X, "raw", 0, n_sub=5, n_draws=5)
        assert len(spec["rows"]) == ss.GRID_N == len(spec["_labels"])
        ks = [r["k"] for r in spec["rows"]]
        assert ks == sorted(ks, reverse=True)
        assert all(r["delta"] == pytest.approx(r["r"] * spec["median"]) for r in spec["rows"])
        pl = [{"start": g, "end": g} for g, r in enumerate(spec["rows"]) if r["k_sub"] == 2]
        assert pl
        out = ss.planted_ari(spec, pl, {"coarse": lab})
        assert ss.found_scales(out, ("coarse",)) == {"coarse": True}


class TestOnAKnownClusteredCloud:
    """LESSONS 6 (2026-10-04): a tail is computed on a known clustered cloud first.
    Pins the first check's finding: on the synthetic without the opening, stability
    and count find both planted scales, and (b)'s count tail rejects them."""

    def test_count_tail_rejects_planted_scales_that_stability_finds(self):
        s = ss.synthetic(0, open_t=0.0)
        spec = ss.spectrum(s["Y"], "centred", 0, n_sub=10, n_draws=10)
        planted = {"coarse": s["coarse"], "fine": s["fine"]}
        without = ss.planted_ari(spec, ss.robust_plateaus(spec["rows"], use_p=False), planted)
        assert ss.found_scales(without) == {"coarse": True, "fine": True}
        with_b = ss.planted_ari(spec, ss.robust_plateaus(spec["rows"]), planted)
        assert ss.found_scales(with_b) == {"coarse": False, "fine": False}


class TestOption1Helpers:
    rows = [{"r": 0.1, "k_sub": 0, "stability": None, "p": 1.0},
            {"r": 0.2, "k_sub": 3, "stability": 0.9, "p": 1.0}]
    st = np.array([[np.nan, np.nan], [np.nan, 0.95], [np.nan, 0.5]])
    ks = np.array([[0, 0], [0, 3], [0, 3]])

    def test_empty_draw_rules(self):
        zero = ss.option1_rows(self.rows, self.st, "zero")
        drop = ss.option1_rows(self.rows, self.st, "drop")
        assert zero[0]["p"] == drop[0]["p"] == 1.0  # no substantial cluster: no p
        assert zero[1]["p"] == pytest.approx(2 / 4)  # refs 0, 0.95, 0.5
        assert drop[1]["p"] == pytest.approx(2 / 3)  # refs 0.95, 0.5

    def test_matched_count(self):
        m = ss.matched_count(self.rows, self.ks, self.st, 3)
        assert (m["synthetic_min"], m["n_draws_with_k"], m["draw_max"], m["n_draws_at_or_above"]) == (0.9, 2, 0.95, 1)
