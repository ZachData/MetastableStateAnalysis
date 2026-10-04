"""Unit 3's real-input reader, step 1: specificity (`p1d_cluster_ensemble/scale_real.py`)."""

import json

import numpy as np
import pytest

from p1d_cluster_ensemble import scale_real as sr
from p1d_cluster_ensemble import scale_spectrum as ss
from p1d_cluster_ensemble.arch_null import comparison_models, label, model_ids
from p1d_cluster_ensemble.move_text import V1_PASSAGES

pytestmark = pytest.mark.deps

INITS, REINITS = model_ids("init"), model_ids("reinit")


def _row(z, stability=0.9, informative=True, p=0.5):
    return {"r": 0.1, "k_sub": 3, "stability": stability, "z": z, "informative": informative, "p": p}


class TestSeeds:
    def test_same_for_every_model_distinct_per_cloud(self):
        seeds = {sr.cloud_seed(k, L) for k in V1_PASSAGES for L in range(1, 25)}
        assert len(seeds) == len(V1_PASSAGES) * 24
        assert min(seeds) > 51  # apart from the synthetic's seeds 0-51


class TestRefValues:
    def test_no_cluster_is_below(self):
        v = sr.ref_values([_row(1.5), _row(None, stability=None)])
        assert v[0] == 1.5 and v[1] == -np.inf

    def test_sd0_takes_the_sign_and_drops_only_a_tie(self):
        # #139 finding 1: a reference with a cluster whose draws are all empty is the
        # furthest above its covariance, not a reference to drop.
        rows = [dict(_row(None, stability=0.8), null_stab_mean=0.0),
                dict(_row(None, stability=0.8), null_stab_mean=1.0),
                dict(_row(None, stability=1.0), null_stab_mean=1.0)]
        v = sr.ref_values(rows)
        assert v[0] == np.inf and v[1] == -np.inf and np.isnan(v[2])


class TestRankRows:
    def test_p_counts_ties_and_no_cluster_refs_as_below(self):
        refs = np.full((40, 1), -np.inf)
        refs[:2, 0] = 2.0  # two at or above
        refs[2:5, 0] = 1.0
        (r,) = sr.rank_rows([_row(2.0)], refs)
        assert r["p"] == pytest.approx(3 / 41) and r["n_ref"] == 40 and r["informative"]
        assert r["p_gauss"] == 0.5

    def test_one_at_or_above_passes_both_arms(self):
        for n in (40, 39):
            refs = np.zeros((n, 1))
            refs[0, 0] = 5.0
            (r,) = sr.rank_rows([_row(1.0)], refs)
            assert r["p"] <= ss.ALPHA

    def test_dropped_refs_shrink_n_and_below_min_ref_refuses(self):
        refs = np.zeros((40, 1))
        refs[:10, 0] = np.nan
        (r,) = sr.rank_rows([_row(1.0)], refs)
        assert r["n_ref"] == 30 and r["informative"] and r["p"] == pytest.approx(1 / 31)
        refs[:11, 0] = np.nan
        (r,) = sr.rank_rows([_row(1.0)], refs)
        assert r["n_ref"] == 29 and not r["informative"] and r["p"] == 1.0

    @pytest.mark.parametrize("row", [_row(None), _row(3.0, informative=False), _row(None, stability=None)])
    def test_own_z_undefined_or_uninformative_is_not_admissible(self, row):
        (r,) = sr.rank_rows([row], np.zeros((40, 1)))
        assert not r["informative"] and r["p"] == 1.0


class TestMinRun:
    def test_robust_plateaus_takes_min_run(self):
        rows = [{"r": float(g), "k_sub": 2, "stability": 0.9, "p": 0.01, "informative": True} for g in range(4)]
        lab = [np.repeat([0, 1], 5)] * 4
        assert len(ss.robust_plateaus(rows, lab, min_size=2)) == 1
        assert len(ss.robust_plateaus(rows, lab, min_size=2, min_run=4)) == 1
        assert ss.robust_plateaus(rows, lab, min_size=2, min_run=5) == []


def _table(hit_share_by_mid, layers=range(1, 9)):
    """A plateau table: per band, model m has a plateau in its first round(share x clouds) clouds."""
    from p1d_cluster_ensemble.move_text import band
    tab = {}
    for b in sorted({band(L) for L in layers}):
        keys = [(k, L) for k in V1_PASSAGES for L in layers if band(L) == b]
        for mid, s in hit_share_by_mid.items():
            n_hit = int(round(s * len(keys)))
            for i, (k, L) in enumerate(keys):
                tab[(mid, k, L)] = [{"start": 0}] if i < n_hit else []
    return tab


class TestShareAndVerdicts:
    def test_share_counts_band_clouds(self):
        tab = _table({"reinit:0": 0.5, "reinit:1": 0.0}, layers=range(1, 17))
        assert sr.share(tab, ["reinit:0"], "L1-8") == (28, 56)
        assert sr.share(tab, ["reinit:0", "reinit:1"], "L9-16") == (28, 112)
        assert sr.share(tab, ["reinit:0"], "L17-24") == (0, 0)

    def test_subset_quantile_of_equal_models_is_their_share(self):
        tab = _table({m: 0.25 for m in REINITS})
        assert sr.subset_quantile(tab, REINITS, "L1-8", n_subsets=50) == pytest.approx(14 / 56)

    def _tabs(self, reinit_by_run, init_share):
        tab = {"reinit": {}, "real_init": {}}
        for m in range(ss.MIN_RUN, sr.MAX_MIN_RUN + 1):
            tab["reinit"][m] = _table({x: reinit_by_run[m] for x in REINITS}, layers=range(1, 25))
            tab["real_init"][m] = _table({x: init_share for x in INITS}, layers=range(1, 25))
        return tab

    def test_rate_raises_min_run_until_it_passes(self):
        rows = sr.verdicts(self._tabs({3: 0.2, 4: 0.04, 5: 0.0}, 0.0))
        r = rows[0]
        assert r["band"] == "L1-8" and r["min_run"] == 4 and r["rate_pass"]
        assert list(r["reinit_rate_by_min_run"]) == [3, 4]
        assert r["route_pass"] and r["pass"]

    def test_rate_failing_at_max_refuses(self):
        r = sr.verdicts(self._tabs({3: 0.3, 4: 0.2, 5: 0.1}, 0.0))[0]
        assert r["min_run"] is None and not r["rate_pass"] and not r["pass"]

    def test_route_fails_when_real_inits_exceed_the_subsets(self):
        r = sr.verdicts(self._tabs({3: 0.02, 4: 0.0, 5: 0.0}, 0.5))[0]
        assert r["rate_pass"] and not r["route_pass"] and not r["pass"]


class TestLoadSets:
    def test_refuses_another_union(self, tmp_path):
        f = tmp_path / "s.json"
        sets = {k: {"kept": [1, 2, 3]} for k in V1_PASSAGES}
        f.write_text(json.dumps({"comparison": [label(st, m) for st, m in comparison_models("first")],
                                 "sets": sets}))
        with pytest.raises(SystemExit):
            sr.load_sets(f)
        f.write_text(json.dumps({"comparison": [label(st, m) for st, m in comparison_models("trained")],
                                 "sets": sets}))
        assert sr.load_sets(f)["wiki_paragraph"] == [1, 2, 3]


def _cloud(z, n_tok=10):
    rows = [{"r": float(g), "k_sub": 2, "stability": 0.9, "z": z, "informative": True, "p": 0.5,
             "null_stab_mean": 0.5} for g in range(ss.GRID_N)]
    return {"rows": {a: rows for a in ss.ARMS}}, np.tile(np.repeat([0, 1], n_tok // 2), (ss.GRID_N, 1))


class TestPlateauTable:
    """#139 finding 4: the reference sets (real inits among all 40, each re-init among the other 39)."""

    def _run(self, monkeypatch, high):
        monkeypatch.setattr(sr, "V1_PASSAGES", ("wiki_paragraph",))
        monkeypatch.setattr(sr, "LAYERS", (1,))
        recs, labs = {}, {}
        for m in INITS + REINITS:
            rec, lab = _cloud(5.0 if m in high else 0.0)
            recs[m] = {"wiki_paragraph": {1: rec}}
            labs[m] = {"wiki_paragraph": {1: lab}}
        return sr.plateau_table(recs, labs, "main", (3,))

    def test_a_reinit_is_not_its_own_reference(self, monkeypatch):
        # Two re-inits at z = 5: each sees one other at or above, p = 2/40 = 0.05 (a plateau);
        # counted against itself it would be 3/41. A real init at z = 5 sees both: 3/41.
        t = self._run(monkeypatch, high={"reinit:0", "reinit:1", "init:0"})
        assert t["reinit"][3][("reinit:0", "wiki_paragraph", 1)]
        assert not t["reinit"][3][("reinit:2", "wiki_paragraph", 1)]
        assert not t["real_init"][3][("init:0", "wiki_paragraph", 1)]
        assert t["n_ref_min"][("init:0", "wiki_paragraph", 1)] == 40
        assert t["n_ref_min"][("reinit:0", "wiki_paragraph", 1)] == 39

    def test_without_b_ignores_the_reference(self, monkeypatch):
        t = self._run(monkeypatch, high=set())
        assert all(t["without_b"].values())
        assert not any(t["reinit"][3].values())


class TestLoadRun:
    def _write(self, out, mids, error=False):
        for m in mids:
            rec, lab = _cloud(0.0)
            layer = {"layer": 1, "error": "refused"} if error else {"layer": 1, **rec}
            sr._write(out, m, "wiki_paragraph", {1: (layer, None if error else lab)},
                      {"git": "abc", "sets_sha256": "x"})

    def test_refuses_missing_and_refused_trees(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr, "V1_PASSAGES", ("wiki_paragraph",))
        self._write(tmp_path, INITS + REINITS[:-1])
        with pytest.raises(SystemExit, match="missing"):
            sr.load_run(tmp_path)
        self._write(tmp_path, REINITS[-1:], error=True)
        with pytest.raises(SystemExit, match="refused tree"):
            sr.load_run(tmp_path)
        self._write(tmp_path, REINITS[-1:])
        recs, labs = sr.load_run(tmp_path)
        assert len(recs) == 50 and labs["init:0"]["wiki_paragraph"][1].shape == (ss.GRID_N, 10)


class TestReadCloud:
    def test_rows_and_labels_populated(self):
        rng = np.random.default_rng(0)
        Y = np.vstack([rng.normal(size=(15, 32)) + 6 * rng.normal(size=32) for _ in range(3)])
        rec, lab = sr.read_cloud(Y, sr.cloud_seed("wiki_paragraph", 1))
        assert lab.shape == (ss.GRID_N, 45) and lab.dtype == np.uint16
        assert set(rec["rows"]) == set(ss.ARMS)
        assert len(rec["rows"]["main"]) == ss.GRID_N
        assert any(r["z"] is not None for r in rec["rows"]["main"])
