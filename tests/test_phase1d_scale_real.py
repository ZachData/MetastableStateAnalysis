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
        assert r["p_rank"] == pytest.approx(3 / 41) and r["n_ref"] == 40 and r["informative"]
        assert r["p_gauss"] == 0.5

    def test_one_at_or_above_passes_both_arms(self):
        for n in (40, 39):
            refs = np.zeros((n, 1))
            refs[0, 0] = 5.0
            (r,) = sr.rank_rows([_row(1.0)], refs)
            assert r["p_rank"] <= ss.ALPHA

    def test_dropped_refs_shrink_n_and_below_min_ref_refuses(self):
        refs = np.zeros((40, 1))
        refs[:10, 0] = np.nan
        (r,) = sr.rank_rows([_row(1.0)], refs)
        assert r["n_ref"] == 30 and r["informative"] and r["p_rank"] == pytest.approx(1 / 31)
        refs[:11, 0] = np.nan
        (r,) = sr.rank_rows([_row(1.0)], refs)
        assert r["n_ref"] == 29 and not r["informative"] and r["p"] == r["p_rank"] == 1.0

    @pytest.mark.parametrize("row", [_row(None), _row(3.0, informative=False), _row(None, stability=None)])
    def test_own_z_undefined_or_uninformative_is_not_admissible(self, row):
        (r,) = sr.rank_rows([row], np.zeros((40, 1)))
        assert not r["informative"] and r["p"] == 1.0


class TestConjunction:
    """Option 4 (Blocked 19): (b) is p_gauss <= alpha and the rank among the references <= alpha."""

    @pytest.mark.parametrize("p_gauss, top, admitted", [(0.01, False, True), (0.5, False, False),
                                                        (0.01, True, False), (0.5, True, False)])
    def test_both_terms_must_pass(self, p_gauss, top, admitted):
        refs = np.zeros((40, 1))
        if top:
            refs[:10, 0] = 5.0  # the cloud is not above its references
        (r,) = sr.rank_rows([_row(1.0, p=p_gauss)], refs)
        assert r["p"] == max(p_gauss, r["p_rank"])
        assert (r["p"] <= ss.ALPHA) is admitted

    def test_references_without_a_cluster_leave_the_gaussian_reader(self):
        # The pilot's fine scales: every re-init all singletons, so the rank is 1/41 < alpha
        # and admission is p_gauss's alone.
        for p_gauss in (0.02, 0.04, 0.06):
            (r,) = sr.rank_rows([_row(-3.0, p=p_gauss)], np.full((40, 1), -np.inf))
            assert r["p_rank"] == pytest.approx(1 / 41)
            assert (r["p"] <= ss.ALPHA) is (p_gauss <= ss.ALPHA)


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


def _cloud(z, n_tok=10, p_gauss=0.01):
    rows = [{"r": float(g), "k_sub": 2, "stability": 0.9, "z": z, "informative": True, "p": p_gauss,
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


class TestGate:
    def test_cells_per_band_distinct_trained_and_fixed(self):
        from p1d_cluster_ensemble.move_text import band
        cells = sr.gate_cells()
        assert len(cells) == len(set(cells)) == sr.GATE_N == 50
        assert [sum(band(L) == b for _, _, L in cells) for b in sr.BANDS] == list(sr.GATE_PER_BAND)
        assert {m for m, _, _ in cells} <= set(INITS)
        assert sr.gate_cells() == cells and sr.gate_cells(seed=1) != cells

    def test_cloud_is_a_populated_gaussian_not_the_input(self):
        rng = np.random.default_rng(0)
        Y = np.vstack([rng.normal(size=(15, 32)) + 6 * rng.normal(size=32) for _ in range(3)])
        rec, lab = sr.gate_cloud(Y, 0)
        own, _ = sr.read_cloud(Y, sr.GATE_SEED + 1)
        assert lab.shape == (ss.GRID_N, 45)
        assert any(r["z"] is not None for r in rec["rows"]["main"])
        assert rec["median"] != own["median"]
        assert sr.gate_cloud(Y, 0)[0]["median"] == rec["median"]

    def _rec(self, p_gauss, z=1.0):
        rec, lab = _cloud(z, p_gauss=p_gauss)
        pl = [{"start": 0}] if p_gauss <= ss.ALPHA else []
        rec["beside"] = {a: {"plateaus_gauss": pl, "plateaus_without_b": [{"start": 0}]} for a in ss.ARMS}
        return rec, lab

    def test_plateaus_need_both_terms(self):
        low = {a: np.zeros((40, ss.GRID_N)) for a in ss.ARMS}
        high = {a: np.full((40, ss.GRID_N), 5.0) for a in ss.ARMS}
        pl = sr.gate_plateaus(*self._rec(0.01), low)["main"]
        assert pl["conjunction"] and pl["gauss_only"] and pl["rank_only"] and pl["n_ref_min"] == 40
        pl = sr.gate_plateaus(*self._rec(0.01), high)["main"]
        assert not pl["conjunction"] and pl["gauss_only"] and not pl["rank_only"]
        pl = sr.gate_plateaus(*self._rec(0.5), low)["main"]
        assert not pl["conjunction"] and not pl["gauss_only"] and pl["rank_only"] and pl["without_b"]

    def test_verdict_counts_the_main_conjunction(self):
        cells = sr.gate_cells()
        hit = {a: {"conjunction": [1], "gauss_only": [1], "rank_only": [], "without_b": [1]} for a in ss.ARMS}
        miss = {a: {"conjunction": [], "gauss_only": [], "rank_only": [], "without_b": [1]} for a in ss.ARMS}
        for n_hit, ok in ((2, True), (3, False)):
            v = sr.gate_verdict(cells, [hit] * n_hit + [miss] * (len(cells) - n_hit))
            assert v["hits"] == n_hit and v["pass"] is ok
            assert v["counts"]["main"]["without_b"]["all"] == 50
            assert v["counts"]["main"]["conjunction"]["L1-8"] == n_hit


class TestTrainedReading:
    """Blocked 20, option (a): the window, partitions, replication, and the refusals."""

    def test_window_drops_uninformative_and_unbeatable_points(self):
        rows = [_row(1.0) for _ in range(6)]
        rows[1] = _row(1.0, informative=False)
        refs = np.zeros((40, 6))
        refs[:2, 4] = np.inf  # two references a finite z cannot beat: 3/41 > 0.05
        refs[:1, 5] = np.inf  # one: 2/41 <= 0.05, still readable
        w = sr.window(rows, refs)
        assert w["points"] == [0, 2, 3, 5] and w["longest_run"] == 2 and not w["readable"]
        refs[:2, 4] = 0.0
        w = sr.window(rows, refs)
        assert w["points"] == [0, 2, 3, 4, 5] and w["longest_run"] == 4 and w["readable"]

    def test_window_is_the_points_where_b_can_pass(self):
        # Whatever the cloud's own z, a readable point with every reference below gives p_rank = 1/41.
        rows = [_row(-3.0, p=0.01) for _ in range(4)]
        refs = np.full((40, 4), -np.inf)
        assert sr.window(rows, refs)["points"] == [0, 1, 2, 3]
        assert all(r["p_rank"] <= ss.ALPHA for r in sr.rank_rows(rows, refs))

    def test_partition_maps_kept_positions_to_tokens(self):
        lab = np.array([0, 0, 1, 2, 2, 2])
        kept = [3, 5, 6, 8, 9, 11]
        tokens = [f"t{i}" for i in range(12)]
        assert sr.partition(lab, 2, kept, tokens) == [
            {"positions": [8, 9, 11], "tokens": ["t8", "t9", "t11"]},
            {"positions": [3, 5], "tokens": ["t3", "t5"]}]
        assert sr.partition(lab, 4, kept, tokens) == []

    def test_pair_ari_over_tokens_clustered_in_both(self):
        a = np.array([0, 0, 1, 1, 2])
        assert sr.pair_ari(a, a, 2) == 1.0
        assert sr.pair_ari(a, np.array([5, 5, 7, 7, 9]), 2) == 1.0  # labels are names
        assert sr.pair_ari(a, np.arange(5), 2) is None

    def test_replication_counts_inits_per_cell(self):
        lab = np.tile(np.repeat([0, 1], 5), (ss.GRID_N, 1))
        hit = {a: {"conjunction": [{"start": 0}]} for a in ss.ARMS}
        miss = {a: {"conjunction": []} for a in ss.ARMS}
        cells = [{"model": m, "prompt": "wiki_paragraph", "layer": 1, "plateaus": hit if i < 3 else miss}
                 for i, m in enumerate(INITS)]
        labs = {m: {"wiki_paragraph": {1: lab}} for m in INITS}
        r = sr.replication(cells, labs, "main")
        assert r["L1-8"] == {"cells_by_n_inits": {3: 1}, "n_pairs": 3, "median_pair_ari": 1.0}
        assert r["L9-16"]["cells_by_n_inits"] == {} and r["L9-16"]["median_pair_ari"] is None

    def test_run_refuses_reinits_when_trained(self):
        with pytest.raises(SystemExit, match="do not exist"):
            sr.run_cmd(["--out", "x", "--step", "step143000", "--only", "reinit:0"])

    def _write(self, out, mids, step, sha="x"):
        for m in mids:
            rec, lab = _cloud(0.0)
            sr._write(out, m, "wiki_paragraph", {1: ({"layer": 1, **rec}, lab)},
                      {"git": "abc", "sets_sha256": sha, "step": step, "sets": "nowhere"})

    def test_trained_refuses_wrong_steps_and_other_token_sets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr, "V1_PASSAGES", ("wiki_paragraph",))
        out, ref = tmp_path / "t", tmp_path / "r"
        self._write(out, INITS, "step0")
        self._write(ref, INITS + REINITS, "step0")
        with pytest.raises(SystemExit, match="steps"):
            sr.trained_cmd(["--out", str(out), "--ref", str(ref)])
        self._write(out, INITS, "step143000", sha="y")
        with pytest.raises(SystemExit, match="different token sets"):
            sr.trained_cmd(["--out", str(out), "--ref", str(ref)])

    def test_point_failures_flags_a_free_rank_term_and_each_condition(self):
        rows = [_row(1.0, p=0.01), _row(1.0, stability=0.5, p=0.01), _row(1.0, p=0.5)]
        refs = np.full((40, 3), -np.inf)
        refs[:, 2] = 0.0
        pts = sr.point_failures(rows, refs)
        assert [p["rank_free"] for p in pts] == [True, True, False]
        assert [p["stable"] for p in pts] == [True, False, True]
        assert [p["gauss"] for p in pts] == [True, True, False]
        assert [p["admissible"] for p in pts] == [True, False, False]
        assert all(p["window"] for p in pts)
