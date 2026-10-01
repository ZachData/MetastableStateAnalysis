"""
tests/test_phase1d_admit.py — admission (`p1d_cluster_ensemble/admit.py`):
level-set HDBSCAN against hdbscan's own tree, planted caps admitted, a pure
Gaussian admitting at most alpha, and the two-group invariance that ruled
out `cluster_persistence_` and the shipped call's tie order (`design-1d.md`
build step 1).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble.admit import (
    NOT_ADMITTED, NOT_TESTED, _job, _mst, admit_record, branch,
    condense_and_select, fit_hdbscan, labels_out, layer_groups, level_set_hdbscan,
    mutual_reachability, rank_p, shipped_check, table,
)
from p1d_cluster_ensemble.methods import LayerData, _fit_hdbscan


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _background(n=100, d=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 0] += 2.0
    return X


def _cap(axis, k, spread, d=30, seed=1):
    rng = np.random.default_rng(seed)
    c = np.zeros(d)
    c[axis] = 6.0
    return c + spread * rng.standard_normal((k, d))


#: Background (rows 0-99), a looser cap C (rows 100-107) and, far from both,
#: a tighter cap T (rows 108-115).
BG, C, T = _background(), _cap(1, 8, 0.25, seed=1), _cap(2, 8, 0.05, seed=2)


def _group_of(rows, member):
    return next(r for r in rows if member in r["members"])


def _binary_tree(single_linkage, n):
    """hdbscan's single-linkage tree (scipy format) as `condense_and_select`'s nodes."""
    nodes = {i: (None, (i,), 1) for i in range(n)}
    for i, (a, b, d, size) in enumerate(single_linkage):
        nodes[n + i] = (float(d), (nodes.pop(int(a)), nodes.pop(int(b))), int(size))
    (tree,) = nodes.values()
    return tree


def _three_blobs(seed, n=80):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    X[:25, 0] += 3.0
    X[25:40, 1] += 4.0
    return LayerData.from_normed(_unit(X)).cos_dist


class TestGroupStats:
    @pytest.mark.parametrize("mcs", [2, 4])
    def test_mutual_reachability_is_hdbscans(self, mcs):
        """Same labels as Phase 1's call, and our graph's MST weights are
        hdbscan's single-linkage heights."""
        data = LayerData.from_normed(_unit(np.vstack([BG, C, T])))
        h = fit_hdbscan(data.cos_dist, mcs)
        np.testing.assert_array_equal(
            h.labels_, _fit_hdbscan({"min_cluster_size": mcs}, data, 0))
        w = [e[0] for e in _mst(mutual_reachability(data.cos_dist, mcs))]
        np.testing.assert_allclose(sorted(w), sorted(h.single_linkage_tree_.to_numpy()[:, 2]))

    @pytest.mark.parametrize("mcs", [2, 4])
    def test_reproduces_hdbscan_on_its_own_tree(self, mcs):
        """Handed hdbscan's binary tree, condensing + EOM give hdbscan's labels
        and its persistence times max lambda, so the level-set version differs
        from the shipped call only in how tied edges are merged. (While
        building: 400 of 400 fits, 1257 groups.)"""
        from sklearn.metrics import adjusted_rand_score
        n_groups = 0
        for seed in range(20):
            D = _three_blobs(seed)
            h = fit_hdbscan(D, mcs)
            labels, rows = condense_and_select(
                _binary_tree(h.single_linkage_tree_.to_numpy(), D.shape[0]), mcs)
            assert adjusted_rand_score(labels, h.labels_) == 1.0
            np.testing.assert_array_equal(labels == -1, h.labels_ == -1)
            max_lambda = h.condensed_tree_.to_numpy()["lambda_val"].max()
            for r in rows:
                assert r["excess"] / max_lambda == pytest.approx(
                    h.cluster_persistence_[h.labels_[r["members"][0]]], rel=1e-9)
            n_groups += len(rows)
        assert n_groups >= 20

    def test_hand_example(self):
        """Two triples, internal distances 0.1 and 0.2, 1.0 between. Every
        core distance is the internal one, so each triple is born at
        lambda 1 and all three points leave at once: S = 3 (10 - 1), 3 (5 - 1)."""
        D = np.full((6, 6), 1.0)
        D[:3, :3], D[3:, 3:] = 0.1, 0.2
        np.fill_diagonal(D, 0.0)
        labels, rows = level_set_hdbscan(D, 2)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
        assert [r["stability"] for r in rows] == pytest.approx([27.0, 12.0])
        assert [r["excess"] for r in rows] == pytest.approx([9.0, 4.0])
        assert [r["log_life"] for r in rows] == pytest.approx([np.log(10), np.log(5)])

    @pytest.mark.parametrize("mcs", [2, 4])
    def test_groups_are_level_set_components_and_branch_agrees(self, mcs):
        """Brute force: every level-set group, and no shipped tie artefact, is
        a connected component of {mutual reachability <= t} for some t; and
        `branch`, from the group's own edges alone, gives the same row."""
        from scipy.sparse.csgraph import connected_components
        D = _three_blobs(5)
        mr = mutual_reachability(D, mcs)
        comps = set()
        for t in np.unique(mr[np.triu_indices_from(mr, 1)]):
            _, lab = connected_components(mr <= t, directed=False)
            comps |= {frozenset(np.flatnonzero(lab == c).tolist()) for c in np.unique(lab)}
        _, rows = level_set_hdbscan(D, mcs)
        assert rows
        for r in rows:
            assert frozenset(r["members"]) in comps
            b = branch(mr, r["members"], mcs)
            for k in ("birth", "death", "stability"):
                assert b[k] == pytest.approx(r[k], rel=1e-12)
        shipped = fit_hdbscan(D, mcs).labels_
        groups = sorted(set(shipped.tolist()) - {-1})
        not_comps = sum(frozenset(np.flatnonzero(shipped == g).tolist()) not in comps
                        for g in groups)
        assert shipped_check(D, shipped, mcs)["tie_artefacts"] == not_comps

    def test_refuses_duplicate_points(self):
        Z = _unit(np.vstack([BG[:20], BG[:1]]))
        with pytest.raises(ValueError, match="zero distance"):
            layer_groups(Z, 2)

    def test_rank_p(self):
        assert rank_p(5.0, np.array([1.0, 5.0, 6.0])) == pytest.approx(3 / 4)
        assert rank_p(7.0, np.zeros(19)) == pytest.approx(1 / 20)


class TestInvariance:
    """The defect that ruled out `cluster_persistence_` (`/challenge-pr` on #121)."""

    def test_looser_group_unchanged_by_a_tighter_one(self):
        rows_c = layer_groups(_unit(np.vstack([BG, C])), 2)[1]
        rows_ct = layer_groups(_unit(np.vstack([BG, C, T])), 2)[1]
        a, b = _group_of(rows_c, 100), _group_of(rows_ct, 100)
        assert a["members"] == list(range(100, 108)) == b["members"]
        assert b["excess"] == pytest.approx(a["excess"], rel=1e-12)
        assert b["log_life"] == pytest.approx(a["log_life"], rel=1e-12)

    def test_verdict_unchanged_and_persistence_is_not(self):
        rec_c = admit_record(_unit(np.vstack([BG, C])), "raw", 99, 0, min_cluster_sizes=(2,))
        rec_ct = admit_record(_unit(np.vstack([BG, C, T])), "raw", 99, 0, min_cluster_sizes=(2,))
        for rec in (rec_c, rec_ct):
            g = _group_of(rec["arms"]["2"]["groups"], 100)
            assert g["admitted_excess"] and g["admitted_log_life"]
        # The reason for not using hdbscan's own number: for C it collapses.
        pers = []
        for X in (np.vstack([BG, C]), np.vstack([BG, C, T])):
            h = fit_hdbscan(LayerData.from_normed(_unit(X)).cos_dist, 2)
            pers.append(h.cluster_persistence_[h.labels_[100]])
        assert pers[1] < 0.1 * pers[0]


class TestAdmission:
    def test_planted_caps_admitted_background_not(self):
        rec = admit_record(_unit(np.vstack([BG, C, T])), "raw", 99, 0)
        for arm in ("2", "4"):
            rows = rec["arms"][arm]["groups"]
            for member in (100, 108):
                assert _group_of(rows, member)["admitted_excess"], (arm, member)
            assert len(rec["arms"][arm]["null"]["excess"]) == 99
        assert rec["arms"]["2"]["n_admitted"]["excess"] == 2

    def test_pure_gaussian_admits_at_most_alpha(self):
        """40 records of an anisotropic Gaussian, the null refitted (plug-in)
        to each: P(any admitted) <= 0.05 per record. Measured while building
        (100 records, B = 39, n 60 and 150): excess 0-3 %, log_life 0-6 %."""
        n_rec, hits = 40, {"2": 0, "4": 0}
        for s in range(n_rec):
            rng = np.random.default_rng(1000 + s)
            X = rng.standard_normal((60, 20)) * np.linspace(2.0, 0.3, 20)
            X[:, 0] += 1.5
            rec = admit_record(_unit(X), "raw", 19, s)
            for arm in hits:
                hits[arm] += rec["arms"][arm]["n_admitted"]["excess"] > 0
        assert all(h <= 0.05 * n_rec for h in hits.values()), hits

    def test_pure_gaussian_more_dims_than_points(self):
        """The real regime (n < d, centred): 30 records of 40 points in 80
        dimensions. Measured while building (60 records, B = 19, and 120 x
        300, B = 39): 0-1 admitting records, both arms and statistics."""
        n_rec, hits = 30, {"2": 0, "4": 0}
        for s in range(n_rec):
            rng = np.random.default_rng(5000 + s)
            X = rng.standard_normal((40, 80)) * np.geomspace(3.0, 0.1, 80)
            X[:, 0] += 2.0
            rec = admit_record(_unit(X), "centred", 19, s)
            for arm in hits:
                hits[arm] += rec["arms"][arm]["n_admitted"]["excess"] > 0
        assert all(h <= 0.05 * n_rec for h in hits.values()), hits

    def test_draws_are_gaussian_nulls(self):
        """Same seed, same draws: the shipped call's group counts equal
        `gaussian_null`'s hdb_k."""
        from p1d_cluster_ensemble.gaussian_null import null_record
        Y = _unit(np.vstack([BG, C]))
        for calibrate in (False, True):
            g = null_record(Y, "centred", 5, 3, calibrate=calibrate)
            a = admit_record(Y, "centred", 5, 3, min_cluster_sizes=(2,), calibrate=calibrate)
            assert a["arms"]["2"]["null"]["shipped_k"] == g["draws"]["hdb_k"]
            assert a["arms"]["2"]["shipped"]["k"] == g["stats"]["hdb_k"]["obs"]


def _fake_run(tmp_path, step, tokens, cap=True, n_layers=3, seed=0):
    d = tmp_path / "2026-01-01_00-00-00" / f"pythia-410m-{step}_wiki_paragraph"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    X = _background(len(tokens), seed=seed)
    if cap:
        X[-8:] = _cap(1, 8, 0.2, seed=seed + 1)
    acts = np.stack([X + 0.01 * rng.standard_normal(X.shape) for _ in range(n_layers)])
    np.savez(d / "activations.npz", activations=acts.astype(np.float32))
    (d / "geometry.json").write_text(json.dumps({"tokens": tokens}))
    return d


def _files(tmp_path, control_cap):
    """Real and calibration files over a trained run (planted cap) and its
    step-0 control (a cap only if ``control_cap``), L1, raw frame."""
    tokens = [f"t{i}" for i in range(60)]
    tokens[5] = tokens[3]                        # a later occurrence, not tested
    runs = [_fake_run(tmp_path / "a", "step143000", tokens),
            _fake_run(tmp_path / "b", "step0", tokens, cap=control_cap, seed=7)]
    settings = {"seed": 0, "alpha": 0.05, "min_cluster_sizes": [2, 4], "n_draws": 19,
                "inputs": [str(r) for r in runs]}
    real = [_job((str(r), 1, "raw", 19, 0, False, 0)) for r in runs]
    cal = [_job((str(r), 1, "raw", 19, 0, True, 0)) for r in runs]
    return {**settings, "records": real}, {**settings, "calibrate": True, "records": cal}


def _cell_of(rows, step):
    return next(r for r in rows if r["stat"] == "excess" and r["arm"] == "2"
                and r["step"] == step)


class TestDriverAndReport:
    def test_released_when_calibration_and_control_pass(self, tmp_path):
        R, Cal = _files(tmp_path, control_cap=False)
        trained = R["records"][0]
        assert trained["n_kept"] == 59 and 5 not in trained["keep"]
        rows = table(R, Cal)
        cell = _cell_of(rows, "step143000")
        assert cell["band"] == "L1-8" and cell["real"]["records_admitting"] == 1
        assert cell["cal"]["records_admitting"] == 0
        assert _cell_of(rows, "step0")["real"]["records_admitting"] == 0
        assert cell["released"] and cell["withheld"] is None
        assert _cell_of(rows, "step0")["withheld"] == "the control step"
        out = {(r["step"], r["arm"]): r for r in labels_out(R, rows)["records"]}
        lab = np.array(out[("step143000", 2)]["labels"])
        admitted = {g["label"] for g in trained["arms"]["2"]["groups"] if g["admitted_excess"]}
        assert lab.size == 60 and lab[5] == NOT_TESTED
        assert set(lab[trained["keep"]]) == {NOT_ADMITTED} | admitted
        assert "labels" not in out[("step0", 2)]

    def test_withheld_when_control_admits(self, tmp_path):
        R, Cal = _files(tmp_path, control_cap=True)
        rows = table(R, Cal)
        cell = _cell_of(rows, "step143000")
        assert cell["cal"]["records_admitting"] == 0
        assert not cell["released"] and "step0 control admits in 1 of 1" in cell["withheld"]

    def test_withheld_without_a_control(self, tmp_path):
        R, Cal = _files(tmp_path, control_cap=False)
        for d in (R, Cal):
            d["records"], d["inputs"] = d["records"][:1], d["inputs"][:1]
        cell = _cell_of(table(R, Cal), "step143000")
        assert not cell["released"] and "no step0 control" in cell["withheld"]

    def test_withheld_when_calibration_over_admits(self, tmp_path):
        R, _ = _files(tmp_path, control_cap=False)
        # A calibration that admits wherever the real file does.
        rows = table(R, {**R, "calibrate": True})
        cell = _cell_of(rows, "step143000")
        assert not cell["released"] and cell["withheld"].startswith("calibration admits")

    def test_refuses_mismatched_calibration(self):
        rec = {"step": "step0", "prompt": "x", "layer": 1, "info": {"frame": "raw"}}
        base = {"seed": 0, "alpha": 0.05, "min_cluster_sizes": [2, 4], "n_draws": 19,
                "inputs": ["a/x"], "records": [rec]}
        with pytest.raises(ValueError, match="calibrate"):
            table(base, base)
        for k, v, msg in (("seed", 1, "seed"), ("n_draws", 20, "n_draws"),
                          ("inputs", ["a/y"], "prompt sets"),
                          ("records", [{**rec, "layer": 2}], "different")):
            with pytest.raises(ValueError, match=msg):
                table(base, {**base, "calibrate": True, k: v})


class TestMinPosition:
    def test_drops_the_opening_and_keeps_dedup(self, tmp_path):
        tokens = [f"t{i}" for i in range(60)]
        tokens[12] = tokens[3]                   # later occurrence of a dropped string
        run = _fake_run(tmp_path, "step0", tokens, cap=False)
        rec = _job((str(run), 1, "raw", 5, 0, False, 10))
        assert rec["min_position"] == 10 and min(rec["keep"]) == 10
        assert 12 not in rec["keep"] and rec["n_kept"] == 49

    def test_report_refuses_a_calibration_at_another_min_position(self, tmp_path):
        R, Cal = _files(tmp_path, control_cap=False)
        with pytest.raises(ValueError, match="min_position"):
            table({**R, "min_position": 8}, Cal)
