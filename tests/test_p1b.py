"""
test_p1b.py — Comprehensive tests for Phase 1h (run_1b pipeline).

Run with:
    pytest tests/p1b_hemisphere/test_p1b.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
for _candidate in [_HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / "p1b_hemisphere" / "__init__.py").exists() or (
        _candidate / "p1b_hemisphere" / "bipartition_detect.py"
    ).exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _l2(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return X / norms


def make_antipodal(n_layers=4, n_tokens=20, d=8, rng=None):
    rng = rng or np.random.default_rng(42)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        a = +v + 0.05 * rng.standard_normal((half, d))
        b = -v + 0.05 * rng.standard_normal((n_tokens - half, d))
        acts[L] = np.vstack([a, b])
    return _l2(acts)


def make_cone(n_layers=4, n_tokens=20, d=8, rng=None):
    rng = rng or np.random.default_rng(7)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        acts[L] = v + 0.05 * rng.standard_normal((n_tokens, d))
    return _l2(acts)


def make_birth_then_stable(n_layers=8, n_tokens=40, d=16, rng=None):
    rng = rng or np.random.default_rng(55)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        if L < 2:
            acts[L] = v + 0.02 * rng.standard_normal((n_tokens, d))
        else:
            a = +v + 0.04 * rng.standard_normal((half, d))
            b = -v + 0.04 * rng.standard_normal((n_tokens - half, d))
            acts[L] = np.vstack([a, b])
    return _l2(acts)


def make_collapse_at_end(n_layers=8, n_tokens=40, d=16, rng=None):
    rng = rng or np.random.default_rng(99)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        if L < 6:
            a = +v + 0.04 * rng.standard_normal((half, d))
            b = -v + 0.04 * rng.standard_normal((n_tokens - half, d))
            acts[L] = np.vstack([a, b])
        else:
            acts[L] = v + 0.01 * rng.standard_normal((n_tokens, d))
    return _l2(acts)


# ===========================================================================
# Block 0  bipartition_detect
# ===========================================================================

class TestBipartitionOutputShapes:
    def test_output_keys(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        required = {
            "eigvals", "fiedler_vecs", "valid", "assignments",
            "hemisphere_sizes", "minority_fraction", "bipartition_eigengap",
            "centroid_angle", "within_half_ip", "between_half_ip",
            "separation_ratio", "fiedler_boundary_frac", "clip_fraction",
            "regime", "n_layers", "n_tokens", "thresholds",
        }
        assert required <= set(r.keys()), f"Missing keys: {required - set(r.keys())}"

    def test_array_shapes(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        nL, nT, d = 5, 30, 12
        acts = make_antipodal(n_layers=nL, n_tokens=nT, d=d)
        r = analyze_bipartition(acts)
        assert r["eigvals"].shape           == (nL, 3)
        assert r["fiedler_vecs"].shape      == (nL, nT)
        assert r["valid"].shape             == (nL,)
        assert r["assignments"].shape       == (nL, nT)
        assert r["hemisphere_sizes"].shape  == (nL, 2)
        assert r["minority_fraction"].shape == (nL,)
        assert r["bipartition_eigengap"].shape == (nL,)
        assert r["centroid_angle"].shape    == (nL,)
        assert r["within_half_ip"].shape    == (nL, 2)
        assert r["between_half_ip"].shape   == (nL,)
        assert r["separation_ratio"].shape  == (nL,)
        assert r["fiedler_boundary_frac"].shape == (nL,)
        assert r["clip_fraction"].shape     == (nL,)
        assert r["regime"].shape            == (nL,)

    def test_n_layers_n_tokens_fields(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=3, n_tokens=24, d=8)
        r = analyze_bipartition(acts)
        assert r["n_layers"] == 3
        assert r["n_tokens"] == 24

    def test_hemisphere_sizes_sum_to_n_tokens(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        for L in range(4):
            if r["valid"][L]:
                assert r["hemisphere_sizes"][L].sum() == 20


class TestBipartitionRegimeClassification:
    def setup_method(self):
        from p1b_hemisphere.bipartition_detect import classify_regime
        self.cr = classify_regime

    def test_collapsed_minority(self):
        assert self.cr(0.04, np.pi, 0.9, 0.9) == "collapsed"

    def test_collapsed_at_exact_threshold(self):
        assert self.cr(0.049, np.pi, 0.9, 0.9) == "collapsed"

    def test_weak_bipartition_minority_range(self):
        assert self.cr(0.07, np.pi, 0.9, 0.9) == "weak_bipartition"

    def test_weak_bipartition_small_angle(self):
        assert self.cr(0.30, np.pi / 3, 0.9, 0.9) == "weak_bipartition"

    def test_diffuse_low_within_both(self):
        assert self.cr(0.30, np.pi, 0.1, 0.1) == "diffuse"

    def test_diffuse_one_half_low(self):
        assert self.cr(0.30, np.pi, 0.5, 0.2) == "diffuse"

    def test_strong_bipartition(self):
        assert self.cr(0.30, np.pi, 0.5, 0.5) == "strong_bipartition"

    def test_strong_bipartition_at_within_ip_threshold(self):
        assert self.cr(0.30, np.pi, 0.30, 0.30) == "strong_bipartition"

    def test_nan_input_collapses(self):
        assert self.cr(float("nan"), np.pi, 0.5, 0.5) == "collapsed"
        assert self.cr(0.30, float("nan"), 0.5, 0.5)  == "collapsed"
        assert self.cr(0.30, np.pi, float("nan"), 0.5) == "collapsed"


class TestBipartitionNewFields:
    def test_between_half_ip_negative_for_antipodal(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["between_half_ip"][np.isfinite(r["between_half_ip"])]
        assert finite.size > 0
        assert (finite < 0).all(), finite

    def test_between_half_ip_positive_for_cone(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_cone(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["between_half_ip"][np.isfinite(r["between_half_ip"])]
        if finite.size:
            assert (finite > 0).all(), finite

    def test_separation_ratio_negative_for_antipodal(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["separation_ratio"][np.isfinite(r["separation_ratio"])]
        assert finite.size > 0
        assert (finite < 0).all(), finite

    def test_fiedler_boundary_frac_low_for_antipodal(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        valid_bf = r["fiedler_boundary_frac"][r["valid"]]
        assert valid_bf.size > 0
        assert valid_bf.mean() < 0.40, valid_bf

    def test_clip_fraction_finite_and_in_unit_interval(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        cf = r["clip_fraction"]
        assert np.all(np.isfinite(cf))
        assert np.all(cf >= 0) and np.all(cf <= 1)

    def test_clip_fraction_zero_when_no_clipping(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_cone(n_layers=3, n_tokens=20, d=8)
        r = analyze_bipartition(acts, clip_negative=False)
        cf = r["clip_fraction"]
        assert np.all((cf == 0) | ~np.isfinite(cf))

    def test_clip_fraction_denominator_fix(self):
        """
        Regression: clip_fraction was computed as n_neg / n*(n-1) instead of
        n_neg / (n*(n-1)//2), halving the reported value.  For a geometry where
        all cross-hemisphere pairs are negative (pure antipodal, ~50% minority),
        the corrected clip_fraction should be ≥ 0.20 (roughly minority_frac^2 * 2).
        """
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        acts = make_antipodal(n_layers=3, n_tokens=40, d=8)
        r = analyze_bipartition(acts)
        cf_valid = r["clip_fraction"][r["valid"]]
        assert cf_valid.size > 0
        # With ~50% minority and genuinely antipodal geometry, at least ~25% of
        # upper-triangle pairs span the two hemispheres and are negative.
        assert cf_valid.mean() > 0.20, (
            f"clip_fraction too low ({cf_valid.mean():.3f}); "
            "may indicate denominator regression."
        )

    def test_bipartition_to_json_serialisable(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition, bipartition_to_json
        acts = make_antipodal(n_layers=3, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        j = bipartition_to_json(r)
        json.dumps(j)   # must not raise


# ===========================================================================
# Block 1  hemisphere_tracking
# ===========================================================================

class TestAlignHemisphereLabels:
    def test_identity_path(self):
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels
        n_L, n_T = 4, 10
        a = np.zeros((n_L, n_T), dtype=np.int8)
        a[:, n_T // 2:] = 1
        valid = np.ones(n_L, dtype=bool)
        out = align_hemisphere_labels(a, valid)
        assert np.array_equal(out["aligned_assignments"], a)
        assert not out["flips_applied"].any()

    def test_forced_flip(self):
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels
        n_L, n_T = 3, 10
        a = np.zeros((n_L, n_T), dtype=np.int8)
        a[:, n_T // 2:] = 1
        a[1] = 1 - a[1]   # flipped layer
        valid = np.ones(n_L, dtype=bool)
        out = align_hemisphere_labels(a, valid)
        assert out["flips_applied"][1]
        assert np.array_equal(out["aligned_assignments"][1], a[0])


class TestAxisRotation:
    def test_zero_rotation_for_identical_vecs(self):
        from p1b_hemisphere.hemisphere_tracking import compute_axis_rotation
        fv = np.ones((4, 10), dtype=np.float64)
        valid = np.ones(4, dtype=bool)
        ar = compute_axis_rotation(fv, valid)
        assert np.allclose(ar, 0.0, atol=1e-6)

    def test_nan_at_invalid_transition(self):
        from p1b_hemisphere.hemisphere_tracking import compute_axis_rotation
        fv = np.ones((4, 10), dtype=np.float64)
        valid = np.array([True, False, True, True])
        ar = compute_axis_rotation(fv, valid)
        assert np.isnan(ar[0])   # L=0→1: valid[1]=False
        assert np.isnan(ar[1])   # L=1→2: valid[1]=False
        assert np.isfinite(ar[2])


class TestCumulativeRotation:
    def test_monotone_non_decreasing(self):
        from p1b_hemisphere.hemisphere_tracking import compute_cumulative_rotation
        ar = np.array([0.1, 0.2, np.nan, 0.3])
        cr = compute_cumulative_rotation(ar)
        assert cr[0] == pytest.approx(0.1)
        assert cr[1] == pytest.approx(0.3)
        assert cr[2] == pytest.approx(0.3)   # nan contributes 0
        assert cr[3] == pytest.approx(0.6)


class TestPersistenceLengths:
    def test_monotone_increasing_with_no_disruptions(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        pl = b1["persistence_length"]
        strong_mask = np.array([str(r) == "strong_bipartition" for r in b0["regime"]])
        if strong_mask.all():
            strong_pl = pl[strong_mask]
            assert (strong_pl > 0).all()
            assert (np.diff(strong_pl) >= 0).all()


class TestEventDetection:
    def _make_regime(self, n, *patches):
        regime = np.array(["strong_bipartition"] * n, dtype=object)
        for start, end, label in patches:
            regime[start:end] = label
        return regime

    def test_birth_event(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 8
        regime = self._make_regime(n, (0, 2, "collapsed"))
        mo, cc = np.full(n - 1, 0.9), np.zeros(n - 1, dtype=np.int32)
        valid, ar = np.ones(n, dtype=bool), np.full(n - 1, 0.01)
        births = [e for e in detect_events(regime, mo, cc, valid, ar) if e["type"] == "birth"]
        assert len(births) >= 1 and births[0]["layer"] == 2

    def test_collapse_event(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 8
        regime = self._make_regime(n, (5, 8, "collapsed"))
        mo, cc = np.full(n - 1, 0.9), np.zeros(n - 1, dtype=np.int32)
        valid, ar = np.ones(n, dtype=bool), np.full(n - 1, 0.01)
        collapses = [e for e in detect_events(regime, mo, cc, valid, ar) if e["type"] == "collapse"]
        assert len(collapses) >= 1 and collapses[0]["layer"] == 5

    def test_swap_event(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 6
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9); mo[2] = 0.3
        cc = np.zeros(n - 1, dtype=np.int32)
        valid, ar = np.ones(n, dtype=bool), np.full(n - 1, 0.01)
        swaps = [e for e in detect_events(regime, mo, cc, valid, ar) if e["type"] == "swap"]
        assert len(swaps) >= 1 and swaps[0]["layer"] == 3

    def test_shear_event(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 10
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9)
        cc = np.zeros(n - 1, dtype=np.int32); cc[5] = 30
        valid, ar = np.ones(n, dtype=bool), np.full(n - 1, 0.01)
        shears = [e for e in detect_events(regime, mo, cc, valid, ar, shear_absolute_floor=3)
                  if e["type"] == "shear"]
        assert len(shears) >= 1 and shears[0]["layer"] == 6

    def test_drift_event(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 12
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9)
        cc = np.zeros(n - 1, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.02); ar[3:8] = 0.40
        drifts = [e for e in detect_events(regime, mo, cc, valid, ar,
                  drift_window_layers=5, drift_window_rad=1.5)
                  if e["type"] == "drift"]
        assert len(drifts) >= 1

    def test_no_spurious_events_stable(self):
        from p1b_hemisphere.hemisphere_tracking import detect_events
        n = 8
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.95)
        cc = np.ones(n - 1, dtype=np.int32)
        valid, ar = np.ones(n, dtype=bool), np.full(n - 1, 0.01)
        critical = [e for e in detect_events(regime, mo, cc, valid, ar, shear_absolute_floor=3)
                    if e["type"] in ("swap", "birth", "collapse")]
        assert not critical, critical


class TestCrossrefPhase1:
    def test_event_tagged_at_merge(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1
        events = [{"type": "swap", "layer": 3, "from_layer": 2, "detail": {}}]
        ar, cc = np.full(5, 0.01), np.zeros(5, dtype=np.int32)
        out = crossref_phase1(events, ar, cc, merge_transition_indices={2})
        # FIX: key is now "at_merge" (was "merge_at_transition")
        assert out["events"][0]["detail"]["phase1"]["at_merge"] is True

    def test_event_tagged_at_violation_layer(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1
        events = [{"type": "birth", "layer": 4, "from_layer": 3, "detail": {}}]
        ar, cc = np.full(6, 0.01), np.zeros(6, dtype=np.int32)
        out = crossref_phase1(events, ar, cc, violation_layers={4})
        assert out["events"][0]["detail"]["phase1"]["at_violation_layer"] is True

    def test_no_tag_when_no_phase1_data(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1
        events = [{"type": "collapse", "layer": 2, "from_layer": 1, "detail": {}}]
        ar, cc = np.full(4, 0.01), np.zeros(4, dtype=np.int32)
        out = crossref_phase1(events, ar, cc)
        p1 = out["events"][0]["detail"].get("phase1", {})
        assert p1.get("at_merge") is False
        assert p1.get("at_violation_layer") is False

    def test_aggregates_present(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1
        out = crossref_phase1([], np.full(4, 0.01), np.zeros(4, dtype=np.int32))
        assert "agg" in out


class TestHemisphereTrackingEndToEnd:
    def test_required_keys(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b1 = analyze_hemisphere_tracking(analyze_bipartition(acts))
        for k in ("aligned_assignments", "flips_applied", "match_overlap",
                  "axis_rotation", "cumulative_axis_rotation", "crossing_count",
                  "persistence_length", "events", "crossref", "thresholds"):
            assert k in b1, f"Missing key: {k}"

    def test_birth_in_pipeline(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        b1 = analyze_hemisphere_tracking(analyze_bipartition(
            make_birth_then_stable(n_layers=8, n_tokens=40, d=16)))
        assert any(e["type"] == "birth" for e in b1["events"]), b1["events"]

    def test_collapse_in_pipeline(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        b1 = analyze_hemisphere_tracking(analyze_bipartition(
            make_collapse_at_end(n_layers=8, n_tokens=40, d=16)))
        assert any(e["type"] == "collapse" for e in b1["events"]), b1["events"]


# ===========================================================================
# Block 2  hemisphere_membership
# ===========================================================================

class TestTokenTrajectories:
    def test_high_stability_for_antipodal(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories
        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"])
        ss = traj["stability_score"]
        finite_ss = ss[np.isfinite(ss)]
        assert finite_ss.size > 0
        assert (finite_ss > 0.9).all(), finite_ss

    def test_output_shapes(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories
        n_T = 25
        b0 = analyze_bipartition(make_antipodal(n_layers=4, n_tokens=n_T, d=8))
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"])
        assert traj["stability_score"].shape     == (n_T,)
        assert traj["border_index"].shape        == (n_T,)
        assert traj["first_stable_layer"].shape  == (n_T,)
        assert traj["dominant_hemisphere"].shape == (n_T,)

    def test_dominant_hemisphere_values(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories
        b0 = analyze_bipartition(make_antipodal(n_layers=5, n_tokens=30, d=16))
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"])
        assert set(np.unique(traj["dominant_hemisphere"])).issubset({-1, 0, 1})

    def test_all_invalid_layers_returns_nan_stability(self):
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories
        n_L, n_T = 4, 10
        aligned = np.full((n_L, n_T), -1, dtype=np.int8)
        fiedler = np.zeros((n_L, n_T))
        valid   = np.zeros(n_L, dtype=bool)
        traj = compute_token_trajectories(aligned, fiedler, valid)
        assert np.all(np.isnan(traj["stability_score"]))


class TestHdbscanNesting:
    def _make_assignments(self, n_L, n_T, half):
        aa = np.zeros((n_L, n_T), dtype=np.int8); aa[:, half:] = 1
        return aa

    def test_fully_nested(self):
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting
        n_L, n_T, half = 4, 20, 10
        aa = self._make_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)
        labels = {L: np.array([0]*half + [1]*(n_T-half), dtype=np.int32) for L in range(n_L)}
        overall = compute_hdbscan_nesting(aa, labels, valid)["overall"]
        assert overall["fully_nested_fraction"] == 1.0
        assert overall["mixed_fraction"]        == 0.0

    def test_mixed_clusters(self):
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting
        n_L, n_T, half = 4, 20, 10
        aa = self._make_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)
        mixed = np.array([0]*5 + [0]*5 + [1]*5 + [1]*5, dtype=np.int32)
        labels = {L: mixed for L in range(n_L)}
        overall = compute_hdbscan_nesting(aa, labels, valid)["overall"]
        assert overall["mixed_fraction"] > 0.0

    def test_nesting_classes_semantics(self):
        """r_c near 0 → nested_B (hemi 1); r_c near 1 → nested_A (hemi 0)."""
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting
        n_L, n_T = 2, 20
        aa = np.zeros((n_L, n_T), dtype=np.int8); aa[:, 10:] = 1
        valid = np.ones(n_L, dtype=bool)
        # Cluster 0: all in hemisphere 1 (labels 10–19 → hemi 1 → r_c ≈ 0)
        # Cluster 1: all in hemisphere 0 (labels 0–9  → hemi 0 → r_c ≈ 1)
        lbl = np.array([1]*10 + [0]*10, dtype=np.int32)  # cluster 1 first, cluster 0 second
        labels = {L: lbl for L in range(n_L)}
        result = compute_hdbscan_nesting(aa, labels, valid)
        for L, layer_data in result["per_layer"].items():
            for rec in layer_data["clusters"]:
                if rec["r_c"] < 0.1:
                    assert rec["nesting_class"] == "nested_B", rec
                elif rec["r_c"] > 0.9:
                    assert rec["nesting_class"] == "nested_A", rec

    def test_plateau_layer_filtering(self):
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting
        n_L, n_T, half = 6, 20, 10
        aa = self._make_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)
        labels = {L: np.array([0]*half + [1]*(n_T-half)) for L in range(n_L)}
        result = compute_hdbscan_nesting(aa, labels, valid, plateau_layers=[2, 4])
        assert set(result["per_layer"].keys()) == {2, 4}


class TestMembershipToJson:
    def test_trajectory_is_actual_sequence(self):
        """
        FIX regression: hemisphere_trajectory must be the real per-layer
        assignment list, not the placeholder [n_layers].
        """
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership, membership_to_json,
        )
        n_L, n_T = 5, 30
        acts = make_antipodal(n_layers=n_L, n_tokens=n_T, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j  = membership_to_json(b2)

        for entry in j["per_token"]:
            traj = entry["hemisphere_trajectory"]
            assert traj is not None, "hemisphere_trajectory is None"
            assert len(traj) == n_L, (
                f"Expected trajectory of length {n_L}, got {len(traj)}"
            )
            assert all(v in (-1, 0, 1) for v in traj), (
                f"Unexpected values in trajectory: {set(traj)}"
            )

    def test_per_token_sorted_by_stability(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership, membership_to_json,
        )
        acts = make_antipodal(n_layers=5, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j  = membership_to_json(b2)
        scores = [e["stability_score"] for e in j["per_token"]
                  if e["stability_score"] is not None]
        assert scores == sorted(scores), "per_token not sorted by stability_score"


# ===========================================================================
# Block 3  cone_collapse
# ===========================================================================
#
# FIX: All tests in this class previously called analyze_cone_collapse and
# then accessed result["per_layer"], which does not exist in its return dict.
# The per_layer structure is produced by cone_collapse_to_json.
# Every test now wraps with cone_collapse_to_json before inspecting per_layer.

class TestClassifyConeRegime:
    def setup_method(self):
        from p1b_hemisphere.cone_collapse import classify_cone_regime
        self.cr = classify_cone_regime

    def test_large_positive_is_cone_collapse(self):
        assert self.cr(0.5)  == "cone_collapse"

    def test_large_negative_is_split(self):
        assert self.cr(-0.5) == "split"

    def test_near_zero_is_borderline(self):
        assert self.cr(0.0)  == "borderline"

    def test_nan_is_invalid(self):
        assert self.cr(float("nan")) == "invalid"

    def test_just_above_tol_is_collapse(self):
        from p1b_hemisphere.cone_collapse import CONE_BORDERLINE_TOL
        assert self.cr(CONE_BORDERLINE_TOL + 1e-6) == "cone_collapse"

    def test_just_below_neg_tol_is_split(self):
        from p1b_hemisphere.cone_collapse import CONE_BORDERLINE_TOL
        assert self.cr(-CONE_BORDERLINE_TOL - 1e-6) == "split"


class TestAnalyzeConeCollapse:
    # Helper: run analyze + to_json so per_layer is accessible.
    @staticmethod
    def _run(acts, **kw):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse, cone_collapse_to_json
        return cone_collapse_to_json(analyze_cone_collapse(acts, **kw))  # FIX

    def test_cone_geometry_yields_cone_collapse(self):
        result = self._run(make_cone(n_layers=4, n_tokens=30, d=8))
        solved_regimes = [e["cone_regime"] for e in result["per_layer"] if e["solved"]]
        assert solved_regimes, "No LP solved"
        assert all(r == "cone_collapse" for r in solved_regimes), solved_regimes

    def test_antipodal_geometry_yields_split(self):
        result = self._run(make_antipodal(n_layers=4, n_tokens=30, d=8))
        solved_regimes = [e["cone_regime"] for e in result["per_layer"] if e["solved"]]
        assert solved_regimes, "No LP solved"
        assert all(r == "split" for r in solved_regimes), solved_regimes

    def test_output_shape_per_layer(self):
        n_L = 5
        result = self._run(make_antipodal(n_layers=n_L, n_tokens=20, d=8))
        assert len(result["per_layer"]) == n_L  # FIX: was KeyError before

    def test_per_layer_keys(self):
        result = self._run(make_cone(n_layers=3, n_tokens=20, d=8))
        for entry in result["per_layer"]:
            for k in ("layer", "cone_regime", "cone_margin", "solved", "lp_at_limit"):
                assert k in entry, f"Missing per-layer key: {k}"

    def test_summary_keys(self):
        result = self._run(make_cone(n_layers=4, n_tokens=20, d=8))
        for k in ("n_layers", "n_tokens", "regime_counts",
                  "cone_collapse_fraction", "split_fraction", "n_lp_at_limit"):
            assert k in result["summary"], f"Missing summary key: {k}"

    def test_summary_fractions_sum_le_1(self):
        s = self._run(make_cone(n_layers=4, n_tokens=20, d=8))["summary"]
        assert s["cone_collapse_fraction"] + s["split_fraction"] <= 1.0 + 1e-9

    def test_lp_at_limit_field_is_bool(self):
        result = self._run(make_cone(n_layers=3, n_tokens=20, d=8))
        for entry in result["per_layer"]:
            assert isinstance(entry["lp_at_limit"], bool)

    def test_valid_mask_respected(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse, cone_collapse_to_json
        acts  = make_cone(n_layers=4, n_tokens=20, d=8)
        valid = np.array([True, False, True, True])
        result = cone_collapse_to_json(analyze_cone_collapse(acts, valid=valid))
        assert result["per_layer"][1]["cone_regime"] == "invalid"

    def test_cone_collapse_to_json_serialisable(self):
        json.dumps(self._run(make_cone(n_layers=3, n_tokens=20, d=8)))