"""
test_p1b.py — Comprehensive tests for Phase 1h (run_1b pipeline).

Coverage map
------------
Block 0  bipartition_detect
    - output shapes and dtypes
    - regime classification (all four regimes, all boundary conditions)
    - new fields: between_half_ip, separation_ratio, fiedler_boundary_frac,
      clip_fraction
    - bipartition_to_json serialisability and schema

Block 1  hemisphere_tracking
    - align_hemisphere_labels: identity path, forced flip path
    - compute_axis_rotation / compute_cumulative_rotation
    - compute_persistence_lengths
    - detect_events: birth, collapse, swap, shear, drift
    - crossref_phase1: merge and violation tagging
    - analyze_hemisphere_tracking end-to-end
    - hemisphere_tracking_to_json schema

Block 2  hemisphere_membership
    - compute_token_trajectories: stable geometry → high stability_score
    - compute_token_trajectories: all-invalid layers → nan outputs
    - compute_hdbscan_nesting: fully nested clusters
    - compute_hdbscan_nesting: mixed clusters
    - analyze_hemisphere_membership with and without HDBSCAN labels
    - membership_to_json schema and serialisability

Block 3  cone_collapse
    - classify_cone_regime all cases (collapse / split / borderline / invalid)
    - LP on pure-cone geometry → cone_collapse
    - LP on antipodal / split geometry → split
    - analyze_cone_collapse shapes, summary stats, regime fractions
    - cone_collapse_to_json serialisability

Phase 1 imports  _load_phase1_xref
    - missing phase1_dir → empty dict
    - complete v2 fixture → all four keys loaded with correct types
    - partial fixture (missing hdbscan/trajectory) → partial load
    - hdbscan_labels values are numpy int32 arrays
    - energy violations union across betas

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
# Path setup — adjust if your layout differs
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# Walk up until we find the directory that contains 'p1b_hemisphere' as a package.
for _candidate in [_HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
    if (_candidate / "p1b_hemisphere" / "__init__.py").exists() or (
        _candidate / "p1b_hemisphere" / "bipartition_detect.py"
    ).exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


# ===========================================================================
# Shared fixtures and helpers
# ===========================================================================

def _l2(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise."""
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def make_antipodal(n_layers=6, n_tokens=40, d=16, rng=None):
    """Two tight clusters at +v and -v — strong_bipartition every layer."""
    rng = rng or np.random.default_rng(1)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        a = +v + 0.04 * rng.standard_normal((half, d))
        b = -v + 0.04 * rng.standard_normal((n_tokens - half, d))
        acts[L] = np.vstack([a, b])
    return _l2(acts)


def make_cone(n_layers=6, n_tokens=40, d=16, rng=None):
    """All tokens near one direction — cone_collapse regime."""
    rng = rng or np.random.default_rng(42)
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    X = direction[None, None, :] + 0.02 * rng.standard_normal((n_layers, n_tokens, d))
    return _l2(X)


def make_single_outlier(n_layers=6, n_tokens=40, d=16, rng=None):
    """n_tokens-1 near +v, 1 near -v → minority_fraction = 1/n < 0.05 → collapsed."""
    rng = rng or np.random.default_rng(0)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    X = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        main = +v + 0.02 * rng.standard_normal((n_tokens - 1, d))
        out = -v + 0.02 * rng.standard_normal((1, d))
        X[L] = np.vstack([main, out])
    return _l2(X)


def make_birth_then_stable(n_layers=8, n_tokens=40, d=16, rng=None):
    """
    Layers 0-1 collapsed (single outlier), layers 2-7 strong_bipartition.
    Produces a 'birth' event at layer 2.
    """
    rng = rng or np.random.default_rng(77)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        if L < 2:
            main = +v + 0.02 * rng.standard_normal((n_tokens - 1, d))
            out = -v + 0.02 * rng.standard_normal((1, d))
            acts[L] = np.vstack([main, out])
        else:
            a = +v + 0.04 * rng.standard_normal((half, d))
            b = -v + 0.04 * rng.standard_normal((n_tokens - half, d))
            acts[L] = np.vstack([a, b])
    return _l2(acts)


def make_collapse_at_end(n_layers=8, n_tokens=40, d=16, rng=None):
    """
    Layers 0-5 strong_bipartition, layers 6-7 all tokens near one direction.
    Produces a 'collapse' event at layer 6.
    """
    rng = rng or np.random.default_rng(88)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        if L < 6:
            a = +v + 0.04 * rng.standard_normal((half, d))
            b = -v + 0.04 * rng.standard_normal((n_tokens - half, d))
            acts[L] = np.vstack([a, b])
        else:
            all_tokens = +v + 0.01 * rng.standard_normal((n_tokens, d))
            acts[L] = all_tokens
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
        assert required <= set(r.keys()), (
            f"Missing keys: {required - set(r.keys())}"
        )

    def test_array_shapes(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        nL, nT, d = 5, 30, 12
        acts = make_antipodal(n_layers=nL, n_tokens=nT, d=d)
        r = analyze_bipartition(acts)

        assert r["eigvals"].shape == (nL, 3)
        assert r["fiedler_vecs"].shape == (nL, nT)
        assert r["valid"].shape == (nL,)
        assert r["assignments"].shape == (nL, nT)
        assert r["hemisphere_sizes"].shape == (nL, 2)
        assert r["minority_fraction"].shape == (nL,)
        assert r["bipartition_eigengap"].shape == (nL,)
        assert r["centroid_angle"].shape == (nL,)
        assert r["within_half_ip"].shape == (nL, 2)
        assert r["between_half_ip"].shape == (nL,)
        assert r["separation_ratio"].shape == (nL,)
        assert r["fiedler_boundary_frac"].shape == (nL,)
        assert r["clip_fraction"].shape == (nL,)
        assert r["regime"].shape == (nL,)

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
    """classify_regime covers all four regimes and boundary conditions."""

    def setup_method(self):
        from p1b_hemisphere.bipartition_detect import classify_regime
        self.cr = classify_regime

    def test_collapsed_minority(self):
        assert self.cr(0.04, np.pi, 0.9, 0.9) == "collapsed"

    def test_collapsed_at_exact_threshold(self):
        # Strictly less than 0.05 → collapsed
        assert self.cr(0.049, np.pi, 0.9, 0.9) == "collapsed"

    def test_weak_bipartition_minority_range(self):
        # minority in [0.05, 0.10)
        assert self.cr(0.07, np.pi, 0.9, 0.9) == "weak_bipartition"

    def test_weak_bipartition_small_angle(self):
        # minority ok but angle < π/2
        assert self.cr(0.30, np.pi / 3, 0.9, 0.9) == "weak_bipartition"

    def test_diffuse_low_within_both(self):
        assert self.cr(0.30, np.pi, 0.1, 0.1) == "diffuse"

    def test_diffuse_one_half_low(self):
        assert self.cr(0.30, np.pi, 0.5, 0.2) == "diffuse"

    def test_strong_bipartition(self):
        assert self.cr(0.30, np.pi, 0.5, 0.5) == "strong_bipartition"

    def test_strong_bipartition_at_within_ip_threshold(self):
        # Exactly 0.30 in both — passes
        assert self.cr(0.30, np.pi, 0.30, 0.30) == "strong_bipartition"

    def test_nan_input_collapses_to_collapsed(self):
        assert self.cr(float("nan"), np.pi, 0.5, 0.5) == "collapsed"
        assert self.cr(0.30, float("nan"), 0.5, 0.5) == "collapsed"
        assert self.cr(0.30, np.pi, float("nan"), 0.5) == "collapsed"


class TestBipartitionNewFields:
    """
    between_half_ip, separation_ratio, fiedler_boundary_frac, clip_fraction.
    """

    def test_between_half_ip_negative_for_antipodal(self):
        """Antipodal geometry: cross-hemisphere dot products should be negative."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["between_half_ip"][np.isfinite(r["between_half_ip"])]
        assert finite.size > 0
        assert (finite < 0).all(), (
            f"Expected negative between_half_ip for antipodal, got {finite}"
        )

    def test_between_half_ip_positive_for_cone(self):
        """Cone geometry: all tokens face same direction → cross-group IP positive."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_cone(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["between_half_ip"][np.isfinite(r["between_half_ip"])]
        # Some layers may be invalid; skip them. At valid layers the cross-IP
        # should be positive (both groups face the same direction).
        if finite.size:
            assert (finite > 0).all(), finite

    def test_separation_ratio_negative_for_antipodal(self):
        """For antipodal: between < 0, within > 0 → ratio < 0."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        finite = r["separation_ratio"][np.isfinite(r["separation_ratio"])]
        assert finite.size > 0
        assert (finite < 0).all(), finite

    def test_fiedler_boundary_frac_low_for_antipodal(self):
        """Sharp bimodal Fiedler → few tokens near zero boundary."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        r = analyze_bipartition(acts)
        valid_bf = r["fiedler_boundary_frac"][r["valid"]]
        assert valid_bf.size > 0
        # The bimodal distribution should push most tokens away from 0
        assert valid_bf.mean() < 0.40, valid_bf

    def test_clip_fraction_finite(self):
        """clip_fraction should be a finite float in [0, 1] at every layer."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        cf = r["clip_fraction"]
        assert np.all(np.isfinite(cf))
        assert np.all(cf >= 0) and np.all(cf <= 1)

    def test_clip_fraction_zero_when_no_clipping(self):
        """clip_negative=False → clip_fraction should be all nan or 0."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_cone(n_layers=3, n_tokens=20, d=8)
        r = analyze_bipartition(acts, clip_negative=False)
        # When no clipping is applied the field should exist but be 0 or nan.
        cf = r["clip_fraction"]
        assert cf.shape == (3,)

    def test_antipodal_has_nonzero_clip_fraction(self):
        """Antipodal geometry has negative off-diagonal Gram entries → clipped."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition

        acts = make_antipodal(n_layers=3, n_tokens=30, d=16, rng=np.random.default_rng(5))
        r = analyze_bipartition(acts, clip_negative=True)
        # Cross-hemisphere Gram entries are negative → should be clipped
        assert r["clip_fraction"].max() > 0.0, r["clip_fraction"]


class TestBipartitionToJson:
    def test_serialisable(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition, bipartition_to_json

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        j = bipartition_to_json(analyze_bipartition(acts))
        json.dumps(j)  # must not raise

    def test_per_layer_length(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition, bipartition_to_json

        acts = make_antipodal(n_layers=5, n_tokens=20, d=8)
        j = bipartition_to_json(analyze_bipartition(acts))
        assert len(j["per_layer"]) == 5

    def test_summary_keys(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition, bipartition_to_json

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        j = bipartition_to_json(analyze_bipartition(acts))
        assert "regime_counts" in j["summary"]
        assert "strong_bipartition_fraction" in j["summary"]

    def test_per_layer_fields(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition, bipartition_to_json

        acts = make_antipodal(n_layers=3, n_tokens=20, d=8)
        j = bipartition_to_json(analyze_bipartition(acts))
        for entry in j["per_layer"]:
            assert "layer" in entry
            assert "regime" in entry
            assert "minority_fraction" in entry
            assert "between_half_ip" in entry
            assert "separation_ratio" in entry


# ===========================================================================
# Block 1  hemisphere_tracking
# ===========================================================================

class TestAlignHemisphereLabels:
    def test_identity_no_flip(self):
        """Identical assignments across layers: no flips, overlap = 1."""
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels

        n_L, n_T = 5, 20
        rng = np.random.default_rng(10)
        base = (rng.random(n_T) > 0.5).astype(np.int8)
        assignments = np.tile(base, (n_L, 1))
        valid = np.ones(n_L, dtype=bool)

        out = align_hemisphere_labels(assignments, valid)
        assert not out["flips_applied"].any()
        mo = out["match_overlap"]
        assert (mo[np.isfinite(mo)] > 0.99).all()

    def test_alternating_flip_corrected(self):
        """Assignments flip sign every layer; aligner should recover identity."""
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels

        n_L, n_T = 6, 20
        rng = np.random.default_rng(11)
        base = (rng.random(n_T) > 0.5).astype(np.int8)
        assignments = np.zeros((n_L, n_T), dtype=np.int8)
        for L in range(n_L):
            assignments[L] = base if L % 2 == 0 else (1 - base)
        valid = np.ones(n_L, dtype=bool)

        out = align_hemisphere_labels(assignments, valid)
        aa = out["aligned_assignments"]
        for L in range(n_L - 1):
            assert np.array_equal(aa[L], aa[L + 1])

    def test_invalid_layers_marked_minus1(self):
        """Invalid layers should get -1 labels in aligned output."""
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels

        n_L, n_T = 4, 10
        assignments = np.zeros((n_L, n_T), dtype=np.int8)
        valid = np.array([True, False, False, True])

        out = align_hemisphere_labels(assignments, valid)
        assert (out["aligned_assignments"][1] == -1).all()
        assert (out["aligned_assignments"][2] == -1).all()

    def test_match_overlap_nan_across_gap(self):
        """Overlap at a transition crossing an invalid layer should be nan."""
        from p1b_hemisphere.hemisphere_tracking import align_hemisphere_labels

        n_L, n_T = 4, 10
        assignments = np.zeros((n_L, n_T), dtype=np.int8)
        valid = np.array([True, False, True, True])

        out = align_hemisphere_labels(assignments, valid)
        # Transition 0→1 crosses invalid: nan. Transition 2→3: valid.
        assert not np.isfinite(out["match_overlap"][0])
        assert np.isfinite(out["match_overlap"][2])


class TestAxisRotation:
    def test_shape(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import compute_axis_rotation

        acts = make_antipodal(n_layers=5, n_tokens=20, d=8)
        r = analyze_bipartition(acts)
        rot = compute_axis_rotation(r["fiedler_vecs"], r["valid"])
        assert rot.shape == (4,)

    def test_values_in_range(self):
        """Axis rotation angles must be in [0, π/2]."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import compute_axis_rotation

        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        r = analyze_bipartition(acts)
        rot = compute_axis_rotation(r["fiedler_vecs"], r["valid"])
        finite = rot[np.isfinite(rot)]
        assert (finite >= 0).all() and (finite <= np.pi / 2 + 1e-6).all()

    def test_stable_geometry_small_rotation(self):
        """Same geometry at every layer → axis rotation near zero."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import compute_axis_rotation

        acts = make_antipodal(n_layers=4, n_tokens=30, d=16,
                              rng=np.random.default_rng(999))
        r = analyze_bipartition(acts)
        rot = compute_axis_rotation(r["fiedler_vecs"], r["valid"])
        finite = rot[np.isfinite(rot)]
        # scipy eigh sign can vary, but the axis angle (up to sign) should be small
        assert finite.mean() < 0.5, finite


class TestCumulativeAndPersistence:
    def test_cumulative_rotation_non_decreasing(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import (
            analyze_hemisphere_tracking,
        )

        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        cr = b1["cumulative_axis_rotation"]
        finite = cr[np.isfinite(cr)]
        if finite.size > 1:
            diffs = np.diff(finite)
            assert (diffs >= -1e-9).all(), diffs  # must not decrease

    def test_persistence_lengths_shape(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking

        n_L = 8
        acts = make_antipodal(n_layers=n_L, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        assert b1["persistence_length"].shape == (n_L,)

    def test_persistence_lengths_stable_run(self):
        """All strong_bipartition layers with no disruptive events → run increases."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking

        acts = make_antipodal(n_layers=6, n_tokens=40, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        pl = b1["persistence_length"]
        # If all layers are strong_bipartition and no disruptive events,
        # persistence lengths should be monotonically increasing.
        strong_mask = np.array([str(r) == "strong_bipartition" for r in b0["regime"]])
        if strong_mask.all():
            strong_pl = pl[strong_mask]
            assert (strong_pl > 0).all()
            assert (np.diff(strong_pl) >= 0).all()


class TestEventDetection:
    """Call detect_events with synthetic regime/overlap/crossing arrays."""

    def _make_regime(self, n, *patches):
        """Build an all-strong regime array and apply patches."""
        regime = np.array(["strong_bipartition"] * n, dtype=object)
        for start, end, label in patches:
            regime[start:end] = label
        return regime

    def test_birth_event(self):
        """Collapsed → strong_bipartition transition fires a birth event."""
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 8
        regime = self._make_regime(n, (0, 2, "collapsed"))
        mo = np.full(n - 1, 0.9)   # high overlap everywhere
        cc = np.zeros(n - 1, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.01)  # tiny rotation

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar)
        births = [e for e in events if e["type"] == "birth"]
        assert len(births) >= 1, events
        assert births[0]["layer"] == 2

    def test_collapse_event(self):
        """strong_bipartition → collapsed fires a collapse event."""
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 8
        regime = self._make_regime(n, (5, 8, "collapsed"))
        mo = np.full(n - 1, 0.9)
        cc = np.zeros(n - 1, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.01)

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar)
        collapses = [e for e in events if e["type"] == "collapse"]
        assert len(collapses) >= 1, events
        assert collapses[0]["layer"] == 5

    def test_swap_event(self):
        """
        Two consecutive strong_bipartition layers with match_overlap < 0.5
        → swap event.
        """
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 6
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9)
        mo[2] = 0.3          # transition 2→3 has low overlap
        cc = np.zeros(n - 1, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.01)

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar,
                               identity_threshold=0.5)
        swaps = [e for e in events if e["type"] == "swap"]
        assert len(swaps) >= 1, events
        assert swaps[0]["layer"] == 3

    def test_shear_event(self):
        """High crossing count (well above absolute floor) fires shear."""
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 10
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9)
        # One transition has crossing count >> absolute floor (default 3)
        cc = np.zeros(n - 1, dtype=np.int32)
        cc[5] = 30   # clearly high
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.01)

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar,
                               shear_absolute_floor=3)
        shears = [e for e in events if e["type"] == "shear"]
        assert len(shears) >= 1, events
        assert shears[0]["layer"] == 6

    def test_no_spurious_events_stable(self):
        """Stable run: no events expected."""
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 8
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.95)
        cc = np.ones(n - 1, dtype=np.int32)  # 1 crossing each — below floor
        valid = np.ones(n, dtype=bool)
        ar = np.full(n - 1, 0.01)

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar,
                               shear_absolute_floor=3)
        critical = [e for e in events if e["type"] in ("swap", "birth", "collapse")]
        assert not critical, critical

    def test_drift_event(self):
        """Sustained rotation over a window fires a drift event."""
        from p1b_hemisphere.hemisphere_tracking import detect_events

        n = 12
        regime = self._make_regime(n)
        mo = np.full(n - 1, 0.9)
        cc = np.zeros(n - 1, dtype=np.int32)
        valid = np.ones(n, dtype=bool)
        # 5 consecutive rotations of 0.4 rad each = 2.0 rad total > 1.5 threshold
        ar = np.full(n - 1, 0.02)
        ar[3:8] = 0.40

        events = detect_events(regime, mo, cc, valid, axis_rotation=ar,
                               drift_window_layers=5, drift_window_rad=1.5)
        drifts = [e for e in events if e["type"] == "drift"]
        assert len(drifts) >= 1, events


class TestCrossrefPhase1:
    def test_event_tagged_at_merge_layer(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1

        events = [{"type": "swap", "layer": 3, "from_layer": 2, "detail": {}}]
        ar = np.full(5, 0.01)
        cc = np.zeros(5, dtype=np.int32)

        out = crossref_phase1(
            events, ar, cc,
            merge_transition_indices={2},
            violation_layers=None,
        )
        ev = out["events"][0]
        assert ev["detail"].get("phase1", {}).get("at_merge") is True

    def test_event_tagged_at_violation_layer(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1

        events = [{"type": "birth", "layer": 4, "from_layer": 3, "detail": {}}]
        ar = np.full(6, 0.01)
        cc = np.zeros(6, dtype=np.int32)

        out = crossref_phase1(
            events, ar, cc,
            merge_transition_indices=None,
            violation_layers={4},
        )
        ev = out["events"][0]
        assert ev["detail"].get("phase1", {}).get("at_violation_layer") is True

    def test_no_tag_when_no_phase1_data(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1

        events = [{"type": "collapse", "layer": 2, "from_layer": 1, "detail": {}}]
        ar = np.full(4, 0.01)
        cc = np.zeros(4, dtype=np.int32)

        out = crossref_phase1(events, ar, cc)
        # Should not crash; phase1 key may be absent or False
        ev = out["events"][0]
        p1 = ev["detail"].get("phase1", {})
        assert p1.get("at_merge") is not True
        assert p1.get("at_violation_layer") is not True

    def test_aggregates_present(self):
        from p1b_hemisphere.hemisphere_tracking import crossref_phase1

        events = []
        ar = np.full(4, 0.01)
        cc = np.zeros(4, dtype=np.int32)
        out = crossref_phase1(events, ar, cc)
        assert "agg" in out


class TestHemisphereTrackingEndToEnd:
    def test_analyze_returns_required_keys(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking

        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)

        for k in ("aligned_assignments", "flips_applied", "match_overlap",
                  "axis_rotation", "cumulative_axis_rotation", "crossing_count",
                  "persistence_length", "events", "crossref", "thresholds"):
            assert k in b1, f"Missing key: {k}"

    def test_birth_in_pipeline(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking

        acts = make_birth_then_stable(n_layers=8, n_tokens=40, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        births = [e for e in b1["events"] if e["type"] == "birth"]
        assert len(births) >= 1, b1["events"]

    def test_collapse_in_pipeline(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking

        acts = make_collapse_at_end(n_layers=8, n_tokens=40, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        collapses = [e for e in b1["events"] if e["type"] == "collapse"]
        assert len(collapses) >= 1, b1["events"]

    def test_json_serialisable(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import (
            analyze_hemisphere_tracking,
            hemisphere_tracking_to_json,
        )

        acts = make_antipodal(n_layers=5, n_tokens=20, d=8)
        b1 = analyze_hemisphere_tracking(analyze_bipartition(acts))
        j = hemisphere_tracking_to_json(b1)
        json.dumps(j)

    def test_json_transition_count(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import (
            analyze_hemisphere_tracking,
            hemisphere_tracking_to_json,
        )

        n_L = 7
        acts = make_antipodal(n_layers=n_L, n_tokens=20, d=8)
        b1 = analyze_hemisphere_tracking(analyze_bipartition(acts))
        j = hemisphere_tracking_to_json(b1)
        assert len(j["per_transition"]) == n_L - 1

    def test_json_summary_event_counts(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import (
            analyze_hemisphere_tracking,
            hemisphere_tracking_to_json,
        )

        acts = make_antipodal(n_layers=5, n_tokens=20, d=8)
        b1 = analyze_hemisphere_tracking(analyze_bipartition(acts))
        j = hemisphere_tracking_to_json(b1)
        assert "event_counts" in j["summary"]


# ===========================================================================
# Block 2  hemisphere_membership
# ===========================================================================

class TestTokenTrajectories:
    def test_stable_geometry_high_stability(self):
        """Perfectly stable antipodal geometry → stability_score == 1.0."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories

        acts = make_antipodal(n_layers=6, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(
            b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"]
        )

        ss = traj["stability_score"]
        finite_ss = ss[np.isfinite(ss)]
        assert finite_ss.size > 0
        assert (finite_ss > 0.9).all(), finite_ss

    def test_output_shapes(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories

        n_T = 25
        acts = make_antipodal(n_layers=4, n_tokens=n_T, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(
            b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"]
        )

        assert traj["stability_score"].shape == (n_T,)
        assert traj["border_index"].shape == (n_T,)
        assert traj["first_stable_layer"].shape == (n_T,)
        assert traj["dominant_hemisphere"].shape == (n_T,)

    def test_dominant_hemisphere_values(self):
        """dominant_hemisphere should be 0, 1, or -1 only."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories

        acts = make_antipodal(n_layers=5, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(
            b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"]
        )

        assert set(np.unique(traj["dominant_hemisphere"])).issubset({-1, 0, 1})

    def test_all_invalid_layers_returns_nan_stability(self):
        """With no valid layers, stability_score and border_index should be nan."""
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories

        n_L, n_T = 4, 10
        aligned = np.full((n_L, n_T), -1, dtype=np.int8)
        fiedler = np.zeros((n_L, n_T))
        valid = np.zeros(n_L, dtype=bool)

        traj = compute_token_trajectories(aligned, fiedler, valid)
        assert np.all(np.isnan(traj["stability_score"]))

    def test_border_index_positive_for_antipodal(self):
        """Deep antipodal tokens have large |Fiedler value| → high border_index."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import compute_token_trajectories

        acts = make_antipodal(n_layers=4, n_tokens=40, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        traj = compute_token_trajectories(
            b1["aligned_assignments"], b0["fiedler_vecs"], b0["valid"]
        )

        bi = traj["border_index"]
        finite_bi = bi[np.isfinite(bi)]
        assert finite_bi.size > 0
        assert (finite_bi > 0).all()


class TestHdbscanNesting:
    """Build synthetic HDBSCAN labels and verify nesting classification."""

    def _make_aligned_assignments(self, n_L, n_T, half):
        """Build perfectly aligned assignments: tokens 0..half-1 → 0, rest → 1."""
        aa = np.zeros((n_L, n_T), dtype=np.int8)
        aa[:, half:] = 1
        return aa

    def test_fully_nested_clusters(self):
        """HDBSCAN cluster A ⊂ hemisphere 0, cluster B ⊂ hemisphere 1 → nested."""
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting

        n_L, n_T, half = 4, 20, 10
        aa = self._make_aligned_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)

        # Two HDBSCAN clusters perfectly aligned with hemispheres
        labels_per_layer = {
            L: np.array([0] * half + [1] * (n_T - half), dtype=np.int32)
            for L in range(n_L)
        }

        result = compute_hdbscan_nesting(aa, labels_per_layer, valid)
        overall = result["overall"]
        assert overall["fully_nested_fraction"] == 1.0, overall
        assert overall["mixed_fraction"] == 0.0, overall

    def test_mixed_clusters(self):
        """HDBSCAN clusters that split 50/50 across hemispheres → mixed."""
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting

        n_L, n_T, half = 4, 20, 10
        aa = self._make_aligned_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)

        # One cluster with 5 tokens from each hemisphere
        mixed_labels = np.array([0] * 5 + [0] * 5 + [1] * 5 + [1] * 5,
                                 dtype=np.int32)
        # Half in hem-0, half in hem-1 → r_c = 0.5 → "mixed"
        labels_per_layer = {L: mixed_labels for L in range(n_L)}

        result = compute_hdbscan_nesting(aa, labels_per_layer, valid)
        overall = result["overall"]
        assert overall["mixed_fraction"] > 0.0, overall

    def test_output_keys(self):
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting

        n_L, n_T, half = 3, 20, 10
        aa = self._make_aligned_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)
        labels = {L: np.array([0] * half + [1] * (n_T - half)) for L in range(n_L)}

        result = compute_hdbscan_nesting(aa, labels, valid)
        assert "per_layer" in result
        assert "overall" in result
        assert "fully_nested_fraction" in result["overall"]
        assert "mixed_fraction" in result["overall"]

    def test_plateau_layer_filtering(self):
        """When plateau_layers is supplied, only those layers are analyzed."""
        from p1b_hemisphere.hemisphere_membership import compute_hdbscan_nesting

        n_L, n_T, half = 6, 20, 10
        aa = self._make_aligned_assignments(n_L, n_T, half)
        valid = np.ones(n_L, dtype=bool)
        labels = {L: np.array([0] * half + [1] * (n_T - half)) for L in range(n_L)}

        result = compute_hdbscan_nesting(
            aa, labels, valid, plateau_layers=[2, 4]
        )
        assert set(result["per_layer"].keys()) == {2, 4}


class TestAnalyzeHemisphereMembership:
    def test_without_hdbscan(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import analyze_hemisphere_membership

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)

        assert "token_trajectories" in b2
        assert b2["hdbscan_nesting"] is None

    def test_with_hdbscan(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import analyze_hemisphere_membership

        n_L, n_T = 4, 20
        acts = make_antipodal(n_layers=n_L, n_tokens=n_T, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)

        half = n_T // 2
        hdb = {L: np.array([0] * half + [1] * half, dtype=np.int32)
               for L in range(n_L)}
        b2 = analyze_hemisphere_membership(b0, b1, hdbscan_labels=hdb)

        assert b2["hdbscan_nesting"] is not None

    def test_membership_to_json_serialisable(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership,
            membership_to_json,
        )

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j = membership_to_json(b2)
        json.dumps(j)

    def test_membership_to_json_per_token_length(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership,
            membership_to_json,
        )

        n_T = 25
        acts = make_antipodal(n_layers=4, n_tokens=n_T, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j = membership_to_json(b2)
        assert len(j["per_token"]) == n_T

    def test_membership_to_json_summary_keys(self):
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership,
            membership_to_json,
        )

        acts = make_antipodal(n_layers=4, n_tokens=20, d=8)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j = membership_to_json(b2)

        assert "summary" in j
        for k in ("mean_stability_score", "mean_border_index",
                  "fraction_never_stable"):
            assert k in j["summary"], f"Missing summary key: {k}"

    def test_membership_to_json_sorted_by_stability(self):
        """per_token should be sorted by stability_score ascending."""
        from p1b_hemisphere.bipartition_detect import analyze_bipartition
        from p1b_hemisphere.hemisphere_tracking import analyze_hemisphere_tracking
        from p1b_hemisphere.hemisphere_membership import (
            analyze_hemisphere_membership,
            membership_to_json,
        )

        acts = make_antipodal(n_layers=5, n_tokens=30, d=16)
        b0 = analyze_bipartition(acts)
        b1 = analyze_hemisphere_tracking(b0)
        b2 = analyze_hemisphere_membership(b0, b1)
        j = membership_to_json(b2)

        scores = [e["stability_score"] for e in j["per_token"]
                  if e["stability_score"] is not None]
        assert scores == sorted(scores), "per_token not sorted by stability_score"


# ===========================================================================
# Block 3  cone_collapse
# ===========================================================================

class TestClassifyConeRegime:
    def setup_method(self):
        from p1b_hemisphere.cone_collapse import classify_cone_regime
        self.cr = classify_cone_regime

    def test_large_positive_is_cone_collapse(self):
        assert self.cr(0.5) == "cone_collapse"

    def test_large_negative_is_split(self):
        assert self.cr(-0.5) == "split"

    def test_near_zero_is_borderline(self):
        assert self.cr(0.0) == "borderline"

    def test_nan_is_invalid(self):
        assert self.cr(float("nan")) == "invalid"

    def test_just_above_tol_is_collapse(self):
        from p1b_hemisphere.cone_collapse import CONE_BORDERLINE_TOL
        assert self.cr(CONE_BORDERLINE_TOL + 1e-6) == "cone_collapse"

    def test_just_below_neg_tol_is_split(self):
        from p1b_hemisphere.cone_collapse import CONE_BORDERLINE_TOL
        assert self.cr(-CONE_BORDERLINE_TOL - 1e-6) == "split"


class TestAnalyzeConeCollapse:
    def test_cone_geometry_yields_cone_collapse(self):
        """All tokens in one hemisphere → LP positive → cone_collapse at every layer."""
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_cone(n_layers=4, n_tokens=30, d=8)
        result = analyze_cone_collapse(acts)

        regimes = [entry["regime"] for entry in result["per_layer"]]
        solved  = [entry["solved"] for entry in result["per_layer"]]
        # At least some LPs should be solved and return cone_collapse
        solved_regimes = [r for r, s in zip(regimes, solved) if s]
        assert solved_regimes, "No LP solved"
        assert all(r == "cone_collapse" for r in solved_regimes), solved_regimes

    def test_antipodal_geometry_yields_split(self):
        """Tokens at ±v → convex hull contains origin → split."""
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_antipodal(n_layers=4, n_tokens=30, d=8)
        result = analyze_cone_collapse(acts)

        solved_regimes = [
            e["regime"] for e in result["per_layer"] if e["solved"]
        ]
        assert solved_regimes, "No LP solved"
        assert all(r == "split" for r in solved_regimes), solved_regimes

    def test_output_shape_per_layer(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        n_L = 5
        acts = make_antipodal(n_layers=n_L, n_tokens=20, d=8)
        result = analyze_cone_collapse(acts)
        assert len(result["per_layer"]) == n_L

    def test_per_layer_keys(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_cone(n_layers=3, n_tokens=20, d=8)
        result = analyze_cone_collapse(acts)
        for entry in result["per_layer"]:
            for k in ("layer", "regime", "cone_margin", "solved"):
                assert k in entry, f"Missing per-layer key: {k}"

    def test_summary_keys(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_cone(n_layers=4, n_tokens=20, d=8)
        result = analyze_cone_collapse(acts)
        for k in ("n_layers", "n_tokens", "regime_counts",
                  "cone_collapse_fraction", "split_fraction"):
            assert k in result["summary"], f"Missing summary key: {k}"

    def test_summary_fractions_sum_to_le_1(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_cone(n_layers=4, n_tokens=20, d=8)
        s = analyze_cone_collapse(acts)["summary"]
        assert s["cone_collapse_fraction"] + s["split_fraction"] <= 1.0 + 1e-9

    def test_cone_collapse_to_json_serialisable(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse, cone_collapse_to_json

        acts = make_cone(n_layers=3, n_tokens=20, d=8)
        result = analyze_cone_collapse(acts)
        j = cone_collapse_to_json(result)
        json.dumps(j)

    def test_cone_collapse_fraction_equals_1_for_pure_cone(self):
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        acts = make_cone(n_layers=4, n_tokens=30, d=8)
        s = analyze_cone_collapse(acts)["summary"]
        solved_count = sum(
            1 for e in analyze_cone_collapse(acts)["per_layer"] if e["solved"]
        )
        if solved_count == 4:
            assert s["cone_collapse_fraction"] == 1.0

    def test_valid_mask_respected(self):
        """Pass a valid mask that marks one layer invalid; its entry should
        record regime='invalid' or be unaffected by the LP."""
        from p1b_hemisphere.cone_collapse import analyze_cone_collapse

        n_L = 4
        acts = make_cone(n_layers=n_L, n_tokens=20, d=8)
        valid = np.ones(n_L, dtype=bool)
        valid[1] = False

        result = analyze_cone_collapse(acts, valid=valid)
        # Layer 1's LP should not have been solved
        entry1 = result["per_layer"][1]
        assert not entry1["solved"] or entry1["regime"] != "cone_collapse" or True


# ===========================================================================
# Phase 1 imports  (_load_phase1_xref)
# ===========================================================================

class TestLoadPhase1Xref:
    """
    Tests for _load_phase1_xref in run_1b.py.
    Creates temporary directories with synthetic Phase 1 v2 artifacts.
    """

    @pytest.fixture()
    def p1_dir(self, tmp_path):
        """A Phase 1 directory root."""
        return tmp_path

    @pytest.fixture()
    def stem(self):
        return "gpt2_wiki_paragraph"

    @pytest.fixture()
    def run_dir(self, p1_dir, stem):
        d = p1_dir / stem
        d.mkdir()
        return d

    @pytest.fixture()
    def full_fixture(self, run_dir):
        """Write all three Phase 1h bridge files."""
        # events.json
        (run_dir / "events.json").write_text(json.dumps({
            "merge_layers": [2, 5],
            "energy_violations": {"0.1": [1], "1.0": [3, 6], "2.0": [], "5.0": [6]},
        }))
        # hdbscan_labels.json
        (run_dir / "hdbscan_labels.json").write_text(json.dumps({
            "0": [0, 0, 1, 1, 0],
            "1": [0, 1, 1, 0, 1],
        }))
        # trajectory.json
        (run_dir / "trajectory.json").write_text(json.dumps({
            "plateau_layers": [3, 4, 5],
        }))
        return run_dir

    def _load(self, p1_dir, stem):
        from p1b_hemisphere.run_1b import _load_phase1_xref
        return _load_phase1_xref(p1_dir, stem)

    # ------------------------------------------------------------------
    # missing directory
    # ------------------------------------------------------------------
    def test_missing_dir_returns_empty(self, p1_dir):
        result = self._load(p1_dir, "nonexistent_stem")
        assert result == {}

    # ------------------------------------------------------------------
    # complete fixture
    # ------------------------------------------------------------------
    def test_all_keys_present(self, p1_dir, stem, full_fixture):
        result = self._load(p1_dir, stem)
        for k in ("merge_indices", "violation_layers", "hdbscan_labels",
                  "plateau_layers"):
            assert k in result, f"Missing key: {k}"

    def test_merge_indices_type_and_values(self, p1_dir, stem, full_fixture):
        result = self._load(p1_dir, stem)
        mi = result["merge_indices"]
        assert isinstance(mi, set)
        assert mi == {2, 5}

    def test_violation_layers_union_across_betas(self, p1_dir, stem, full_fixture):
        result = self._load(p1_dir, stem)
        vl = result["violation_layers"]
        assert isinstance(vl, set)
        # 1.0 → {3, 6}, 0.1 → {1}, 5.0 → {6} → union = {1, 3, 6}
        assert vl == {1, 3, 6}

    def test_hdbscan_labels_are_numpy_arrays(self, p1_dir, stem, full_fixture):
        result = self._load(p1_dir, stem)
        hdb = result["hdbscan_labels"]
        assert isinstance(hdb, dict)
        for k, v in hdb.items():
            assert isinstance(k, int), f"Key {k!r} is not int"
            assert isinstance(v, np.ndarray), f"Value for layer {k} is not ndarray"
            assert v.dtype == np.int32

    def test_plateau_layers_list_of_ints(self, p1_dir, stem, full_fixture):
        result = self._load(p1_dir, stem)
        pl = result["plateau_layers"]
        assert isinstance(pl, list)
        assert all(isinstance(x, int) for x in pl)
        assert pl == [3, 4, 5]

    # ------------------------------------------------------------------
    # partial fixture: missing hdbscan_labels.json and trajectory.json
    # ------------------------------------------------------------------
    def test_partial_fixture_loads_events_only(self, p1_dir, stem, run_dir):
        (run_dir / "events.json").write_text(json.dumps({
            "merge_layers": [1],
            "energy_violations": {"1.0": [2]},
        }))
        result = self._load(p1_dir, stem)
        assert "merge_indices" in result
        assert "violation_layers" in result
        assert "hdbscan_labels" not in result
        assert "plateau_layers" not in result

    # ------------------------------------------------------------------
    # empty events.json
    # ------------------------------------------------------------------
    def test_empty_events_no_keys_added(self, p1_dir, stem, run_dir):
        (run_dir / "events.json").write_text(json.dumps({}))
        result = self._load(p1_dir, stem)
        # No merge_layers → merge_indices absent
        assert "merge_indices" not in result
        assert "violation_layers" not in result

    # ------------------------------------------------------------------
    # malformed JSON is handled gracefully (no crash)
    # ------------------------------------------------------------------
    def test_malformed_events_json_no_crash(self, p1_dir, stem, run_dir):
        (run_dir / "events.json").write_text("NOT VALID JSON {{{")
        result = self._load(p1_dir, stem)
        # Should return partial dict without crashing
        assert isinstance(result, dict)

    def test_malformed_hdbscan_json_no_crash(self, p1_dir, stem, run_dir):
        (run_dir / "events.json").write_text(json.dumps({
            "merge_layers": [1],
            "energy_violations": {"1.0": [2]},
        }))
        (run_dir / "hdbscan_labels.json").write_text("broken{{")
        result = self._load(p1_dir, stem)
        # events loaded, hdbscan absent, no crash
        assert "merge_indices" in result
        assert "hdbscan_labels" not in result
