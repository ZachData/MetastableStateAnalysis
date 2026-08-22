"""
tests/test_phase4_plateaus.py

Tests for p4_mstate_features modules that detect metastable feature structure:
  - activation_trajectories: detect_feature_plateaus, feature_cluster_mi,
                              plateau_alignment
  - chorus:                   analyze_chorus_at_layer

All tests are self-contained synthetic data; no model loading, no I/O.

Standard fixture dimensions (shared across all tests unless overridden):
  n_layers=8, n_tokens=30, d=16 (unused here), n_features=20, n_clusters=2
"""

import sys
import numpy as np
import pytest

# conftest.py ensures stubs are in place before this module is imported.
from p4_mstate_features.activation_trajectories import (
    ActivationTrajectory,
    detect_feature_plateaus,
    feature_cluster_mi,
    plateau_alignment,
    _mutual_information,
)
from p4_mstate_features.chorus import analyze_chorus_at_layer


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

N_LAYERS   = 8
N_TOKENS   = 30
N_FEATURES = 20
LAYER_IDX  = list(range(N_LAYERS))    # identity: model layer i == cc index i


def _make_traj(
    z: np.ndarray,
    active: np.ndarray | None = None,
    layer_indices: list | None = None,
    key: str = "test",
) -> ActivationTrajectory:
    """Convenience wrapper; active defaults to (z != 0)."""
    if active is None:
        active = (z != 0.0)
    if layer_indices is None:
        layer_indices = list(range(z.shape[2]))
    return ActivationTrajectory(key, z, active, layer_indices)


def _binary_labels(n: int = N_TOKENS) -> np.ndarray:
    """[0]*half + [1]*half."""
    half = n // 2
    return np.array([0] * half + [1] * (n - half))


def _feature_plateau_stub(plateau_ranges: list[tuple[int, int]]) -> dict:
    """
    Build a minimal feature_plateaus dict (output of detect_feature_plateaus)
    with a single feature whose plateaus are specified by (start, end) pairs.
    Used to test plateau_alignment without going through detect_feature_plateaus.
    """
    plateaus = [
        {"start": s, "end": e, "length": e - s + 1}
        for s, e in plateau_ranges
    ]
    return {
        "per_feature": [
            {
                "feature_idx": 0,
                "plateaus": plateaus,
                "n_plateaus": len(plateaus),
                "max_plateau_length": max(p["length"] for p in plateaus),
            }
        ],
        "summary": {
            "n_features_with_plateaus": 1,
            "n_features_total": 1,
            "fraction_with_plateaus": 1.0,
        },
    }


# ===========================================================================
# 1.  detect_feature_plateaus
# ===========================================================================


class TestDetectFeaturePlateaus:
    """Plateau detection via rolling variance across layers."""

    def test_constant_activation_flags_all_layers(self):
        """
        A feature with the same value at every layer has zero rolling variance
        at every window position → single plateau spanning all layers.
        """
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)

        # Every feature should have at least one plateau
        assert result["summary"]["n_features_with_plateaus"] == N_FEATURES
        assert result["summary"]["fraction_with_plateaus"] == pytest.approx(1.0)

        # Each plateau should span the full depth
        for feat_info in result["per_feature"]:
            longest = feat_info["max_plateau_length"]
            assert longest == N_LAYERS, (
                f"feature {feat_info['feature_idx']}: expected plateau length "
                f"{N_LAYERS}, got {longest}"
            )

    def test_spike_excluded_from_plateau(self):
        """
        A feature that is constant except for a spike at one layer should
        form plateaus on either side of the spike but not include that layer
        in any plateau window.
        """
        SPIKE_LAYER = 4
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        z[:, 0, SPIKE_LAYER] = 10.0          # spike on feature 0
        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)

        feat0 = next(
            f for f in result["per_feature"] if f["feature_idx"] == 0
        )
        for plateau in feat0["plateaus"]:
            assert not (plateau["start"] <= SPIKE_LAYER <= plateau["end"]), (
                f"Spike layer {SPIKE_LAYER} included in plateau {plateau}"
            )

    def test_spike_produces_pre_and_post_plateaus(self):
        """
        With a spike at layer 4 and constant values elsewhere, the pre-spike
        layers (0-3) and post-spike layers (5-7) each form a plateau.
        """
        SPIKE_LAYER = 4
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        z[:, 0, SPIKE_LAYER] = 10.0
        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)
        feat0 = next(
            f for f in result["per_feature"] if f["feature_idx"] == 0
        )
        starts = {p["start"] for p in feat0["plateaus"]}
        ends   = {p["end"]   for p in feat0["plateaus"]}

        assert 0 in starts, "Pre-spike plateau should start at layer 0"
        assert (N_LAYERS - 1) in ends, (
            f"Post-spike plateau should end at layer {N_LAYERS-1}"
        )
        assert feat0["n_plateaus"] >= 2, (
            "Expected at least one plateau before and one after the spike"
        )

    def test_inactive_feature_not_reported(self):
        """
        A feature whose active_per_layer is all-False should be skipped
        regardless of what z values are present.
        """
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.ones_like(z, dtype=bool)
        active[:, 0, :] = False              # feature 0 is never active

        traj = _make_traj(z, active)
        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)

        reported_features = {f["feature_idx"] for f in result["per_feature"]}
        assert 0 not in reported_features, (
            "Inactive feature 0 should not appear in per_feature results"
        )

    def test_high_variance_signal_no_plateau(self):
        """
        A feature whose per-layer mean alternates +1/-1 should not pass
        a tight variance threshold.
        """
        z = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        # Alternating signal: mean per layer = [1,-1,1,-1,...] for feature 0
        for l in range(N_LAYERS):
            z[:, 0, l] = 1.0 if l % 2 == 0 else -1.0
        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)
        reported_features = {f["feature_idx"] for f in result["per_feature"]}
        assert 0 not in reported_features, (
            "Alternating feature should not form any plateau with tight threshold"
        )

    def test_summary_counts_are_consistent(self):
        """n_features_with_plateaus + fraction_with_plateaus must be consistent."""
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        result = detect_feature_plateaus(traj, var_threshold=0.01, min_plateau_len=3)
        s = result["summary"]

        expected_fraction = s["n_features_with_plateaus"] / s["n_features_total"]
        assert s["fraction_with_plateaus"] == pytest.approx(expected_fraction)
        assert s["n_features_total"] == N_FEATURES


class TestDetectFeaturePlateaus_Extra:

    def test_run_shorter_than_min_plateau_len_not_reported(self):
        """
        A constant run of exactly min_plateau_len - 1 layers must NOT
        be reported as a plateau.
        """
        min_len = 3
        # Constant for 2 layers (< min_len), then high variance
        z = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        for l in range(2):         # layers 0-1: constant 1.0
            z[:, 0, l] = 1.0
        for l in range(2, N_LAYERS):
            z[:, 0, l] = float(l) * 5.0    # diverges; no plateau

        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)
        result = detect_feature_plateaus(traj, var_threshold=0.01,
                                         min_plateau_len=min_len)

        reported = {f["feature_idx"] for f in result["per_feature"]}
        assert 0 not in reported, (
            f"Feature 0 had only {min_len-1} constant layers but was "
            "reported as having a plateau"
        )

    def test_features_without_plateaus_absent_from_per_feature(self):
        """
        A feature that never forms a plateau should not appear in
        per_feature (or appear with an empty plateaus list).
        We use high-variance alternating values to prevent any plateau.
        """
        z = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        for l in range(N_LAYERS):
            z[:, 0, l] = 100.0 if l % 2 == 0 else -100.0  # alternating

        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)
        result = detect_feature_plateaus(traj, var_threshold=0.001,
                                         min_plateau_len=3)

        for f in result["per_feature"]:
            if f["feature_idx"] == 0:
                assert f["n_plateaus"] == 0 or len(f["plateaus"]) == 0, (
                    "High-variance alternating feature should have no plateaus"
                )

    def test_var_threshold_controls_sensitivity(self):
        """
        A mildly drifting signal (small but non-zero variance per window)
        should form a plateau under a loose threshold but not a tight one.
        """
        z = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        for l in range(N_LAYERS):
            z[:, 0, l] = l * 0.1    # slow drift; inter-window variance ~0.1

        active = np.ones_like(z, dtype=bool)
        traj = _make_traj(z, active)

        # Tight threshold: no plateau
        tight = detect_feature_plateaus(traj, var_threshold=1e-4, min_plateau_len=3)
        tight_features = {f["feature_idx"] for f in tight["per_feature"]}

        # Loose threshold: plateau allowed
        loose = detect_feature_plateaus(traj, var_threshold=10.0, min_plateau_len=3)
        loose_features = {f["feature_idx"] for f in loose["per_feature"]}

        assert 0 not in tight_features, (
            "Drifting signal should not form plateau under tight threshold"
        )
        assert 0 in loose_features, (
            "Drifting signal should form plateau under loose threshold"
        )

# ===========================================================================
# 2.  feature_cluster_mi
# ===========================================================================


class TestFeatureClusterMI:
    """Mutual information between binary feature activations and cluster labels."""

    def _make_perfect_traj(self):
        """
        Feature 0 fires exactly when label == 1 (binary perfect predictor).
        All other features fire randomly at ~50 % regardless of label.
        """
        rng = np.random.default_rng(7)
        labels = _binary_labels()                    # (N_TOKENS,)
        z = rng.standard_normal(
            (N_TOKENS, N_FEATURES, N_LAYERS)
        ).astype(np.float32)

        active = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        # Feature 0: active iff label == 1
        active[:, 0, :] = labels[:, None]
        # Other features: random, uncorrelated with labels
        active[:, 1:, :] = rng.random((N_TOKENS, N_FEATURES - 1, N_LAYERS)) > 0.5

        traj = _make_traj(z, active)
        return traj, labels

    def test_perfect_predictor_mi_equals_log2(self):
        """
        A feature perfectly correlated with a balanced binary label
        achieves MI = log(2) ≈ 0.6931.
        """
        traj, labels = self._make_perfect_traj()
        hdbscan = {"layer_0": labels.tolist()}
        result = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        assert "layer_0" in result
        top = result["layer_0"]["top_features"]
        f0 = next(f for f in top if f["feature_idx"] == 0)

        assert f0["mi"] == pytest.approx(np.log(2), abs=1e-4), (
            f"Expected MI ≈ log(2)={np.log(2):.4f}, got {f0['mi']:.4f}"
        )

    def test_perfect_predictor_nmi_equals_1(self):
        """Same feature should have NMI = 1.0 (equal to min-entropy)."""
        traj, labels = self._make_perfect_traj()
        hdbscan = {"layer_0": labels.tolist()}
        result = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        f0 = next(
            f for f in result["layer_0"]["top_features"]
            if f["feature_idx"] == 0
        )
        assert f0["nmi"] == pytest.approx(1.0, abs=1e-4)

    def test_independent_feature_mi_near_zero(self):
        """
        A feature whose activations are constructed to be independent of the
        binary label should have MI ≈ 0.
        """
        # Construct exact independence: each cluster has exactly 50 % active
        labels = _binary_labels()
        active = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        # 8 tokens from cluster-0, 8 from cluster-1 → 16/30 ≈ 53 % rate,
        # but crucially the same proportion in each cluster
        active[0:8, 1, :]  = True
        active[15:23, 1, :] = True

        z = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        traj = _make_traj(z, active)
        hdbscan = {"layer_0": labels.tolist()}
        result = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        f1 = next(
            f for f in result["layer_0"]["top_features"]
            if f["feature_idx"] == 1
        )
        assert abs(f1["mi"]) < 0.01, (
            f"Independent feature MI should be ≈ 0, got {f1['mi']:.6f}"
        )

    def test_layer_not_in_layer_indices_is_skipped(self):
        """
        An hdbscan_labels entry whose model layer is not in layer_indices
        must not appear in the output.
        """
        labels = _binary_labels()
        active = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        z = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        traj = _make_traj(z, active)

        # Layer 99 does not exist in LAYER_IDX (0-7)
        hdbscan = {"layer_99": labels.tolist(), "layer_0": labels.tolist()}
        result = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        assert "layer_99" not in result
        assert "layer_0" in result

    def test_top_features_sorted_by_nmi_descending(self):
        """The returned top_features list must be sorted by NMI, highest first."""
        traj, labels = self._make_perfect_traj()
        hdbscan = {"layer_0": labels.tolist()}
        result = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        nmis = [f["nmi"] for f in result["layer_0"]["top_features"]]
        assert nmis == sorted(nmis, reverse=True), (
            "top_features should be sorted by NMI descending"
        )

    def test_direct_mi_computation_boundary_cases(self):
        """
        White-box: _mutual_information on two identical arrays should equal
        their entropy; on independent arrays it should be 0.
        """
        a = np.array([0, 0, 1, 1])
        # MI(a, a) == H(a) == log(2)
        assert _mutual_information(a, a) == pytest.approx(np.log(2), abs=1e-9)
        # MI(a, ~a) == log(2) as well (perfect predictor, just flipped)
        b = np.array([1, 1, 0, 0])
        assert _mutual_information(a, b) == pytest.approx(np.log(2), abs=1e-9)
        # MI with an independent uniform variable
        c = np.array([0, 1, 0, 1])
        assert _mutual_information(a, c) == pytest.approx(0.0, abs=1e-9)


class TestFeatureClusterMI_Extra:

    def test_noise_tokens_excluded_from_mi(self):
        """
        Tokens with label == -1 (HDBSCAN noise) must be excluded.
        If we make half the tokens noise but the remaining tokens are a
        perfect predictor, MI should still equal log(2), not be diluted.
        """
        rng = np.random.default_rng(42)
        n   = N_TOKENS

        # First half noise (-1), second half balanced 0/1
        labels = np.full(n, -1, dtype=int)
        labels[n // 2 :]   = np.array([0] * (n // 4) + [1] * (n - n // 2 - n // 4))

        z      = rng.standard_normal((n, N_FEATURES, N_LAYERS)).astype(np.float32)
        active = np.zeros((n, N_FEATURES, N_LAYERS), dtype=bool)
        # Feature 0: active iff label == 1
        active[:, 0, :] = (labels == 1)[:, None]

        traj    = _make_traj(z, active)
        hdbscan = {"layer_0": labels.tolist()}
        result  = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        f0 = next(f for f in result["layer_0"]["top_features"]
                  if f["feature_idx"] == 0)
        assert f0["nmi"] == pytest.approx(1.0, abs=0.05), (
            f"Expected NMI≈1.0 after excluding noise tokens, got {f0['nmi']:.4f}"
        )

    def test_result_has_key_per_hdbscan_layer(self):
        """
        If hdbscan_labels contains keys 'layer_0' and 'layer_4', the result
        dict should contain both keys.
        """
        rng = np.random.default_rng(0)
        z      = rng.standard_normal((N_TOKENS, N_FEATURES, N_LAYERS)).astype(np.float32)
        active = rng.random((N_TOKENS, N_FEATURES, N_LAYERS)) > 0.5
        traj   = _make_traj(z, active)

        labels = _binary_labels()
        hdbscan = {"layer_0": labels.tolist(), "layer_4": labels.tolist()}
        result  = feature_cluster_mi(traj, hdbscan, LAYER_IDX)

        assert "layer_0" in result, "Missing 'layer_0' in result"
        assert "layer_4" in result, "Missing 'layer_4' in result"


# ===========================================================================
# 3.  plateau_alignment
# ===========================================================================


class TestPlateauAlignment:
    """IoU-based overlap between feature and cluster plateau windows."""

    def test_identical_plateau_iou_is_1(self):
        """
        Feature plateau [0, 7] and cluster plateau [0, 7] share all layers
        → IoU = 1.0.
        """
        fp = _feature_plateau_stub([(0, N_LAYERS - 1)])
        cp = [{"start": 0, "end": N_LAYERS - 1, "length": N_LAYERS}]

        result = plateau_alignment(fp, cp, LAYER_IDX)

        assert result["mean_overlap_iou"] == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_plateau_iou_is_0(self):
        """
        Feature plateau [0, 3] and cluster plateau [5, 7] have no layers
        in common → IoU = 0.0.
        """
        fp = _feature_plateau_stub([(0, 3)])
        cp = [{"start": 5, "end": 7, "length": 3}]

        result = plateau_alignment(fp, cp, LAYER_IDX)

        assert result["mean_overlap_iou"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_aligned"] == 0

    def test_partial_overlap_iou_correct(self):
        """
        Feature plateau [0, 5] and cluster plateau [3, 7]:
        intersection = {3,4,5} = 3 layers
        union        = {0..7}  = 8 layers
        IoU = 3/8 = 0.375
        """
        fp = _feature_plateau_stub([(0, 5)])
        cp = [{"start": 3, "end": 7, "length": 5}]

        result = plateau_alignment(fp, cp, LAYER_IDX)

        assert result["mean_overlap_iou"] == pytest.approx(3 / 8, abs=1e-6)

    def test_empty_feature_plateaus_returns_error(self):
        """
        When feature_plateaus contains no per_feature entries the function
        must return a dict with an 'error' key, not raise.
        """
        fp_empty = {"per_feature": [], "summary": {}}
        cp = [{"start": 0, "end": 7, "length": 8}]

        result = plateau_alignment(fp_empty, cp, LAYER_IDX)

        assert "error" in result or result.get("falsification") == "untestable"

    def test_empty_cluster_plateaus_returns_error(self):
        """
        When cluster_plateaus is an empty list the function must return a
        dict with an 'error' key.
        """
        fp = _feature_plateau_stub([(0, 7)])

        result = plateau_alignment(fp, [], LAYER_IDX)

        assert "error" in result or result.get("falsification") == "untestable"

    def test_n_aligned_counts_iou_above_threshold(self):
        """
        n_aligned counts plateaus with best IoU > 0.3.
        Two feature plateaus: [0,7] overlaps fully, [0,1] overlaps minimally.
        """
        # Two feature plateaus on the same feature
        fp = _feature_plateau_stub([(0, N_LAYERS - 1), (0, 1)])
        cp = [{"start": 0, "end": N_LAYERS - 1, "length": N_LAYERS}]

        result = plateau_alignment(fp, cp, LAYER_IDX)

        # [0,7] vs [0,7] → IoU=1.0 → aligned
        # [0,1] vs [0,7] → IoU=2/8=0.25 < 0.3 → not aligned
        assert result["n_aligned"] == 1
        assert result["n_feature_plateaus"] == 2


# ===========================================================================
# 4.  analyze_chorus_at_layer  (chorus.py)
# ===========================================================================


class TestAnalyzeChorusAtLayer:
    """Co-activation cliques and their correspondence to cluster labels."""

    def test_zero_activations_produce_no_cliques(self):
        """
        When all z values are zero no feature co-activates with any other
        → the co-activation graph has no edges → no cliques.
        """
        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        traj   = _make_traj(z, active)
        labels = _binary_labels()

        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)

        assert result["n_cliques"] == 0, (
            "Zero activations should produce no cliques"
        )

    def test_perfect_cluster_split_yields_two_pure_cliques(self):
        """
        Features 0-9 fire only on cluster-0 tokens; features 10-19 only on
        cluster-1 tokens.  The co-activation graph splits cleanly into two
        connected components, each pure.
        """
        labels = _binary_labels()
        half   = N_TOKENS // 2

        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)

        # Cluster-0 features
        active[:half, :10, :]  = True
        z[:half, :10, :]       = 1.0
        # Cluster-1 features
        active[half:, 10:, :]  = True
        z[half:, 10:, :]       = 1.0

        traj   = _make_traj(z, active)
        result = analyze_chorus_at_layer(
            traj, labels, cc_layer_idx=0, coact_threshold=0.3
        )

        assert result["n_cliques"] == 2, (
            f"Expected 2 cliques for perfect split, got {result['n_cliques']}"
        )
        assert result["purity"]["summary"]["mean_purity"] == pytest.approx(1.0), (
            "Both cliques should be perfectly pure"
        )

    def test_result_contains_required_keys(self):
        """analyze_chorus_at_layer must always return the expected top-level keys."""
        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        traj   = _make_traj(z, active)
        labels = _binary_labels()

        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)

        for key in ("n_cliques", "purity", "ari", "coact_density"):
            assert key in result, f"Missing key '{key}' in chorus result"

    def test_coact_density_in_unit_interval(self):
        """Co-activation graph density must always be in [0, 1]."""
        rng    = np.random.default_rng(13)
        z      = rng.standard_normal((N_TOKENS, N_FEATURES, N_LAYERS)).astype(np.float32)
        active = rng.random((N_TOKENS, N_FEATURES, N_LAYERS)) > 0.5
        traj   = _make_traj(z, active)
        labels = _binary_labels()

        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)

        assert 0.0 <= result["coact_density"] <= 1.0, (
            f"Density out of bounds: {result['coact_density']}"
        )

    def test_all_tokens_same_cluster_no_valid_purity(self):
        """
        If all tokens carry the same label the chorus still runs without
        error (though purity / ARI may be degenerate).
        """
        labels_uniform = np.zeros(N_TOKENS, dtype=int)  # all cluster-0
        z      = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)
        traj   = _make_traj(z, active)

        # Should not raise
        result = analyze_chorus_at_layer(traj, labels_uniform, cc_layer_idx=0)
        assert isinstance(result, dict)



class TestAnalyzeChorusAtLayer_Extra:

    def test_result_contains_required_keys(self):
        """
        Smoke-test the result structure for the known required keys:
        n_cliques, cliques, ari (or cluster_ari).
        Prevents silent schema drift.
        """
        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros_like(z, dtype=bool)
        # Two cliques: features 0-4 on tokens 0-14, features 5-9 on tokens 15-29
        active[:15,  :5,  0] = True
        active[15:, 5:10, 0] = True
        z[active] = 1.0
        traj   = _make_traj(z, active)
        labels = _binary_labels()

        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)

        assert "n_cliques" in result, "Missing 'n_cliques'"
        assert "cliques"   in result, "Missing 'cliques'"

    def test_single_cluster_labels_does_not_raise(self):
        """
        When all tokens share the same label the function must return
        without raising, even if ARI is undefined.
        """
        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros_like(z, dtype=bool)
        active[:, :5, 0] = True
        z[active] = 1.0
        traj   = _make_traj(z, active)
        labels = np.zeros(N_TOKENS, dtype=int)   # single cluster

        # Must not raise
        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)
        assert isinstance(result, dict)

    def test_two_pure_cliques_ari_is_1(self):
        """
        Features 0-9 fire only on cluster-0 tokens; features 10-19 only on
        cluster-1 tokens.  Each clique is pure → ARI between clique membership
        and cluster labels should be 1.0.
        Extends the existing test to check the ARI value, not just n_cliques.
        """
        half   = N_TOKENS // 2
        labels = _binary_labels()
        z      = np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        active = np.zeros_like(z, dtype=bool)

        # Group A: features 0-9 active on tokens 0..half-1
        active[:half, :10,  0] = True
        # Group B: features 10-19 active on tokens half..end
        active[half:, 10:, 0]  = True
        z[active] = 1.0
        traj = _make_traj(z, active)

        result = analyze_chorus_at_layer(traj, labels, cc_layer_idx=0)

        # Accept either 'ari' or 'cluster_ari' key
        ari_key = "ari" if "ari" in result else "cluster_ari"
        if ari_key in result:
            assert result[ari_key] == pytest.approx(1.0, abs=0.05), (
                f"Expected ARI ≈ 1.0, got {result[ari_key]}"
            )