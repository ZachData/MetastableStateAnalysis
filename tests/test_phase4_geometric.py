"""
tests/test_phase4_geometric.py

Tests for p4_mstate_features/geometric.py:
  - lda_stability_across_layers
  - probe_accuracy_trajectory
  - pca_on_deltas

geometric.py has no imports from phase3 or torch; it only requires numpy.
It can be imported directly from the project root without any stubs.

Standard fixture dimensions: n_layers=8, n_tokens=30, d=16, n_clusters=2
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so geometric.py can be found.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest

from geometric import (
    lda_stability_across_layers,
    lda_directions,
    probe_accuracy_trajectory,
    pca_on_deltas,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N_LAYERS = 8
N_TOKENS = 30
D        = 16


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _binary_labels(n: int = N_TOKENS) -> np.ndarray:
    """Balanced binary labels: [0]*half + [1]*half."""
    half = n // 2
    return np.array([0] * half + [1] * (n - half))


def _antipodal_activations(
    n_tokens: int = N_TOKENS,
    d: int = D,
    n_layers: int = N_LAYERS,
    separation: float = 10.0,
    noise_scale: float = 0.01,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Two clusters placed at ±separation*e_0 in R^d with a tiny noise.
    Returns (activations_per_layer, labels) where activations is a dict
    mapping each layer index to an (n_tokens, d) array.
    Clusters are perfectly linearly separable by construction.
    """
    half   = n_tokens // 2
    labels = np.array([0] * half + [1] * (n_tokens - half))

    acts: dict[int, np.ndarray] = {}
    for l in range(n_layers):
        rng = np.random.default_rng(l)
        X   = np.zeros((n_tokens, d))
        X[:half,  0] = +separation
        X[half:,  0] = -separation
        X += rng.standard_normal((n_tokens, d)) * noise_scale
        acts[l] = X

    return acts, labels


def _random_label_activations(
    n_tokens: int = N_TOKENS,
    d: int = D,
    n_layers: int = N_LAYERS,
    seed: int = 0,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Activations drawn from N(0,1) with binary labels assigned randomly,
    independent of the activations.
    """
    rng    = np.random.default_rng(seed)
    labels = (rng.random(n_tokens) > 0.5).astype(int)
    acts   = {l: rng.standard_normal((n_tokens, d)) for l in range(n_layers)}
    return acts, labels


# ===========================================================================
# 1.  lda_stability_across_layers
# ===========================================================================


class TestLDAStabilityAcrossLayers:
    """LDA directions and accuracy across model layers."""

    def test_antipodal_clusters_achieve_perfect_accuracy(self):
        """
        Two clusters at ±10 * e_0 are perfectly separable.  LDA must reach
        accuracy = 1.0 at every layer.
        """
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = lda_stability_across_layers(acts, labs)

        for layer, info in result["per_layer"].items():
            assert info.get("accuracy") == pytest.approx(1.0, abs=1e-6), (
                f"Layer {layer}: expected accuracy=1.0, got {info['accuracy']}"
            )

    def test_random_labels_accuracy_near_chance(self):
        """
        With n=200 >> d=16, the LDA projection cannot memorise random labels.
        Accuracy at each layer must stay well below the perfectly separable
        baseline of 1.0.  We allow up to 0.85 to avoid flakiness.
        """
        n_large = 200
        rng     = np.random.default_rng(42)
        labels  = (np.arange(n_large) % 2).astype(int)      # alternating 0/1
        acts    = {l: rng.standard_normal((n_large, D)) for l in range(N_LAYERS)}
        labs    = {l: labels for l in range(N_LAYERS)}

        result  = lda_stability_across_layers(acts, labs)

        for layer, info in result["per_layer"].items():
            acc = info.get("accuracy", 0.5)
            assert acc <= 0.85, (
                f"Layer {layer}: random-label LDA accuracy {acc:.3f} "
                f"unexpectedly high (> 0.85)"
            )

    def test_cosine_trajectory_length(self):
        """
        With n_layers consecutive layers there should be n_layers-1
        inter-layer cosine similarity entries.
        """
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = lda_stability_across_layers(acts, labs)

        assert len(result["cosine_trajectory"]) == N_LAYERS - 1, (
            f"Expected {N_LAYERS-1} cosine entries, "
            f"got {len(result['cosine_trajectory'])}"
        )

    def test_cosine_trajectory_values_in_unit_interval(self):
        """All cosine similarity values must be in [0, 1] (abs taken)."""
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = lda_stability_across_layers(acts, labs)

        for entry in result["cosine_trajectory"]:
            c = entry["cosine"]
            assert 0.0 <= c <= 1.0, f"Cosine out of [0,1]: {c}"

    def test_antipodal_cosine_near_1(self):
        """
        With d=2 the LDA subspace is essentially one-dimensional (e_0) and
        cannot be rotated by within-class scatter noise → consecutive cosines
        should all exceed 0.9.  (High-d cases can produce lower cosines even
        for perfectly separable clusters because S_W has random orientation
        in the orthogonal complement; this test pins d=2 to eliminate that.)
        """
        n_tokens = 100
        d_small  = 2
        half     = n_tokens // 2
        labels   = np.array([0] * half + [1] * (n_tokens - half))

        acts: dict[int, np.ndarray] = {}
        for l in range(N_LAYERS):
            rng      = np.random.default_rng(l)
            X        = np.zeros((n_tokens, d_small))
            X[:half,  0] = +10.0
            X[half:,  0] = -10.0
            X       += rng.standard_normal((n_tokens, d_small)) * 0.001
            acts[l]  = X

        labs   = {l: labels for l in range(N_LAYERS)}
        result = lda_stability_across_layers(acts, labs)

        for entry in result["cosine_trajectory"]:
            assert entry["cosine"] >= 0.9, (
                f"d=2 antipodal cosine should be ≥ 0.9, got {entry['cosine']:.4f}"
            )

    def test_per_layer_keys_present(self):
        """Each per_layer entry must contain 'accuracy' and 'n_classes'."""
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = lda_stability_across_layers(acts, labs)

        for layer, info in result["per_layer"].items():
            assert "accuracy" in info, f"Layer {layer} missing 'accuracy'"
            assert "n_classes" in info, f"Layer {layer} missing 'n_classes'"

    def test_single_layer_no_cosine(self):
        """A single layer produces no cosine trajectory entries."""
        acts   = {0: np.random.default_rng(0).standard_normal((N_TOKENS, D))}
        labels = _binary_labels()
        labs   = {0: labels}

        result = lda_stability_across_layers(acts, labs)

        assert result["cosine_trajectory"] == []

    def test_lda_directions_low_level_accuracy_shape(self):
        """
        lda_directions (single layer) returns accuracy in [0,1] and directions
        of shape (n_classes-1, d).
        """
        rng    = np.random.default_rng(0)
        X      = np.zeros((N_TOKENS, D))
        X[:N_TOKENS // 2, 0] = 10.0
        X[N_TOKENS // 2:, 0] = -10.0
        labels = _binary_labels()

        result = lda_directions(X, labels)

        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["directions"].shape == (1, D), (
            f"Expected shape (1, {D}), got {result['directions'].shape}"
        )
        assert result["n_classes"] == 2


# ===========================================================================
# 2.  probe_accuracy_trajectory
# ===========================================================================


class TestProbeAccuracyTrajectory:
    """Linear probe accuracy across layers."""

    def test_separable_clusters_accuracy_1(self):
        """
        Two clusters separated along the first dimension by 20 units.
        A linear probe with zero regularisation should fit perfectly.
        """
        acts, labels = _antipodal_activations(separation=20.0, noise_scale=0.0)
        labs = {l: labels for l in range(N_LAYERS)}

        result = probe_accuracy_trajectory(acts, labs, reg=1e-6)

        for layer, info in result["per_layer"].items():
            assert info["accuracy"] == pytest.approx(1.0, abs=1e-6), (
                f"Layer {layer}: expected accuracy=1.0, got {info['accuracy']}"
            )

    def test_random_labels_accuracy_not_perfect(self):
        """
        With n=200 >> d=16 the ridge solver cannot memorise random labels:
        max training accuracy across all layers must stay below 0.8.

        At n=30 (n/d ≈ 2) the probe can interpolate any label vector, so
        training accuracy reaches ~1.0 and the test would give a false failure.
        n=200 gives n/d ≈ 12.5, which is well past the interpolation threshold.
        """
        n_large = 200
        rng     = np.random.default_rng(99)
        labels  = (np.arange(n_large) % 2).astype(int)   # alternating; no signal
        acts    = {l: rng.standard_normal((n_large, D)) for l in range(N_LAYERS)}
        labs    = {l: labels for l in range(N_LAYERS)}

        result  = probe_accuracy_trajectory(acts, labs)

        max_acc = result["summary"]["max_accuracy"]
        assert max_acc <= 0.8, (
            f"Random-label probe reached {max_acc:.3f}; expected ≤ 0.8 "
            f"(n=200, d={D})"
        )

    def test_summary_contains_required_keys(self):
        """summary must contain mean_accuracy, max_accuracy, min_accuracy."""
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = probe_accuracy_trajectory(acts, labs)

        for key in ("mean_accuracy", "max_accuracy", "min_accuracy"):
            assert key in result["summary"], f"Missing summary key '{key}'"

    def test_summary_ordering(self):
        """min_accuracy ≤ mean_accuracy ≤ max_accuracy."""
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = probe_accuracy_trajectory(acts, labs)
        s = result["summary"]

        assert s["min_accuracy"] <= s["mean_accuracy"] + 1e-9
        assert s["mean_accuracy"] <= s["max_accuracy"] + 1e-9

    def test_probe_directions_stored_per_layer(self):
        """
        probe_directions should contain one weight matrix per processed layer,
        each of shape (n_classes, d).
        """
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = probe_accuracy_trajectory(acts, labs)

        assert len(result["probe_directions"]) == N_LAYERS
        for layer, W in result["probe_directions"].items():
            assert W.shape == (2, D), (
                f"Layer {layer}: expected probe shape (2, {D}), got {W.shape}"
            )

    def test_per_layer_n_classes_correct(self):
        """Each per_layer entry should record n_classes = 2."""
        acts, labels = _antipodal_activations()
        labs = {l: labels for l in range(N_LAYERS)}

        result = probe_accuracy_trajectory(acts, labs)

        for layer, info in result["per_layer"].items():
            assert info.get("n_classes") == 2, (
                f"Layer {layer}: expected n_classes=2, got {info.get('n_classes')}"
            )


# ===========================================================================
# 3.  pca_on_deltas
# ===========================================================================


class TestPCAOnDeltas:
    """PCA of layer-to-layer activation deltas."""

    def test_rank1_deltas_top_component_explains_nearly_all(self):
        """
        When all token deltas are proportional to a single direction v,
        the first PC explains ≈ 100 % of variance.
        """
        rng  = np.random.default_rng(42)
        v    = rng.standard_normal(D)
        v   /= np.linalg.norm(v)
        base = rng.standard_normal((N_TOKENS, D))

        acts: dict[int, np.ndarray] = {}
        for l in range(N_LAYERS):
            coeffs    = rng.standard_normal(N_TOKENS)
            acts[l]   = base + l * np.outer(coeffs, v)

        result = pca_on_deltas(acts)

        for trans in result["per_transition"]:
            top1 = trans["top1_explained"]
            assert top1 == pytest.approx(1.0, abs=0.02), (
                f"Rank-1 delta: expected top1≈1.0, "
                f"got {top1:.4f} at transition {trans['layer_from']}→{trans['layer_to']}"
            )

    def test_random_deltas_explained_ratios_sum_to_1(self):
        """
        For arbitrary Gaussian deltas, the eigenvalues in explained_ratio
        should sum to 1 (up to floating-point rounding).
        """
        rng  = np.random.default_rng(7)
        acts = {l: rng.standard_normal((N_TOKENS, D)) for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        for trans in result["per_transition"]:
            ratio_sum = sum(trans["explained_ratio"])
            assert ratio_sum == pytest.approx(1.0, abs=1e-6), (
                f"Explained ratios sum {ratio_sum:.8f} ≠ 1.0 at "
                f"{trans['layer_from']}→{trans['layer_to']}"
            )

    def test_n_transitions_equals_n_layers_minus_1(self):
        """
        Given n_layers layer keys, there are n_layers - 1 transitions.
        """
        rng  = np.random.default_rng(0)
        acts = {l: rng.standard_normal((N_TOKENS, D)) for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        assert len(result["per_transition"]) == N_LAYERS - 1, (
            f"Expected {N_LAYERS-1} transitions, "
            f"got {len(result['per_transition'])}"
        )

    def test_transition_keys_identify_layer_pairs(self):
        """
        Each transition entry must contain 'layer_from' and 'layer_to'
        with layer_to == layer_from + 1 for consecutive integer keys.
        """
        rng  = np.random.default_rng(1)
        acts = {l: rng.standard_normal((N_TOKENS, D)) for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        for i, trans in enumerate(result["per_transition"]):
            assert trans["layer_from"] == i,     f"Unexpected layer_from: {trans}"
            assert trans["layer_to"]   == i + 1, f"Unexpected layer_to: {trans}"

    def test_zero_delta_total_variance_is_zero(self):
        """
        When all layers carry the same activations the deltas are all zero
        → total_variance = 0.
        """
        rng  = np.random.default_rng(5)
        base = rng.standard_normal((N_TOKENS, D))
        acts = {l: base.copy() for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        for trans in result["per_transition"]:
            assert trans["total_variance"] == pytest.approx(0.0, abs=1e-10), (
                f"Zero delta should give total_variance=0, "
                f"got {trans['total_variance']}"
            )

    def test_summary_transition_count(self):
        """summary['n_transitions'] should equal len(per_transition)."""
        rng  = np.random.default_rng(3)
        acts = {l: rng.standard_normal((N_TOKENS, D)) for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        assert result["summary"]["n_transitions"] == len(result["per_transition"])

    def test_explained_ratios_all_non_negative(self):
        """All explained ratio values must be ≥ 0 (eigenvalues non-negative)."""
        rng  = np.random.default_rng(99)
        acts = {l: rng.standard_normal((N_TOKENS, D)) for l in range(N_LAYERS)}

        result = pca_on_deltas(acts)

        for trans in result["per_transition"]:
            for r in trans["explained_ratio"]:
                assert r >= -1e-9, f"Negative explained ratio {r} in {trans}"
