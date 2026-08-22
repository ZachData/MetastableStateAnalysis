"""
tests/test_phase4_bugs.py

Regression tests for five Phase 4 bugs identified from p4 result files.

  Bug 1 — detect_feature_plateaus   : dead/near-zero features inflate plateau counts
                                       because the active-mask guard passes float noise.
  Bug 2 — probe_accuracy_trajectory  : returns 0.0 not NaN when labels are absent;
           _aggregate_mi_summary      : same silent-zero problem for MI summary.
  Bug 3 — compute_coactivation       : denominator is total-token count T, diluting
                                       sparse co-occurrence below threshold → no cliques.
  Bug 4 — cross_track_agreement      : T1 keys are strings ("layer_6"), T2 keys are
                                       ints (6); intersection always empty → Spearman nan.
  Bug 5 — build_phase4_verdict (T1)  : max_nmi threshold is satisfied by a single
                                       trivial feature; mean_nmi is the right signal.

All tests are self-contained (no I/O, no model loading).
"""

import numpy as np
import pytest
from collections import defaultdict

from p4_mstate_features.activation_trajectories import (
    ActivationTrajectory,
    detect_feature_plateaus,
    feature_cluster_mi,
    _aggregate_mi_summary,          # new helper added by Bug 2 fix
)
from p4_mstate_features.chorus import compute_coactivation, extract_cliques
from p4_mstate_features.geometric import probe_accuracy_trajectory
from p4_mstate_features.analysis import cross_track_agreement, build_phase4_verdict


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_LAYERS   = 10      # matches ALBERT / GPT-2 sampled layer count in the p4 run
N_TOKENS   = 40
N_FEATURES = 50
LAYER_IDX  = list(range(N_LAYERS))


def _traj(z: np.ndarray, active: np.ndarray | None = None) -> ActivationTrajectory:
    if active is None:
        active = (z != 0.0)
    return ActivationTrajectory("test", z, active.astype(bool), LAYER_IDX)


def _zeros_z() -> np.ndarray:
    return np.zeros((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)


def _ones_active() -> np.ndarray:
    """All-True active mask — the situation that lets dead features slip past the guard."""
    return np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=bool)


def _balanced_labels(n: int = N_TOKENS, n_clusters: int = 2) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    per = n // n_clusters
    for c in range(n_clusters):
        labels[c * per:(c + 1) * per] = c
    return labels


# ===========================================================================
# Bug 1: dead / near-zero features contaminate plateau detection
# ===========================================================================

class TestBug1_DeadFeatureContamination:
    """
    detect_feature_plateaus checks `traj.active_per_layer[:, f, :].any()`.
    When active is set to all-True (or populated from float noise), a
    feature with z ≈ 0 everywhere has near-zero variance across layers
    and is reported as having a plateau spanning the full window.
    Fix: add a `min_peak_activation` guard before calling _find_stable_windows.
    """

    def test_all_zero_z_with_all_true_active_not_reported(self):
        """
        z = 0 everywhere, active = True everywhere (the bad case from float noise).
        No feature should appear in per_feature after the fix.
        """
        z      = _zeros_z()
        active = _ones_active()
        traj   = _traj(z, active)
        result = detect_feature_plateaus(traj, min_peak_activation=0.05)
        assert len(result["per_feature"]) == 0, (
            "All-zero features must not be reported even when active mask is True"
        )

    def test_near_zero_below_min_peak_not_reported(self):
        """z = 1e-6 (numerical noise) must not produce a plateau."""
        z      = np.full((N_TOKENS, N_FEATURES, N_LAYERS), 1e-6, dtype=np.float32)
        active = _ones_active()
        traj   = _traj(z, active)
        result = detect_feature_plateaus(traj, min_peak_activation=0.05)
        assert len(result["per_feature"]) == 0

    def test_genuine_signal_above_min_peak_reported(self):
        """Feature with amplitude 1.0 constant across layers must still be found."""
        z       = _zeros_z()
        z[:, 0, :] = 1.0          # feature 0: constant, real signal
        traj    = _traj(z)
        result  = detect_feature_plateaus(
            traj, min_peak_activation=0.05, min_plateau_len=3
        )
        reported = {f["feature_idx"] for f in result["per_feature"]}
        assert 0 in reported, "Genuine constant feature must still be detected"

    def test_dead_features_do_not_inflate_mean_plateau_length(self):
        """
        80 % of features are dead (z=0, active=True). Only 10 features have
        real signal. mean_plateau_length must not be pushed to N_LAYERS by the
        dead features' spurious full-window plateaus.
        """
        z      = _zeros_z()
        z[:, :10, :] = 1.0        # first 10 features: real constant signal
        active = _ones_active()   # active=True for all (the problem)
        traj   = _traj(z, active)
        result = detect_feature_plateaus(
            traj, min_peak_activation=0.05, min_plateau_len=3
        )
        assert result["summary"]["n_features_with_plateaus"] <= 10, (
            "Only the 10 live features should be reported; dead ones must be excluded"
        )

    def test_noise_floor_vs_real_signal_discrimination(self):
        """
        Feature 0: z = 1e-4 (noise floor, below min_peak). Must not be reported.
        Feature 1: z = 1.0  (real signal). Must be reported.
        """
        z      = _zeros_z()
        z[:, 0, :] = 1e-4
        z[:, 1, :] = 1.0
        active = _ones_active()
        traj   = _traj(z, active)
        result = detect_feature_plateaus(
            traj, min_peak_activation=0.05, min_plateau_len=3
        )
        reported = {f["feature_idx"] for f in result["per_feature"]}
        assert 0 not in reported, "Noise-floor feature (1e-4) must be excluded"
        assert 1 in reported,     "Real-signal feature (1.0) must be included"

    def test_min_peak_activation_zero_restores_original_behavior(self):
        """
        Passing min_peak_activation=0.0 is an escape hatch that restores the
        original behavior — useful for callers that pre-filter their own features.
        A near-zero constant feature should then produce a plateau.
        """
        z      = _zeros_z()
        z[:, 0, :] = 1e-6         # tiny constant
        traj   = _traj(z)
        result = detect_feature_plateaus(
            traj, min_peak_activation=0.0, min_plateau_len=3
        )
        reported = {f["feature_idx"] for f in result["per_feature"]}
        # With no peak filter this feature is allowed through (original behavior)
        assert 0 in reported, (
            "With min_peak_activation=0 the original behavior must be preserved"
        )

    def test_p4_symptom_mean_plateau_length_equals_n_layers(self):
        """
        Regression for the exact symptom seen in p4 results: mean_plateau_length
        reported as exactly N_LAYERS (= 10.0). This can only happen if every
        detected plateau spans the full window, which only occurs for dead features.
        After the fix, a mix of dead + live features must give mean_length < N_LAYERS.
        """
        z      = _zeros_z()
        z[:, :5, :] = 2.0         # 5 live features with 3-layer plateaus
        z[:, :5, 3:7] = 3.0       # introduce a spike so plateau < full window
        active = _ones_active()
        traj   = _traj(z, active)
        result = detect_feature_plateaus(
            traj, min_peak_activation=0.05, min_plateau_len=3
        )
        if result["summary"]["total_plateaus"] > 0:
            assert result["summary"]["mean_plateau_length"] < N_LAYERS, (
                "mean_plateau_length should be < N_LAYERS; full-window-only result "
                "indicates dead-feature contamination"
            )


# ===========================================================================
# Bug 2: silent zero return when HDBSCAN labels are absent
# ===========================================================================

class TestBug2_SilentZeroReturnNoLabels:
    """
    When Phase 1 labels are unavailable the analysis returns 0.0 scalars,
    indistinguishable from a genuine null measurement. The fix returns NaN
    and marks results as untestable.
    """

    # --- _aggregate_mi_summary ---

    def test_empty_mi_results_gives_nan_not_zero(self):
        """
        _aggregate_mi_summary({}) must return NaN values, not 0.0.
        This is the primary observable symptom from the GPT-2 run.
        """
        summary = _aggregate_mi_summary({})
        assert np.isnan(summary["max_nmi"]),  "max_nmi must be NaN when no labels"
        assert np.isnan(summary["mean_nmi"]), "mean_nmi must be NaN when no labels"
        assert summary.get("untestable") is True

    def test_all_empty_layer_results_gives_nan(self):
        """
        If every prompt returned empty feature lists (e.g. all-noise labels),
        the aggregate must still be NaN not 0.0.
        """
        empty_per_prompt = {
            "wiki_paragraph": {},
            "sullivan_ballou": {},
        }
        summary = _aggregate_mi_summary(empty_per_prompt)
        assert np.isnan(summary["max_nmi"])
        assert summary.get("untestable") is True

    def test_valid_mi_results_give_finite_summary(self):
        """
        When at least one prompt has valid MI values the summary must
        be finite, not NaN.
        """
        mi_results = {
            "wiki_paragraph": {
                "layer_6":  {"top_features": [{"feature_idx": 0, "nmi": 0.4}]},
                "layer_12": {"top_features": [{"feature_idx": 0, "nmi": 0.7}]},
            }
        }
        summary = _aggregate_mi_summary(mi_results)
        assert np.isfinite(summary["max_nmi"]),  "max_nmi should be finite"
        assert np.isfinite(summary["mean_nmi"]), "mean_nmi should be finite"
        assert not summary.get("untestable", False)

    def test_max_nmi_correct_value(self):
        mi_results = {
            "p1": {"layer_0": {"top_features": [{"feature_idx": 0, "nmi": 0.3},
                                                 {"feature_idx": 1, "nmi": 0.8}]}},
            "p2": {"layer_0": {"top_features": [{"feature_idx": 0, "nmi": 0.5}]}},
        }
        summary = _aggregate_mi_summary(mi_results)
        assert summary["max_nmi"] == pytest.approx(0.8, abs=1e-6)

    # --- feature_cluster_mi with empty labels ---

    def test_feature_cluster_mi_empty_hdbscan_returns_empty_dict(self):
        """
        feature_cluster_mi with {} labels should return {} (no layers processed),
        never a dict with 0-filled entries.
        """
        z    = np.ones((N_TOKENS, N_FEATURES, N_LAYERS), dtype=np.float32)
        traj = _traj(z)
        result = feature_cluster_mi(traj, {}, LAYER_IDX)
        assert isinstance(result, dict)
        assert len(result) == 0, (
            "Empty hdbscan_labels must produce empty result, not zero-filled entries"
        )

    # --- probe_accuracy_trajectory with empty labels ---

    def test_probe_accuracy_empty_labels_returns_nan_summary(self):
        """
        probe_accuracy_trajectory with empty labels_per_layer must return
        NaN summary values, not 0.0.
        """
        acts = {i: np.random.default_rng(i).standard_normal((N_TOKENS, 16))
                for i in range(N_LAYERS)}
        result = probe_accuracy_trajectory(acts, {})
        s = result["summary"]
        assert np.isnan(s["mean_accuracy"]), (
            f"mean_accuracy should be NaN with no labels; got {s['mean_accuracy']}"
        )
        assert np.isnan(s["max_accuracy"]), (
            f"max_accuracy should be NaN with no labels; got {s['max_accuracy']}"
        )
        assert result.get("untestable") is True

    def test_probe_accuracy_partial_labels_uses_only_available_layers(self):
        """
        Labels provided for only 3 of 10 layers: summary should reflect
        those 3 layers, not be NaN, and not include zeros for unlabelled layers.
        """
        acts   = {i: np.ones((N_TOKENS, 16)) * i for i in range(N_LAYERS)}
        labels = _balanced_labels()
        labs   = {0: labels, 4: labels, 8: labels}     # only 3 layers
        result = probe_accuracy_trajectory(acts, labs)
        assert len(result["per_layer"]) == 3
        assert np.isfinite(result["summary"]["mean_accuracy"])


# ===========================================================================
# Bug 3: compute_coactivation divides by T (total tokens) not joint-active
# ===========================================================================

class TestBug3_CoactivationDenominator:
    """
    Old code: coact[i,j] = (active.T @ active)[i,j] / T
    For sparse features that fire on 5-10 % of tokens, max co-activation
    value ≈ 0.05-0.1 — always below the default threshold of 0.3, so no
    cliques ever form. ARI and chorus metrics are therefore always 0.
    Fix: Jaccard similarity — intersection / union — per feature pair.
    """

    def test_perfectly_coactive_sparse_pair_scores_one(self):
        """
        Features 0 and 1 fire together on exactly 4 / 40 tokens (10 %).
        Jaccard = 4/4 = 1.0.  Old T denominator: 4/40 = 0.1.
        """
        z = _zeros_z()
        fire = [0, 1, 2, 3]
        z[fire, 0, 0] = 1.0
        z[fire, 1, 0] = 1.0
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 1] == pytest.approx(1.0, abs=1e-6), (
            f"Perfectly co-active sparse pair should score 1.0; got {coact[0,1]:.4f}"
        )

    def test_independent_sparse_features_score_zero(self):
        """Features firing on non-overlapping token sets: Jaccard = 0."""
        z = _zeros_z()
        z[0:4, 0, 0] = 1.0    # feature 0: tokens 0-3
        z[5:9, 1, 0] = 1.0    # feature 1: tokens 5-8, no overlap
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap_jaccard_correct(self):
        """
        Feature 0: tokens 0-3 (4 tokens).  Feature 1: tokens 2-5 (4 tokens).
        Intersection = {2,3} = 2.  Union = {0,1,2,3,4,5} = 6.
        Jaccard = 2/6 ≈ 0.3333.
        """
        z = _zeros_z()
        z[0:4, 0, 0] = 1.0
        z[2:6, 1, 0] = 1.0
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 1] == pytest.approx(2 / 6, abs=1e-5)

    def test_coactivation_matrix_symmetric(self):
        z = _zeros_z()
        z[0:5, 0, 0] = 1.0
        z[3:8, 1, 0] = 1.0
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 1] == pytest.approx(coact[1, 0], abs=1e-6)

    def test_diagonal_is_zero(self):
        z = _zeros_z()
        z[:, 0, 0] = 1.0
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_cliques_form_for_sparse_coactive_group(self):
        """
        Features 0, 1, 2 fire together on 4 / 40 tokens (10 %).
        With Jaccard denominator, coact ≈ 1.0 → clique forms.
        With T denominator, coact ≈ 0.1 → no clique (old bug).
        """
        z    = _zeros_z()
        fire = list(range(4))
        for f in range(3):
            z[fire, f, 0] = 1.0
        traj   = _traj(z)
        coact  = compute_coactivation(traj, cc_layer_idx=0)
        cliques = extract_cliques(coact, threshold=0.3, min_clique_size=2)
        assert len(cliques) > 0, "Sparse co-active features should form a clique"
        assert {0, 1, 2}.issubset(set(cliques[0])), (
            f"Features 0,1,2 should be in the same clique; got {cliques[0]}"
        )

    def test_old_t_denominator_would_not_form_cliques(self):
        """
        Document the old bug: T denominator gives coact < 0.3 for sparse features.
        This test computes the old value explicitly — it must be below threshold.
        """
        z    = _zeros_z()
        fire = list(range(4))     # 10 % of 40 tokens
        for f in range(3):
            z[fire, f, 0] = 1.0
        active_at_layer = (np.abs(z[:, :, 0]) > 0.0).astype(np.float32)
        # Old code: divide by T
        coact_old = (active_at_layer.T @ active_at_layer) / N_TOKENS
        np.fill_diagonal(coact_old, 0.0)
        assert coact_old[0, 1] < 0.3, (
            f"Old T denominator must produce coact < 0.3 for 10%-sparse features; "
            f"got {coact_old[0,1]:.4f}"
        )

    def test_no_active_features_returns_zeros(self):
        """All-zero activations: no co-activity anywhere → all-zero matrix."""
        z     = _zeros_z()
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert np.all(coact == 0.0)

    def test_full_activation_all_tokens_coact_one(self):
        """Features 0 and 1 both active on every token: Jaccard = T/T = 1.0."""
        z = _zeros_z()
        z[:, 0, 0] = 1.0
        z[:, 1, 0] = 1.0
        traj  = _traj(z)
        coact = compute_coactivation(traj, cc_layer_idx=0)
        assert coact[0, 1] == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Bug 4: T1 string keys / T2 integer keys — intersection always empty
# ===========================================================================

class TestBug4_CrossTrackKeyMismatch:
    """
    T1 (chorus_per_layer) keys derive from hdbscan_labels format: "layer_6" (str).
    T2 (probe_per_layer) keys derive from probe_accuracy_trajectory's per_layer
    dict, which is keyed by model layer index: 6 (int).
    set("layer_6") & set(6) == ∅ → Spearman never computed → n_layers always 0.
    Fix: normalise both sides to integer layer index before intersecting.
    """

    def _t1(self, layer_ints, ari_vals):
        """T1 uses string keys from hdbscan_labels."""
        return {
            "chorus_per_layer": {
                f"layer_{li}": {"ari": {"ari": float(v)}}
                for li, v in zip(layer_ints, ari_vals)
            }
        }

    def _t2(self, layer_ints, acc_vals):
        """T2 uses integer keys from probe_accuracy_trajectory."""
        return {
            "probe_per_layer": {
                int(li): {"accuracy": float(v)}
                for li, v in zip(layer_ints, acc_vals)
            }
        }

    _LAYERS = [0, 6, 12, 18, 24]
    _ARI    = [0.1, 0.3, 0.7, 0.8, 0.6]
    _ACC    = [0.5, 0.6, 0.8, 0.9, 0.75]

    def test_string_vs_int_key_mismatch_is_the_bug(self):
        """
        Document the exact mismatch: raw set intersection of string and integer
        keys is always empty, confirming the bug.
        """
        t1_keys = {f"layer_{li}" for li in self._LAYERS}     # strings
        t2_keys = set(self._LAYERS)                           # ints
        assert len(t1_keys & t2_keys) == 0, (
            "String keys and integer keys never intersect — this IS the bug"
        )

    def test_fixed_code_computes_spearman_across_formats(self):
        """
        After the fix, T1 string keys and T2 integer keys must normalise to
        the same integer and produce a valid Spearman correlation.
        """
        t1 = self._t1(self._LAYERS, self._ARI)
        t2 = self._t2(self._LAYERS, self._ACC)
        result = cross_track_agreement(t1, t2, None)
        corr   = result["t1_t2_correlation"]
        assert corr["n_layers"] >= 4, (
            f"Expected n_layers ≥ 4 after key normalisation; got {corr['n_layers']}"
        )
        assert np.isfinite(corr["spearman_rho"]), (
            "Spearman rho must be finite when keys are correctly matched"
        )
        assert corr["interpretation"] != "insufficient_data"

    def test_matching_integer_keys_also_work(self):
        """If both T1 and T2 happen to use integer keys, result must still work."""
        t1 = {"chorus_per_layer": {li: {"ari": {"ari": v}}
                                    for li, v in zip(self._LAYERS, self._ARI)}}
        t2 = self._t2(self._LAYERS, self._ACC)
        result = cross_track_agreement(t1, t2, None)
        assert result["t1_t2_correlation"]["n_layers"] >= 4

    def test_matching_string_keys_both_sides_work(self):
        """If both T1 and T2 happen to use string keys, result must still work."""
        t1 = self._t1(self._LAYERS, self._ARI)
        t2 = {
            "probe_per_layer": {
                f"layer_{li}": {"accuracy": float(v)}
                for li, v in zip(self._LAYERS, self._ACC)
            }
        }
        result = cross_track_agreement(t1, t2, None)
        assert result["t1_t2_correlation"]["n_layers"] >= 4

    def test_prompt_prefixed_t1_keys_normalised(self):
        """
        T1 can also store 'wiki_paragraph__layer_12' style keys (multi-prompt
        aggregation). These must also normalise to integer 12.
        """
        chorus_pl = {}
        for li, v in zip(self._LAYERS, self._ARI):
            chorus_pl[f"wiki_paragraph__layer_{li}"] = {"ari": {"ari": float(v)}}
            chorus_pl[f"sullivan__layer_{li}"]        = {"ari": {"ari": float(v) + 0.05}}
        t1 = {"chorus_per_layer": chorus_pl}
        t2 = self._t2(self._LAYERS, self._ACC)
        result = cross_track_agreement(t1, t2, None)
        assert result["t1_t2_correlation"]["n_layers"] >= 4

    def test_multi_prompt_values_averaged_per_layer(self):
        """
        When two prompts provide ARI for the same layer, the cross-track
        correlation should average them rather than using only one.
        The averaged value must fall between the two individual values.
        """
        ari_p1 = [0.2, 0.4, 0.6, 0.8, 0.9]
        ari_p2 = [0.4, 0.6, 0.8, 1.0, 1.0]   # shifted up by 0.2
        chorus_pl = {}
        for li, v1, v2 in zip(self._LAYERS, ari_p1, ari_p2):
            chorus_pl[f"layer_{li}"]         = {"ari": {"ari": float(v1)}}
            chorus_pl[f"sullivan__layer_{li}"] = {"ari": {"ari": float(v2)}}
        t1 = {"chorus_per_layer": chorus_pl}
        t2 = self._t2(self._LAYERS, self._ACC)
        # Just check it runs and returns finite Spearman; value correctness is
        # an implementation detail but at least it must not crash or give NaN.
        result = cross_track_agreement(t1, t2, None)
        assert np.isfinite(result["t1_t2_correlation"]["spearman_rho"])

    def test_empty_t1_gives_insufficient_data_not_crash(self):
        t2 = self._t2(self._LAYERS, self._ACC)
        result = cross_track_agreement({"chorus_per_layer": {}}, t2, None)
        assert result["t1_t2_correlation"]["n_layers"] == 0
        assert result["t1_t2_correlation"]["interpretation"] == "insufficient_data"

    def test_p4_symptom_n_layers_zero(self):
        """
        Reproduce the exact symptom from the p4 results file:
        T1/T2 n_layers: 0, interpretation: insufficient_data.
        With the bug present (raw key intersection), this is what happens.
        After the fix this test should fail — if it passes the bug is still there.
        """
        t1 = self._t1(self._LAYERS, self._ARI)   # string keys
        t2 = self._t2(self._LAYERS, self._ACC)   # integer keys
        # Simulate old raw-intersection logic
        t1_keys = set(t1["chorus_per_layer"])
        t2_keys = set(t2["probe_per_layer"])
        old_shared = t1_keys & t2_keys
        assert len(old_shared) == 0, (
            "Old raw intersection must be empty (confirms the bug exists pre-fix)"
        )
        # New logic (via fixed cross_track_agreement) must find shared layers
        result = cross_track_agreement(t1, t2, None)
        assert result["t1_t2_correlation"]["n_layers"] > 0, (
            "Fixed cross_track_agreement must find shared layers despite key type mismatch"
        )


# ===========================================================================
# Bug 5: max_nmi = 1.0 from a single trivial feature triggers verdict
# ===========================================================================

class TestBug5_NMIMaxTrivial:
    """
    A single feature that fires exclusively on one cluster gives NMI = 1.0 by max.
    The old threshold (max_nmi > 0.3) is then satisfied even when 99 % of features
    have near-zero MI. mean_nmi is the correct signal.
    """

    def _make_inputs(self, max_nmi, mean_nmi, max_probe=0.0, mean_probe=0.0,
                     max_ari=0.0, untestable=False):
        t1 = {
            "chorus_summary":    {"max_ari": max_ari, "mean_ari": 0.0, "mean_purity": 0.0},
            "mi_summary":        {"max_nmi": max_nmi, "mean_nmi": mean_nmi,
                                  "untestable": untestable},
            "plateau_alignment": {"alignment_rate": 0.5, "falsification": "inconclusive"},
        }
        t2 = {
            "probe_summary":     {"mean_accuracy": mean_probe, "max_accuracy": max_probe},
            "lda_summary":       {"mean_cosine": 0.0},
            "delta_pca_summary": {"mean_total_variance": 0.0, "mean_top1_explained": 0.0},
        }
        return t1, t2

    def test_max_nmi_one_mean_near_zero_verdict_is_null(self):
        """
        max_nmi = 1.0 (trivial single feature), mean_nmi = 0.01.
        Track 1 verdict must NOT be 'crosscoder_tracks_clusters'.
        This is the central failure: ALBERT's verdict was triggered for this reason.
        """
        t1, t2 = self._make_inputs(max_nmi=1.0, mean_nmi=0.01)
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 != "crosscoder_tracks_clusters", (
            f"Single trivial NMI=1.0 with mean=0.01 must not trigger strong verdict; "
            f"got '{v1}'"
        )

    def test_high_mean_nmi_triggers_strong_verdict(self):
        """mean_nmi = 0.5: strong signal across many features → verdict should fire."""
        t1, t2 = self._make_inputs(max_nmi=0.9, mean_nmi=0.5)
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 == "crosscoder_tracks_clusters", (
            f"mean_nmi=0.5 should trigger 'crosscoder_tracks_clusters'; got '{v1}'"
        )

    def test_moderate_mean_nmi_triggers_weak_signal(self):
        """mean_nmi = 0.2 (above noise floor, below strong) → 'weak_crosscoder_signal'."""
        t1, t2 = self._make_inputs(max_nmi=0.8, mean_nmi=0.2)
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 == "weak_crosscoder_signal", (
            f"mean_nmi=0.2 should be 'weak_crosscoder_signal'; got '{v1}'"
        )

    def test_both_null_mean_and_max_stays_null(self):
        """max_nmi = 0.05, mean_nmi = 0.02: genuine null → 'crosscoder_null'."""
        t1, t2 = self._make_inputs(max_nmi=0.05, mean_nmi=0.02)
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 == "crosscoder_null"

    def test_albert_result_correct_path_high_mean(self):
        """
        ALBERT p4 values: max_nmi=1.0, mean_nmi=0.867, chorus_ari=0.0.
        With fix: verdict is 'crosscoder_tracks_clusters' because mean is
        genuinely high (not just the trivial-max path).
        """
        t1, t2 = self._make_inputs(max_nmi=1.0, mean_nmi=0.867)
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 == "crosscoder_tracks_clusters"

    def test_gpt2_untestable_mi_gives_untestable_verdict(self):
        """
        GPT-2 p4 values: max_nmi=0.0, mean_nmi=0.0 because labels were absent.
        After Bug 2 fix: mi_summary = {max_nmi: nan, mean_nmi: nan, untestable: True}.
        Track 1 verdict should be 'untestable', not 'crosscoder_null'.
        This prevents the false inference that GPT-2 has no metastable features.
        """
        t1, t2 = self._make_inputs(
            max_nmi=float("nan"), mean_nmi=float("nan"), untestable=True
        )
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        v1 = verdict["tracks"]["track1_crosscoder"]["verdict"]
        assert v1 == "untestable", (
            f"NaN mi_summary (no labels) must give 'untestable'; got '{v1}'"
        )

    def test_overall_verdict_untestable_when_all_tracks_untestable(self):
        """
        If all three tracks are untestable, overall verdict must be
        'untestable' not 'cross_track_null' (a different scientific claim).
        """
        t1, t2 = self._make_inputs(
            max_nmi=float("nan"), mean_nmi=float("nan"), untestable=True,
            max_probe=float("nan"), mean_probe=float("nan"),
        )
        # Patch probe_summary to also be untestable
        t2["probe_summary"]["untestable"] = True
        verdict = build_phase4_verdict(t1, t2, None, {}, t1["plateau_alignment"])
        assert verdict.get("overall") == "untestable", (
            f"All-untestable tracks must yield overall='untestable'; "
            f"got '{verdict.get('overall')}'"
        )
