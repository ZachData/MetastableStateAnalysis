"""
test_phase1_clustering.py — Pure-computation tests for spectral, clustering,
and cluster-tracking modules.

Modules under test (flat imports from /mnt/project/ via pytest pythonpath):
  spectral.py        : spectral_eigengap_k
  clustering.py      : cluster_count_sweep
  cluster_tracking.py: track_clusters

No model loading.  All inputs are pre-computed synthetic fixtures from conftest.py.
"""

import numpy as np
import pytest

from p1_mstate_tracking.spectral         import spectral_eigengap_k
from p1_mstate_tracking.clustering       import cluster_count_sweep
from p1_mstate_tracking.cluster_tracking import track_clusters

from tests.conftest import N_LAYERS, N_TOKENS, D
from core.config import DISTANCE_THRESHOLDS

# Pre-compute mid-threshold once (matches the value used inside cluster_count_sweep).
_THRESHOLD_LIST = list(DISTANCE_THRESHOLDS)
_MID_THRESH = float(_THRESHOLD_LIST[len(_THRESHOLD_LIST) // 2])

# ============================================================================
# spectral_eigengap_k
# ============================================================================

class TestSpectralEigengap:
    """
    spectral_eigengap_k clips negative Gram entries to 0, builds the
    normalised Laplacian, and returns k from the largest eigengap.

    Antipodal Gram
    --------------
    After clipping, the between-cluster block becomes all-zeros → the graph
    is effectively two disconnected components.  λ₂ ≈ 0, and the eigengap
    is large after position 1 (between λ₂ and λ₃).  k_eigengap = 2.

    Collapsed Gram
    --------------
    All inner products ≈ +1 → the graph is near-complete K_n.  Eigenvalues
    are {0} ∪ {n/(n−1)}^{n−1}: the only large gap is at position 0 (trivial
    zero mode → first non-zero eigenvalue).  k_eigengap = 1.
    """

    def test_antipodal_gram_k_equals_2(self, antipodal_gram):
        result = spectral_eigengap_k(antipodal_gram)
        assert result["k_eigengap"] == 2, (
            f"Antipodal clusters should give k_eigengap=2, got {result['k_eigengap']}"
        )

    def test_collapsed_gram_k_equals_1(self, collapsed_gram):
        result = spectral_eigengap_k(collapsed_gram)
        assert result["k_eigengap"] == 1, (
            f"Collapsed cluster should give k_eigengap=1, got {result['k_eigengap']}"
        )

    def test_returns_required_keys(self, antipodal_gram):
        result = spectral_eigengap_k(antipodal_gram)
        for key in ("k_eigengap", "eigenvalues", "eigengaps"):
            assert key in result, f"Missing key '{key}' in spectral_eigengap_k output"

    def test_eigenvalues_non_negative(self, antipodal_gram):
        """Laplacian eigenvalues are always ≥ 0."""
        result = spectral_eigengap_k(antipodal_gram)
        eigs   = np.array(result["eigenvalues"])
        assert (eigs >= -1e-10).all(), (
            f"Some Laplacian eigenvalues are negative: {eigs[eigs < -1e-10]}"
        )

    def test_eigengaps_non_negative(self, antipodal_gram):
        """Consecutive differences of a non-decreasing sequence are ≥ 0."""
        result = spectral_eigengap_k(antipodal_gram)
        gaps   = np.array(result["eigengaps"])
        assert (gaps >= -1e-10).all()

    def test_k_eigengap_in_valid_range(self, antipodal_gram, collapsed_gram):
        """k must be a positive integer not exceeding SPECTRAL_MAX_K."""
        from core.config import SPECTRAL_MAX_K
        for gram, name in [(antipodal_gram, "antipodal"), (collapsed_gram, "collapsed")]:
            k = spectral_eigengap_k(gram)["k_eigengap"]
            assert 1 <= k <= SPECTRAL_MAX_K, (
                f"{name}: k_eigengap={k} outside [1, {SPECTRAL_MAX_K}]"
            )

    def test_fiedler_vec_returned_when_requested(self, antipodal_gram):
        """With return_fiedler_vec=True the result must contain 'fiedler_vec'."""
        result = spectral_eigengap_k(antipodal_gram, return_fiedler_vec=True)
        assert "fiedler_vec" in result
        assert result["fiedler_vec"] is not None
        assert len(result["fiedler_vec"]) == N_TOKENS

    def test_fiedler_vec_bipartitions_antipodal(self, antipodal_gram):
        """
        For two perfectly separated clusters, the Fiedler vector (second
        Laplacian eigenvector) partitions tokens by sign: the first N/2
        tokens should have the same sign, the second N/2 the opposite sign.

        This is robust as long as noise is small enough that the block
        structure dominates.
        """
        result   = spectral_eigengap_k(antipodal_gram, return_fiedler_vec=True)
        fv       = np.array(result["fiedler_vec"])
        half     = N_TOKENS // 2
        sign_a   = np.sign(fv[:half])
        sign_b   = np.sign(fv[half:])
        # All tokens in each half should have a consistent sign, opposite across halves.
        assert np.all(sign_a == sign_a[0]), "First cluster tokens have mixed Fiedler signs"
        assert np.all(sign_b == sign_b[0]), "Second cluster tokens have mixed Fiedler signs"
        assert sign_a[0] != sign_b[0],     "Both clusters have the same Fiedler sign"


# ============================================================================
# cluster_count_sweep
# ============================================================================

class TestClusterCountSweep:
    """
    cluster_count_sweep runs agglomerative clustering at 12 thresholds plus
    KMeans silhouette sweep.

    Antipodal  →  KMeans best_k = 2  (silhouette peak at k=2)
    Collapsed  →  agglomerative count = 1 at mid-threshold
                  (all cosine distances ≪ 0.05, so even the smallest
                  threshold merges every token into one cluster)
    """

    def test_antipodal_kmeans_best_k_is_2(self, antipodal_normed):
        """
        With tight antipodal clusters (cosine distance between poles ≈ 2,
        within-cluster ≈ 0), the silhouette score for k=2 dominates.
        """
        result = cluster_count_sweep(antipodal_normed)
        best_k = result["kmeans"]["best_k"]
        assert best_k == 2, (
            f"Antipodal activations should yield KMeans best_k=2, got {best_k}"
        )

    def test_collapsed_agglomerative_count_is_1(self, collapsed_normed):
        """
        Collapsed tokens have cosine distances ~(noise/1)² ≈ 1e-6, far below
        every threshold in DISTANCE_THRESHOLDS (min 0.05).
        Agglomerative clustering at the mid-threshold must return 1 cluster.
        """
        result = cluster_count_sweep(collapsed_normed)
        count  = result["agglomerative"][_MID_THRESH]
        assert count == 1, (
            f"Collapsed activations should give 1 agglomerative cluster at "
            f"threshold={_MID_THRESH:.3f}, got {count}"
        )

    def test_antipodal_agglomerative_count_is_2_at_mid_threshold(
        self, antipodal_normed
    ):
        """
        Between-cluster cosine distance ≈ 2 ≫ mid-threshold ≈ 0.35.
        The two clusters must not merge until a much larger threshold.
        """
        result = cluster_count_sweep(antipodal_normed)
        count  = result["agglomerative"][_MID_THRESH]
        assert count == 2, (
            f"Antipodal activations should give 2 agglomerative clusters at "
            f"threshold={_MID_THRESH:.3f}, got {count}"
        )

    def test_returns_required_keys(self, antipodal_normed):
        result = cluster_count_sweep(antipodal_normed)
        assert "agglomerative" in result
        assert "kmeans"        in result
        assert "best_k"        in result["kmeans"]
        assert "best_silhouette" in result["kmeans"]
        assert "labels"        in result["kmeans"]

    def test_agglomerative_has_mid_labels(self, antipodal_normed):
        """mid_labels must be a list of length N_TOKENS."""
        result      = cluster_count_sweep(antipodal_normed)
        mid_labels  = result["agglomerative"].get("mid_labels")
        assert mid_labels is not None
        assert len(mid_labels) == N_TOKENS

    def test_kmeans_labels_length(self, antipodal_normed):
        result = cluster_count_sweep(antipodal_normed)
        assert len(result["kmeans"]["labels"]) == N_TOKENS

    def test_silhouette_antipodal_near_plus_one(self, antipodal_normed):
        """
        Tight antipodal clusters are nearly perfectly separated in cosine space.
        The best silhouette score should be high (> 0.8).
        """
        result = cluster_count_sweep(antipodal_normed)
        sil    = result["kmeans"]["best_silhouette"]
        assert sil > 0.8, (
            f"Antipodal clusters should yield silhouette > 0.8, got {sil:.4f}"
        )


# ============================================================================
# track_clusters
# ============================================================================

class TestTrackClusters:
    """
    track_clusters matches HDBSCAN clusters across adjacent layers via
    Hungarian-assigned Jaccard overlap, then records births, deaths, and merges.

    Stable assignments (same labels at every layer)
    ------------------------------------------------
    Every cluster matches itself at 100 % Jaccard overlap → no births, no
    deaths, no merges.  summary["total_merges"] == 0.

    One merge event
    ---------------
    Layers 0–2: two clusters (tokens 0-19 in cluster 0, 20-39 in cluster 1).
    Layers 3–5: single cluster (all 40 tokens in cluster 0).

    At the 2→3 transition:
      Jaccard(prev-0, curr-0) = |{0..19}∩{0..39}| / |{0..19}∪{0..39}| = 20/40 = 0.5
      Jaccard(prev-1, curr-0) = |{20..39}∩{0..39}| / |{20..39}∪{0..39}| = 20/40 = 0.5
    Hungarian matches prev-0 → curr-0.  Merge-detection finds prev-1 also
    overlaps curr-0, which is already matched → records one merge event.

    summary["total_merges"] == 1.
    """

    def test_stable_assignments_zero_merges(self, stable_tracking_results):
        tracking = track_clusters(stable_tracking_results)
        total_merges = tracking["summary"]["total_merges"]
        assert total_merges == 0, (
            f"Stable cluster assignments should produce 0 merges, got {total_merges}"
        )

    def test_stable_assignments_zero_births(self, stable_tracking_results):
        tracking = track_clusters(stable_tracking_results)
        assert tracking["summary"]["total_births"] == 0

    def test_stable_assignments_zero_deaths(self, stable_tracking_results):
        tracking = track_clusters(stable_tracking_results)
        assert tracking["summary"]["total_deaths"] == 0

    def test_stable_assignments_correct_event_count(
        self, stable_tracking_results
    ):
        """One event per layer transition → N_LAYERS − 1 events total."""
        tracking = track_clusters(stable_tracking_results)
        assert len(tracking["events"]) == N_LAYERS - 1

    def test_one_merge_event_recorded(self, one_merge_tracking_results):
        """
        The single 2-cluster → 1-cluster transition produces exactly one
        merge in summary["total_merges"].
        """
        tracking     = track_clusters(one_merge_tracking_results)
        total_merges = tracking["summary"]["total_merges"]
        assert total_merges == 1, (
            f"Expected exactly 1 merge event, got {total_merges}"
        )

    def test_one_merge_event_in_correct_transition(
        self, one_merge_tracking_results
    ):
        """The merge must appear at the layer-2 → layer-3 transition."""
        tracking = track_clusters(one_merge_tracking_results)
        merge_events = [ev for ev in tracking["events"] if ev["n_merges"] > 0]
        assert len(merge_events) == 1
        ev = merge_events[0]
        assert ev["layer_from"] == 2
        assert ev["layer_to"]   == 3

    def test_returns_required_summary_keys(self, stable_tracking_results):
        tracking = track_clusters(stable_tracking_results)
        for key in ("total_births", "total_deaths", "total_merges", "max_alive"):
            assert key in tracking["summary"], f"Missing summary key '{key}'"

    def test_max_alive_is_2_for_stable(self, stable_tracking_results):
        """With two clusters across all layers, max_alive = 2."""
        tracking = track_clusters(stable_tracking_results)
        assert tracking["summary"]["max_alive"] == 2

    def test_trajectories_list_present(self, stable_tracking_results):
        """track_clusters must return a 'trajectories' key."""
        tracking = track_clusters(stable_tracking_results)
        assert "trajectories" in tracking
        assert isinstance(tracking["trajectories"], list)

    def test_stable_trajectories_span_all_layers(self, stable_tracking_results):
        """
        With perfectly stable assignments, each trajectory starts at layer 0
        and ends at layer N_LAYERS − 1 (lifespan = N_LAYERS).
        """
        tracking = track_clusters(stable_tracking_results)
        for traj in tracking["trajectories"]:
            assert traj["lifespan"] == N_LAYERS, (
                f"Trajectory {traj['id']} has lifespan {traj['lifespan']}, "
                f"expected {N_LAYERS}"
            )
