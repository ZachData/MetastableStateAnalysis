"""
p4_mstate_features — Phase 4: Identifying Metastable Features
==============================================================

Core question
-------------
Phase 3 confirmed that crosscoder features split into short-lived and
long-lived populations, but their decoder directions are geometrically
random with respect to V's eigensubspaces (SNR 0.18×).  Phase 4 asks:
do crosscoder features track metastable cluster structure through their
*activation patterns*, even though their decoder directions don't align?

Three parallel tracks, each independently falsifiable
------------------------------------------------------
Track 1 — Crosscoder activation patterns  (activation_trajectories, chorus)
    Do features fire preferentially on tokens belonging to one HDBSCAN
    cluster?  Do co-active feature cliques (the chorus hypothesis) track
    cluster identity even when individual features don't?

Track 2 — Direct geometric methods  (geometric)
    Does cluster structure live in a linearly separable subspace of the
    residual stream at plateau layers?  LDA stability, PCA on layer-to-layer
    deltas, and supervised linear probes — no learned dictionary required.

Track 3 — Low-rank autoencoder  (low_rank_ae)
    A bottleneck AE with rank = cluster count, no sparsity pressure.
    Tests whether sparsity was the confound preventing decoder directions
    from aligning with V in Phase 3.

Cross-track comparison and verdict  (analysis)
    Correlates per-layer signals across tracks, assembles the structured
    verdict consumed by Phases 5 and 6.

Module map
----------
activation_trajectories.py  — ActivationTrajectory, plateau detection, MI
chorus.py                   — co-activation matrix, clique extraction, ARI
geometric.py                — LDA, PCA on deltas, linear probes, V-alignment
low_rank_ae.py              — LowRankAE model, training, bottleneck diagnostics
analysis.py                 — cross-track comparison, verdict, I/O helpers
run.py                      — CLI entry point  (python -m p4_mstate_features.run)

Bug fixes applied (see tests/test_phase4_bugs.py for regression coverage)
--------------------------------------------------------------------------
B1  detect_feature_plateaus   — added min_peak_activation guard; dead/noise-floor
                                 features no longer produce full-window plateaus.
B2  probe_accuracy_trajectory  — returns NaN (not 0.0) when labels are absent.
    _aggregate_mi_summary      — new helper; returns NaN + untestable=True flag
                                 instead of 0.0 when Phase 1 labels are missing.
B3  compute_coactivation       — denominator changed from T (total tokens) to
                                 per-pair union count (Jaccard similarity); sparse
                                 co-active features now exceed the 0.3 threshold.
B4  cross_track_agreement      — T1 string keys ("layer_6") and T2 integer keys
                                 (6) are normalised to int before intersection;
                                 multi-prompt T1 values are averaged per layer.
B5  build_phase4_verdict T1    — verdict now driven by mean_nmi, not max_nmi;
                                 NaN mi_summary sets verdict to "untestable".

Downstream consumers
--------------------
Phase 5 (cluster identity + merge characterisation):
    LDA directions and cluster identity feature sets exported as .npz.
Phase 6 (tuned lens backward tracing):
    Cluster centroids saved per (prompt, layer, cluster_id) in centroids.npz.
"""

# ---------------------------------------------------------------------------
# Track 1 — crosscoder activation patterns
# ---------------------------------------------------------------------------

from .activation_trajectories import (
    # Data structure
    ActivationTrajectory,

    # Extraction
    extract_activation_trajectories,

    # Plateau detection  (B1: min_peak_activation parameter added)
    detect_feature_plateaus,

    # Cluster correspondence
    feature_cluster_mi,
    _aggregate_mi_summary,          # B2: NaN-safe summary helper; used by run_track1
    plateau_alignment,
    merge_feature_dynamics,
)

from .chorus import (
    compute_coactivation,           # B3: Jaccard denominator
    extract_cliques,
    analyze_chorus_at_layer,
    sweep_thresholds,
)

# ---------------------------------------------------------------------------
# Track 2 — direct geometric methods
# ---------------------------------------------------------------------------

from .geometric import (
    # Core analyses
    lda_directions,
    lda_stability_across_layers,
    pca_on_deltas,
    probe_accuracy_trajectory,      # B2: NaN summary when labels absent
    probe_v_alignment,

    # Utilities consumed by run_track2 and Phase 5
    train_linear_probe,
    extract_per_layer_activations,
    build_labels_per_layer,
)

# ---------------------------------------------------------------------------
# Track 3 — low-rank autoencoder
# ---------------------------------------------------------------------------

from .low_rank_ae import (
    # Configuration
    LowRankAEConfig,
    LRAETrainingConfig,

    # Model
    LowRankAE,

    # Training and persistence
    train_low_rank_ae,
    load_low_rank_ae,

    # Diagnostics
    bottleneck_v_alignment,
    compare_reconstruction,

    # Data adapter
    ActivationBufferAdapter,
)

# ---------------------------------------------------------------------------
# Cross-track comparison and I/O
# ---------------------------------------------------------------------------

from .analysis import (
    # Cross-track comparison  (B4: key normalisation)
    cross_track_agreement,

    # Verdict assembly  (B5: mean_nmi threshold; untestable state)
    build_phase4_verdict,

    # Per-track save helpers (called immediately after each track in run.py)
    save_track1_outputs,
    save_track2_outputs,
    save_track3_outputs,

    # Full-pipeline save wrapper (backward-compatible; calls per-track helpers)
    save_phase4_outputs,

    # LLM-friendly summary
    write_llm_summary,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Track 1 — trajectories
    "ActivationTrajectory",
    "extract_activation_trajectories",
    "detect_feature_plateaus",
    "feature_cluster_mi",
    "_aggregate_mi_summary",
    "plateau_alignment",
    "merge_feature_dynamics",
    # Track 1 — chorus
    "compute_coactivation",
    "extract_cliques",
    "analyze_chorus_at_layer",
    "sweep_thresholds",
    # Track 2
    "lda_directions",
    "lda_stability_across_layers",
    "pca_on_deltas",
    "probe_accuracy_trajectory",
    "probe_v_alignment",
    "train_linear_probe",
    "extract_per_layer_activations",
    "build_labels_per_layer",
    # Track 3
    "LowRankAEConfig",
    "LRAETrainingConfig",
    "LowRankAE",
    "train_low_rank_ae",
    "load_low_rank_ae",
    "bottleneck_v_alignment",
    "compare_reconstruction",
    "ActivationBufferAdapter",
    # Cross-track
    "cross_track_agreement",
    "build_phase4_verdict",
    "save_track1_outputs",
    "save_track2_outputs",
    "save_track3_outputs",
    "save_phase4_outputs",
    "write_llm_summary",
]