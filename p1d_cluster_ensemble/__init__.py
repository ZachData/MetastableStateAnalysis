"""
p1d_cluster_ensemble — Phase 1d: tuning the other methods, and what a
conglomeration of them says that one categorical label cannot.

An auxiliary re-analysis phase. Every cluster-conditioned result in this
project rests on one clusterer at one untuned setting —
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")` — and Phase 1
persists three more partitions of the same tokens whose agreement with it
`p1_visualization/cluster_methods.py` already measures. What has never
happened is tuning: that agreement statistic compares four sets of
library defaults, not four methods.

Phase 1d tunes seven families against criteria that do not presuppose
HDBSCAN's answer (subsample stability, calibrated against a matched
shuffled-dimension null), builds a weighted consensus from the tuned
partitions, and exports a per-particle CONTINUOUS annotation —
confidence, with a null-calibrated core/halo/contested trichotomy — in
place of the categorical clustered/unclustered bit.

Depends on : Phase 1 artifacts (activations.npz required,
             hdbscan_labels.json for P-C2/P-C3, geometry.json for
             identity), core/nulls.py, core/particles.py.
Depends NOT on Phase 2, and needs no model weights and no forward pass.
Runnable against any existing Phase 1 run directory.

Sub-experiments:
    A  selection.py   tune every family per layer                    [R]
    B  ensemble.py    consensus + calibrated per-particle confidence [R]
    C  comparison.py  the shipped partition and its refusals   P-C2, P-C3
    D  comparison.py  persistence prediction                        P-C4
    E  p1d_io.py      particle-table export                          [R]

Predictions P-C1..P-C4 were registered in PREDICTIONS.md before this code
existed. P-C4 is the phase's own falsification: if graded confidence does
not out-predict the binary flag, the conglomeration is presentation, and
that is the sentence the status doc will carry.
"""

from .constants import (
    DISTANCE_THRESHOLDS, K_VALUES, SHIPPED_HDBSCAN_PARAMS,
    SUBSTANTIAL_CLUSTER_SIZE,
)
from .methods import (
    FAMILIES, CAN_REFUSE, STOCHASTIC, LayerData, affinity_matrix,
    available_families, fit, hdbscan_backend, modularity, mutual_knn_graph,
    param_grid, spherical_kmeans,
)
from .selection import (
    NULL_ALPHA, Candidate, apply_gate, calibrate, null_distributions,
    partition_summary, select_all_families, select_family, selected_labels,
    selection_weights, separation_score, subsample_stability, sweep_family,
)
from .ensemble import (
    build, co_association, confidence, confidence_thresholds,
    consensus_partition, consensus_recall, consensus_strength,
    noise_as_singletons, refusal_fraction, trichotomy,
)
from .comparison import (
    adjudicate_p_c1, adjudicate_p_c2, adjudicate_p_c3, adjudicate_p_c4,
    auc, concordance_matrix, delta_auc_report, noise_rescue,
    persistence_target, shipped_comparison,
)
from .p1d_io import (
    P1D_SCHEMA_VERSION, build_particle_table, layer_activations, load_run,
    run_identity, save_p1d,
)

__all__ = [
    "DISTANCE_THRESHOLDS", "K_VALUES", "SHIPPED_HDBSCAN_PARAMS",
    "SUBSTANTIAL_CLUSTER_SIZE",
    "FAMILIES", "CAN_REFUSE", "STOCHASTIC", "LayerData", "affinity_matrix",
    "available_families", "fit", "hdbscan_backend", "modularity",
    "mutual_knn_graph", "param_grid", "spherical_kmeans",
    "NULL_ALPHA", "Candidate", "apply_gate", "calibrate",
    "null_distributions", "partition_summary", "select_all_families",
    "select_family", "selected_labels", "selection_weights",
    "separation_score", "subsample_stability", "sweep_family",
    "build", "co_association", "confidence", "confidence_thresholds",
    "consensus_partition", "consensus_recall", "consensus_strength",
    "noise_as_singletons", "refusal_fraction", "trichotomy",
    "adjudicate_p_c1", "adjudicate_p_c2", "adjudicate_p_c3",
    "adjudicate_p_c4", "auc", "concordance_matrix", "delta_auc_report",
    "noise_rescue", "persistence_target", "shipped_comparison",
    "P1D_SCHEMA_VERSION", "build_particle_table", "layer_activations",
    "load_run", "run_identity", "save_p1d",
]
