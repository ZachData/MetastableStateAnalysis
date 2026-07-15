"""
p5b_manifold — Phase 5b: Metastable States as Activation Manifold Control Points

Public API (imported by run_5b.py and tests):

  manifold_fit
    pca_reduce(centroids, k) → (scores, basis, evr)
    arc_length_params(pts, periodic) → u
    fit_activation_manifold(centroids_pca, u, periodic) → mh dict
    eval_manifold(mh, t) → pts
    fit_behavior_manifold(distributions, u, periodic) → my dict
    eval_behavior_manifold(my, t) → p
    load_plateau_centroids(path, plateau_layers, cluster_ids) → (centroids, mask)
    compute_fit_summary(mh, my, evr) → dict

  isometry_test
    hellinger_distance(p, q) → float
    geodesic_distance_manifold(mh, u_i, u_j) → float
    geodesic_distance_behavior(my, u_i, u_j) → float
    pairwise_distances(mh, my, u, raw_c) → dict
    isometry_score(d_m, d_b, d_l) → dict
    mds_embed(dist_matrix) → coords
    run_isometry_test(mh, my, u, raw_c) → dict

  merge_teleportation
    teleportation_score(p_before, p_at, p_after) → dict
    compare_merge_plateau(merge_scores, plateau_scores) → dict
    run_merge_teleportation(logit_dists, merge_layers, plateau_layers) → dict

  subspace_isometry  (defined in merge_teleportation.py)
    project_centroids(centroids, U) → projected
    subspace_isometry_score(centroids, U_S, U_A, d_behavior) → dict

  logit_cache
    validate_logit_output(logits, expected_vocab) → None
    logits_to_distribution(logits) → p
    extract_layer_logits(model, tokenizer, prompt, layer_idxs) → dict
    save_logit_cache(distributions, path) → None
    load_logit_cache(path) → dict

  report
    write_report(out_dir, results, model, prompt) → Path

Reference: Wurgaft et al. (2026), arXiv:2605.05115
Code:      https://github.com/goodfire-ai/causalab/tree/manifold_steering
"""

from .manifold_fit import (
    pca_reduce,
    arc_length_params,
    fit_activation_manifold,
    eval_manifold,
    fit_behavior_manifold,
    eval_behavior_manifold,
    load_plateau_centroids,
    compute_fit_summary,
)
from .isometry_test import (
    hellinger_distance,
    geodesic_distance_manifold,
    geodesic_distance_behavior,
    pairwise_distances,
    isometry_score,
    mds_embed,
    run_isometry_test,
)
from .merge_teleportation_subspace import (
    teleportation_score,
    compare_merge_plateau,
    run_merge_teleportation,
)
from .subspace_isometry import (
    project_centroids,
    subspace_isometry_score,
)
from .logit_cache import (
    validate_logit_output,
    logits_to_distribution,
    save_logit_cache,
    load_logit_cache,
)
from .p5b_report import write_report

__all__ = [
    "pca_reduce", "arc_length_params",
    "fit_activation_manifold", "eval_manifold",
    "fit_behavior_manifold", "eval_behavior_manifold",
    "load_plateau_centroids", "compute_fit_summary",
    "hellinger_distance",
    "geodesic_distance_manifold", "geodesic_distance_behavior",
    "pairwise_distances", "isometry_score", "mds_embed", "run_isometry_test",
    "teleportation_score", "compare_merge_plateau", "run_merge_teleportation",
    "project_centroids", "subspace_isometry_score",
    "validate_logit_output", "logits_to_distribution",
    "save_logit_cache", "load_logit_cache",
    "write_report",
]
