"""
phase1h — Hemispheric structure investigation.

Characterizes the k=2 Laplacian bipartition found in Phase 1: when it
exists as geometric structure vs. eigengap artifact, how its identity
tracks across layers, whether it nests HDBSCAN clusters, whether it
coexists with or contradicts the paper's cone-collapse regime, what
weight-level object determines its axis, and what token semantics it
carries.

Block roster (see README.md for detail):
  Block 0  bipartition_detect      regime classification per layer
  Block 1  hemisphere_tracking     identity matching, events, axis rotation
  Block 2  hemisphere_membership   per-token trajectories, HDBSCAN nesting
  Block 3  paper_alignment         cone-collapse LP test, regime
  Block 4  size_symmetry           asymmetry distribution, correlations
  Block 5  hemisphere_mechanism    axis alignment vs OV / PCA / embedding
  Block 6  hemisphere_semantics    attribute contingency and MI
  Block 7  hemisphere_ablation     causal projection (stretch; deferred)
"""
