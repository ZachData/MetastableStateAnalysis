"""
constants.py — Tunable thresholds and weights for Phase 5.

Centralized so that the CLI can override any of them without touching
module internals. All defaults match the ranges specified in the plan.
"""

# ---------------------------------------------------------------------------
# Candidate gates (hard filters — a trajectory that fails any gate is dropped
# from the candidate pool before scoring)
# ---------------------------------------------------------------------------

MIN_LIFESPAN                = 6     # plan A: at least 6 layers continuous
MIN_SIZE                    = 4     # plan A.5: cluster of at least 4 tokens
MIN_SIZE_FRACTION_OF_ALIVE  = 0.75  # tolerate brief dips — size>=MIN at >= this
                                    # fraction of alive layers
REJECT_PROMPTS              = ("repeated_tokens",)

# ---------------------------------------------------------------------------
# Semantic-tag rule (not in plan; needs a concrete threshold)
# ---------------------------------------------------------------------------
# A cluster is "semantic at layer L" if the within-cluster mutual-NN pairs at
# L are predominantly semantic (share-cluster) rather than artifact. A
# trajectory earns the semantic bonus when it is semantic at >= this fraction
# of alive layers.
SEMANTIC_PAIR_MIN_COUNT     = 1     # need at least one mutual-NN pair in cluster
SEMANTIC_PAIR_FRACTION      = 0.6   # fraction of pairs tagged "semantic"
SEMANTIC_LAYER_FRACTION     = 0.5   # fraction of alive layers meeting the above

# ---------------------------------------------------------------------------
# Preferred-prompt bonus
# ---------------------------------------------------------------------------
# Plan B.4 says prefer sullivan_ballou or paper_excerpt for ALBERT-xlarge.
# Implemented as an additive score bonus, not a gate — cluster from another
# prompt can still rank high if everything else is strong.
PREFERRED_PROMPTS           = ("sullivan_ballou", "paper_excerpt")

# ---------------------------------------------------------------------------
# Sibling-availability rule
# ---------------------------------------------------------------------------
# The merge-partner trajectory (or nearest non-merged neighbour) must be
# alive for at least this many layers to enable contrast analysis.
MIN_SIBLING_LIFESPAN        = 6

# ---------------------------------------------------------------------------
# Score weights (plan §Cluster selection, criteria 1-6)
# ---------------------------------------------------------------------------
# Linear combination of per-criterion sub-scores. Each sub-score is already
# normalized to roughly [0, 1] inside select_cluster._score_trajectory.
SCORE_WEIGHTS = {
    "lifespan":         2.0,   # criterion 1
    "merge":            2.0,   # criterion 2 — trajectory ends in / contains merge
    "semantic":         2.0,   # criterion 3 — P1-4 semantic
    "preferred_prompt": 1.0,   # criterion 4
    "size":             1.0,   # criterion 5
    "sibling":          1.0,   # criterion 6
}

# Normalisation caps for the lifespan and size sub-scores. A trajectory
# with lifespan >= LIFESPAN_FULL_SCORE gets the full lifespan sub-score;
# anything shorter is scaled linearly.
LIFESPAN_FULL_SCORE         = 18    # ~half of ALBERT-xlarge 48-iter depth
SIZE_FULL_SCORE             = 10    # median cluster size in practice

# ---------------------------------------------------------------------------
# V-projector usage (Groups B, D)
# ---------------------------------------------------------------------------
# Cap the number of V eigenvectors used to build the attractive/repulsive
# subspaces. None means "use all"; smaller numbers give a tighter subspace
# dominated by the largest-magnitude eigenvalues. 64 is a compromise between
# subspace coverage and numerical noise in the tail.
V_PROJECTOR_K_TOP           = 64
