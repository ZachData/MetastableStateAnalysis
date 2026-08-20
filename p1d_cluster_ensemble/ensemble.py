"""
p1d_cluster_ensemble/ensemble.py — the conglomeration.

Once each family has a tuned setting (selection.py), this module turns the
set of partitions into one object: a co-association matrix, a consensus
partition extracted from it, and — the part that is not just another
categorical label — a per-particle confidence that is continuous and
calibrated.

The claim structure, stated once
--------------------------------
A co-association matrix is **not** a ground truth. It is an aggregation of
biases: if five of seven families assume clusters are blobs, the consensus
will find blobs. What it does buy is that no *single* method's inductive
bias can be blamed for a structure that survives it, which is why
p1_visualization/cluster_methods.py already calls the unweighted version
"the strongest available evidence that the clusters are a property of the
geometry". Everything in this module is an aggregation statement, and the
only claims that get calibrated against a null are the per-particle ones
in `confidence_thresholds`.

Three things this adds over the version in p1_visualization
-----------------------------------------------------------
1. **Weights.** Families vote in proportion to the reproducibility of
   their selected setting. A family that cleared the floor at 0.3 and one
   that reproduces at 0.95 are not the same evidence.
2. **Abstention.** A family that failed the gate at a layer does not vote
   there at all (PREDICTIONS.md, Phase 1d constraint 3), and the number
   that did vote is carried alongside every consensus statistic — an
   agreement among two families is not the same measurement as an
   agreement among seven.
3. **Per-particle confidence, with a null-calibrated threshold.** The
   categorical HDBSCAN label answers "is this token in a cluster" with a
   bit. Confidence answers it with a number whose scale is set by what a
   structureless cloud with the same marginals produces.

Relationship to p1_visualization/cluster_methods.py
---------------------------------------------------
`co_association` there is this module's function at unit weights and no
abstention, and `consensus_strength` is the same statistic. They are not
imported because that module lives inside the visualization package,
whose `__init__` pulls in matplotlib and the whole figure pipeline, and
this phase must stay importable in a numpy/scipy/sklearn environment.
The duplication is real and is not left to a comment:
tests/test_phase1d_ensemble.py asserts the two implementations agree
whenever the visualization package can be imported, so a change to either
that breaks the correspondence fails a test rather than drifting quietly.
If this phase outlives its first run, both belong in core/.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import SUBSTANTIAL_CLUSTER_SIZE

NOISE_POLICIES = ("singleton", "exclude")


# ---------------------------------------------------------------------------
# Noise policy
# ---------------------------------------------------------------------------

def noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    """
    Replace every -1 with a distinct new cluster id, above the existing
    maximum so no refused token can collide with a real cluster.

    Same function, same semantics, as the visualization package's; see the
    module docstring for why it is here rather than imported.
    """
    labels = np.asarray(labels, dtype=np.int64)
    is_noise = labels == -1
    if not is_noise.any():
        return labels.copy()
    out = labels.copy()
    start = int(labels.max()) + 1 if (~is_noise).any() else 0
    out[is_noise] = np.arange(start, start + int(is_noise.sum()))
    return out


# ---------------------------------------------------------------------------
# Co-association
# ---------------------------------------------------------------------------

def co_association(
    labels_by_family: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    noise_policy: str = "exclude",
) -> Dict[str, object]:
    """
    Weighted co-association: entry (i, j) is the weight-share of voting
    families that place i and j in the same cluster.

    The denominator is per-pair, not global: under "exclude" a family that
    refused either token contributes to neither numerator nor denominator
    for that pair, so C is always "of the families with an opinion about
    this pair, what fraction agreed". A pair no family had an opinion
    about comes back 0 with `support` 0, and downstream code reads
    `support` rather than treating 0 as disagreement — those are different
    statements and collapsing them is how a refusal turns into evidence of
    separation.

    Parameters
    ----------
    labels_by_family : {family: (n,) int array}. -1 is a refusal.
    weights          : {family: float}; missing families default to 1.0.
                        Negative weights are refused rather than clipped —
                        a negative vote is not a concept this has.
    noise_policy     : "exclude" (default) or "singleton". Default differs
                        from the visualization module's on purpose: this
                        module's whole subject is what happens to refused
                        tokens, and turning each refusal into its own
                        cluster silently answers that question before it
                        is asked.

    Returns
    -------
    dict with keys C (n, n), support (n, n), families (list), weights
    (dict), n_families (int), noise_policy (str).
    """
    if noise_policy not in NOISE_POLICIES:
        raise ValueError(f"unknown noise policy {noise_policy!r}; use one of {NOISE_POLICIES}")
    families = [f for f in labels_by_family]
    if not families:
        return {"C": np.zeros((0, 0)), "support": np.zeros((0, 0)),
                "families": [], "weights": {}, "n_families": 0,
                "noise_policy": noise_policy}

    n = int(np.asarray(labels_by_family[families[0]]).size)
    weights = dict(weights or {})
    for f in families:
        w = float(weights.get(f, 1.0))
        if w < 0:
            raise ValueError(f"negative weight {w} for family {f!r}")
        weights[f] = w

    total = np.zeros((n, n), dtype=float)
    denom = np.zeros((n, n), dtype=float)
    used: List[str] = []

    for family in families:
        lab = np.asarray(labels_by_family[family])
        if lab.size != n:
            # A family whose array is a different length is describing a
            # different token set; dropping it is the only safe reading,
            # and it is dropped loudly by being absent from `families`.
            continue
        w = weights[family]
        if w <= 0:
            continue
        if noise_policy == "singleton":
            lab = noise_as_singletons(lab)
            valid = np.ones(n, dtype=bool)
        else:
            valid = lab >= 0
        same = (lab[:, None] == lab[None, :]).astype(float)
        pair_valid = (valid[:, None] & valid[None, :]).astype(float)
        total += w * same * pair_valid
        denom += w * pair_valid
        used.append(family)

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(denom > 0, total / np.maximum(denom, 1e-12), 0.0)
    np.fill_diagonal(C, 1.0)
    return {
        "C": C, "support": denom, "families": used,
        "weights": {f: weights[f] for f in used},
        "n_families": len(used), "noise_policy": noise_policy,
    }


def consensus_strength(C: np.ndarray) -> float:
    """
    Fraction of off-diagonal pairs the voting families are unanimous about
    (co-association exactly 0 or 1). Near 1 means the partition is
    method-independent; near 0.5 means cluster membership is largely a
    statement about which algorithm was run. This is P-C1's instrument.
    """
    n = C.shape[0]
    if n < 2:
        return float("nan")
    vals = C[np.triu_indices(n, k=1)]
    if vals.size == 0:
        return float("nan")
    return float((np.isclose(vals, 0.0) | np.isclose(vals, 1.0)).mean())


# ---------------------------------------------------------------------------
# Consensus partition
# ---------------------------------------------------------------------------

def consensus_partition(C: np.ndarray) -> Dict[str, object]:
    """
    The partition that best represents the co-association matrix.

    Average-linkage on (1 - C), cut at the height that minimizes the
    squared disagreement with C itself:

        J(P) = mean over i<j of (C_ij - 1[i ~_P j])^2

    which is the Mirkin/consensus objective specialized to a soft target.
    Every merge height in the dendrogram is a candidate cut, plus the
    all-singletons cut, so the number of clusters is *derived* rather than
    chosen — there is no k to pick, which matters because a consensus
    whose k came from the same family of assumptions the members made
    would not be a consensus.

    Falls back to the all-singletons partition when scipy is unavailable,
    and says so in `branch` rather than returning a partition whose
    provenance is invisible (standing rule 2).

    Returns dict: labels (n,), n_clusters, objective, cut_height,
    objective_curve [(height, J)], branch.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    if n == 0:
        return {"labels": np.zeros(0, dtype=np.int32), "n_clusters": 0,
                "objective": float("nan"), "cut_height": float("nan"),
                "objective_curve": [], "branch": "empty"}
    if n < 3:
        labels = np.zeros(n, dtype=np.int32)
        return {"labels": labels, "n_clusters": 1, "objective": _mirkin(C, labels),
                "cut_height": 0.0, "objective_curve": [], "branch": "n<3"}

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        labels = np.arange(n, dtype=np.int32)
        return {"labels": labels, "n_clusters": n, "objective": _mirkin(C, labels),
                "cut_height": 0.0, "objective_curve": [],
                "branch": "scipy_unavailable_singletons"}

    D = 1.0 - C
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    try:
        Z = linkage(squareform(D, checks=False), method="average")
    except (ValueError, RuntimeError):
        labels = np.arange(n, dtype=np.int32)
        return {"labels": labels, "n_clusters": n, "objective": _mirkin(C, labels),
                "cut_height": 0.0, "objective_curve": [],
                "branch": "linkage_failed_singletons"}

    heights = np.unique(np.concatenate([[0.0], Z[:, 2]]))
    curve: List[Tuple[float, float]] = []
    best = (np.inf, 0.0, np.arange(n, dtype=np.int32))
    for h in heights:
        labels = fcluster(Z, t=float(h), criterion="distance").astype(np.int32) - 1
        j = _mirkin(C, labels)
        curve.append((float(h), float(j)))
        if j < best[0]:
            best = (j, float(h), labels)

    objective, cut, labels = best
    return {
        "labels": labels,
        "n_clusters": int(np.unique(labels).size),
        "objective": float(objective),
        "cut_height": float(cut),
        "objective_curve": curve,
        "branch": "mirkin_cut",
    }


def _mirkin(C: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared disagreement between C and a partition's 0/1 pattern."""
    n = C.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    same = (labels[:, None] == labels[None, :]).astype(float)
    return float(np.mean((C[iu] - same[iu]) ** 2))


# ---------------------------------------------------------------------------
# Per-particle quantities — the graded annotation
# ---------------------------------------------------------------------------

def confidence(C: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    (n,) per-particle consensus confidence: mean co-association with its
    own consensus cluster, minus the best mean co-association with any
    other cluster.

    A silhouette in co-association space rather than in distance space,
    and the difference matters: the units are "fraction of methods that
    agree", so +0.8 means four fifths of the weighted vote puts this
    particle with its own group and almost none of it puts the particle
    anywhere else. Near 0 means the methods are split about where this
    particle goes — which is a statement the categorical label cannot
    make at all.

    A particle alone in its consensus cluster has no within-cluster
    support; its own term is 0, so its confidence is the negative of its
    best claim elsewhere. That is the right reading (nothing supports it
    here, something supports it there), and it is why the scale runs to
    -1 rather than starting at 0.
    """
    C = np.asarray(C, dtype=float)
    labels = np.asarray(labels)
    n = C.shape[0]
    out = np.full(n, np.nan)
    if n == 0:
        return out
    ids = np.unique(labels)
    masks = {cid: (labels == cid) for cid in ids}

    for i in range(n):
        own = labels[i]
        own_mask = masks[own].copy()
        own_mask[i] = False
        a = float(C[i, own_mask].mean()) if own_mask.any() else 0.0
        b = 0.0
        for cid in ids:
            if cid == own:
                continue
            m = masks[cid]
            if m.any():
                b = max(b, float(C[i, m].mean()))
        out[i] = a - b
    return out


def consensus_recall(
    labels_by_family: Dict[str, np.ndarray],
    consensus_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Per particle, what fraction of its consensus-cluster co-members each
    family also places with it.

    This is the disagreement structure the confidence number compresses:
    `min_recall` near 0 with `mean_recall` high says one family
    specifically dissents about this particle, which is a different
    situation from every family being half-right about it. Refused
    tokens score 0 recall for that family — a refusal is a failure to
    reproduce the consensus, not a missing value, and treating it as
    missing would make a method look better the more often it declined.

    Returns {mean_recall, min_recall, per_family: {family: (n,)}}.
    """
    consensus_labels = np.asarray(consensus_labels)
    n = int(consensus_labels.size)
    same_consensus = consensus_labels[:, None] == consensus_labels[None, :]
    np.fill_diagonal(same_consensus, False)
    denom = same_consensus.sum(axis=1)

    per_family: Dict[str, np.ndarray] = {}
    for family, raw in labels_by_family.items():
        lab = np.asarray(raw)
        if lab.size != n:
            continue
        same_family = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        np.fill_diagonal(same_family, False)
        hit = (same_consensus & same_family).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            # A singleton consensus cluster has nothing to recall; NaN
            # rather than 1.0, which would read as unanimous agreement
            # about a particle no method grouped with anything.
            per_family[family] = np.where(denom > 0, hit / np.maximum(denom, 1), np.nan)

    if not per_family:
        empty = np.full(n, np.nan)
        return {"mean_recall": empty, "min_recall": empty.copy(), "per_family": {}}

    stacked = np.vstack([per_family[f] for f in per_family])
    return {
        "mean_recall": np.nanmean(stacked, axis=0),
        "min_recall": np.nanmin(stacked, axis=0),
        "per_family": per_family,
    }


def refusal_fraction(
    labels_by_family: Dict[str, np.ndarray],
    substantial: int = SUBSTANTIAL_CLUSTER_SIZE,
) -> np.ndarray:
    """
    (n,) fraction of voting families that leave each particle outside
    substantial structure — either refused outright (-1, HDBSCAN's idiom)
    or placed in a cluster smaller than `substantial`, which is the same
    refusal in a family that cannot spell it.

    This is what makes "unclustered" comparable across families that have
    no noise label. Without it, every non-density method looks like it
    placed 100% of tokens in structure by construction.
    """
    families = list(labels_by_family)
    if not families:
        return np.zeros(0)
    n = int(np.asarray(labels_by_family[families[0]]).size)
    counts = np.zeros(n, dtype=float)
    voters = 0
    for family in families:
        lab = np.asarray(labels_by_family[family])
        if lab.size != n:
            continue
        voters += 1
        ids, sizes = np.unique(lab, return_counts=True)
        size_of = dict(zip(ids.tolist(), sizes.tolist()))
        counts += np.array([
            1.0 if (int(l) < 0 or size_of.get(int(l), 0) < substantial) else 0.0
            for l in lab
        ])
    return counts / voters if voters else np.full(n, np.nan)


# ---------------------------------------------------------------------------
# Calibration of the confidence scale
# ---------------------------------------------------------------------------

def confidence_thresholds(
    null_confidences: Sequence[np.ndarray],
    q_core: float = 95.0,
    q_contested: float = 50.0,
) -> Dict[str, float]:
    """
    Turn pooled per-particle confidences measured on matched-null draws
    into the two cut points the trichotomy uses.

    A confidence of 0.4 means nothing on its own — it depends on how many
    families voted, how many clusters they found, and how large the
    token set is. Against the same pipeline run on structureless data
    with the same marginals it means something specific: above the 95th
    null percentile, a structureless cloud produces this level of
    cross-method agreement about a particle 5% of the time.

    The percentiles are the placed part and are named as such; the
    *values* they produce are calibrated. Returns {core, contested,
    n_null_particles, q_core, q_contested}.
    """
    pooled = np.concatenate([np.asarray(v, dtype=float).ravel()
                             for v in null_confidences]) if null_confidences else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return {"core": float("nan"), "contested": float("nan"),
                "n_null_particles": 0, "q_core": q_core, "q_contested": q_contested}
    return {
        "core": float(np.percentile(pooled, q_core)),
        "contested": float(np.percentile(pooled, q_contested)),
        "n_null_particles": int(pooled.size),
        "q_core": float(q_core),
        "q_contested": float(q_contested),
    }


def trichotomy(conf: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """
    (n,) array of "core" / "halo" / "contested" from a confidence array
    and calibrated cut points.

    This is the population tag Phase 1d exports, and it is deliberately
    three-valued rather than two: "halo" is the population the binary
    clustered/unclustered split has nowhere to put — particles the
    methods mostly agree about but not at a level structureless data
    could not reach.

    An uncalibrated threshold set (NaN cut points, i.e. no null draws)
    returns every particle as "uncalibrated" rather than guessing.
    """
    conf = np.asarray(conf, dtype=float)
    core_t, cont_t = thresholds.get("core"), thresholds.get("contested")
    if core_t is None or cont_t is None or not (np.isfinite(core_t) and np.isfinite(cont_t)):
        return np.array(["uncalibrated"] * conf.size)
    out = np.where(conf >= core_t, "core",
                   np.where(conf <= cont_t, "contested", "halo"))
    return np.where(np.isfinite(conf), out, "uncalibrated")


def build(
    labels_by_family: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    noise_policy: str = "exclude",
) -> Dict[str, object]:
    """
    One layer's whole ensemble: co-association, consensus partition, and
    every per-particle annotation, in one dict.

    Everything a figure or a downstream phase needs is here; nothing is
    recomputed from a partial result, because the confidence numbers and
    the consensus partition have to come from the same C or they describe
    different objects.
    """
    coas = co_association(labels_by_family, weights=weights, noise_policy=noise_policy)
    C = coas["C"]
    part = consensus_partition(C)
    labels = part["labels"]
    voting = {f: labels_by_family[f] for f in coas["families"]}
    conf = confidence(C, labels)
    recall = consensus_recall(voting, labels)
    return {
        "co_association": coas,
        "consensus": part,
        "confidence": conf,
        "consensus_strength": consensus_strength(C),
        "mean_recall": recall["mean_recall"],
        "min_recall": recall["min_recall"],
        "per_family_recall": recall["per_family"],
        "refusal_fraction": refusal_fraction(voting),
        "n_families": coas["n_families"],
        "families": coas["families"],
        "weights": coas["weights"],
    }
