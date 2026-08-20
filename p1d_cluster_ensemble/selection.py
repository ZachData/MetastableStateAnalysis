"""
p1d_cluster_ensemble/selection.py — what "tuned" means here.

The problem this module exists for
----------------------------------
Every off-the-shelf way of choosing a clustering hyperparameter is
already known to fail on this project's data, and the failure is
documented inside the project:

  "K_RANGE starts at 2, so best_k=2 is a floor, not a finding. In the
   collapsed regime all tokens are near-collinear and any 2-way split
   scores a silhouette of ~0.1-0.3 from geometry alone."
   (p1_visualization/cluster_methods.py)

That is not a quirk of silhouette. Every internal index — silhouette,
Calinski-Harabasz, Davies-Bouldin — is a ratio of within- to
between-cluster spread, and a collapsed cloud has a perfectly good best
split at every k. Tuning on one would produce a "tuned" method that is
tuned to the collapse.

So selection here rests on two statistics, both calibrated against the
same matched null, and neither sufficient alone:

1. **Stability under resampling.** Cluster twice on overlapping
   subsamples of the same tokens and measure ARI on the overlap
   (Ben-Hur/Elisseeff/Guyon). A partition driven by geometry survives
   dropping 20% of the tokens; a partition driven by where the cut
   happened to fall does not.

2. **Separation** — silhouette on the same cosine distances. The index
   the module docstring above just called unusable, which it is *against
   an absolute threshold*. Against a matched baseline it is exactly the
   construction UPDATE_PLAN.md §5.7 already forced on Q_k: "P-S1 is
   adjudicated on the ratio to a matched random baseline", because
   E[Q_k] = 1/n makes every large-n configuration look like a spherical
   design under a fixed cutoff. Same disease, same cure.

**The matched null.** `core.nulls.shuffled_dimension_null`: same
per-dimension marginals, same token count, cross-token geometry
destroyed, re-normalized onto the sphere, and then the *whole pipeline*
re-run on it — distance matrix, fit, and both statistics. What survives
is "more reproducible, and better separated, than a structureless cloud
with these marginals", each in units of that null's own sigma.

**Why both.** Each catches a failure the other cannot:

  - Stability alone admits structureless data. k-means at k=2 on i.i.d.
    points on the sphere is highly reproducible — the split it finds is a
    real property of the sample, just not a cluster.
  - Separation alone admits a partition nobody could reproduce: a
    linkage that peels a different handful of outliers off each
    subsample can score a fine silhouette on the tokens it did assign.

So they are not used symmetrically, and this is the one design decision
in the module worth arguing with:

  **Separation is the significance gate.** Rank test against the matched
  null, at `alpha`. This is what decides whether there is structure here
  at all.

  **Stability is a floor and a ranking, not a second significance test.**
  A setting must be at least as reproducible as the matched null's mean
  — it may not be *worse* than structureless data — and among settings
  that pass, the most reproducible one wins. Its full calibration (p
  value, sigma, percentile) is reported either way.

Why not demand significance from both: stability is bounded at 1 and
saturates. On three cleanly planted caps, spectral clustering scores a
perfect 1.00 while two of twenty structureless draws also score 1.00 —
p = 0.14, a failure by ties alone, for a partition that recovers the
planted structure exactly and separates at p = 0.05. Requiring
significance from a statistic whose null piles up on the ceiling
discards true structure, which is the more expensive error here: an
abstaining family removes a whole inductive bias from the consensus.

Two-stage, and why
------------------
Computing null distributions for every grid point costs
n_grid x n_null x n_repeats fits. Stage 1 ranks the grid by stability
alone (cheap); stage 2 computes the null gate for the top `top_m`
candidates only, and selects the best-ranked one that passes. This is a
real approximation — a candidate ranked 4th on raw stability that would
have passed the gate while the top 3 fail is never examined — and it is
recorded in the artifact as `top_m` rather than presented as an
exhaustive search.

Refusal
-------
If no candidate passes the gate, this module does not return the
best-of-a-bad-grid. It returns `selected=None` with a reason, and the
family abstains from the ensemble at that layer (PREDICTIONS.md, Phase 1d
adjudication constraint 3). "Refuse rather than degrade" is standing rule
4 in UPDATE_PLAN.md §6; a tuned setting for a family that found nothing
is exactly the kind of number that is unfalsifiable from the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sklearn.metrics import adjusted_rand_score, silhouette_score

from core.nulls import shuffled_dimension_null, sigma_from_null

from .constants import SUBSTANTIAL_CLUSTER_SIZE
from .methods import LayerData, fit, param_grid

# ---------------------------------------------------------------------------
# Thresholds. Both PLACED, not calibrated — there is no distribution to
# derive either from before the phase runs, which is exactly why they are
# named here with the label on them rather than buried at a call site.
# ---------------------------------------------------------------------------

#: Significance level for both matched-null gates. The decision is made
#: on a RANK test — p = (1 + #{null >= observed}) / (n_null + 1) — and not
#: on the N-sigma summary this project reports elsewhere, for a reason
#: worth stating because it is a departure:
#:
#: both statistics are bounded above (ARI and silhouette by 1), and on
#: data with real structure the observation sits at or near that bound.
#: A z-score there is compressed by the null's own spread and reads as
#: "not significant" for a partition that exceeded every single null draw
#: — measured, not hypothesised: k-means at k=3 on three planted caps
#: scores 1.00 against a null mean of 0.85, which is 1.9σ and outside
#: every one of the null draws. The rank test says what actually
#: happened. `z_score` is still computed and written into every artifact
#: so these numbers stay readable next to the project's other null
#: comparisons; it is just not what decides.
NULL_ALPHA = 0.05

#: A partition where one cluster holds this much of the token set is a
#: relabeling of "everything", whatever its stability.
TRIVIAL_DOMINANCE = 0.95

#: Fraction of tokens each subsample keeps.
SUBSAMPLE_FRACTION = 0.8

#: Branch labels written into every gate result, so a reloaded artifact
#: says how the decision was reached and not merely what it was.
_STABILITY_FLOOR = "stability_floor"
_STABILITY_DEGENERATE = "stability_null_degenerate"


# ---------------------------------------------------------------------------
# Partition shape
# ---------------------------------------------------------------------------

def partition_summary(
    labels: np.ndarray, substantial: int = SUBSTANTIAL_CLUSTER_SIZE,
) -> Dict:
    """
    The shape of a partition, and whether it is trivial.

    "Substantial" repeats `noise_audit`'s argument from
    p1_visualization/cluster_methods.py: a cluster of one is the same
    refusal HDBSCAN's -1 makes, spelled differently. A partition of 400
    tokens into 380 singletons and two pairs has k=382 and says nothing,
    so `k_substantial` — not k — is what the triviality test reads.
    """
    labels = np.asarray(labels)
    n = int(labels.size)
    out = {
        "n": n, "k": 0, "k_substantial": 0, "assigned_fraction": 0.0,
        "largest_fraction": 0.0, "singleton_fraction": 0.0,
        "trivial": True, "trivial_reason": "empty",
    }
    if n == 0:
        return out

    assigned = labels[labels >= 0]
    out["assigned_fraction"] = float(assigned.size / n)
    if assigned.size == 0:
        out["trivial_reason"] = "nothing assigned"
        return out

    ids, sizes = np.unique(assigned, return_counts=True)
    out["k"] = int(ids.size)
    out["k_substantial"] = int((sizes >= substantial).sum())
    out["largest_fraction"] = float(sizes.max() / n)
    out["singleton_fraction"] = float((sizes == 1).sum() / n)

    if out["k_substantial"] < 2:
        out["trivial_reason"] = (
            f"fewer than two clusters of size >= {substantial} "
            f"(k={out['k']}, k_substantial={out['k_substantial']})"
        )
    elif out["largest_fraction"] > TRIVIAL_DOMINANCE:
        out["trivial_reason"] = (
            f"one cluster holds {out['largest_fraction']:.2f} of all tokens"
        )
    else:
        out["trivial"] = False
        out["trivial_reason"] = ""
    return out


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def subsample_stability(
    family: str,
    params: Dict,
    data: LayerData,
    n_repeats: int = 5,
    fraction: float = SUBSAMPLE_FRACTION,
    seed: int = 0,
) -> Dict:
    """
    Mean ARI between two independent subsample clusterings, over their
    shared tokens, repeated `n_repeats` times.

    Two independent subsamples rather than subsample-vs-full: comparing a
    subsample against the partition of the whole set rewards a method for
    being *insensitive*, since the full-set partition is one of the two
    every time. Two draws put both partitions at the same disadvantage,
    which is what makes the number a reproducibility statistic rather than
    an agreement-with-myself statistic.

    Noise is compared as-is (-1 is a label like any other for ARI's
    purposes here). Treating each refusal as its own singleton cluster —
    the "singleton" policy in p1_visualization/cluster_methods.py — would
    make an unstable *set* of refusals read as instability of the
    clusters, and the noise question is asked separately and directly in
    comparison.py. Which convention was used matters and is stated here
    rather than left to be inferred from the number.

    Returns {mean_ari, std_ari, per_repeat, n_overlap_mean, seed}. An
    unusable repeat (overlap < 2 tokens, or a fit that returns fewer than
    two labels on one side) contributes NaN and is reported, not dropped
    silently.
    """
    rng = np.random.default_rng(seed)
    n = data.n
    size = int(max(2, round(fraction * n)))
    scores: List[float] = []
    overlaps: List[int] = []

    for _ in range(max(1, n_repeats)):
        idx_a = np.sort(rng.choice(n, size=size, replace=False))
        idx_b = np.sort(rng.choice(n, size=size, replace=False))
        shared = np.intersect1d(idx_a, idx_b)
        overlaps.append(int(shared.size))
        if shared.size < 2:
            scores.append(np.nan)
            continue
        lab_a = fit(family, params, data.subset(idx_a), seed=int(rng.integers(1 << 30)))
        lab_b = fit(family, params, data.subset(idx_b), seed=int(rng.integers(1 << 30)))
        pos_a = {t: p for p, t in enumerate(idx_a)}
        pos_b = {t: p for p, t in enumerate(idx_b)}
        a = np.array([lab_a[pos_a[t]] for t in shared])
        b = np.array([lab_b[pos_b[t]] for t in shared])
        scores.append(float(adjusted_rand_score(a, b)))

    arr = np.asarray(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "mean_ari": float(finite.mean()) if finite.size else float("nan"),
        "std_ari": float(finite.std()) if finite.size else float("nan"),
        "per_repeat": [None if not np.isfinite(v) else float(v) for v in arr],
        "n_repeats_usable": int(finite.size),
        "n_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "fraction": float(fraction),
        "seed": int(seed),
    }


def separation_score(labels: np.ndarray, data: LayerData) -> float:
    """
    Silhouette of a partition on the cosine distances it was built from,
    over the assigned tokens only.

    Refused tokens are excluded rather than treated as their own cluster:
    a silhouette that counts every HDBSCAN refusal as a singleton
    measures the refusal rate, not the separation of the clusters that
    were found. This is the "exclude" noise policy in
    p1_visualization/cluster_methods.py, and it is the fairer question
    about the clusters a method does commit to.

    NaN when fewer than two clusters survive that exclusion, or fewer
    than three tokens do — not 0.0, which would read as "no separation"
    rather than "not measurable".
    """
    labels = np.asarray(labels)
    keep = labels >= 0
    sub = labels[keep]
    if sub.size < 3 or np.unique(sub).size < 2:
        return float("nan")
    D = data.cos_dist[np.ix_(np.flatnonzero(keep), np.flatnonzero(keep))]
    return float(silhouette_score(D, sub, metric="precomputed"))


def calibrate(observed: float, null_values: np.ndarray, alpha: float = NULL_ALPHA) -> Dict:
    """
    `core.nulls.sigma_from_null` plus the rank test the gate actually
    reads: p = (1 + #{null >= observed}) / (n_null + 1), the standard
    conservative permutation p-value.

    `p_floor` is the smallest p this many draws can produce. A gate whose
    alpha is below its own p_floor can never pass; that is a
    configuration error, and `select_family` refuses rather than running
    a sweep whose every outcome is predetermined.
    """
    null_values = np.asarray(null_values, dtype=np.float64)
    finite = null_values[np.isfinite(null_values)]
    if finite.size == 0 or not np.isfinite(observed):
        return {
            "observed": (float(observed) if np.isfinite(observed) else None),
            "null_mean": None, "null_std": None, "z_score": None,
            "percentile": None, "n_null": int(finite.size),
            "p_value": None, "p_floor": None, "alpha": float(alpha),
            "usable": False,
        }
    summary = sigma_from_null(observed, finite)
    n_ge = int((finite >= observed).sum())
    summary.update({
        "n_null_ge": n_ge,
        "p_value": float((1 + n_ge) / (finite.size + 1)),
        "p_floor": float(1.0 / (finite.size + 1)),
        "alpha": float(alpha),
        "usable": True,
    })
    return summary


def null_distributions(
    family: str,
    params: Dict,
    data: LayerData,
    n_null: int = 20,
    n_repeats: int = 3,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Both statistics on `n_null` shuffled-dimension null draws.

    Two calls into `core.nulls.shuffled_dimension_null` rather than one
    loop computing both: the null construction is shared project-wide and
    a second, subtly different shuffle in one phase is how two "N sigma
    from null" numbers stop being comparable. The two calls are given
    identically-seeded generators, so they see the *same* sequence of
    shuffled clouds — the pairing is free and makes the two sigmas
    statements about one null population rather than two.

    The separation pass is roughly 1/(2*n_repeats) of the cost of the
    stability pass: one fit per draw against 2*n_repeats.
    """
    def _stability(shuffled: np.ndarray) -> float:
        null_data = LayerData.from_normed(shuffled)
        return subsample_stability(
            family, params, null_data, n_repeats=n_repeats, seed=seed,
        )["mean_ari"]

    def _separation(shuffled: np.ndarray) -> float:
        null_data = LayerData.from_normed(shuffled)
        return separation_score(fit(family, params, null_data, seed=seed), null_data)

    stab = shuffled_dimension_null(
        data.normed, _stability, n_shuffles=int(n_null), renormalize=True,
        rng=np.random.default_rng(seed + 977),
    )
    sep = shuffled_dimension_null(
        data.normed, _separation, n_shuffles=int(n_null), renormalize=True,
        rng=np.random.default_rng(seed + 977),
    )
    return {"stability": stab, "separation": sep}


# ---------------------------------------------------------------------------
# Candidates and selection
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One grid point, with everything that was measured about it."""
    family: str
    params: Dict
    labels: np.ndarray
    shape: Dict
    stability: Dict
    separation: float = float("nan")
    null: Optional[Dict] = None          # {stability, separation} sigma summaries
    admissible: Optional[bool] = None    # None = gate not evaluated
    reason: str = ""
    branch: str = ""                     # which gate path the decision took

    def to_dict(self, include_labels: bool = False) -> Dict:
        out = {
            "family": self.family,
            "params": _jsonable(self.params),
            "shape": self.shape,
            "stability": self.stability,
            "separation": (None if not np.isfinite(self.separation)
                           else float(self.separation)),
            "null": self.null,
            "admissible": self.admissible,
            "reason": self.reason,
            "branch": self.branch,
        }
        if include_labels:
            out["labels"] = np.asarray(self.labels, dtype=int).tolist()
        return out


def apply_gate(cand: Candidate, alpha: float = NULL_ALPHA) -> Candidate:
    """
    Decide admissibility from the two calibrated statistics already
    attached to `cand.null`, recording which branch the decision took.

    Separation is the significance test — a partition no better separated
    than a structureless cloud with the same marginals is not admitted
    however reproducible it is. Stability is a floor: it may not be worse
    than that same null's mean. See the module docstring for why the two
    are not symmetric; the short version is that stability's null piles
    up on the ceiling, so demanding significance from it discards true
    structure by ties.

    Both branch labels are recorded, per standing rule 2, so a reloaded
    artifact says how each decision was reached and not merely what it
    was.
    """
    if cand.null is None:
        raise ValueError("apply_gate called before null distributions were computed")

    sep, stab = cand.null["separation"], cand.null["stability"]
    reasons: List[str] = []

    if not np.isfinite(cand.separation):
        sep_ok, sep_branch = False, "separation_not_measurable"
        reasons.append("separation not measurable (fewer than two assigned clusters)")
    elif not sep.get("usable"):
        sep_ok, sep_branch = False, "separation_null_unusable"
        reasons.append("no usable separation null draws — no calibration possible")
    else:
        sep_branch = "separation_rank_test"
        sep_ok = bool(sep["p_value"] <= alpha and sep["observed"] > sep["null_mean"])
        if not sep_ok:
            reasons.append(
                f"separation p={sep['p_value']:.3f} against the matched null "
                f"(observed {sep['observed']:.3f}, null mean {sep['null_mean']:.3f})"
            )

    if not stab.get("usable"):
        stab_branch, stab_ok = _STABILITY_DEGENERATE, False
        reasons.append("no usable stability null draws — no calibration possible")
    else:
        stab_branch = _STABILITY_FLOOR
        stab_ok = bool(stab["observed"] >= stab["null_mean"] - 1e-9)
        if not stab_ok:
            reasons.append(
                f"stability {stab['observed']:.2f} below the matched null's mean "
                f"{stab['null_mean']:.2f} — structureless draws of this cloud "
                "reproduce better than the real one does"
            )

    cand.admissible = bool(sep_ok and stab_ok)
    cand.branch = f"{sep_branch}+{stab_branch}"
    if cand.admissible:
        cand.reason = (
            f"separation p={sep['p_value']:.3f}; stability {stab['observed']:.2f} "
            f"against a null mean of {stab['null_mean']:.2f} "
            f"(p={stab['p_value']:.3f}, not gated on)"
        )
    else:
        cand.reason = "; ".join(reasons)
    return cand


def sweep_family(
    family: str,
    data: LayerData,
    grid: str = "full",
    n_repeats: int = 5,
    seed: int = 0,
) -> List[Candidate]:
    """
    Stage 1: fit every grid point once, measure its shape and its
    subsample stability. No gate applied yet, nothing discarded — a
    trivial or unstable candidate is kept with its numbers, because "the
    whole surface" is the artifact and the argmax is a view of it.
    """
    out: List[Candidate] = []
    for params in param_grid(family, data.n, grid=grid):
        labels = fit(family, params, data, seed=seed)
        shape = partition_summary(labels)
        stab = (subsample_stability(family, params, data,
                                    n_repeats=n_repeats, seed=seed)
                if not shape["trivial"]
                # A trivial partition's stability is meaningless and its
                # fits are not free; skipping them is the one shortcut
                # taken here, and it can only remove candidates that the
                # gate would reject anyway.
                else {"mean_ari": float("nan"), "std_ari": float("nan"),
                      "per_repeat": [], "n_repeats_usable": 0,
                      "n_overlap_mean": 0.0, "fraction": SUBSAMPLE_FRACTION,
                      "seed": int(seed), "skipped": "trivial partition"})
        out.append(Candidate(
            family=family, params=params, labels=labels, shape=shape,
            stability=stab,
            separation=(separation_score(labels, data) if not shape["trivial"]
                        else float("nan")),
        ))
    return out


def select_family(
    family: str,
    data: LayerData,
    grid: str = "full",
    n_repeats: int = 5,
    n_null: int = 20,
    n_null_repeats: int = 3,
    top_m: int = 3,
    alpha: float = NULL_ALPHA,
    seed: int = 0,
) -> Dict:
    """
    Tune one family at one layer.

    Returns
    -------
    dict with keys:
      family        : str
      selected      : the winning Candidate as a dict, or None
      selected_labels : (n,) int list, or None
      reason        : why nothing was selected, when selected is None
      candidates    : every grid point's numbers (the surface)
      gated         : the top_m candidates the null gate was computed for
      settings      : the knobs this selection was made under

    `selected` is None whenever no candidate clears both the triviality
    and the null gate, and that is a legitimate, reportable outcome: it
    says this family found nothing at this layer that a structureless
    cloud with the same marginals would not have produced.
    """
    p_floor = 1.0 / (int(n_null) + 1)
    if p_floor > alpha:
        raise ValueError(
            f"n_null={n_null} gives a smallest attainable p-value of "
            f"{p_floor:.3f}, above the requested alpha={alpha}: no candidate "
            f"could pass the gate whatever the data says. Raise n_null to at "
            f"least {int(np.ceil(1.0 / alpha)) - 1}, or raise alpha "
            f"deliberately."
        )

    candidates = sweep_family(family, data, grid=grid,
                              n_repeats=n_repeats, seed=seed)

    ranked = sorted(
        [c for c in candidates if not c.shape["trivial"]
         and np.isfinite(c.stability["mean_ari"])],
        key=lambda c: c.stability["mean_ari"], reverse=True,
    )

    settings = {
        "grid": grid, "n_repeats": n_repeats, "n_null": n_null,
        "n_null_repeats": n_null_repeats, "top_m": top_m,
        "alpha": alpha, "seed": seed,
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "trivial_dominance": TRIVIAL_DOMINANCE,
        "substantial_cluster_size": SUBSTANTIAL_CLUSTER_SIZE,
    }

    if not ranked:
        return {
            "family": family, "selected": None, "selected_labels": None,
            "reason": "every grid point produced a trivial partition",
            "candidates": [c.to_dict() for c in candidates],
            "gated": [], "settings": settings,
        }

    gated: List[Candidate] = []
    winner: Optional[Candidate] = None
    for cand in ranked[: max(1, top_m)]:
        nulls = null_distributions(family, cand.params, data, n_null=n_null,
                                   n_repeats=n_null_repeats, seed=seed)
        cand.null = {
            "stability": calibrate(cand.stability["mean_ari"],
                                   nulls["stability"], alpha=alpha),
            "separation": calibrate(cand.separation,
                                    nulls["separation"], alpha=alpha),
        }
        apply_gate(cand, alpha=alpha)
        gated.append(cand)
        if winner is None and cand.admissible:
            winner = cand

    if winner is None:
        return {
            "family": family, "selected": None, "selected_labels": None,
            "reason": (
                f"no candidate among the top {min(top_m, len(ranked))} by "
                f"stability cleared the matched-null gates at alpha={alpha} "
                "(separation and stability)"
            ),
            "candidates": [c.to_dict() for c in candidates],
            "gated": [c.to_dict() for c in gated], "settings": settings,
        }

    return {
        "family": family,
        "selected": winner.to_dict(),
        "selected_labels": np.asarray(winner.labels, dtype=int).tolist(),
        "reason": "",
        "candidates": [c.to_dict() for c in candidates],
        "gated": [c.to_dict() for c in gated],
        "settings": settings,
    }


def select_all_families(
    data: LayerData,
    families: Sequence[str],
    **kwargs,
) -> Dict[str, Dict]:
    """`select_family` for each family, keyed by family name."""
    return {f: select_family(f, data, **kwargs) for f in families}


def selected_labels(selection: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """
    {family: labels} for the families that selected something. Families
    that abstained are absent rather than present-and-empty, so a caller
    counting votes counts only real votes.
    """
    out: Dict[str, np.ndarray] = {}
    for family, res in selection.items():
        if res.get("selected_labels") is not None:
            out[family] = np.asarray(res["selected_labels"], dtype=np.int32)
    return out


def selection_weights(selection: Dict[str, Dict]) -> Dict[str, float]:
    """
    {family: weight} for the ensemble, where the weight is the selected
    setting's own stability (mean subsample ARI, clipped at 0).

    A family that barely cleared the gate should not carry the same vote
    as one that is reproducible at 0.9. Clipped rather than shifted
    because a negative ARI means "worse than chance agreement with
    itself", which is not a small vote — it is no vote.
    """
    out: Dict[str, float] = {}
    for family, res in selection.items():
        sel = res.get("selected")
        if sel is None:
            continue
        out[family] = float(max(0.0, sel["stability"]["mean_ari"]))
    return out


def _jsonable(params: Dict) -> Dict:
    """numpy scalars out of the param dicts, so json.dump does not choke."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
