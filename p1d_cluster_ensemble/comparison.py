"""
p1d_cluster_ensemble/comparison.py — what the ensemble adds, if anything.

Everything upstream of this module is descriptive. A tuned method is not
a result; a prettier annotation is not a result. This module holds the
three comparisons that can come back negative, and the negative readings
are written out as plainly as the positive ones:

  P-C2  is the shipped HDBSCAN setting the tuned one?      shipped_comparison
  P-C3  is the unclustered population homogeneous?          noise_rescue
  P-C4  does the graded annotation beat the binary flag?    delta_auc_report

P-C4 in particular is the phase's own falsification: if graded confidence
does not predict layer-to-layer cluster persistence better than "HDBSCAN
called it clustered", then the conglomeration is presentation and the
categorical label loses nothing.

The persistence target
----------------------
Both predictors are scored against the same thing: for each particle at
layer L, the fraction of its consensus-cluster co-members at L that are
still its co-members at L+1. That target is computed from the *consensus*
partition at both layers, never from HDBSCAN — scoring a graded
annotation against a target one of the two predictors defined would rig
the comparison, which PREDICTIONS.md fixes as Phase 1d adjudication
constraint 4.

It also means the target is only defined where consecutive layers share a
token set, and that both layers' ensembles must exist. Layers where
either is missing are absent from the report rather than filled in.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from sklearn.metrics import adjusted_rand_score

from .constants import SHIPPED_HDBSCAN_PARAMS

#: A particle "persists" when at least this fraction of its consensus
#: co-members are still co-members one layer deeper. PLACED, not
#: calibrated: there is no distribution to derive it from, and the
#: continuous fraction is reported alongside every boolean so a reader
#: can see how much the cut point is carrying.
PERSISTENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# P-C2 — the shipped setting against the tuned one
# ---------------------------------------------------------------------------

def shipped_comparison(
    tuned_labels: np.ndarray,
    shipped_labels: np.ndarray,
    tuned_params: Dict,
) -> Dict:
    """
    How far the tuned HDBSCAN partition is from the one every existing
    cluster-conditioned result in this project was computed on.

    `shipped_labels` must be Phase 1's own persisted labels — the
    partition on disk, not a re-run — so that the comparison carries no
    implementation difference inside it. p1d_io.load_run reads them from
    hdbscan_labels.json for exactly this reason.

    ARI is reported under both noise conventions because they answer
    different questions and, on a partition with many refusals, differ by
    a lot: `ari_raw` treats -1 as an ordinary shared label ("do the two
    settings refuse the same tokens"), `ari_assigned` compares only the
    tokens both settings assigned ("where both committed, did they
    agree").
    """
    tuned = np.asarray(tuned_labels)
    shipped = np.asarray(shipped_labels)
    out: Dict[str, object] = {
        "tuned_params": dict(tuned_params),
        "shipped_params": dict(SHIPPED_HDBSCAN_PARAMS),
        "params_identical": _params_match(tuned_params, SHIPPED_HDBSCAN_PARAMS),
        "n": int(tuned.size),
        "ari_raw": float("nan"),
        "ari_assigned": float("nan"),
        "n_both_assigned": 0,
        "tuned_noise_fraction": float("nan"),
        "shipped_noise_fraction": float("nan"),
    }
    if tuned.size != shipped.size or tuned.size < 2:
        out["error"] = (
            f"token counts differ (tuned {tuned.size}, shipped {shipped.size}); "
            "no comparison made"
        )
        return out

    out["ari_raw"] = float(adjusted_rand_score(tuned, shipped))
    both = (tuned >= 0) & (shipped >= 0)
    out["n_both_assigned"] = int(both.sum())
    if both.sum() >= 2 and np.unique(tuned[both]).size + np.unique(shipped[both]).size > 2:
        out["ari_assigned"] = float(adjusted_rand_score(tuned[both], shipped[both]))
    out["tuned_noise_fraction"] = float((tuned < 0).mean())
    out["shipped_noise_fraction"] = float((shipped < 0).mean())
    return out


def _params_match(a: Dict, b: Dict) -> bool:
    keys = set(a) | set(b)
    return all(a.get(k) == b.get(k) for k in keys)


def adjudicate_p_c2(
    per_layer: Dict[int, Dict], ari_threshold: float = 0.9,
) -> Dict:
    """
    P-C2's verdict over a run's layers: was the shipped setting the tuned
    one, and where it was not, how different is the partition?

    Reports the fraction of comparable layers whose `ari_assigned` falls
    below `ari_threshold`, and the fraction where the tuned params
    equalled the shipped params exactly. A confirmed P-C2 is
    "params_identical rare AND most layers below the ARI threshold". The
    two are reported separately because they can disagree: a different
    min_cluster_size that produces nearly the same partition would
    satisfy the letter of P-C2 while meaning that nothing downstream
    changes, and that reading should be visible rather than resolved by
    the verdict string.
    """
    layers = sorted(per_layer)
    aris = np.array([per_layer[l].get("ari_assigned", np.nan) for l in layers], dtype=float)
    identical = np.array([bool(per_layer[l].get("params_identical", False)) for l in layers])
    usable = np.isfinite(aris)
    n_usable = int(usable.sum())
    below = float((aris[usable] < ari_threshold).mean()) if n_usable else float("nan")
    return {
        "n_layers": len(layers),
        "n_usable": n_usable,
        "ari_threshold": float(ari_threshold),
        "fraction_below_threshold": below,
        "fraction_params_identical": float(identical.mean()) if identical.size else float("nan"),
        "median_ari_assigned": float(np.nanmedian(aris)) if n_usable else float("nan"),
        "verdict": _p_c2_verdict(below, identical),
    }


def _p_c2_verdict(below: float, identical: np.ndarray) -> str:
    if not np.isfinite(below) or identical.size == 0:
        return "UNDECIDED — no comparable layers"
    frac_id = float(identical.mean())
    if frac_id > 0.5:
        return (
            f"P-C2 FALSIFIED — the shipped setting was selected at "
            f"{frac_id:.0%} of layers; the default was already stability-optimal"
        )
    if below > 0.5:
        return (
            f"P-C2 CONFIRMED — tuned setting differs at {1 - frac_id:.0%} of "
            f"layers and the partition differs (ARI below threshold at {below:.0%})"
        )
    return (
        f"P-C2 PARTIAL — the setting changed at {1 - frac_id:.0%} of layers but "
        f"the partition barely moved (ARI below threshold at only {below:.0%}); "
        "downstream cluster-conditioned results are largely unaffected"
    )


# ---------------------------------------------------------------------------
# P-C3 — is the unclustered population homogeneous?
# ---------------------------------------------------------------------------

def noise_rescue(
    hdbscan_labels: np.ndarray,
    confidence: np.ndarray,
    consensus_labels: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict:
    """
    What the ensemble says about the particles HDBSCAN refused.

    `core_fraction` is P-C3's instrument: the fraction of refused
    particles whose consensus confidence clears the null-calibrated
    "core" threshold — i.e. other methods place them in structure at a
    level a structureless cloud with the same marginals reaches 5% of the
    time.

    `into_shared` separates the two readings that fraction can have, the
    same split `noise_audit` makes in p1_visualization:

      high core_fraction, high into_shared — the refused particles were
        absorbed into clusters HDBSCAN itself found. Its density
        criterion drew a tighter boundary than the other methods.
      high core_fraction, low into_shared — the refused particles form
        their own coherent group. That is a different claim entirely:
        not "HDBSCAN was strict" but "there is a population here that
        HDBSCAN's criterion cannot represent", which is the reading that
        would matter to Phase 5c.
    """
    hdb = np.asarray(hdbscan_labels)
    conf = np.asarray(confidence, dtype=float)
    cons = np.asarray(consensus_labels)
    n = int(hdb.size)
    is_noise = hdb < 0
    n_noise = int(is_noise.sum())

    out: Dict[str, object] = {
        "n": n,
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / n) if n else float("nan"),
        "core_fraction": float("nan"),
        "into_shared": float("nan"),
        "own_cluster_fraction": float("nan"),
        "mean_confidence_noise": float("nan"),
        "mean_confidence_assigned": float("nan"),
        "confidence_gap": float("nan"),
        "thresholds": dict(thresholds),
    }
    if n_noise == 0 or conf.size != n or cons.size != n:
        return out

    core_t = thresholds.get("core", np.nan)
    if np.isfinite(core_t):
        out["core_fraction"] = float(np.nanmean(conf[is_noise] >= core_t))

    # Which consensus clusters also hold particles HDBSCAN did assign?
    shared_ids = set(np.unique(cons[~is_noise]).tolist()) if (~is_noise).any() else set()
    in_shared = np.array([int(c) in shared_ids for c in cons[is_noise]], dtype=float)
    if np.isfinite(core_t):
        rescued = conf[is_noise] >= core_t
        out["into_shared"] = float((rescued & (in_shared > 0)).mean())
        out["own_cluster_fraction"] = float((rescued & (in_shared == 0)).mean())

    out["mean_confidence_noise"] = float(np.nanmean(conf[is_noise]))
    if (~is_noise).any():
        out["mean_confidence_assigned"] = float(np.nanmean(conf[~is_noise]))
        out["confidence_gap"] = out["mean_confidence_assigned"] - out["mean_confidence_noise"]
    return out


def adjudicate_p_c3(per_layer: Dict[int, Dict], core_floor: float = 0.20) -> Dict:
    """
    P-C3's verdict: is at least `core_floor` of the refused population
    above the calibrated core threshold, at a typical layer?

    The floor is the placed number registered in PREDICTIONS.md. It is
    adjudicated on the median layer rather than the mean, because the
    noise fraction itself varies by an order of magnitude with depth and
    a mean would be dominated by whichever layers refuse most.
    """
    layers = sorted(per_layer)
    core = np.array([per_layer[l].get("core_fraction", np.nan) for l in layers], dtype=float)
    shared = np.array([per_layer[l].get("into_shared", np.nan) for l in layers], dtype=float)
    own = np.array([per_layer[l].get("own_cluster_fraction", np.nan) for l in layers], dtype=float)
    usable = np.isfinite(core)
    if not usable.any():
        return {"n_layers": len(layers), "n_usable": 0, "core_floor": core_floor,
                "verdict": "UNDECIDED — no layer had both refusals and a calibrated threshold"}

    median_core = float(np.nanmedian(core))
    median_shared = float(np.nanmedian(shared)) if np.isfinite(shared).any() else float("nan")
    median_own = float(np.nanmedian(own)) if np.isfinite(own).any() else float("nan")
    if median_core >= core_floor:
        reading = ("absorbed into clusters HDBSCAN also found"
                   if median_shared >= median_own
                   else "forming their own group, which HDBSCAN's criterion cannot represent")
        verdict = (f"P-C3 CONFIRMED — {median_core:.0%} of refused particles clear the "
                   f"calibrated core threshold at the median layer, mostly {reading}")
    else:
        verdict = (f"P-C3 FALSIFIED — only {median_core:.0%} of refused particles clear "
                   f"the calibrated threshold; the unclustered population is homogeneous "
                   f"at this resolution")
    return {
        "n_layers": len(layers), "n_usable": int(usable.sum()),
        "core_floor": float(core_floor),
        "median_core_fraction": median_core,
        "median_into_shared": median_shared,
        "median_own_cluster": median_own,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# P-C4 — the graded annotation against the binary flag
# ---------------------------------------------------------------------------

def persistence_target(
    consensus_here: np.ndarray,
    consensus_next: np.ndarray,
    threshold: float = PERSISTENCE_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """
    Per particle: what fraction of its consensus co-members at this layer
    are still co-members at the next one.

    Requires the two label arrays to describe the same tokens in the same
    order, which is what a Phase 1 run directory gives (one prompt, one
    token order, every layer). Particles alone in their cluster here have
    no co-members to keep and come back NaN — excluded from scoring
    rather than counted as a failure to persist, since there was nothing
    to persist.

    Returns {fraction: (n,), persisted: (n,) bool-with-NaN-as-False,
    scorable: (n,) bool}.
    """
    a = np.asarray(consensus_here)
    b = np.asarray(consensus_next)
    n = int(a.size)
    if b.size != n:
        raise ValueError(f"layer token counts differ: {n} vs {b.size}")

    same_here = (a[:, None] == a[None, :])
    np.fill_diagonal(same_here, False)
    same_next = (b[:, None] == b[None, :])
    np.fill_diagonal(same_next, False)

    denom = same_here.sum(axis=1)
    hit = (same_here & same_next).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(denom > 0, hit / np.maximum(denom, 1), np.nan)
    scorable = np.isfinite(frac)
    return {
        "fraction": frac,
        "persisted": np.where(scorable, frac >= threshold, False),
        "scorable": scorable,
        "threshold": float(threshold),
    }


def concordance_matrix(scores: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """
    (n_pos, n_neg) matrix of pairwise concordance: 1 where the positive
    outranks the negative, 0.5 on a tie, 0 otherwise. Its mean is the AUC.

    Materialized rather than summarized because the permutation test
    below needs the per-pair terms, and because computing AUC this way
    makes the tie handling explicit — the binary predictor is nearly all
    ties, and an AUC implementation that resolved them differently would
    change the comparison this module exists to make.
    """
    s = np.asarray(scores, dtype=float)
    pos = s[positive]
    neg = s[~positive]
    if pos.size == 0 or neg.size == 0:
        return np.zeros((0, 0))
    diff = pos[:, None] - neg[None, :]
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUC as the mean pairwise concordance. NaN when either class is empty."""
    M = concordance_matrix(scores, positive)
    return float(M.mean()) if M.size else float("nan")


def delta_auc_report(
    graded: np.ndarray,
    binary: np.ndarray,
    target: Dict[str, np.ndarray],
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict:
    """
    P-C4's instrument: does the graded confidence outrank the binary
    clustered/noise flag at predicting persistence?

    Two readings, deliberately, and the verdict refuses when they
    disagree — the same discipline UPDATE_PLAN.md §5.2 forced on T_eff,
    where three definitions of a step size straddled the threshold:

      permutation  a paired sign-flip test on the per-pair concordance
                   differences (the registered instrument). Exact for
                   independent pairs; the pairs here share particles, so
                   the p-value is approximate and biased anti-
                   conservative — stated, not hidden.
      bootstrap    resampling particles with replacement, which respects
                   the particle-level dependence the permutation test
                   does not, and gives a CI on the same quantity.

    A confirmed P-C4 needs both: delta_auc > 0, outside the permutation
    null's 2 sigma band, and a bootstrap CI excluding 0.
    """
    graded = np.asarray(graded, dtype=float)
    binary = np.asarray(binary, dtype=float)
    scorable = np.asarray(target["scorable"], dtype=bool) & np.isfinite(graded)
    positive = np.asarray(target["persisted"], dtype=bool)[scorable]
    g, b = graded[scorable], binary[scorable]

    out: Dict[str, object] = {
        "n_scorable": int(scorable.sum()),
        "n_positive": int(positive.sum()),
        "auc_graded": float("nan"), "auc_binary": float("nan"),
        "delta_auc": float("nan"), "permutation": None, "bootstrap": None,
        "verdict": "P-C4 UNDECIDED — not enough scorable particles",
    }
    if positive.sum() < 2 or (~positive).sum() < 2:
        return out

    Mg = concordance_matrix(g, positive)
    Mb = concordance_matrix(b, positive)
    out["auc_graded"] = float(Mg.mean())
    out["auc_binary"] = float(Mb.mean())
    delta = float(Mg.mean() - Mb.mean())
    out["delta_auc"] = delta

    D = Mg - Mb
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_permutations), dtype=float)
    for i in range(int(n_permutations)):
        signs = rng.choice([-1.0, 1.0], size=D.shape)
        null[i] = float((D * signs).mean())
    null_std = float(null.std())
    out["permutation"] = {
        "n_permutations": int(n_permutations),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "z_score": float(delta / null_std) if null_std > 1e-12 else float("nan"),
        "p_value": float((1 + int((np.abs(null) >= abs(delta)).sum())) / (n_permutations + 1)),
        "caveat": "pairs share particles; p is approximate and anti-conservative",
    }

    idx = np.arange(g.size)
    boots = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        take = rng.choice(idx, size=idx.size, replace=True)
        pos_b = positive[take]
        if pos_b.sum() < 2 or (~pos_b).sum() < 2:
            boots[i] = np.nan
            continue
        boots[i] = auc(g[take], pos_b) - auc(b[take], pos_b)
    finite = boots[np.isfinite(boots)]
    out["bootstrap"] = {
        "n_bootstrap": int(n_bootstrap),
        "n_usable": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
    }
    out["verdict"] = _p_c4_verdict(out)
    return out


def _p_c4_verdict(res: Dict) -> str:
    delta = res["delta_auc"]
    perm, boot = res["permutation"], res["bootstrap"]
    if perm is None or boot is None or not np.isfinite(delta):
        return "P-C4 UNDECIDED — not enough scorable particles"
    z = perm["z_score"]
    perm_pass = bool(np.isfinite(z) and z >= 2.0 and delta > 0)
    boot_pass = bool(np.isfinite(boot["ci_low"]) and boot["ci_low"] > 0)
    if perm_pass and boot_pass:
        return (f"P-C4 CONFIRMED — graded confidence beats the binary flag by "
                f"{delta:+.3f} AUC ({z:.1f}σ, bootstrap CI "
                f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}])")
    if not perm_pass and not boot_pass:
        return (f"P-C4 FALSIFIED — delta AUC {delta:+.3f} is not distinguishable "
                f"from zero; the categorical label loses nothing the ensemble recovers")
    return (f"P-C4 UNDECIDED — the two readings disagree (delta {delta:+.3f}, "
            f"permutation {z:.1f}σ, bootstrap CI "
            f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]); the permutation "
            f"test ignores particle-level dependence, so the bootstrap is the "
            f"conservative reading")


def adjudicate_p_c4(per_layer: Dict[int, Dict]) -> Dict:
    """
    P-C4 across a run's layer boundaries. Reports the median delta AUC and
    how many boundaries confirmed, falsified, or came back undecided —
    not a single pooled number, because pooling AUCs across layers with
    very different persistence rates is not a well-defined quantity.
    """
    layers = sorted(per_layer)
    deltas = np.array([per_layer[l].get("delta_auc", np.nan) for l in layers], dtype=float)
    verdicts = [str(per_layer[l].get("verdict", "")) for l in layers]
    counts = {
        "confirmed": sum(v.startswith("P-C4 CONFIRMED") for v in verdicts),
        "falsified": sum(v.startswith("P-C4 FALSIFIED") for v in verdicts),
        "undecided": sum(v.startswith("P-C4 UNDECIDED") for v in verdicts),
    }
    usable = np.isfinite(deltas)
    median = float(np.nanmedian(deltas)) if usable.any() else float("nan")
    if counts["confirmed"] > counts["falsified"] + counts["undecided"]:
        verdict = (f"P-C4 CONFIRMED at {counts['confirmed']}/{len(layers)} layer "
                   f"boundaries, median delta AUC {median:+.3f}")
    elif counts["falsified"] >= max(1, counts["confirmed"]):
        verdict = (f"P-C4 FALSIFIED — {counts['falsified']}/{len(layers)} boundaries "
                   f"show no improvement, median delta AUC {median:+.3f}")
    else:
        verdict = (f"P-C4 UNDECIDED — {counts['undecided']}/{len(layers)} boundaries "
                   f"undecided, median delta AUC {median:+.3f}")
    return {"n_boundaries": len(layers), "counts": counts,
            "median_delta_auc": median, "verdict": verdict}


# ---------------------------------------------------------------------------
# P-C1 — did tuning dissolve the agreement the defaults showed?
# ---------------------------------------------------------------------------

def adjudicate_p_c1(
    tuned_strength: Dict[int, float],
    agreement_layers: Optional[Sequence[int]] = None,
    strength_floor: float = 0.9,
) -> Dict:
    """
    P-C1's verdict: at the layers where Phase 1 already reported the
    methods agreeing (cluster counts within +/-1 — `agreement_layers`,
    from p1_visualization.cluster_methods.cluster_count_table), does the
    tuned ensemble still reach `strength_floor` consensus strength?

    When `agreement_layers` is None the test runs over every layer with a
    consensus, and says so: that is a weaker statement than the
    registered one, because Phase 1's agreement layers are where the
    prediction was aimed.
    """
    layers = sorted(tuned_strength)
    scoped = ([l for l in layers if l in set(agreement_layers)]
              if agreement_layers is not None else layers)
    vals = np.array([tuned_strength[l] for l in scoped], dtype=float)
    usable = np.isfinite(vals)
    if not usable.any():
        return {"n_layers": len(scoped), "n_usable": 0,
                "scope": "phase1_agreement_layers" if agreement_layers is not None else "all_layers",
                "verdict": "UNDECIDED — no layer produced a consensus"}
    frac = float((vals[usable] >= strength_floor).mean())
    scope = ("Phase 1's own agreement layers" if agreement_layers is not None
             else "ALL layers — Phase 1's agreement layers were unavailable, "
                  "so this is a weaker test than the registered one")
    verdict = (
        f"P-C1 CONFIRMED — tuned consensus strength stays above "
        f"{strength_floor} at {frac:.0%} of {len(scoped)} layers ({scope})"
        if frac > 0.5 else
        f"P-C1 FALSIFIED — tuned consensus strength falls below "
        f"{strength_floor} at {1 - frac:.0%} of {len(scoped)} layers ({scope}); "
        f"the agreement the defaults showed does not survive tuning"
    )
    return {
        "n_layers": len(scoped), "n_usable": int(usable.sum()),
        "scope": "phase1_agreement_layers" if agreement_layers is not None else "all_layers",
        "strength_floor": float(strength_floor),
        "fraction_above_floor": frac,
        "median_strength": float(np.nanmedian(vals)),
        "verdict": verdict,
    }
