"""
tools/dry_run_p6_r2_r4.py — P6-R2 and P6-R4 on inputs whose answer is known.

`claims/EVALUABILITY.md`'s queue: every converted row is owed a run on an input
whose correct verdict is fixed a priori, ahead of converting the next one.
`CLAIM-C` had it (2026-08-25), `P-ST1` had it (2026-08-26), and these two are
the third and fourth. The record is `claims/audits/p6_r2_r4_dry_run.json`.

WHY THESE TWO WERE NEXT, AND IT IS NOT THE QUEUE ORDER

`POPPER_PLAN.md` 6m retired the null `P-ST1` adjudicated: a MATCHED-DIMENSION
random orthogonal subspace pair, which randomises the union of the two
subspaces together with the split between them, so it rejects when the pair is
unusual AS A PAIR rather than when the labelling predicts anything.

That construction is 6h's, and 6h introduced it HERE -- for `P6-R2` and
`P6-R4`. A defect found in a construction one claim borrowed is a defect to
check wherever it was borrowed from, which `EVALUABILITY.md`'s opening argument
makes non-optional: the product is only as valid as its weakest factor, and
these two entries share a construction with a third.

WHAT THE CHECK FOUND, AND THE TWO ENTRIES DIFFER

`P6-R2` has the defect. Measured on an H0 whose split is uniformly random by
construction -- so the correct answer is "do not reject" -- the retired null's
rejection rate is a monotone TREND in how far the union sits above chance
against the layer's separating direction, and the trend is the mechanism rather
than a rate: 0.000 at chance, rising past 0.15 at ~3.9x chance, against a
nominal 0.05. The null that replaces it holds the union fixed and re-splits it,
and does not trend -- 0.047 and 0.068 at the two ends of the same sweep, which
pools with a 1000-replicate run during construction to about 0.056 at the
aligned end. At or marginally above nominal, flat, and 9.7 standard errors
below the retired null where that null fails.

`P6-R4` does NOT have it, and the reason is structural rather than lucky. It
compares ONE subspace against matched-dimension random ones: there is no union
and no split for this defect to reach. Measured where a high-variance `U_S`
captures 3.4x the population variance a random subspace of its dimension would,
its rate holds at 0.040-0.048. It is left alone -- changing it would have been a
change with no measurement behind it.

**So 6h's construction is safe or not according to the STATISTIC built on it**,
which is the transferable finding and the reason this file measures both:

  * a SIGN of a difference (`P-ST1`) saturates, so a common elevation of both
    arms does not cancel at all and the null fails hardest -- 6m measured 0.20
    where each arm held 1.27x chance;
  * a DIFFERENCE of two chance-normalized alignments (`P6-R2`) cancels a common
    elevation to FIRST order, so the null survives until the union is strongly
    aligned and then fails;
  * a SINGLE subspace against matched-dimension controls (`P6-R4`) has no
    common elevation to mismatch and is unaffected.

`EVALUABILITY.md` lists `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1` and
`P-I4` as queued rows whose predictions already name a matched control. That
taxonomy is what they should be read against before the control is built.

FOUR ARMS

A. `known_answer` -- the separating direction planted inside `U_neg` (P6-R2
   must reject), inside `U_A` (it must return p = 1, the arms reversed), and in
   neither. The unit tests already assert these; what this arm adds is the
   whole sweep at one place with the retired null scored on the same draws.

B. `union_alignment_sweep` -- the retirement evidence. Both nulls on the same
   runs, so the comparison is paired rather than two experiments.

C. `r4_variance_sweep` -- the evidence that `P6-R4` is not affected, which is
   what makes leaving it alone a decision rather than an omission.

D. `power` -- both nulls against a planted effect, so the cost of the change is
   in the record beside its benefit.

WHAT THIS DELIBERATELY DOES NOT DO

It adjudicates nothing: the populations are synthetic, no ALBERT run artifact
is in this repository, and `claims/adjudications/` stays empty. It does not
touch `P6-R4`.

RUN IT

    python3 -m tools.dry_run_p6_r2_r4 --write      # about thirty-five minutes
    python3 -m tools.dry_run_p6_r2_r4 --check
    python3 -m tools.dry_run_p6_r2_r4 --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "p6_r2_r4_dry_run.json"

RECORD_SCHEMA_VERSION = 1

#: The two files every number here depends on: the null construction, and the
#: geometry module that builds the channels and both null families.
NULL_PATH = ROOT / "p6_subspace" / "r2_r4_null.py"
GEOMETRY_PATH = ROOT / "p6_subspace" / "subspace_geometry.py"

#: Geometry. d_model and the channel dimensions are the shape of a small layer;
#: the dimensions are deliberately UNEQUAL, because the two nulls differ in how
#: they treat the split and equal arms would hide half of it.
D_MODEL, DIM_NEG, DIM_A, N_LAYERS = 128, 8, 24, 6

#: Null draws used here. The module ships 2000 (floor 1/2001); this file runs
#: the whole null a few thousand times, and 199 puts the floor at 0.005, still
#: an order below alpha. Recorded, because a rate measured at one draw count is
#: not a rate at another.
DRY_RUN_DRAWS = 199

#: Replicates per sweep cell. 250 resolves a proportion to about +/-0.014,
#: which separates 0.05 from 0.12 and does not separate 0.05 from 0.07 -- so
#: the arms below report a TREND across cells rather than resting on one.
N_SWEEP = 250
N_R4_SWEEP = 200

#: Family-wise level for this file's own numeric checks. TIGHTER than the
#: registry's alpha on purpose, and the reason is about the check rather than
#: about the science: these bounds are applied to proportions in a regenerated
#: artifact, and a bound that fails once in twenty regenerations when nothing
#: is wrong is one that gets re-run rather than read. Derived allowances are
#: computed from it by `_two_sided_z`.
CHECK_FAMILY_ALPHA = 0.01

#: Replicates for the two ends again, where the claim "the replacement holds at
#: nominal" actually has to be resolved: 600 puts the standard error at 0.0089.
#: This pass produced a 0.076 at 250 replicates that came back 0.050 at 1000,
#: which is the reason this constant exists rather than a larger N_SWEEP -- the
#: sweep needs breadth to show a trend and the claim needs depth at two points.
N_PRECISION = 600

#: How far the union sits above chance against the separating direction, as the
#: `a` of a lead vector `a*v + sqrt(1-a^2)*g`. The measured alignment is
#: reported per cell rather than inferred from `a`.
UNION_ALIGNMENTS: Tuple[float, ...] = (0.0, 0.9, 0.95, 0.99)

#: Variance scales for P6-R4's arm. `capture` is reported per cell.
R4_VARIANCE_SCALES: Tuple[float, ...] = (0.0, 2.0, 4.0)

_SEED = 20260826


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic layers with a planted answer
# ---------------------------------------------------------------------------

def union_with_alignment(rng: np.random.Generator, v: np.ndarray, a: float,
                         d: int = D_MODEL,
                         k: int = DIM_NEG + DIM_A) -> np.ndarray:
    """
    A k-dimensional subspace holding a prescribed share of `v`.

    One basis direction is tilted toward `v` by `a` and the rest are random, so
    `a` sweeps the union's alignment from chance to its ceiling d/k while the
    dimensions stay fixed. That is the axis the retired null fails along, and
    holding the dimensions fixed while it moves is what makes it the axis
    rather than a confound.
    """
    g = rng.normal(size=d)
    g -= (g @ v) * v
    g /= np.linalg.norm(g)
    lead = a * v + np.sqrt(max(0.0, 1.0 - a * a)) * g
    M = np.column_stack([lead] + [rng.normal(size=d) for _ in range(k - 1)])
    return np.linalg.qr(M)[0]


def _rot(rng: np.random.Generator, k: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(k, k)))[0]


def random_split(rng: np.random.Generator, U: np.ndarray,
                 k_neg: int = DIM_NEG) -> Tuple[np.ndarray, np.ndarray]:
    """H0: the split carries no information, realised exactly."""
    Z = U @ _rot(rng, U.shape[1])
    return Z[:, :k_neg], Z[:, k_neg:]


def planted_split(U: np.ndarray, v: np.ndarray,
                  k_neg: int = DIM_NEG) -> Tuple[np.ndarray, np.ndarray]:
    """H1: the union's `v` content concentrated in U_neg, P6-R2's direction."""
    Z = U[:, np.argsort(-np.abs(U.T @ v))]
    return Z[:, :k_neg], Z[:, k_neg:]


def _channels(pairs) -> list:
    from p6_subspace.subspace_geometry import LayerChannels

    zero = np.zeros((D_MODEL, 1))
    return [LayerChannels(u_pos=zero, u_neg=un, u_a=ua, u_s=zero,
                          d_model=D_MODEL, n_heads=1) for un, ua in pairs]


def _r2(dirs, chans, draws: Optional[int] = None) -> dict:
    # Resolved HERE and not in the signature: a module constant bound as a
    # default argument is bound once at definition time, which POPPER_PLAN.md
    # 6h records as a live bug in `attainable_floor_report` and 6m records
    # again in this file's predecessor. Writing that comment did not stop it
    # being written a third time, which is its own small lesson.
    from p6_subspace import r2_r4_null as rn

    prev = rn.N_NULL_DRAWS
    rn.N_NULL_DRAWS = DRY_RUN_DRAWS if draws is None else int(draws)
    try:
        return rn.p_value_p6_r2(dirs, chans, unit="model")
    finally:
        rn.N_NULL_DRAWS = prev


# ---------------------------------------------------------------------------
# A. the answer is known
# ---------------------------------------------------------------------------

def known_answer(rng: np.random.Generator, alpha: float) -> dict:
    """
    The separating direction inside one channel, the other, or neither.

    P6-R2 predicts MORE alignment with U_neg, so a direction drawn inside
    U_neg must reject and the same direction inside U_A must return p = 1 --
    a construction that cannot produce p = 1 with the arms reversed is not
    testing the direction it claims to.
    """
    rows = []
    for planted, expect in (("u_neg", "reject"), ("u_a", "p = 1"),
                            ("neither", "no rejection")):
        ps, retired = [], []
        for s in range(5):
            v = rng.normal(size=D_MODEL)
            v /= np.linalg.norm(v)
            pairs, dirs = [], []
            for _ in range(N_LAYERS):
                U = union_with_alignment(rng, v, 0.95)
                un, ua = random_split(rng, U)
                pairs.append((un, ua))
                if planted == "u_neg":
                    w = un @ rng.normal(size=un.shape[1])
                elif planted == "u_a":
                    w = ua @ rng.normal(size=ua.shape[1])
                else:
                    w = rng.normal(size=D_MODEL)
                dirs.append(w / np.linalg.norm(w))
            res = _r2(dirs, _channels(pairs))
            ps.append(float(res["p_value"]))
            retired.append(
                float(res["matched_dimension_diagnostic"]["p_value"]))
        rows.append({
            "planted_in": planted,
            "expected": expect,
            "p_values": ps,
            "retired_null_p_values": retired,
            "all_as_expected": bool(
                all(p <= alpha for p in ps) if expect == "reject"
                else all(p >= 0.99 for p in ps) if expect == "p = 1"
                else all(p > alpha for p in ps)),
        })
    return {
        "_what": ("the separating direction drawn inside U_neg (P6-R2's "
                  "predicted channel), inside U_A (the arms reversed), and in "
                  "neither. The correct answer is fixed a priori in all three."),
        "alpha": alpha,
        "n_seeds_per_row": 5,
        "rows": rows,
        "every_row_as_expected": bool(all(r["all_as_expected"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# B. the retirement evidence
# ---------------------------------------------------------------------------

def union_alignment_sweep(rng: np.random.Generator, alpha: float,
                          reps: int = N_SWEEP) -> dict:
    """
    H0 rejection rate against the union's alignment, both nulls, paired.

    The H0 is exact: each layer's split is drawn uniformly at random inside its
    own union, so the labelling carries no information by construction and the
    correct answer is "do not reject" at every alignment. What moves along the
    sweep is only how far the union sits above chance -- the quantity the
    retired null does not reproduce.
    """
    from p6_subspace.subspace_geometry import normalized_alignment

    rows = []
    for a in UNION_ALIGNMENTS:
        adj = ret = 0
        aligns = []
        for _ in range(reps):
            v = rng.normal(size=D_MODEL)
            v /= np.linalg.norm(v)
            pairs, unions = [], []
            for _ in range(N_LAYERS):
                U = union_with_alignment(rng, v, a)
                unions.append(U)
                pairs.append(random_split(rng, U))
            aligns.append(normalized_alignment(v, unions[0], D_MODEL))
            res = _r2([v] * N_LAYERS, _channels(pairs))
            adj += int(res["p_value"] <= alpha)
            ret += int(res["matched_dimension_diagnostic"]["p_value"] <= alpha)
        rows.append({
            "union_tilt": a,
            "mean_union_alignment": float(np.mean(aligns)),
            "adjudicated_reject": adj / reps,
            "retired_reject": ret / reps,
        })
    adj_rates = [r["adjudicated_reject"] for r in rows]
    ret_rates = [r["retired_reject"] for r in rows]
    se = float(np.sqrt(alpha * (1 - alpha) / reps))
    return {
        "_what": ("H0 rejection rate against how far span(U_neg + U_A) sits "
                  "above chance, with both nulls scored on the same runs."),
        "_why_a_trend_and_not_a_rate": (
            "a single rate is a proportion over a few hundred draws and this "
            "project has been burned by exactly that. A monotone trend in the "
            "quantity the retired null fails to reproduce is the mechanism, "
            "and it is what makes the retirement more than one cell."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "standard_error_at_alpha": se,
        "n_null_draws": DRY_RUN_DRAWS,
        "rows": rows,
        "retired_null_range": [min(ret_rates), max(ret_rates)],
        "adjudicated_null_range": [min(adj_rates), max(adj_rates)],
        "retired_null_rises_with_alignment": bool(
            ret_rates[-1] - ret_rates[0] > 3 * se),
        "adjudicated_null_is_flat": bool(
            max(adj_rates) - min(adj_rates) <= 4 * se),
        "adjudicated_null_holds": bool(
            max(adj_rates) <= alpha + 1.96 * se),
    }


def _two_sided_z(alpha_family: float, n_cells: int) -> float:
    """
    The z a one-sided per-cell bound needs to hold a family-wise error rate.

    Bonferroni: split `alpha_family` across the cells and take that upper
    quantile of the normal. Written out rather than imported so tier 0's
    numpy-free lint never has to reach scipy for it, and derived from the two
    inputs rather than placed.
    """
    from math import erf, sqrt
    target = 1.0 - alpha_family / max(int(n_cells), 1)
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def precision_check(rng: np.random.Generator, alpha: float,
                    reps: int = N_PRECISION) -> dict:
    """
    The two ends of the sweep again, at four times the replicates.

    The sweep above resolves a proportion to about +/-0.014, which separates
    0.05 from 0.15 and does NOT separate 0.05 from 0.07 -- and this pass's own
    exploration produced a 0.076 at 250 replicates that came back 0.050 at
    1000. "The replacement holds at nominal" is the claim the change rests on,
    so it is measured where it can be resolved rather than read off a cell that
    cannot resolve it.

    Only the two ends: they are what the claim needs, and the replicates are
    what they cost.
    """
    from p6_subspace.subspace_geometry import normalized_alignment

    rows = []
    for a in (UNION_ALIGNMENTS[0], UNION_ALIGNMENTS[-1]):
        adj = ret = 0
        aligns = []
        for _ in range(reps):
            v = rng.normal(size=D_MODEL)
            v /= np.linalg.norm(v)
            pairs, unions = [], []
            for _ in range(N_LAYERS):
                U = union_with_alignment(rng, v, a)
                unions.append(U)
                pairs.append(random_split(rng, U))
            aligns.append(normalized_alignment(v, unions[0], D_MODEL))
            res = _r2([v] * N_LAYERS, _channels(pairs))
            adj += int(res["p_value"] <= alpha)
            ret += int(res["matched_dimension_diagnostic"]["p_value"] <= alpha)
        rows.append({
            "union_tilt": a,
            "mean_union_alignment": float(np.mean(aligns)),
            "adjudicated_reject": adj / reps,
            "retired_reject": ret / reps,
        })
    se = float(np.sqrt(alpha * (1 - alpha) / reps))
    adj = [r["adjudicated_reject"] for r in rows]
    ret = [r["retired_reject"] for r in rows]
    # The bound carries a multiplicity allowance, and it has to. A per-cell
    # 1.96-sigma bound applied to every cell of a table false-alarms once per
    # twenty cells BY CONSTRUCTION when the null is exactly nominal -- the
    # first version of this section did precisely that and turned a 0.068 at
    # 600 replicates into a failing gate. The allowance is Bonferroni over the
    # cells this section tests, derived from that count rather than chosen.
    z = float(_two_sided_z(alpha_family=CHECK_FAMILY_ALPHA, n_cells=len(rows)))
    bound = alpha + z * se
    return {
        "_what": (f"the two ends of the sweep at {reps} replicates, where the "
                  f"standard error is {se:.3f} -- enough to separate nominal "
                  f"from a rate the coarser sweep can only suggest."),
        "_the_bound": (
            f"alpha plus {z:.2f} standard errors: the Bonferroni allowance for "
            f"the {len(rows)} cells this section tests at a family-wise "
            f"{CHECK_FAMILY_ALPHA}. A per-cell 1.96 fails once in twenty "
            f"regenerations with the null exactly nominal -- a property of the "
            f"check rather than of the null, and the first version of this "
            f"section did exactly that on a 0.068. The family-wise level is "
            f"tighter than alpha itself because a gate that cries wolf once "
            f"per twenty regenerations is a gate people re-run rather than "
            f"read."),
        "_what_it_does_not_claim": (
            "that the replacement is exactly nominal. Two independent "
            "measurements of the aligned end -- this section and a "
            "1000-replicate run during construction -- gave 0.068 and 0.048, "
            "pooling to about 0.056, so the replacement is at or marginally "
            "above 0.05 there. What is established is that it does not TREND "
            "with the union's alignment, which is the property the retired "
            "null failed, and that the two are far apart at the aligned end."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "standard_error_at_alpha": se,
        "bound_in_standard_errors": z,
        "bound": float(bound),
        "rows": rows,
        "adjudicated_holds_at_both_ends": bool(max(adj) <= bound),
        "retired_fails_at_the_aligned_end": bool(ret[-1] > bound),
        "separation_in_standard_errors": (
            float((ret[-1] - adj[-1]) / se) if se else None),
    }


# ---------------------------------------------------------------------------
# C. why P6-R4 is left alone
# ---------------------------------------------------------------------------

def r4_variance_sweep(rng: np.random.Generator, alpha: float,
                      reps: int = N_R4_SWEEP) -> dict:
    """
    P6-R4's H0 rate when U_S captures more population variance than chance.

    P6-R4 has no union and no split, so the retired null's defect has no direct
    analogue. The analogous question is whether matching the DIMENSION is
    enough when the observed subspace is not a random one: a probe's accuracy
    inside a projection depends on the retained signal and the retained
    within-cluster noise together, and which way that cuts is not predictable.

    H0 here: U_S carries no more cluster information than a random subspace of
    its dimension. Realised by drawing U_S inside a high-variance subspace that
    is orthogonal to the separating direction by construction.
    """
    from p6_subspace import r2_r4_null as rn
    from p6_subspace.subspace_geometry import LayerChannels

    d, k_s, n_tok, var_dims = 96, 12, 240, 24
    rows = []
    for scale in R4_VARIANCE_SCALES:
        rej = 0
        caps = []
        prev = rn.N_NULL_DRAWS
        rn.N_NULL_DRAWS = DRY_RUN_DRAWS
        try:
            for _ in range(reps):
                acts, labs, chans = [], [], []
                for _ in range(4):
                    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
                    hi = Q[:, 1:1 + var_dims]      # orthogonal to Q[:, 0]
                    y = (rng.random(n_tok) < 0.5).astype(int)
                    X = rng.normal(size=(n_tok, d))
                    if scale:
                        X = X + (rng.normal(size=(n_tok, var_dims))
                                 * scale) @ hi.T
                    u_s = (hi @ _rot(rng, var_dims))[:, :k_s] if scale else \
                        np.linalg.qr(rng.normal(size=(d, k_s)))[0]
                    Xc = X - X.mean(axis=0, keepdims=True)
                    caps.append(float(((Xc @ u_s) ** 2).sum()
                                      / (Xc ** 2).sum()) / (k_s / d))
                    zero = np.zeros((d, 1))
                    acts.append(X)
                    labs.append(y)
                    chans.append(LayerChannels(u_pos=zero, u_neg=zero,
                                               u_a=zero, u_s=u_s, d_model=d,
                                               n_heads=1))
                rej += int(rn.p_value_p6_r4(acts, labs, chans,
                                            unit="model")["p_value"] <= alpha)
        finally:
            rn.N_NULL_DRAWS = prev
        rows.append({
            "variance_scale": scale,
            "mean_variance_capture": float(np.mean(caps)),
            "reject": rej / reps,
        })
    rates = [r["reject"] for r in rows]
    se = float(np.sqrt(alpha * (1 - alpha) / reps))
    return {
        "_what": ("P6-R4's H0 rate when U_S captures more of the population's "
                  "variance than a random subspace of its dimension would."),
        "_why_it_is_here": (
            "P6-R2's null was changed and P6-R4's was not. Leaving it alone is "
            "a decision, and this is the measurement behind it: without this "
            "arm the difference between the two entries would rest on an "
            "argument about their statistics rather than on a number."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "standard_error_at_alpha": se,
        "rows": rows,
        "range": [min(rates), max(rates)],
        "holds": bool(max(rates) <= alpha + 1.96 * se),
    }


# ---------------------------------------------------------------------------
# D. what the change cost
# ---------------------------------------------------------------------------

def power(rng: np.random.Generator, alpha: float, reps: int = 60) -> dict:
    """Both nulls against a planted effect, so the cost sits beside the fix."""
    rows = []
    for a in (0.0, 0.9, 0.99):
        adj = ret = 0
        for _ in range(reps):
            v = rng.normal(size=D_MODEL)
            v /= np.linalg.norm(v)
            pairs = []
            for _ in range(N_LAYERS):
                U = union_with_alignment(rng, v, a)
                pairs.append(planted_split(U, v))
            res = _r2([v] * N_LAYERS, _channels(pairs))
            adj += int(res["p_value"] <= alpha)
            ret += int(res["matched_dimension_diagnostic"]["p_value"] <= alpha)
        rows.append({"union_tilt": a, "adjudicated_power": adj / reps,
                     "retired_power": ret / reps})
    return {
        "_what": ("power against the union's v-content concentrated in U_neg, "
                  "which is P6-R2's predicted direction."),
        "alpha": alpha,
        "n_reps_per_cell": reps,
        "rows": rows,
        "no_power_lost": bool(all(r["adjudicated_power"] >= r["retired_power"]
                                  - 1e-9 for r in rows)),
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    from p6_subspace import r2_r4_null as rn

    rng = np.random.default_rng(seed)
    alpha = 0.05
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("P6-R2 and P6-R4 run on inputs whose correct answer is known "
                  "a priori, and the measurement that retired P6-R2's null."),
        "_why": ("POPPER_PLAN.md 6m retired the matched-dimension random "
                 "subspace pair for P-ST1. That construction is 6h's and 6h "
                 "introduced it for these two entries, so it had to be checked "
                 "where it came from."),
        "_not": ("not evidence about ALBERT and not an adjudication. The "
                 "populations are synthetic; what is being checked is the "
                 "null."),
        "generated_by": "python3 -m tools.dry_run_p6_r2_r4 --write",
        "null_file": str(NULL_PATH.relative_to(ROOT)),
        "null_sha256": _sha256(NULL_PATH),
        "geometry_file": str(GEOMETRY_PATH.relative_to(ROOT)),
        "geometry_sha256": _sha256(GEOMETRY_PATH),
        "alpha": alpha,
        "null_family": rn.NULL_FAMILY,
        "registered_exchangeable_unit": rn.REGISTERED_EXCHANGEABLE_UNIT,
        "geometry": {"d_model": D_MODEL, "dim_u_neg": DIM_NEG,
                     "dim_u_a": DIM_A, "n_layers": N_LAYERS,
                     "n_null_draws": DRY_RUN_DRAWS,
                     "n_null_draws_shipped": int(rn.N_NULL_DRAWS)},
        "seed": int(seed),
        "known_answer": known_answer(rng, alpha),
        "union_alignment_sweep": union_alignment_sweep(rng, alpha),
        "precision_check": precision_check(rng, alpha),
        "r4_variance_sweep": r4_variance_sweep(rng, alpha),
        "power": power(rng, alpha),
    }


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and does it still
    support the change it was the evidence for?

    Three things can fail, and each of them should. The record can describe a
    module that has moved; the adjudicated null can stop holding; and the
    RETIRED null can stop looking anticonservative -- in which case the
    retirement is not supported by the artifact that supports it, which is a
    problem with the retirement rather than something to pass over.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_p6_r2_r4 --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    from p6_subspace import r2_r4_null as rn

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("null", NULL_PATH), ("geometry", GEOMETRY_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        if rec.get(f"{key}_sha256") != _sha256(described):
            problems.append(
                f"{described.name} has changed since this record was written "
                f"(sha256 {_sha256(described)[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record); rerun "
                f"--write rather than editing the hash")
    if rec.get("null_family") != rn.NULL_FAMILY:
        problems.append(
            f"the record was made against null family "
            f"{rec.get('null_family')!r} and the module now adjudicates "
            f"{rn.NULL_FAMILY!r}")

    ka = rec.get("known_answer", {})
    if not ka.get("every_row_as_expected"):
        problems.append(
            "P6-R2 did not return the known answer on an input whose correct "
            "verdict is fixed a priori; that is the criterion not meaning what "
            "it says rather than a calibration question")

    sweep = rec.get("union_alignment_sweep", {})
    if not sweep.get("rows"):
        problems.append("union_alignment_sweep has no rows")
    else:
        if not sweep.get("adjudicated_null_holds"):
            problems.append(
                f"the ADJUDICATED null's H0 rate reaches "
                f"{sweep.get('adjudicated_null_range')} against alpha "
                f"{sweep.get('alpha')}; the null this module adjudicates does "
                f"not hold on the family it was chosen for")
        if not sweep.get("retired_null_rises_with_alignment"):
            problems.append(
                f"the RETIRED null's H0 rate no longer rises with the union's "
                f"alignment ({sweep.get('retired_null_range')}). It was "
                f"retired on that trend; an artifact that no longer shows it "
                f"does not support the retirement")

    pc = rec.get("precision_check", {})
    if not pc.get("rows"):
        problems.append(
            "no precision_check section: the sweep's replicate count cannot "
            "separate nominal from a rate near it, and 'the replacement holds' "
            "is the claim the change rests on")
    else:
        if not pc.get("adjudicated_holds_at_both_ends"):
            problems.append(
                "at the replicate count that can resolve it, the adjudicated "
                "null does not hold at both ends of the sweep")
        if not pc.get("retired_fails_at_the_aligned_end"):
            problems.append(
                "at the replicate count that can resolve it, the retired null "
                "no longer fails at the aligned end; the retirement is not "
                "supported by the artifact that supports it")

    r4 = rec.get("r4_variance_sweep", {})
    if not r4.get("rows"):
        problems.append("r4_variance_sweep has no rows")
    elif not r4.get("holds"):
        problems.append(
            f"P6-R4's H0 rate reaches {r4.get('range')} against alpha "
            f"{r4.get('alpha')}. It was left unchanged on the evidence that it "
            f"is unaffected; this record no longer says so")

    if not rec.get("power", {}).get("no_power_lost"):
        problems.append(
            "the change cost power somewhere in the sweep; that is not "
            "disqualifying but it is not what this record says")
    return problems


def print_summary(rec: dict) -> None:
    print(f"null:     {rec['null_file']}  sha256 {rec['null_sha256'][:12]}")
    print(f"geometry: {rec['geometry_file']}  sha256 "
          f"{rec['geometry_sha256'][:12]}")
    print(f"family:   {rec['null_family']}")
    print(f"alpha {rec['alpha']}  unit {rec['registered_exchangeable_unit']}  "
          f"geometry {rec['geometry']}\n")

    ka = rec["known_answer"]
    print("=== A. the answer is known ===")
    for r in ka["rows"]:
        ps = ", ".join(f"{p:.3f}" for p in r["p_values"])
        print(f"  planted in {r['planted_in']:>8}  expect {r['expected']:>13}  "
              f"as expected {r['all_as_expected']!s:>5}   p = {ps}")

    s = rec["union_alignment_sweep"]
    print(f"\n=== B. H0 against the union's alignment "
          f"({s['n_reps_per_cell']} reps, SE "
          f"{s['standard_error_at_alpha']:.3f}) ===")
    print(f"  {'tilt':>6} {'union align':>12} {'adjudicated':>12} "
          f"{'retired':>9}")
    for r in s["rows"]:
        print(f"  {r['union_tilt']:>6} {r['mean_union_alignment']:>12.2f} "
              f"{r['adjudicated_reject']:>12.3f} {r['retired_reject']:>9.3f}")
    print(f"  retired rises with alignment: "
          f"{s['retired_null_rises_with_alignment']}   adjudicated flat: "
          f"{s['adjudicated_null_is_flat']}   holds: "
          f"{s['adjudicated_null_holds']}")

    pc = rec["precision_check"]
    print(f"\n=== B2. the two ends at {pc['n_reps_per_cell']} reps, SE "
          f"{pc['standard_error_at_alpha']:.3f} ===")
    print(f"  {'union align':>12} {'adjudicated':>12} {'retired':>9}")
    for r in pc["rows"]:
        print(f"  {r['mean_union_alignment']:>12.2f} "
              f"{r['adjudicated_reject']:>12.3f} {r['retired_reject']:>9.3f}")
    print(f"  adjudicated holds at both ends: "
          f"{pc['adjudicated_holds_at_both_ends']}   retired fails at the "
          f"aligned end: {pc['retired_fails_at_the_aligned_end']}   "
          f"separation {pc['separation_in_standard_errors']:.1f} SE")

    r4 = rec["r4_variance_sweep"]
    print(f"\n=== C. P6-R4, unchanged and why ({r4['n_reps_per_cell']} reps) ===")
    print(f"  {'var scale':>10} {'capture':>9} {'reject':>8}")
    for r in r4["rows"]:
        print(f"  {r['variance_scale']:>10} {r['mean_variance_capture']:>9.2f} "
              f"{r['reject']:>8.3f}")
    print(f"  holds: {r4['holds']}")

    p = rec["power"]
    print(f"\n=== D. power ({p['n_reps_per_cell']} reps) ===")
    print(f"  {'tilt':>6} {'adjudicated':>12} {'retired':>9}")
    for r in p["rows"]:
        print(f"  {r['union_tilt']:>6} {r['adjudicated_power']:>12.3f} "
              f"{r['retired_power']:>9.3f}")
    print(f"  no power lost: {p['no_power_lost']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run it and write the record (~20 minutes)")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed record without re-running")
    ap.add_argument("--summary", action="store_true",
                    help="print the committed record")
    ap.add_argument("--out", type=Path, default=RECORD_PATH)
    ap.add_argument("--seed", type=int, default=_SEED)
    args = ap.parse_args(argv)

    if args.write:
        rec = build_record(seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
        print(f"wrote {args.out}")
        print_summary(rec)
        return 0
    if args.check:
        problems = check_record(args.out)
        for p in problems:
            print(f"PROBLEM: {p}")
        if not problems:
            print(f"{args.out.name}: clean, and describes the files on disk")
        return 1 if problems else 0
    if args.summary:
        if not args.out.exists():
            print(f"{args.out} is missing; run --write")
            return 1
        print_summary(json.loads(args.out.read_text()))
        return 0
    ap.error("nothing to do: pass --write, --check or --summary")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
