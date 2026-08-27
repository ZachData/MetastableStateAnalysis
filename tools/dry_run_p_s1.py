"""
tools/dry_run_p_s1.py — P-S1 on inputs whose answer is known.

`claims/EVALUABILITY.md`'s queue owes every converted row a run on an input
whose correct verdict is fixed a priori. `P-S1` is the ninth and last of the
nine adjudicable rows, after `CLAIM-C`, `P-ST1`, `P6-R2`/`P6-R4`,
`CLAIM-B`/`P-I1` and `P-T1`/`P-M1`. The record is
`claims/audits/p_s1_dry_run.json`; `POPPER_PLAN.md` 6p reads it.

WHAT IT FOUND, AND IT IS THE LARGEST TYPE-I NUMBER IN THE REGISTRY

`p_value_p_s1` takes `m` and `d` from the TRAINED arm and draws its null
there. Both arms are then re-referenced against that one baseline -- which is
6d's fix and is right when the two arms sit at the same configuration. Until
2026-08-27 nothing checked that they did.

For i.i.d. points on the sphere E[Q_k] = 1/m exactly, so the baseline scales
like 1/m. A step-0 arm with a different cluster count is therefore divided by
a baseline that is not its own, and its ratio is off by roughly
m_trained/m_step0 -- which enters the statistic as a DIFFERENCE between the
arms and is indistinguishable from the effect P-S1 predicts. Measured on TWO
I.I.D. ARMS, where the correct verdict is "no difference" at every row:

    32 against 28 clusters    rejects at 1.000
    32 against 24 clusters    rejects at 1.000
    32 against 40 clusters    p = 1.000, so the design can never win

A four-cluster difference in thirty-two -- twelve percent -- turns a null
input into certain rejection, in the direction that CONFIRMS the prediction.
And unequal cluster counts are the expected case rather than the exception:
clustering runs per checkpoint, and a random-weight model's activation
geometry is not the trained one's.

WHAT THE FIX IS, AND WHY IT IS MORE THAN A GUARD

`p_value_p_s1` now refuses when the two arms report different (m, d), and
refuses when the step-0 arm does not report them at all. It is a degeneracy
and not a tolerance -- the counts are equal or they are not -- so no threshold
is placed.

It is also a statement about the statistic rather than about the code. Q_k's
i.i.d. floor depends on m, so "closer to a spherical design" is not a
comparison that exists ACROSS different m; there is no baseline choice that
rescues it. The requirement that puts on a run is that both arms be clustered
to the same count rather than each to its own best k -- the sixth
pre-computed requirement in six passes.

AND A SECOND THING, WHICH IS A CORRECTION RATHER THAN A DEFECT

The module warned that its `Q_ratio` fallback -- taken when raw `Q` is absent
-- leaves the p-value "mildly anticonservative", citing a null-p mean of 0.40.
That number was measured on the pre-2026-08-24 code. On the code that exists
the statistic is a DIFFERENCE of two ratios formed against the SAME caller
baseline, so a common per-degree factor cancels to first order; measured, the
two paths are indistinguishable over 120 replicates. The note stopped
describing the path it was attached to and nothing noticed, which is 6m's
lesson about inlined figures arriving for the second time.

FIVE ARMS

A. `known_answer`      -- the gate on inputs whose verdict is fixed a priori,
                          in both directions, at matched (m, d).
B. `mismatched_arms`   -- the finding: two I.I.D. arms at different cluster
                          counts, which is H0 by construction.
C. `refusal`           -- what the refusal turns away and what it costs.
D. `the_fallback_note` -- the two paths measured against each other, which is
                          what retires the 0.40 the note carried.
E. `floor_and_power`   -- whether a perfect input reaches the reported floor,
                          and what the design can do when it is used correctly.

WHAT THIS DELIBERATELY DOES NOT DO

It adjudicates nothing: the configurations are synthetic, no Phase 1c run
artifact is in this repository, and `claims/adjudications/` stays empty. It
does not touch the statistic, the null or the alternative.

RUN IT

    python3 -m tools.dry_run_p_s1 --write
    python3 -m tools.dry_run_p_s1 --check
    python3 -m tools.dry_run_p_s1 --summary

The generation cost is measured on every write and stored as `elapsed_seconds`
rather than quoted here, where it would go stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from p1c_frames.centroids import (
    P_S1_ALTERNATIVE,
    P_S1_T_MAX,
    _standardised_improvement,
    p_value_p_s1,
    random_band,
)
from p1c_frames.design_test import design_report, gegenbauer_moments, random_baseline_Q

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "p_s1_dry_run.json"

RECORD_SCHEMA_VERSION = 1

#: The two files every number here depends on: the gate, and the moment
#: machinery whose i.i.d. baseline is the quantity the finding is about.
GATE_PATH = ROOT / "p1c_frames" / "centroids.py"
DESIGN_PATH = ROOT / "p1c_frames" / "design_test.py"

ALPHA = 0.05

#: Geometry. d is the residual dimension and is the same for both arms by
#: construction; m is the CLUSTER COUNT, which is not, and is the axis this
#: pass is about.
D_MODEL = 256
M_TRAINED = 32

#: Cluster counts for the step-0 arm, against a trained arm at M_TRAINED. The
#: range is deliberately narrow: the finding is that a FEW clusters' difference
#: is enough, not that a wild mismatch breaks it.
M_STEP0: Tuple[int, ...] = (32, 30, 28, 24, 36, 40)

#: Replicates. 80 resolves a proportion to about +/-0.024, which separates
#: 0.05 from 1.000 several times over -- this arm does not need precision, it
#: needs the two ends.
N_REPS = 80
N_NULL = 120

_SEED = 20260827


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic configurations
# ---------------------------------------------------------------------------

def iid_arm(m: int, rng, d: int = D_MODEL) -> dict:
    """An i.i.d. uniform configuration of m unit vectors -- P-S1's own null."""
    X = rng.normal(size=(m, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return design_report(X, d=d, t_max=P_S1_T_MAX)


def spread_arm(m: int, rng, d: int = D_MODEL) -> dict:
    """
    A configuration that is genuinely better spread than i.i.d.: a random
    orthonormal frame padded with its negatives, which drives the low-order
    Gegenbauer moments down. This is P-S1's H1 realised directly.
    """
    k = min(m // 2, d)
    Q = np.linalg.qr(rng.normal(size=(d, k)))[0].T
    X = np.vstack([Q, -Q])[:m]
    if X.shape[0] < m:
        pad = rng.normal(size=(m - X.shape[0], d))
        pad /= np.linalg.norm(pad, axis=1, keepdims=True)
        X = np.vstack([X, pad])
    return design_report(X, d=d, t_max=P_S1_T_MAX)


def _strip_raw_Q(arm: dict) -> dict:
    return {k: v for k, v in arm.items() if k != "Q"}


# ---------------------------------------------------------------------------
# The gate's arithmetic WITHOUT the refusal
# ---------------------------------------------------------------------------
#
# Arm B has to reach a state the module now refuses, so it reproduces the
# calls the module makes and skips only the (m, d) check. That is a second
# implementation of the gate's arithmetic, which POPPER_PLAN.md 6g records as
# a real risk on CLAIM-C's fast path -- so `module_agreement` below pins it
# against `p_value_p_s1` itself on every MATCHED pair, where the module still
# emits and the two must agree exactly.

def scored_without_the_refusal(trained: dict, step0: dict, m: int, d: int,
                               n_null: int, seed: int) -> float:
    """`p_value_p_s1`'s p, computed the way it was before the (m, d) check."""
    from core.nulls import p_from_null

    band = random_band(m, d, t_max=P_S1_T_MAX, n_trials=n_null, seed=seed)
    base = random_baseline_Q(m, d, t_max=P_S1_T_MAX, n_trials=n_null, seed=seed)
    ref = np.maximum(base["mean"], 1e-30)
    r_trained = np.asarray(trained["Q"], dtype=np.float64)[:P_S1_T_MAX] / ref[:P_S1_T_MAX]
    r_step0 = np.asarray(step0["Q"], dtype=np.float64)[:P_S1_T_MAX] / ref[:P_S1_T_MAX]
    observed = _standardised_improvement(r_trained, r_step0, band["sd"])

    rng = np.random.default_rng(seed + 977)

    def _ratio() -> np.ndarray:
        X = rng.normal(size=(m, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = gegenbauer_moments(X @ X.T, t_max=P_S1_T_MAX, d=d, check=False)["Q"]
        return Q / ref

    null_stats = np.array([
        _standardised_improvement(_ratio(), _ratio(), band["sd"])
        for _ in range(n_null)])
    return float(p_from_null(observed, null_stats,
                             alternative=P_S1_ALTERNATIVE)["p_value"])


# ---------------------------------------------------------------------------
# A. the answer is known
# ---------------------------------------------------------------------------

def known_answer(rng, alpha: float, reps: int = 30) -> dict:
    """
    The gate at MATCHED (m, d) on inputs whose verdict is fixed a priori.

    Matched, because arm B is about to show that unmatched is where the gate
    goes wrong; asking a gate that now refuses whether it returns the right
    verdict would not be a check.
    """
    rows = []
    for label, make_t, make_s, expect in (
            ("trained better spread than step 0", spread_arm, iid_arm, "reject"),
            ("step 0 better spread than trained", iid_arm, spread_arm, "p high"),
            ("both arms i.i.d. (H0)", iid_arm, iid_arm, "no rejection"),
    ):
        ps = []
        for i in range(reps):
            t = make_t(M_TRAINED, rng)
            s = make_s(M_TRAINED, rng)
            ps.append(p_value_p_s1(t, s, n_null=N_NULL, seed=i)["p_value"])
        a = np.asarray([p for p in ps if p is not None], dtype=float)
        rate = float(np.mean(a <= alpha)) if a.size else float("nan")
        rows.append({
            "input": label, "expected": expect,
            "n_reps": reps, "reject_rate": rate,
            "median_p": float(np.median(a)) if a.size else None,
            # H0 and the reversed arm are RATES, not identities: a one-sided
            # test rejects at alpha under H0 by design, so demanding zero
            # rejections would be an assertion about the draw rather than
            # about the gate. POPPER_PLAN.md 6m and 6o both record the same
            # correction being needed.
            "as_expected": bool(
                rate >= 0.8 if expect == "reject"
                else (np.median(a) > 0.5 if a.size else False) if expect == "p high"
                else rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)),
        })
    return {
        "_what": ("the gate at MATCHED (m, d) on inputs whose correct verdict "
                  "is fixed a priori, in both directions."),
        "_why_matched": (
            "arm B shows the gate is wrong at UNMATCHED (m, d) and the module "
            "now refuses there. Asking a gate that refuses whether it returns "
            "the right verdict is not a check."),
        "alpha": alpha,
        "m": M_TRAINED, "d": D_MODEL,
        "rows": rows,
        "every_row_as_expected": bool(all(r["as_expected"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# B. the finding
# ---------------------------------------------------------------------------

def mismatched_arms(rng, alpha: float, reps: Optional[int] = None) -> dict:
    """
    Two I.I.D. arms at different cluster counts.

    Both arms are i.i.d. uniform, so the correct verdict is "no difference" at
    every row -- this is P-S1's own H0, realised exactly. The only thing that
    varies is the step-0 arm's cluster count.

    Scored WITHOUT the refusal this pass added, which is what the gate did
    before it: the module is called with `m` and `d` supplied explicitly, which
    is the path a caller took when both arms came from `design_report` and
    nothing compared them.
    """
    n = N_REPS if reps is None else int(reps)   # resolved here, not in the
    rows = []                                   # signature; see 6h/6m/6n
    agree_checked, agree_worst = 0, 0.0
    for m0 in M_STEP0:
        ps = []
        for i in range(n):
            t = iid_arm(M_TRAINED, rng)
            s = iid_arm(m0, rng)
            p_old = scored_without_the_refusal(t, s, M_TRAINED, D_MODEL,
                                               N_NULL, i)
            ps.append(p_old)
            if m0 == M_TRAINED:
                # Where the module still emits, this file's arithmetic and the
                # module's must agree exactly, or the rates above are about a
                # second implementation rather than about the gate.
                p_mod = p_value_p_s1(t, s, n_null=N_NULL, seed=i)["p_value"]
                if p_mod is not None:
                    agree_worst = max(agree_worst, abs(p_mod - p_old))
                    agree_checked += 1
        a = np.asarray([p for p in ps if p is not None], dtype=float)
        rows.append({
            "m_trained": M_TRAINED, "m_step0": m0,
            "cluster_difference": m0 - M_TRAINED,
            "relative_difference": (m0 - M_TRAINED) / M_TRAINED,
            "reject_rate": float(np.mean(a <= alpha)),
            "mean_p": float(a.mean()),
            "n_reps": n,
        })
    matched = [r for r in rows if r["m_step0"] == M_TRAINED][0]
    off = [r for r in rows if r["m_step0"] != M_TRAINED]
    return {
        "_what": ("two I.I.D. arms at different cluster counts. Both arms are "
                  "i.i.d. uniform, so the correct verdict is 'no difference' "
                  "at every row -- this is P-S1's own H0, realised exactly."),
        "_the_mechanism": (
            "E[Q_k] = 1/m for i.i.d. points, so the i.i.d. baseline scales "
            "like 1/m. The null is drawn at the TRAINED arm's (m, d) and both "
            "arms are re-referenced against it, so a step-0 arm at a different "
            "count is divided by a baseline that is not its own and its ratio "
            "is off by roughly m_trained/m_step0. That enters the statistic as "
            "a difference BETWEEN the arms, which is exactly the shape of the "
            "effect P-S1 predicts."),
        "_the_direction_matters": (
            "fewer step-0 clusters inflates the step-0 ratio and the statistic "
            "is (step0 - trained), so the error CONFIRMS the prediction. More "
            "step-0 clusters sends p to 1.000 and the design can never win. "
            "Neither is a wrong answer the analyst would notice."),
        "_scored_without_the_refusal": (
            "the module now refuses every one of these rows, so this arm "
            "reproduces the calls it makes and skips only the (m, d) check. "
            "These are the rates the gate had before this pass. The second "
            "implementation is pinned against the module on every MATCHED "
            "pair, where the module still emits -- see module_agreement."),
        "alpha": alpha,
        "n_reps_per_cell": n,
        "d": D_MODEL,
        "rows": rows,
        "matched_reject_rate": matched["reject_rate"],
        "worst_mismatched_reject_rate": max(r["reject_rate"] for r in off),
        "smallest_difference_that_breaks_it": min(
            (abs(r["cluster_difference"]) for r in off
             if r["reject_rate"] >= 0.5), default=None),
        "a_mismatch_can_also_make_it_unwinnable": bool(
            any(r["mean_p"] >= 0.99 for r in off)),
        "module_agreement": {
            "_what": ("this file's re-implementation against p_value_p_s1 "
                      "itself, on every matched pair where the module emits."),
            "n_compared": agree_checked,
            "max_absolute_difference": float(agree_worst),
            "agrees": bool(agree_checked > 0 and agree_worst == 0.0),
        },
    }


# ---------------------------------------------------------------------------
# C. the refusal
# ---------------------------------------------------------------------------

def refusal(rng, alpha: float, reps: int = 20) -> dict:
    """
    What the refusal turns away, and what it costs.

    It costs nothing measurable and the reason is not a measurement: at
    unmatched (m, d) there is no correct p for it to have removed. Q_k's
    i.i.d. floor depends on m, so "closer to a spherical design" is not a
    comparison that exists across different m -- there is no baseline choice
    that rescues the row, which is why this is a refusal rather than a
    correction.
    """
    rows = []
    for m0 in M_STEP0:
        refused = 0
        for i in range(reps):
            t = iid_arm(M_TRAINED, rng)
            s = iid_arm(m0, rng)
            r = p_value_p_s1(t, s, n_null=N_NULL, seed=i)   # no m=/d= override
            refused += int(r["p_value"] is None)
        rows.append({"m_step0": m0, "matched": m0 == M_TRAINED,
                     "refused_rate": refused / reps, "n_reps": reps})

    # The unverifiable case: an arm that does not say what configuration it
    # sits at cannot be checked against the one the null is drawn at.
    t = iid_arm(M_TRAINED, rng)
    s = iid_arm(M_TRAINED, rng)
    s_no_m = {k: v for k, v in s.items() if k != "n_centroids"}
    unverifiable = p_value_p_s1(t, s_no_m, n_null=N_NULL, seed=0)

    return {
        "_what": "which inputs the (m, d) refusal turns away.",
        "_what_it_costs": (
            "nothing, and by construction rather than by measurement. At "
            "unmatched (m, d) there is no correct p that the refusal removed: "
            "Q_k's i.i.d. floor depends on m, so the comparison the statistic "
            "makes does not exist across different m. That is a different kind "
            "of zero from CLAIM-C's informative-row refusal, which had to be "
            "re-scored, and the same kind as P-ST1's 2m bound."),
        "alpha": alpha,
        "rows": rows,
        "refuses_every_mismatch": bool(
            all(r["refused_rate"] == 1.0 for r in rows if not r["matched"])),
        "refuses_no_matched_arm": bool(
            all(r["refused_rate"] == 0.0 for r in rows if r["matched"])),
        "unverifiable_arm_refused": bool(unverifiable["p_value"] is None),
        "unverifiable_reason": unverifiable.get("reason"),
    }


# ---------------------------------------------------------------------------
# D. the note that stopped describing its own path
# ---------------------------------------------------------------------------

def the_fallback_note(rng, alpha: float, reps: int = 120) -> dict:
    """
    The raw-Q path against the `Q_ratio` fallback, on the same draws.

    The module warned that the fallback leaves the p-value "mildly
    anticonservative" and cited a null-p mean of 0.40 against 0.50. That was
    measured on the pre-2026-08-24 code. On the code that exists the statistic
    is a DIFFERENCE of two ratios formed against the SAME caller baseline, so
    a common per-degree factor cancels to first order and what is left is a
    rescaling of about a percent.

    Paired: both paths are scored on the same pair of arms, so the comparison
    is not two experiments.
    """
    fixed, fell_back = [], []
    for i in range(reps):
        t = iid_arm(M_TRAINED, rng)
        s = iid_arm(M_TRAINED, rng)
        fixed.append(p_value_p_s1(t, s, n_null=N_NULL, seed=i)["p_value"])
        fell_back.append(p_value_p_s1(_strip_raw_Q(t), _strip_raw_Q(s),
                                      n_null=N_NULL, seed=i)["p_value"])
    a = np.asarray(fixed, dtype=float)
    b = np.asarray(fell_back, dtype=float)
    return {
        "_what": ("the raw-Q path against the Q_ratio fallback on the SAME "
                  "draws, both under H0 (two i.i.d. arms at matched (m, d))."),
        "_why_it_is_here": (
            "the module's note quoted a null-p mean of 0.40 for the fallback. "
            "That number was measured on retired code and stopped describing "
            "the path it was attached to; nothing noticed. POPPER_PLAN.md 6m "
            "records the same pattern for three rates in P-ST1's docstring, "
            "and this is the second arrival."),
        "alpha": alpha,
        "n_reps": reps,
        "raw_Q_path": {"reject_rate": float(np.mean(a <= alpha)),
                       "mean_p": float(a.mean())},
        "fallback_path": {"reject_rate": float(np.mean(b <= alpha)),
                          "mean_p": float(b.mean())},
        "max_absolute_p_difference": float(np.max(np.abs(a - b))),
        "the_two_paths_are_indistinguishable": bool(
            abs(a.mean() - b.mean()) < 0.05
            and abs(np.mean(a <= alpha) - np.mean(b <= alpha)) < 0.05),
        "the_retired_number_was": 0.40,
        "and_neither_path_is_near_it": bool(
            min(a.mean(), b.mean()) > 0.42),
    }


# ---------------------------------------------------------------------------
# E. the floor, and what the design can do used correctly
# ---------------------------------------------------------------------------

def floor_and_power(rng, alpha: float, reps: int = 30) -> dict:
    """
    Does a perfect input reach the reported floor, and does the design work?

    P-S1's statistic is CONTINUOUS -- a sum of standardised ratio differences
    over Gaussian draws -- so ties with the observation have probability zero
    and 1/(n_null + 1) really is attainable. That is worth measuring rather
    than assuming, because it is exactly the claim that failed for `P-ST1`
    (6m), `P-T1` and `P-M1` (this pass), all of which have discrete statistics.
    """
    hits, ps = 0, []
    for i in range(reps):
        t = spread_arm(M_TRAINED, rng)
        s = iid_arm(M_TRAINED, rng)
        r = p_value_p_s1(t, s, n_null=N_NULL, seed=i)
        ps.append(r["p_value"])
        hits += int(bool(r.get("at_resolution_floor")))
    a = np.asarray(ps, dtype=float)
    return {
        "_what": ("whether a strongly-spread trained arm reaches the reported "
                  "1/(n_null+1) floor, and at what rate the design rejects "
                  "when it is used at matched (m, d)."),
        "_why_it_matters_here": (
            "the reported floor being the attainable one is the claim that "
            "failed for P-ST1, P-T1 and P-M1 -- all discrete statistics. "
            "P-S1's is continuous, so it should hold, and that is a thing to "
            "check rather than to assume."),
        "alpha": alpha,
        "n_reps": reps,
        "n_null": N_NULL,
        "reported_floor": 1.0 / (N_NULL + 1.0),
        "reached_the_floor_rate": hits / reps,
        "power_at_alpha": float(np.mean(a <= alpha)),
        "median_p": float(np.median(a)),
        "the_floor_is_attainable": bool(hits > 0),
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rec = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("P-S1 run on inputs whose correct answer is known a priori, "
                  "and the measurement that gave it an (m, d) refusal."),
        "_why": ("EVALUABILITY.md's queue owes every converted row a run on an "
                 "input whose verdict is fixed in advance. P-S1 is the ninth "
                 "and last of the nine."),
        "_not": ("not evidence about any model and not an adjudication. The "
                 "configurations are synthetic; what is being checked is the "
                 "gate."),
        "generated_by": "python3 -m tools.dry_run_p_s1 --write",
        "gate_file": str(GATE_PATH.relative_to(ROOT)),
        "gate_sha256": _sha256(GATE_PATH),
        "design_file": str(DESIGN_PATH.relative_to(ROOT)),
        "design_sha256": _sha256(DESIGN_PATH),
        "alpha": ALPHA,
        "alternative": P_S1_ALTERNATIVE,
        "t_max": P_S1_T_MAX,
        "geometry": {"d": D_MODEL, "m_trained": M_TRAINED,
                     "m_step0_swept": list(M_STEP0), "n_null": N_NULL},
        "seed": int(seed),
        "known_answer": known_answer(rng, ALPHA),
        "mismatched_arms": mismatched_arms(rng, ALPHA),
        "refusal": refusal(rng, ALPHA),
        "the_fallback_note": the_fallback_note(rng, ALPHA),
        "floor_and_power": floor_and_power(rng, ALPHA),
    }
    rec["elapsed_seconds"] = round(time.time() - t0, 1)
    return rec


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and does it still
    support the change it was the evidence for?
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_p_s1 --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("gate", GATE_PATH), ("design", DESIGN_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        if rec.get(f"{key}_sha256") != _sha256(described):
            problems.append(
                f"{described.name} has changed since this record was written "
                f"(sha256 {_sha256(described)[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record); rerun "
                f"--write rather than editing the hash")

    ka = rec.get("known_answer", {})
    if not ka.get("every_row_as_expected"):
        problems.append(
            "the gate did not return the known answer at matched (m, d) on an "
            "input whose correct verdict is fixed a priori; that is a "
            "criterion not meaning what it says rather than a calibration "
            "question")

    ma = rec.get("mismatched_arms", {})
    agr = ma.get("module_agreement", {})
    if not agr.get("agrees"):
        problems.append(
            f"this file's re-implementation of the gate disagrees with the "
            f"module by {agr.get('max_absolute_difference')} over "
            f"{agr.get('n_compared')} matched comparisons; the rates in "
            f"mismatched_arms are then about a second implementation rather "
            f"than about the gate")
    if not ma.get("rows"):
        problems.append("mismatched_arms has no rows")
    else:
        worst = ma.get("worst_mismatched_reject_rate")
        if worst is None or worst < 0.5:
            problems.append(
                f"a mismatched pair of I.I.D. arms no longer rejects at a high "
                f"rate (worst {worst}). The (m, d) refusal added on 2026-08-27 "
                f"rests on that number; an artifact that no longer shows it "
                f"does not support the change it is the evidence for")
        matched = ma.get("matched_reject_rate")
        if matched is None or matched > 0.15:
            problems.append(
                f"the MATCHED row rejects at {matched}, so this table no "
                f"longer separates the mismatch from the gate itself and the "
                f"finding it records is not the one it measured")
        if not ma.get("a_mismatch_can_also_make_it_unwinnable"):
            problems.append(
                "no row shows a mismatch driving p to 1. The finding is that "
                "the error runs in BOTH directions, and half of it is missing")

    rf = rec.get("refusal", {})
    if not rf.get("refuses_every_mismatch"):
        problems.append("the refusal no longer turns away every mismatched pair")
    if not rf.get("refuses_no_matched_arm"):
        problems.append(
            "the refusal turns away a MATCHED pair; a refusal that fires on "
            "everything is not a check (POPPER_PLAN.md 6h)")
    if not rf.get("unverifiable_arm_refused"):
        problems.append(
            "an arm that does not report its own (m, d) is no longer refused, "
            "so a mismatch it cannot be checked for would be scored")

    fn = rec.get("the_fallback_note", {})
    if not fn.get("the_two_paths_are_indistinguishable"):
        problems.append(
            "the raw-Q path and the Q_ratio fallback are no longer "
            "indistinguishable, so the note this pass rewrote should be "
            "rewritten again rather than left")
    if not fn.get("and_neither_path_is_near_it"):
        problems.append(
            "a path's null-p mean has drifted toward the 0.40 the retired note "
            "quoted; that number was retired on this measurement")

    fp = rec.get("floor_and_power", {})
    if not fp.get("the_floor_is_attainable"):
        problems.append(
            "no draw reached the reported 1/(n_null+1) floor. P-S1's statistic "
            "is continuous and the floor should be attainable -- if it is not, "
            "P-S1 has the defect P-T1, P-M1 and P-ST1 had")
    return problems


def print_summary(rec: dict) -> None:
    print(f"gate:   {rec['gate_file']}  sha256 {rec['gate_sha256'][:12]}")
    print(f"design: {rec['design_file']}  sha256 {rec['design_sha256'][:12]}")
    print(f"alpha {rec['alpha']}  alternative {rec['alternative']}  "
          f"t_max {rec['t_max']}  generated in {rec.get('elapsed_seconds')}s")
    print(f"geometry {rec['geometry']}\n")

    ka = rec["known_answer"]
    print(f"=== A. the answer is known, at MATCHED (m, d) ===")
    for r in ka["rows"]:
        print(f"  {r['input']:36s} expect {r['expected']:14s} "
              f"reject {r['reject_rate']:.3f}  median p "
              f"{r['median_p']:.3f}  ok {r['as_expected']}")

    ma = rec["mismatched_arms"]
    print(f"\n=== B. TWO I.I.D. ARMS at different cluster counts "
          f"({ma['n_reps_per_cell']} reps; correct verdict is 'no difference' "
          f"at every row) ===")
    print(f"  {'m trained':>10} {'m step0':>8} {'difference':>11} "
          f"{'reject':>8} {'mean p':>8}")
    for r in ma["rows"]:
        print(f"  {r['m_trained']:>10} {r['m_step0']:>8} "
              f"{r['cluster_difference']:>+11d} {r['reject_rate']:>8.3f} "
              f"{r['mean_p']:>8.3f}")
    agr = ma["module_agreement"]
    print(f"  re-implementation vs the module: {agr['n_compared']} matched "
          f"comparisons, worst difference {agr['max_absolute_difference']}, "
          f"agrees {agr['agrees']}")
    print(f"  matched {ma['matched_reject_rate']:.3f}   worst mismatched "
          f"{ma['worst_mismatched_reject_rate']:.3f}   smallest difference "
          f"that breaks it: {ma['smallest_difference_that_breaks_it']}")

    rf = rec["refusal"]
    print(f"\n=== C. what the refusal turns away ===")
    for r in rf["rows"]:
        print(f"  m_step0 {r['m_step0']:>3}  matched {str(r['matched']):>5}  "
              f"refused {r['refused_rate']:.2f}")
    print(f"  refuses every mismatch: {rf['refuses_every_mismatch']}   "
          f"refuses no matched arm: {rf['refuses_no_matched_arm']}   "
          f"unverifiable arm refused: {rf['unverifiable_arm_refused']}")

    fn = rec["the_fallback_note"]
    print(f"\n=== D. the note that stopped describing its own path "
          f"({fn['n_reps']} paired draws) ===")
    print(f"  raw Q   reject {fn['raw_Q_path']['reject_rate']:.3f}  "
          f"mean p {fn['raw_Q_path']['mean_p']:.3f}")
    print(f"  fallback reject {fn['fallback_path']['reject_rate']:.3f}  "
          f"mean p {fn['fallback_path']['mean_p']:.3f}")
    print(f"  indistinguishable: {fn['the_two_paths_are_indistinguishable']}   "
          f"the retired note said {fn['the_retired_number_was']}")

    fp = rec["floor_and_power"]
    print(f"\n=== E. the floor and the power ({fp['n_reps']} reps) ===")
    print(f"  reported floor {fp['reported_floor']:.5f}  reached on "
          f"{fp['reached_the_floor_rate']:.2f} of draws  power "
          f"{fp['power_at_alpha']:.3f}  median p {fp['median_p']:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run it and write the record; it prints and stores "
                         "its own elapsed time")
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
