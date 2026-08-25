"""
tools/calibrate_steering_sign.py — what P-ST1's construction does, measured.

`p7_motifs/steering_gate.py` is the construction. This is the evidence that it
is the one that works, and the evidence behind the four decisions inside it.
Generated offline and committed to `claims/calibration/steering_sign.json`,
same arrangement as `claims/calibration/changepoint_colocation.json`
(`docs/CI_BASELINE.md`): the gate reads none of it at runtime, so a lost file
breaks nothing — what it would lose is the reason the gate is built this way,
which is the thing this project keeps finding it cannot reconstruct later.

The whole intervention is exact linear algebra -- adding alpha*v to every row
of an (n_tokens, d) matrix and recomputing effective rank -- so unlike every
other entry in the registry, P-ST1's statistic can be exercised end to end on
synthetic populations with a planted answer and no model at all. That is why
this file can measure what it measures.

SEVEN THINGS IT RECORDS, and four of them decided the module's constants.

1. `attainable_floor`. Under the REGISTERED null the floor is
   (2^(m-k) + 1)/(2^m + 1) over the k INFORMATIVE pairs, not 2/(2^m + 1) over
   the m drawn. Tabulated so the number in the gate's diagnostic is checkable.
   The adjudicated null has no such floor -- see 6.

2. `sign_symmetry` -- decided ER_MODE. With the baseline mean removed the
   centred population has zero mean, the first-order Gram term vanishes, and
   dER is even in v. Row-normalization is not linear and reintroduces an odd
   term. Measured: raw agrees with itself under v -> -v in 60 of 60 draws at
   every alpha; normed falls to 0.00 at small alpha and manufactures D = -2 in
   20-22% of pairs there. The two are indistinguishable at the working alpha,
   which is exactly how the wrong one nearly shipped.

3. `mean_offset` -- decided DEBIAS_BASELINE_MEAN, put to the author before the
   module was written. Steering is a pure mean effect, so the cloud's own mean
   offset competes with the injected one. Each configuration is measured at ITS
   OWN best alpha, because the readable window moves when the baseline mean is
   left in and comparing them at one shared alpha would flatter whichever
   choice set it.

4. `alpha_window` -- decided ALPHA_SPREAD_FRACTION, and shows what a `placed`
   constant is placed against. There is a PLATEAU at 0.17-0.24 x spread where
   the per-pair rate is 1.000, with sharp edges either side, and its location
   in units of the population's own spread is stable across mean offsets --
   which is the whole reason alpha multiplies a derived scale. The first value
   written into the module was 0.1, chosen on a grid of (0.03, 0.1, 0.3) that
   could not see the plateau because both neighbours read zero.

5. `dimension_cliff` -- not a decision, a precondition. The informative rate
   falls as dim U_pos exceeds the dimension the population occupies.

6. `registered_null_inflation` -- the reason the registered null is NOT the one
   adjudicated. Permuting the decomposition label across pairs is
   anticonservative and the inflation GROWS WITH THE PAIR COUNT (0.037 at 8
   pairs, 0.051 at 24, 0.143 at 40, 0.172 at 150, against a nominal 0.05),
   because every pair at one layer shares the tokens and both subspaces and
   more pairs shrink the null's spread while leaving the shared tilt untouched.
   Invisible in the clean regime, where the gate refuses and the unconditional
   rate reads 0.000 -- calibrated BY REFUSING, which is the conditioning error
   POPPER_PLAN.md 6g records for CLAIM-C.

7. `adjudicated_null_validity_and_power` -- the replacement, measured on the
   same draws so the comparison is paired. Randomising over matched-dimension
   random ORTHOGONAL subspace pairs is POPPER_PLAN.md 6h's construction
   arriving for the fourth time, and it removes the informative-pair floor as
   well as the inflation.

RUN IT

    python3 -m tools.calibrate_steering_sign --write     # about 12 minutes
    python3 -m tools.calibrate_steering_sign --check
    python3 -m tools.calibrate_steering_sign --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "calibration" / "steering_sign.json"
GATE_PATH = ROOT / "p7_motifs" / "steering_gate.py"

RECORD_SCHEMA_VERSION = 1

#: Synthetic layer geometry. d and n_tokens are the shape of a small real
#: layer's activation block; kept fixed across every arm so the arms are
#: comparable to each other rather than each to its own baseline.
D_MODEL = 160
N_TOKENS = 64

#: Dimension the population actually occupies. Every `dim U_pos` in the
#: dimension cliff is read as a ratio against this.
DIM_OCCUPIED = 16

#: How strongly the cloud concentrates into the occupied subspace, relative to
#: isotropic noise. It is a swept parameter and not a detail: at high
#: concentration the two arms separate cleanly and every pair under H0 is
#: uninformative, so the gate REFUSES and its Type-I rate is 0 by refusal
#: rather than by control. Weak concentration is where effective-rank changes
#: are near-tied, spurious informative pairs appear symmetrically, and the
#: label-permutation null has to do real work. Measuring validity only at the
#: clean end would let the gate look calibrated BY REFUSING -- the conditioning
#: error POPPER_PLAN.md 6g records for CLAIM-C.
CONCENTRATION = 3.0
H0_EMITTING_CONCENTRATIONS: Tuple[float, ...] = (0.0, 0.3, 0.6)

#: Replicate counts. The per-pair rates are proportions over N_PAIR_TRIALS
#: draws; the null-validity rates are over N_GATE_TRIALS whole gate runs, which
#: is the expensive one.
N_PAIR_TRIALS = 200

#: Whole-gate runs per row. The subspace null costs n_draws * n_pairs effective
#: ranks per run, which is what keeps this number small and the pair counts
#: modest -- and modest pair counts are what the design wants anyway, since at
#: the plateau alpha the informative rate is near 1 and eight pairs already
#: saturate the null.
N_GATE_TRIALS = 50
N_PAIRS_SWEPT: Tuple[int, ...] = (8, 24)

#: Trials per cell of the mean-offset sweep, which is a full alpha grid per
#: configuration and so costs 13x a single-alpha row.
N_MEAN_OFFSET_TRIALS = 100

#: Null draws used HERE. The module ships 199 (floor 0.005); the calibration
#: uses fewer because what it measures is a rejection rate at alpha = 0.05 and
#: a floor of 0.01 is already well under that, while the draws are what the
#: runtime is made of. Recorded in the artifact so the difference is visible.
CALIB_SUBSPACE_DRAWS = 99

#: Permutation-only runs per row. Cheap enough to reach 150 pairs, which is
#: where the registered null's inflation is worst.
N_PERM_TRIALS = 100

#: Null draws, mirroring the module so the two cannot drift.
from p7_motifs.steering_gate import N_SUBSPACE_DRAWS  # noqa: E402

#: Mean offsets swept, in units of the population's own spread. A real residual
#: stream sits at the high end.
MEAN_OFFSETS: Tuple[float, ...] = (0.0, 2.0, 5.0)

#: Per-pair rate at or above which an alpha fraction counts as inside the
#: window's plateau. Comparing plateaus rather than argmaxes is what makes the
#: stability claim about the window instead of about sampling noise: several
#: fractions reach 1.000 and which one is highest is not a measurement.
PLATEAU_RATE = 0.95

#: Alpha fractions swept for the window.
ALPHA_FRACTIONS: Tuple[float, ...] = (
    0.01, 0.05, 0.09, 0.1, 0.12, 0.15, 0.17, 0.2, 0.22, 0.24, 0.26, 0.3, 1.0)

#: Alpha fractions swept for the v -> -v symmetry, reaching far enough below
#: the window to catch the regime where the odd term dominates.
SYMMETRY_FRACTIONS: Tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1)

_SEED = 20260825


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sg_alpha_fraction() -> float:
    """The module's own placed fraction, so the two cannot drift apart."""
    from p7_motifs.steering_gate import ALPHA_SPREAD_FRACTION
    return float(ALPHA_SPREAD_FRACTION)


# ---------------------------------------------------------------------------
# One synthetic layer, with a planted answer
# ---------------------------------------------------------------------------

def synthetic_layer(rng: np.random.Generator, *, dim_u_pos: int, dim_u_neg: int,
                    occupied: str, mean_offset: float,
                    concentration: float = CONCENTRATION,
                    dim_occupied: int = DIM_OCCUPIED,
                    d: int = D_MODEL, n: int = N_TOKENS):
    """
    A token population and two subspaces, with which one the cloud occupies
    planted by construction.

    `occupied` is "pos" (H1: the cloud lives inside U_pos), "neg" (the planted
    INVERSION) or "other" (H0: the cloud lives in a subspace unrelated to
    both). The occupied subspace always has `dim_occupied` dimensions whatever
    `dim_u_pos` is, so the dimension cliff varies the RATIO and nothing else --
    confounding the two is what a first pass of this measurement did.
    """
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    u_pos = Q[:, :dim_u_pos]
    u_neg = Q[:, dim_u_pos:dim_u_pos + dim_u_neg]
    start = {"pos": 0, "neg": dim_u_pos,
             "other": dim_u_pos + dim_u_neg}[occupied]
    u_occ = Q[:, start:start + dim_occupied]
    X = rng.normal(size=(n, d))
    if concentration:
        X = X + (rng.normal(size=(n, dim_occupied)) * concentration) @ u_occ.T
    if mean_offset:
        scale = float(np.sqrt((X ** 2).sum(axis=1).mean()))
        X = X + (Q[:, -1] * mean_offset * scale)[None, :]
    return X, u_pos, u_neg


def _pair_rates(rng: np.random.Generator, n_trials: int, *, alpha_fraction: float,
                debias: bool, mode: str, alpha_scale: str = "spread",
                **layer) -> dict:
    """
    Per-pair D rates over `n_trials` independent layers.

    `alpha_scale` names what the fraction multiplies, and it has to be named
    rather than assumed. A design that does not centre has no "spread" to scale
    by -- it would naturally use the RMS token norm -- and with a mean offset
    those two differ by the offset itself, which is more than the readable
    window is wide. Measuring the undebiased arm at the centred scale would
    compare it at an alpha it would never have chosen.
    """
    from p7_motifs import steering_gate as sg

    prev = sg.DEBIAS_BASELINE_MEAN
    sg.DEBIAS_BASELINE_MEAN = bool(debias)
    try:
        D = []
        for _ in range(n_trials):
            X, u_pos, u_neg = synthetic_layer(rng, **layer)
            scale = (sg.population_spread(X) if alpha_scale == "spread"
                     else float(np.sqrt((X ** 2).sum(axis=1).mean())))
            a = alpha_fraction * scale
            vp = sg.draw_unit_in_subspace(u_pos, rng)
            vn = sg.draw_unit_in_subspace(u_neg, rng)
            D.append(sg.pair_statistic(X, vp, vn, a, mode)["D"])
    finally:
        sg.DEBIAS_BASELINE_MEAN = prev
    D = np.asarray(D, dtype=np.float64)
    return {"informative_rate": float((D != 0).mean()),
            "frac_predicted": float((D == 2).mean()),
            "frac_inverted": float((D == -2).mean()),
            "mean_D": float(D.mean())}


# ---------------------------------------------------------------------------
# The six sections
# ---------------------------------------------------------------------------

def attainable_floor_table(alpha: float) -> dict:
    from p7_motifs.steering_gate import attainable_floor, min_informative_pairs

    rows = []
    for m in (8, 12, 20, 30, 60):
        for k in range(1, 9):
            if k > m:
                continue
            f = attainable_floor(m, k)
            rows.append({"n_pairs": m, "n_informative": k,
                         "floor": float(f), "clears_alpha": bool(f <= alpha)})
    return {
        "_what": ("the smallest p the label-permutation null can express, as a "
                  "function of the INFORMATIVE pair count. A pair with D = 0 "
                  "contributes identically to the observed sum and to every "
                  "null pattern, so it cannot separate them."),
        "alpha": alpha,
        "min_informative_pairs": min_informative_pairs(alpha),
        "floor_is_independent_of_n_pairs": True,
        "rows": rows,
    }


def sign_symmetry(rng: np.random.Generator, n_trials: int = 60) -> dict:
    """Does sign(dER) survive v -> -v? The measurement that fixed ER_MODE."""
    from p7_motifs import steering_gate as sg

    out = []
    for mode in ("raw", "normed"):
        for f in SYMMETRY_FRACTIONS:
            agree, inverted, sizes = 0, 0, []
            for _ in range(n_trials):
                X, u_pos, u_neg = synthetic_layer(
                    rng, dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                    occupied="pos", mean_offset=2.0)
                a = f * sg.population_spread(X)
                v = sg.draw_unit_in_subspace(u_pos, rng)
                w = sg.draw_unit_in_subspace(u_neg, rng)
                dp = sg.delta_effective_rank(X, v, a, mode)
                dm = sg.delta_effective_rank(X, -v, a, mode)
                agree += int(np.sign(dp) == np.sign(dm))
                dn = sg.delta_effective_rank(X, w, a, mode)
                inverted += int(np.sign(dn) - np.sign(dp) == -2)
                sizes.append(abs(dp))
            out.append({"er_mode": mode, "alpha_spread_fraction": float(f),
                        "sign_agreement_under_v_to_minus_v": agree / n_trials,
                        "frac_pairs_reading_inverted": inverted / n_trials,
                        "mean_abs_delta_er": float(np.mean(sizes))})
    return {
        "_what": ("sign(dER(v)) == sign(dER(-v))? A steering DIRECTION and its "
                  "negation are the same object, so a criterion that answers "
                  "differently for them is not a criterion about the "
                  "decomposition. Debiasing the baseline mean zeroes the "
                  "first-order Gram term and makes raw dER even in v; L2 row "
                  "normalization is not linear and puts an odd term back."),
        "_decided": "ER_MODE = 'raw'",
        "n_trials": n_trials,
        "rows": out,
    }


def mean_offset_table(rng: np.random.Generator) -> dict:
    """
    The measurement that decided DEBIAS_BASELINE_MEAN.

    Each configuration is swept over the whole alpha grid and reported at ITS
    OWN best fraction, with the H0 rate read at that same fraction. Comparing
    them at one shared alpha would be unfair in the direction that flatters the
    choice actually made: the readable window sits at a different fraction when
    the baseline mean is left in, because the baseline's own mean direction is
    then part of the spectrum being perturbed.
    """
    rows = []
    for debias, scale in ((False, "token_norm"), (False, "spread"),
                          (True, "spread")):
        for off in MEAN_OFFSETS:
            best = None
            for f in ALPHA_FRACTIONS:
                h1 = _pair_rates(rng, N_MEAN_OFFSET_TRIALS, alpha_fraction=f,
                                 debias=debias, mode="raw", alpha_scale=scale,
                                 dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                                 occupied="pos", mean_offset=off)
                if best is None or h1["frac_predicted"] > best[1]["frac_predicted"]:
                    best = (f, h1)
            f, h1 = best
            h0 = _pair_rates(rng, N_MEAN_OFFSET_TRIALS, alpha_fraction=f,
                             debias=debias, mode="raw", alpha_scale=scale,
                             dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                             occupied="other", mean_offset=off)
            rows.append({
                "debias_baseline_mean": debias, "alpha_scale": scale,
                "mean_offset_in_spreads": float(off),
                "best_alpha_fraction": float(f),
                "h1_frac_predicted": h1["frac_predicted"],
                "h1_informative_rate": h1["informative_rate"],
                "h0_frac_predicted": h0["frac_predicted"],
                "h0_informative_rate": h0["informative_rate"],
            })
    return {
        "_what": ("per-pair rates against the cloud's own mean offset, each "
                  "configuration at its own best alpha. Steering is a pure mean "
                  "effect -- re-centring AFTER injection makes dER identically "
                  "zero -- so the pre-existing offset competes directly with "
                  "the injected one."),
        "_decided": "DEBIAS_BASELINE_MEAN = True (author's call, 2026-08-25)",
        "n_trials_per_cell": N_MEAN_OFFSET_TRIALS,
        "alpha_fractions_swept": list(ALPHA_FRACTIONS),
        "rows": rows,
    }


def alpha_window(rng: np.random.Generator) -> dict:
    """What ALPHA_SPREAD_FRACTION is placed against, and whether it travels."""
    rows = []
    for off in MEAN_OFFSETS:
        for f in ALPHA_FRACTIONS:
            for occ in ("pos", "other"):
                r = _pair_rates(rng, N_PAIR_TRIALS, alpha_fraction=f,
                                debias=True, mode="raw",
                                dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                                occupied=occ, mean_offset=off)
                rows.append({"mean_offset_in_spreads": float(off),
                             "alpha_spread_fraction": float(f),
                             "family": "H1" if occ == "pos" else "H0", **r})
    # The PLATEAU, not the argmax. Several fractions reach a rate of 1.000 and
    # which one wins is decided by sampling noise, so an argmax comparison
    # across offsets reports instability that the rows plainly do not show.
    # The plateau is the set of fractions whose rate is at or above
    # PLATEAU_RATE, and it is the plateaus that must coincide.
    best, plateau = {}, {}
    for off in MEAN_OFFSETS:
        cand = [r for r in rows if r["family"] == "H1"
                and r["mean_offset_in_spreads"] == off]
        best[str(off)] = max(cand, key=lambda r: r["frac_predicted"])[
            "alpha_spread_fraction"]
        plateau[str(off)] = sorted(r["alpha_spread_fraction"] for r in cand
                                   if r["frac_predicted"] >= PLATEAU_RATE)
    return {
        "_what": ("per-pair rates across the injection scale. Below the window "
                  "the sign is dominated by terms the criterion is not about; "
                  "above it the rank-1 spike n*alpha^2*v v^T dominates the Gram "
                  "matrix and BOTH arms reduce effective rank for any "
                  "direction."),
        "_decided": ("ALPHA_SPREAD_FRACTION = 0.1, labelled `placed` per Phase 7 "
                     "adjudication constraint 4. The FRACTION is placed; the "
                     "spread it multiplies is derived from the data."),
        "n_trials_per_row": N_PAIR_TRIALS,
        "best_fraction_by_mean_offset": best,
        "plateau_rate": PLATEAU_RATE,
        "plateau_by_mean_offset": plateau,
        "window_location_is_stable_in_spread_units":
            len({tuple(v) for v in plateau.values()}) == 1,
        "shipped_fraction_is_in_every_plateau":
            all(sg_alpha_fraction() in v for v in plateau.values()),
        "rows": rows,
    }


def dimension_cliff(rng: np.random.Generator) -> dict:
    """The precondition: dim U_pos against the dimension the cloud occupies."""
    from p7_motifs.steering_gate import min_informative_pairs

    need = min_informative_pairs(0.05)
    rows = []
    for kpos in (16, 24, 32, 48, 64, 96):
        r = _pair_rates(rng, N_PAIR_TRIALS, alpha_fraction=sg_alpha_fraction(),
                        debias=True,
                        mode="raw", dim_u_pos=kpos, dim_u_neg=DIM_OCCUPIED,
                        occupied="pos", mean_offset=2.0)
        rate = r["informative_rate"]
        rows.append({"dim_u_pos": kpos, "dim_occupied": DIM_OCCUPIED,
                     "ratio": kpos / DIM_OCCUPIED,
                     # what the REGISTERED permutation would have needed. The
                     # adjudicated subspace null has no informative-pair floor,
                     # so this column is the diagnostic arm's requirement and
                     # is kept to show what replacing the null bought.
                     "pairs_needed_under_registered_null":
                         (None if rate == 0 else int(np.ceil(need / rate))),
                     **r})
    return {
        "_what": ("a uniform draw from U_pos carries only "
                  "dim(occupied)/dim(U_pos) of its energy into the subspace the "
                  "cloud lives in. Below about half, the sign contrast vanishes "
                  "and NO number of pairs suffices. POPPER_PLAN.md 6h measured "
                  "that U_pos is the un-shrunk bucket in the projector build's "
                  "resolution order, which is the unfavourable side of this."),
        "_not_a_decision": ("a precondition on the pilot. The obvious fix -- "
                            "drawing from the intersection of U_pos with the "
                            "occupied subspace -- is circular and is refused: "
                            "a probe aligned with the cloud by construction "
                            "concentrates it by construction."),
        "n_trials_per_row": N_PAIR_TRIALS,
        "min_informative_pairs_at_alpha_0_05": need,
        "rows": rows,
    }


def registered_null_inflation(rng: np.random.Generator, alpha: float) -> dict:
    """
    The REGISTERED label-permutation null's Type-I rate against the pair count.

    Measured without ever drawing the subspace null, which is what makes it
    cheap enough to reach 150 pairs. Rates are CONDITIONAL ON EMISSION: in the
    clean regime every H0 pair is uninformative, the gate refuses, and an
    unconditional rate would read 0.000 -- calibrated by refusing rather than
    by controlling, which is the conditioning error POPPER_PLAN.md 6g records.
    """
    from p7_motifs.steering_gate import (label_permutation_diagnostic,
                                         pair_statistic, population_spread,
                                         draw_unit_in_subspace, _baseline,
                                         ER_MODE)
    from core.metrics import effective_rank

    frac = sg_alpha_fraction()
    rows = []
    for conc, occ, family in ((0.0, "other", "H0-noisy(conc=0.0)"),
                              (0.3, "other", "H0-noisy(conc=0.3)"),
                              (CONCENTRATION, "other", "H0-separated"),
                              (CONCENTRATION, "pos", "H1")):
        for m in (8, 24, 40, 150):
            rej = emitted = 0
            for _ in range(N_PERM_TRIALS):
                X, u_pos, u_neg = synthetic_layer(
                    rng, dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                    occupied=occ, mean_offset=2.0, concentration=conc)
                base = _baseline(X)
                er0 = float(effective_rank(base, mode=ER_MODE))
                a = frac * population_spread(X)
                D = [pair_statistic(None, draw_unit_in_subspace(u_pos, rng),
                                    draw_unit_in_subspace(u_neg, rng), a,
                                    ER_MODE, base=base, er0=er0)["D"]
                     for _ in range(m)]
                d = label_permutation_diagnostic(D, alpha)
                if d.get("p_value") is not None:
                    emitted += 1
                    rej += int(d["p_value"] <= alpha)
            rows.append({"family": family, "concentration": conc, "n_pairs": m,
                         "emission_rate": emitted / N_PERM_TRIALS,
                         "reject_rate_given_emitted":
                             (None if not emitted else rej / emitted)})
    return {
        "_what": ("Type-I rate of the null the registry's wording names, "
                  "against the pair count. The inflation GROWS with the pair "
                  "count: more pairs shrink the permutation null's spread like "
                  "sqrt(m) and leave the layer's shared tilt untouched."),
        "_decided": ("the registered null is NOT adjudicated; "
                     "NULL_FAMILY replaces it"),
        "alpha": alpha,
        "n_trials_per_row": N_PERM_TRIALS,
        "rows": rows,
    }


def adjudicated_null_validity_and_power(rng: np.random.Generator,
                                        alpha: float) -> dict:
    """
    The gate as it actually runs: the matched-dimension subspace null, with the
    registered permutation computed alongside on the SAME draws.

    Both rates come from one set of runs, so the comparison is paired rather
    than two separate experiments -- the two nulls see identical data.
    """
    from p7_motifs.steering_gate import p_value_p_st1

    rows = []
    for conc, occ, family in ((0.0, "other", "H0-noisy(conc=0.0)"),
                              (0.3, "other", "H0-noisy(conc=0.3)"),
                              (CONCENTRATION, "other", "H0-separated"),
                              (CONCENTRATION, "pos", "H1"),
                              (CONCENTRATION, "neg", "INVERTED")):
        for m in N_PAIRS_SWEPT:
            sub = {"rej": 0, "emit": 0, "recip": 0}
            perm = {"rej": 0, "emit": 0}
            verdicts = {"TRACKS-DECOMPOSITION": 0, "INVERTS": 0,
                        "INSUFFICIENT": 0}
            informative = []
            for _ in range(N_GATE_TRIALS):
                X, u_pos, u_neg = synthetic_layer(
                    rng, dim_u_pos=DIM_OCCUPIED, dim_u_neg=DIM_OCCUPIED,
                    occupied=occ, mean_offset=2.0, concentration=conc)
                res = p_value_p_st1(X, u_pos, u_neg, m, gate_alpha=alpha,
                                    seed=int(rng.integers(1 << 30)),
                                    n_draws=CALIB_SUBSPACE_DRAWS,
                                    with_profile=False)
                verdicts[res["verdict"]] += 1
                informative.append(res["n_informative_pairs"])
                if res.get("p_value") is not None:
                    sub["emit"] += 1
                    sub["rej"] += int(res["p_value"] <= alpha)
                    sub["recip"] += int(res["p_reciprocal"] <= alpha)
                d = res.get("label_permutation_diagnostic", {})
                if d.get("p_value") is not None:
                    perm["emit"] += 1
                    perm["rej"] += int(d["p_value"] <= alpha)
            n = float(N_GATE_TRIALS)
            rows.append({
                "family": family, "concentration": conc, "n_pairs": m,
                "subspace_emission_rate": sub["emit"] / n,
                "subspace_reject_given_emitted":
                    (None if not sub["emit"] else sub["rej"] / sub["emit"]),
                "subspace_reciprocal_given_emitted":
                    (None if not sub["emit"] else sub["recip"] / sub["emit"]),
                "permutation_emission_rate": perm["emit"] / n,
                "permutation_reject_given_emitted":
                    (None if not perm["emit"] else perm["rej"] / perm["emit"]),
                "tracks_decomposition": verdicts["TRACKS-DECOMPOSITION"] / n,
                "inverts": verdicts["INVERTS"] / n,
                "insufficient": verdicts["INSUFFICIENT"] / n,
                "mean_informative_pairs": float(np.mean(informative)),
            })
    return {
        "_what": ("the gate as it runs. `subspace_*` is the adjudicated null; "
                  "`permutation_*` is the registered one computed on the same "
                  "draws, so the comparison is paired. H1 and INVERTED rows "
                  "are power in each direction."),
        "alpha": alpha,
        "n_trials_per_row": N_GATE_TRIALS,
        "n_subspace_draws_used_here": CALIB_SUBSPACE_DRAWS,
        "n_subspace_draws_shipped": N_SUBSPACE_DRAWS,
        "n_pairs_swept": list(N_PAIRS_SWEPT),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    from p7_motifs import steering_gate as sg

    rng = np.random.default_rng(seed)
    alpha = sg._alpha()
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": ("what P-ST1's steering-sign construction does, measured on "
                  "synthetic populations with a planted answer."),
        "_why": ("the construction has four constants and a precondition; this "
                 "is the evidence for each, measured rather than reasoned "
                 "about."),
        "_not": ("not evidence about any model and not an adjudication. The "
                 "populations are synthetic; what is measured is the gate."),
        "generated_by": "python3 -m tools.calibrate_steering_sign --write",
        "gate_file": str(GATE_PATH.relative_to(ROOT)),
        "gate_sha256": _sha256(GATE_PATH),
        "alpha": float(alpha),
        "constants": {
            "unit": sg.P_ST1_UNIT,
            "alternative": sg.ALTERNATIVE,
            "reciprocal_alternative": sg.RECIPROCAL_ALTERNATIVE,
            "debias_baseline_mean": sg.DEBIAS_BASELINE_MEAN,
            "alpha_spread_fraction": sg.ALPHA_SPREAD_FRACTION,
            "alpha_is_placed": sg.ALPHA_IS_PLACED,
            "er_mode": sg.ER_MODE,
        },
        "geometry": {"d_model": D_MODEL, "n_tokens": N_TOKENS,
                     "dim_occupied": DIM_OCCUPIED},
        "seed": int(seed),
        "attainable_floor": attainable_floor_table(alpha),
        "sign_symmetry": sign_symmetry(rng),
        "mean_offset": mean_offset_table(rng),
        "alpha_window": alpha_window(rng),
        "dimension_cliff": dimension_cliff(rng),
        "registered_null_inflation": registered_null_inflation(rng, alpha),
        "adjudicated_null_validity_and_power":
            adjudicated_null_validity_and_power(rng, alpha),
    }


def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the module on disk, and does it still
    support the constants that module holds?

    Does NOT re-run the simulation -- that is the four minutes the commit
    exists to avoid. What it checks is that the record describes the gate as it
    now stands, and that each measured section still points the way the
    corresponding constant was set.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.calibrate_steering_sign --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    from p7_motifs import steering_gate as sg

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    if not GATE_PATH.exists():
        problems.append(f"{GATE_PATH} is missing")
    elif rec.get("gate_sha256") != _sha256(GATE_PATH):
        problems.append(
            f"{GATE_PATH.name} has changed since the calibration was written "
            f"(sha256 {_sha256(GATE_PATH)[:12]} on disk vs "
            f"{str(rec.get('gate_sha256'))[:12]} on record). The rates "
            f"describe a construction that no longer exists in that form; "
            f"rerun --write rather than editing the hash.")

    live = {"unit": sg.P_ST1_UNIT, "alternative": sg.ALTERNATIVE,
            "reciprocal_alternative": sg.RECIPROCAL_ALTERNATIVE,
            "debias_baseline_mean": sg.DEBIAS_BASELINE_MEAN,
            "alpha_spread_fraction": sg.ALPHA_SPREAD_FRACTION,
            "alpha_is_placed": sg.ALPHA_IS_PLACED, "er_mode": sg.ER_MODE}
    for k, v in live.items():
        if rec.get("constants", {}).get(k) != v:
            problems.append(
                f"constant {k}: record says "
                f"{rec.get('constants', {}).get(k)!r}, module says {v!r}. The "
                f"measured rates are rates of a different test.")

    sym = {(r["er_mode"], r["alpha_spread_fraction"]): r
           for r in rec.get("sign_symmetry", {}).get("rows", [])}
    raw_small = [r for (m, f), r in sym.items() if m == "raw" and f <= 1e-3]
    if raw_small and not all(
            r["sign_agreement_under_v_to_minus_v"] == 1.0 for r in raw_small):
        problems.append(
            "the record no longer shows raw dER even in v at small alpha, "
            "which is the whole reason ER_MODE is 'raw'")
    if sg.ER_MODE != "raw":
        problems.append(
            f"ER_MODE is {sg.ER_MODE!r}; this calibration's sign-symmetry "
            f"section is the argument for 'raw'")
    return problems


def print_summary(rec: dict) -> None:
    print(f"gate: {rec['gate_file']}  sha256 {rec['gate_sha256'][:12]}")
    print(f"alpha: {rec['alpha']}   constants: {rec['constants']}\n")

    fl = rec["attainable_floor"]
    print(f"=== attainable floor: {fl['min_informative_pairs']} INFORMATIVE "
          f"pairs are needed at alpha={fl['alpha']} ===")
    for m in (8, 30):
        row = [r for r in fl["rows"] if r["n_pairs"] == m]
        print(f"  m={m:3d}  " + "  ".join(
            f"k={r['n_informative']}:{r['floor']:.4f}" for r in row[:6]))

    print("\n=== sign symmetry under v -> -v (decided ER_MODE) ===")
    print(f"  {'mode':>7} {'alpha/spread':>13} {'agree':>7} {'reads inverted':>15}")
    for r in rec["sign_symmetry"]["rows"]:
        print(f"  {r['er_mode']:>7} {r['alpha_spread_fraction']:>13.0e} "
              f"{r['sign_agreement_under_v_to_minus_v']:>7.2f} "
              f"{r['frac_pairs_reading_inverted']:>15.2f}")

    print("\n=== mean offset, each configuration at its own best alpha "
          "(decided DEBIAS_BASELINE_MEAN) ===")
    print(f"  {'debias':>7} {'alpha scale':>11} {'offset':>7} {'best a':>7} "
          f"{'H1 P(+2)':>9} {'H0 P(+2)':>9}")
    for r in rec["mean_offset"]["rows"]:
        print(f"  {str(r['debias_baseline_mean']):>7} {r['alpha_scale']:>11} "
              f"{r['mean_offset_in_spreads']:>7.1f} "
              f"{r['best_alpha_fraction']:>7.2f} "
              f"{r['h1_frac_predicted']:>9.3f} {r['h0_frac_predicted']:>9.3f}")

    aw = rec["alpha_window"]
    print(f"\n=== alpha window (plateau stable in spread units: "
          f"{aw['window_location_is_stable_in_spread_units']}; plateaus "
          f"{aw['plateau_by_mean_offset']}; shipped fraction inside every "
          f"plateau: {aw['shipped_fraction_is_in_every_plateau']}) ===")
    print(f"  {'offset':>7} " + " ".join(
        f"{f:>8.3g}" for f in ALPHA_FRACTIONS))
    for off in MEAN_OFFSETS:
        cells = [next(r for r in aw["rows"] if r["mean_offset_in_spreads"] == off
                      and r["alpha_spread_fraction"] == f and r["family"] == "H1")
                 for f in ALPHA_FRACTIONS]
        print(f"  {off:>7.1f} " + " ".join(f"{c['frac_predicted']:>8.3f}"
                                           for c in cells))

    print("\n=== dimension cliff (precondition, not a decision) ===")
    print(f"  {'dim U_pos':>10} {'ratio':>6} {'informative':>12} {'pairs (reg null)':>17}")
    for r in rec["dimension_cliff"]["rows"]:
        need = r["pairs_needed_under_registered_null"]
        print(f"  {r['dim_u_pos']:>10} {r['ratio']:>6.1f} "
              f"{r['informative_rate']:>12.3f} "
              f"{'no number suffices' if need is None else need:>17}")

    print("\n=== the REGISTERED null's Type-I rate against the pair count ===")
    print(f"  {'family':>20} " + " ".join(f"{m:>7}" for m in (8, 24, 40, 150)))
    rows = rec["registered_null_inflation"]["rows"]
    for fam in dict.fromkeys(r["family"] for r in rows):
        cells = []
        for m in (8, 24, 40, 150):
            r = next((x for x in rows if x["family"] == fam
                      and x["n_pairs"] == m), None)
            v = r and r["reject_rate_given_emitted"]
            cells.append("-" if v is None else f"{v:.3f}")
        print(f"  {fam:>20} " + " ".join(f"{c:>7}" for c in cells))

    print("\n=== the ADJUDICATED null, and the registered one on the same draws ===")
    print(f"  {'family':>20} {'pairs':>5} {'sub rej|em':>11} {'perm rej|em':>12} "
          f"{'TRACKS':>7} {'INVERTS':>8} {'mean inf':>9}")
    for r in rec["adjudicated_null_validity_and_power"]["rows"]:
        sv = r["subspace_reject_given_emitted"]
        pv = r["permutation_reject_given_emitted"]
        print(f"  {r['family']:>20} {r['n_pairs']:>5} "
              f"{'-' if sv is None else f'{sv:.3f}':>11} "
              f"{'-' if pv is None else f'{pv:.3f}':>12} "
              f"{r['tracks_decomposition']:>7.3f} {r['inverts']:>8.3f} "
              f"{r['mean_informative_pairs']:>9.1f}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run the calibration and write the record (~4 min)")
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
        args.out.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"wrote {args.out}")
        print_summary(rec)
        return 0
    if args.check:
        problems = check_record(args.out)
        for p in problems:
            print(f"PROBLEM: {p}")
        if not problems:
            print(f"{args.out.name}: clean, and describes the module on disk")
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
