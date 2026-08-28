#!/usr/bin/env python3
"""
tools/calibrate_patching_exponent.py — offline calibration of P-AB1's growth
exponent construction (`p7_motifs/patching_gate.py`).

Run once, offline; the result is committed to
`claims/calibration/patching_exponent.json` and pinned by
`tests/test_patching_gate.py::TestCommittedCalibration`.

WHY THIS IS A TOOL AND NOT A TEST

The same division of labour as the other three calibrations, for the same
reason: measuring a rejection rate to three digits takes hundreds of replicates
of an end-to-end gate call, which is minutes, and the CI gating tier runs in
tens of seconds. What the pure tier pins is the MECHANISM -- deterministically
and in milliseconds -- plus the committed numbers, so the record cannot drift
without a failure.

WHAT IS MEASURED, AND WHY EACH SECTION EXISTS

`validity` -- the H0 families, each an attempt to break the sign-flip null,
run under BOTH readings of the exchangeable unit on the SAME draws so the
comparison is paired.

  independent          real and control arms drawn from the same law, cell by
                       cell. The plain H0. Both units should sit at or under
                       alpha.
  shared-prompt-factor the row that decides the unit. A component common to a
                       whole prompt -- one prompt whose geometry tilts all its
                       ablation points the same way -- with `rho` its share of
                       each difference. `status-6.md`'s "49 layers are not 49
                       independent observations" in this design's clothing, and
                       the per-ablation-point unit is the reading the registry's
                       wording implies.
  fixed-offset         the LIMITATION arm, and the one that is supposed to
                       reject under BOTH units. Real ablation directions are not
                       isotropic and the control directions are, so every cell
                       is nudged the same way for a reason with nothing to do
                       with the field account. No label-swap null separates
                       that from the effect -- a confound present in every cell
                       is present under every sign pattern -- which is 6i's
                       shared-per-unit-factor rejecting at 1.00, here. It is in
                       the file because POPPER_PLAN.md 6m records that a
                       calibration whose H0 families cannot express the failure
                       it is meant to rule out is an audit arm incapable of
                       failing.
  saturating-both-arms both arms bend identically. The window contributes to
                       both exponents and must cancel in the pair; if it does
                       not, the common-window construction is wrong and this row
                       is where it shows.
  differential-saturation
                       the SECOND sensitivity arm, and the one aimed at the
                       reciprocal branch. Both arms have the SAME true exponent
                       but the real arm saturates sooner, which is what a real
                       ablation that propagates further does. The contrast is
                       biased downward, so PROPAGATES is attenuated and
                       RECAPTURES can be manufactured -- measured here rather
                       than described, and the reason `p_reciprocal` reaches no
                       ledger.

`power` -- a planted exponent gap in each direction, so both verdict branches
are shown to be branches that can actually fire (6h found an audit arm
reporting PASS while incapable of failing; 6i made checking the reciprocal
branch a requirement).

`window_dependence` -- deterministic, no replicates: one fixed set of dynamics
fitted over windows of different lengths. This is what "the exponent is partly a
property of the fit window" means as a number, and it is the measurement that
forced the common window.

`statistic_choice` -- the sign sum against a mean of paired differences on a
grid whose windows are deliberately unequal. The choice of statistic was made on
this measurement rather than on the argument.

`grid_arithmetic` -- exact, closed form, no simulation: what an even number of
ablation points per prompt costs. A prompt contributes the sum of its points'
signs, so an even count can split evenly and contribute nothing to the
observation or to any sign pattern (POPPER_PLAN.md 6l's informative-row
structure, reached by a second construction).

Usage
-----
    python3 -m tools.calibrate_patching_exponent --write
    python3 -m tools.calibrate_patching_exponent --replicates 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from p7_motifs.patching_gate import (
    MIN_FIT_POINTS,
    P_AB1_UNITS,
    attainable_floor_report,
    fit_growth_exponent,
    bend_contrast_arm,
    p_value_p_ab1,
    paired_exponents,
    signflip_arm,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "claims" / "calibration" / "patching_exponent.json"

SCHEMA_VERSION = 1

#: The design the rates are rates for. Six prompts is the smallest number that
#: clears alpha under the prompt unit (floor 2/65 = 0.031) and seven ablation
#: points is odd, so no prompt can split evenly -- both are consequences of the
#: arithmetic in `grid_arithmetic` rather than choices made here.
N_PROMPTS = 6
N_ABLATION_POINTS = 7
FIT_WINDOW = 8

#: The synthetic divergence family. ALL PLACED: they define the family the
#: rates are rates under, and no distribution was consulted to pick them.
BASE_EXPONENT = 1.4
LOG_NOISE_SD = 0.20

#: The scale in which a shared component and a fixed offset are expressed. At
#: this window and noise level one fitted exponent has sd ~0.115 and a paired
#: difference ~0.163, so a jitter of 0.25 is about 1.5 difference-sds: large
#: enough that "rho = 1" really means the difference is essentially the
#: prompt's, which is the regime the unit question lives in.
EXPONENT_JITTER = 0.25

#: Saturation timescale in layers for the rows that bend. Placed, same status.
TAU_BENT = 5.0
TAU_STRAIGHT = 1.0e6

ALPHA_FALLBACK = 0.05


def _alpha() -> float:
    from core.adjudication import load_registry
    try:
        return float(load_registry().get("alpha", ALPHA_FALLBACK))
    except Exception:
        return ALPHA_FALLBACK


def _curve(rng, exponent: float, tau: float, window: int) -> np.ndarray:
    """A saturating power law with multiplicative noise. `tau -> inf` is the
    pure power law; a finite tau bends, which is what makes the fitted exponent
    a property of the window."""
    k = np.arange(1, window + 1, dtype=np.float64)
    mean = 1.0 - np.exp(-((k / tau) ** exponent))
    return mean * np.exp(rng.normal(0.0, LOG_NOISE_SD, size=window))


def _draw(rng, family: str, rho: float, offset: float, gap: float,
          n_prompts: int, n_points: int, window: int):
    """One design's worth of curves for both arms."""
    real, ctrl = [], []
    for _ in range(n_prompts):
        # A per-prompt shared component enters as a shift of BOTH arms' true
        # exponents in opposite directions, which is how a per-prompt factor
        # reaches the DIFFERENCE the statistic reads.
        c = (rng.normal(0.0, math.sqrt(rho)) * EXPONENT_JITTER
             if rho > 0 else 0.0)
        rr, cc = [], []
        for _ in range(n_points):
            e = (rng.normal(0.0, math.sqrt(max(0.0, 1.0 - rho)))
                 * EXPONENT_JITTER)
            br = BASE_EXPONENT + gap + offset * EXPONENT_JITTER + c + e
            bc = BASE_EXPONENT
            tr = TAU_BENT if family in ("saturating-both-arms",
                                        "differential-saturation") else TAU_STRAIGHT
            tc = TAU_BENT if family == "saturating-both-arms" else TAU_STRAIGHT
            if family == "differential-saturation":
                tc = TAU_BENT * 3.0
            rr.append(_curve(rng, br, tr, window))
            cc.append(_curve(rng, bc, tc, window))
        real.append(rr)
        ctrl.append(cc)
    return real, ctrl


FAMILIES = (
    # name,                       rho,  offset, gap
    ("independent",               0.0,  0.0,    0.0),
    ("shared-prompt-factor@0.5",  0.5,  0.0,    0.0),
    ("shared-prompt-factor@1.0",  1.0,  0.0,    0.0),
    ("fixed-offset@0.5jitter",    0.0,  0.5,    0.0),
    ("fixed-offset@1.0jitter",    0.0,  1.0,    0.0),
    ("saturating-both-arms",      0.0,  0.0,    0.0),
    ("differential-saturation",   0.0,  0.0,    0.0),
    ("power@+0.15",               0.0,  0.0,    0.15),
    ("power@-0.15",               0.0,  0.0,   -0.15),
)


def measure_family(name, rho, offset, gap, replicates, seed, alpha) -> list:
    """
    Every rate here is CONDITIONAL ON EMISSION, and the bend refusal changes
    what that conditions on -- so each row also carries the COUNTERFACTUAL rate,
    re-scored on the refused draws through the gate's own arm. 6o re-scored its
    refusal in every cell rather than asserting the cost was small; the same is
    owed here, and more sharply, because this refusal is one that costs
    verdicts.
    """
    rng = np.random.default_rng(seed)
    mag = np.ones((N_PROMPTS, N_ABLATION_POINTS))
    acc = {u: {"rej": 0, "rec": 0, "emit": 0, "pl_refused": 0,
               "bend_flagged": 0, "other_refused": 0,
               "cf_rej": 0, "cf_rec": 0, "cf_n": 0,
               "p": [], "sat": [], "shared": []}
           for u in P_AB1_UNITS}
    for _ in range(replicates):
        real, ctrl = _draw(rng, name.split("@")[0], rho, offset, gap,
                           N_PROMPTS, N_ABLATION_POINTS, FIT_WINDOW)
        pair = paired_exponents(real, ctrl)
        for unit in P_AB1_UNITS:
            a = acc[unit]
            res = p_value_p_ab1(real, ctrl, mag, mag, unit=unit, alpha=alpha)
            # The counterfactual: the same arm, scored whether or not the bend
            # refusal fired. Computed from the gate's own pieces, not a copy.
            try:
                cf = signflip_arm(pair["difference"], unit, alpha)
                a["cf_n"] += 1
                a["cf_rej"] += int(cf["p_value"] <= alpha)
                a["cf_rec"] += int(cf["p_reciprocal"] <= alpha)
            except Exception:
                pass
            if (res.get("bend_contrast") or {}).get("confounded"):
                a["bend_flagged"] += 1
            if res["p_value"] is None:
                if (res.get("power_law") or {}).get("not_a_power_law"):
                    a["pl_refused"] += 1
                else:
                    a["other_refused"] += 1
                continue
            a["emit"] += 1
            a["rej"] += int(res["p_value"] <= alpha)
            a["rec"] += int(res["p_reciprocal"] <= alpha)
            a["p"].append(res["p_value"])
            a["sat"].append(res["saturation_diagnostic"]["paired_mean"])
            a["shared"].append(
                res["shared_prompt_factor"]["shared_share_estimate"])
    rows = []
    for unit in P_AB1_UNITS:
        a = acc[unit]
        n = max(1, a["emit"])
        cfn = max(1, a["cf_n"])
        rows.append({
            "family": name,
            "unit": unit,
            "replicates": replicates,
            "emitted": a["emit"],
            "refused_not_a_power_law": a["pl_refused"],
            "refused_other": a["other_refused"],
            "bend_contrast_flagged": a["bend_flagged"],
            "emission_rate": a["emit"] / replicates,
            "rejection_rate": a["rej"] / n,
            "reciprocal_rejection_rate": a["rec"] / n,
            "counterfactual_rejection_rate_no_bend_refusal": a["cf_rej"] / cfn,
            "counterfactual_reciprocal_rate_no_bend_refusal": a["cf_rec"] / cfn,
            "mean_p": float(np.mean(a["p"])) if a["p"] else None,
            "mean_paired_window_sensitivity":
                float(np.mean(a["sat"])) if a["sat"] else None,
            "mean_shared_share_estimate":
                float(np.mean(a["shared"])) if a["shared"] else None,
        })
    return rows


def window_dependence() -> list:
    """Deterministic. One set of dynamics, several fit windows."""
    out = []
    for beta in (1.0, 1.5, 2.0):
        for tau in (4.0, 8.0, 16.0, TAU_STRAIGHT):
            k = np.arange(1, 25, dtype=np.float64)
            mean = 1.0 - np.exp(-((k / tau) ** beta))
            row = {"beta_true": beta, "tau": tau, "fitted": {}}
            for w in (3, 4, 6, 8, 12, 16, 24):
                row["fitted"][str(w)] = round(
                    fit_growth_exponent(mean[:w])["exponent"], 4)
            out.append(row)
    return out


def sampling_spread() -> dict:
    """
    Deterministic, exact: the spread a log-log slope has at each window, which
    is `sigma / sqrt(Sxx)` with `Sxx` fixed by the window alone. This is why the
    units are combined by a SIGN and not by a mean -- a mean of paired
    differences is dominated by whichever ablation point sits nearest the output.
    """
    rows = {}
    for w in (3, 4, 5, 6, 8, 12, 16, 24):
        x = np.log(np.arange(1, w + 1, dtype=np.float64))
        sxx = float(((x - x.mean()) ** 2).sum())
        rows[str(w)] = {
            "sxx": round(sxx, 6),
            "slope_sd_per_unit_log_noise": round(1.0 / math.sqrt(sxx), 6),
            "slope_sd_at_this_files_noise":
                round(LOG_NOISE_SD / math.sqrt(sxx), 6),
        }
    return {
        "by_window": rows,
        "_note": (
            "Exact arithmetic on the window, no replicates. The ratio between "
            "the shortest and longest window tabulated here is what a mean of "
            "paired differences would weight by and a sign sum does not."),
    }


def power_law_arm_operating_curve(replicates, seed) -> list:
    """
    What `power_law_arm` does as the curve bends, over the design's own cell
    count. `tau -> inf` is the pure power law, where the refusal must sit at
    alpha and not above it: a refusal that fires on the shape it is supposed to
    admit costs verdicts for nothing.
    """
    rng = np.random.default_rng(seed)
    n_cells = N_PROMPTS * N_ABLATION_POINTS
    out = []
    for tau in (TAU_STRAIGHT, 30.0, 15.0, 8.0, 5.0, 3.0):
        fired = 0
        for _ in range(replicates):
            zs = np.array([
                fit_growth_exponent(
                    _curve(rng, BASE_EXPONENT, tau, FIT_WINDOW))["bend_z"]
                for _ in range(n_cells)])
            pooled = float(zs.sum() / math.sqrt(n_cells))
            fired += int(math.erfc(abs(pooled) / math.sqrt(2.0)) <= 0.05)
        out.append({
            "tau": tau,
            "n_cells": n_cells,
            "replicates": replicates,
            "refusal_rate_one_arm_at_0.05": fired / replicates,
        })
    return out


def discarded_refusal(replicates, seed, alpha) -> dict:
    """
    The first refusal this construction tried, measured, and the measurement is
    why it is not the one that shipped.

    The obvious condition is that the two arms bend by DIFFERENT amounts, since
    an equal bend cancels inside the pair, and it is testable two-sided with the
    gate's own sign-flip null. It is the right shape and it is too weak: on the
    differential-saturation family it turns away only some of the draws, and
    among the draws it lets through the reciprocal branch still fires at
    essentially 1. A refusal that thins a defect is not a refusal, and this
    section is in the artifact for the reason 6o's rank sweep is in its own:
    it is what changed the design.
    """
    rng = np.random.default_rng(seed)
    stats = {u: {"flagged": 0, "n": 0, "rec_given_passed": 0, "passed": 0}
             for u in P_AB1_UNITS}
    for _ in range(replicates):
        real, ctrl = _draw(rng, "differential-saturation", 0.0, 0.0, 0.0,
                           N_PROMPTS, N_ABLATION_POINTS, FIT_WINDOW)
        pair = paired_exponents(real, ctrl)
        for unit in P_AB1_UNITS:
            a = stats[unit]
            a["n"] += 1
            bend = bend_contrast_arm(pair["bend_contrast"], unit, alpha)
            if bend["confounded"]:
                a["flagged"] += 1
                continue
            a["passed"] += 1
            arm = signflip_arm(pair["difference"], unit, alpha)
            a["rec_given_passed"] += int(arm["p_reciprocal"] <= alpha)
    return {
        "family": "differential-saturation",
        "replicates": replicates,
        "per_unit": {
            u: {
                "flagged_by_the_discarded_refusal": stats[u]["flagged"],
                "let_through": stats[u]["passed"],
                "reciprocal_rejection_among_those_let_through":
                    (stats[u]["rec_given_passed"] / stats[u]["passed"]
                     if stats[u]["passed"] else None),
            } for u in P_AB1_UNITS},
        "_note": (
            "The reciprocal branch is P-AB1's registered FALSIFICATION branch, "
            "and the correct verdict on this family is INSUFFICIENT -- both "
            "arms carry the same true exponent and only their saturation "
            "differs. `power_law_arm` refuses the whole family instead."),
    }


def statistic_choice(replicates, seed, alpha) -> list:
    """Sign sum against a mean of paired differences, on unequal windows."""
    rng = np.random.default_rng(seed)
    windows = [3 + 4 * j for j in range(N_ABLATION_POINTS)]
    out = []
    for gap in (0.0, 0.05, 0.10, 0.20):
        rs = rm = 0
        for _ in range(replicates):
            d = np.zeros((N_PROMPTS, N_ABLATION_POINTS))
            for p in range(N_PROMPTS):
                for j, w in enumerate(windows):
                    br = fit_growth_exponent(
                        _curve(rng, BASE_EXPONENT + gap, TAU_STRAIGHT, w))
                    bc = fit_growth_exponent(
                        _curve(rng, BASE_EXPONENT, TAU_STRAIGHT, w))
                    d[p, j] = br["exponent"] - bc["exponent"]
            rs += int(signflip_arm(d, "prompt", alpha)["p_value"] <= alpha)
            # The mean-difference variant, scored through the same enumeration.
            blocks = d.sum(axis=1)
            from itertools import product
            pats = np.array(list(product((-1.0, 1.0), repeat=N_PROMPTS)))
            obs = blocks.sum()
            pm = (int((pats @ blocks >= obs - 1e-12).sum()) + 1) / (
                pats.shape[0] + 1)
            rm += int(pm <= alpha)
        out.append({
            "planted_exponent_gap": gap,
            "windows": windows,
            "replicates": replicates,
            "sign_sum_rejection": rs / replicates,
            "mean_difference_rejection": rm / replicates,
        })
    return out


def grid_arithmetic(alpha) -> dict:
    """Exact. What an even number of ablation points per prompt costs."""
    rows = []
    for n_points in range(3, 12):
        p0 = (math.comb(n_points, n_points // 2) / 2 ** n_points
              if n_points % 2 == 0 else 0.0)
        can = 0.0
        for k in range(0, N_PROMPTS + 1):
            pk = (math.comb(N_PROMPTS, k) * (1 - p0) ** k
                  * p0 ** (N_PROMPTS - k))
            if attainable_floor_report(N_PROMPTS, k, alpha)["sufficient"]:
                can += pk
        rows.append({
            "ablation_points_per_prompt": n_points,
            # NOT rounded: C(n, n/2)/2^n has a power-of-two denominator and is
            # exactly representable, and a stored 0.273438 against an exact
            # 0.2734375 is the shape of 6g's rounding defect. `can` is a sum of
            # products and is not, so it is rounded and read with a tolerance.
            "p_prompt_non_informative_under_h0": p0,
            "p_design_can_reject_at_all": round(can, 6),
        })
    floors = {str(n): round(2.0 / (2 ** n + 1), 6) for n in range(1, 11)}
    return {
        "n_prompts": N_PROMPTS,
        "floor_by_informative_unit_count": floors,
        "min_informative_units_for_alpha":
            attainable_floor_report(1, 1, alpha)[
                "min_informative_units_for_alpha"],
        "even_vs_odd": rows,
        "_note": (
            "p_prompt_non_informative_under_h0 is exact: under H0 each cell's "
            "sign is a fair coin, so a prompt with an even number of usable "
            "ablation points splits evenly with probability C(n, n/2)/2^n and "
            "contributes nothing to the observation or to any sign pattern. An "
            "odd count cannot."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--write", action="store_true",
                    help="write the artifact; without it, measure and print only")
    args = ap.parse_args(argv)

    alpha = _alpha()
    t0 = time.time()

    validity = []
    for i, (name, rho, offset, gap) in enumerate(FAMILIES):
        rows = measure_family(name, rho, offset, gap, args.replicates,
                              args.seed + 101 * i, alpha)
        validity.extend(rows)
        for r in rows:
            print(f"{r['family']:26s} unit={r['unit']:15s} "
                  f"reject={r['rejection_rate']:.4f} "
                  f"recip={r['reciprocal_rejection_rate']:.4f} "
                  f"emit={r['emitted']:4d} pl-refused={r['refused_not_a_power_law']:4d} "
                  f"cf_recip={r['counterfactual_reciprocal_rate_no_bend_refusal']:.4f}",
                  flush=True)

    print("\npower-law arm operating curve ...", flush=True)
    pl_curve = power_law_arm_operating_curve(
        max(50, args.replicates // 4), args.seed + 555)
    for r in pl_curve:
        print(f"  tau={r['tau']:>9.1f} refusal={r['refusal_rate_one_arm_at_0.05']:.3f}",
              flush=True)

    print("\nthe discarded refusal ...", flush=True)
    discarded = discarded_refusal(max(50, args.replicates // 4),
                                  args.seed + 777, alpha)
    for u, v in discarded["per_unit"].items():
        print(f"  unit={u:15s} flagged={v['flagged_by_the_discarded_refusal']:4d} "
              f"let_through={v['let_through']:4d} "
              f"reciprocal_among_them={v['reciprocal_rejection_among_those_let_through']}",
              flush=True)

    print("\nstatistic choice ...", flush=True)
    stat = statistic_choice(max(50, args.replicates // 3), args.seed + 991, alpha)
    for r in stat:
        print(f"  gap={r['planted_exponent_gap']:.2f} "
              f"sign={r['sign_sum_rejection']:.3f} "
              f"mean={r['mean_difference_rejection']:.3f}", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/calibrate_patching_exponent.py",
        "alpha": alpha,
        "design": {
            "n_prompts": N_PROMPTS,
            "n_ablation_points": N_ABLATION_POINTS,
            "fit_window": FIT_WINDOW,
            "min_fit_points": MIN_FIT_POINTS,
        },
        "synthetic_family": {
            "curve": "D(k) = 1 - exp(-(k/tau)^beta), times exp(N(0, sd))",
            "base_exponent": BASE_EXPONENT,
            "log_noise_sd": LOG_NOISE_SD,
            "tau_bent": TAU_BENT,
            "tau_straight": TAU_STRAIGHT,
            "_placed": "all PLACED, not calibrated: they define the family the "
                       "rates are rates under and no distribution was consulted",
        },
        "replicates": args.replicates,
        "seed": args.seed,
        "validity": validity,
        "power_note": (
            "the two power@ rows are the verdict branches. power@+0.15 must "
            "reject in the predicted direction and power@-0.15 in the "
            "reciprocal one, or a branch of the stop rule is one nothing can "
            "trigger -- POPPER_PLAN.md 6h's audit arm, and 6i's requirement "
            "that the falsification branch be checked to be one that can fire."),
        "window_dependence": window_dependence(),
        "sampling_spread": sampling_spread(),
        "power_law_arm_operating_curve": pl_curve,
        "discarded_refusal": discarded,
        "statistic_choice": stat,
        "grid_arithmetic": grid_arithmetic(alpha),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if not args.write:
        print("\n(not written: pass --write)")
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} in {doc['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
