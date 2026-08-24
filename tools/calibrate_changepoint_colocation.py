#!/usr/bin/env python3
"""
tools/calibrate_changepoint_colocation.py — offline calibration of the
changepoint co-location construction (`core/changepoint_colocation.py`).

Run once, offline; the result is committed to
`claims/calibration/changepoint_colocation.json` and pinned by
`tests/test_changepoint_colocation.py::TestCommittedCalibration`.

WHY THIS IS A TOOL AND NOT A TEST

Same division of labour as `tools/calibrate_claim_c_homogeneity.py` and
`tools/audit_p6_projector_labels.py`, for the same reason: measuring a rejection
rate to three digits takes hundreds of replicates times two thousand
permutations, which is minutes, and the CI gating tier runs in ten seconds. What
the pure tier pins is the MECHANISM -- deterministically and in milliseconds --
plus the committed numbers, so the record cannot drift without a failure.

WHAT IS MEASURED, AND WHY EACH ROW EXISTS

The construction's validity rests on one assumption: under H0 the two series'
per-unit change locations are independent, so which unit of A is paired with
which unit of B is arbitrary. Every row below is an attempt to break it.

  independent          both series' onsets drawn independently, log-uniform
                       over the sweep. The plain H0. Should sit at alpha.
  common-trend         both drawn from an EARLY-concentrated distribution --
                       "everything moves early in training". This is the row
                       that kills the alternatives: a permutation over
                       checkpoint order rejects at 0.32-0.45 here and an
                       enumerated circular shift at 0.103, because both assert
                       that a change could equally have been anywhere. The
                       pairing null holds each series' real locations fixed on
                       both sides, so the trend is held fixed rather than
                       assumed away.
  shared-unit-factor   the SENSITIVITY ARM, and the one that is supposed to
                       reject. Both series' onsets are pushed by a common
                       per-unit factor (a head that forms late forms late in
                       both) that has nothing to do with the claim. The pairing
                       null tests ASSOCIATION, and a common per-unit factor is
                       an association, so a high rate here is not a defect --
                       it is the construction's real limitation, measured
                       instead of described. Compare POPPER_PLAN.md 6f: prompts
                       run on one model are not independent, the cost was
                       measured rather than argued, and the measurement is what
                       the record carries.
  reversed             the observed pairing deliberately anti-aligned. p must go
                       to 1 and the RECIPROCAL test must fire, or the RE-ANCHORS
                       branch is a verdict that cannot happen -- the defect
                       POPPER_PLAN.md 6h found in an audit arm that could not
                       fail.
  power                both series sharing an onset per unit.

Usage
-----
    python3 -m tools.calibrate_changepoint_colocation --write
    python3 -m tools.calibrate_changepoint_colocation --replicates 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core.changepoint_colocation import paired_colocation_arm
from core.checkpoint_frames import step_x

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "claims" / "calibration" / "changepoint_colocation.json"

SCHEMA_VERSION = 1

#: A plausible 20-30 checkpoint cheap-tier Pythia sweep: the log-spaced releases
#: to 512, then the every-1000 releases thinned. CLAIM-B's instrument field says
#: "20-30 checkpoint cheap-tier sweep".
SWEEP = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
         8000, 13000, 23000, 33000, 43000, 63000, 83000, 103000, 123000, 143000)

#: Logistic width in log10-step, and per-checkpoint noise. PLACED: they set the
#: synthetic family the rates are rates under, and no distribution was consulted
#: for either. They are reported in the artifact for that reason.
ONSET_WIDTH_LOG_STEP = 0.35
NOISE_SD = 0.02

ALPHA = 0.05
N_UNITS = (8, 16, 24)


def _curve(steps, mid_step, rng):
    x = step_x(steps)
    return (1.0 / (1.0 + np.exp(-(x - np.log10(mid_step + 1.0)) / ONSET_WIDTH_LOG_STEP))
            + NOISE_SD * rng.standard_normal(x.size))


def _onsets(kind, n, rng):
    if kind == "common-trend":
        lo, hi = np.log10(2.0), np.log10(2000.0)
    else:
        lo, hi = np.log10(2.0), np.log10(60000.0)
    return 10.0 ** rng.uniform(lo, hi, n)


def _one(kind, n_units, rng, steps):
    if kind == "power":
        m = _onsets("independent", n_units, rng)
        a = [_curve(steps, x, rng) for x in m]
        b = [_curve(steps, x, rng) for x in m]
    elif kind == "reversed":
        m = np.sort(_onsets("independent", n_units, rng))
        a = [_curve(steps, x, rng) for x in m]
        b = [_curve(steps, x, rng) for x in m[::-1]]
    elif kind == "shared-unit-factor":
        # A per-unit factor that pushes BOTH series' onsets the same way and has
        # nothing to do with the claim.
        shared = rng.uniform(np.log10(2.0), np.log10(60000.0), n_units)
        ja = shared + 0.25 * rng.standard_normal(n_units)
        jb = shared + 0.25 * rng.standard_normal(n_units)
        a = [_curve(steps, 10.0 ** x, rng) for x in ja]
        b = [_curve(steps, 10.0 ** x, rng) for x in jb]
    else:
        a = [_curve(steps, x, rng) for x in _onsets(kind, n_units, rng)]
        b = [_curve(steps, x, rng) for x in _onsets(kind, n_units, rng)]
    return paired_colocation_arm(steps, a, "rise", b, "rise", alpha=ALPHA,
                                 unit_name="unit", arm_name="calibration")


def measure(kind, n_units, replicates, seed, steps):
    rng = np.random.default_rng(seed)
    ps, pr = [], []
    for _ in range(replicates):
        r = _one(kind, n_units, rng, steps)
        ps.append(r["p_value"])
        pr.append(r["p_reciprocal"])
    ps = np.asarray(ps)
    pr = np.asarray(pr)
    return {
        "family": kind,
        "n_units": int(n_units),
        "replicates": int(replicates),
        "rejection_rate": float(np.mean(ps <= ALPHA)),
        "reciprocal_rejection_rate": float(np.mean(pr <= ALPHA)),
        "mean_p": float(np.mean(ps)),
        "median_p": float(np.median(ps)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--write", action="store_true",
                    help="write the artifact; without it, measure and print only")
    args = ap.parse_args(argv)

    steps = np.asarray(SWEEP, dtype=float)
    t0 = time.time()
    rows = []
    for kind in ("independent", "common-trend", "shared-unit-factor",
                 "reversed", "power"):
        for n in N_UNITS:
            row = measure(kind, n, args.replicates, args.seed + 7 * n, steps)
            rows.append(row)
            print(f"{row['family']:20s} n_units={n:3d}  "
                  f"reject={row['rejection_rate']:.4f}  "
                  f"reciprocal={row['reciprocal_rejection_rate']:.4f}  "
                  f"mean p={row['mean_p']:.3f}", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/calibrate_changepoint_colocation.py",
        "alpha": ALPHA,
        "sweep_steps": list(SWEEP),
        "n_checkpoints": len(SWEEP),
        "replicates": args.replicates,
        "seed": args.seed,
        "synthetic_family": {
            "curve": "logistic in log10(step+1)",
            "onset_width_log_step": ONSET_WIDTH_LOG_STEP,
            "noise_sd": NOISE_SD,
            "_placed": "both PLACED, not calibrated: they define the family the "
                       "rates are rates under and no distribution was consulted",
        },
        "rows": rows,
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
