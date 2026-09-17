#!/usr/bin/env python3
"""
tools/calibrate_p_i5_joint_null.py — offline calibration of P-I5's joint
sign-flip null (`p7_motifs/p_i5_gate.py`).

Run once, offline; the result is committed to
`claims/calibration/p_i5_joint_null.json`.

WHY THIS IS A TOOL AND NOT A TEST

Same division of labour as every calibration before it (6m, 6q, 6s and the
tools those sections built): the pure-tier tests in
`tests/test_p_i5_gate.py` pin the MECHANISM deterministically and in
milliseconds (exact enumeration matching a brute-force reference,
attainable_floor matching direct construction for n=1..15, a handful of
seeded calibration checks with loose tolerances so they don't flake). This
script measures the actual REJECTION RATES to three digits, which needs
thousands of replicates per (n, statistic) cell — seconds, not
milliseconds, and the numbers the module docstring quotes are read off
this file's output rather than off a scratch run.

WHAT IS MEASURED

`floor` — attainable_floor(n) for n = 1..15, both alternatives, doubling as
a machine-readable version of the closed-form table the module docstring
proves and the tests enumeration-check.

`naive_and_corner` and `joint_rank` — calibrate_naive_and_corner /
calibrate_joint_rank at n = 6, 8, 10, 12, under a TRUE joint H0 (both axes
independent noise, zero effect on either). The first construction tried
(AND-corner) over-rejects by roughly 4x at every n measured; the min-rank
statistic does not — under THIS null. That null is the intersection of the
two axes' nulls, and P-I5 is a conjunction whose null is their UNION.

`intersection_union` — calibrate_intersection_union at the same n grid on
THREE arms: complete null, geometry-null with a real logit effect (the
falsifier's configuration), and logit-null with a real geometric effect.
This is the calibration P-I5's statistic needs and the min-rank statistic
never had; the self-check fails if any arm exceeds nominal (schema 2,
2026-09-17).

`partial_pass` — partial_pass_risk_demo on the falsifier's own
configuration (real logit effect, no geometric effect), at the n = 8 the
measurement grid below actually supports. Records the min-rank statistic's
rejection rate there — ~0.17-0.19 against nominal 0.05, an uncontrolled
Type-I rate on a point in the null, which is why the intersection-union
test replaced it.

`measurement_grid` — count_matched_pairs_by_prompt against a real cached
Pythia tokenizer (pythia-70m step143000, offline) plus
informative_prompt_count with the repeated_tokens exclusion applied. Only
run when transformers + the cached tokenizer are actually reachable
(HF_HUB_OFFLINE mode) — recorded as `null` with a reason otherwise, never
faked.

Usage
-----
    python tools/calibrate_p_i5_joint_null.py            # measure, print
    python tools/calibrate_p_i5_joint_null.py --write     # and commit it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p7_motifs.p_i5_gate import (            # noqa: E402
    DEGENERATE_PROMPT,
    attainable_floor,
    calibrate_joint_rank,
    calibrate_naive_and_corner,
    count_matched_pairs_by_prompt,
    informative_prompt_count,
    partial_pass_risk_demo,
    calibrate_intersection_union,
)

SCHEMA_VERSION = 2
OUT_PATH = ROOT / "claims" / "calibration" / "p_i5_joint_null.json"
CONSTRUCTION_PATH = ROOT / "p7_motifs" / "p_i5_gate.py"

_SEED = 20260916
N_GRID = (6, 8, 10, 12)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floor_table() -> dict:
    return {
        str(n): {
            "greater": attainable_floor(n, "greater"),
            "two_sided": attainable_floor(n, "two-sided"),
        }
        for n in range(1, 16)
    }


def _measurement_grid() -> dict:
    """Real (tokenization-only) measurement against a cached Pythia
    checkpoint. Refuses rather than fakes a number when the cache or
    transformers isn't reachable."""
    prev = {
        k: os.environ.get(k)
        for k in ("HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_XET")
    }
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        # Only the optional dependency and the cache lookup may make the grid
        # "unreachable". A failure in the measurement itself is a defect and
        # must stop the calibration, not be recorded as an absent cache.
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-70m", revision="step143000",
            )
        except (ImportError, OSError, ValueError) as exc:  # pragma: no cover — environment-dependent
            return {
                "reachable": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        counts = count_matched_pairs_by_prompt(tok)
        n_informative = informative_prompt_count(counts)
        return {
            "tokenizer": "EleutherAI/pythia-70m@step143000",
            "counts": counts,
            "degenerate_prompt_excluded": DEGENERATE_PROMPT,
            "n_informative_prompts": n_informative,
            "reachable": True,
        }
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_record(doc: dict) -> list:
    """Self-check: findings that would mean this record is wrong, not just
    unsurprising. Mirrors cross_head_association.json's self_check."""
    findings = []

    naive_005 = doc["naive_and_corner"]["8"]["rejection_rate"]["0.05"]
    if naive_005 < 0.10:
        findings.append(
            f"naive_and_corner rejection rate at n=8, alpha=0.05 is {naive_005}, "
            f"expected clearly above nominal (the module's central finding is "
            f"that it over-rejects) — either the bug was fixed without updating "
            f"this script, or something else changed."
        )

    joint_005 = doc["joint_rank"]["8"]["rejection_rate"]["0.05"]
    if not (0.02 <= joint_005 <= 0.10):
        findings.append(
            f"joint_rank rejection rate at n=8, alpha=0.05 is {joint_005}, "
            f"expected close to nominal 0.05 — the fix may have regressed."
        )

    pp = doc["partial_pass"]
    if not (pp["joint_reject_rate"] < 0.5 * pp["logit_only_reject_rate"]):
        findings.append(
            "partial_pass: joint_reject_rate is not clearly below "
            "logit_only_reject_rate on the falsifier's own configuration."
        )

    iu = doc["intersection_union"]
    for n, rec in iu.items():
        for arm, r in rec["arms"].items():
            rate = r["rejection_rate"]["0.05"]
            if rate > 0.07:
                findings.append(
                    f"intersection_union n={n} arm={arm}: rejection rate at alpha=0.05 "
                    f"is {rate}, above nominal — the statistic does not control the "
                    f"union null on this arm."
                )
    pp_iu = iu["8"]["arms"]["geometry_null_logit_effect"]["rejection_rate"]["0.05"]
    if not (pp_iu <= 0.07 < pp["joint_reject_rate"]):
        findings.append(
            f"on the falsifier's configuration the min-rank statistic rejects at "
            f"{pp['joint_reject_rate']} and intersection-union at {pp_iu}; expected "
            f"the first above nominal and the second at or below it."
        )

    mg = doc["measurement_grid"]
    if mg.get("reachable") and mg.get("n_informative_prompts", 0) < 1:
        findings.append("measurement_grid: reachable but found 0 informative prompts.")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="commit the result to claims/calibration/")
    parser.add_argument("--replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=_SEED)
    args = parser.parse_args()

    t0 = time.time()

    naive = {}
    joint = {}
    for n in N_GRID:
        naive[str(n)] = calibrate_naive_and_corner(
            n_units=n, n_trials=args.replicates, rng=np.random.default_rng(args.seed),
        )
        joint[str(n)] = calibrate_joint_rank(
            n_units=n, n_trials=args.replicates, rng=np.random.default_rng(args.seed),
        )
        print(f"n={n:2d}  naive@0.05={naive[str(n)]['rejection_rate']['0.05']:.3f}  "
              f"joint@0.05={joint[str(n)]['rejection_rate']['0.05']:.3f}", flush=True)

    iu = {}
    for n in N_GRID:
        iu[str(n)] = calibrate_intersection_union(
            n_units=n, n_trials=args.replicates, effect=1.0,
            rng=np.random.default_rng(args.seed),
        )
        arms = iu[str(n)]["arms"]
        print(f"n={n:2d}  IU@0.05  complete={arms['complete_null']['rejection_rate']['0.05']:.3f}  "
              f"geom-null={arms['geometry_null_logit_effect']['rejection_rate']['0.05']:.3f}  "
              f"logit-null={arms['logit_null_geometry_effect']['rejection_rate']['0.05']:.3f}",
              flush=True)

    partial_pass = partial_pass_risk_demo(
        n_units=8, n_trials=args.replicates, logit_effect=1.0, alpha=0.05,
        rng=np.random.default_rng(args.seed),
    )
    print(f"partial_pass: logit_only={partial_pass['logit_only_reject_rate']:.3f} "
          f"joint={partial_pass['joint_reject_rate']:.3f}", flush=True)

    grid = _measurement_grid()
    if grid.get("reachable"):
        print(f"measurement_grid: n_informative_prompts={grid['n_informative_prompts']}", flush=True)
    else:
        print(f"measurement_grid: NOT REACHABLE ({grid.get('reason')})", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/calibrate_p_i5_joint_null.py",
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "replicates": args.replicates,
        "seed": args.seed,
        "floor": _floor_table(),
        "naive_and_corner": naive,
        "joint_rank": joint,
        "intersection_union": iu,
        "partial_pass": partial_pass,
        "measurement_grid": grid,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    findings = check_record(doc)
    doc["self_check"] = {"findings": findings, "ok": not findings}
    print("\nself-check:", "ok" if not findings else f"{len(findings)} finding(s)")
    for f in findings:
        print("  FAIL", f)

    if not args.write:
        print("\n(not written: pass --write)")
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} in {doc['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
