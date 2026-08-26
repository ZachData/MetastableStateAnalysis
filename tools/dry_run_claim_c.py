"""
tools/dry_run_claim_c.py — CLAIM-C's gate, run on inputs whose answer is known.

Five passes built apparatus and `claims/adjudications/` is still empty. This
one adds none: it RUNS the gate that `POPPER_PLAN.md` §§6f-6g shipped, on two
families of input whose correct verdict is fixed a priori, and commits what
came back to `claims/audits/claim_c_dry_run.json`.

THE SELF-COMPARISON, AND WHY IT IS THE SHARP ONE

gpt2-large as BOTH the reference and the candidate. The contrast tables are
then identical, every cell is concordant, and the statistic is at its maximum
in the full set and in all six leave-one-out subsets at once. If the gate does
not return TRANSFERS on that, the criterion does not mean what it says.

There was a live reason to expect it might not, and it is not a subtlety: a
self-comparison inherits whatever prompt-to-prompt sign consistency the
reference phenomenology has, §6g's derived refusal fires above sign homogeneity
of roughly 0.80-0.85, and §6f's identical-rows refusal fires at exactly 1.0.
So the question the dry run actually asks is not "does the criterion work" but
"over what part of its own input space can this gate return TRANSFERS at all".

WHAT CAME BACK (n_prompts = 8, alpha = 0.05; see the committed record)

  sign homogeneity <= 0.8125   TRANSFERS, p = max(2/257, R(h, 2/257))
  0.8333 .. 0.9583             refused: derived homogeneity refusal
  0.9792                       refused: the curve has no measurement in the
                               top bin, because under H0 nearly every draw
                               there hits a refusal and there is no emitted
                               distribution to calibrate against
  1.0000                       refused: identical sign rows

Four readings follow, and the second is the one that matters.

1. **The criterion is not broken, and the tool axis costs nothing here.** On a
   perfect input the observed statistic is the maximum of its own null in every
   subset, so each subset returns exactly the attainable floor and the
   intersection-union max over the seven is that same floor. Unanimity cannot
   bite on a unanimous input, which is the right behaviour and was not
   previously checked.

2. **The gate has an ADMISSIBLE BAND in homogeneity, and outside it the gate is
   a constant function.** At eight prompts the band is `sign_homogeneity` at or
   below 0.8125, which is at least 9 of the 48 candidate cells carrying the
   minority sign for their metric -- on average at least 1.5 of the 8 prompts
   dissenting on each of the six metrics. Above it the gate returns
   INSUFFICIENT for every input: the power curve confirms it at every k from 0
   to 48, so neither TRANSFERS nor FAILS-TO-TRANSFER can be reached and the
   hard stop fires unconditionally. §6g's own caution -- *"a stop rule that
   always fires carries no information"* -- names exactly that region, and this
   is where CLAIM-C's stop rule has one.

   This is not a Type-I defect and not an argument for a weaker correction.
   `sign_homogeneity` is a within-candidate statistic: under H0 it measures the
   prompt redundancy §6g measured and corrected for; under H1 the same number
   also rises with the strength and uniformity of a real effect, and the
   correction cannot tell the two apart. So the cost lands as power, and it
   lands hardest where the effect is most uniform.

3. **How wide the band is, against the two references that fix its scale.**
   Under INDEPENDENT prompt signs -- the most favourable candidate the design
   can be handed -- homogeneity concentrates at 0.637 at eight prompts and the
   refusal fires with probability 1e-4. The band is not tight against chance.
   It is tight against a clean effect: a contrast pointing the same way on
   every prompt sits at exactly 1.0 and is refused with certainty. So the
   design is powered against a contrast carrying a prompt-specific signature
   that transfers, and unpowered against a contrast carrying one uniform
   direction that transfers. Blog 1's phenomenology is the second kind.

   **What that makes this: a requirement on what the pilot must measure,
   computed before it runs** -- the same shape as CLAIM-B's 19 control series
   (§6i). At least ~19% of the candidate's 48 cells must dissent in sign, and
   whether they do is an empirical fact about the run that nothing in this
   repository yet knows. The remedy the refusal names is a remedy on the PROMPT
   SET and not on the correction, and the dry run adds that more prompts will
   not supply it. Expressed as the curve bin the refusal starts in -- the unit
   that is comparable across prompt counts, since the attainable homogeneities
   themselves lie on a grid of step 1/(n_prompts * n_metrics) -- the boundary
   is 0.800-0.825 at six prompts, 0.850-0.875 at seven and nine, and
   0.825-0.850 at eight, ten, eleven and twelve. Three bins of 0.025, with no
   trend: doubling the prompt count does not move it anywhere in particular.

4. **The derived refusal is TIGHT, and that is a positive result.** R(h, .) is
   non-decreasing in p in all 264 tabulated bins, so `R(h, floor) > alpha`
   implies `R(h, p) > alpha` for every attainable p: whenever the refusal
   fires, no input whatsoever could have cleared alpha. It never costs a
   verdict the gate could otherwise have reached; it converts an uninformative
   "p > alpha" into a refusal that says why. §6h found an audit arm reporting
   PASS while incapable of failing; this is the same question asked of a
   refusal, and the answer is the good one.

THE POWER CURVE

The second thing a run wants to know before spending forward passes: how much
concordance does the gate need? For each k from 0 to 48 concordant cells, over
randomly placed arrangements at a fixed candidate sign table, the fraction of
arrangements reaching each verdict. Reported alongside two counterfactual
rates -- the full-set-only p and the uncorrected intersection-union max -- so
the cost of the tool axis and the cost of the homogeneity correction are
separated rather than summed.

PART C -- WHAT THE INFORMATIVE-ROW FLOOR COSTS (added 2026-08-25, §6l)

§6j asked of the derived homogeneity refusal whether it ever refuses something
that could have passed, and answered it from the curve's monotonicity. The
informative-row floor -- a row whose usable cells split exactly half and half,
or whose cells were all dropped, cannot move the statistic, and with fewer than
five rows that can, no p below alpha is expressible -- needs the same question
asked, and here it is asked the expensive way. Tables are drawn at a range of H1
strengths, the whole gate is run, and every table the floor refused is re-scored
through the gate's own `_subset_result` and `apply_homogeneity_correction` to
see what it WOULD have reported. `counterfactual_rejections` must be zero in
every row.

`costs_no_power` is None rather than True when the refusal never fired: a sweep
with nothing to re-score would report success while being incapable of
reporting anything else, which is the audit arm §6h found reporting PASS
without being able to fail.

WHAT THIS DELIBERATELY DOES NOT DO

It adds no third robustness axis -- §6g records why CLAIM-C in particular cannot
afford one. It adjudicates nothing: a dry run on synthetic inputs is not
evidence about pythia-1.4b, and `claims/adjudications/` stays empty.

It no longer says "it changes nothing in the gate", because §6l did: the
informative-row floor is a new refusal in `replication_gate.py` and the
homogeneity curve gained a cell-drop dimension. What is unchanged is this
file's relationship to it -- the dry run measures the gate and never adjusts
it.

RUN IT

    python3 -m tools.dry_run_claim_c --write      # about 5 minutes
    python3 -m tools.dry_run_claim_c --check
    python3 -m tools.dry_run_claim_c --summary

Committed rather than recomputed for the same reason as the other three
artifacts (`docs/CI_BASELINE.md`): about five minutes is far too slow for a
gate people wait on. `tests/test_claim_c_dry_run.py` pins the record, pins the
sha256 of both files it describes, and re-derives the headline boundary from
scratch in milliseconds so the pinned number is checked and not merely stored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "claims" / "audits" / "claim_c_dry_run.json"

#: Bump when the record's shape changes. A reader at a different version
#: refuses rather than reinterpreting fields that may have moved.
RECORD_SCHEMA_VERSION = 2

#: The two files every verdict in this record depends on. The gate decides the
#: verdicts; the curve decides where the refusal boundary sits. Either changing
#: makes the record a description of something that no longer exists, and
#: nothing else in the suite would notice.
GATE_PATH = ROOT / "p1_mstate_tracking" / "replication_gate.py"
CURVE_PATH = ROOT / "claims" / "calibration" / "claim_c_homogeneity.json"

#: Prompt counts the self-comparison is run at. The same set the homogeneity
#: curve tabulates, so every level has a correction to read.
N_PROMPTS_SWEPT: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)

#: Independent placements of the minority signs per homogeneity level. The
#: self-comparison's verdict should depend on the homogeneity and not on where
#: the minority cells sit; running several placements is what turns that from
#: an assumption into a recorded check (`placement_invariant`).
N_PLACEMENTS = 3

#: The power curve's prompt count -- the eight metastability prompts the gate
#: was designed around -- and its arrangement count per (homogeneity, k) cell.
POWER_N_PROMPTS = 8
N_ARRANGEMENTS = 120

#: Fewer arrangements above the refusal boundary: the self-comparison already
#: establishes that a PERFECT input is refused there, so every k is refused
#: there and the row exists to show the zero rather than to estimate a rate.
N_ARRANGEMENTS_ABOVE_BOUNDARY = 20

#: Layer counts. gpt2-large has 36 and pythia-1.4b has 24; the gate resamples
#: both onto DEPTH_GRID_POINTS, and using the real counts means the dry run
#: exercises the depth normalization rather than side-stepping it.
N_LAYERS_REFERENCE = 36
N_LAYERS_CANDIDATE = 24

#: H1 strengths for the informative-row floor's cost measurement -- the
#: per-cell probability that the candidate's contrast agrees with the
#: reference's. 0.50 is H0 and is included because the refusal fires most there.
#: The range stops at 0.70 on purpose: past it the candidate's homogeneity rises
#: into the band where the DERIVED homogeneity refusal fires first, so the
#: informative-row floor is never reached and the row would measure nothing.
INFORMATIVE_FLOOR_STRENGTHS: Tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)
N_INFORMATIVE_FLOOR_DRAWS = 150

_SEED = 20260824

#: Passed to the gate in place of a step-0 arm. The two-baseline policy refuses
#: a silently absent arm, and the honest reason here is that there is no run.
STEP0_ABSENT_REASON = (
    "synthetic dry run: no step-0 arm exists because no checkpoint sweep has "
    "been executed. The step-0 sensitivity arm is reported and never "
    "adjudicated, so its absence changes no verdict in this record."
)


# ---------------------------------------------------------------------------
# Synthetic arms with an exactly prescribed contrast sign table
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_arms(signs: np.ndarray, n_layers: int,
               rng: np.random.Generator) -> Tuple[dict, dict]:
    """
    A (trained, random) arm pair whose per-cell contrast has exactly `signs`.

    `delta` is the mean over normalized depth of (trained - random), so adding
    a positive constant to a shared base profile fixes the sign while leaving
    the profiles themselves unremarkable. The magnitude is randomised because
    the criterion is ordinal and must not depend on it -- if it ever did, this
    is where it would show.
    """
    from p1_mstate_tracking.replication_gate import CLAIM_C_METRICS

    trained: Dict[str, Dict[str, list]] = {}
    random_: Dict[str, Dict[str, list]] = {}
    for i in range(signs.shape[0]):
        prompt = f"prompt_{i:02d}"
        trained[prompt], random_[prompt] = {}, {}
        for j, metric in enumerate(CLAIM_C_METRICS):
            base = rng.normal(size=n_layers)
            bump = float(signs[i, j]) * (0.5 + float(rng.random()))
            random_[prompt][metric] = base.tolist()
            trained[prompt][metric] = (base + bump).tolist()
    return trained, random_


def minority_counts(n_prompts: int, total: int, n_metrics: int) -> List[int]:
    """
    `total` minority-sign prompts spread round-robin over the metrics.

    Homogeneity is the mean over metrics of max(frac +, frac -), which for a
    complete table is `1 - total / (n_prompts * n_metrics)` for ANY spread --
    the distribution over metrics cancels. So sweeping `total` sweeps every
    attainable homogeneity in steps of 1/(n_prompts * n_metrics), and the
    round-robin spread is chosen only because concentrating the minority in one
    metric makes that metric's leave-one-out subset degenerate for reasons that
    have nothing to do with homogeneity.
    """
    cap = n_prompts // 2
    if total > cap * n_metrics:
        raise ValueError(f"total {total} exceeds {cap * n_metrics}")
    out = [0] * n_metrics
    for k in range(total):
        out[k % n_metrics] += 1
    return out


def sign_table(n_prompts: int, mino: Sequence[int],
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """`mino[j]` prompts carry -1 for metric j; the rest carry +1."""
    s = np.ones((n_prompts, len(mino)), dtype=np.int8)
    for j, k in enumerate(mino):
        if not k:
            continue
        if rng is None:
            idx = [(j + 2 * q) % n_prompts for q in range(k)]
            # A deterministic stride can collide; fall back to the first free
            # rows rather than silently planting fewer minority cells.
            idx = sorted(set(idx))
            q = 0
            while len(idx) < k:
                if q not in idx:
                    idx.append(q)
                q += 1
        else:
            idx = rng.choice(n_prompts, size=k, replace=False).tolist()
        s[list(idx), j] = -1
    return s


def homogeneity_of(signs: np.ndarray) -> float:
    """The gate's `sign_homogeneity`, recomputed here for the record."""
    fr = [max((signs[:, j] > 0).mean(), (signs[:, j] < 0).mean())
          for j in range(signs.shape[1])]
    return float(np.mean(fr))


# ---------------------------------------------------------------------------
# Classifying what the gate did
# ---------------------------------------------------------------------------

def refusal_kind(out: dict) -> Optional[str]:
    """
    Which refusal fired, as a stable label rather than a prose match.

    Ordered the way `p_value_claim_c` checks them, so the label names the
    refusal that actually fired and not merely one that would have.
    """
    if out.get("p_value") is not None:
        return None
    if out.get("n_prompts", 0) < 2:
        return "too-few-prompts"
    if out.get("n_cells") == 0:
        return "no-signed-cell"
    if out.get("best_attainable_p") is not None and \
            out["best_attainable_p"] > out.get("alpha", 0.05):
        return "attainable-floor"
    if out.get("sign_rows_identical"):
        return "identical-rows"
    # The data refusals come first in the gate and so they come first here: a
    # table that cannot carry a statistic is never asked whether the curve
    # covers it. Which keys are PRESENT is what orders them, with no prose
    # matching: `informative_rows` is absent until every subset carried a
    # p-value, and `homogeneity_correction` is absent until the informative-row
    # floor has been cleared. See POPPER_PLAN.md 6l.
    if out.get("informative_rows") is None:
        return "subset-undefined"
    if "homogeneity_correction" not in out:
        return "informative-rows-floor"

    corr = out["homogeneity_correction"]
    if not corr.get("available", False):
        # The correction's own branches carry a stable `refusal` marker, so the
        # drop-dimension refusals are distinguishable from the older ones
        # without matching prose.
        marker = corr.get("refusal")
        if marker in ("drop-fraction-above-tabulated", "drop-slab-missing"):
            return marker
        return "no-correction-available"
    return "derived-homogeneity"


def _run_gate(ref_signs: np.ndarray, can_signs: np.ndarray,
              rng: np.random.Generator) -> dict:
    from p1_mstate_tracking.replication_gate import p_value_claim_c

    ref_t, ref_r = build_arms(ref_signs, N_LAYERS_REFERENCE, rng)
    can_t, can_r = build_arms(can_signs, N_LAYERS_CANDIDATE, rng)
    return p_value_claim_c(ref_t, ref_r, can_t, can_r,
                           candidate_step0=None,
                           step0_absent_reason=STEP0_ABSENT_REASON)


# ---------------------------------------------------------------------------
# Part A -- the self-comparison
# ---------------------------------------------------------------------------

def self_comparison_for_n(n_prompts: int, rng: np.random.Generator) -> dict:
    """
    The gate run with one model as both arms, across every attainable
    homogeneity at this prompt count.

    Concordance is perfect by construction, so the ONLY thing that varies is
    the homogeneity -- which is exactly the point. `placement_invariant`
    records whether the verdict depended on where the minority cells sat; it
    should not, and a False there would be a finding rather than a nuisance.
    """
    from p1_mstate_tracking.replication_gate import CLAIM_C_METRICS

    n_metrics = len(CLAIM_C_METRICS)
    n_cells = n_prompts * n_metrics
    levels: List[dict] = []
    for total in range(0, (n_prompts // 2) * n_metrics + 1):
        mino = minority_counts(n_prompts, total, n_metrics)
        verdicts, kinds, ps, uncs, obs = set(), set(), set(), set(), set()
        for placement in range(N_PLACEMENTS):
            signs = sign_table(n_prompts, mino,
                               None if placement == 0 else rng)
            out = _run_gate(signs, signs, rng)
            verdicts.add(out["verdict"])
            kinds.add(refusal_kind(out))
            ps.add(None if out.get("p_value") is None
                   else round(float(out["p_value"]), 12))
            uncs.add(None if out.get("p_value_uncorrected") is None
                     else round(float(out["p_value_uncorrected"]), 12))
            obs.add(out.get("observed"))
            last = out
        levels.append({
            "n_minority_cells": total,
            "minority_per_metric": mino,
            "homogeneity": float(last.get("sign_homogeneity")),
            "sign_rows_identical": bool(last.get("sign_rows_identical")),
            "observed": last.get("observed"),
            "n_cells": last.get("n_cells"),
            "verdict": last["verdict"],
            "refusal_kind": refusal_kind(last),
            "p_value": (None if last.get("p_value") is None
                        else float(last["p_value"])),
            "p_value_uncorrected": (None if last.get("p_value_uncorrected") is None
                                    else float(last["p_value_uncorrected"])),
            "placement_invariant": (len(verdicts) == 1 and len(kinds) == 1
                                    and len(ps) == 1 and len(uncs) == 1),
            "n_placements": N_PLACEMENTS,
        })

    passing = [lv for lv in levels if lv["verdict"] == "TRANSFERS"]
    floor = 2.0 / (2 ** n_prompts + 1.0)
    return {
        "n_prompts": n_prompts,
        "n_cells": n_cells,
        "best_attainable_p": floor,
        # On a perfect input every subset's statistic is the maximum of its own
        # null, so the uncorrected IUT max is exactly the attainable floor. If
        # this is ever False the tool axis is doing something to a unanimous
        # input, which it must not.
        "perfect_input_hits_floor": all(
            lv["p_value_uncorrected"] is not None
            and abs(lv["p_value_uncorrected"] - floor) < 1e-12
            for lv in passing),
        "all_cells_concordant": all(lv["observed"] == lv["n_cells"]
                                    for lv in levels),
        "max_passing_homogeneity": (max(lv["homogeneity"] for lv in passing)
                                    if passing else None),
        "min_refusing_homogeneity": (
            min((lv["homogeneity"] for lv in levels
                 if lv["verdict"] != "TRANSFERS"), default=None)),
        "min_minority_cells_to_pass": (
            min((lv["n_minority_cells"] for lv in passing), default=None)),
        "n_levels": len(levels),
        "n_levels_passing": len(passing),
        "all_placements_invariant": all(lv["placement_invariant"]
                                        for lv in levels),
        "refusal_kinds_seen": sorted(
            {lv["refusal_kind"] for lv in levels if lv["refusal_kind"]}),
        "levels": levels,
    }


def refusal_boundary_bins(alpha: float = 0.05) -> dict:
    """
    The curve bin in which the derived homogeneity refusal starts, per prompt
    count -- the boundary expressed in a unit that is comparable ACROSS prompt
    counts.

    `max_passing_homogeneity` is not: the attainable homogeneities form a grid
    of step 1/(n_prompts * n_metrics), so that number moves with the grid as
    well as with the boundary. The bin edge is the boundary itself.

    Read on the COMPLETE-TABLE slab of the curve, which since 2026-08-25 is one
    of several (POPPER_PLAN.md 6l). That is the slab the band reported in 6j was
    measured on, so reading it here keeps this number comparable with what is on
    record; the boundary at other drop rates is a separate question and the
    curve now answers it per slab.
    """
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_METRICS, apply_homogeneity_correction,
        homogeneity_correction, load_homogeneity_curve)

    curve = load_homogeneity_curve()
    out = {}
    for n, row in sorted(curve["curves"].items(), key=lambda kv: int(kv[0])):
        floor = row["best_attainable_p"]
        n_cells = int(n) * len(CLAIM_C_METRICS)
        first_bad = None
        for b in row["drop_bins"][0]["bins"]:
            mid = (b["lo"] + b["hi"]) / 2.0
            corr = homogeneity_correction(int(n), mid, 0, n_cells)
            if not corr["available"]:
                r = None
            else:
                r = apply_homogeneity_correction(corr, floor, CLAIM_C_ALTERNATIVE)
            if r is None or r > alpha:
                first_bad = {"bin_lo": b["lo"], "bin_hi": b["hi"],
                             "rate_at_floor": r}
                break
        out[str(int(n))] = {"best_attainable_p": floor,
                            "first_refusing_bin": first_bad,
                            "drop_slab": "no cells dropped"}
    return out


def independent_sign_reference(n_prompts: int, n_metrics: int) -> dict:
    """
    What `sign_homogeneity` looks like when the candidate's prompt signs are
    INDEPENDENT fair coins -- the most favourable case the design can be handed.

    Computed exactly rather than sampled: per metric the majority count is
    Binomial(n, 1/2), per-metric homogeneity is max(k, n-k)/n, and the mean over
    n_metrics independent metrics follows by convolution. Reported because it
    fixes the scale the refusal boundary has to be read against -- homogeneity
    0.5 is not a thing a real run can be near, so the question is not "is 0.81
    high" but "how much of the independent-prompt distribution already sits
    above it".
    """
    from math import comb

    per = {}
    for k in range(n_prompts + 1):
        v = max(k, n_prompts - k)          # majority count
        per[v] = per.get(v, 0.0) + comb(n_prompts, k) / 2.0 ** n_prompts
    # distribution of the SUM of majority counts over n_metrics
    dist = {0: 1.0}
    for _ in range(n_metrics):
        nxt: dict = {}
        for s, ps in dist.items():
            for v, pv in per.items():
                nxt[s + v] = nxt.get(s + v, 0.0) + ps * pv
        dist = nxt
    denom = float(n_prompts * n_metrics)
    mean = sum(s * p for s, p in dist.items()) / denom
    return {
        "n_prompts": n_prompts,
        "n_metrics": n_metrics,
        "mean_homogeneity": mean,
        "distribution": sorted(
            ({"homogeneity": s / denom, "probability": p}
             for s, p in dist.items() if p > 1e-12),
            key=lambda d: d["homogeneity"]),
    }

def correction_is_monotone() -> dict:
    """
    Is R(h, .) non-decreasing in p in every tabulated bin?

    This is what makes the derived refusal TIGHT: if `R(h, floor) > alpha` and
    R is non-decreasing, then no attainable p clears alpha, so the refusal
    never costs a verdict the gate could otherwise have reached. Checked here
    rather than asserted, because the property belongs to the committed curve
    and not to the code that reads it.

    Since 2026-08-25 it walks every DROP slab as well as every homogeneity bin,
    so the tightness claim covers the cell-drop dimension rather than only the
    complete-table one it was first made on.
    """
    from p1_mstate_tracking.replication_gate import load_homogeneity_curve

    curve = load_homogeneity_curve()
    violations = []
    n_checked = 0
    for n, row in sorted(curve["curves"].items()):
        for slab in row["drop_bins"]:
            for b in slab["bins"]:
                for key in ("quantiles_greater", "quantiles_less"):
                    q = b.get(key)
                    if q is None:
                        continue
                    n_checked += 1
                    for i in range(len(q) - 1):
                        if q[i + 1] < q[i] - 1e-12:
                            violations.append(
                                {"n_prompts": int(n), "bin_lo": b["lo"],
                                 "drop_bin_index": slab["drop_bin_index"],
                                 "tail": key, "index": i,
                                 "values": [q[i], q[i + 1]]})
    return {"n_quantile_vectors_checked": n_checked,
            "monotone": not violations,
            "violations": violations[:8]}


def informative_row_floor_cost(n_prompts: int, strengths: Sequence[float],
                               n_draws: int, rng: np.random.Generator) -> dict:
    """
    What the informative-row floor refusal costs, measured on inputs that carry
    signal rather than on H0.

    POPPER_PLAN.md 6j asked of the derived homogeneity refusal whether it ever
    refuses something that could have passed, and answered it from the curve's
    monotonicity. The informative-row floor (6l) needs the same question asked,
    and here it is asked the expensive way: draw tables at a range of H1
    strengths, run the whole gate, and for every draw the floor refuses,
    recompute what the gate WOULD have reported and check that neither tail
    clears alpha.

    `counterfactual_rejections` is the number that matters and it must be zero.
    A single non-zero entry means the refusal took a verdict away, and the
    refusal would have to go.

    The counterfactual is built from `_subset_result` and
    `apply_homogeneity_correction` -- the gate's own scoring functions, called
    directly -- rather than from a second implementation of the arithmetic.
    """
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_METRICS, CLAIM_C_RECIPROCAL_ALTERNATIVE,
        DEFAULT_N_PERM, _alpha, _metric_subsets, _subset_result,
        apply_homogeneity_correction, homogeneity_correction)

    n_m = len(CLAIM_C_METRICS)
    alpha = float(_alpha())
    rows = []
    for q in strengths:
        counts = {"TRANSFERS": 0, "FAILS-TO-TRANSFER": 0, "INSUFFICIENT": 0}
        n_refused = counterfactual_rejections = n_checked = 0
        for _ in range(n_draws):
            can = rng.choice([-1, 1], size=(n_prompts, n_m)).astype(np.int8)
            agree = rng.random((n_prompts, n_m)) < q
            ref = (can * np.where(agree, 1, -1)).astype(np.int8)
            out = _run_gate(ref, can, rng)
            counts[out["verdict"]] += 1
            if refusal_kind(out) != "informative-rows-floor":
                continue
            n_refused += 1

            # The gate never computed a correction for this table -- the
            # floor refusal fires before that block -- so the counterfactual
            # has to look it up here. That is the whole point: what WOULD the
            # gate have reported had the floor not turned the table away.
            full = homogeneity_correction(
                out["n_prompts"], out["sign_homogeneity"],
                out["n_cells_dropped"], out["n_prompts"] * n_m)
            if not full.get("available"):
                continue
            concordant = np.asarray(out["concordant"], dtype=bool)
            usable = np.asarray(out["usable"], dtype=bool)
            sign_can = np.sign(np.asarray(out["contrast_candidate"],
                                          dtype=np.float64))
            pg = pl = 0.0
            for _name, cols in _metric_subsets():
                sub = _subset_result(concordant, usable, sign_can, cols,
                                     n_perm=DEFAULT_N_PERM, seed=0)
                if sub.get("p_value") is None:
                    pg = pl = 1.0
                    break
                pg = max(pg, sub["p_value"])
                pl = max(pl, sub["p_reciprocal"])
            n_checked += 1
            if (apply_homogeneity_correction(full, pg, CLAIM_C_ALTERNATIVE) <= alpha
                    or apply_homogeneity_correction(
                        full, pl, CLAIM_C_RECIPROCAL_ALTERNATIVE) <= alpha):
                counterfactual_rejections += 1
        rows.append({
            "h1_strength": float(q),
            "n_draws": int(n_draws),
            "refusal_rate": n_refused / float(n_draws),
            "verdicts": counts,
            "p_transfers": counts["TRANSFERS"] / float(n_draws),
            "n_refusals_rescored": int(n_checked),
            "counterfactual_rejections": int(counterfactual_rejections),
        })
    return {
        "_what": ("the informative-row floor refusal, measured on inputs that "
                  "carry signal: how often it fires at each H1 strength, and "
                  "how many of the tables it refused could have cleared alpha "
                  "in either tail"),
        "_the_number_that_matters": (
            "counterfactual_rejections, which must be 0 in every row. The "
            "refusal is only safe because both tails share the floor; a "
            "non-zero entry means it took a verdict away and it would have to "
            "go. POPPER_PLAN.md 6j asked this of the derived homogeneity "
            "refusal and answered it from the curve's monotonicity; this is the "
            "same question asked of 6l's refusal by running the gate."),
        "n_prompts": int(n_prompts),
        "alpha": alpha,
        "rows": rows,
        "n_refusals_rescored": sum(r["n_refusals_rescored"] for r in rows),
        # None, never True, when nothing was rescored. A sweep in which the
        # refusal never fired would report "costs no power" while being
        # INCAPABLE of reporting anything else, which is exactly the audit arm
        # POPPER_PLAN.md 6h found reporting PASS without being able to fail. The
        # verdict is evidence or it is absent.
        "costs_no_power": (
            all(r["counterfactual_rejections"] == 0 for r in rows)
            if any(r["n_refusals_rescored"] for r in rows) else None),
        "_costs_no_power_is_none_when": (
            "the refusal never fired in this sweep, so there was nothing to "
            "rescore and the answer would be true by vacuity rather than by "
            "measurement"),
    }


# ---------------------------------------------------------------------------
# Part B -- the power curve
# ---------------------------------------------------------------------------

def power_curve_at(homogeneity_total: int, n_arrangements: int,
                   rng: np.random.Generator) -> dict:
    """
    Verdict rates against the number of concordant cells, at one fixed
    candidate sign table.

    Concordance is varied by flipping REFERENCE cells, which leaves the
    candidate's sign table -- and therefore the homogeneity, and therefore the
    correction and both refusals -- untouched. That is what makes the row a
    power curve at a fixed correction rather than two things moving at once.
    """
    from p1_mstate_tracking.replication_gate import CLAIM_C_METRICS

    n_p, n_m = POWER_N_PROMPTS, len(CLAIM_C_METRICS)
    n_cells = n_p * n_m
    alpha = 0.05
    can = sign_table(n_p, minority_counts(n_p, homogeneity_total, n_m))
    h = homogeneity_of(can)

    rows = []
    for k in range(n_cells + 1):
        counts = {"TRANSFERS": 0, "FAILS-TO-TRANSFER": 0, "INSUFFICIENT": 0}
        refused = 0
        full_set_clears = iut_unc_clears = 0
        ps: List[float] = []
        for _ in range(n_arrangements):
            disc = np.zeros(n_cells, dtype=bool)
            if k < n_cells:
                disc[rng.choice(n_cells, size=n_cells - k, replace=False)] = True
            ref = can * np.where(disc.reshape(n_p, n_m), -1, 1).astype(np.int8)
            out = _run_gate(ref, can, rng)
            counts[out["verdict"]] += 1
            alpha = float(out.get("alpha", alpha))
            if out.get("p_value") is None:
                refused += 1
                continue
            ps.append(float(out["p_value"]))
            full_set_clears += int(out["p_full_set"] <= alpha)
            iut_unc_clears += int(out["p_value_uncorrected"] <= alpha)
        n = float(n_arrangements)
        rows.append({
            "k_concordant": k,
            "frac_concordant": k / n_cells,
            "transfers": counts["TRANSFERS"] / n,
            "fails_to_transfer": counts["FAILS-TO-TRANSFER"] / n,
            "insufficient": counts["INSUFFICIENT"] / n,
            "refused": refused / n,
            "p_min": (min(ps) if ps else None),
            "p_median": (float(np.median(ps)) if ps else None),
            "p_max": (max(ps) if ps else None),
            "full_set_only_clears": full_set_clears / n,
            "iut_uncorrected_clears": iut_unc_clears / n,
        })

    def _first(pred) -> Optional[int]:
        for r in rows:
            if pred(r):
                return r["k_concordant"]
        return None

    def _last(pred) -> Optional[int]:
        found = None
        for r in rows:
            if pred(r):
                found = r["k_concordant"]
        return found

    k_half = _first(lambda r: r["transfers"] >= 0.5)
    k_all = _first(lambda r: all(x["transfers"] >= 1.0 for x in rows
                                 if x["k_concordant"] >= r["k_concordant"]))
    k_fail_half = _last(lambda r: r["fails_to_transfer"] >= 0.5)
    return {
        "homogeneity": h,
        "n_minority_cells": homogeneity_total,
        "n_arrangements": n_arrangements,
        "n_cells": n_cells,
        "alpha": alpha,
        "thresholds": {
            "k_transfers_half": k_half,
            "frac_transfers_half": (None if k_half is None else k_half / n_cells),
            "k_transfers_always": k_all,
            "frac_transfers_always": (None if k_all is None else k_all / n_cells),
            "k_fails_half": k_fail_half,
            "frac_fails_half": (None if k_fail_half is None
                                else k_fail_half / n_cells),
            "insufficient_band": (
                None if (k_half is None and k_fail_half is None)
                else [(0 if k_fail_half is None else k_fail_half + 1),
                      (n_cells if k_half is None else k_half - 1)]),
            "k_full_set_only_half": _first(
                lambda r: r["full_set_only_clears"] >= 0.5),
            "k_iut_uncorrected_half": _first(
                lambda r: r["iut_uncorrected_clears"] >= 0.5),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Assembling the record
# ---------------------------------------------------------------------------

def build_record(seed: int = _SEED) -> dict:
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_EXCHANGEABLE_UNIT, CLAIM_C_METRICS,
        CLAIM_C_TOOL_AXIS, CLAIM_C_TOOL_RULE, DEPTH_GRID_POINTS, _alpha)

    rng = np.random.default_rng(seed)
    n_m = len(CLAIM_C_METRICS)
    self_cmp = {str(n): self_comparison_for_n(n, rng) for n in N_PROMPTS_SWEPT}

    # Three levels at or below the boundary the self-comparison locates, and
    # one above it. The one above exists to show the zero: a perfect input is
    # refused there, so no k can clear.
    boundary_total = self_cmp[str(POWER_N_PROMPTS)]["min_minority_cells_to_pass"]
    if boundary_total is None:
        raise RuntimeError(
            "the self-comparison never returned TRANSFERS at "
            f"{POWER_N_PROMPTS} prompts, so there is no boundary to sit the "
            "power curve below. That is itself the finding -- do not paper "
            "over it by picking a homogeneity.")
    totals = sorted({boundary_total, boundary_total + 3, boundary_total + 9})
    curves = [power_curve_at(t, N_ARRANGEMENTS, rng) for t in totals]
    if boundary_total >= 1:
        curves.insert(0, power_curve_at(boundary_total - 1,
                                        N_ARRANGEMENTS_ABOVE_BOUNDARY, rng))

    alpha = float(_alpha())
    floor_cost = informative_row_floor_cost(
        POWER_N_PROMPTS, INFORMATIVE_FLOOR_STRENGTHS,
        N_INFORMATIVE_FLOOR_DRAWS, rng)
    bins = refusal_boundary_bins(alpha)
    indep = {}
    for n in N_PROMPTS_SWEPT:
        ref = independent_sign_reference(n, n_m)
        edge = (bins[str(n)]["first_refusing_bin"] or {}).get("bin_lo")
        ref["refusal_boundary_bin_lo"] = edge
        ref["p_above_refusal_boundary"] = (
            None if edge is None
            else float(sum(d["probability"] for d in ref["distribution"]
                           if d["homogeneity"] >= edge)))
        # The full distribution is 6 * n_prompts values wide and the record is
        # read by people; keep the summary and drop the tail listing.
        ref.pop("distribution")
        indep[str(n)] = ref

    boundaries = {str(n): self_cmp[str(n)]["max_passing_homogeneity"]
                  for n in N_PROMPTS_SWEPT}
    passing_bounds = [v for v in boundaries.values() if v is not None]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "_what": (
            "CLAIM-C's replication gate run on inputs whose correct verdict is "
            "known a priori: a self-comparison (one model as both arms, so "
            "every cell is concordant) and a power curve over the number of "
            "concordant cells."),
        "_why": (
            "Five passes built apparatus and claims/adjudications/ is still "
            "empty. This validates the shipped gate rather than extending it, "
            "and it can fail in a way no synthetic unit test catches."),
        "_not": (
            "Not evidence about pythia-1.4b and not an adjudication. The "
            "inputs are synthetic; what is being checked is the gate."),
        "generated_by": "tools/dry_run_claim_c.py --write",
        "gate_file": str(GATE_PATH.relative_to(ROOT)),
        "gate_sha256": _sha256(GATE_PATH),
        "curve_file": str(CURVE_PATH.relative_to(ROOT)),
        "curve_sha256": _sha256(CURVE_PATH),
        "alpha": alpha,
        "metrics": list(CLAIM_C_METRICS),
        "n_metrics": n_m,
        "exchangeable_unit": CLAIM_C_EXCHANGEABLE_UNIT,
        "alternative": CLAIM_C_ALTERNATIVE,
        "tool_axis": CLAIM_C_TOOL_AXIS,
        "tool_rule": CLAIM_C_TOOL_RULE,
        "depth_grid_points": int(DEPTH_GRID_POINTS),
        "n_layers_reference": N_LAYERS_REFERENCE,
        "n_layers_candidate": N_LAYERS_CANDIDATE,
        "step0_absent_reason": STEP0_ABSENT_REASON,
        "seed": int(seed),
        "self_comparison": {
            "_what": (
                "one model as BOTH reference and candidate, so the contrast "
                "tables are identical and every cell is concordant. The "
                "correct verdict is TRANSFERS at every homogeneity; where it "
                "is not, the reason is recorded."),
            "n_prompts_swept": list(N_PROMPTS_SWEPT),
            "max_passing_homogeneity_by_n": boundaries,
            "boundary_range": [min(passing_bounds), max(passing_bounds)]
                              if passing_bounds else None,
            "boundary_moves_with_n": (len(set(passing_bounds)) > 1
                                      if passing_bounds else None),
            "per_n_prompts": self_cmp,
        },
        "refusal_boundary_bins": bins,
        "independent_prompt_reference": {
            "_what": (
                "sign_homogeneity when the candidate's prompt signs are "
                "independent fair coins -- the most favourable input the "
                "design can be handed. `p_above_refusal_boundary` is the share "
                "of that distribution the derived refusal already rejects, so "
                "it reads as the gate's refusal rate on the friendliest "
                "possible candidate."),
            "per_n_prompts": indep,
        },
        "correction_monotonicity": correction_is_monotone(),
        "informative_row_floor": floor_cost,
        "power_curve": {
            "_what": (
                "verdict rates against the number of concordant cells, at a "
                "candidate sign table held fixed so the homogeneity correction "
                "does not move with the effect size. `full_set_only_clears` "
                "and `iut_uncorrected_clears` are counterfactual rates that "
                "separate what the tool axis costs from what the correction "
                "costs."),
            "n_prompts": POWER_N_PROMPTS,
            "n_cells": POWER_N_PROMPTS * n_m,
            "levels": curves,
        },
    }


# ---------------------------------------------------------------------------
# Staleness, summary, CLI
# ---------------------------------------------------------------------------

def check_record(path: Path = RECORD_PATH) -> List[str]:
    """
    Is the committed record still about the files on disk, and self-consistent?

    Deliberately does NOT re-run the gate: the whole reason the record is
    committed is that five minutes is too slow for a gate people wait on. What
    can go stale is the pair of files it describes, so that is what is hashed.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with "
                f"`python3 -m tools.dry_run_claim_c --write`"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != RECORD_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{RECORD_SCHEMA_VERSION}; regenerate with --write")
    for key, described in (("gate", GATE_PATH), ("curve", CURVE_PATH)):
        if not described.exists():
            problems.append(f"{described} is missing")
            continue
        on_disk = _sha256(described)
        if rec.get(f"{key}_sha256") != on_disk:
            problems.append(
                f"{described.name} has changed since the dry run was written "
                f"(sha256 {on_disk[:12]} on disk vs "
                f"{str(rec.get(f'{key}_sha256'))[:12]} on record). The record "
                f"describes something that no longer exists in that form; "
                f"rerun --write rather than editing the hash.")

    from p1_mstate_tracking.replication_gate import CLAIM_C_METRICS
    if list(rec.get("metrics", [])) != list(CLAIM_C_METRICS):
        problems.append(
            "the dry run was made on a different metric set than "
            "CLAIM_C_METRICS; its verdicts are verdicts of a different test")

    sc = rec.get("self_comparison", {}).get("per_n_prompts", {})
    if not sc:
        problems.append("self_comparison is missing from the record")
    for n, row in sc.items():
        if not row.get("all_cells_concordant"):
            problems.append(
                f"n_prompts={n}: the self-comparison did NOT score every cell "
                f"concordant, so it was not a self-comparison")
        if not row.get("perfect_input_hits_floor"):
            problems.append(
                f"n_prompts={n}: a perfect input did not reach the attainable "
                f"floor, so the tool axis is biting on a unanimous input")
        if not row.get("all_placements_invariant"):
            problems.append(
                f"n_prompts={n}: the verdict depended on WHERE the minority "
                f"signs were placed, not only on the homogeneity")
    if not rec.get("correction_monotonicity", {}).get("monotone"):
        problems.append(
            "R(h, .) is not non-decreasing in p, so the derived homogeneity "
            "refusal is no longer tight: it can refuse a run that would have "
            "cleared alpha")

    floor = rec.get("informative_row_floor", {})
    if floor.get("costs_no_power") is not True:
        problems.append(
            "the informative-row floor's cost was not established: "
            f"costs_no_power={floor.get('costs_no_power')!r} over "
            f"{floor.get('n_refusals_rescored')} rescored refusals. None means "
            f"the refusal never fired in the sweep, so the record proves "
            f"nothing about it; False means it took a verdict away and the "
            f"refusal has to go.")
    if not floor.get("n_refusals_rescored"):
        problems.append(
            "no refusal was rescored, so `costs_no_power` would be true by "
            "vacuity -- an arm reporting PASS while incapable of failing, "
            "which POPPER_PLAN.md 6h records as a defect in its own right")
    return problems


def print_summary(rec: dict) -> None:
    print(f"gate:  {rec['gate_file']}  sha256 {rec['gate_sha256'][:12]}")
    print(f"curve: {rec['curve_file']}  sha256 {rec['curve_sha256'][:12]}")
    print(f"alpha: {rec['alpha']}\n")

    sc = rec["self_comparison"]
    print("=== self-comparison: one model as BOTH arms, every cell concordant ===")
    print(f"boundary moves with n: {sc['boundary_moves_with_n']}   "
          f"range {sc['boundary_range']}")
    print(f"{'n':>3}  {'floor':>8}  {'max h passing':>13}  {'min minority cells':>18}"
          f"  refusals seen")
    for n in rec["self_comparison"]["n_prompts_swept"]:
        row = sc["per_n_prompts"][str(n)]
        print(f"{n:>3}  {row['best_attainable_p']:>8.5f}  "
              f"{row['max_passing_homogeneity']:>13.4f}  "
              f"{row['min_minority_cells_to_pass']:>18d}"
              f"  {', '.join(row['refusal_kinds_seen'])}")

    row = sc["per_n_prompts"][str(rec["power_curve"]["n_prompts"])]
    print(f"\nat {row['n_prompts']} prompts, level by level:")
    for lv in row["levels"]:
        p = "-" if lv["p_value"] is None else f"{lv['p_value']:.5f}"
        print(f"  h={lv['homogeneity']:.4f}  minority={lv['n_minority_cells']:2d}"
              f"  p={p:>8}  {lv['verdict']:<17} {lv['refusal_kind'] or ''}")

    print("\nrefusal boundary as a curve bin (comparable across prompt counts),")
    print("and where an INDEPENDENT-prompt candidate would sit:")
    print(f"{'n':>3}  {'first refusing bin':>19}  {'R at floor':>10}  "
          f"{'E[h] indep':>10}  {'P(refuse) indep':>15}")
    for n in rec["self_comparison"]["n_prompts_swept"]:
        b = rec["refusal_boundary_bins"][str(n)]["first_refusing_bin"] or {}
        ind = rec["independent_prompt_reference"]["per_n_prompts"][str(n)]
        lo, hi = b.get("bin_lo"), b.get("bin_hi")
        rate = b.get("rate_at_floor")
        print(f"{n:>3}  {f'{lo:.3f}-{hi:.3f}' if lo is not None else '-':>19}  "
              f"{'-' if rate is None else f'{rate:.4f}':>10}  "
              f"{ind['mean_homogeneity']:>10.4f}  "
              f"{ind['p_above_refusal_boundary']:>15.2e}")

    mono = rec["correction_monotonicity"]
    print(f"\nR(h,.) non-decreasing in p over {mono['n_quantile_vectors_checked']} "
          f"tabulated bins: {mono['monotone']}  -> the derived refusal is "
          f"{'tight' if mono['monotone'] else 'NOT tight'}")

    fl = rec["informative_row_floor"]
    print(f"\n=== the informative-row floor at {fl['n_prompts']} prompts ===")
    print(f"{'H1 strength':>11}  {'refusal rate':>12}  {'P(TRANSFERS)':>12}  "
          f"{'rescored':>8}  {'could have cleared':>18}")
    for r in fl["rows"]:
        print(f"{r['h1_strength']:>11.2f}  {r['refusal_rate']:>12.4f}  "
              f"{r['p_transfers']:>12.4f}  {r['n_refusals_rescored']:>8d}  "
              f"{r['counterfactual_rejections']:>18d}")
    print(f"costs_no_power: {fl['costs_no_power']}  "
          f"({fl['n_refusals_rescored']} refusals re-scored; None means the "
          f"refusal never fired and the answer would be vacuous)")

    print("\n=== power curve at "
          f"{rec['power_curve']['n_prompts']} prompts, "
          f"{rec['power_curve']['n_cells']} cells ===")
    for lvl in rec["power_curve"]["levels"]:
        t = lvl["thresholds"]
        def _f(k):
            return "-" if k is None else f"{k:d}"
        print(f"  h={lvl['homogeneity']:.4f} ({lvl['n_arrangements']} arrangements)"
              f"  TRANSFERS>=50% at k={_f(t['k_transfers_half'])}"
              f", always at k={_f(t['k_transfers_always'])}"
              f"; FAILS>=50% at k<={_f(t['k_fails_half'])}"
              f"; INSUFFICIENT band {t['insufficient_band']}")
        print(f"        counterfactuals: full-set-only 50% at "
              f"k={_f(t['k_full_set_only_half'])}, uncorrected IUT 50% at "
              f"k={_f(t['k_iut_uncorrected_half'])}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run the dry run and write the record (~5 minutes)")
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
