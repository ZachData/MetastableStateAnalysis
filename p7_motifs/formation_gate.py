"""
p7_motifs/formation_gate.py — P-I1's gate (experiment 7-A, the formation curve).

    P-I1  `relay` motif strength above N1 and N2 first rises in the same
          checkpoint window as the behavioral induction score.
    Falsifier: motif already above nulls at step 0, or absent at step 143,000
          despite a high behavioral score.

The construction is `core/changepoint_colocation.py`'s and is deliberately NOT
reinvented here. `claims/EVALUABILITY.md` closes by naming this: P-I1 "shares a
construction with CLAIM-B -- the same changepoint co-location across a
checkpoint sweep -- so the two should be built together rather than each
inventing one." This module is the thin half: which series, which direction,
which unit, and the two preconditions that belong to P-I1 alone.

WHAT IS DIFFERENT FROM CLAIM-B, AND WHY

**One arm, not three.** P-I1 names no literature anchor -- it asks only that the
two curves rise together -- so there is nothing for an anchor arm to test and
none is invented. The mutual arm is the whole gate.

**The unit is the HEAD, and that is a registered constraint rather than a
convenience.** `PREDICTIONS.md`'s first adjudication constraint for Phase 7:
*"Effective n is the number of heads, not the number of edges. Edges within a
head are not independent samples. Any significance computed over edge counts is
wrong by orders of magnitude, in the direction that manufactures findings."*
The pairing null permutes which head's motif curve is matched with which head's
behavioral curve, so the head is the unit by construction and an edge-level n
cannot enter.

**The series handed in must already be the ABOVE-NULL excess.** P-I1's wording
is "strength above N1 and N2", and a pass requires clearing both
(`core/qk_offset_null.py`). Clearing them is `p7_motifs/motif_stats.py`'s job,
not this module's; what arrives here is the excess, per head, per checkpoint.
This module cannot check that and says so rather than implying it did.

**The falsifier's second half is a PRECONDITION, not a p-value.** "Motif already
above nulls at step 0, or absent at step 143,000 despite a high behavioral
score" is a statement about the endpoints of the curve, not about co-location,
and folding it into the co-location statistic would put two different questions
in one number. `endpoint_flags` reports it beside the result, computed from the
series' own first and last checkpoints, and it enters no p-value.

THE TAUTOLOGY RISK, WHICH THIS MODULE CANNOT DISCHARGE

`PREDICTIONS.md`'s second adjudication constraint: the behavioral induction
score is "mean attention on induction pairs", and a motif defined as "attentive
edge on induction pairs" is the same number. It is stated against P-I3 and it
applies here with equal force -- two identical series co-locate perfectly and
the gate would report p at its floor. No null detects that, because the null is
over the PAIRING and a tautological pair is tautological at every head. So the
gate records `series_identity` -- the correlation between the two curves' change
locations is what it tests, and a caller handing in the same series twice gets a
refusal -- and the independence source stays a claim the analyst has to make in
the record, exactly as the constraint requires.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from core.changepoint_colocation import (
    ALTERNATIVE,
    ColocationRefused,
    _SEED,
    combine_arms,
    gate_verdict,
    paired_colocation_arm,
    spacing_change_report,
)

#: Both series are predicted to RISE. Registered, with no default: the module
#: that owns the constant is the one that can be checked against the prediction.
P_I1_SERIES: Dict[str, Dict[str, str]] = {
    "relay_strength": {
        "field": "p7_motifs.motif_stats: `relay` strength MINUS the N1/N2 "
                 "offset-null envelope (core/qk_offset_null.py)",
        "direction": "rise",
        "why": "P-I1 says the above-null strength first RISES",
    },
    "induction_score": {
        "field": "behavioral induction score (mean attention on induction pairs)",
        "direction": "rise",
        "why": "the behavioral score rises as the head forms",
    },
}

#: The exchangeable unit. See the module docstring: PREDICTIONS.md fixes it.
P_I1_UNIT = "head"

#: Pythia's first and last released checkpoints, which are the two the
#: falsifier names. Taken from the falsifier's wording, not chosen here.
P_I1_ENDPOINT_STEPS = (0, 143000)


def endpoint_flags(steps: Sequence[float],
                   relay_strength: Sequence[Sequence[float]],
                   induction_score: Sequence[Sequence[float]]) -> dict:
    """
    The falsifier's second half, reported and never scored.

    "Motif already above nulls at step 0" is `relay_strength > 0` at the first
    checkpoint -- the series is an above-null EXCESS, so zero is the null
    envelope and the comparison needs no placed threshold. "Absent at step
    143,000 despite a high behavioral score" is the excess at or below zero at
    the last checkpoint while the behavioral score is at its own maximum there.

    Both are per head and both are counts, not verdicts. A reader of the record
    decides what they mean; this function makes sure they are in the record.
    """
    s = np.asarray(steps, dtype=np.float64)
    r = np.asarray([np.asarray(v, dtype=np.float64) for v in relay_strength])
    b = np.asarray([np.asarray(v, dtype=np.float64) for v in induction_score])
    first_at_or_above = r[:, 0] > 0.0
    last_absent = r[:, -1] <= 0.0
    behav_peaks_last = b[:, -1] >= b.max(axis=1)
    return {
        "first_step": float(s[0]),
        "last_step": float(s[-1]),
        "endpoint_steps_expected": list(P_I1_ENDPOINT_STEPS),
        "n_heads": int(r.shape[0]),
        "n_heads_above_null_at_first_step": int(first_at_or_above.sum()),
        "n_heads_absent_at_last_step": int(last_absent.sum()),
        "n_heads_absent_at_last_step_with_peak_behaviour":
            int((last_absent & behav_peaks_last).sum()),
        "_note": (
            "P-I1's falsifier names both as failure modes. They are reported "
            "and enter NO p-value: they are statements about the curve's "
            "endpoints and the gate's statistic is about where it rises, and "
            "one number cannot carry both questions."),
    }


def p_value_p_i1(steps: Sequence[float],
                 relay_strength: Sequence[Sequence[float]],
                 induction_score: Sequence[Sequence[float]],
                 *,
                 alpha: Optional[float] = None,
                 seed: int = _SEED) -> dict:
    """
    P-I1's p-value. One arm: do the two curves' rises co-locate across heads,
    more than an arbitrary pairing of the same two populations allows?

    Refuses -- `p_value` None with a `reason` -- rather than returning a number
    the design cannot support.
    """
    out: dict = {
        "prediction_id": "P-I1",
        "claim": "H-BRIDGE",
        "series": P_I1_SERIES,
        "unit": P_I1_UNIT,
        "spacing": None,
        "endpoint_flags": None,
        "p_value": None,
        "p_reciprocal": None,
        "reason": None,
    }
    try:
        out["spacing"] = spacing_change_report(steps)
        rs, bs = list(relay_strength), list(induction_score)
        if len(rs) != len(bs):
            raise ColocationRefused(
                f"{len(rs)} heads of relay strength against {len(bs)} of "
                f"behavioral score; these index the same heads and must match")
        for i, (a, b) in enumerate(zip(rs, bs)):
            if np.array_equal(np.asarray(a, dtype=np.float64),
                              np.asarray(b, dtype=np.float64)):
                raise ColocationRefused(
                    f"head {i}: the two series are identical. That is the "
                    f"tautology PREDICTIONS.md's second adjudication constraint "
                    f"names -- the behavioral score is 'mean attention on "
                    f"induction pairs' and a motif defined as 'attentive edge on "
                    f"induction pairs' is the same number -- and no null detects "
                    f"it, because a tautological pair is tautological at every "
                    f"head.")
        out["endpoint_flags"] = endpoint_flags(steps, rs, bs)
        arm = paired_colocation_arm(
            steps, rs, P_I1_SERIES["relay_strength"]["direction"],
            bs, P_I1_SERIES["induction_score"]["direction"],
            alpha=(alpha if alpha is not None else _gate_alpha()),
            unit_name=P_I1_UNIT, arm_name="mutual", seed=seed)
        comb = combine_arms([arm])
    except ColocationRefused as exc:
        out["reason"] = str(exc)
        out.update(gate_verdict(None, None, alpha))
        return out

    out.update({k: comb[k] for k in
                ("p_value", "p_reciprocal", "n_arms", "binding_arm", "rule", "arms")})
    out.update(gate_verdict(comb["p_value"], comb["p_reciprocal"], alpha))
    return out


def _gate_alpha() -> float:
    from core.changepoint_colocation import _alpha
    return _alpha()


def adjudicate_p_i1(steps: Sequence[float],
                    relay_strength: Sequence[Sequence[float]],
                    induction_score: Sequence[Sequence[float]],
                    *,
                    alpha: Optional[float] = None,
                    seed: int = _SEED,
                    artifact_hashes: Sequence[str] = (),
                    run_manifest: Optional[dict] = None,
                    adjudicate: bool = False,
                    adjudications_dir=None) -> dict:
    """
    `p_value_p_i1` plus, optionally, an entry in the falsification ledger.
    Opt-in behind a flag, for the reason given in `core.changepoint_colocation`.
    """
    res = p_value_p_i1(steps, relay_strength, induction_score,
                       alpha=alpha, seed=seed)
    res["adjudication"] = None
    if not (adjudicate and res.get("p_value") is not None):
        return res

    from core.adjudication import adjudicate_if_registered
    ef = res.get("endpoint_flags") or {}
    res["adjudication"] = adjudicate_if_registered(
        "P-I1", res["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=(
            f"changepoint co-location on the log-step axis; location = centroid "
            f"of the change-mass profile; null = permutation of the "
            f"{P_I1_UNIT}-pairing between the two series; one-sided "
            f"'{ALTERNATIVE}'"),
        notes=(
            f"verdict={res['verdict']} "
            f"p_reciprocal={res['p_reciprocal']:.4f} (RE-ANCHORS input only, NOT "
            f"calibrated into E) "
            f"endpoints: above-null at step 0 in "
            f"{ef.get('n_heads_above_null_at_first_step')} heads, absent at the "
            f"last step in {ef.get('n_heads_absent_at_last_step')} "
            f"({ef.get('n_heads_absent_at_last_step_with_peak_behaviour')} of "
            f"those with peak behaviour) -- reported, not scored; "
            f"shares its estimator with CLAIM-B under H-EMERGE: an estimator "
            f"defect moves both, so their e-values are not two independent "
            f"factors"),
        adjudications_dir=adjudications_dir,
    )
    return res
