"""
p1_mstate_tracking/replication_gate.py — CLAIM-C's null construction.

CLAIM-C is the transfer claim under H-TRANSFER and the only prediction in the
registry with a hard stop attached: `PREDICTIONS.md` says "If this fails, no
checkpoint-sweep work (items 9-11) proceeds past the gate." Its instrument is
the replication gate (`UPDATE_PLAN.md` execution-order item 6).

WHAT THE REGISTERED WORDING LEFT OPEN

The statement is prose: "pythia-1.4b-random reproduces gpt2-large-random
phenomenology, and checkpoint 143,000 reproduces trained gpt2-large." Which
statistics count as phenomenology, and what tolerance counts as reproduction,
are not stated. A criterion chosen after seeing gate data would void exactly
the guarantee the gate exists to provide, so both were fixed in advance and
are recorded in `claims/registry.json` under CLAIM-C's `null_construction`
before any real adjudication. Every choice below is a MODULE CONSTANT rather
than a parameter, so it cannot be re-made per run.

THE CRITERION: SIGN-CONCORDANCE OF THE TRAINED-MINUS-RANDOM CONTRAST

Blog 1's phenomenology is not a set of absolute levels; it is a *contrast* —
what trained weights do that random weights do not. So "reproduces" is read as
"the contrast points the same way on the new architecture":

  For each (metric, prompt) cell, take the per-layer profile of each arm,
  resample it onto a common normalized-depth grid (gpt2-large has 36 layers,
  pythia-1.4b has 24 — depth has to be normalized before anything can be
  compared at all), reduce to the scalar

      delta = mean over normalized depth of (trained - random)

  and score the cell concordant when sign(delta_pythia) == sign(delta_gpt2).

  Statistic = the number of concordant cells. One-sided, `greater`.

**What this criterion is blind to, stated rather than discovered later.** It is
ordinal, so it needs no tolerance and no per-metric standardisation — but it is
also scale-blind. A pythia contrast at one tenth of gpt2-large's magnitude
scores concordant in every cell. "Reproduces" here means "the same things move
the same way", not "by the same amount". The absolute per-arm profile distances
are computed and reported as a diagnostic (`arm_distances`) precisely so a
reader can see the levels, and they deliberately do NOT enter the p-value: a
second statistic entering the same test is a second chance for the prediction.

THE SECOND AGREEMENT AXIS: METRIC LEAVE-ONE-OUT, UNANIMOUS BOTH WAYS

Added 2026-08-24, still before any gate data exists. The cross-architecture
comparison above is the CLAIM. It is now run once on the full metric set and
once per metric-leave-one-out subset, and the gate requires them to agree.

**Why the two axes are separate factors rather than one p-value.** The
gpt/pythia axis is what CLAIM-C asserts; the metric axis is a statement about
whether the instrument is trustworthy. Folded into one number, a failure is
ambiguous between "the phenomenology does not transfer" and "one of our six
measurements is quirky" -- and those have opposite consequences for the sweep.
Kept as factors over a verdict lattice, each keeps its own meaning.

**Intersection-union, so no multiplicity correction.** The alternative is the
CONJUNCTION over subsets ("every subset agrees"), and for a conjunction the max
of the sub-p-values is itself a valid p-value. That holds *regardless of how
dependent the sub-tests are*, which is what makes it the right tool here: six
leave-one-out runs share five sixths of their data, and any Bonferroni-style
correction over them would be absurdly conservative while an IUT is exact.
Every subset shares one prompt set and therefore one null size -- eligibility
is decided once from the full six-metric requirement and leave-one-out drops
COLUMNS only -- so the max compares like with like.

**Unanimity in BOTH directions, and what that does and does not mean.**
TRANSFERS needs every subset to clear; FAILS-TO-TRANSFER needs every subset to
show the inversion; anything mixed is INSUFFICIENT. Both directions get harder
and the INSUFFICIENT middle grows. The hard stop already fires on INSUFFICIENT,
so the gate is not weakened -- only the word *falsified* is reserved for an
inversion no single metric is carrying.

The rule is "no subset may fail", **not** "no metric may dissent". Five of six
metrics inverting on every prompt survives every leave-one-out, because no one
metric is carrying it, and it is correctly recorded as a falsification. Reading
the rule the other way would let one quirky measurement veto a real result and
make the gate unfalsifiable in practice, which is the failure this whole
apparatus exists to prevent. What the axis catches is the other case: a verdict
that evaporates when one metric is dropped was a verdict about that metric.

The attainable floor is unchanged by the axis -- a max of p-values each at or
above 2/(2^n + 1) is at or above 2/(2^n + 1) -- so the refusal below still
works as written. What changes is power, and that cost is the point.

THE NULL, AND WHY THE EXCHANGEABLE UNIT IS THE PROMPT

The null is CLAIM-C's own falsifier — "the Blog-1 contrast is
architecture-dependent, not a general property of trained transformers" —
realised by permuting the trained/random condition labels on the *pythia* side
while gpt2-large is held fixed. gpt2-large is the reference phenomenology, not
a random draw, so the test conditions on it.

The label "trained" attaches to a RUN, not to a metric. Swapping it therefore
negates all six metric contrasts for that prompt at once, and the permutation
is a sign flip per prompt rather than per cell. This matters: six metrics on
one prompt are not six independent observations, and flipping them
independently would be the same error `p6_subspace/status-6.md` records for
"49 ALBERT layers are not 49 independent observations". Because delta is
antisymmetric in (trained, random), a flip negates it exactly, so the null
reduces to a closed form — row i contributes either its concordant count or
its discordant count — and is enumerated EXHAUSTIVELY when 2^n_prompts is
small enough, which it is for the eight metastability prompts (256 patterns).

**The limitation that remains, recorded rather than papered over.** Prompts run
on one model share that model's weights, so rows are not fully independent
either. A pythia-wide effect present in every prompt (a metric that moves the
same way with training for reasons unrelated to gpt2-large) correlates the rows
and the enumeration cannot see it. The prompt is the coarsest unit the gate's
own design makes available; a coarser one would need independent training runs,
which do not exist. The cost was measured, not assumed -- about 0.015 at alpha = 0.05
with independent rows and about 0.34 with identical ones (3000 draws each) -- so the module refuses at
the degenerate end, reports `sign_homogeneity` in between, and every record it
emits carries both numbers in its notes.

REFUSING RATHER THAN DEGRADING

Following `p2d_operator_activation.gradient_flow_condition.p_value_p_m1`, which
refuses when its three aggregates disagree in sign: this module emits no
p-value when

  - the four arms share fewer than two usable prompts, so there is no null;
  - the best attainable p exceeds alpha. With n prompts the smallest p the
    enumeration can express is 2/(2^n + 1), so at n = 4 a *perfect* result
    gives p = 0.118 and the test cannot reject at alpha = 0.05 however clean
    the data is. A test that cannot reject on a perfect result is not a test,
    and reporting its p-value as "not significant" would read as evidence
    against CLAIM-C when it is evidence of nothing;
  - no cell survives (every delta non-finite or exactly zero);
  - ANY leave-one-out subset cannot carry a p-value. The unanimity rule is a
    max, and a max over a set with an undefined member is undefined; reporting
    the rest would silently drop whichever subset was hardest to satisfy, which
    is precisely the one the rule exists to catch;
  - every usable prompt carries the SAME candidate sign pattern. The prompts
    then contribute one observation, and enumerating 2^n patterns over one
    observation is the wrong null rather than a conservative one. Measured
    rather than argued: with independent rows the rejection rate at
    alpha = 0.05 is about 0.015, and with identical rows it is about **0.34** -- the same
    fourfold-plus inflation POPPER reports when its relevance checker is
    removed (0.082 -> 0.340), reached here by a different route. This is a
    degeneracy, not a tolerance: the rows are either all equal or they are not,
    so nothing is being thresholded. `sign_homogeneity` reports the continuous
    version so a reader can see where between those two rates a real run sits.

A prompt is dropped whole when any of the six metrics is unavailable for it in
any of the four arms. The metric set is fixed; running on four of six because
two artifact fields were missing would be a per-run re-choice of the statistic.

THE THREE-WAY VERDICT, AND WHICH p ENTERS THE LEDGER

A difference-form criterion has three outcomes, not two, and the hard stop
must not conflate them:

  TRANSFERS          p_greater <= alpha
  FAILS-TO-TRANSFER  p_less    <= alpha — systematic DIScordance, i.e. the
                     contrast inverts on the new architecture. This is
                     CLAIM-C's falsifier positively demonstrated.
  INSUFFICIENT       neither. Transfer was not shown; nothing was shown.

The hard stop fires on both FAILS-TO-TRANSFER and INSUFFICIENT — the sweep does
not proceed on an unadjudicated gate — but only FAILS-TO-TRANSFER is recorded
as a falsification. An e-process records "insufficient evidence", never "null
accepted", and `claims/CLAIMS.md` already says that is the right shape here.

**Only `p_greater` is adjudicated.** CLAIM-C's registered H1 is "both
reproductions hold", so that is the prediction's p-value and the only number
calibrated into an e-value. `p_less` exists solely to separate
FAILS-TO-TRANSFER from INSUFFICIENT in the stop rule; it is recorded in the
record's notes and never enters the claim's product. Two one-sided tests on one
statistic would otherwise double the claim's Type-I rate.

THE RANDOM ARM, AND THE TWO-BASELINE POLICY

`PREDICTIONS.md` attaches the two-baseline policy to this claim: the true step-0
init and the norm-matched `pythia-1.4b-random` are separate objects and
`design-5c.md` / `design-1.md` are emphatic that they cannot be collapsed into
one "random" condition — GPT-NeoX's init variance-scaling is not comparable to
GPT-2's, and step 0 is the only checkpoint sitting in the attractive regime
eq. (3.6) describes.

The p-value runs on `pythia-1.4b-random`, which is what CLAIM-C's statement
names. Step 0 is computed in the same call as a MANDATORY sensitivity arm and
reported beside the result, but it does not enter the p-value: step 0 is
CLAIM-A's object, and letting the same data settle both entries is
`claims/EVALUABILITY.md`'s third recurring pattern. The arm cannot be dropped
by omission — `claim_c_concordance` requires either the arm or a written reason
for its absence, the same refusal `centroids.load_centroids` makes rather than
silently falling back to the primary arm.

TWO FURTHER DECISIONS THE ORIGINAL WORDING DID NOT SETTLE

- **`effective_rank` is read from `effective_rank_normed`.** `status-1.md`
  defect D1: the raw-mode field mixes directional collapse with residual-stream
  norm growth, so a single massive-norm token drives it toward 2 with no
  directional collapse at all. Both fields are persisted (`p1_io.py:154-158`).
  Baking the known-defective one into the gate with the hard stop attached
  would be knowingly wrong.
- **Full normalized depth, no band restriction.** Blog 1 quotes layers 5-30 of
  gpt2-large, but a depth band is a choice with as many options as there are
  bands, and picking one after seeing profiles is the selection this apparatus
  exists to prevent. The whole profile is used.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# The fixed choices. Constants, not parameters -- a per-run choice of which
# statistics count as phenomenology is a per-run choice of what CLAIM-C says.
# ---------------------------------------------------------------------------

#: The six per-layer series that constitute "phenomenology" for CLAIM-C. These
#: are exactly the keys of `visualization/checkpoints.py::CHECKPOINT_METRICS`,
#: the project's existing registry of per-layer series that participate in
#: checkpoint comparison. Named here rather than imported because that module
#: imports matplotlib, and the pure test tier runs with matplotlib genuinely
#: unimportable -- the same reason `core/model_family.py` exists.
CLAIM_C_METRICS: Tuple[str, ...] = (
    "mass_near_1",
    "effective_rank",
    "cluster_membership",
    "cluster_count",
    "cka_prev",
    "fiedler_mean",
)

#: Which artifact field each metric is read from, so the record says where the
#: number came from instead of leaving it to be inferred. `effective_rank` maps
#: to `effective_rank_normed` per status-1.md D1 -- see the module docstring.
CLAIM_C_METRIC_FIELDS: Dict[str, Tuple[str, str]] = {
    "mass_near_1":        ("geometry.json", "ip_mass_near_1"),
    "effective_rank":     ("geometry.json", "effective_rank_normed"),
    "cluster_membership": ("clustering.json", "1 - hdbscan.noise_fraction"),
    "cluster_count":      ("clustering.json", "hdbscan.n_clusters"),
    "cka_prev":           ("geometry.json", "cka_prev"),
    "fiedler_mean":       ("sinkhorn.json", "fiedler_mean"),
}

#: Points on the normalized-depth grid both architectures are resampled onto.
#: gpt2-large has 36 layers, pythia-1.4b has 24; nothing can be compared before
#: depth is normalized. 32 sits between the two so neither arm is upsampled by
#: more than ~1.4x. Matches the convention `checkpoint_scalars._resampled_l2`
#: already uses for cross-model profile distance.
DEPTH_GRID_POINTS = 32

#: CLAIM-C predicts CONCORDANCE, so the statistic is expected to be LARGE.
#: Recorded as a constant so the tail cannot be picked after seeing the data.
CLAIM_C_ALTERNATIVE = "greater"

#: The reciprocal direction, which distinguishes "the contrast inverts on the
#: new architecture" (CLAIM-C's falsifier, positively shown) from "nothing was
#: shown". Never calibrated into an e-value -- see the module docstring.
CLAIM_C_RECIPROCAL_ALTERNATIVE = "less"

#: The exchangeable unit of the permutation. The label "trained" attaches to a
#: run, so a swap moves all six metrics of one prompt together.
CLAIM_C_EXCHANGEABLE_UNIT = "prompt"

#: The second agreement axis, added 2026-08-24 and fixed before any gate data
#: exists. The cross-architecture test is re-run once per metric-leave-one-out
#: subset, so "the tools agree" means the verdict does not depend on any single
#: metric. See the module docstring: the claim axis and the instrument axis are
#: kept as separate factors rather than folded into one p-value.
CLAIM_C_TOOL_AXIS = "metric-leave-one-out"

#: Unanimity, in BOTH directions. Confirmation and falsification each require
#: every subset to agree; anything mixed is INSUFFICIENT.
CLAIM_C_TOOL_RULE = "unanimity"

#: Enumerate the sign-flip null exhaustively while 2^n_prompts is at most this;
#: fall back to sampling above it. 2^16 = 65536 patterns is well inside the
#: pure tier's budget and the eight metastability prompts need only 256.
EXHAUSTIVE_ENUMERATION_LIMIT = 1 << 16

#: Sampled draws when the prompt set is too large to enumerate.
DEFAULT_N_PERM = 5000

_MIN_VALID_DEPTH_POINTS = 2


# ---------------------------------------------------------------------------
# Depth normalization and the cell statistic
# ---------------------------------------------------------------------------

def resample_depth(series: Sequence[float],
                   n: int = DEPTH_GRID_POINTS) -> Optional[np.ndarray]:
    """
    A per-layer series on a normalized-depth grid of `n` points, NaN-aware.

    Returns None when fewer than two finite points survive, which is the
    honest answer for a profile that cannot define a depth trend -- not a
    zero-filled array that would silently contribute a delta of 0.

    Same construction as `checkpoint_scalars._resampled_l2`'s inner
    `_resample`, re-implemented here rather than imported because that module
    imports matplotlib.
    """
    # None is mapped to NaN rather than raising: `profiles_from_run_dir` already
    # does that for absent artifact fields, and a caller passing raw JSON should
    # get the same "unavailable" answer rather than a TypeError.
    v = np.array([np.nan if x is None else x for x in series], dtype=np.float64)
    if v.size < _MIN_VALID_DEPTH_POINTS:
        return None
    m = np.isfinite(v)
    if m.sum() < _MIN_VALID_DEPTH_POINTS:
        return None
    x = np.linspace(0.0, 1.0, v.size)
    return np.interp(np.linspace(0.0, 1.0, n), x[m], v[m])


def contrast(trained: Sequence[float], random_: Sequence[float],
             n: int = DEPTH_GRID_POINTS) -> float:
    """
    delta = mean over normalized depth of (trained - random), for one
    (metric, prompt) cell. NaN when either profile cannot be resampled.

    Antisymmetric in its arguments by construction, which is what makes the
    condition-label permutation an exact sign flip.
    """
    a, b = resample_depth(trained, n), resample_depth(random_, n)
    if a is None or b is None:
        return float("nan")
    return float(np.mean(a - b))


def profile_distance(a: Sequence[float], b: Sequence[float],
                     n: int = DEPTH_GRID_POINTS) -> float:
    """
    RMS distance between two profiles on the normalized-depth grid. Reported
    as a DIAGNOSTIC only -- the criterion is ordinal and this number does not
    enter any p-value. See the module docstring on scale-blindness.
    """
    ra, rb = resample_depth(a, n), resample_depth(b, n)
    if ra is None or rb is None:
        return float("nan")
    return float(np.sqrt(np.mean((ra - rb) ** 2)))


def _prompt_ok(arms: Sequence[dict], prompt: str) -> Optional[str]:
    """None when every arm has every metric for this prompt; else the reason."""
    for arm in arms:
        block = arm.get(prompt)
        if block is None:
            return "missing from at least one arm"
        for metric in CLAIM_C_METRICS:
            series = block.get(metric)
            # An all-NaN series is "unavailable", not "available and empty".
            # Checking length alone would keep the prompt and silently drop
            # only that metric's cells -- which is the per-run re-choice of the
            # statistic set this rule exists to prevent.
            if series is None or resample_depth(series) is None:
                return f"metric {metric!r} unavailable in at least one arm"
    return None


def _contrast_table(trained: dict, random_: dict, prompts: Sequence[str]) -> np.ndarray:
    """(n_prompts, n_metrics) of delta, NaN where a cell is unusable."""
    out = np.full((len(prompts), len(CLAIM_C_METRICS)), np.nan)
    for i, p in enumerate(prompts):
        for j, metric in enumerate(CLAIM_C_METRICS):
            out[i, j] = contrast(trained[p][metric], random_[p][metric])
    return out


# ---------------------------------------------------------------------------
# The concordance statistic
# ---------------------------------------------------------------------------

def claim_c_concordance(
    reference_trained: dict,
    reference_random: dict,
    candidate_trained: dict,
    candidate_random: dict,
    *,
    candidate_step0: Optional[dict],
    step0_absent_reason: Optional[str] = None,
) -> dict:
    """
    The observed sign-concordance table, before any p-value.

    Each arm is `{prompt: {metric: [per-layer values]}}`. `reference_*` is
    gpt2-large (the phenomenology being reproduced), `candidate_*` is
    pythia-1.4b. `candidate_step0` is the two-baseline policy's second object.

    `candidate_step0` is a REQUIRED keyword and may only be None when
    `step0_absent_reason` says why in writing. `PREDICTIONS.md` attaches the
    two-baseline policy to this claim specifically, and an arm that can be
    dropped by forgetting to pass it is not a policy -- the same refusal
    `centroids.load_centroids` makes rather than falling back silently.

    Returns the per-row concordant/discordant counts the null is built from,
    plus the step-0 sensitivity arm and the (non-adjudicated) arm distances.
    """
    if candidate_step0 is None and not str(step0_absent_reason or "").strip():
        raise ValueError(
            "candidate_step0 is required by the two-baseline policy "
            "(PREDICTIONS.md, design-5c.md, design-1.md): the true step-0 init "
            "and the norm-matched random model are separate objects. Pass the "
            "arm, or state step0_absent_reason in writing. Refusing to run a "
            "one-baseline gate that reads as a two-baseline one."
        )

    arms = [reference_trained, reference_random, candidate_trained, candidate_random]
    shared = sorted(set(reference_trained) & set(reference_random)
                    & set(candidate_trained) & set(candidate_random))

    prompts, dropped = [], {}
    for p in sorted(set().union(*(set(a) for a in arms))):
        reason = _prompt_ok(arms, p) if p in shared else "missing from at least one arm"
        if reason is None:
            prompts.append(p)
        else:
            dropped[p] = reason

    out = {
        "prompts_used": prompts,
        "prompts_dropped": dropped,
        "metrics": list(CLAIM_C_METRICS),
        "n_prompts": len(prompts),
        "exchangeable_unit": CLAIM_C_EXCHANGEABLE_UNIT,
        "step0_absent_reason": (None if candidate_step0 is not None
                                else str(step0_absent_reason).strip()),
    }
    if not prompts:
        out.update({"observed": None, "n_cells": 0,
                    "reason": "no prompt has all six metrics in all four arms"})
        return out

    d_ref = _contrast_table(reference_trained, reference_random, prompts)
    d_can = _contrast_table(candidate_trained, candidate_random, prompts)

    # A cell needs a SIGN, so a non-finite or exactly-zero delta in either
    # architecture is dropped. This is a degeneracy, not a tolerance: the
    # criterion is ordinal and introducing a magnitude cut here would be
    # choosing the tolerance the criterion was picked to avoid needing.
    usable = (np.isfinite(d_ref) & np.isfinite(d_can)
              & (d_ref != 0.0) & (d_can != 0.0))
    concordant = usable & (np.sign(d_ref) == np.sign(d_can))

    per_row_valid = usable.sum(axis=1).astype(int)
    per_row_conc = concordant.sum(axis=1).astype(int)

    out.update({
        "contrast_reference": d_ref.tolist(),
        "contrast_candidate": d_can.tolist(),
        "concordant": concordant.tolist(),
        "usable": usable.tolist(),
        "per_row_valid": per_row_valid.tolist(),
        "per_row_concordant": per_row_conc.tolist(),
        "n_cells": int(per_row_valid.sum()),
        "n_cells_dropped": int(usable.size - usable.sum()),
        "observed": int(per_row_conc.sum()),
        "concordance_fraction": (float(per_row_conc.sum() / per_row_valid.sum())
                                 if per_row_valid.sum() else None),
        "per_metric_concordant": {
            m: int(concordant[:, j].sum()) for j, m in enumerate(CLAIM_C_METRICS)},
        "per_metric_valid": {
            m: int(usable[:, j].sum()) for j, m in enumerate(CLAIM_C_METRICS)},
        "arm_distances": _arm_distances(reference_trained, reference_random,
                                        candidate_trained, candidate_random, prompts),
    })
    out.update(_row_independence(d_can, usable))
    out["step0_sensitivity"] = _step0_sensitivity(
        d_ref, candidate_trained, candidate_step0, prompts,
        out["concordance_fraction"])
    return out


def _row_independence(candidate_contrast: np.ndarray, usable: np.ndarray) -> dict:
    """
    How much the candidate's own prompts agree with each other, which is how
    close the design sits to the case where its permutation null is wrong.

    The sign-flip null enumerates 2^n_prompts patterns on the premise that the
    prompts carry n_prompts pieces of information. They do not carry n
    INDEPENDENT pieces -- they share a model's weights -- and the cost of that
    was measured rather than reasoned about: with independent rows the
    rejection rate at alpha = 0.05 is about 0.015 (conservative, as a discrete
    statistic should be); with every row identical it is about **0.34**. That is the
    same fourfold-plus inflation POPPER reports when its relevance checker is
    removed (0.082 -> 0.340), arriving here by a different route.

    `sign_rows_identical` is the exactly-degenerate end of that range: every
    usable prompt carrying the same sign pattern means the prompts contribute
    ONE observation, and enumerating 2^n patterns over one observation is not
    a conservative approximation, it is the wrong null. `p_value_claim_c`
    refuses there. It is a degeneracy, not a tolerance -- nothing is being
    thresholded, so the criterion stays free of the magnitude cut it was picked
    to avoid needing.

    `sign_homogeneity` is the continuous version, reported so a reader can see
    where between the two measured rates the real design sits: 1/2 when the
    prompts disagree as much as coin flips, 1.0 at the degenerate end.
    """
    signs = np.sign(np.nan_to_num(candidate_contrast, nan=0.0))

    # Rows are compared only on the metrics usable in EVERY row. A row that
    # differs from the others solely because one of its cells lost its sign is
    # not carrying independent information, and comparing full rows would let a
    # single dropped cell disable the refusal below -- masking with `signs *
    # usable` would do it too, since a NaN delta times False is still NaN and a
    # row carrying NaN compares unequal to an identical one.
    common = usable.all(axis=0)
    identical = False
    if common.any() and signs.shape[0] > 1:
        rows = {tuple(r) for r in signs[:, common].tolist()}
        identical = len(rows) == 1

    fracs = []
    for j in range(signs.shape[1]):
        col = signs[usable[:, j].astype(bool), j]
        if col.size:
            fracs.append(max((col > 0).mean(), (col < 0).mean()))
    return {
        "sign_rows_identical": bool(identical),
        "sign_homogeneity": float(np.mean(fracs)) if fracs else None,
        "n_metrics_usable_in_every_prompt": int(common.sum()),
    }


def _arm_distances(ref_t: dict, ref_r: dict, can_t: dict, can_r: dict,
                   prompts: Sequence[str]) -> dict:
    """
    Mean profile distance between the matching arms, per metric. DIAGNOSTIC:
    the chosen criterion is scale-blind by construction, and this is what a
    reader looks at to see the levels the sign test threw away. It enters no
    p-value -- a second statistic in the same test is a second chance for the
    prediction.
    """
    out = {}
    for metric in CLAIM_C_METRICS:
        for label, (a, b) in (("trained", (ref_t, can_t)), ("random", (ref_r, can_r))):
            vals = [profile_distance(a[p][metric], b[p][metric]) for p in prompts]
            vals = [v for v in vals if np.isfinite(v)]
            out[f"{metric}/{label}"] = float(np.mean(vals)) if vals else None
    return out


def _step0_sensitivity(d_ref: np.ndarray, can_t: dict,
                       can_step0: Optional[dict], prompts: Sequence[str],
                       primary_fraction: Optional[float]) -> dict:
    """
    The same concordance computed against the true step-0 init instead of the
    norm-matched random model. REPORTED, NEVER ADJUDICATED: step 0 is CLAIM-A's
    object, and letting one dataset settle two registry entries is
    EVALUABILITY.md's third recurring pattern.

    `disagrees_with_primary` is the flag worth reading. The two-baseline policy
    exists because these are different objects; if they give different answers
    about transfer, that is a finding about which random baseline the claim was
    ever about, and it belongs in the record rather than in a footnote.
    """
    if can_step0 is None:
        return {"available": False}
    missing = [p for p in prompts
               if _prompt_ok([can_step0], p) is not None]
    if missing:
        return {"available": False,
                "reason": f"step-0 arm lacks all six metrics for {len(missing)} prompt(s)"}

    d_s0 = _contrast_table(can_t, can_step0, prompts)
    usable = (np.isfinite(d_ref) & np.isfinite(d_s0)
              & (d_ref != 0.0) & (d_s0 != 0.0))
    conc = usable & (np.sign(d_ref) == np.sign(d_s0))
    n_valid = int(usable.sum())
    frac = float(conc.sum() / n_valid) if n_valid else None
    disagrees = (primary_fraction is not None and frac is not None
                 and (frac > 0.5) != (primary_fraction > 0.5))
    return {
        "available": True,
        "n_cells": n_valid,
        "n_concordant": int(conc.sum()),
        "concordance_fraction": frac,
        "primary_concordance_fraction": primary_fraction,
        "disagrees_with_primary": bool(disagrees),
        "note": ("step-0 and the norm-matched baseline point OPPOSITE ways about "
                 "transfer; the two-baseline policy exists because they are "
                 "different objects and here they behave like it"
                 if disagrees else
                 "step-0 agrees in direction with the norm-matched baseline"),
    }


# ---------------------------------------------------------------------------
# The sign-flip null
# ---------------------------------------------------------------------------

def _null_size(n_prompts: int, n_perm: int) -> Tuple[int, bool]:
    """
    (n_patterns, exhaustive) for a sign-flip null over `n_prompts` prompts.

    Depends only on the prompt count, which is why the attainable-floor refusal
    can be decided before any subset is scored: every leave-one-out subset uses
    the SAME prompt set (see `p_value_claim_c`), so they all share this null
    size and their p-values are directly comparable.
    """
    if 2 ** n_prompts <= EXHAUSTIVE_ENUMERATION_LIMIT:
        return 2 ** n_prompts, True
    return int(n_perm), False


def _null_counts(per_row_valid: np.ndarray, per_row_concordant: np.ndarray,
                 n_perm: int, seed: int) -> Tuple[np.ndarray, bool, int]:
    """
    The null distribution of the concordance count under condition-label
    swaps on the candidate side.

    Flipping prompt i's label negates every delta in row i, so every usable
    cell in that row changes concordance state: the row contributes `conc_i`
    unflipped and `valid_i - conc_i` flipped. That closed form is what makes
    exhaustive enumeration cheap.

    Returns (null_counts, exhaustive, n_patterns).
    """
    n = len(per_row_valid)
    disc = per_row_valid - per_row_concordant
    if 2 ** n <= EXHAUSTIVE_ENUMERATION_LIMIT:
        rows = np.stack([per_row_concordant, disc])          # (2, n)
        pats = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.intp)
        null = rows[pats, np.arange(n)].sum(axis=1).astype(np.float64)
        return null, True, int(pats.shape[0])
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_perm, n))
    null = np.where(flips == 0, per_row_concordant, disc).sum(axis=1).astype(np.float64)
    return null, False, int(n_perm)


def _subset_result(concordant: np.ndarray, usable: np.ndarray,
                   sign_can: np.ndarray, cols: Sequence[int],
                   n_perm: int, seed: int) -> dict:
    """
    Score one column subset of the concordance table.

    The prompt set, and therefore the null's pattern count, is identical across
    subsets -- only the columns change. That is what makes taking a max over
    their p-values meaningful rather than a comparison of differently-shaped
    tests.

    Returns `p_value: None` with a `reason` on the same degeneracies the full
    table refuses on, so a subset cannot quietly contribute a number the design
    does not support.
    """
    from core.nulls import p_from_null

    idx = list(cols)
    u = usable[:, idx]
    c = concordant[:, idx]
    valid = u.sum(axis=1).astype(np.intp)
    conc = c.sum(axis=1).astype(np.intp)
    out = {"n_cells": int(valid.sum()), "observed": int(conc.sum())}

    if out["n_cells"] == 0:
        out.update({"p_value": None,
                    "reason": "no cell in this subset has a defined sign"})
        return out

    # The degeneracy check is per subset: dropping a metric can leave the
    # remaining sign rows identical even when the full table's are not, and a
    # subset in that state carries one observation rather than n_prompts.
    common = u.all(axis=0)
    if common.any() and sign_can.shape[0] > 1:
        if len({tuple(r) for r in sign_can[:, idx][:, common].tolist()}) == 1:
            out.update({"p_value": None,
                        "reason": ("every prompt carries the same candidate sign "
                                   "pattern within this subset")})
            return out

    null, exhaustive, n_patterns = _null_counts(valid, conc, n_perm=n_perm, seed=seed)
    observed = float(out["observed"])
    out["p_value"] = float(
        p_from_null(observed, null, alternative=CLAIM_C_ALTERNATIVE)["p_value"])
    out["p_reciprocal"] = float(
        p_from_null(observed, null,
                    alternative=CLAIM_C_RECIPROCAL_ALTERNATIVE)["p_value"])
    out["n_null_patterns"] = int(n_patterns)
    out["null_exhaustive"] = bool(exhaustive)
    return out


def _metric_subsets() -> List[Tuple[str, Tuple[int, ...]]]:
    """
    The full metric set, then one subset per metric with that metric dropped.

    The full set is included in the unanimity requirement rather than treated
    as separate: "the test clears AND is not carried by any one metric" is one
    conjunction, and a max over all seven is the p-value for it.
    """
    n = len(CLAIM_C_METRICS)
    subsets = [("all", tuple(range(n)))]
    for j, m in enumerate(CLAIM_C_METRICS):
        subsets.append((f"drop:{m}", tuple(k for k in range(n) if k != j)))
    return subsets


def _alpha() -> float:
    from core.adjudication import load_registry
    from core.evalues import DEFAULT_ALPHA
    try:
        return float(load_registry().get("alpha", DEFAULT_ALPHA))
    except Exception:
        return float(DEFAULT_ALPHA)


def gate_verdict(p_greater: Optional[float], p_less: Optional[float],
                 alpha: Optional[float] = None) -> dict:
    """
    The three-way stop rule. See the module docstring: the hard stop fires on
    both FAILS-TO-TRANSFER and INSUFFICIENT, but only the former is a
    falsification of CLAIM-C.
    """
    a = _alpha() if alpha is None else float(alpha)
    if p_greater is None:
        return {"verdict": "INSUFFICIENT", "hard_stop": True, "falsified": False,
                "alpha": a,
                "reading": "no p-value could be emitted; the gate is unadjudicated "
                           "and items 9-11 do not proceed"}
    if p_greater <= a:
        return {"verdict": "TRANSFERS", "hard_stop": False, "falsified": False,
                "alpha": a,
                "reading": "the trained-vs-random contrast agrees in sign across "
                           "architectures more often than label swaps allow"}
    if p_less is not None and p_less <= a:
        return {"verdict": "FAILS-TO-TRANSFER", "hard_stop": True, "falsified": True,
                "alpha": a,
                "reading": "the contrast systematically INVERTS on the new "
                           "architecture. CLAIM-C's falsifier, positively shown: "
                           "STOP AND RE-BASELINE"}
    return {"verdict": "INSUFFICIENT", "hard_stop": True, "falsified": False,
            "alpha": a,
            "reading": "transfer was not shown and non-transfer was not shown. "
                       "The sweep does not proceed, but nothing is falsified -- an "
                       "e-process records insufficient evidence, never a null accepted"}


def p_value_claim_c(
    reference_trained: dict,
    reference_random: dict,
    candidate_trained: dict,
    candidate_random: dict,
    *,
    candidate_step0: Optional[dict],
    step0_absent_reason: Optional[str] = None,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
) -> dict:
    """
    CLAIM-C's calibrated p-value.

    Refuses -- `p_value` None with a `reason` -- rather than returning a number
    the design cannot support. See the module docstring for the three refusal
    conditions; the sharpest is the attainable-floor one, since a test whose
    best possible p exceeds alpha reports "not significant" on a perfect result
    and that reads as evidence against CLAIM-C when it is evidence of nothing.

    Emitting the p-value into the ledger is deliberately NOT this function's
    job -- see `adjudicate_claim_c` -- so that computing the number and
    entering it in the falsification record stay separable.
    """
    out = claim_c_concordance(
        reference_trained, reference_random, candidate_trained, candidate_random,
        candidate_step0=candidate_step0, step0_absent_reason=step0_absent_reason)
    alpha = _alpha()
    out["alpha"] = alpha

    if out.get("observed") is None or out["n_prompts"] < 2:
        out["p_value"] = None
        out.setdefault("reason",
                       f"{out['n_prompts']} usable prompt(s); the permutation "
                       f"unit is the prompt, so there is no null to draw")
        out.update(gate_verdict(None, None, alpha))
        return out

    if out["n_cells"] == 0:
        out["p_value"] = None
        out["reason"] = ("no cell has a defined sign in both architectures; every "
                         "contrast is non-finite or exactly zero")
        out.update(gate_verdict(None, None, alpha))
        return out

    # The attainable floor depends only on the prompt count, which every subset
    # shares, so it is decided once before any subset is scored.
    n_patterns, exhaustive = _null_size(out["n_prompts"], n_perm)
    best_attainable = (2.0 if exhaustive else 1.0) / (n_patterns + 1.0)
    out.update({
        "n_null_patterns": n_patterns,
        "null_exhaustive": bool(exhaustive),
        "best_attainable_p": float(best_attainable),
    })
    if best_attainable > alpha:
        out["p_value"] = None
        out["reason"] = (
            f"the permutation null over {out['n_prompts']} prompts can express no "
            f"p smaller than {best_attainable:.3f}, above alpha={alpha}. A test "
            f"that cannot reject on a PERFECT result is not a test, and its "
            f"'not significant' would read as evidence against CLAIM-C. Needs "
            f"more prompts, not a different threshold.")
        out.update(gate_verdict(None, None, alpha))
        return out

    if out.get("sign_rows_identical"):
        out["p_value"] = None
        out["reason"] = (
            "every usable prompt carries the SAME candidate sign pattern, so the "
            "prompts contribute one observation and enumerating 2^n_prompts "
            "patterns over it is the wrong null, not a conservative one. Measured: "
            "the rejection rate at this degeneracy is about 0.34 against a nominal "
            "0.05. Needs prompts that are not all telling the same story, or a "
            "coarser exchangeable unit than this design provides.")
        out.update(gate_verdict(None, None, alpha))
        return out

    # ---- the tool axis: one run per metric-leave-one-out subset -------------
    concordant = np.asarray(out["concordant"], dtype=bool)
    usable = np.asarray(out["usable"], dtype=bool)
    sign_can = np.sign(np.nan_to_num(np.asarray(out["contrast_candidate"],
                                                dtype=np.float64), nan=0.0))

    subsets = {}
    for name, cols in _metric_subsets():
        subsets[name] = _subset_result(concordant, usable, sign_can, cols,
                                       n_perm=n_perm, seed=seed)
    out["subsets"] = subsets
    out["tool_axis"] = CLAIM_C_TOOL_AXIS
    out["tool_rule"] = CLAIM_C_TOOL_RULE

    refused = {k: v["reason"] for k, v in subsets.items() if v.get("p_value") is None}
    if refused:
        out["p_value"] = None
        out["reason"] = (
            f"{len(refused)} of {len(subsets)} metric subsets cannot carry a "
            f"p-value ({'; '.join(f'{k}: {v}' for k, v in sorted(refused.items()))}). "
            f"The unanimity rule takes a MAX over subsets, and a max over a set "
            f"with an undefined member is undefined -- reporting the rest would "
            f"silently drop whichever subset was hardest to satisfy.")
        out.update(gate_verdict(None, None, alpha))
        return out

    # Intersection-union: the alternative is the CONJUNCTION over subsets, so
    # max(p) is a valid p-value for it and needs NO multiplicity correction --
    # and that holds regardless of how dependent the subsets are, which matters
    # here because six leave-one-out runs share five sixths of their data.
    p_iut = max(v["p_value"] for v in subsets.values())
    p_recip_iut = max(v["p_reciprocal"] for v in subsets.values())
    worst = max(subsets, key=lambda k: subsets[k]["p_value"])
    worst_recip = max(subsets, key=lambda k: subsets[k]["p_reciprocal"])

    out.update({
        "p_value": float(p_iut),
        "p_reciprocal": float(p_recip_iut),
        "alternative": CLAIM_C_ALTERNATIVE,
        "p_full_set": float(subsets["all"]["p_value"]),
        "p_reciprocal_full_set": float(subsets["all"]["p_reciprocal"]),
        "binding_subset": worst,
        "binding_subset_reciprocal": worst_recip,
        "n_subsets": len(subsets),
        "statistic": (
            f"count of (metric, prompt) cells where the trained-minus-random "
            f"contrast agrees in SIGN between architectures, over "
            f"{len(CLAIM_C_METRICS)} metrics x {out['n_prompts']} prompts; null "
            f"permutes the trained/random condition label per prompt on the "
            f"candidate side, reference held fixed. Reported p is the "
            f"intersection-union MAX over the full set and the six "
            f"metric-leave-one-out subsets, so it is the p of the subset that "
            f"agrees least"),
    })
    out.update(gate_verdict(out["p_value"], out["p_reciprocal"], alpha))
    return out


# ---------------------------------------------------------------------------
# Ledger emission -- opt-in
# ---------------------------------------------------------------------------

def adjudicate_claim_c(
    reference_trained: dict,
    reference_random: dict,
    candidate_trained: dict,
    candidate_random: dict,
    *,
    candidate_step0: Optional[dict],
    step0_absent_reason: Optional[str] = None,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    artifact_hashes: Iterable[str] = (),
    run_manifest: Optional[dict] = None,
    adjudicate: bool = False,
    adjudications_dir=None,
) -> dict:
    """
    `p_value_claim_c` plus, optionally, an entry in the falsification ledger.

    Adjudication is opt-in for the reason it is everywhere else in this
    project: these functions are exercised by tests, and `core.adjudication`
    refuses to overwrite an existing record, so one accidental test run would
    permanently occupy CLAIM-C's slot with a synthetic p-value -- on the one
    prediction that carries a hard stop.

    Only `p_greater` is adjudicated. `p_reciprocal` lands in the record's notes
    beside it, because the gate's stop decision depends on it and a reader of
    the ledger should not have to re-derive which of the three verdicts fired.
    """
    res = p_value_claim_c(
        reference_trained, reference_random, candidate_trained, candidate_random,
        candidate_step0=candidate_step0, step0_absent_reason=step0_absent_reason,
        n_perm=n_perm, seed=seed)
    res["adjudication"] = None
    if not (adjudicate and res.get("p_value") is not None):
        return res

    from core.adjudication import adjudicate_if_registered
    s0 = res.get("step0_sensitivity", {})
    res["adjudication"] = adjudicate_if_registered(
        "CLAIM-C", res["p_value"],
        artifact_hashes=tuple(artifact_hashes), run_manifest=run_manifest,
        test_name=(
            f"sign-flip permutation over prompts (condition label swapped on the "
            f"candidate side, reference held fixed); statistic = count of "
            f"(metric, prompt) cells whose trained-minus-random contrast agrees "
            f"in sign across architectures; one-sided "
            f"'{CLAIM_C_ALTERNATIVE}', reported as the intersection-union MAX "
            f"over the full metric set and the {len(CLAIM_C_METRICS)} "
            f"metric-leave-one-out subsets (unanimity, no multiplicity "
            f"correction needed for a conjunction); "
            f"{res['n_null_patterns']} patterns"
            f"{' (exhaustive)' if res['null_exhaustive'] else ' (sampled)'}"),
        notes=(
            f"verdict={res['verdict']} hard_stop={res['hard_stop']} "
            f"p_reciprocal={res['p_reciprocal']:.4f} (stop-rule input only, NOT "
            f"calibrated into E) "
            f"cells={int(res['observed'])}/{res['n_cells']} "
            f"tool_axis={res['tool_axis']}/{res['tool_rule']} "
            f"p_full_set={res['p_full_set']:.4f} "
            f"binding_subset={res['binding_subset']} "
            f"prompts={res['n_prompts']} dropped={len(res['prompts_dropped'])} "
            f"best_attainable_p={res['best_attainable_p']:.4f} "
            f"step0_arm={'reported' if s0.get('available') else 'absent'}"
            + (f" step0_DISAGREES_WITH_PRIMARY" if s0.get("disagrees_with_primary")
               else "")
            + f" sign_homogeneity={res.get('sign_homogeneity')}"
            + f" | prompts on one model are not independent runs: the sign-flip "
              f"unit is the prompt, which is the coarsest unit this design "
              f"provides, and a pythia-wide effect common to every prompt would "
              f"not be visible to the enumeration. Measured rejection rate at "
              f"alpha=0.05: ~0.015 with independent rows, ~0.34 with identical "
              f"ones; sign_homogeneity says where this run sits between them"),
        adjudications_dir=adjudications_dir,
    )
    return res


# ---------------------------------------------------------------------------
# Loading the six profiles from a Phase 1 run directory
# ---------------------------------------------------------------------------

def profiles_from_run_dir(run_dir) -> Dict[str, List[float]]:
    """
    `{metric: [per-layer values]}` for the six CLAIM_C_METRICS, read straight
    from a Phase 1 run directory's JSON artifacts.

    Deliberately does not reuse `visualization/series.py`'s extractors, which
    compute the same six series: that module imports `core.naming`, which
    imports `core.style`, which imports matplotlib -- and the pure tier runs
    with matplotlib genuinely unimportable. The same reason the checkpoint name
    grammar was moved out of `visualization/checkpoints.py` into
    `core/model_family.py`. If the two ever disagree, `series.py` is the older
    copy but THIS one is the one the gate reads, and `effective_rank` differs
    on purpose (status-1.md D1).

    A missing file or field yields a MISSING KEY rather than a zero-filled or
    all-NaN series, so `claim_c_concordance` drops the whole prompt instead of
    quietly running on five of six metrics. A series with some NaN layers is
    kept -- interpolation across a gap is a different thing from having no
    measurement at all.
    """
    import json

    run_dir = Path(run_dir)

    def _load(name: str) -> dict:
        p = run_dir / name
        if not p.exists():
            return {}
        try:
            with open(p) as f:
                return json.load(f)
        except (ValueError, OSError):
            return {}

    geo = _load("geometry.json").get("layers", []) or []
    clu = _load("clustering.json").get("layers", []) or []
    snk = _load("sinkhorn.json").get("layers", []) or []

    def _plain(layers, key):
        if not layers:
            return None
        out = [float(lr[key]) if lr.get(key) is not None else float("nan")
               for lr in layers]
        return out if any(v == v for v in out) else None

    def _hdbscan(layers, key, transform=lambda v: v):
        if not layers:
            return None
        out = []
        for lr in layers:
            v = lr.get("clustering", {}).get("hdbscan", {}).get(key)
            out.append(float(transform(v)) if v is not None else float("nan"))
        return out if any(v == v for v in out) else None

    candidates = {
        "mass_near_1":        _plain(geo, "ip_mass_near_1"),
        "effective_rank":     _plain(geo, "effective_rank_normed"),
        "cka_prev":           _plain(geo, "cka_prev"),
        "cluster_membership": _hdbscan(clu, "noise_fraction", lambda v: 1.0 - v),
        "cluster_count":      _hdbscan(clu, "n_clusters"),
        "fiedler_mean":       _plain(snk, "fiedler_mean"),
    }
    return {k: v for k, v in candidates.items() if v is not None}
