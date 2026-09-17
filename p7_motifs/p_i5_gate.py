"""
p7_motifs/p_i5_gate.py — P-I5's gate, PART ONE: the floor, the degeneracy,
and the measurement grid. The control is not built here — see "WHAT THIS
MODULE DOES NOT BUILD" at the end of this docstring.

    P-I5   Ablating an induction head changes the pairwise-distance
           distribution among the particles it couples (the matched
           positions), not only the logit at the copied token.
    H0     Induction-head ablation leaves inter-particle geometry at the
           matched positions indistinguishable from a matched-magnitude
           random-direction ablation, while still moving the copied-token
           logit.
    falsifier
           A large logit effect with a pairwise-distance change
           indistinguishable from the matched control: the head moves the
           readout without moving the particles, and the transport reading
           is wrong. Adjudicated on the JOINT outcome — a geometric effect
           with no logit effect falsifies it equally, in the other
           direction.

BUILT IN THE ORDER `claims/EVALUABILITY.md` PRESCRIBES

That document's rule for every row naming a matched control: "compute the
attainable floor, name what the statistic degenerates on, check what the
measurement grid contributes, and only then build the control." `P-AB1`
(`POPPER_PLAN.md` 6q) and `P-I3` (6s) were built that way and each session
before the control changed the design. `P-I5` is EVALUABILITY.md's own
flagged exception: "the first row here whose statistic is two-dimensional,
which no construction in this project has built" — so this pass does not
get to reuse either precedent's null verbatim, only their method.

1. THE ATTAINABLE FLOOR — a genuinely joint sign-flip null
------------------------------------------------------------
`P-AB1`'s justification for a sign-flip null carries over unchanged: under
H0, a real ablation direction and a structureless direction of equal
magnitude at the same site are exchangeable, so swapping the real/control
label at one unit is an exact symmetry of the null. What is new here is
that swapping the label at unit `i` flips the sign of BOTH
`delta_geometric[i]` and `delta_logit[i]` TOGETHER — the same underlying
(real, control) pair produced both readouts, so the null is a single
2**n-point sign-pattern space, not two independent 2**n-point spaces run
side by side.

2. WHAT THE STATISTIC DEGENERATES ON — THE FIRST STATISTIC TRIED DID NOT
   HOLD, AND THE SECOND ONE REPLACES IT (found by running it on inputs
   whose answer is known, `POPPER_PLAN.md` 6m's method applied here)
--------------------------------------------------------------------------------
**What was tried first, and what broke.** The obvious joint statistic is an
AND-corner: `count(s) = #{s : T_g(s) >= T_g_obs AND T_l(s) >= T_l_obs}`,
`p = count / 2**n`. It reduces correctly to the plain one-dimensional
sign-flip test when one axis is held at zero (verified:
`tests/test_p_i5_gate.py::TestNaiveAndCornerPvalue`
`test_reduces_to_one_dimensional_when_one_axis_is_zero`), so it looked
right. **Run on synthetic data with a TRUE joint H0 — both
`delta_geometric` and `delta_logit` drawn as pure independent noise, zero
effect on either axis — it rejects at 0.207 against a nominal 0.05, at
n = 8, 2026-09-16 (`claims/calibration/p_i5_joint_null.json`,
`calibrate_naive_and_corner`).** A ~4x inflation, not a rounding error, and
present across the n values checked (`calibrate_naive_and_corner` sweeps
n = 6..12). The AND-corner is not a valid test.

**Why.** A single-dimension sign-flip statistic is valid because the
observed pattern's RANK among the `2**n` values of `T(s)` is, by
exchangeability, uniformly distributed under H0 — that is the entire
content of a permutation test. The AND-corner treats two DIFFERENT
rankings (`T_g`'s and `T_l`'s, induced by independent `delta_geometric`
and `delta_logit`) as if intersecting their "at least this extreme" sets
preserved that uniformity. It does not: `|A cap B| / 2**n`, for two
independent approximately-half-sized random subsets `A`, `B` of the same
`2**n`-point space, is not itself uniformly distributed — its own sampling
distribution puts more mass near small values than a valid p-value's does,
which is exactly an inflated rejection rate at any fixed alpha. This is a
general fact about intersecting two independently-derived "extremeness"
sets, not specific to sign-flip nulls — verified here rather than assumed
because it is the kind of thing that reads as obviously fine until it is
run (P-ST1, P6-R2/R4 and P-I3's own gates each record one of these).

**The second construction: rank by the WEAKER axis, not the intersection.**
`joint_rank_pvalue` computes, for every sign pattern `s`, `rank_g(s)` and
`rank_l(s)` — each axis's own valid ascending rank among all reference
patterns (ties to the minimum rank, `scipy.stats.rankdata(method="min")`) —
and combines them as `combined(s) = min(rank_g(s), rank_l(s))`, a
Tippett-style minimum-rank statistic. That is a genuine scalar function of
`s`, so the single-dimension exchangeability argument applies to it: under
the COMPLETE null (no effect on either axis) the observed pattern's rank in
`combined` is uniform, and `calibrate_joint_rank` measures 0.049 against
nominal 0.05 at n = 8 (`claims/calibration/p_i5_joint_null.json`).

**Why that is still not P-I5's statistic (2026-09-17, found in review).**
The prediction is a CONJUNCTION — a geometric effect AND a logit effect, and
the falsifier says either one missing falsifies it. So the null P-I5 must
control is the UNION `H0_geometric OR H0_logit`: any configuration where at
least one axis carries no effect. The min-rank statistic is calibrated only
at the intersection of those two (both absent); at the falsifier's own
configuration — a real logit effect, pure noise on geometry —
`partial_pass_risk_demo` measures it rejecting at **0.17–0.19** against
nominal 0.05 (`p_i5_joint_null.json`, `partial_pass_risk`). An earlier
draft of this docstring read that as "close to what a correctly-calibrated
test should do when one of its two required conditions is absent". It is
not: that configuration is a point IN the null, and a rate of 0.19 there is
an uncontrolled Type-I rate, not a power trade. A Tippett minimum is the
right combination for "at least one effect"; it is the wrong one for
"both".

**The statistic: intersection-union.** `intersection_union_pvalue` computes
each axis's own exact one-dimensional sign-flip p-value over the SAME
`2**n` joint sign-pattern space (`one_dimensional_pvalue` on each axis) and
reports `p = max(p_geometric, p_logit)`. Rejecting only when BOTH axes
individually clear alpha is the intersection-union test, valid under the
union null at level alpha with no multiplicity correction, however
dependent the two axes are — the same device `CLAIM-C`'s gate uses across
its metric-leave-one-out subsets and `CLAIM-A`'s recommended null uses
across its three criteria. `calibrate_intersection_union` measures it on
three arms — the complete null and BOTH partial nulls — and it must sit at
or below nominal on all three, which is the calibration the min-rank
statistic never had. `joint_rank_pvalue` is kept because the committed
exploratory records (`p_i5_real_ablation.json`, `p_i5_validation.json`,
`p_i5_structured_control.json`) were scored with it; the scripts now report
both, with `intersection_union` as `gate` and the min-rank value beside it
as `gate_min_rank_superseded`, and the stored deltas re-score under the
new statistic without a model run.

**Caveat found while testing the min-rank statistic: an identically-zero
axis degenerates `joint_rank_pvalue`, it does not reduce it.** `_axis_rank`
gives every sign pattern the SAME (minimum) rank on a perfectly-tied axis,
so `min(rank_g, rank_l)` collapses to that shared value for every pattern
regardless of the other axis, and `p_value` comes back `1.0` no matter how
extreme the informative axis is. `one_dimensional_pvalue` is the real
single-axis statistic, and the intersection-union test is built from it
directly. On an identically-zero axis `one_dimensional_pvalue` returns 1.0
(every pattern ties the observed one), so the intersection-union p is 1.0
there too — correctly: an axis with no variation cannot show an effect.

**Attainable floor is the same for all three constructions and is
re-verified** (`tests/test_p_i5_gate.py::TestAttainableFloor`): the best
case — every unit's `delta_geometric` and `delta_logit` both nonzero and
same-signed — makes the observed pattern the UNIQUE global maximizer of
BOTH `T_g` and `T_l`, hence the unique top rank on both axes, hence the
unique top `combined` score, and each axis's one-dimensional p is
`1 / 2**n`, so their max is too:

    one-sided: p_min(n) = 1 / 2**n
    two-sided: p_min(n) = 2 / 2**n = 1 / 2**(n-1)

3. THE MEASUREMENT GRID — matched pairs per prompt, measured, not assumed
------------------------------------------------------------------------------
`count_matched_pairs_by_prompt` runs `core.battery_structure.
induction_candidates` (the same primitive `p7_motifs/run_7.py` and
`core/interactions.py` already use to build every `InteractionTable` in
this project) against `core.config.PROMPTS`, tokenized at a real cached
Pythia checkpoint. Measured 2026-09-16 against `EleutherAI/pythia-70m`
step143000:

    short_heterogeneous       20 tokens        1 pair
    hdbscan_code             242 tokens       426 pairs
    paper_excerpt             286 tokens       554 pairs
    camus_letranger           465 tokens       963 pairs
    wiki_paragraph             467 tokens      1598 pairs
    sullivan_ballou            482 tokens      2038 pairs
    latex_monograph            446 tokens      2873 pairs
    homer_iliad                562 tokens      2518 pairs
    repeated_tokens            265 tokens     34191 pairs   -- excluded, see below

`repeated_tokens` reproduces the 34,191-pair figure the resume block and
`p7_motifs/motif_alphabet.py::relay_strength`'s docstring already quote for
this exact prompt, which cross-validates this count against an independent
prior measurement rather than trusting a fresh one blind. It is excluded
from `n` the same way `P_I1_DOMINANT_PROMPT` already is everywhere else in
this project ("reported and never scored" — every token repeats, so its
pair count is a fact about the prompt, not about a checkpoint).

**This sets, not assumes, the exchangeable unit.** Per-pair counts in the
thousands would make `n` (in the floor formula above) enormous if the unit
were "one matched pair" — but pairs within one prompt are not independent
draws: they share the same forward pass, the same model state, and (by the
design choice below) the same one random-direction draw for the control
arm. That is exactly `P-AB1`'s finding restated (`POPPER_PLAN.md` 6q: the
per-ablation-point reading inflated to 0.235 under a shared per-prompt
factor where the per-prompt reading held at 0.029) — this module reuses
that conclusion rather than re-deriving it, because the mechanism is the
same one: many correlated readings inside one prompt masquerading as many
independent units.

**Design choice this step lands on, following that reasoning: the
exchangeable unit is the PROMPT, not the matched pair.** One
matched-magnitude random direction is drawn per (prompt, ablation site);
every matched pair within that prompt is scored against it and the
prompt's `(delta_geometric, delta_logit)` is their aggregate (mean).
Excluding `repeated_tokens`, that gives `n = 8` — floor `1/2**8 = 1/256`
one-sided, `1/128` two-sided, both comfortably below the usual 0.05
working threshold and in the same regime `P-AB1` needed six prompts to
reach.

**This is put to the author before the control is built, matching how
`P-AB1`'s prompt-as-unit choice and `P-I3`'s `score_and_layer` matching key
were both registered: after the measurement that motivated them, not
before.** Nothing here commits it to `claims/registry.json`.

WHAT THIS MODULE DOES NOT BUILD
---------------------------------
The matched-magnitude random-direction ablation itself: drawing a random
direction of the same norm as the real ablation's effect, running it
through `core/intervention.py` on cached pythia-70m/410m checkpoints, and
scoring the result with `core/dual_reading.py::pairwise_geometric_reading`
(the geometric half) and `core/intervention.py::next_token_kl` (the logit
half — corrected 2026-09-16: earlier drafts of this docstring, and of
`PROJECT.md` §3.29 before it, misattributed this function to
`core/functional_distance.py`, which holds a *different* pairwise-KL
primitive for clustering, not this one) to produce real
`(delta_geometric, delta_logit)` arrays. That is the next PR, not this
one — steps 1–3 above are validated on synthetic data and a real but
model-forward-pass-free measurement (tokenization only); nothing here has
touched a real activation.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# 1. The attainable floor — exact joint sign-flip permutation null
# ---------------------------------------------------------------------------

# 2**20 ~ 1e6 patterns, enumerable in well under a second — the same regime
# P-ST1/P-AB1's exact-enumeration constructions work in. Above this, the
# Monte Carlo path (mc_draws) is required.
MAX_EXACT_UNITS = 20


def enumerate_sign_patterns(n: int) -> np.ndarray:
    """All 2**n sign vectors in {-1,+1}^n, one per row. n <= MAX_EXACT_UNITS."""
    if n > MAX_EXACT_UNITS:
        raise ValueError(
            f"enumerate_sign_patterns: n={n} exceeds the exact cap of "
            f"{MAX_EXACT_UNITS} (2**{n} patterns) — use the Monte Carlo path "
            f"instead."
        )
    if n == 0:
        return np.zeros((1, 0), dtype=np.int8)
    return np.array(list(itertools.product([1, -1], repeat=n)), dtype=np.int8)


def _find_observed_row(patterns: np.ndarray) -> int:
    """Index of the all-+1 row within `patterns` — the observed label
    assignment by this module's convention (real minus control)."""
    rows = np.flatnonzero(np.all(patterns == 1.0, axis=1))
    if rows.size == 0:
        raise ValueError(
            "_find_observed_row: no all-+1 row in patterns — the reference "
            "set must include the observed configuration."
        )
    return int(rows[0])


def _axis_extreme_mask(t_null: np.ndarray, t_obs: float, alternative: str) -> np.ndarray:
    if alternative == "greater":
        return t_null >= t_obs
    if alternative == "two-sided":
        return np.abs(t_null) >= abs(t_obs)
    raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")


def _axis_rank(t: np.ndarray, alternative: str) -> np.ndarray:
    """0-indexed ascending rank, ties -> the MINIMUM rank (scipy's
    method='min') so a tie cannot inflate an observed pattern's apparent
    extremeness. 'two-sided' ranks by |t| instead of t."""
    values = t if alternative == "greater" else np.abs(t)
    return rankdata(values, method="min").astype(np.int64) - 1


def one_dimensional_pvalue(delta: np.ndarray, alternative: str = "greater") -> float:
    """
    A single axis's own exact sign-flip permutation p-value — the
    one-dimensional reading `partial_pass_risk_demo` compares
    `joint_rank_pvalue` against. NOT the same as calling
    `joint_rank_pvalue` with the other axis zeroed out: an identically-zero
    axis is fully tied under `_axis_rank` (every pattern gets rank 0 on
    it), which collapses `min(rank_g, rank_l)` to 0 for every pattern
    regardless of the informative axis — a real degeneracy of the min-rank
    statistic, not a bug, but one that makes the zero-out trick invalid for
    "what would a one-dimensional reading alone say" (found by the test
    that tried exactly that reduction — see tests/test_p_i5_gate.py).
    """
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    if n > MAX_EXACT_UNITS:
        raise ValueError(
            f"one_dimensional_pvalue: n={n} exceeds the exact cap of {MAX_EXACT_UNITS}."
        )
    patterns = enumerate_sign_patterns(n).astype(np.float64)
    t_null = patterns @ d
    obs_row = _find_observed_row(patterns)
    t_obs = float(t_null[obs_row])
    extreme = _axis_extreme_mask(t_null, t_obs, alternative)
    return float(extreme.mean())


def naive_and_corner_pvalue(
    delta_geometric: np.ndarray,
    delta_logit: np.ndarray,
    alternative: str = "greater",
    mc_draws: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    RETIRED — kept for the record, not for adjudication. See module
    docstring section 2: this AND-corner construction over-rejects under a
    true joint H0 (measured 0.207 vs nominal 0.05 at n=8). Use
    joint_rank_pvalue instead. This function exists so "the null that did
    not hold" is reproducible from code, matching how this project keeps
    every retired construction on record rather than deleting it
    (P6-R2/R4, the steering-sign null's three retired nulls, P-AB1's
    literal-reading floor).

    Same parameters/returns shape as joint_rank_pvalue.
    """
    dg = np.asarray(delta_geometric, dtype=np.float64)
    dl = np.asarray(delta_logit, dtype=np.float64)
    if dg.shape != dl.shape or dg.ndim != 1:
        raise ValueError(
            f"naive_and_corner_pvalue: delta_geometric {dg.shape} and "
            f"delta_logit {dl.shape} must be same-length 1-D arrays."
        )
    n = dg.shape[0]
    if n == 0:
        raise ValueError("naive_and_corner_pvalue: no units.")

    if mc_draws is None:
        if n > MAX_EXACT_UNITS:
            raise ValueError(
                f"naive_and_corner_pvalue: n={n} exceeds the exact cap of "
                f"{MAX_EXACT_UNITS}; pass mc_draws explicitly."
            )
        patterns = enumerate_sign_patterns(n).astype(np.float64)
        t_g_null = patterns @ dg
        t_l_null = patterns @ dl
        obs_row = _find_observed_row(patterns)
        t_g_obs = float(t_g_null[obs_row])
        t_l_obs = float(t_l_null[obs_row])
        extreme = _axis_extreme_mask(t_g_null, t_g_obs, alternative) & \
            _axis_extreme_mask(t_l_null, t_l_obs, alternative)
        n_extreme = int(extreme.sum())
        n_null = patterns.shape[0]
        p_value = n_extreme / n_null
        exact = True
    else:
        rng = rng if rng is not None else np.random.default_rng()
        signs = rng.choice(np.array([-1.0, 1.0]), size=(mc_draws, n))
        t_g_obs = float(dg.sum())
        t_l_obs = float(dl.sum())
        t_g_null = signs @ dg
        t_l_null = signs @ dl
        extreme = _axis_extreme_mask(t_g_null, t_g_obs, alternative) & \
            _axis_extreme_mask(t_l_null, t_l_obs, alternative)
        n_extreme = int(extreme.sum())
        n_null = mc_draws
        p_value = (n_extreme + 1) / (n_null + 1)
        exact = False

    return {
        "p_value": float(p_value), "n_units": int(n), "exact": exact,
        "t_geometric_obs": t_g_obs, "t_logit_obs": t_l_obs,
        "n_null_patterns": int(n_null), "n_extreme": n_extreme,
        "alternative": alternative,
    }


def joint_rank_pvalue(
    delta_geometric: np.ndarray,
    delta_logit: np.ndarray,
    alternative: str = "greater",
    mc_draws: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    P-I5's joint permutation p-value: a Tippett-style minimum-rank
    combined statistic over the sign-flip null. See module docstring
    section 2 for why this replaces naive_and_corner_pvalue and why it is
    valid where that one was not.

    Parameters
    ----------
    delta_geometric, delta_logit : (n,) arrays — one paired difference
        (real ablation arm minus matched-magnitude random-direction control
        arm) per EXCHANGEABLE UNIT. What a unit is (one matched pair? one
        prompt?) is section 3's question — this function scores whatever
        the caller already aggregated to.
    alternative : "greater" (H1's predicted direction — real > control on
        BOTH axes) or "two-sided".
    mc_draws : None -> exact enumeration (n <= MAX_EXACT_UNITS, raises above
        it unless mc_draws is given). An int -> that many Monte Carlo sign
        patterns, PLUS the observed pattern, form the reference set the
        ranks are computed over — the same rank-based statistic, evaluated
        on a finite random sample instead of the full 2**n population, the
        standard way large-n permutation p-values are approximated.

    Returns
    -------
    dict: p_value, n_units, exact, rank_geometric_obs, rank_logit_obs,
    combined_rank_obs, n_reference_patterns, n_extreme, alternative.
    """
    dg = np.asarray(delta_geometric, dtype=np.float64)
    dl = np.asarray(delta_logit, dtype=np.float64)
    if dg.shape != dl.shape or dg.ndim != 1:
        raise ValueError(
            f"joint_rank_pvalue: delta_geometric {dg.shape} and delta_logit "
            f"{dl.shape} must be same-length 1-D arrays."
        )
    n = dg.shape[0]
    if n == 0:
        raise ValueError("joint_rank_pvalue: no units.")
    if alternative not in ("greater", "two-sided"):
        raise ValueError(
            f"joint_rank_pvalue: alternative must be 'greater' or "
            f"'two-sided', got {alternative!r}."
        )

    if mc_draws is None:
        if n > MAX_EXACT_UNITS:
            raise ValueError(
                f"joint_rank_pvalue: n={n} exceeds the exact cap of "
                f"{MAX_EXACT_UNITS}; pass mc_draws explicitly."
            )
        patterns = enumerate_sign_patterns(n).astype(np.float64)
        exact = True
    else:
        rng = rng if rng is not None else np.random.default_rng()
        mc_signs = rng.choice(np.array([-1.0, 1.0]), size=(mc_draws, n))
        patterns = np.concatenate([mc_signs, np.ones((1, n))], axis=0)
        exact = False

    t_g = patterns @ dg
    t_l = patterns @ dl
    obs_row = _find_observed_row(patterns)

    rank_g = _axis_rank(t_g, alternative)
    rank_l = _axis_rank(t_l, alternative)
    combined = np.minimum(rank_g, rank_l)
    combined_obs = int(combined[obs_row])

    n_extreme = int((combined >= combined_obs).sum())
    n_reference = int(patterns.shape[0])
    p_value = n_extreme / n_reference

    return {
        "p_value": float(p_value),
        "n_units": int(n),
        "exact": exact,
        "rank_geometric_obs": int(rank_g[obs_row]),
        "rank_logit_obs": int(rank_l[obs_row]),
        "combined_rank_obs": combined_obs,
        "n_reference_patterns": n_reference,
        "n_extreme": n_extreme,
        "alternative": alternative,
    }


def intersection_union_pvalue(
    delta_geometric: np.ndarray,
    delta_logit: np.ndarray,
    alternative: str = "greater",
) -> dict:
    """
    P-I5's statistic: the intersection-union test over the two axes.

    Each axis gets its own exact one-dimensional sign-flip p-value over the
    joint 2**n sign-pattern space, and the reported p is their MAXIMUM. The
    test rejects only when both axes individually clear alpha, which is
    valid under the union null (either effect absent) at level alpha with
    no multiplicity correction, whatever the dependence between the axes
    — module docstring section 2. `joint_rank_pvalue` (min-rank) is NOT
    this test; it controls only the complete null.

    Returns
    -------
    dict: p_value, p_geometric, p_logit, binding_axis, n_units, exact,
    alternative.
    """
    dg = np.asarray(delta_geometric, dtype=np.float64)
    dl = np.asarray(delta_logit, dtype=np.float64)
    if dg.shape != dl.shape or dg.ndim != 1:
        raise ValueError(
            f"intersection_union_pvalue: delta_geometric {dg.shape} and "
            f"delta_logit {dl.shape} must be same-length 1-D arrays."
        )
    if dg.shape[0] == 0:
        raise ValueError("intersection_union_pvalue: no units.")
    if alternative not in ("greater", "two-sided"):
        raise ValueError(
            f"intersection_union_pvalue: alternative must be 'greater' or "
            f"'two-sided', got {alternative!r}."
        )
    p_g = one_dimensional_pvalue(dg, alternative)
    p_l = one_dimensional_pvalue(dl, alternative)
    return {
        "p_value": float(max(p_g, p_l)),
        "p_geometric": float(p_g),
        "p_logit": float(p_l),
        "binding_axis": "geometric" if p_g >= p_l else "logit",
        "n_units": int(dg.shape[0]),
        "exact": True,
        "alternative": alternative,
    }


def attainable_floor(n: int, alternative: str = "greater") -> float:
    """
    Closed-form best-case floor for n exchangeable units, shared by
    naive_and_corner_pvalue, joint_rank_pvalue and intersection_union_pvalue
    (proved in module
    docstring sections 1–2, enumeration-checked in tests/test_p_i5_gate.py
    for n = 1..15): every unit fully informative and same-signed on both
    axes makes the observed sign pattern the UNIQUE occupant of the top
    AND-corner / the unique top combined rank, either way the statistic is
    computed.
    """
    if n < 1:
        raise ValueError(f"attainable_floor: n must be >= 1, got {n}")
    if alternative == "greater":
        return 1.0 / (2 ** n)
    if alternative == "two-sided":
        return 2.0 / (2 ** n)
    raise ValueError(f"attainable_floor: alternative must be 'greater' or 'two-sided', got {alternative!r}")


# ---------------------------------------------------------------------------
# 2. Calibration — run on inputs whose answer is known
# ---------------------------------------------------------------------------

def _synthetic_null_draw(n_units: int, rng: np.random.Generator) -> tuple:
    """One draw of (delta_geometric, delta_logit) under a TRUE joint H0:
    both axes pure independent standard-normal noise, no effect on
    either."""
    return rng.normal(0.0, 1.0, size=n_units), rng.normal(0.0, 1.0, size=n_units)


def calibrate_naive_and_corner(
    n_units: int, n_trials: int = 3000, rng: Optional[np.random.Generator] = None,
) -> dict:
    """Rejection rate of naive_and_corner_pvalue under a true joint H0, at
    several alpha thresholds. This is the measurement module docstring
    section 2 reports (0.207 at n=8, alpha=0.05) — regenerate rather than
    quote a stale number if this function or its inputs change."""
    rng = rng if rng is not None else np.random.default_rng(11)
    alphas = (0.01, 0.05, 0.10, 0.20)
    ps = np.empty(n_trials)
    for i in range(n_trials):
        dg, dl = _synthetic_null_draw(n_units, rng)
        ps[i] = naive_and_corner_pvalue(dg, dl, alternative="greater")["p_value"]
    return {
        "n_units": n_units, "n_trials": n_trials,
        "rejection_rate": {str(a): float((ps <= a).mean()) for a in alphas},
    }


def calibrate_joint_rank(
    n_units: int, n_trials: int = 3000, rng: Optional[np.random.Generator] = None,
) -> dict:
    """Same measurement as calibrate_naive_and_corner, for joint_rank_pvalue
    — the corrected statistic. Should land close to nominal at every
    alpha; module docstring section 2 quotes 0.049 at n=8, alpha=0.05."""
    rng = rng if rng is not None else np.random.default_rng(11)
    alphas = (0.01, 0.05, 0.10, 0.20)
    ps = np.empty(n_trials)
    for i in range(n_trials):
        dg, dl = _synthetic_null_draw(n_units, rng)
        ps[i] = joint_rank_pvalue(dg, dl, alternative="greater")["p_value"]
    return {
        "n_units": n_units, "n_trials": n_trials,
        "rejection_rate": {str(a): float((ps <= a).mean()) for a in alphas},
    }


def calibrate_intersection_union(
    n_units: int, n_trials: int = 3000, effect: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Rejection rate of intersection_union_pvalue on the THREE arms of the
    union null: complete (both axes pure noise), geometry-null (real logit
    effect, noise on geometry — the falsifier's configuration), and
    logit-null (real geometric effect, noise on logit). A valid test for a
    conjunction sits at or below nominal on all three; the min-rank
    statistic fails the second arm at ~0.17-0.19 (partial_pass_risk_demo).
    """
    rng = rng if rng is not None else np.random.default_rng(11)
    alphas = (0.01, 0.05, 0.10, 0.20)
    arms = {
        "complete_null": (0.0, 0.0),
        "geometry_null_logit_effect": (0.0, effect),
        "logit_null_geometry_effect": (effect, 0.0),
    }
    out = {"n_units": n_units, "n_trials": n_trials, "effect": effect, "arms": {}}
    for name, (mu_g, mu_l) in arms.items():
        ps = np.empty(n_trials)
        for i in range(n_trials):
            dg = rng.normal(loc=mu_g, scale=1.0, size=n_units)
            dl = rng.normal(loc=mu_l, scale=1.0, size=n_units)
            ps[i] = intersection_union_pvalue(dg, dl, alternative="greater")["p_value"]
        out["arms"][name] = {
            "rejection_rate": {str(a): float((ps <= a).mean()) for a in alphas},
        }
    return out


def partial_pass_risk_demo(
    n_units: int,
    n_trials: int = 2000,
    logit_effect: float = 1.0,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Quantifies EVALUABILITY.md's warning, on the falsifier's own
    configuration: a real logit effect (delta_logit ~ N(logit_effect, 1)
    per unit), pure noise on the geometric axis (delta_geometric ~ N(0, 1)
    per unit — no geometric effect at all). On each of n_trials synthetic
    draws, compares:

      logit_only_reject_rate : one_dimensional_pvalue on delta_logit ALONE
                                (what a reader sees if only the logit half
                                is reported) — NOT joint_rank_pvalue against
                                a zeroed geometric channel, which degenerates
                                instead of reducing to this (see
                                one_dimensional_pvalue's docstring).
      joint_reject_rate       : joint_rank_pvalue on the SAME draw, real
                                geometric noise included.

    Under the falsifier's configuration, H1 should NOT be supported — the
    geometric channel carries no effect, and joint_reject_rate should sit
    well below logit_only_reject_rate.

    Returns
    -------
    dict: n_units, n_trials, logit_effect, alpha, logit_only_reject_rate,
    joint_reject_rate, attainable_floor_one_sided.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    logit_only_rejects = 0
    joint_rejects = 0

    for _ in range(n_trials):
        delta_logit = rng.normal(loc=logit_effect, scale=1.0, size=n_units)
        delta_geometric = rng.normal(loc=0.0, scale=1.0, size=n_units)

        logit_p = one_dimensional_pvalue(delta_logit, alternative="greater")
        if logit_p <= alpha:
            logit_only_rejects += 1

        joint_p = joint_rank_pvalue(
            delta_geometric, delta_logit, alternative="greater",
        )["p_value"]
        if joint_p <= alpha:
            joint_rejects += 1

    return {
        "n_units": n_units,
        "n_trials": n_trials,
        "logit_effect": logit_effect,
        "alpha": alpha,
        "logit_only_reject_rate": logit_only_rejects / n_trials,
        "joint_reject_rate": joint_rejects / n_trials,
        "attainable_floor_one_sided": attainable_floor(n_units, "greater"),
    }


# ---------------------------------------------------------------------------
# 3. The measurement grid — matched pairs per prompt, measured
# ---------------------------------------------------------------------------

#: Excluded from n the same way P_I1_DOMINANT_PROMPT is excluded everywhere
#: else in this project: every token repeats, so its pair count is a fact
#: about the prompt, not about a checkpoint. See module docstring section 3.
DEGENERATE_PROMPT = "repeated_tokens"


def count_matched_pairs_by_prompt(tokenizer, min_offset: int = 2) -> dict:
    """
    Real (not synthetic) measurement: how many induction-matched (query,
    key) position pairs core.battery_structure.induction_candidates finds
    in each of core.config.PROMPTS, under the given tokenizer. No forward
    pass — tokenization only, so this needs no ablation machinery and no
    model weights beyond the tokenizer.

    Returns
    -------
    dict: {prompt_key: {"n_tokens": int, "n_pairs": int}}
    """
    from core.config import PROMPTS
    from core.battery_structure import induction_candidates

    out = {}
    for key, text in PROMPTS.items():
        ids = [int(i) for i in tokenizer(text)["input_ids"]]
        pairs = induction_candidates(ids, min_offset=min_offset)
        out[key] = {"n_tokens": len(ids), "n_pairs": len(pairs)}
    return out


def informative_prompt_count(pair_counts: dict, exclude=(DEGENERATE_PROMPT,)) -> int:
    """n for the floor formula: prompts in pair_counts minus the excluded
    degenerate ones, matching the P_I1_DOMINANT_PROMPT convention."""
    return sum(1 for k in pair_counts if k not in exclude)
