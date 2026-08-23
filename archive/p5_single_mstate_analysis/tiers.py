"""
p5_single_mstate_analysis/tiers.py — Group G, re-cut as an adjudicable
three-tier contrast.

What Group G claims and what it was missing
-------------------------------------------
Group G's result across six models was an ordering: the selected cluster is
more compact than a sibling cluster, which is more compact than a random
size-matched set. That ordering is what licenses reading Groups A-F as being
about a real object rather than an arbitrary subset of a cloud.

It had no significance statement. `sibling_contrast.py` built a hand-rolled
random baseline and compared point values. `core/nulls.py` now exists, so the
ordering can be stated as "Nσ from null" and put in a falsification table.

The control tier and the null are different objects
---------------------------------------------------
This is the distinction the module is built around, and conflating the two is
the easy error:

  **control tier** — ONE fixed random draw, frozen at selection time (D-5),
      measured at every checkpoint. It is a comparison *object* with a
      developmental trajectory: "these particular unrelated particles did
      this over training." Its variance across checkpoints is signal.

  **label-permutation null** — 200 random draws at ONE checkpoint, discarded
      immediately. It is a *distribution*: "what would any size-matched
      subset of this population have scored here." Its spread is the
      yardstick, not a result.

Reporting only the control gives an ordering with no error bar. Reporting
only the null loses the developmental comparison. Both are produced.

Which null for which question
-----------------------------
`label_permutation_null` (permute membership, activations fixed) answers "is
this set more compact than a random size-matched subset of the same tokens" —
the right chance baseline for every set-level statistic here.

`shuffled_dimension_null` (permute each feature dimension across tokens,
destroying joint structure while preserving per-dimension marginals) answers
"is there any geometric structure in this population at all" — the right
baseline for the population-level effective rank, and the one that says
whether a checkpoint's cloud is structured before asking whether a subset of
it is.

The pos0 caveat, and what it is NOT
-----------------------------------
The sink is excluded from the permutation pool under
`pos0_policy="excluded"`, so a null draw can never place position 0 inside the
set. The restriction is done by subsetting the arrays handed to the null, and
the remapped indices never escape that call — token positions outside it
always mean original positions (D-11).

The reason is **not** norm inflation. Every set-level statistic here is
computed on L2-normalized rows, so a token with a norm two orders above the
rest contributes a unit vector like any other and the null's spread barely
moves — measured at 0.0880 vs 0.0877 on a synthetic sink, i.e. nothing. An
earlier draft of this module asserted otherwise and the test caught it.

The reasons that do hold:

  1. **Raw-mode statistics.** `population_structure_null` and every raw
     quantity in `sweep_geometry` are norm-sensitive, and there the sink can
     set the spectrum single-handedly. The policy has to be one decision
     applied identically across sphere and raw quantities, or the two stop
     describing the same population.
  2. **The sink is functionally a different kind of particle.** With no BOS
     prepended it absorbs attention mass on behalf of every position. Whether
     it belongs in a "random size-matched subset of ordinary tokens" is a
     question about what the null is a null *of*, not a numerical convenience.

Stated plainly because the sphere-frame invariance is easy to mistake for
the policy doing nothing — it is doing something, just not the thing the
obvious argument suggests.

Pure numpy.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from core.nulls import (
    label_permutation_null,
    shuffled_dimension_null,
    nsigma_verdict,
)
from .token_sets import TokenSet
from .sweep_geometry import (
    DEFAULT_POS0_POLICY,
    _pos0_keep,
    eff_rank_from_gram,
    l2_normalize_rows,
    set_geometry,
)

__all__ = [
    "TIERS",
    "DEFAULT_N_PERMUTATIONS",
    "resultant_of_labelled",
    "separation_of_labelled",
    "tier_contrast",
    "tier_nulls",
    "population_structure_null",
    "ordering_consistency",
    "sweep_tier_records",
    "falsification_table_lines",
]

TIERS = ("primary", "sibling", "control")
DEFAULT_N_PERMUTATIONS = 200

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Metrics in metric_fn(activations, labels) form
# ---------------------------------------------------------------------------

def resultant_of_labelled(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Norm of the mean unit vector over the tokens labelled 1.

    In `metric_fn(activations, labels)` form so `label_permutation_null` can
    drive it directly. Bounded in [0, 1], scale-free, 1.0 iff every member
    points the same way — the same statistic `set_geometry` reports as
    `resultant_length`, so the observed value and the null are computed by
    identical code rather than by two definitions that might drift.
    """
    labels = np.asarray(labels)
    sel = labels == 1
    if not sel.any():
        return float("nan")
    U = l2_normalize_rows(np.asarray(X, dtype=np.float64))
    return float(np.linalg.norm(U[sel].mean(axis=0)))


def separation_of_labelled(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-set cosine minus mean member-to-nonmember cosine, in
    metric_fn form. NaN for sets below two members or with no complement —
    both are undefined, not zero."""
    labels = np.asarray(labels)
    sel = labels == 1
    n_in = int(sel.sum())
    if n_in < 2 or n_in == len(labels):
        return float("nan")
    U = l2_normalize_rows(np.asarray(X, dtype=np.float64))
    Us = U[sel]
    C = Us @ Us.T
    iu = np.triu_indices(n_in, k=1)
    within = float(np.mean(C[iu]))
    between = float(np.mean(Us @ U[~sel].T))
    return within - between


def _rank_of_labelled(X: np.ndarray, labels: np.ndarray) -> float:
    """Normed effective rank of the labelled subset. Lower means the set
    occupies fewer directions than a random subset would."""
    sel = np.asarray(labels) == 1
    if sel.sum() < 1:
        return float("nan")
    U = l2_normalize_rows(np.asarray(X, dtype=np.float64))[sel]
    return eff_rank_from_gram(U @ U.T)


_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "resultant_length": resultant_of_labelled,
    "separation": separation_of_labelled,
    "eff_rank_normed": _rank_of_labelled,
}


# ---------------------------------------------------------------------------
# Tier contrast
# ---------------------------------------------------------------------------

def tier_contrast(
    X: np.ndarray,
    token_set: TokenSet,
    pos0_policy: str = DEFAULT_POS0_POLICY,
) -> dict:
    """
    All three tiers' set geometry at one layer, plus the pairwise gaps.

    `ordering_holds` is the three-tier claim as a boolean:
    primary > sibling > control on `resultant_length`. Recorded per layer so
    the sweep-level statement is "held at N of M checkpoints" rather than an
    assertion from one favourable layer.

    A missing tier (no sibling was available at selection, say) gives
    `ordering_holds = None` — not False. "The claim failed" and "the claim
    could not be evaluated" must not be the same value in a falsification
    table.
    """
    X = np.asarray(X, dtype=np.float64)
    positions = {
        "primary": token_set.positions,
        "sibling": token_set.sibling_positions,
        "control": token_set.control_positions,
    }
    tiers = {t: set_geometry(X, p, pos0_policy)
             for t, p in positions.items() if p}

    out: dict = {"tiers": tiers, "ordering_holds": None, "gaps": {}}

    def _r(t):
        v = tiers.get(t, {}).get("resultant_length", float("nan"))
        return v if np.isfinite(v) else None

    p, s, c = _r("primary"), _r("sibling"), _r("control")
    if p is not None and s is not None:
        out["gaps"]["primary_minus_sibling"] = p - s
    if p is not None and c is not None:
        out["gaps"]["primary_minus_control"] = p - c
    if s is not None and c is not None:
        out["gaps"]["sibling_minus_control"] = s - c

    if p is not None and s is not None and c is not None:
        out["ordering_holds"] = bool(p > s > c)
    elif p is not None and c is not None:
        # Two-tier fallback: still a real claim, flagged as partial.
        out["ordering_holds"] = bool(p > c)
        out["partial"] = "no sibling tier; primary vs control only"
    return out


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

def tier_nulls(
    X: np.ndarray,
    positions: Sequence[int],
    metrics: Sequence[str] = ("resultant_length", "separation"),
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    sigma_threshold: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, dict]:
    """
    "Is this set more compact than a random size-matched subset of the same
    tokens?", per metric, as an Nσ verdict.

    The permutation pool is restricted by `pos0_policy`: under "excluded" the
    sink cannot be drawn into a null set. A token with a norm one to two
    orders above the rest, permuted into the membership slot, would widen the
    null and make a real effect look insignificant.

    That restriction is implemented by subsetting the arrays handed to
    `label_permutation_null`, so indices inside that call are pool-relative.
    They never escape: only summary statistics are returned. Every token
    position in this module's output is an original position (D-11).

    Set members excluded by the pos0 policy are dropped from the observed set
    too, so observed and null describe the same object — comparing a
    sink-inclusive observation against a sink-free null would be the exact
    mismatch this guards against.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    keep = _pos0_keep(n, pos0_policy)
    pool_idx = np.where(keep)[0]

    pos = sorted({int(p) for p in positions if 0 <= int(p) < n})
    pos_in_pool = [p for p in pos if keep[p]]

    out: Dict[str, dict] = {}
    if len(pos_in_pool) < 2 or len(pool_idx) <= len(pos_in_pool):
        for m in metrics:
            out[m] = {
                "observed": float("nan"), "z_score": float("nan"),
                "significant": False,
                "verdict_str": "not evaluable (set too small or pool exhausted)",
                "n_null": 0, "n_set_in_pool": len(pos_in_pool),
            }
        return out

    Xp = X[pool_idx]
    remap = {orig: k for k, orig in enumerate(pool_idx.tolist())}
    labels = np.zeros(len(pool_idx), dtype=np.int64)
    labels[[remap[p] for p in pos_in_pool]] = 1

    for m in metrics:
        fn = _METRICS[m]
        observed = fn(Xp, labels)
        null = label_permutation_null(Xp, labels, fn,
                                      n_permutations=n_permutations, rng=rng)
        finite = null[np.isfinite(null)]
        if not np.isfinite(observed) or finite.size < 2:
            out[m] = {
                "observed": float(observed), "z_score": float("nan"),
                "significant": False,
                "verdict_str": "not evaluable (degenerate null)",
                "n_null": int(finite.size), "n_set_in_pool": len(pos_in_pool),
            }
            continue
        v = nsigma_verdict(float(observed), finite,
                           sigma_threshold=sigma_threshold)
        v["n_set_in_pool"] = len(pos_in_pool)
        v["n_dropped_by_pos0"] = len(pos) - len(pos_in_pool)
        out[m] = v
    return out


def population_structure_null(
    X: np.ndarray,
    n_shuffles: int = DEFAULT_N_PERMUTATIONS,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    sigma_threshold: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    "Is there any joint geometric structure in this population at all?" —
    normed effective rank against `shuffled_dimension_null`.

    Logically prior to every set-level question: if a checkpoint's cloud is
    indistinguishable from independently-shuffled dimensions, then no subset
    of it can be a meaningful object, and a significant tier ordering there
    would need explaining rather than reporting. Cheap enough to run at every
    layer, and it is the row that makes an early-checkpoint null result
    interpretable instead of merely absent.
    """
    X = np.asarray(X, dtype=np.float64)
    keep = _pos0_keep(X.shape[0], pos0_policy)
    Xk = X[keep]
    if Xk.shape[0] < 3:
        return {"observed": float("nan"), "z_score": float("nan"),
                "significant": False,
                "verdict_str": "not evaluable (too few tokens)", "n_null": 0}

    def _rank(A):
        U = l2_normalize_rows(A)
        return eff_rank_from_gram(U @ U.T)

    observed = _rank(Xk)
    null = shuffled_dimension_null(Xk, _rank, n_shuffles=n_shuffles, rng=rng)
    finite = null[np.isfinite(null)]
    if finite.size < 2:
        return {"observed": float(observed), "z_score": float("nan"),
                "significant": False,
                "verdict_str": "not evaluable (degenerate null)", "n_null": 0}
    return nsigma_verdict(float(observed), finite,
                          sigma_threshold=sigma_threshold)


# ---------------------------------------------------------------------------
# Across the sweep
# ---------------------------------------------------------------------------

def sweep_tier_records(
    token_set: TokenSet,
    activations_by_step: Dict[int, np.ndarray],
    layers: Optional[Sequence[int]] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    sigma_threshold: float = 2.0,
    seed: int = 0,
) -> dict:
    """
    Tier contrast + nulls at each (checkpoint, layer).

    `activations_by_step` maps step -> (n_layers, n_tokens, d). Passed in
    rather than loaded so this is testable without a run directory; the
    caller uses `sweep_geometry.load_activations`.

    `layers` defaults to the deepest layer common to every checkpoint. Nulls
    at every layer of every checkpoint are affordable but rarely wanted — the
    claim is about where the object exists, not about the embedding layer.

    The RNG is seeded per (step, layer) so the null is reproducible and
    independent between cells. A single shared generator would make the null
    at step 143000 depend on how many layers were evaluated at step 0.
    """
    steps = sorted(activations_by_step)
    if not steps:
        return {"token_set": token_set.name, "prompt_key": token_set.prompt_key,
                "records": [], "skipped": ["no activations supplied"]}

    n_layers_common = min(activations_by_step[s].shape[0] for s in steps)
    if layers is None:
        layers = [n_layers_common - 1]

    records: List[dict] = []
    skipped: List[str] = []

    for step in steps:
        X_all = activations_by_step[step]
        if token_set.n_tokens_prompt and X_all.shape[1] != token_set.n_tokens_prompt:
            skipped.append(
                f"step {step}: token count {X_all.shape[1]} != anchor's "
                f"{token_set.n_tokens_prompt}; positions no longer identify "
                "the same particles"
            )
            continue
        for li in layers:
            if li >= X_all.shape[0]:
                skipped.append(f"step {step}: layer {li} beyond "
                               f"{X_all.shape[0]} layers")
                continue
            X = X_all[li]
            rng = np.random.default_rng(seed + 1_000_003 * step + li)
            contrast = tier_contrast(X, token_set, pos0_policy)
            records.append({
                "step": step,
                "layer": int(li),
                "contrast": contrast,
                "nulls": {
                    tier: tier_nulls(
                        X, pos, n_permutations=n_permutations,
                        pos0_policy=pos0_policy,
                        sigma_threshold=sigma_threshold,
                        rng=np.random.default_rng(
                            seed + 1_000_003 * step + li + 7919 * i),
                    )
                    for i, (tier, pos) in enumerate((
                        ("primary", token_set.positions),
                        ("sibling", token_set.sibling_positions),
                        ("control", token_set.control_positions),
                    )) if pos
                },
                "population": population_structure_null(
                    X, n_shuffles=n_permutations, pos0_policy=pos0_policy,
                    sigma_threshold=sigma_threshold, rng=rng),
            })

    return {
        "token_set": token_set.name,
        "prompt_key": token_set.prompt_key,
        "anchor_step": token_set.anchor_step,
        "pos0_policy": pos0_policy,
        "frame_kind": "l2_sphere",
        "n_permutations": n_permutations,
        "sigma_threshold": sigma_threshold,
        "layers": list(layers),
        "records": records,
        "skipped": skipped,
        "ordering": ordering_consistency(records),
    }


def ordering_consistency(records: Sequence[dict]) -> dict:
    """
    How often the three-tier ordering held, and where it didn't.

    Evaluable cells only: a cell with no sibling tier contributes to neither
    numerator nor denominator, so the fraction is not diluted by cells where
    the claim was never testable. `n_not_evaluable` is reported separately —
    a claim that held 4 of 4 times out of 27 cells is a different object from
    one that held 4 of 4 out of 4.
    """
    evaluable = [r for r in records if r["contrast"]["ordering_holds"] is not None]
    held = [r for r in evaluable if r["contrast"]["ordering_holds"]]
    failures = [{"step": r["step"], "layer": r["layer"],
                 "gaps": r["contrast"]["gaps"]}
                for r in evaluable if not r["contrast"]["ordering_holds"]]
    return {
        "n_cells": len(records),
        "n_evaluable": len(evaluable),
        "n_not_evaluable": len(records) - len(evaluable),
        "n_held": len(held),
        "fraction_held": (round(len(held) / len(evaluable), 4)
                          if evaluable else None),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Falsification table
# ---------------------------------------------------------------------------

def falsification_table_lines(sweep: dict) -> List[str]:
    """
    The table `status-5.md` was missing: one row per checkpoint, with the
    tier ordering and the Nσ verdict that adjudicates it.

    `n/a` rows are printed, not dropped. A checkpoint where the claim could
    not be evaluated is a fact about that checkpoint, and a table listing only
    the cells that produced numbers is the failure mode this whole rebuild
    keeps finding.
    """
    recs = sweep.get("records", [])
    lines = [
        f"Group G falsification — {sweep.get('token_set')} / "
        f"{sweep.get('prompt_key')} / frame={sweep.get('frame_kind')} "
        f"pos0={sweep.get('pos0_policy')} / "
        f"n_perm={sweep.get('n_permutations')}",
    ]
    if not recs:
        lines.append("(no cells evaluated)")
        for s in sweep.get("skipped", []):
            lines.append(f"  [skip] {s}")
        return lines

    lines.append(
        f"{'step':>8} {'layer':>5} {'order':>6} "
        f"{'primary_z':>10} {'sibling_z':>10} {'control_z':>10} {'pop_z':>8}"
    )

    def _z(rec, tier):
        d = rec["nulls"].get(tier, {}).get("resultant_length")
        if not d:
            return "n/a"
        z = d.get("z_score", float("nan"))
        return "n/a" if not np.isfinite(z) else f"{z:.1f}"

    for r in recs:
        oh = r["contrast"]["ordering_holds"]
        order = "n/a" if oh is None else ("yes" if oh else "NO")
        pz = r["population"].get("z_score", float("nan"))
        lines.append(
            f"{r['step']:>8} {r['layer']:>5} {order:>6} "
            f"{_z(r, 'primary'):>10} {_z(r, 'sibling'):>10} "
            f"{_z(r, 'control'):>10} "
            f"{('n/a' if not np.isfinite(pz) else f'{pz:.1f}'):>8}"
        )

    o = sweep.get("ordering", {})
    if o:
        lines.append(
            f"ordering held {o['n_held']}/{o['n_evaluable']} evaluable cells"
            + (f" ({o['fraction_held']:.2f})" if o.get("fraction_held") is not None else "")
            + (f"; {o['n_not_evaluable']} not evaluable"
               if o.get("n_not_evaluable") else "")
        )
        for f in o.get("failures", []):
            lines.append(f"  [fail] step {f['step']} layer {f['layer']}: "
                         + ", ".join(f"{k}={v:+.4f}" for k, v in f["gaps"].items()))
    for s in sweep.get("skipped", []):
        lines.append(f"  [skip] {s}")
    return lines
