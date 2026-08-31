"""
p7_motifs/formation_curve.py — the checkpoint series P-I1 is adjudicated on.

The other half of `status-7.md`'s build step 8. `run_7.py` produces one
checkpoint's `interaction_table.npz`; this module turns a series of them
into `formation_curve.json`, whose contract has existed in
`core/artifacts.py` since 2026-08-22 and which `p_value_p_i1` reads.

Two things here are the AUTHOR's decisions and are therefore required
arguments with no default, in the idiom `p7_io.load_sign_channel` already
uses for `sign_channel`. A default would make the choice invisible in a
result that depends on it.

1. `relay_owner` — WHICH HEAD OWNS A RELAY
   -------------------------------------
   `motif_alphabet.relay_strength` is keyed by
   `(layer_1, head_1, layer_2, head_2)`: a relay is a composition of two
   edges in two different heads, and that pairing is the whole content of
   the motif. But `formation_gate.P_I1_UNIT` is `"head"`, singular, and
   PREDICTIONS.md's first adjudication constraint fixes it there —
   "effective n is the number of heads, not the number of edges". So a
   pair-keyed statistic has to be collapsed onto a head axis, and how it
   collapses is a definition of what the motif measures:

     "tag_writer"  credit head_1, which writes the tag into the particle.
                   The prev-token side. Formation of stage 1.
     "matcher"     credit head_2, which matches on it. The induction side,
                   and the one whose behavioural score P-I1 pairs against —
                   which is also the direction in which the tautology risk
                   design-7.md names is largest.
     "both"        credit each relay to both heads. Doubles the total mass
                   and makes the two arms share heads, so a head can appear
                   as both stages of the same relay.

   The three are not rescalings of one another: a head that writes many
   tags and matches none has a large "tag_writer" strength and zero
   "matcher" strength.

2. `independence_source` — already required by the artifact contract, for
   the reason recorded there: a result that cannot name what makes the
   motif independent of the behavioural score has measured the same
   quantity twice.

WHAT THIS MODULE DOES NOT DO, AND WILL NOT PRETEND TO
-----------------------------------------------------
`formation_gate`'s docstring: "The series handed in must already be the
ABOVE-NULL excess" — relay strength minus the N1/N2 offset-null envelope.
Producing those null VALUES is not `motif_stats`'s job (it adjudicates
them) and not this module's; `core/qk_offset_null.py` computes N1/N2 for
the QK antisymmetry statistic, not for relay counts, and a relay-count
null does not exist in this repository. So `above_null_excess` is a
required flag, it is stamped into the artifact, and `assert_gate_ready`
refuses a payload that is not one. A raw relay count handed to
`p_value_p_i1` would produce a p-value against a null the series never
cleared.

THE BEHAVIOURAL SCORE IS COMPUTED FROM THE ATTENTION TENSOR, NOT THE TABLE
--------------------------------------------------------------------------
It would be one line to average `weight` over the interaction table's
`pair_type == "induction"` rows, and it would be wrong. The table is
thinned by a top-k-by-force retention cutoff, so that average is over the
induction pairs that SURVIVED retention — which selects on force magnitude,
the very quantity the motif side is built from. The two arms would then
share a selection step and the pairing null could not separate them.
Reading the full attention matrix costs an extra artifact and keeps the
arms independent.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from core.interactions import InteractionTable
from .motif_alphabet import find_relays

#: Who a relay is credited to. No default; see the module docstring.
RELAY_OWNER_CHOICES = ("tag_writer", "matcher", "both")

#: The three the artifact contract admits.
INDEPENDENCE_SOURCES = ("two_stage", "force_channel", "particle_event")


class FormationCurveRefused(ValueError):
    """A payload this module will not assemble or hand onward."""


def per_head_relay_strength(t: InteractionTable, relay_owner: str,
                            thresholds: Optional[dict] = None) -> Dict[tuple, float]:
    """
    Relay counts collapsed onto (layer, head) under `relay_owner`.

    Counts, not rates, for `relay_strength`'s reason: the denominator is
    prompt-dependent and belongs with the null comparison rather than in
    the definition.
    """
    if relay_owner not in RELAY_OWNER_CHOICES:
        raise FormationCurveRefused(
            f"relay_owner must be one of {list(RELAY_OWNER_CHOICES)}; got "
            f"{relay_owner!r}. There is no default: a relay is a composition "
            "of two heads and which one carries it is what the per-head "
            "series means."
        )
    out: Dict[tuple, float] = {}
    for r in find_relays(t, thresholds):
        keys = []
        if relay_owner in ("tag_writer", "both"):
            keys.append((r.layer_1, r.head_1))
        if relay_owner in ("matcher", "both"):
            keys.append((r.layer_2, r.head_2))
        for k in keys:
            out[k] = out.get(k, 0.0) + 1.0
    return out


def behavioural_induction_score(attentions, induction_pairs) -> Dict[tuple, float]:
    """
    Mean post-softmax attention on induction pairs, per (layer, head).

    PREDICTIONS.md's wording for the behavioural score. `attentions` is the
    (n_layers, n_heads, n_tokens, n_tokens) tensor Phase 1 writes;
    `induction_pairs` are (query, key) with query > key, the convention
    `core.battery_structure.induction_candidates` returns and the one
    `a_frac_mat[query, key]` is indexed by.

    Empty pair set returns {} rather than zeros: "this prompt carried no
    induction pairs" and "these heads attend to none of them" are the two
    readings that must not collapse.
    """
    a = np.asarray(attentions)
    if a.ndim != 4:
        raise FormationCurveRefused(
            f"attentions must be (n_layers, n_heads, n_tokens, n_tokens); "
            f"got shape {a.shape}")
    pairs = list(induction_pairs)
    if not pairs:
        return {}
    q = np.asarray([p[0] for p in pairs], dtype=int)
    k = np.asarray([p[1] for p in pairs], dtype=int)
    n = a.shape[-1]
    if q.max() >= n or k.max() >= n:
        raise FormationCurveRefused(
            f"induction pair index {max(int(q.max()), int(k.max()))} is outside "
            f"an attention matrix of {n} tokens; the pairs and the tensor are "
            "from different tokenizations")
    vals = a[:, :, q, k]                       # (n_layers, n_heads, n_pairs)
    means = vals.mean(axis=-1)
    return {(int(l), int(h)): float(means[l, h])
            for l in range(means.shape[0]) for h in range(means.shape[1])}


def _head_axis(per_step: Sequence[Dict[tuple, float]]) -> list:
    """The (layer, head) keys present at EVERY checkpoint, sorted.

    Intended for a DENSE series only — one computed for every head the
    model has, so that a missing key means the checkpoint did not have that
    head rather than that the head scored nothing. Handed a sparse series
    this intersection is actively wrong; see `formation_curve_payload`.
    """
    if not per_step:
        return []
    common = set(per_step[0])
    for d in per_step[1:]:
        common &= set(d)
    return sorted(common)


def formation_curve_payload(
    steps: Sequence[int],
    relay_by_step: Sequence[Dict[tuple, float]],
    score_by_step: Sequence[Dict[tuple, float]],
    *,
    independence_source: str,
    relay_owner: str,
    above_null_excess: bool,
    thresholds: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Assemble `formation_curve.json` against its registered contract.

    `relay_by_step` and `score_by_step` are per-checkpoint maps from
    (layer, head) to that head's value, in the same order as `steps`. The
    two series come out indexed by a shared head axis, which is what
    `p_value_p_i1` requires — it pairs rs[i] against bs[i] as the same head.
    """
    if independence_source not in INDEPENDENCE_SOURCES:
        raise FormationCurveRefused(
            f"independence_source must be one of {list(INDEPENDENCE_SOURCES)}; "
            f"got {independence_source!r}. The contract requires it because a "
            "result that cannot name one has measured the same quantity twice."
        )
    if relay_owner not in RELAY_OWNER_CHOICES:
        raise FormationCurveRefused(
            f"relay_owner must be one of {list(RELAY_OWNER_CHOICES)}")
    n = len(steps)
    if not (len(relay_by_step) == len(score_by_step) == n):
        raise FormationCurveRefused(
            f"{n} steps, {len(relay_by_step)} relay maps, "
            f"{len(score_by_step)} score maps; these index the same "
            "checkpoints and must match")
    if n < 2:
        raise FormationCurveRefused(
            f"a formation curve needs at least two checkpoints; got {n}. "
            "P-I1 is about whether two curves RISE together, which one point "
            "cannot show.")

    order = np.argsort(np.asarray(steps, dtype=float))
    steps_sorted = [int(np.asarray(steps)[i]) for i in order]
    relay_sorted = [relay_by_step[i] for i in order]
    score_sorted = [score_by_step[i] for i in order]

    # The axis comes from the BEHAVIOURAL series, which is dense: it is
    # computed for every (layer, head) in the attention tensor, so a head
    # missing from it is a head the checkpoint did not have. The relay
    # series is sparse by construction — `per_head_relay_strength` omits a
    # head with no relays — and 0.0 is the correct fill for it, because
    # "measured, and had no relays" is exactly what P-I1 predicts of an
    # early checkpoint.
    #
    # Intersecting on the relay side instead drops every head that had no
    # relay at any one checkpoint, which is precisely the set of heads that
    # go on to form. Measured on the real sweep at step 1000 vs step 54000,
    # that axis would have been empty: step 1000 has no relays at all at the
    # registered threshold.
    heads = _head_axis(score_sorted)
    if not heads:
        raise FormationCurveRefused(
            "no (layer, head) appears at every checkpoint of the behavioural "
            "series, so there is no head axis the two arms share. That means "
            "the checkpoints disagree about the model's head geometry, not "
            "that a head scored nothing.")

    motif = [[float(r.get(h, 0.0)) for r in relay_sorted] for h in heads]
    behav = [[float(s[h]) for s in score_sorted] for h in heads]

    payload = {
        "checkpoint_steps": steps_sorted,
        "motif_strength": motif,
        "behavioral_induction_score": behav,
        "independence_source": independence_source,
        "relay_owner": relay_owner,
        "series_is_above_null_excess": bool(above_null_excess),
        "heads": [list(h) for h in heads],
        "n_heads": len(heads),
        "thresholds": dict(thresholds) if thresholds else None,
    }
    if extra:
        payload.update(extra)
    return payload


def assert_gate_ready(payload: dict) -> dict:
    """
    Refuse a payload `p_value_p_i1` must not be handed.

    The gate's own docstring requires the ABOVE-NULL excess — "strength
    above N1 and N2". A raw relay count is not that, and handing one over
    yields a p-value against a null the series never cleared, which is the
    reading `compare_against_nulls` refuses one level down for the same
    reason. This is the same refusal at the series level.
    """
    if not payload.get("series_is_above_null_excess"):
        raise FormationCurveRefused(
            "this curve holds raw relay strength, not the excess above the "
            "N1/N2 offset-null envelope that P-I1's wording requires. No "
            "relay-count null exists in this repository — core/qk_offset_null.py "
            "computes N1/N2 for the QK antisymmetry statistic, not for relay "
            "counts. Adjudicating this series would report a p-value against "
            "a null it never cleared."
        )
    return payload
