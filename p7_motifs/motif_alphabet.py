"""
p7_motifs/motif_alphabet.py — the seven named motifs, as predicates over an
InteractionTable.

Fixed in advance, deliberately. The obvious alternative is to enumerate
every typed subgraph up to size 3 or 4 and report which are
over-represented; with millions of edges per checkpoint that produces a
motif zoo whose significant entries are limited only by patience and which
is not falsifiable by anything. Open-ended discovery is a later,
explicitly exploratory pass — see design-7.md, "Why the motif alphabet is
fixed in advance."

ALPHABET_VERSION is written into every motif_counts.json. If the alphabet
changes, counts computed under different versions are not comparable, and
the artifact says which one it was.

Six of the seven are single-edge predicates: a row either is or is not a
`prev_token` edge. The seventh, `relay`, is a two-edge composition across
layers and is the reason this module has more than a table of lambdas —
it is also the one that restates the induction head, so it gets the
attention.

Thresholds
----------
Every threshold here is PLACED, not calibrated (standing rule 6): none has
been derived from an observed force distribution, because no Pythia
interaction table exists yet. They are collected in DEFAULTS with that
status attached, so a run's artifact can record what it used and a later
pass can replace them with measured values without hunting through the
code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.interactions import InteractionTable

ALPHABET_VERSION = "1.0"

MOTIF_NAMES = ("prev_token", "match", "sink", "relay", "mutual", "hub", "repulsor")

# All PLACED, none calibrated. See module docstring.
DEFAULTS: Dict[str, float] = {
    # An edge counts as "attractive" when this much of its force lies in U_pos.
    "attractive_frac_min": 0.5,
    # An edge counts as "repulsive" when this much lies in U_neg.
    "repulsive_frac_min": 0.5,
    # Ignore edges carrying negligible force: they are noise in every count.
    "force_magnitude_min": 0.0,
    # `hub`: in-degree this many standard deviations above the LEAVE-ONE-OUT
    # mean of the other particles' in-degrees (see hub_mask for why the
    # candidate must be excluded from its own baseline).
    "hub_indegree_sigma": 2.0,
    # `hub` fallback when the leave-one-out spread is exactly zero (every
    # other particle has identical in-degree): require this multiple of the
    # leave-one-out mean instead. Without it, any excess at all would count.
    "hub_flat_multiple": 2.0,
}

THRESHOLD_STATUS = {k: "placed" for k in DEFAULTS}


def _attractive(t: InteractionTable, thresholds: Dict[str, float]) -> np.ndarray:
    """
    Attractive-channel mask.

    NaN attractive_frac means no U_pos projector was supplied, which is not
    the same as "not attractive" — it is "unknown". Unknown must not count
    as a motif occurrence, so it reads False here, and `motif_mask` reports
    how many rows were unknown so a caller can tell an honest zero from a
    missing projector.
    """
    frac = t.columns["attractive_frac"]
    mag = t.columns["force_magnitude"]
    with np.errstate(invalid="ignore"):
        return (frac >= thresholds["attractive_frac_min"]) & (
            mag > thresholds["force_magnitude_min"]
        )


def _repulsive(t: InteractionTable, thresholds: Dict[str, float]) -> np.ndarray:
    frac = t.columns["repulsive_frac"]
    mag = t.columns["force_magnitude"]
    with np.errstate(invalid="ignore"):
        return (frac >= thresholds["repulsive_frac_min"]) & (
            mag > thresholds["force_magnitude_min"]
        )


# ---------------------------------------------------------------------------
# Single-edge motifs
# ---------------------------------------------------------------------------

def prev_token_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """Attractive edge at offset -1: stage 1 of the induction circuit."""
    th = {**DEFAULTS, **(thresholds or {})}
    return _attractive(t, th) & (t.columns["offset"] == 1)


def match_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """
    Attractive edge on an induction pair: stage 2 of the induction circuit.

    "induction" and "strict" are both accepted — they are two formulations
    of the same idea and core/battery_structure.py reports both precisely
    because the repo has historically tested one while citing the other.
    Collapsing them here would hide that; what this does instead is count
    both as `match` and leave the divergence visible in battery_structure's
    own report, where it is a property of the prompt rather than of a motif.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    pt = t.columns["pair_type"]
    return _attractive(t, th) & ((pt == "induction") | (pt == "strict"))


def sink_mask(t: InteractionTable, thresholds=None, sink_position: int = 0) -> np.ndarray:
    """
    Edge into the attention sink.

    In the alphabet because Phase 6's null analysis found the same-content
    null set can collapse almost entirely onto position-0 pairs, at which
    point a "content" comparison is really comparing against sink
    behaviour. Sink edges must be separable or they will masquerade as
    every other motif. Note this deliberately does NOT require the edge to
    be attractive: sink behaviour is defined by where the attention goes,
    and asking whether sink edges are attractive is a question, not a
    definition.
    """
    return t.columns["source"] == sink_position


def repulsor_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """Edge in the repulsive channel — individuating pressure."""
    th = {**DEFAULTS, **(thresholds or {})}
    return _repulsive(t, th)


# ---------------------------------------------------------------------------
# Structural motifs (need the whole table, not a per-row predicate)
# ---------------------------------------------------------------------------

def mutual_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """
    Reciprocal attractive edges — a bound pair, the smallest metastable
    structure the dynamics admit. An edge is `mutual` when its reverse
    (same layer, same head, target and source swapped) is also present and
    also attractive.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    attr = _attractive(t, th)
    present = {
        (int(l), int(h), int(a), int(b))
        for l, h, a, b, ok in zip(
            t.columns["layer"], t.columns["head"],
            t.columns["target"], t.columns["source"], attr,
        )
        if ok
    }
    out = np.zeros(len(t), dtype=bool)
    for i, (l, h, a, b, ok) in enumerate(zip(
        t.columns["layer"], t.columns["head"],
        t.columns["target"], t.columns["source"], attr,
    )):
        if ok and (int(l), int(h), int(b), int(a)) in present:
            out[i] = True
    return out


def hub_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """
    Edges pointing INTO a local attractor: a source particle whose
    attractive in-degree, within its (layer, head), stands out from the
    other particles' in-degrees.

    "In-degree" here counts edges for which the particle is the SOURCE —
    the particle others are being pulled toward. That is the attractor,
    and it is the opposite of the graph-theoretic reading of the column
    name, which is why it is spelled out.

    Why leave-one-out, and not mean + k*sigma over all counts
    ---------------------------------------------------------
    A hub inflates the very statistic it would be compared against. Worse,
    the inflation is bounded in a way that makes the naive rule not merely
    conservative but *impossible* to satisfy: for n values, the largest
    achievable z-score is (n-1)/sqrt(n), which is below 2 for every
    n <= 4. So a "2 sigma above the mean" hub rule cannot fire at all in a
    small population, and fires only reluctantly in a large one — a single
    dominant attractor against 4 background particles scores exactly at
    the cutoff and is missed.

    That was found by the planted-attractor oracle test, not reasoned out
    in advance, which is the argument for having written the test first.

    The candidate is therefore excluded from its own baseline: its count is
    compared against the mean and standard deviation of the *other*
    particles' counts. When that leave-one-out spread is exactly zero
    (every other particle has identical in-degree, which is common on
    small synthetic inputs and on uniform attention), sigma-based
    comparison is undefined and `hub_flat_multiple` is used instead — a
    placed threshold, and recorded as one.

    A layer where every particle has the same in-degree has no hub, which
    is correct: a uniform attention pattern has no attractor.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    attr = _attractive(t, th)
    out = np.zeros(len(t), dtype=bool)

    layers, heads, sources = t.columns["layer"], t.columns["head"], t.columns["source"]
    for key in {(int(l), int(h)) for l, h in zip(layers, heads)}:
        sel = (layers == key[0]) & (heads == key[1]) & attr
        if not sel.any():
            continue
        uniq, counts = np.unique(sources[sel], return_counts=True)
        if len(uniq) < 2:
            # One attractor and nothing to compare it against. Undecidable,
            # not a hub: calling it one would make every single-source head
            # a hub by default.
            continue

        counts = counts.astype(np.float64)
        total, n = counts.sum(), len(counts)
        # Leave-one-out mean and std, vectorised.
        loo_mean = (total - counts) / (n - 1)
        loo_var = ((counts**2).sum() - counts**2) / (n - 1) - loo_mean**2
        loo_std = np.sqrt(np.maximum(loo_var, 0.0))

        flat = loo_std == 0
        is_hub = np.where(
            flat,
            counts > th["hub_flat_multiple"] * loo_mean,
            counts > loo_mean + th["hub_indegree_sigma"] * loo_std,
        )
        hubs = set(uniq[is_hub].tolist())
        if hubs:
            out |= sel & np.isin(sources, list(hubs))
    return out


# ---------------------------------------------------------------------------
# relay — the induction head, restated
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelayInstance:
    """One two-stage composition. `tag_position` is the particle that
    carries the tag: it is the TARGET of the stage-1 prev_token edge and
    the SOURCE of the stage-2 match edge, which is the whole content of the
    composition."""
    layer_1: int
    head_1: int
    layer_2: int
    head_2: int
    tag_position: int
    match_target: int


def find_relays(t: InteractionTable, thresholds=None) -> list:
    """
    Every `relay`: a prev_token edge at layer L1 whose target particle is
    the source of a match edge at layer L2 > L1.

    The strict L2 > L1 ordering is the point. Information has to be written
    into the tag-carrying particle before a later layer can match on it, so
    a composition within one layer, or backwards, is not a relay — it is a
    coincidence of two edges. Enforcing this is also what keeps `relay`
    from reducing to "a match edge exists", which is the behavioural
    induction score and would make the whole comparison tautological (see
    design-7.md, "The tautology risk").
    """
    th = {**DEFAULTS, **(thresholds or {})}
    prev = np.flatnonzero(prev_token_mask(t, th))
    match = np.flatnonzero(match_mask(t, th))
    if prev.size == 0 or match.size == 0:
        return []

    col = t.columns
    # Index stage-2 edges by the particle that sources them.
    by_source: Dict[int, list] = {}
    for j in match:
        by_source.setdefault(int(col["source"][j]), []).append(j)

    out = []
    for i in prev:
        tag = int(col["target"][i])          # the particle the tag was written into
        l1 = int(col["layer"][i])
        for j in by_source.get(tag, ()):
            l2 = int(col["layer"][j])
            if l2 <= l1:
                continue
            out.append(RelayInstance(
                layer_1=l1, head_1=int(col["head"][i]),
                layer_2=l2, head_2=int(col["head"][j]),
                tag_position=tag, match_target=int(col["target"][j]),
            ))
    return out


def relay_strength(t: InteractionTable, thresholds=None) -> Dict[tuple, int]:
    """
    Relay count per (layer_1, head_1, layer_2, head_2) — the head-pair
    resolved form P-I1 and P-I3 are adjudicated on.

    Counts, not rates. The denominator (how many relays were *possible*
    given the prompt's induction-pair structure) is prompt-dependent and
    belongs with the null comparison in motif_stats, not baked in here
    where it would silently become part of the definition.
    """
    counts: Dict[tuple, int] = {}
    for r in find_relays(t, thresholds):
        key = (r.layer_1, r.head_1, r.layer_2, r.head_2)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SINGLE_EDGE = {
    "prev_token": prev_token_mask,
    "match":      match_mask,
    "sink":       sink_mask,
    "mutual":     mutual_mask,
    "hub":        hub_mask,
    "repulsor":   repulsor_mask,
}


def motif_mask(name: str, t: InteractionTable, thresholds=None) -> dict:
    """
    Boolean mask for one single-edge or structural motif, plus the counts a
    reader needs to interpret it.

    `unknown_channel` is the number of rows whose channel fraction was NaN
    — no projector supplied. Those rows read False in the mask, so without
    this number a zero count from "we never loaded the projectors" is
    indistinguishable from an honest zero. Reporting it is standing rule 3
    (every gate records what it read) applied at the motif level.

    `relay` is not available here: it is a two-edge composition and returns
    instances rather than a row mask. Use find_relays / relay_strength.
    """
    if name == "relay":
        raise ValueError(
            "relay is a two-edge composition and has no per-row mask; "
            "use find_relays() or relay_strength()."
        )
    if name not in _SINGLE_EDGE:
        raise ValueError(f"Unknown motif {name!r}. Known: {sorted(MOTIF_NAMES)}")

    th = {**DEFAULTS, **(thresholds or {})}
    mask = _SINGLE_EDGE[name](t, th)
    frac = ("repulsive_frac" if name == "repulsor" else "attractive_frac")
    unknown = int(np.isnan(t.columns[frac]).sum()) if name != "sink" else 0

    return {
        "motif": name,
        "alphabet_version": ALPHABET_VERSION,
        "mask": mask,
        "count": int(mask.sum()),
        "n_edges": len(t),
        "unknown_channel": unknown,
        "thresholds": {k: th[k] for k in DEFAULTS},
        "threshold_status": dict(THRESHOLD_STATUS),
    }
