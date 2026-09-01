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
#
# Every motif below joins rows BY PARTICLE POSITION, and `target` / `source`
# are per-prompt token indices. Position 7 of one prompt and position 7 of
# another are different particles, and p7_motifs/run_7.py writes all 8
# battery prompts into a single table — so an ungrouped join is the reading
# production takes, and it does not error. `_context_ids` names the scope a
# position is meaningful in; anything joining on position groups by it first.


def _composite_ids(*arrays) -> np.ndarray:
    """One int64 id per row for the tuple of column values.

    Lets a group be formed by a single sort rather than one full-length
    boolean scan per group, which matters here: a battery table carries
    ~10^7 rows and thousands of (context, layer, head) groups.
    """
    n = len(arrays[0]) if arrays else 0
    ids = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ids
    stride = 1
    for a in arrays:
        _, inv = np.unique(a, return_inverse=True)
        inv = np.asarray(inv, dtype=np.int64).ravel()
        ids += inv * stride
        stride *= int(inv.max()) + 1
    return ids


def _context_ids(t: InteractionTable) -> np.ndarray:
    """The (model, checkpoint_step, prompt_key) each row's positions belong
    to, as an integer id. Checkpoint and model are in the key because
    InteractionTable.concat does not forbid mixing them either, and the
    same collision follows."""
    c = t.columns
    return _composite_ids(c["model"], c["checkpoint_step"], c["prompt_key"])


def _groups(keys: np.ndarray):
    """Row indices grouped by `keys`, one sort for the whole table."""
    order = np.argsort(keys, kind="stable")
    if order.size == 0:
        return
    sk = keys[order]
    bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield order[a:b]



def mutual_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """
    Reciprocal attractive edges — a bound pair, the smallest metastable
    structure the dynamics admit. An edge is `mutual` when its reverse
    (same context, same layer, same head, target and source swapped) is
    also present and also attractive.

    "Same context" is load-bearing in principle: without it, an edge 5<-4
    in one prompt and an edge 4<-5 in another read as each other's
    reverse, and BOTH are reported as a bound pair that neither prompt
    contains. On pythia-410m at step 54000 the collision does not in fact
    fire — pooled and grouped both give 105,752 — so this is a latent
    defect, fixed for correctness and not because a number moved. The
    oracle test plants the collision rather than relying on the data to
    produce it.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    attr = _attractive(t, th)
    ctx = _context_ids(t)
    cols = t.columns
    rows = np.flatnonzero(attr)
    present = set(zip(
        ctx[rows].tolist(), cols["layer"][rows].tolist(),
        cols["head"][rows].tolist(), cols["target"][rows].tolist(),
        cols["source"][rows].tolist(),
    ))
    out = np.zeros(len(t), dtype=bool)
    for i in rows:
        key = (int(ctx[i]), int(cols["layer"][i]), int(cols["head"][i]),
               int(cols["source"][i]), int(cols["target"][i]))   # reversed
        if key in present:
            out[i] = True
    return out


def hub_mask(t: InteractionTable, thresholds=None) -> np.ndarray:
    """
    Edges pointing INTO a local attractor: a source particle whose
    attractive in-degree, within its (context, layer, head), stands out
    from the other particles' in-degrees.

    The context is part of the group because in-degrees would otherwise be
    POOLED across prompts: a position's count becomes a sum over texts it
    has nothing to do with, and so does the leave-one-out baseline it is
    compared against. Both directions of error are reachable — two flat
    prompts can pool into an apparent attractor neither contains (the
    oracle test plants exactly that), and a real attractor can be buried
    under a baseline the other prompts inflated.

    On pythia-410m at step 54000 the second dominates: 305,233 hub edges
    pooled against 437,508 grouped, so pooling LOST 30% of the real
    per-prompt hubs rather than inventing any. That direction was not the
    one predicted before measuring it.

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
    for rows in _groups(_composite_ids(_context_ids(t), layers, heads)):
        rows = rows[attr[rows]]
        if rows.size == 0:
            continue
        uniq, counts = np.unique(sources[rows], return_counts=True)
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
        if is_hub.any():
            out[rows[np.isin(sources[rows], uniq[is_hub])]] = True
    return out


# ---------------------------------------------------------------------------
# relay — the induction head, restated
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelayInstance:
    """One two-stage composition. `tag_position` is the particle that
    carries the tag: it is the TARGET of the stage-1 prev_token edge and
    the SOURCE of the stage-2 match edge, which is the whole content of the
    composition.

    `prompt_key` is carried because `tag_position` is meaningless without
    it: positions are per-prompt token indices, so the same integer names
    different particles in different prompts. Anything that resolves a
    relay back to a particle needs both (see p7_motifs/events.py's
    relay_target_flags)."""
    layer_1: int
    head_1: int
    layer_2: int
    head_2: int
    tag_position: int
    match_target: int
    prompt_key: str


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

    The join is confined to one (model, checkpoint_step, prompt_key)
    context, because that is the only scope in which a particle position
    names a particle. A table holding more than one prompt — which is what
    p7_motifs/run_7.py writes, all 8 of the battery into one artifact — is
    otherwise joined tag-to-match ACROSS prompts, composing a tag written
    in one text with a match found in another. Nothing errors: the
    positions collide and the result reads as a relay. Measured on
    pythia-410m at step 54000 that was 23,050,007 relays against the
    2,560,483 the per-prompt join gives, a 9.0x inflation, entirely
    spurious. The grouping belongs here rather than in each caller: every
    caller would otherwise have to know it, and the ones that forgot would
    fail silently.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    prev = np.flatnonzero(prev_token_mask(t, th))
    match = np.flatnonzero(match_mask(t, th))
    if prev.size == 0 or match.size == 0:
        return []

    col = t.columns
    ctx = _context_ids(t)
    prompts = col["prompt_key"]

    # Index stage-2 edges by the particle that sources them, WITHIN the
    # context that particle belongs to.
    by_source: Dict[tuple, list] = {}
    for j in match:
        by_source.setdefault((int(ctx[j]), int(col["source"][j])), []).append(j)

    out = []
    for i in prev:
        tag = int(col["target"][i])          # the particle the tag was written into
        l1 = int(col["layer"][i])
        prompt = str(prompts[i])
        for j in by_source.get((int(ctx[i]), tag), ()):
            l2 = int(col["layer"][j])
            if l2 <= l1:
                continue
            out.append(RelayInstance(
                layer_1=l1, head_1=int(col["head"][i]),
                layer_2=l2, head_2=int(col["head"][j]),
                tag_position=tag, match_target=int(col["target"][j]),
                prompt_key=prompt,
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

    HOW MUCH that denominator explains is now measured rather than left as
    a caveat, because the size of it decides what a null has to hold fixed.
    Across the 8 battery prompts at step 54000, the raw relay count against
    the prompt's own induction-pair supply (`n_induction` from
    `core.battery_structure.analyze_prompt`) runs r = **+0.9958** — 99% of
    the cross-prompt variance in this count is the prompt's combinatorics,
    not the model's circuitry. Excluding `repeated_tokens` it is +0.8908.
    No other structural quantity comes close: n_tokens -0.39,
    n_same_content -0.36, n_distinct_tokens -0.79.

    That also explains `repeated_tokens` carrying 61% of the battery's
    relays. ". . . ." x 265 holds **34,191** induction pairs against the
    next prompt's 2,873 — twelve times as many, because every repeated
    token pairs with every other. Its share is a fact about the prompt, not
    about the checkpoint.

    The consequence for the null, and it is a constraint rather than an
    observation: a relay-count null that does not hold `n_induction` fixed
    per prompt is testing whether the prompt has induction pairs, which is
    known before the model runs. That is EVALUABILITY.md's "a null that
    randomised more than the claim is about". What is left after
    normalising is not nothing — relays per induction pair still spans 45
    to 133 across the battery, a factor of three — and that residue is
    where a formation signal would have to live.
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
