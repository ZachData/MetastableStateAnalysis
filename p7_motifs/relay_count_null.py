"""
p7_motifs/relay_count_null.py — P-I1's relay-count null (PROJECT.md §3.4).

The construction the author registered 2026-09-04, after the walk-through
PROJECT.md §3.4 asked for: degree-preserving AT THE HEAD LEVEL, not per
particle. `core/qk_offset_null.py` computes N1/N2 for the QK antisymmetry
statistic; this is the sibling for relay counts, which that module does not
and was never meant to cover.

THE MECHANICS THE CONSTRUCTION LEANS ON
----------------------------------------
`motif_alphabet.py` types a relay from two independent facts about an edge:
`pair_type` (induction / strict / same_content / neither) and `offset`
(target - source) are pure facts about WHERE an edge points, given the
prompt's tokenisation — computed once by `classify_pair_types`, never by the
model. `attractive_frac` / `repulsive_frac` / `force_magnitude` / `weight`
are facts about the edge's FORCE, computed from the OV circuit. A relay is a
stage-1 edge (offset==1, attractive) whose target is the source of a stage-2
edge (pair_type in {induction, strict}, attractive) in a later layer, same
prompt (`find_relays`).

Because those two axes are independent, the null is a payload shuffle: for
each (prompt, layer, head), draw a fresh set of DISTINCT causal positions,
uniformly at random from every (source, target) with 0 <= source < target <
n_tokens, one per real retained edge, and reattach that edge's entire
force-derived payload to it unchanged. `offset` and `pair_type` are
recomputed from the new position; every other column — including which
edges are "attractive" at all — is untouched.

What that holds fixed, and why it is enough
--------------------------------------------
* Each head's edge COUNT, exactly (a resample of the same size).
* The ENTIRE force distribution — not just an aggregate like "attractive
  fraction" the way PROJECT.md §3.4 first phrased it: the actual weight,
  force_magnitude, attractive_frac, repulsive_frac, real_frac, imag_frac
  values ride along with whichever row they came from, so the attractive
  fraction (and every other channel statistic) is preserved exactly, not
  approximately.
* `n_induction` PER PROMPT, automatically. §3.4's first constraint —
  "otherwise it tests whether the prompt has induction pairs, which is known
  before the model runs" — falls out with no separate bookkeeping, because
  the position pool and the induction/strict/same-content candidate sets are
  properties of the prompt's tokenisation alone (`PromptNullContext`) and
  are identical at every checkpoint and every replicate.

What is NOT held fixed, by the author's decision (2026-09-04)
---------------------------------------------------------------
Per-particle in/out-degree. A double-edge-swap configuration-model null
would additionally keep how many edges touch each token position fixed —
the standard construction for network-motif over-representation testing —
and would control for a hub particle attracting disproportionate incoming
force regardless of content, the way `motif_alphabet.hub_mask`'s
leave-one-out baseline already does for a different statistic. The author
considered it and chose the lighter head-level construction; this module
does not build the heavier one.

WHY THE RELAY COUNT ITSELF IS SCORED BY MONTE CARLO, NOT A CLOSED FORM
------------------------------------------------------------------------
A single-edge motif's null count under this shuffle is an exact hypergeometric
(or a short compound of two), because "is this edge attractive" and "does its
new position carry offset==1 / an induction pair_type" are independent given
the shuffle. A RELAY is a composition of two such edges across two heads,
joined by "stage 1's target equals stage 2's source" — position equality
across two INDEPENDENT random draws — and getting that composition's exact
null distribution analytically risks exactly the class of subtle error this
project's own history warns about (UPDATE_PLAN.md §5.6; `core/interactions.py`
projector-shape bug). So this module does not re-derive relay-composition
math: it draws K shuffled replicates of the real table and reruns
`motif_alphabet.find_relays` / `formation_curve.per_head_relay_strength`
unchanged on each one — the same tested pipeline the real series is scored
with — and reports the mean and sd across replicates as the N1/N2 envelope.
Slower than a formula; verified against nothing but itself and cheap to
audit, which is the trade EVALUABILITY.md's order generally prefers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.interactions import InteractionTable

#: (target, source) is encoded as one int64 key = target * BASE + source, for
#: a vectorised replacement of `classify_pair_types`' per-row Python loop.
#: Every prompt this project uses is well under 4096 tokens (§1's grid tops
#: out at 512), so this leaves an enormous margin while staying a pure
#: shift-and-add.
_POSITION_ENCODING_BASE = 1 << 20


class RelayNullRefused(ValueError):
    """An input this null will not shuffle or score."""


def _encode(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    return (np.asarray(targets, dtype=np.int64) * _POSITION_ENCODING_BASE
            + np.asarray(sources, dtype=np.int64))


def causal_pool(n_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Every (source, target) with 0 <= source < target < n_tokens: the full
    causal position space a head's edges could occupy, independent of which
    ones a real head actually kept under top-k-by-force retention.
    """
    if n_tokens < 2:
        raise RelayNullRefused(
            f"n_tokens={n_tokens} has no causal (source, target) pair to "
            f"draw from")
    source, target = np.triu_indices(n_tokens, k=1)
    return source.astype(np.int64), target.astype(np.int64)


@dataclass(frozen=True)
class PromptNullContext:
    """
    Everything the null needs about ONE prompt, computed once from its
    tokenisation and reused across every checkpoint and every replicate —
    nothing here is a function of either, which is what makes `n_induction`
    fixed per prompt automatic rather than a separate constraint to enforce.
    """
    prompt_key: str
    n_tokens: int
    source_pool: np.ndarray
    target_pool: np.ndarray
    induction_keys: np.ndarray      # sorted, encoded
    strict_keys: np.ndarray
    same_content_keys: np.ndarray

    @property
    def pool_size(self) -> int:
        return int(len(self.source_pool))


def build_prompt_context(prompt_key: str, ids: Sequence[int]) -> PromptNullContext:
    """
    From a prompt's token ids — already truncated to the width its
    attention/interaction tables were built at; callers are responsible for
    that, exactly as `run_7.py` and `tools/run/behavioural.py` are — compute
    the induction / strict / same-content candidate sets and the causal
    pool ONCE.
    """
    from core.battery_structure import induction_candidates, same_content_candidates

    n = len(ids)
    source_pool, target_pool = causal_pool(n)
    ind = induction_candidates(ids)
    strict = induction_candidates(ids, strict=True)
    same = same_content_candidates(ids, ind)

    def _keys(pairs) -> np.ndarray:
        if not pairs:
            return np.array([], dtype=np.int64)
        q = np.asarray([p[0] for p in pairs], dtype=np.int64)
        k = np.asarray([p[1] for p in pairs], dtype=np.int64)
        return np.sort(_encode(q, k))

    return PromptNullContext(
        prompt_key=prompt_key, n_tokens=n,
        source_pool=source_pool, target_pool=target_pool,
        induction_keys=_keys(ind), strict_keys=_keys(strict),
        same_content_keys=_keys(same),
    )


def _pair_types(targets: np.ndarray, sources: np.ndarray,
                ctx: PromptNullContext) -> np.ndarray:
    """
    Same precedence as `core.interactions.classify_pair_types` — induction >
    strict > same_content > neither — vectorised via the encoded key rather
    than a per-row Python loop. Pinned equal to it in
    `tests/test_p7_relay_count_null.py::TestPairTypesAgreeWithClassifyPairTypes`.
    """
    key = _encode(targets, sources)
    out = np.full(len(key), "neither", dtype="<U12")
    if ctx.induction_keys.size:
        out[np.isin(key, ctx.induction_keys)] = "induction"
    rem = out == "neither"
    if ctx.strict_keys.size and rem.any():
        out[rem & np.isin(key, ctx.strict_keys)] = "strict"
    rem = out == "neither"
    if ctx.same_content_keys.size and rem.any():
        out[rem & np.isin(key, ctx.same_content_keys)] = "same_content"
    return out


def shuffle_replicate(t: InteractionTable,
                      contexts: Dict[str, PromptNullContext],
                      rng: np.random.Generator) -> InteractionTable:
    """
    One null replicate of `t`.

    Every (prompt, layer, head) group's edges keep every column EXCEPT
    `target`, `source`, `offset` and `pair_type`; those four are recomputed
    for `len(group)` positions drawn uniformly at random, without
    replacement, from that prompt's causal pool. `checkpoint_step`, `model`,
    `layer`, `head`, and the entire force-derived payload travel with the
    row they started on, unchanged.

    Refuses rather than silently mis-shuffling when a prompt in `t` has no
    entry in `contexts`, or when a head's retained edge count exceeds its
    prompt's pool size (retention promising more edges than the prompt has
    causal pairs to give it).
    """
    c = t.columns
    n = len(t)
    prompts = c["prompt_key"]
    table_prompts = set(np.unique(prompts).tolist())
    missing = table_prompts - set(contexts)
    if missing:
        raise RelayNullRefused(
            f"no PromptNullContext for {sorted(missing)}; every prompt in "
            f"the table needs one or its edges cannot be reassigned")

    new_target = c["target"].copy()
    new_source = c["source"].copy()
    layers = c["layer"]
    heads = c["head"]

    for p in sorted(table_prompts):
        ctx = contexts[p]
        pmask = prompts == p
        idx = np.flatnonzero(pmask)
        # Group within this prompt's context, matching `_context_ids`'
        # own scoping (`motif_alphabet.py`) — the shuffle must respect the
        # same boundary the real join does, or a position collision across
        # prompts would read as a relay exactly as commit f7e95bc's finding
        # describes for the unshuffled table.
        lh = (layers[idx].astype(np.int64) * 4096 + heads[idx].astype(np.int64))
        for g in np.unique(lh):
            gm = idx[lh == g]
            k = len(gm)
            if k > ctx.pool_size:
                l_val, h_val = divmod(int(g), 4096)
                raise RelayNullRefused(
                    f"{p!r} layer {l_val} head {h_val}: {k} retained edges "
                    f"but the causal pool for {ctx.n_tokens} tokens holds "
                    f"only {ctx.pool_size} positions")
            draw = rng.choice(ctx.pool_size, size=k, replace=False)
            new_target[gm] = ctx.target_pool[draw]
            new_source[gm] = ctx.source_pool[draw]

    new_cols = dict(c)
    new_cols["target"] = new_target
    new_cols["source"] = new_source
    new_cols["offset"] = new_target - new_source
    pair_type = np.empty(n, dtype="<U12")
    for p in sorted(table_prompts):
        pmask = prompts == p
        pair_type[pmask] = _pair_types(new_target[pmask], new_source[pmask],
                                       contexts[p])
    new_cols["pair_type"] = pair_type
    return InteractionTable(columns=new_cols, extra=dict(t.extra),
                            retention=t.retention)


def null_envelope(t: InteractionTable,
                  contexts: Dict[str, PromptNullContext],
                  relay_owner: str,
                  n_replicates: int,
                  seed: int,
                  heads: Optional[Sequence[tuple]] = None) -> Dict[tuple, dict]:
    """
    Run `n_replicates` shuffles of `t` and score each with
    `formation_curve.per_head_relay_strength(_, relay_owner)` — the SAME
    collapse the real series uses, so a null head and a real head mean the
    same thing.

    Returns {(layer, head): {"mean": N1, "sd": N2, "n_replicates": K}}, over
    every head named in `heads` (default: the union of what any replicate
    or the real table itself produced). 0.0-fills a replicate that drew no
    relay for a head that some OTHER replicate did — the same "measured,
    and had none" reading `per_head_relay_strength`'s own sparsity already
    means, not a missing value.
    """
    if n_replicates < 2:
        raise RelayNullRefused(
            f"n_replicates={n_replicates}; at least 2 are needed for a "
            f"standard deviation to mean anything")
    from .formation_curve import per_head_relay_strength

    rng = np.random.default_rng(seed)
    per_rep: List[Dict[tuple, float]] = []
    for _ in range(n_replicates):
        shuf = shuffle_replicate(t, contexts, rng)
        per_rep.append(per_head_relay_strength(shuf, relay_owner))

    all_heads = set(heads) if heads is not None else set()
    for rep in per_rep:
        all_heads |= set(rep)

    out: Dict[tuple, dict] = {}
    for k in all_heads:
        vals = np.array([rep.get(k, 0.0) for rep in per_rep], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "n_replicates": len(vals),
        }
    return out
