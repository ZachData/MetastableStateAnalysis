"""
p5_single_mstate_analysis/token_sets.py — the object of study, frozen.

Replaces `select_cluster.py`.

Why this replaces trajectory selection
--------------------------------------
An HDBSCAN trajectory is Jaccard-chained *within one run*. Cluster ids and
trajectory ids carry no meaning across checkpoints, so "select the best
trajectory" run at 27 checkpoints selects 27 different objects and every
cross-checkpoint statement is about a different thing. `select_cluster.py` is
correct for a single-checkpoint study and structurally unable to support a
sweep.

The one index that survives the whole sweep is **token position**. The battery
is fixed and `core/pythia_registry.py` records that the NeoX tokenizer is byte
identical from step 0 to step 143,000, so position *i* is the same particle at
every checkpoint. So: use a trajectory as a *discovery device* at one anchor
checkpoint, extract the token positions, freeze them, and discard the
trajectory identity. Every downstream group then measures the same particles
at every step, and "does this cluster still exist at step 512" becomes a
measurement rather than a second selection.

Two anchors, both produced
--------------------------
    anchor_final (step 143000)  what became of these particles?  (backward)
    anchor_init  (step 0)       what happened to particles that
                                started together?                (forward)

Selecting only at the final checkpoint biases toward mature structure;
selecting only at init biases toward whatever the initialization happens to
group. `anchor_overlap` reports the Jaccard between the two sets, which is
itself a result and costs nothing.

Core membership, not union membership
-------------------------------------
A trajectory's membership churns along its chain. `positions` is the set of
tokens in the cluster at >= CORE_MEMBERSHIP_FRACTION of the chain's layers —
the particles that actually stayed together, not everything that ever touched
it. `union_positions` is recorded alongside for the size comparison, but the
core set is the object. Strict intersection (fraction 1.0) can be empty on a
churning trajectory, which is why the default is below 1.0 and why an empty
core is a rejection with a stated reason rather than a crash.

Four scoring criteria, not six
------------------------------
Dropped from `constants.SCORE_WEIGHTS`:

  `semantic` (weight 2.0) — never fired. `clustering.pair_hdbscan_agreement`
      emits `tag` as an alias for `cross_method_tag`, whose value set is
      {"same_cluster", "diff_cluster", "noise"}; `select_cluster` tested it
      against the string "semantic", which is not in that set, in both the
      main branch and the fallback. `s["semantic"]` was 0.0 for every
      trajectory on every model. Repairing it means switching to
      `ext_semantic_tag`, whose reference frame is the model's own layer-0
      embedding Gram against a fixed cutoff — status-1 defect D6: the frame
      *trains*, so a repaired criterion would drift across checkpoints for
      reasons unrelated to any cluster. Recorded as an annotation instead,
      under a field name that carries the caveat.

  `preferred_prompt` (weight 1.0) — selection is now per-prompt, so there is
      no cross-prompt competition for this term to break.

Remaining: lifespan, merge, size, sibling. Weights renormalized so a perfect
score is 1.0 rather than an arbitrary sum — the old 9.0 maximum invited
exactly the confusion that design-5.md's "9.000 for 4/6" now needs checking
against (that figure is unreachable if a 2.0 term is structurally zero).

Pure numpy. No torch, no disk I/O — the loaders live in p5_io / core.io and
hand this module plain trajectories/events/labels, so every function here is
testable without a model or a run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TokenSet",
    "SelectionRejected",
    "MIN_LIFESPAN",
    "MIN_CORE_SIZE",
    "CORE_MEMBERSHIP_FRACTION",
    "SCORE_WEIGHTS",
    "REJECT_PROMPTS",
    "core_membership",
    "pick_sibling",
    "score_trajectory",
    "rank_trajectories",
    "select_token_set",
    "random_control_positions",
    "anchor_overlap",
    "save_token_sets",
    "load_token_sets",
]


# ---------------------------------------------------------------------------
# Constants (Pythia-recalibrated; see PHASE5_PYTHIA.md §6)
# ---------------------------------------------------------------------------

# Hard gate on the ANCHOR trajectory only. status-1 records mean cluster
# lifespan falling 7.0 -> 4.5 across training, so a gate applied at every
# checkpoint would admit a survivor set whose bias grows monotonically with
# step. Applied once, at one anchor, that objection does not arise — but the
# gate is deliberately lower than select_cluster's 6 because at anchor_init
# (step 0) the lifespan distribution is a different object entirely.
MIN_LIFESPAN = 4

# Minimum size of the CORE set. Below this the per-particle statistics
# downstream (silhouette, effective-rank contribution) are not estimable.
MIN_CORE_SIZE = 4

# A token joins the core set if it is in the cluster at >= this fraction of
# the trajectory's alive layers. 1.0 = strict intersection, which is often
# empty; 0.75 tolerates brief departures without admitting drive-bys.
CORE_MEMBERSHIP_FRACTION = 0.75

# Normalisation caps. Pythia-410M has 25 analyzed layers (embeddings + 24
# blocks), so select_cluster's LIFESPAN_FULL_SCORE = 18 (calibrated to
# ALBERT-xlarge's 48-iteration depth) would leave the term effectively
# unsaturable. Half the stack is the meaningful "long-lived" threshold.
LIFESPAN_FULL_SCORE = 12
SIZE_FULL_SCORE = 10

# Weights sum to 1.0 by construction — a perfect score is 1.0, not a sum
# whose maximum has to be recomputed whenever a criterion is added or dies.
SCORE_WEIGHTS: Dict[str, float] = {
    "lifespan": 0.35,
    "merge":    0.30,
    "size":     0.20,
    "sibling":  0.15,
}

# status-2 V4: `repeated_tokens` is ". " x ~264, one distinct token id, so the
# particles are degenerate at embedding and its 27 zero-violation runs are the
# eff_rank guard firing, not monotonicity. It is a collapse control, not a
# metastability prompt.
REJECT_PROMPTS: Tuple[str, ...] = ("repeated_tokens",)

MIN_SIBLING_LIFESPAN = 4


class SelectionRejected(RuntimeError):
    """No trajectory in this run passes the gates. Carries the per-gate tally
    so a rejection reads as a diagnosis rather than an empty result."""


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

def core_membership(
    trajectory: dict,
    hdb_labels: Sequence[np.ndarray],
    min_fraction: float = CORE_MEMBERSHIP_FRACTION,
) -> dict:
    """
    Token positions belonging to a trajectory, resolved to a frozen set.

    Returns
    -------
    dict with:
      core        : sorted tuple[int] — in cluster at >= min_fraction of
                    alive layers. THE object.
      union       : sorted tuple[int] — in cluster at any alive layer.
      counts      : {position: n_layers_present}
      n_layers    : number of chain layers actually evaluated
      per_layer_size : list[int], cluster size at each evaluated layer
      churn       : 1 - len(core)/len(union), or 0.0 when union is empty.
                    A high churn with a healthy core means tokens drifted in
                    and out around a stable centre; a high churn with a small
                    core means there was no stable centre.
    """
    counts: Dict[int, int] = {}
    per_layer_size: List[int] = []
    n_layers = 0

    for layer, cid in trajectory["chain"]:
        layer = int(layer)
        if layer >= len(hdb_labels):
            continue
        labels = np.asarray(hdb_labels[layer])
        members = np.where(labels == int(cid))[0]
        n_layers += 1
        per_layer_size.append(int(members.size))
        for p in members.tolist():
            counts[int(p)] = counts.get(int(p), 0) + 1

    if n_layers == 0:
        return {"core": (), "union": (), "counts": {}, "n_layers": 0,
                "per_layer_size": [], "churn": 0.0}

    threshold = min_fraction * n_layers
    core = tuple(sorted(p for p, c in counts.items() if c >= threshold))
    union = tuple(sorted(counts))
    churn = 1.0 - (len(core) / len(union)) if union else 0.0

    return {
        "core": core,
        "union": union,
        "counts": counts,
        "n_layers": n_layers,
        "per_layer_size": per_layer_size,
        "churn": round(float(churn), 4),
    }


# ---------------------------------------------------------------------------
# Merge / sibling
# ---------------------------------------------------------------------------

def _assert_tracking_events(events: Sequence[dict]) -> None:
    """
    Refuse the wrong event schema loudly.

    Phase 1 emits two: cluster_tracking's
    ({"layer_from": int, "layer_to": int, "merges": [...]}), and the Phase 3
    bridge file events.json, which `p1_io._load_events` normalises to
    ({"type": "merge", "layer_name": str, "layer_from": str}) — no `merges`
    key at all.

    Handed the second, a tolerant implementation returns None for every
    trajectory and the merge criterion silently contributes zero. That is
    blocker 1, and it survived a whole six-model study precisely because
    nothing objected. So this objects: an event list without `merges` is a
    wiring error, not an absence of merges, and the two must not look alike.

    An empty list is fine — a run with no merges is a real run.
    """
    for ev in events:
        if "merges" in ev and "layer_from" in ev:
            continue
        raise ValueError(
            "merge_event_for was given the wrong event schema: "
            f"keys {sorted(ev)}. This looks like p1_io.load_phase1_run's "
            "`events` (the Phase 3 bridge file events.json), which carries "
            "no `merges` and a string `layer_from`. Merge detection needs "
            "trajectory.json -> cluster_tracking.events; read it with "
            "p5_single_mstate_analysis.anchors.load_cluster_tracking."
        )


def merge_event_for(trajectory: dict, events: Sequence[dict]) -> Optional[dict]:
    """
    The merge event this trajectory participates in, or None.

    Unpacks the raw per-transition event schema
    ({"layer_from", "layer_to", "merges": [(prev_ids, curr_id), ...]}) —
    the same unpacking FIX-B7 had to retrofit into `_run_group_B` after the
    original Group B filtered on a top-level "prev_ids" key that schema never
    had. Done once, here, so no consumer re-derives it.

    Deterministic: earliest layer_from wins, then lowest curr_id.
    """
    _assert_tracking_events(events)
    if not trajectory.get("chain"):
        return None
    chain_at = {int(l): int(c) for l, c in trajectory["chain"]}
    hits = []
    for ev in events:
        lf = int(ev["layer_from"])
        lt = int(ev["layer_to"])
        own = chain_at.get(lf)
        if own is None:
            continue
        for prev_ids, curr_id in ev.get("merges", []):
            prev_ids = [int(x) for x in prev_ids]
            if own in prev_ids:
                hits.append({
                    "layer_from": lf,
                    "layer_to": lt,
                    "prev_ids": prev_ids,
                    "curr_id": int(curr_id),
                    "own_cluster_id": own,
                    "role": "participant",
                })
    if not hits:
        return None
    hits.sort(key=lambda h: (h["layer_from"], h["curr_id"]))
    return hits[0]


def pick_sibling(
    trajectory: dict,
    all_trajectories: Sequence[dict],
    events: Sequence[dict],
) -> Optional[dict]:
    """
    The contrast object: (a) the trajectory that fuses with this one at its
    merge, else (b) the contemporary trajectory sharing the most alive layers.

    Deterministic throughout — `select_cluster._pick_sibling`'s case (b) had a
    tie-break that dereferenced `best_id` while it could still be None and
    whose outcome depended on iteration order. Here ties break on
    (-overlap, -lifespan, id), a total order.
    """
    tid = int(trajectory["id"])
    by_id = {int(t["id"]): t for t in all_trajectories}

    merge = merge_event_for(trajectory, events)
    if merge is not None:
        lf = merge["layer_from"]
        prev_ids = set(merge["prev_ids"])
        partners = []
        for other in all_trajectories:
            oid = int(other["id"])
            if oid == tid:
                continue
            for layer, cid in other["chain"]:
                if int(layer) == lf and int(cid) in prev_ids:
                    partners.append(oid)
                    break
        if partners:
            return by_id[min(partners)]

    t_layers = {int(l) for l, _ in trajectory["chain"]}
    scored = []
    for other in all_trajectories:
        oid = int(other["id"])
        if oid == tid:
            continue
        o_layers = {int(l) for l, _ in other["chain"]}
        overlap = len(t_layers & o_layers)
        if overlap >= max(1, MIN_LIFESPAN // 2):
            scored.append((-overlap, -int(other["lifespan"]), oid))
    if not scored:
        return None
    return by_id[min(scored)[2]]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trajectory(
    trajectory: dict,
    all_trajectories: Sequence[dict],
    events: Sequence[dict],
    hdb_labels: Sequence[np.ndarray],
    min_fraction: float = CORE_MEMBERSHIP_FRACTION,
) -> dict:
    """
    Score one trajectory. Always returns a dict; `passed` is False with a
    `reject_reason` when a gate fails, so the caller can tally rejections
    instead of watching candidates silently disappear.
    """
    out: dict = {
        "id": int(trajectory["id"]),
        "lifespan": int(trajectory["lifespan"]),
        "start_layer": int(trajectory["start_layer"]),
        "end_layer": int(trajectory["end_layer"]),
        "passed": False,
        "reject_reason": None,
        "sub_scores": {},
        "total_score": 0.0,
    }

    if out["lifespan"] < MIN_LIFESPAN:
        out["reject_reason"] = f"lifespan {out['lifespan']} < {MIN_LIFESPAN}"
        return out

    mem = core_membership(trajectory, hdb_labels, min_fraction)
    out["n_core"] = len(mem["core"])
    out["n_union"] = len(mem["union"])
    out["churn"] = mem["churn"]
    out["mean_size"] = (round(float(np.mean(mem["per_layer_size"])), 2)
                        if mem["per_layer_size"] else 0.0)

    if mem["n_layers"] == 0:
        out["reject_reason"] = "no chain layer within label array"
        return out
    if len(mem["core"]) < MIN_CORE_SIZE:
        out["reject_reason"] = (
            f"core size {len(mem['core'])} < {MIN_CORE_SIZE} "
            f"(union {len(mem['union'])}, churn {mem['churn']})"
        )
        return out

    merge = merge_event_for(trajectory, events)
    sibling = pick_sibling(trajectory, all_trajectories, events)

    s = {
        "lifespan": min(1.0, out["lifespan"] / LIFESPAN_FULL_SCORE),
        "merge": 1.0 if merge is not None else 0.0,
        "size": min(1.0, len(mem["core"]) / SIZE_FULL_SCORE),
        "sibling": 1.0 if (sibling is not None
                           and int(sibling["lifespan"]) >= MIN_SIBLING_LIFESPAN)
                   else 0.0,
    }

    out.update({
        "passed": True,
        "sub_scores": {k: round(v, 4) for k, v in s.items()},
        "total_score": round(sum(SCORE_WEIGHTS[k] * v for k, v in s.items()), 4),
        "merge_event": merge,
        "sibling_id": int(sibling["id"]) if sibling is not None else None,
        "_membership": mem,
    })
    return out


def rank_trajectories(
    trajectories: Sequence[dict],
    events: Sequence[dict],
    hdb_labels: Sequence[np.ndarray],
    min_fraction: float = CORE_MEMBERSHIP_FRACTION,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    (passing candidates sorted best-first, rejection tally).

    Sort key is (-total_score, -lifespan, id) — a total order, so the ranking
    is reproducible rather than dependent on input order.
    """
    scored = [score_trajectory(t, trajectories, events, hdb_labels, min_fraction)
              for t in trajectories]
    passing = [c for c in scored if c["passed"]]
    passing.sort(key=lambda c: (-c["total_score"], -c["lifespan"], c["id"]))

    tally: Dict[str, int] = {}
    for c in scored:
        if c["passed"]:
            continue
        key = (c["reject_reason"] or "unknown").split("(")[0].split(str(0))[0]
        key = key.strip()
        # Bucket by gate, not by the specific numbers in the message.
        for gate in ("lifespan", "core size", "no chain layer"):
            if key.startswith(gate):
                key = gate
                break
        tally[key] = tally.get(key, 0) + 1
    return passing, tally


# ---------------------------------------------------------------------------
# The frozen object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenSet:
    """A frozen set of token positions, plus everything needed to justify it.

    `positions` is the object. Everything else is provenance: which anchor
    produced it, which trajectory found it, and what the discovery device
    looked like at the time. Nothing downstream should re-derive positions
    from `source_trajectory_id` — that id is meaningless at any other
    checkpoint, which is the entire reason this class exists.
    """
    name: str                       # "anchor_final" | "anchor_init" | ...
    prompt_key: str
    anchor_model: str
    anchor_step: Optional[int]
    anchor_run_dir: str
    positions: Tuple[int, ...]
    sibling_positions: Tuple[int, ...] = ()
    control_positions: Tuple[int, ...] = ()
    control_seed: Optional[int] = None
    source_trajectory_id: Optional[int] = None
    source_layers: Tuple[int, ...] = ()
    union_positions: Tuple[int, ...] = ()
    churn: float = 0.0
    score: float = 0.0
    sub_scores: Dict[str, float] = field(default_factory=dict)
    merge_event: Optional[dict] = None
    n_tokens_prompt: int = 0
    annotations: Dict[str, object] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def __post_init__(self):
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        if tuple(sorted(self.positions)) != tuple(self.positions):
            raise ValueError("positions must be sorted")
        overlap = set(self.positions) & set(self.control_positions)
        if overlap:
            raise ValueError(
                f"control overlaps primary at positions {sorted(overlap)}"
            )

    @property
    def size(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("positions", "sibling_positions", "control_positions",
                  "source_layers", "union_positions", "notes"):
            d[k] = list(d[k])
        return d


def random_control_positions(
    n: int,
    n_tokens: int,
    exclude: Sequence[int],
    seed: int,
) -> Tuple[int, ...]:
    """
    Size-matched random control drawn from the same prompt, disjoint from
    `exclude` (primary + sibling). Seeded and sorted, so the control is a
    fixed object across every checkpoint the sweep touches — a control
    redrawn per checkpoint would add variance exactly where the comparison
    needs none.

    Returns fewer than `n` positions, silently, only if the prompt cannot
    supply them; the caller checks `len()` and records a note.
    """
    pool = np.setdiff1d(np.arange(n_tokens), np.asarray(sorted(set(exclude)),
                                                        dtype=np.int64))
    if pool.size == 0:
        return ()
    k = int(min(n, pool.size))
    rng = np.random.default_rng(seed)
    picked = rng.choice(pool, size=k, replace=False)
    return tuple(sorted(int(x) for x in picked))


def _ext_semantic_annotation(
    positions: Sequence[int],
    metrics: dict,
    source_layers: Sequence[int],
) -> dict:
    """
    Fraction of within-set mutual-NN pairs tagged `ext_semantic`, averaged
    over the trajectory's layers.

    Deliberately an ANNOTATION, never a score term. Two reasons, both real:

    1. `select_cluster` scored this by testing `pair["tag"] == "semantic"`.
       `tag` is a backward-compat alias for `cross_method_tag`, whose value
       set is {"same_cluster", "diff_cluster", "noise"} — so that test was
       never true and the 2.0-weight criterion contributed nothing on any
       model. The correct field is `ext_semantic_tag`.
    2. `ext_semantic` is defined by the cosine Gram of the model's OWN layer-0
       embeddings against a fixed 0.5 cutoff (clustering.py:318). The
       embedding matrix trains, so the reference frame moves across
       checkpoints — status-1 defect D6. A repaired criterion would drift for
       reasons unrelated to any cluster, which is worse than one that does
       nothing, because it would look like it worked.

    The key name carries the caveat so a consumer cannot pick it up without
    reading it.
    """
    pos = set(int(p) for p in positions)
    layers = metrics.get("layers", []) if metrics else []
    fracs: List[float] = []
    n_pairs_total = 0
    for li in source_layers:
        if li >= len(layers):
            continue
        pa = layers[li].get("pair_agreement", {}) or {}
        pairs = [p for p in pa.get("mutual_pairs", [])
                 if int(p.get("i", -1)) in pos and int(p.get("j", -1)) in pos]
        if not pairs:
            continue
        n_pairs_total += len(pairs)
        n_sem = sum(1 for p in pairs
                    if p.get("ext_semantic_tag") == "ext_semantic")
        fracs.append(n_sem / len(pairs))
    return {
        "ext_semantic_frac__unfrozen_reference": (
            round(float(np.mean(fracs)), 4) if fracs else None
        ),
        "ext_semantic_n_pairs": n_pairs_total,
        "ext_semantic_caveat": (
            "reference frame is the model's own layer-0 embedding Gram "
            "(status-1 D6) and therefore trains; not comparable across "
            "checkpoints. Annotation only — not used in selection."
        ),
    }


def select_token_set(
    name: str,
    prompt_key: str,
    anchor_model: str,
    anchor_step: Optional[int],
    anchor_run_dir: str,
    trajectories: Sequence[dict],
    events: Sequence[dict],
    hdb_labels: Sequence[np.ndarray],
    n_tokens: int,
    metrics: Optional[dict] = None,
    rank: int = 0,
    force_trajectory_id: Optional[int] = None,
    control_seed: int = 0,
    min_fraction: float = CORE_MEMBERSHIP_FRACTION,
) -> TokenSet:
    """
    Discover a coherent set of particles at one anchor checkpoint and freeze
    their positions.

    Raises SelectionRejected — with the per-gate tally — when nothing passes.
    An empty candidate pool is a fact about the checkpoint (late Pythia
    checkpoints have mean cluster lifespan 4.5), not an internal error, and
    the tally is what says which gate did it.
    """
    if prompt_key in REJECT_PROMPTS:
        raise SelectionRejected(
            f"{prompt_key!r} is a collapse control, not a metastability prompt "
            "(status-2 V4: one distinct token id, degenerate at embedding)."
        )

    passing, tally = rank_trajectories(trajectories, events, hdb_labels,
                                       min_fraction)

    if force_trajectory_id is not None:
        chosen = next((c for c in passing
                       if c["id"] == int(force_trajectory_id)), None)
        if chosen is None:
            raise SelectionRejected(
                f"forced trajectory id={force_trajectory_id} is not in the "
                f"passing pool (ids: {[c['id'] for c in passing]}); "
                f"rejections: {tally}"
            )
    else:
        if not passing:
            raise SelectionRejected(
                f"no trajectory passes the gates for prompt={prompt_key!r} at "
                f"{anchor_model} (step {anchor_step}). Rejections by gate: "
                f"{tally}. n_trajectories={len(trajectories)}"
            )
        if rank >= len(passing):
            raise SelectionRejected(
                f"rank {rank} requested but only {len(passing)} trajectories "
                f"pass for prompt={prompt_key!r}; rejections: {tally}"
            )
        chosen = passing[rank]

    mem = chosen["_membership"]
    positions = mem["core"]
    source_layers = tuple(int(l) for l, _ in
                          _traj_by_id(trajectories, chosen["id"])["chain"])

    sibling_positions: Tuple[int, ...] = ()
    notes: List[str] = []
    if chosen.get("sibling_id") is not None:
        sib = _traj_by_id(trajectories, chosen["sibling_id"])
        sib_mem = core_membership(sib, hdb_labels, min_fraction)
        sibling_positions = tuple(p for p in sib_mem["core"]
                                  if p not in set(positions))
        if len(sibling_positions) < MIN_CORE_SIZE:
            notes.append(
                f"sibling core is {len(sibling_positions)} positions after "
                f"removing overlap with primary — below MIN_CORE_SIZE "
                f"({MIN_CORE_SIZE}); Group G's sibling tier is weak here."
            )
    else:
        notes.append("no sibling available; Group G runs primary-vs-control only")

    control = random_control_positions(
        n=len(positions),
        n_tokens=n_tokens,
        exclude=list(positions) + list(sibling_positions),
        seed=control_seed,
    )
    if len(control) < len(positions):
        notes.append(
            f"control is {len(control)} of a requested {len(positions)} — "
            "prompt too short to supply a disjoint size-matched control"
        )

    annotations = _ext_semantic_annotation(positions, metrics or {},
                                           source_layers)
    annotations["rejections_by_gate"] = tally
    annotations["n_candidates_passing"] = len(passing)
    annotations["mean_cluster_size_along_chain"] = chosen["mean_size"]

    return TokenSet(
        name=name,
        prompt_key=prompt_key,
        anchor_model=anchor_model,
        anchor_step=anchor_step,
        anchor_run_dir=str(anchor_run_dir),
        positions=positions,
        sibling_positions=sibling_positions,
        control_positions=control,
        control_seed=control_seed,
        source_trajectory_id=chosen["id"],
        source_layers=source_layers,
        union_positions=mem["union"],
        churn=mem["churn"],
        score=chosen["total_score"],
        sub_scores=chosen["sub_scores"],
        merge_event=chosen.get("merge_event"),
        n_tokens_prompt=int(n_tokens),
        annotations=annotations,
        notes=tuple(notes),
    )


def _traj_by_id(trajectories: Sequence[dict], tid: int) -> dict:
    for t in trajectories:
        if int(t["id"]) == int(tid):
            return t
    raise KeyError(f"trajectory id={tid} not present")


# ---------------------------------------------------------------------------
# Anchor comparison
# ---------------------------------------------------------------------------

def anchor_overlap(a: TokenSet, b: TokenSet) -> dict:
    """
    Jaccard and directional containment between two anchors' token sets.

    Only meaningful within one prompt: positions index into that prompt's
    token sequence and nothing else. Refuses across prompts rather than
    returning a number that would look fine.
    """
    if a.prompt_key != b.prompt_key:
        raise ValueError(
            f"token sets are for different prompts ({a.prompt_key!r} vs "
            f"{b.prompt_key!r}); positions are not comparable across prompts"
        )
    sa, sb = set(a.positions), set(b.positions)
    inter = sa & sb
    union = sa | sb
    return {
        "prompt_key": a.prompt_key,
        "a": a.name, "b": b.name,
        "n_a": len(sa), "n_b": len(sb),
        "n_intersection": len(inter),
        "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "frac_of_a_in_b": round(len(inter) / len(sa), 4) if sa else 0.0,
        "frac_of_b_in_a": round(len(inter) / len(sb), 4) if sb else 0.0,
        "only_a": tuple(sorted(sa - sb)),
        "only_b": tuple(sorted(sb - sa)),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_token_sets(token_sets: Sequence[TokenSet], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "p5.token_sets.v1",
        "token_sets": [ts.to_dict() for ts in token_sets],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def load_token_sets(path: Path) -> List[TokenSet]:
    with open(Path(path)) as f:
        payload = json.load(f)
    if payload.get("schema") != "p5.token_sets.v1":
        raise ValueError(f"unexpected schema {payload.get('schema')!r}")
    out = []
    for d in payload["token_sets"]:
        d = dict(d)
        for k in ("positions", "sibling_positions", "control_positions",
                  "source_layers", "union_positions", "notes"):
            d[k] = tuple(d.get(k) or ())
        out.append(TokenSet(**d))
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
