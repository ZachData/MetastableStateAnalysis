"""
p5_single_mstate_analysis/anchors.py — from run directories to frozen token
sets.

Wires `core/run_discovery.py` (which run is which) to `token_sets.py` (what
the object of study is), and emits the result into `core/particles.py`.

The reader that had to be written (B11)
---------------------------------------
`p1_io.load_phase1_run` returns an `events` key, and it is the wrong events.

Phase 1 produces two different event schemas:

  trajectory.json -> cluster_tracking.events    the real one
      {"layer_from": int, "layer_to": int, "merges": [[prev_ids], curr_id], ...}

  events.json                                    a Phase 3 bridge file
      {"merge_layers": [2, 5], "energy_violations": {"1.0": [3, 4]}}

`load_phase1_run` reads the second and normalises it (`_load_events`) into
`[{"type": "merge", "layer_name": "5", "layer_from": "5"}]` — `layer_from` a
**string**, and no `merges` key at all. Any consumer asking "which merge did
this trajectory participate in" against that list gets `None` for every
trajectory, silently, forever.

That is blocker 1. `status-5.md` records FIX-B7 as having fixed it by routing
`merge_verdict` through `select_cluster`'s own computation instead of
re-deriving it in `_run_group_B` — which was the right plumbing change against
the wrong cause. `select_cluster._merge_event_for_trajectory` was already
reading `run["events"]`, so the merge criterion (weight 3.0 of 9.0, the second
largest term) had been contributing 0.0 on every model for the same reason the
semantic criterion had (B10). Two of six criteria dead, 5.0 of the 9.0 scale.

`load_cluster_tracking` below reads the real schema. It belongs upstream in
`p1_io` as a sibling of `load_phase1_run` — it is a Phase 1 artifact reader,
not a Phase 5 concern — and should be moved there once Phase 1 is next
touched. It lives here for now so this rebuild does not edit a frozen phase.

What the driver produces
------------------------
Per prompt, one `TokenSet` per anchor plus the overlap between them:

    anchor_final (step 143000)   what became of these particles?
    anchor_init  (step 0)        what happened to particles that
                                 started together?

`AnchorBundle.overlaps` carries the Jaccard. That number is a result on its
own — if the two anchors select near-disjoint particle sets, then "the cluster"
at the end of training is not a matured version of anything present at
initialization, and every backward-looking claim in Groups A-G has to be read
as being about an object that formed rather than one that persisted.

Torch-free. Reads JSON and npz only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.run_discovery import RunRef, sweep_for_prompt
from .token_sets import (
    TokenSet,
    SelectionRejected,
    CORE_MEMBERSHIP_FRACTION,
    REJECT_PROMPTS,
    anchor_overlap,
    select_token_set,
)

__all__ = [
    "AnchorSpec",
    "AnchorBundle",
    "DEFAULT_ANCHORS",
    "load_cluster_tracking",
    "load_run_for_selection",
    "build_anchor_token_sets",
    "token_set_particle_table",
    "bundle_report_lines",
]


# ---------------------------------------------------------------------------
# Phase 1 artifact readers (upstream to p1_io when Phase 1 is next touched)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_cluster_tracking(run_dir: Path) -> dict:
    """
    The REAL cluster-tracking record: trajectory.json -> cluster_tracking.

    Returns {"trajectories": [...], "events": [...], "summary": {...}} with
    every index cast to int and every chain entry a (layer, cluster_id) tuple.

    JSON round-trips tuples to lists, so `chain` arrives as [[layer, cid], ...]
    and `merges` as [[[prev...], curr], ...]. Both are normalised here rather
    than at each consumer — an unnormalised chain still unpacks correctly under
    `for layer, cid in chain`, which is exactly why this kind of drift survives
    unnoticed until something tries to hash or compare it.

    Never raises on a missing or malformed file: an absent trajectory.json
    becomes empty lists, and the caller's selection then fails with a
    SelectionRejected naming the run, which is more useful than a traceback
    from inside a JSON decoder.
    """
    tj = _read_json(Path(run_dir) / "trajectory.json") or {}
    ct = tj.get("cluster_tracking") or {}

    trajectories: List[dict] = []
    for t in ct.get("trajectories", []) or []:
        chain = tuple((int(l), int(c)) for l, c in t.get("chain", []))
        if not chain:
            continue
        trajectories.append({
            "id": int(t["id"]),
            "chain": chain,
            "start_layer": int(t.get("start_layer", chain[0][0])),
            "end_layer": int(t.get("end_layer", chain[-1][0])),
            # cluster_tracking defines lifespan as end - start + 1, not
            # len(chain). Identical for the contiguous chains track_clusters
            # produces, but the definition is preserved rather than recomputed.
            "lifespan": int(t.get("lifespan",
                                  chain[-1][0] - chain[0][0] + 1)),
        })

    events: List[dict] = []
    for ev in ct.get("events", []) or []:
        merges = [([int(p) for p in prev], int(curr))
                  for prev, curr in (ev.get("merges") or [])]
        events.append({
            "layer_from": int(ev["layer_from"]),
            "layer_to": int(ev["layer_to"]),
            "merges": merges,
            "n_births": int(ev.get("n_births", 0)),
            "n_deaths": int(ev.get("n_deaths", 0)),
        })

    return {
        "trajectories": trajectories,
        "events": events,
        "summary": ct.get("summary", {}) or {},
        "plateau_layers": [int(l) for l in (tj.get("plateau_layers") or [])],
    }


def _load_hdbscan_labels(run_dir: Path) -> List[np.ndarray]:
    """hdbscan_labels.json ({str(layer): [int, ...]}) -> list[ndarray],
    indexed by layer position with gaps filled by all-noise arrays.

    A list, never the string-keyed dict on disk — `labels[L]` with integer L
    is the access pattern everywhere downstream, and a dict silently returns
    KeyError on int keys that look identical when printed."""
    raw = _read_json(Path(run_dir) / "hdbscan_labels.json")
    if not raw:
        return []
    keyed = {int(k): np.asarray(v, dtype=np.int64) for k, v in raw.items()}
    if not keyed:
        return []
    n_layers = max(keyed) + 1
    n_tokens = len(next(iter(keyed.values())))
    return [keyed.get(i, np.full(n_tokens, -1, dtype=np.int64))
            for i in range(n_layers)]


def load_run_for_selection(run_dir: Path) -> dict:
    """
    Everything `select_token_set` needs from one run directory, and nothing
    else — no activations, no attentions. Selection is a labels-and-chains
    operation, so a whole sweep can be resolved without loading a single
    (n_layers, n_tokens, d_model) array.

    Returns {} when the directory has no usable clustering, which the driver
    reports as a skipped anchor rather than a crash.
    """
    run_dir = Path(run_dir)
    labels = _load_hdbscan_labels(run_dir)
    if not labels:
        return {}
    tracking = load_cluster_tracking(run_dir)
    geo = _read_json(run_dir / "geometry.json") or {}
    clustering = _read_json(run_dir / "clustering.json") or {}
    return {
        "run_dir": str(run_dir),
        "hdbscan_labels": labels,
        "n_tokens": int(len(labels[0])),
        "trajectories": tracking["trajectories"],
        "events": tracking["events"],
        "tokens": list(geo.get("tokens") or []),
        # `metrics` is the pair_agreement carrier for the ext_semantic
        # annotation; clustering.json's shape is {"layers": [ {...}, ... ]}.
        "metrics": clustering,
        "model": geo.get("model", ""),
        "prompt": geo.get("prompt", ""),
    }


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorSpec:
    name: str
    step: int
    rank: int = 0
    force_trajectory_id: Optional[int] = None


# 143000 is the final 410M checkpoint; 0 is the developmental origin, and is
# deliberately NOT the norm-matched random control (core/pythia_registry.py is
# explicit that collapsing the two is a mistake already made once).
DEFAULT_ANCHORS: Tuple[AnchorSpec, ...] = (
    AnchorSpec("anchor_final", step=143000),
    AnchorSpec("anchor_init", step=0),
)


@dataclass
class AnchorBundle:
    """Token sets for every (prompt, anchor), plus what failed and why."""
    token_sets: List[TokenSet] = field(default_factory=list)
    overlaps: List[dict] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)

    def by_prompt(self, prompt_key: str) -> Dict[str, TokenSet]:
        return {ts.name: ts for ts in self.token_sets
                if ts.prompt_key == prompt_key}

    def get(self, prompt_key: str, name: str) -> Optional[TokenSet]:
        return self.by_prompt(prompt_key).get(name)


def build_anchor_token_sets(
    refs: Sequence[RunRef],
    anchors: Sequence[AnchorSpec] = DEFAULT_ANCHORS,
    prompt_keys: Optional[Sequence[str]] = None,
    control_seed: int = 0,
    min_fraction: float = CORE_MEMBERSHIP_FRACTION,
    loader: Callable[[Path], dict] = load_run_for_selection,
) -> AnchorBundle:
    """
    Select once per (prompt, anchor) and freeze the positions.

    `loader` is injectable so the driver is testable without a run directory;
    the default reads from disk.

    Nothing here raises. A prompt whose anchor has no passing trajectory, or
    whose run directory is missing, lands in `bundle.skipped` with the reason
    attached. Partial coverage is the normal case on a real sweep — late
    Pythia checkpoints have mean cluster lifespan 4.5 — and a driver that
    aborts on the first miss would make that fact invisible.

    The control seed is derived per (prompt, anchor) from `control_seed`, so
    the random control is reproducible AND different between anchors. A single
    shared seed would draw the same positions for both anchors, which would
    make the two controls correlated for no reason.
    """
    bundle = AnchorBundle()
    if prompt_keys is None:
        prompt_keys = sorted({r.prompt_key for r in refs})

    for pk in prompt_keys:
        if pk in REJECT_PROMPTS:
            bundle.skipped.append({
                "prompt_key": pk, "anchor": "*",
                "reason": "collapse control, not a metastability prompt "
                          "(status-2 V4)",
            })
            continue

        sweep = sweep_for_prompt(refs, pk)
        available = [r.step for r in sweep]

        for a in anchors:
            ref = next((r for r in sweep if r.step == a.step), None)
            if ref is None:
                bundle.skipped.append({
                    "prompt_key": pk, "anchor": a.name,
                    "reason": f"no run at step {a.step}; available: {available}",
                })
                continue

            run = loader(Path(ref.run_dir))
            if not run:
                bundle.skipped.append({
                    "prompt_key": pk, "anchor": a.name,
                    "reason": f"no usable clustering in {ref.run_dir}",
                })
                continue

            seed = control_seed + 1000 * a.step + hash(pk) % 997
            try:
                ts = select_token_set(
                    name=a.name,
                    prompt_key=pk,
                    anchor_model=ref.model,
                    anchor_step=ref.step,
                    anchor_run_dir=str(ref.run_dir),
                    trajectories=run["trajectories"],
                    events=run["events"],
                    hdb_labels=run["hdbscan_labels"],
                    n_tokens=run["n_tokens"],
                    metrics=run.get("metrics"),
                    rank=a.rank,
                    force_trajectory_id=a.force_trajectory_id,
                    control_seed=int(seed),
                    min_fraction=min_fraction,
                )
            except SelectionRejected as exc:
                bundle.skipped.append({
                    "prompt_key": pk, "anchor": a.name,
                    "reason": str(exc),
                })
                continue
            bundle.token_sets.append(ts)

    # Pairwise overlaps within each prompt, in the order anchors were given.
    for pk in prompt_keys:
        got = bundle.by_prompt(pk)
        names = [a.name for a in anchors if a.name in got]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                bundle.overlaps.append(
                    anchor_overlap(got[names[i]], got[names[j]])
                )
    return bundle


# ---------------------------------------------------------------------------
# Emission into the particle table
# ---------------------------------------------------------------------------

def token_set_particle_table(
    token_set: TokenSet,
    run: dict,
    ref: RunRef,
    ParticleTable=None,
):
    """
    One `ParticleTable` for a run, with the token set's roles attached as
    extra columns.

    Every token gets a row at every layer — not just the set's members. The
    complement is the object 5c is about (the unclustered population, "not a
    failure mode but a distinct phase"), and dropping it here would make that
    population unrecoverable from the table.

    Extra columns:
      token_set_role : "primary" | "sibling" | "control" | "none"
      in_token_set   : 1 for primary, else 0

    `ParticleTable` is injected rather than imported at module scope so this
    module stays importable when core.particles is not on the path.
    """
    if ParticleTable is None:
        from core.particles import ParticleTable  # noqa: F811

    primary = set(token_set.positions)
    sibling = set(token_set.sibling_positions)
    control = set(token_set.control_positions)

    n_tokens = run["n_tokens"]
    roles = []
    for p in range(n_tokens):
        if p in primary:
            roles.append("primary")
        elif p in sibling:
            roles.append("sibling")
        elif p in control:
            roles.append("control")
        else:
            roles.append("none")
    roles_arr = np.array(roles)
    in_set = np.array([1 if r == "primary" else 0 for r in roles],
                      dtype=np.int64)

    tokens = run.get("tokens") or None
    if tokens is not None and len(tokens) != n_tokens:
        tokens = None

    tables = []
    for layer, labels in enumerate(run["hdbscan_labels"]):
        tables.append(ParticleTable.from_layer(
            model=ref.model,
            prompt_key=token_set.prompt_key,
            layer=layer,
            cluster_labels=labels,
            checkpoint_step=ref.step,
            token_str=tokens,
            extra={"token_set_role": roles_arr, "in_token_set": in_set},
        ))
    return ParticleTable.concat(tables)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def bundle_report_lines(bundle: AnchorBundle) -> List[str]:
    """Coverage, sizes, overlap. Skips are printed with their reasons — an
    anchor that produced nothing is a finding about that checkpoint, and a
    report that only lists successes hides it."""
    lines: List[str] = []
    if not bundle.token_sets and not bundle.skipped:
        return ["(no anchors attempted)"]

    lines.append(f"token sets: {len(bundle.token_sets)}")
    for ts in sorted(bundle.token_sets,
                     key=lambda t: (t.prompt_key, t.name)):
        lines.append(
            f"  {ts.prompt_key:22s} {ts.name:14s} "
            f"n={ts.size:3d}  union={len(ts.union_positions):3d}  "
            f"churn={ts.churn:.2f}  score={ts.score:.3f}  "
            f"traj={ts.source_trajectory_id}"
        )
        for note in ts.notes:
            lines.append(f"      [note] {note}")

    if bundle.overlaps:
        lines.append("anchor overlap:")
        for o in bundle.overlaps:
            lines.append(
                f"  {o['prompt_key']:22s} {o['a']} vs {o['b']}: "
                f"J={o['jaccard']:.3f}  "
                f"|A|={o['n_a']} |B|={o['n_b']} |A&B|={o['n_intersection']}"
            )

    if bundle.skipped:
        lines.append(f"skipped: {len(bundle.skipped)}")
        for s in bundle.skipped:
            lines.append(f"  {s['prompt_key']:22s} {s['anchor']:14s} "
                         f"{s['reason']}")
    return lines
