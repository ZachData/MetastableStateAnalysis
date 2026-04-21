"""
select_cluster.py — Rank HDBSCAN trajectories, pick primary + sibling.

Plan reference: §Cluster selection. Six criteria, scored 0-1 each, combined
linearly with weights from constants.SCORE_WEIGHTS. A trajectory is dropped
before scoring if it violates a hard gate (MIN_LIFESPAN, MIN_SIZE,
REJECT_PROMPTS).

Entry point
-----------
select_primary_and_sibling(phase1_runs, ...) -> dict
    Returns one trajectory per prompt ranked, the top pick overall, and its
    sibling. No manual override here — the run script is responsible for
    presenting the ranked list and optionally taking a --force-trajectory
    argument.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from . import constants as C


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

def _traj_by_id(trajs: list, tid: int) -> Optional[dict]:
    for t in trajs:
        if t["id"] == tid:
            return t
    return None


def _chain_dict(traj: dict) -> dict:
    """{layer: cluster_id} from a trajectory chain list."""
    return {int(l): int(c) for l, c in traj["chain"]}


def _cluster_sizes_along_chain(traj: dict, hdb_labels: list) -> list:
    """Size of the HDBSCAN cluster at each layer of the trajectory chain."""
    sizes = []
    for layer, cid in traj["chain"]:
        if layer < len(hdb_labels):
            sizes.append(int(np.sum(hdb_labels[layer] == cid)))
    return sizes


def _pair_agreement_at_layer(metrics: dict, layer: int) -> dict:
    """Extract P1-4 mutual-NN pairs for one layer."""
    layers = metrics.get("layers", [])
    if layer >= len(layers):
        return {"mutual_pairs": []}
    return layers[layer].get("pair_agreement", {"mutual_pairs": []})


def _merge_event_for_trajectory(traj: dict, events: list) -> Optional[dict]:
    """
    Return the merge event in which this trajectory is absorbed (or that
    absorbs others into it), or None.

    The cluster_tracking machinery only tracks the primary through a merge,
    so we check the last chain entry's layer + cluster against the events'
    merges field.
    """
    if not traj["chain"]:
        return None

    # Trajectory could be ABSORBED (its id disappears into another trajectory's
    # primary) or a MERGE HOST (other clusters merge into its tip).
    chain = traj["chain"]

    # Case 1: trajectory dies, absorbed into another. Look at event at
    # (end_layer -> end_layer + 1): is this traj's last cluster in any
    # event["merges"] prev_ids (as non-primary)?
    for ev in events:
        lf = ev["layer_from"]
        lt = ev["layer_to"]
        for prev_ids, curr_id in ev.get("merges", []):
            # prev_ids are cluster ids at lf; curr_id at lt.
            # Find this trajectory's cluster at lf.
            for layer, cid in chain:
                if layer == lf and cid in prev_ids:
                    return {
                        "layer_from": lf, "layer_to": lt,
                        "prev_ids": list(prev_ids), "curr_id": int(curr_id),
                        "role": "participant",
                    }
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_trajectory(
    traj: dict,
    metrics: dict,
    hdb_labels: list,
    events: list,
    prompt_key: str,
    all_trajectories: list,
) -> Optional[dict]:
    """
    Compute score components and total for one trajectory.

    Returns None if the trajectory fails a hard gate.
    """
    # --- Hard gates ---
    if prompt_key in C.REJECT_PROMPTS:
        return None

    lifespan = traj["lifespan"]
    if lifespan < C.MIN_LIFESPAN:
        return None

    sizes = _cluster_sizes_along_chain(traj, hdb_labels)
    if not sizes:
        return None

    frac_large_enough = float(np.mean(np.array(sizes) >= C.MIN_SIZE))
    if frac_large_enough < C.MIN_SIZE_FRACTION_OF_ALIVE:
        return None

    # --- Sub-scores (each in [0, 1]) ---
    s = {}

    # (1) lifespan — linear up to LIFESPAN_FULL_SCORE
    s["lifespan"] = min(1.0, lifespan / C.LIFESPAN_FULL_SCORE)

    # (2) merge participation
    merge_info = _merge_event_for_trajectory(traj, events)
    s["merge"] = 1.0 if merge_info is not None else 0.0

    # (3) semantic content from P1-4 tagging, restricted to this cluster's tokens
    n_semantic_layers = 0
    n_alive = 0
    chain = _chain_dict(traj)
    for layer, cid in traj["chain"]:
        n_alive += 1
        if layer >= len(hdb_labels):
            continue
        pa = _pair_agreement_at_layer(metrics, layer)
        # Restrict to pairs where both endpoints are in this cluster
        in_cluster_pairs = [
            p for p in pa.get("mutual_pairs", [])
            if p.get("cluster_i") == cid and p.get("cluster_j") == cid
        ]
        if len(in_cluster_pairs) < C.SEMANTIC_PAIR_MIN_COUNT:
            # Not enough pairs to evaluate; fall back to: if any semantic pair
            # involves a cluster member on either side, that counts.
            any_semantic = any(
                p.get("tag") == "semantic" and cid in (p.get("cluster_i"),
                                                       p.get("cluster_j"))
                for p in pa.get("mutual_pairs", [])
            )
            if any_semantic:
                n_semantic_layers += 1
            continue
        frac_semantic = sum(1 for p in in_cluster_pairs
                            if p.get("tag") == "semantic") / len(in_cluster_pairs)
        if frac_semantic >= C.SEMANTIC_PAIR_FRACTION:
            n_semantic_layers += 1
    s["semantic"] = (
        1.0 if n_alive > 0
        and n_semantic_layers / n_alive >= C.SEMANTIC_LAYER_FRACTION
        else 0.0
    )

    # (4) preferred prompt
    s["preferred_prompt"] = 1.0 if prompt_key in C.PREFERRED_PROMPTS else 0.0

    # (5) size — use mean cluster size while alive, scaled
    mean_size = float(np.mean(sizes))
    s["size"] = min(1.0, mean_size / C.SIZE_FULL_SCORE)

    # (6) sibling availability
    sibling_id = _pick_sibling(traj, all_trajectories, events)
    sibling = _traj_by_id(all_trajectories, sibling_id) if sibling_id is not None else None
    s["sibling"] = 1.0 if (sibling and
                           sibling["lifespan"] >= C.MIN_SIBLING_LIFESPAN) else 0.0

    total = sum(C.SCORE_WEIGHTS[k] * v for k, v in s.items())

    return {
        "id":            int(traj["id"]),
        "prompt_key":    prompt_key,
        "start_layer":   int(traj["start_layer"]),
        "end_layer":     int(traj["end_layer"]),
        "lifespan":      int(lifespan),
        "mean_size":     round(mean_size, 2),
        "min_size":      int(min(sizes)),
        "max_size":      int(max(sizes)),
        "merge_event":   merge_info,
        "sibling_id":    sibling_id,
        "sub_scores":    {k: round(v, 3) for k, v in s.items()},
        "total_score":   round(total, 3),
    }


# ---------------------------------------------------------------------------
# Sibling selection
# ---------------------------------------------------------------------------

def _pick_sibling(
    traj: dict,
    all_trajectories: list,
    events: list,
) -> Optional[int]:
    """
    Sibling = either (a) the trajectory that fuses with this one at its merge,
    or (b) the longest-lived contemporary trajectory with which it shares
    the most layers, if no merge partner exists.

    "Contemporary" means overlap of alive-layers >= MIN_LIFESPAN/2.
    """
    # Case (a): merge partner
    merge_info = _merge_event_for_trajectory(traj, events)
    if merge_info is not None:
        prev_ids = set(merge_info["prev_ids"])
        lf = merge_info["layer_from"]
        # Find other trajectories whose cluster at layer lf is in prev_ids
        # and that isn't this one.
        for other in all_trajectories:
            if other["id"] == traj["id"]:
                continue
            for layer, cid in other["chain"]:
                if layer == lf and cid in prev_ids:
                    return int(other["id"])

    # Case (b): nearest contemporary by overlap
    t_layers = set(l for l, _ in traj["chain"])
    best_id, best_overlap = None, 0
    for other in all_trajectories:
        if other["id"] == traj["id"]:
            continue
        o_layers = set(l for l, _ in other["chain"])
        overlap = len(t_layers & o_layers)
        # Prefer longer overlap, break ties by longer lifespan
        if overlap > best_overlap or (
            overlap == best_overlap and best_id is not None
            and other["lifespan"] > _traj_by_id(all_trajectories, best_id)["lifespan"]
        ):
            if overlap >= C.MIN_LIFESPAN // 2:
                best_overlap = overlap
                best_id = int(other["id"])
    return best_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rank_trajectories(phase1_runs: dict) -> list:
    """
    Score every trajectory across all supplied Phase 1 runs.

    Parameters
    ----------
    phase1_runs : dict {prompt_key: loaded phase1 run dict from io.load_phase1_run}

    Returns
    -------
    ranked : list of candidate dicts, sorted descending by total_score.
    """
    candidates = []
    for prompt_key, run in phase1_runs.items():
        trajs = run.get("trajectories", [])
        events = run.get("events", [])
        hdb = run.get("hdbscan_labels") or []
        metrics = run.get("metrics", {})
        if not trajs or hdb is None or len(hdb) == 0:
            continue
        for t in trajs:
            scored = _score_trajectory(
                t, metrics, hdb, events, prompt_key, trajs
            )
            if scored is not None:
                candidates.append(scored)

    candidates.sort(key=lambda c: c["total_score"], reverse=True)
    return candidates


def select_primary_and_sibling(
    phase1_runs: dict,
    force_prompt: Optional[str] = None,
    force_trajectory_id: Optional[int] = None,
    runner_up_rank: int = 1,
) -> dict:
    """
    Pick the primary trajectory (rank 0 by default) and its sibling.

    Parameters
    ----------
    force_prompt, force_trajectory_id : manual override — pick this trajectory
        instead of the top-ranked one. Both must be supplied together.
    runner_up_rank : which rank to designate as the replication case. Default 1
        (i.e. 2nd place). Ignored if fewer than 2 candidates.

    Returns
    -------
    dict with keys:
      ranked    : full ranked list
      primary   : chosen candidate dict
      runner_up : next candidate for replication (or None)
      sibling   : candidate dict for the sibling trajectory (or None —
                  the sibling may not itself pass the gates, and that's fine,
                  we still analyze it in groups A+G)
    """
    ranked = rank_trajectories(phase1_runs)

    # Primary
    if force_prompt is not None and force_trajectory_id is not None:
        primary = next(
            (c for c in ranked
             if c["prompt_key"] == force_prompt
             and c["id"] == force_trajectory_id),
            None,
        )
        if primary is None:
            raise ValueError(
                f"Forced trajectory (prompt={force_prompt}, id={force_trajectory_id}) "
                f"not in ranked list (may have failed a hard gate). "
                f"Use --override-gates to bypass."
            )
    else:
        if not ranked:
            raise RuntimeError(
                "No trajectories survived the selection gates. "
                "Check phase1 runs and try relaxing MIN_LIFESPAN / MIN_SIZE."
            )
        primary = ranked[0]

    runner_up = ranked[runner_up_rank] if len(ranked) > runner_up_rank else None

    # Sibling: look up in the same prompt's trajectories
    sibling = None
    sib_id = primary.get("sibling_id")
    if sib_id is not None:
        # Try ranked candidates first
        sibling = next(
            (c for c in ranked
             if c["prompt_key"] == primary["prompt_key"] and c["id"] == sib_id),
            None,
        )
        if sibling is None:
            # Sibling may have failed a gate — still retrieve its basic record
            run = phase1_runs[primary["prompt_key"]]
            raw = _traj_by_id(run["trajectories"], sib_id)
            if raw is not None:
                sibling = {
                    "id":          int(raw["id"]),
                    "prompt_key":  primary["prompt_key"],
                    "start_layer": int(raw["start_layer"]),
                    "end_layer":   int(raw["end_layer"]),
                    "lifespan":    int(raw["lifespan"]),
                    "note":        "sibling did not pass selection gates",
                }

    return {
        "ranked":    ranked,
        "primary":   primary,
        "runner_up": runner_up,
        "sibling":   sibling,
    }


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def save_selection(selection: dict, out_path: Path) -> None:
    """Write cluster_metadata.json with selection + full ranked list."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selection, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
