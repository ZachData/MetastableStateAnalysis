"""
cluster_tracking.py — Track HDBSCAN clusters across adjacent layers.

Match clusters between layers (L, L+1) by maximum Jaccard overlap of
token membership.  Record births, deaths, merges, and matched centroid
trajectories.

Functions
---------
track_clusters       : full layer-by-layer tracking from results dict
match_layer_pair     : Jaccard overlap matching between two label vectors
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _jaccard_overlap_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> tuple:
    """
    Compute Jaccard overlap between every (cluster_a, cluster_b) pair.

    Noise tokens (label == -1) are excluded from matching.

    Returns
    -------
    overlap : (n_a, n_b) float matrix — Jaccard(Cₐ, Cᵦ)
    ids_a   : sorted unique cluster IDs from labels_a (excl. -1)
    ids_b   : sorted unique cluster IDs from labels_b (excl. -1)
    """
    ids_a = sorted(set(labels_a) - {-1})
    ids_b = sorted(set(labels_b) - {-1})
    if not ids_a or not ids_b:
        return np.zeros((len(ids_a), len(ids_b))), ids_a, ids_b

    # Precompute membership sets
    sets_a = {c: set(np.where(labels_a == c)[0]) for c in ids_a}
    sets_b = {c: set(np.where(labels_b == c)[0]) for c in ids_b}

    overlap = np.zeros((len(ids_a), len(ids_b)), dtype=np.float64)
    for i, ca in enumerate(ids_a):
        sa = sets_a[ca]
        for j, cb in enumerate(ids_b):
            sb = sets_b[cb]
            inter = len(sa & sb)
            union = len(sa | sb)
            overlap[i, j] = inter / union if union > 0 else 0.0

    return overlap, ids_a, ids_b


def match_layer_pair(
    labels_prev: np.ndarray,
    labels_curr: np.ndarray,
    min_jaccard: float = 0.1,
) -> dict:
    """
    Match HDBSCAN clusters between two adjacent layers.

    Uses the Hungarian algorithm on the negated Jaccard overlap matrix
    for optimal assignment, then filters matches below min_jaccard.

    Parameters
    ----------
    labels_prev : (n_tokens,) int array — HDBSCAN labels at layer L
    labels_curr : (n_tokens,) int array — HDBSCAN labels at layer L+1
    min_jaccard : minimum overlap to count as a valid match

    Returns
    -------
    dict with keys:
      matches : list of (prev_id, curr_id, jaccard) — matched pairs
      births  : list of curr cluster IDs with no match in prev
      deaths  : list of prev cluster IDs with no match in curr
      merges  : list of (list_of_prev_ids, curr_id) — many-to-one matches
    """
    overlap, ids_prev, ids_curr = _jaccard_overlap_matrix(labels_prev, labels_curr)

    if overlap.size == 0:
        return {
            "matches": [],
            "births": list(ids_curr),
            "deaths": list(ids_prev),
            "merges": [],
        }

    # Hungarian on negated overlap for maximum-weight matching.
    # linear_sum_assignment minimises cost, so negate.
    n_prev, n_curr = overlap.shape
    # Pad to square if needed
    size = max(n_prev, n_curr)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:n_prev, :n_curr] = -overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    # Extract valid matches (within actual cluster range and above threshold)
    matched_prev = set()
    matched_curr = set()
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n_prev and c < n_curr and overlap[r, c] >= min_jaccard:
            matches.append((ids_prev[r], ids_curr[c], float(overlap[r, c])))
            matched_prev.add(ids_prev[r])
            matched_curr.add(ids_curr[c])

    # Now check for merges: unmatched prev clusters that have significant
    # overlap with an already-matched curr cluster.
    merges = []
    unmatched_prev = [c for c in ids_prev if c not in matched_prev]
    for up in list(unmatched_prev):
        i = ids_prev.index(up)
        # Find best curr overlap
        best_j = int(np.argmax(overlap[i, :]))
        if overlap[i, best_j] >= min_jaccard:
            target_curr = ids_curr[best_j]
            if target_curr in matched_curr:
                # This is a merge: up merged into target_curr
                # Find existing merge group or create one
                found = False
                for mg in merges:
                    if mg[1] == target_curr:
                        mg[0].append(up)
                        found = True
                        break
                if not found:
                    # Find the prev cluster that was already matched to target_curr
                    primary_prev = [m[0] for m in matches if m[1] == target_curr]
                    merges.append((primary_prev + [up], target_curr))
                matched_prev.add(up)
                unmatched_prev.remove(up)

    births = [c for c in ids_curr if c not in matched_curr]
    deaths = list(unmatched_prev)

    return {
        "matches": matches,
        "births": births,
        "deaths": deaths,
        "merges": [(sorted(prev_ids), int(curr_id)) for prev_ids, curr_id in merges],
    }


def track_clusters(results: dict) -> dict:
    """
    Full layer-by-layer HDBSCAN cluster tracking.

    Parameters
    ----------
    results : analysis results dict (must contain per-layer clustering.hdbscan.labels)

    Returns
    -------
    dict with keys:
      events          : list of per-layer-transition dicts with matches/births/deaths/merges
      centroid_ids    : list of tracked centroid trajectory IDs
      centroid_layers : (n_trajectories,) list of (start_layer, end_layer) tuples
      centroid_coords : list of (n_layers_alive, d) arrays — centroid positions per trajectory
      summary         : dict with total births, deaths, merges, max_alive
    """
    layers = results["layers"]
    n_layers = len(layers)

    # Check HDBSCAN availability
    has_hdbscan = all(
        "hdbscan" in lr.get("clustering", {})
        for lr in layers
    )
    if not has_hdbscan:
        return {
            "events": [],
            "centroid_ids": [],
            "centroid_layers": [],
            "centroid_coords": [],
            "summary": {"total_births": 0, "total_deaths": 0, "total_merges": 0, "max_alive": 0},
        }

    # Extract label arrays
    label_arrays = [
        np.array(lr["clustering"]["hdbscan"]["labels"], dtype=np.int32)
        for lr in layers
    ]

    # Per-transition matching
    events = []
    for i in range(n_layers - 1):
        ev = match_layer_pair(label_arrays[i], label_arrays[i + 1])
        ev["layer_from"] = i
        ev["layer_to"] = i + 1
        events.append(ev)

    # Build centroid trajectories by chaining matches across layers.
    # Each trajectory is a sequence of (layer, cluster_id) pairs.
    # Start with all clusters at layer 0.
    active_trajectories = {}  # traj_id -> list of (layer, cluster_id)
    next_traj_id = 0

    # Initialize from layer 0
    ids_0 = sorted(set(label_arrays[0]) - {-1})
    for cid in ids_0:
        active_trajectories[next_traj_id] = [(0, cid)]
        next_traj_id += 1

    # Build a reverse lookup: (layer, cluster_id) -> traj_id
    def _lookup(layer, cid):
        for tid, chain in active_trajectories.items():
            if chain[-1] == (layer, cid):
                return tid
        return None

    for ev in events:
        lf = ev["layer_from"]
        lt = ev["layer_to"]

        # Extend matched trajectories
        for prev_id, curr_id, _ in ev["matches"]:
            tid = _lookup(lf, prev_id)
            if tid is not None:
                active_trajectories[tid].append((lt, curr_id))

        # Births: start new trajectories
        for cid in ev["births"]:
            active_trajectories[next_traj_id] = [(lt, cid)]
            next_traj_id += 1

        # Merges: extend the primary trajectory, terminate secondaries
        for prev_ids, curr_id in ev["merges"]:
            # The first prev_id that has an active trajectory becomes primary
            primary_tid = None
            for pid in prev_ids:
                tid = _lookup(lf, pid)
                if tid is not None:
                    if primary_tid is None:
                        primary_tid = tid
                        active_trajectories[tid].append((lt, curr_id))
                    # Secondary trajectories just end (death is implicit)

        # Deaths: trajectories that ended (no action needed — they just stop)

    # Extract centroid coordinates using stored centroids or recompute from labels
    # We use KMeans centroids stored in results if available, but for HDBSCAN
    # tracking we need to compute centroids from the HDBSCAN labels directly.
    # This requires the normed activations, which we don't have here.
    # Instead, store the trajectory chains; centroid coordinates are filled in
    # by save_run when activations are available.

    # Summary stats
    total_births = sum(len(ev["births"]) for ev in events)
    total_deaths = sum(len(ev["deaths"]) for ev in events)
    total_merges = sum(len(ev["merges"]) for ev in events)
    # Count max clusters alive at any layer
    max_alive = max(
        (len(set(la) - {-1}) for la in label_arrays),
        default=0,
    )

    # Trajectory lifespan info
    traj_info = []
    for tid in sorted(active_trajectories):
        chain = active_trajectories[tid]
        start_layer = chain[0][0]
        end_layer = chain[-1][0]
        traj_info.append({
            "id": tid,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "lifespan": end_layer - start_layer + 1,
            "chain": chain,
        })

    return {
        "events": [
            {
                "layer_from": ev["layer_from"],
                "layer_to": ev["layer_to"],
                "n_matches": len(ev["matches"]),
                "n_births": len(ev["births"]),
                "n_deaths": len(ev["deaths"]),
                "n_merges": len(ev["merges"]),
                "matches": [(int(a), int(b), float(j)) for a, b, j in ev["matches"]],
                "births": [int(b) for b in ev["births"]],
                "deaths": [int(d) for d in ev["deaths"]],
                "merges": ev["merges"],
            }
            for ev in events
        ],
        "trajectories": traj_info,
        "summary": {
            "total_births": total_births,
            "total_deaths": total_deaths,
            "total_merges": total_merges,
            "max_alive": max_alive,
            "n_trajectories": len(traj_info),
            "mean_lifespan": float(np.mean([t["lifespan"] for t in traj_info])) if traj_info else 0.0,
            "max_lifespan": max((t["lifespan"] for t in traj_info), default=0),
        },
    }


def compute_centroid_trajectories(
    tracking: dict,
    hidden_states: list,
    label_arrays: list,
) -> dict:
    """
    Compute actual centroid coordinates for each tracked trajectory.

    Parameters
    ----------
    tracking      : output of track_clusters
    hidden_states : list of (n_tokens, d) normed activation arrays per layer
    label_arrays  : list of (n_tokens,) HDBSCAN label arrays per layer

    Returns
    -------
    dict mapping trajectory_id -> (lifespan, d) float32 array of centroid positions
    """
    from core.models import layernorm_to_sphere
    import torch

    centroid_trajs = {}
    for traj in tracking.get("trajectories", []):
        tid = traj["id"]
        coords = []
        for layer_idx, cluster_id in traj["chain"]:
            if layer_idx < len(hidden_states) and layer_idx < len(label_arrays):
                acts = hidden_states[layer_idx]
                if isinstance(acts, torch.Tensor):
                    normed = layernorm_to_sphere(acts).numpy()
                else:
                    normed = acts
                labels = label_arrays[layer_idx]
                mask = labels == cluster_id
                if mask.any():
                    mean_vec = normed[mask].mean(axis=0)
                    norm = np.linalg.norm(mean_vec)
                    centroid = mean_vec / norm if norm > 1e-10 else mean_vec
                    coords.append(centroid)
                else:
                    coords.append(np.zeros(normed.shape[1], dtype=np.float32))
            else:
                break
        if coords:
            centroid_trajs[tid] = np.array(coords, dtype=np.float32)

    return centroid_trajs
