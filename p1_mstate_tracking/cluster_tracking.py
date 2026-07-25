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

    # Initialize from layer 0.
    # tip_map: (layer, cluster_id) -> traj_id for the *current tip* of every
    # active trajectory.  Updated incrementally so each lookup is O(1) instead
    # of the previous O(n_trajectories) linear scan.
    tip_map: dict = {}   # (layer, cid) -> traj_id

    ids_0 = sorted(set(label_arrays[0]) - {-1})
    for cid in ids_0:
        active_trajectories[next_traj_id] = [(0, cid)]
        tip_map[(0, cid)] = next_traj_id
        next_traj_id += 1

    for ev in events:
        lf = ev["layer_from"]
        lt = ev["layer_to"]

        # Extend matched trajectories
        for prev_id, curr_id, _ in ev["matches"]:
            tid = tip_map.get((lf, prev_id))
            if tid is not None:
                active_trajectories[tid].append((lt, curr_id))
                # Move the tip forward: old tip key is now stale, add new one.
                del tip_map[(lf, prev_id)]
                tip_map[(lt, curr_id)] = tid

        # Births: start new trajectories
        for cid in ev["births"]:
            active_trajectories[next_traj_id] = [(lt, cid)]
            tip_map[(lt, cid)] = next_traj_id
            next_traj_id += 1

        # Merges: extend the primary trajectory, terminate secondaries.
        # The first prev_id with a live tip becomes primary; the rest end.
        for prev_ids, curr_id in ev["merges"]:
            primary_tid = None
            for pid in prev_ids:
                tid = tip_map.get((lf, pid))
                if tid is not None:
                    if primary_tid is None:
                        primary_tid = tid
                        active_trajectories[tid].append((lt, curr_id))
                        del tip_map[(lf, pid)]
                        tip_map[(lt, curr_id)] = tid
                    else:
                        # Secondary trajectory ends here — remove its tip entry.
                        tip_map.pop((lf, pid), None)

        # Deaths: stale tip entries for dead clusters.
        # These were never matched or merged, so their tip keys still point to
        # lf.  Remove them to keep tip_map clean.
        for dead_id in ev["deaths"]:
            tip_map.pop((lf, dead_id), None)

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
"""
APPEND THIS BLOCK to the end of p1_mstate_tracking/cluster_tracking.py.

It is not a standalone module — it is written as an addition so nothing in
the existing 359-line file has to be retyped or re-diffed. The only import
it needs (`numpy as np`) is already at the top of that file.

Also update that file's module docstring "Functions" list to add:

    compute_behavior_trajectories : per-trajectory output distributions,
                                    masked by the same chain

--------------------------------------------------------------------------
WHY THIS LIVES IN PHASE 1 AND NOT PHASE 5b

It is the same tracking operation compute_centroid_trajectories already
performs — walk a trajectory's chain of (layer, cluster_id) pairs, mask the
member tokens, aggregate — applied to a different per-token quantity. The
activation-side half of that pair has been here since Phase 1; the
behavior-side half was never written, and Phase 5b compensated by taking a
global mean over ALL tokens at each plateau layer, which is what decoupled
Mh's population from My's and silently disabled Sub-exp B (see
design-5b.md). Putting the sibling next to its twin is also the tracking-
module merge INDEX.md already lists as outstanding.
"""

# ===========================================================================
# --------------------------- BEGIN APPEND BLOCK ---------------------------
# ===========================================================================


def _layer_lookup(container, layer_idx):
    """
    Fetch a per-layer array from either a dict keyed by layer index or a
    list indexed by it. Returns None when the layer is absent.

    Both shapes occur in this codebase and neither is wrong:
    compute_centroid_trajectories is called with lists (Phase 1's own
    analysis loop holds every layer); Phase 5b calls with dicts, because
    its logit cache is deliberately sparse — extract_layer_logits only
    materializes the layers asked for, and a full (n_layers, n_tokens,
    vocab) tensor at GPT-2-large scale is not something to build by
    accident. Accepting both here is cheaper than forcing either caller to
    convert.
    """
    if container is None:
        return None
    if isinstance(container, dict):
        got = container.get(layer_idx)
        if got is None:
            got = container.get(int(layer_idx))
        return got
    try:
        if 0 <= layer_idx < len(container):
            return container[layer_idx]
    except TypeError:
        return None
    return None


def _normalize_rows(p, eps: float = 1e-12):
    """Clip negatives and renormalize rows to sum to 1. float64 out."""
    arr = np.asarray(p, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum(axis=-1, keepdims=True)
    return arr / np.maximum(s, eps)


def compute_behavior_trajectories(
    tracking,
    label_arrays,
    logit_dists,
    space: str = "hellinger",
) -> tuple:
    """
    Per-trajectory output distributions — the behavior-side twin of
    compute_centroid_trajectories.

    For each trajectory, walk its chain of (layer_idx, cluster_id) pairs.
    At each step, mask the tokens belonging to that cluster at that layer —
    THE SAME MASK the centroid used — and average their decoded output
    distributions. The result is one distribution sequence per trajectory,
    over exactly the trajectories and in exactly the order that
    compute_centroid_trajectories produces centroids for.

    Parameters
    ----------
    tracking     : output of track_clusters, OR the bare list of trajectory
                   dicts (each {"id", "chain", ...}). Both accepted;
                   compute_centroid_trajectories takes the former.
    label_arrays : {layer_idx: (n_tokens,) int} or list of the same —
                   HDBSCAN labels per layer. Phase 5b gets these from
                   p1_visualization/loaders.py::_hdbscan_labels.
    logit_dists  : {layer_idx: (n_tokens, vocab) float} or list — decoded
                   output distributions per layer, from
                   p5b_manifold_steering/logit_cache.py::extract_layer_logits.
                   May be SPARSE: layers absent from it are skipped, and
                   the coverage return value records what was actually used.
    space        : how to aggregate distributions across the chain.
                   "hellinger" (default) — mean of √p, renormalized, then
                   squared back. "mixture" — plain arithmetic mean of p.

    Returns
    -------
    behavior_trajs : {trajectory_id: (m, vocab) float32} — one row per
                     chain step that had both labels and logits, in chain
                     order. Trajectories with zero covered steps are absent
                     from the dict entirely (not present-but-empty).
    coverage       : {trajectory_id: {"layers_used": [int],
                                      "layers_in_chain": int,
                                      "frac": float}}

    WHY `space` DEFAULTS TO "hellinger"
    -----------------------------------
    The activation side aggregates by taking the mean of L2-normalized
    vectors and renormalizing (compute_centroid_trajectories, and again in
    load_plateau_centroids) — a spherical mean, not a Euclidean one. The
    exact structural analog on the behavior side is the spherical mean in
    the Hellinger embedding: √p is a unit vector, so mean-then-renormalize
    there is the same operation in the same geometry. My's own fit maps
    p → √p before doing anything else (fit_behavior_manifold), so this also
    avoids aggregating in one space and fitting in another.

    "mixture" is the operationally natural reading — "the cluster's typical
    next-token distribution" really is the arithmetic mean — but it is
    entropy-increasing: averaging peaked distributions in probability space
    blurs them toward uniform faster than in √p space, which compresses the
    behavior-side distances the isometry test is trying to resolve. Both
    are available; whichever is used must be recorded in isometry.json,
    because it changes the numbers.

    COVERAGE IS RETURNED, NOT HIDDEN
    --------------------------------
    A trajectory covered at 1 of 5 chain layers and one covered at 5 of 5
    are not equally-good measurements, and collapsing both to "a
    distribution" is exactly the kind of silent degradation this project's
    artifact-contract discipline exists to prevent. The caller decides
    whether to drop low-coverage trajectories; this function does not
    decide for them.
    """
    if space not in ("hellinger", "mixture"):
        raise ValueError(
            f"compute_behavior_trajectories: space must be 'hellinger' or "
            f"'mixture', got {space!r}"
        )

    if isinstance(tracking, dict):
        trajectories = tracking.get("trajectories", [])
    else:
        trajectories = tracking or []

    behavior_trajs = {}
    coverage = {}

    for traj in trajectories:
        tid = int(traj["id"])
        chain = traj.get("chain", [])
        rows = []
        used = []

        for layer_idx, cluster_id in chain:
            layer_idx = int(layer_idx)
            labels = _layer_lookup(label_arrays, layer_idx)
            probs = _layer_lookup(logit_dists, layer_idx)
            if labels is None or probs is None:
                continue

            labels = np.asarray(labels)
            probs = np.asarray(probs)
            if labels.shape[0] != probs.shape[0]:
                # Token-count disagreement means the label array and the
                # logit array came from different forward passes. Refuse
                # rather than mask with a mismatched index.
                raise ValueError(
                    f"compute_behavior_trajectories: layer {layer_idx} has "
                    f"{labels.shape[0]} labels but {probs.shape[0]} logit "
                    f"rows — labels and logits are not from the same pass"
                )

            mask = labels == cluster_id
            if not mask.any():
                continue

            member = _normalize_rows(probs[mask])
            if space == "hellinger":
                sq = np.sqrt(member)
                m = sq.mean(axis=0)
                nrm = float(np.linalg.norm(m))
                m = m / max(nrm, 1e-12)
                rows.append(m ** 2)
            else:
                rows.append(member.mean(axis=0))

            used.append(layer_idx)

        coverage[tid] = {
            "layers_used": used,
            "layers_in_chain": len(chain),
            "frac": (len(used) / len(chain)) if chain else 0.0,
        }
        if rows:
            behavior_trajs[tid] = np.asarray(rows, dtype=np.float32)

    return behavior_trajs, coverage


def stack_behavior_by_traj_ids(
    behavior_trajs: dict,
    traj_ids: list,
    space: str = "hellinger",
) -> tuple:
    """
    Reduce per-trajectory distribution sequences to one distribution each,
    stacked in the order given by `traj_ids`.

    This is the function that actually enforces the alignment. `traj_ids`
    is load_plateau_centroids' second return value — the identity list for
    the whole phase. Iterating it (rather than iterating behavior_trajs'
    own keys, or a plateau-layer list, or anything else) is what guarantees
    row i of the returned stack and row i of the centroid array describe
    the same cluster.

    Returns
    -------
    dists   : (n_kept, vocab) float32
    kept    : list[int] — the subset of traj_ids that had any coverage, in
              order. Callers MUST re-index their centroid array by this
              list rather than assuming it equals traj_ids; a trajectory
              whose chain layers were all absent from the logit cache has
              a centroid but no distribution.
    """
    stacked = []
    kept = []
    for tid in traj_ids:
        seq = behavior_trajs.get(int(tid))
        if seq is None or len(seq) == 0:
            continue
        seq = _normalize_rows(seq)
        if space == "hellinger":
            sq = np.sqrt(seq)
            m = sq.mean(axis=0)
            m = m / max(float(np.linalg.norm(m)), 1e-12)
            stacked.append(m ** 2)
        else:
            stacked.append(seq.mean(axis=0))
        kept.append(int(tid))

    if not stacked:
        raise ValueError(
            "stack_behavior_by_traj_ids: no trajectory in traj_ids had any "
            "logit coverage. Check that extract_layer_logits was asked for "
            "the layers appearing in these trajectories' chains — the "
            "plateau_layers + merge_layers union is NOT sufficient."
        )

    return np.stack(stacked, axis=0).astype(np.float32), kept


# ===========================================================================
# ---------------------------- END APPEND BLOCK ----------------------------
# ===========================================================================