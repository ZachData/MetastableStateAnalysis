"""
cluster_profile.py — Group A: structural profile across cluster lifespan.

Pure analysis on cached Phase 1 artifacts. No model required.

What gets measured (per layer of the trajectory's chain):
  - Token membership (strings + indices)
  - Size, intra-cluster mean inner product, radius, silhouettes
  - Centroid angular step, cumulative arc length
  - Mass-near-1 contribution
  - CKA on cluster tokens (vs layer's complement)
  - Membership stability (Jaccard between layer L and L+1)
  - Nesting: does the cluster contain stable sub-clusters? (from Phase 1 P1-3)

Output JSON fragment: one row per layer, plus cluster-level summary scalars.
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level geometry on the sphere
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _cluster_centroid(points: np.ndarray) -> np.ndarray:
    """Unit-normed mean direction of a set of unit vectors."""
    mu = points.mean(axis=0)
    return _unit(mu)


def _angular_step(u: np.ndarray, v: np.ndarray) -> float:
    """Arc distance (radians) between two unit vectors."""
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(dot))


# ---------------------------------------------------------------------------
# Per-layer measurements
# ---------------------------------------------------------------------------

def _silhouette_pair(a: np.ndarray, b: np.ndarray) -> float:
    """
    Silhouette-style score for cluster a vs cluster b on the sphere.
    Uses cosine distance.

    s = (d_other - d_intra) / max(d_other, d_intra)

    Returns scalar mean. If either set has <2 points, returns nan.
    """
    if a.shape[0] < 2 or b.shape[0] < 1:
        return float("nan")
    # Intra-cluster: mean cosine distance to other points in a
    ga = a @ a.T
    ga = np.clip(ga, -1.0, 1.0)
    n = a.shape[0]
    d_intra = ((1.0 - ga).sum(axis=1) - 0.0) / (n - 1)  # subtract self (=0)

    gab = a @ b.T
    gab = np.clip(gab, -1.0, 1.0)
    d_other = (1.0 - gab).mean(axis=1)

    denom = np.maximum(d_other, d_intra)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(denom > 0, (d_other - d_intra) / denom, 0.0)
    return float(np.mean(s))


def _layer_profile(
    acts_layer: np.ndarray,
    hdb_labels: np.ndarray,
    cluster_id: int,
    sibling_id: int,
    tokens: list,
    ) -> dict:
    """Compactness + context metrics for one layer."""
    mask = hdb_labels == cluster_id
    indices = np.where(mask)[0]
    points = acts_layer[mask]

    result = {
        "n":               int(mask.sum()),   # <-- ADD: summary code reads p["n"]
        "n_tokens":        int(mask.sum()),
        "token_idxs":      [int(i) for i in indices],
        "tokens":          [tokens[i] if i < len(tokens) else "?"
                            for i in indices],
        "ip_mean":         float("nan"),
        "radius":          float("nan"),
        "silhouette_sib":  float("nan"),
        "silhouette_all":  float("nan"),
        "centroid":        None,
    }

    if points.shape[0] < 1:
        return result

    centroid = _cluster_centroid(points)
    result["centroid"] = centroid

    # Intra-cluster inner product mean (off-diagonal). Radius = max cosine
    # distance from centroid.
    if points.shape[0] >= 2:
        G = points @ points.T
        n = points.shape[0]
        tri = G[np.triu_indices(n, k=1)]
        result["ip_mean"] = float(tri.mean())
    else:
        result["ip_mean"] = 1.0

    dots = np.clip(points @ centroid, -1.0, 1.0)
    result["radius"] = float((1.0 - dots).max())

    # Silhouettes
    complement = acts_layer[~mask]
    if complement.shape[0] >= 1 and points.shape[0] >= 2:
        result["silhouette_all"] = _silhouette_pair(points, complement)

    sib_mask = hdb_labels == sibling_id
    sib_pts = acts_layer[sib_mask]
    if sib_pts.shape[0] >= 1 and points.shape[0] >= 2:
        result["silhouette_sib"] = _silhouette_pair(points, sib_pts)

    return result


# ---------------------------------------------------------------------------
# Cluster-level aggregates
# ---------------------------------------------------------------------------

def _mass_near_1_contribution(
    acts_layer: np.ndarray,
    hdb_labels: np.ndarray,
    cluster_id: int,
    threshold: float = 0.95,
) -> dict:
    """
    Fraction of the layer's 'mass-near-1' pairs (|<x_i, x_j>| > threshold)
    that are cluster-internal.
    """
    n = acts_layer.shape[0]
    G = acts_layer @ acts_layer.T
    iu = np.triu_indices(n, k=1)
    high_mask = G[iu] > threshold
    total = int(high_mask.sum())

    if total == 0:
        return {"total_pairs_near_1": 0, "cluster_pairs": 0, "fraction": 0.0}

    in_cluster = (hdb_labels == cluster_id)
    ii = iu[0][high_mask]
    jj = iu[1][high_mask]
    cluster_pairs = int((in_cluster[ii] & in_cluster[jj]).sum())

    return {
        "total_pairs_near_1": total,
        "cluster_pairs":      cluster_pairs,
        "fraction":           float(cluster_pairs / total),
    }


def _restricted_cka(
    acts_prev: np.ndarray, acts_curr: np.ndarray,
    hdb_prev: np.ndarray, hdb_curr: np.ndarray,
    cid_prev: int, cid_curr: int,
) -> float:
    """
    Linear CKA between prev-layer and curr-layer activations restricted to
    cluster members. Tokens keep their original indices; if cluster membership
    shifts, we CKA on the union of member-indices present in both layers.
    """
    idx_prev = set(np.where(hdb_prev == cid_prev)[0].tolist())
    idx_curr = set(np.where(hdb_curr == cid_curr)[0].tolist())
    shared = sorted(idx_prev & idx_curr)
    if len(shared) < 3:
        return float("nan")
    X = acts_prev[shared]
    Y = acts_curr[shared]
    # Linear CKA (centered Gram norms)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = (X.T @ Y).ravel() @ (X.T @ Y).ravel()  # ||X.T Y||_F^2
    nx = (X.T @ X).ravel() @ (X.T @ X).ravel()
    ny = (Y.T @ Y).ravel() @ (Y.T @ Y).ravel()
    denom = np.sqrt(nx * ny)
    return float(hsic / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_profile(
    activations: np.ndarray,   # (n_layers, n_tokens, d)
    hdb_labels: list,          # list of (n_tokens,) int arrays
    trajectory: dict,          # {id, chain: [(layer, cid), ...], ...}
    sibling_trajectory: dict,  # same shape, or None
    tokens: list,
    metrics: dict,             # full metrics.json, for nesting + mass_near_1
) -> dict:
    """
    Compute Group A structural profile for one trajectory.

    Returns
    -------
    dict with keys:
      per_layer         : list of per-layer dicts
      centroid_angular_steps : list of radians, length = lifespan-1
      cumulative_arc         : list, length = lifespan
      membership_stability   : list of Jaccard floats, length = lifespan-1
      mean_ip_mean, mean_radius, mean_silhouette_sib, mean_silhouette_all
      cka_restricted    : list of floats per layer transition
      nesting           : summary of P1-3 nesting inside the cluster
      summary           : collapsed scalar summary
    """
    chain = trajectory["chain"]
    sibling_chain = {l: c for l, c in sibling_trajectory["chain"]} if sibling_trajectory else {}

    per_layer = []
    centroids = []
    for layer, cid in chain:
        if layer >= activations.shape[0]:
            break
        sib_cid = sibling_chain.get(layer, -999)
        prof = _layer_profile(
            activations[layer], hdb_labels[layer], cid, sib_cid, tokens,
        )
        prof["layer"] = int(layer)
        prof["cluster_id"] = int(cid)

        # Mass-near-1 contribution
        prof["mass_near_1"] = _mass_near_1_contribution(
            activations[layer], hdb_labels[layer], cid,
        )

        # Nesting (from metrics.json P1-3)
        layers_m = metrics.get("layers", [])
        if layer < len(layers_m):
            nesting_per_cluster = (
                layers_m[layer].get("nesting", {}).get("per_cluster", {})
            )
            # Dict keys may be strings after JSON roundtrip
            nest = (nesting_per_cluster.get(str(cid))
                    or nesting_per_cluster.get(cid)
                    or {})
            prof["nesting_sub_k"] = int(nest.get("spectral_k", 1)) if nest else 1
        else:
            prof["nesting_sub_k"] = None

        per_layer.append(prof)
        if prof["centroid"] is not None:
            centroids.append(prof["centroid"])

    # Centroid trajectory geometry
    centroids_arr = np.array(centroids) if centroids else np.zeros((0, 0))
    steps, cum_arc = [], [0.0]
    for k in range(1, len(centroids_arr)):
        step = _angular_step(centroids_arr[k - 1], centroids_arr[k])
        steps.append(step)
        cum_arc.append(cum_arc[-1] + step)

    # Membership Jaccard layer-to-layer
    jaccards = []
    for (l0, c0), (l1, c1) in zip(chain[:-1], chain[1:]):
        if l0 >= len(hdb_labels) or l1 >= len(hdb_labels):
            continue
        s0 = set(np.where(hdb_labels[l0] == c0)[0].tolist())
        s1 = set(np.where(hdb_labels[l1] == c1)[0].tolist())
        union = s0 | s1
        j = len(s0 & s1) / len(union) if union else 0.0
        jaccards.append(round(j, 4))

    # Restricted CKA
    cka_list = []
    for (l0, c0), (l1, c1) in zip(chain[:-1], chain[1:]):
        if l0 >= activations.shape[0] or l1 >= activations.shape[0]:
            continue
        cka = _restricted_cka(
            activations[l0], activations[l1],
            hdb_labels[l0], hdb_labels[l1],
            c0, c1,
        )
        cka_list.append(round(cka, 4) if not np.isnan(cka) else None)

    # Summaries
    def _mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "lifespan":             len(chain),
        "n_layers_processed":   len(per_layer),
        "mean_size":            _mean([p["n"] for p in per_layer]),
        "mean_ip_mean":         _mean([p["ip_mean"] for p in per_layer]),
        "mean_radius":          _mean([p["radius"] for p in per_layer]),
        "mean_silhouette_sib":  _mean([p["silhouette_sib"] for p in per_layer]),
        "mean_silhouette_all":  _mean([p["silhouette_all"] for p in per_layer]),
        "mean_angular_step":    _mean(steps),
        "total_arc_length":     cum_arc[-1] if cum_arc else 0.0,
        "mean_jaccard_stability": _mean(jaccards),
        "mean_cka_restricted":  _mean(cka_list),
        "mean_mass_near_1_frac": _mean(
            [p["mass_near_1"]["fraction"] for p in per_layer]
        ),
    }

    # Strip centroid arrays from per_layer before serialization (kept in npz)
    centroids_to_save = {}
    for p in per_layer:
        if p.get("centroid") is not None:
            centroids_to_save[f"centroid_L{p['layer']}"] = p["centroid"]
        p.pop("centroid", None)

    return {
        "trajectory_id":          int(trajectory["id"]),
        "per_layer":              per_layer,
        "centroid_angular_steps": [round(s, 4) for s in steps],
        "cumulative_arc":         [round(s, 4) for s in cum_arc],
        "membership_jaccard":     jaccards,
        "cka_restricted":         cka_list,
        "summary":                summary,
        "_centroid_arrays":       centroids_to_save,  # stripped before JSON dump
    }


def save_profile(profile: dict, out_dir: Path, tag: str = "primary") -> None:
    """Write profile JSON + centroid npz to out_dir."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    centroids = profile.pop("_centroid_arrays", {})
    with open(out_dir / f"group_A_profile_{tag}.json", "w") as f:
        json.dump(profile, f, indent=2, default=_json_default)
    if centroids:
        np.savez_compressed(
            out_dir / f"group_A_centroids_{tag}.npz", **centroids,
        )


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
