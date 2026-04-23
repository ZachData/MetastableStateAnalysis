"""
sibling_contrast.py — Group G: sibling + random control analysis.

Re-runs subsets of Groups A and C.1 on:
  (a) the sibling trajectory (cluster that fuses with the primary, or the
      nearest contemporary)
  (b) a random control: a same-size subset of non-cluster tokens treated as
      a pseudo-cluster at each layer

The random control is the null baseline — most of A–F should return null
on it. If they don't, the primary signal is weaker than it looks.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .cluster_profile import compute_profile
from .head_contributions import analyze_heads


# ---------------------------------------------------------------------------
# Random control construction
# ---------------------------------------------------------------------------

def build_random_pseudo_trajectory(
    activations: np.ndarray,
    hdb_labels: list,
    primary_chain: list,
    rng: np.random.Generator,
    sibling_chain: list = None,
) -> dict:
    """
    Build a pseudo-trajectory: at each layer of the primary's chain, randomly
    select n_primary tokens from the complement (not in any real cluster the
    primary intersects). These get a synthetic cluster id of -100 (guaranteed
    outside HDBSCAN's id space).

    Returns a trajectory-shaped dict AND augmented hdb_labels with the
    synthetic ids written in. The caller should substitute these labels
    rather than appending; we return fresh copies to avoid mutating input.

    sibling_chain is excluded from candidates to avoid corrupting silhouette
    computations that compare the pseudo-cluster against the primary.
    """
    n_layers = activations.shape[0]
    synth_id = -100

    new_labels = [arr.copy() for arr in hdb_labels]

    new_chain = []
    for layer, cid in primary_chain:
        if layer >= n_layers:
            break
        base = hdb_labels[layer]
        primary_size = int((base == cid).sum())
        if primary_size < 2:
            continue
        # Exclude the primary cluster AND the sibling cluster (if known) so
        # that overwriting sibling tokens doesn't corrupt the contrast metrics.
        exclude_mask = base == cid
        if sibling_chain is not None:
            sib_dict = {l: c for l, c in sibling_chain}
            sib_cid = sib_dict.get(layer)
            if sib_cid is not None:
                exclude_mask = exclude_mask | (base == sib_cid)
        candidates = np.where(~exclude_mask)[0]
        if candidates.size < primary_size:
            continue
        chosen = rng.choice(candidates, size=primary_size, replace=False)
        new_labels[layer][chosen] = synth_id
        new_chain.append((int(layer), synth_id))

    return {
        "id":          -100,
        "chain":       new_chain,
        "start_layer": new_chain[0][0] if new_chain else 0,
        "end_layer":   new_chain[-1][0] if new_chain else 0,
        "lifespan":    len(new_chain),
        "_synthetic":  True,
    }, new_labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_sibling_contrast(
    activations: np.ndarray,
    attentions: np.ndarray,
    hdb_labels: list,
    primary_trajectory: dict,
    sibling_trajectory: dict,
    tokens: list,
    metrics: dict,
    weights: Optional[dict] = None,
    seed: int = 0,
    run_heads_on_sibling: bool = True,
) -> dict:
    """
    Compute structural profile + head contributions for:
      - the sibling
      - a random control pseudo-cluster

    The primary is NOT re-computed here (already done by Groups A/C.1).
    """
    rng = np.random.default_rng(seed)

    results = {}

    # ---- Sibling: Group A + optionally Group C.1 ----
    if sibling_trajectory is not None:
        try:
            profile = compute_profile(
                activations, hdb_labels,
                sibling_trajectory, primary_trajectory,  # primary as its "sibling" for silhouette
                tokens, metrics,
            )
            profile.pop("_centroid_arrays", None)
            results["sibling_profile"] = profile
        except Exception as e:
            results["sibling_profile"] = {"error": str(e)}

        if run_heads_on_sibling and attentions is not None:
            try:
                heads = analyze_heads(
                    activations, attentions, hdb_labels,
                    sibling_trajectory, tokens, weights=weights,
                )
                results["sibling_heads"] = heads
            except Exception as e:
                results["sibling_heads"] = {"error": str(e)}

    # ---- Random control ----
    pseudo_traj, pseudo_labels = build_random_pseudo_trajectory(
        activations, hdb_labels, primary_trajectory["chain"], rng,
        sibling_chain=sibling_trajectory["chain"] if sibling_trajectory is not None else None,
    )
    if pseudo_traj["lifespan"] >= 2:
        try:
            # Use the primary as "sibling" for silhouette contrast
            profile = compute_profile(
                activations, pseudo_labels,
                pseudo_traj, primary_trajectory,
                tokens, metrics,
            )
            profile.pop("_centroid_arrays", None)
            results["random_control_profile"] = profile
        except Exception as e:
            results["random_control_profile"] = {"error": str(e)}

        if attentions is not None:
            try:
                heads = analyze_heads(
                    activations, attentions, pseudo_labels,
                    pseudo_traj, tokens, weights=weights,
                )
                results["random_control_heads"] = heads
            except Exception as e:
                results["random_control_heads"] = {"error": str(e)}

    # ---- Quick contrast summary ----
    def _get_summary(key):
        v = results.get(key, {})
        if isinstance(v, dict):
            return v.get("summary", {})
        return {}

    results["contrast_summary"] = {
        "sibling_mean_ip":  _get_summary("sibling_profile").get("mean_ip_mean"),
        "random_mean_ip":   _get_summary("random_control_profile").get("mean_ip_mean"),
        "sibling_mean_silhouette_all":
            _get_summary("sibling_profile").get("mean_silhouette_all"),
        "random_mean_silhouette_all":
            _get_summary("random_control_profile").get("mean_silhouette_all"),
    }

    return results


def save_sibling_contrast(result: dict, out_dir: Path) -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "group_G_sibling_contrast.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
