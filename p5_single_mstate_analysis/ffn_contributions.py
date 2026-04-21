"""
ffn_contributions.py — Group C.2: FFN projection onto cluster directions.

Requires a forward-pass decomposition of the residual stream update per layer
into attn and FFN components. Phase 2's decompose.py produces this
(attn_deltas.npz, ffn_deltas.npz per layer) — we consume those artifacts here.

What gets measured (per layer of the trajectory):
  - FFN_update projected onto the cluster centroid direction
  - FFN_update projected onto the LDA direction (cluster vs sibling)
  - Fraction of FFN energy explaining cluster cohesion vs cluster separation

If decomposition artifacts are missing, this module can run a forward pass
itself via ffn_contributions.compute_ffn_deltas_inline() — but the Phase 2
path is preferred.
"""

import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# LDA direction between two clusters
# ---------------------------------------------------------------------------

def lda_direction(
    X_primary: np.ndarray,    # (n_p, d)
    X_sibling: np.ndarray,    # (n_s, d)
    reg: float = 1e-4,
) -> np.ndarray:
    """
    Fisher's linear discriminant direction for 2-class separation.

    w ∝ Σ_pooled^{-1} (μ_1 - μ_2)

    With ridge reg for numerical stability when n_p + n_s < d_model.

    Returns unit vector (d,), or zero vector if inputs degenerate.
    """
    d = X_primary.shape[1]
    if X_primary.shape[0] < 2 or X_sibling.shape[0] < 2:
        return np.zeros(d, dtype=np.float32)

    mu_p = X_primary.mean(axis=0)
    mu_s = X_sibling.mean(axis=0)
    Xc_p = X_primary - mu_p
    Xc_s = X_sibling - mu_s
    Sw = (Xc_p.T @ Xc_p + Xc_s.T @ Xc_s) / (
        X_primary.shape[0] + X_sibling.shape[0] - 2
    )
    # Ridge
    Sw += reg * np.eye(d, dtype=Sw.dtype)
    try:
        w = np.linalg.solve(Sw, mu_p - mu_s)
    except np.linalg.LinAlgError:
        w = mu_p - mu_s
    n = np.linalg.norm(w)
    return (w / n).astype(np.float32) if n > 1e-12 else np.zeros(d, dtype=np.float32)


# ---------------------------------------------------------------------------
# Projection metrics
# ---------------------------------------------------------------------------

def project_on_direction(deltas: np.ndarray, direction: np.ndarray) -> dict:
    """
    deltas : (n_tokens, d) update vectors
    direction : (d,) unit vector

    Returns:
      mean_proj  : mean of <δ_i, direction>
      std_proj   : std
      energy_frac: ||proj||^2 / ||deltas||^2
    """
    proj = deltas @ direction                                 # (n,)
    proj_sq = float((proj * proj).sum())
    total_sq = float((deltas * deltas).sum())
    return {
        "mean_proj":   round(float(proj.mean()), 4),
        "std_proj":    round(float(proj.std()), 4),
        "energy_frac": round(proj_sq / total_sq, 4) if total_sq > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Loading Phase 2 FFN deltas
# ---------------------------------------------------------------------------

def load_ffn_deltas(
    phase2_run_dir: Path,
    prompt_key: str,
) -> Optional[dict]:
    """
    Load Phase 2 forward-pass decomposition for one prompt.

    Expected files (from phase2.decompose):
      <phase2_run_dir>/<prompt_key>/attn_deltas.npz  (n_layers, n, d)
      <phase2_run_dir>/<prompt_key>/ffn_deltas.npz   (n_layers, n, d)

    Returns dict {"attn": ndarray, "ffn": ndarray} or None.
    """
    phase2_run_dir = Path(phase2_run_dir)
    candidates = [
        phase2_run_dir / prompt_key,
        phase2_run_dir,
        *phase2_run_dir.glob(f"*{prompt_key}*"),
    ]
    for d in candidates:
        if not d.is_dir():
            continue
        attn_p = d / "attn_deltas.npz"
        ffn_p  = d / "ffn_deltas.npz"
        if attn_p.exists() and ffn_p.exists():
            return {
                "attn": np.load(attn_p)["deltas"] if "deltas" in np.load(attn_p).files
                        else np.load(attn_p)[np.load(attn_p).files[0]],
                "ffn":  np.load(ffn_p)["deltas"] if "deltas" in np.load(ffn_p).files
                        else np.load(ffn_p)[np.load(ffn_p).files[0]],
            }
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_ffn(
    activations: np.ndarray,     # (n_layers, n, d)
    hdb_labels: list,
    trajectory: dict,
    sibling_trajectory: dict,
    ffn_deltas: Optional[np.ndarray] = None,
    attn_deltas: Optional[np.ndarray] = None,
) -> dict:
    """
    Group C.2 analysis.

    If ffn_deltas is None, returns per-layer LDA + centroid directions only
    (no projection metrics) — caller is expected to warn the user.
    """
    chain = trajectory["chain"]
    sibling_chain = {l: c for l, c in sibling_trajectory["chain"]} if sibling_trajectory else {}

    per_layer = []
    lda_directions = {}

    for layer, cid in chain:
        if layer >= activations.shape[0]:
            break
        X_layer = activations[layer]
        mask_p = hdb_labels[layer] == cid
        if mask_p.sum() < 2:
            continue

        X_p = X_layer[mask_p]
        centroid = X_p.mean(axis=0)
        centroid_unit = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

        # LDA direction
        sib_cid = sibling_chain.get(layer)
        w_lda = None
        if sib_cid is not None:
            mask_s = hdb_labels[layer] == sib_cid
            if mask_s.sum() >= 2:
                w_lda = lda_direction(X_p, X_layer[mask_s])
                lda_directions[f"lda_L{layer}"] = w_lda

        entry = {
            "layer":          int(layer),
            "cluster_id":     int(cid),
            "has_ffn_deltas": ffn_deltas is not None and layer < ffn_deltas.shape[0],
            "has_lda":        w_lda is not None and float(np.linalg.norm(w_lda)) > 0,
        }

        if ffn_deltas is not None and layer < ffn_deltas.shape[0]:
            ffn_C = ffn_deltas[layer][mask_p]
            entry["ffn_on_centroid"] = project_on_direction(ffn_C, centroid_unit)
            if attn_deltas is not None and layer < attn_deltas.shape[0]:
                attn_C = attn_deltas[layer][mask_p]
                entry["attn_on_centroid"] = project_on_direction(attn_C, centroid_unit)
            if entry["has_lda"]:
                entry["ffn_on_lda"] = project_on_direction(ffn_C, w_lda)
                if attn_deltas is not None and layer < attn_deltas.shape[0]:
                    entry["attn_on_lda"] = project_on_direction(attn_C, w_lda)

        per_layer.append(entry)

    # Summary: does FFN actively maintain cluster cohesion?
    ffn_cohesion_means = [
        p["ffn_on_centroid"]["mean_proj"] for p in per_layer
        if "ffn_on_centroid" in p
    ]
    attn_cohesion_means = [
        p["attn_on_centroid"]["mean_proj"] for p in per_layer
        if "attn_on_centroid" in p
    ]
    ffn_separation_fracs = [
        p["ffn_on_lda"]["energy_frac"] for p in per_layer
        if "ffn_on_lda" in p
    ]

    summary = {
        "mean_ffn_cohesion":  round(float(np.mean(ffn_cohesion_means)), 4)
                              if ffn_cohesion_means else None,
        "mean_attn_cohesion": round(float(np.mean(attn_cohesion_means)), 4)
                              if attn_cohesion_means else None,
        "mean_ffn_lda_frac":  round(float(np.mean(ffn_separation_fracs)), 4)
                              if ffn_separation_fracs else None,
        "ffn_cohesion_verdict": _cohesion_verdict(
            ffn_cohesion_means, attn_cohesion_means,
        ),
    }

    return {
        "trajectory_id":     int(trajectory["id"]),
        "per_layer":         per_layer,
        "summary":           summary,
        "_lda_directions":   lda_directions,   # saved to npz separately
    }


def _cohesion_verdict(ffn_means: list, attn_means: list) -> str:
    """
    Rough verdict: which channel pulls cluster tokens along the centroid
    direction more strongly?
    """
    if not ffn_means and not attn_means:
        return "indeterminate"
    f = float(np.mean(ffn_means)) if ffn_means else 0.0
    a = float(np.mean(attn_means)) if attn_means else 0.0
    if abs(f) < 1e-4 and abs(a) < 1e-4:
        return "both_near_zero"
    if f > 0 and a > 0:
        return "both_cohesive" if abs(f - a) < 0.25 * max(abs(f), abs(a)) else (
            "ffn_dominant_cohesive" if f > a else "attn_dominant_cohesive"
        )
    if f < 0 and a > 0:
        return "attn_cohesive_ffn_disruptive"
    if f > 0 and a < 0:
        return "ffn_cohesive_attn_disruptive"
    return "both_disruptive"


def save_ffn_contributions(result: dict, out_dir: Path,
                           tag: str = "primary") -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lda_dirs = result.pop("_lda_directions", {})
    with open(out_dir / f"group_C2_ffn_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    if lda_dirs:
        np.savez_compressed(
            out_dir / f"group_C2_lda_directions_{tag}.npz", **lda_dirs,
        )


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
