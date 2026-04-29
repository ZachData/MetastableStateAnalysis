"""
imaginary_ablation.py — Layer-depth ablation of imaginary OV components.

Block 3 of Phase 2i.  Tests the discriminator hypothesis from the
complex-eigenspectra discussion: ablate Im(OV) = antisymmetric part
A = (V-V^T)/2 at increasing depth thresholds, then measure whether output
degrades before, with, or independently of cluster structure.

Outcome classification
----------------------
ARTIFACTUAL       neither measure degrades at any depth
COMPUTATION_ONLY  output degrades while cluster ARI stays high
COUPLED           both degrade at a shared threshold (within ±1 layer)
INDETERMINATE     pattern does not fit any of the above

Implementation note
-------------------
We work on stored residual-stream activations without re-running the model.
Ablation: x_l_ablated = x_l - Pi_A @ x_l  for l >= threshold,
where Pi_A is the projector onto col(A).  For per-layer models each layer
has its own Pi_A; for ALBERT the shared Pi_A is replicated.

This is a post-hoc projection, valid as a first-order approximation in the
near-linear regime confirmed by Phase 2i Block 1b.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional

try:
    import hdbscan as _hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

try:
    from sklearn.metrics import adjusted_rand_score as _ari_fn
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Projector construction
# ---------------------------------------------------------------------------

def build_imaginary_projector(V: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Orthogonal projector onto col(A),  A = (V - V^T) / 2.

    Returns (d, d) matrix.  Zero matrix if V is symmetric (A ≈ 0).
    """
    V = np.asarray(V, dtype=float)
    A = (V - V.T) / 2.0
    frob = np.linalg.norm(A, "fro")
    if frob < tol:
        return np.zeros((V.shape[0], V.shape[0]))
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > tol * s[0]))
    if rank == 0:
        return np.zeros((V.shape[0], V.shape[0]))
    Ur = U[:, :rank]
    return Ur @ Ur.T


def build_projectors_all_layers(ov_data: dict) -> list[np.ndarray]:
    """
    One Pi_A per layer (or one repeated for shared-weight models).

    Returns list of length == number of layer_names entries.
    """
    if ov_data["is_per_layer"]:
        return [build_imaginary_projector(V) for V in ov_data["ov_total"]]
    Pi = build_imaginary_projector(ov_data["ov_total"])
    return [Pi] * len(ov_data["layer_names"])


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def apply_ablation_from_threshold(
    activations: np.ndarray,
    projectors:  list[np.ndarray],
    threshold:   int,
) -> np.ndarray:
    """
    Zero the imaginary component of activations for layers >= threshold.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)
    projectors  : list[n_layers] of (d, d) projectors
    threshold   : first layer index at which ablation is applied

    Returns
    -------
    (n_layers, n_tokens, d) with x_l replaced by x_l - Pi_A @ x_l for l >= threshold
    """
    ablated = activations.copy()
    n = activations.shape[0]
    for l in range(max(0, threshold), n):
        if l < len(projectors):
            Pi = projectors[l]
            # row-wise: (n_tokens, d) -= (Pi @ acts[l].T).T
            ablated[l] = activations[l] - (Pi @ activations[l].T).T
    return ablated


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def output_cosine_degradation(acts_orig: np.ndarray, acts_ablated: np.ndarray) -> float:
    """Mean cosine distance at the final layer.  Range [0, 1]; 0 = identical."""
    a = acts_orig[-1]      # (N, d)
    b = acts_ablated[-1]   # (N, d)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    cos_sim = np.sum(a * b, axis=-1)                  # (N,) ∈ [-1, 1]
    dist = np.clip(1.0 - cos_sim, 0.0, 1.0)           # map to [0, 1]
    return float(np.mean(dist))


def _cluster_labels(X: np.ndarray, min_cluster_size: int = 3) -> np.ndarray:
    if _HAS_HDBSCAN:
        return _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    # Fallback: spectral 2-split on top singular vector
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    scores = X @ Vt[0]
    return (scores >= np.median(scores)).astype(int)


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SKLEARN:
        try:
            return float(_ari_fn(a, b))
        except Exception:
            return float("nan")
    # Fallback: simple agreement fraction
    return float(np.mean(a == b))


def cluster_ari_at_plateau_layers(
    acts_orig:        np.ndarray,
    acts_ablated:     np.ndarray,
    plateau_layers:   list[int],
    min_cluster_size: int = 3,
) -> dict:
    """
    HDBSCAN ARI between original and ablated activations at each plateau layer.

    Returns
    -------
    dict:
      per_plateau : list of {layer, ari, n_clusters_orig, n_clusters_ablated}
      mean_ari    : float
    """
    per = []
    for l in plateau_layers:
        if l >= acts_orig.shape[0]:
            continue
        lo = _cluster_labels(acts_orig[l],    min_cluster_size)
        la = _cluster_labels(acts_ablated[l], min_cluster_size)
        per.append({
            "layer":              l,
            "ari":                _ari(lo, la),
            "n_clusters_orig":    len(set(lo) - {-1}),
            "n_clusters_ablated": len(set(la) - {-1}),
        })
    aris = [p["ari"] for p in per if math.isfinite(p["ari"])]
    return {
        "per_plateau": per,
        "mean_ari":    float(np.mean(aris)) if aris else float("nan"),
    }


# ---------------------------------------------------------------------------
# Depth sweep
# ---------------------------------------------------------------------------

def depth_sweep(
    activations:    np.ndarray,
    ov_data:        dict,
    plateau_layers: list[int],
    thresholds:     Optional[list[int]] = None,
) -> list[dict]:
    """
    Sweep threshold from 0 to n_layers-1, compute metrics at each.

    Returns list of dicts:
      threshold, output_degradation, mean_cluster_ari, per_plateau
    """
    n = activations.shape[0]
    if thresholds is None:
        thresholds = list(range(n))
    projectors = build_projectors_all_layers(ov_data)
    out = []
    for thr in thresholds:
        abl = apply_ablation_from_threshold(activations, projectors, thr)
        out.append({
            "threshold":          thr,
            "output_degradation": output_cosine_degradation(activations, abl),
            **cluster_ari_at_plateau_layers(activations, abl, plateau_layers),
        })
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_CLUSTER_ARI_DISRUPTED  = 0.7   # below this = cluster structure disrupted
_OUTPUT_DEG_DEGRADED    = 0.05  # above this = output degraded


def classify_outcome(sweep: list[dict]) -> dict:
    """
    Classify sweep into ARTIFACTUAL / COMPUTATION_ONLY / COUPLED / INDETERMINATE.

    Scans from the shallowest threshold inward to find the first threshold
    at which each measure crosses its threshold.
    """
    if not sweep:
        return {"outcome": "INDETERMINATE", "reason": "empty_sweep",
                "output_threshold": None, "cluster_threshold": None}

    out_thr = next(
        (s["threshold"] for s in sweep
         if s["output_degradation"] > _OUTPUT_DEG_DEGRADED),
        None,
    )
    clu_thr = next(
        (s["threshold"] for s in sweep
         if math.isfinite(s.get("mean_ari", float("nan")))
         and s["mean_ari"] < _CLUSTER_ARI_DISRUPTED),
        None,
    )

    if out_thr is None and clu_thr is None:
        outcome = "ARTIFACTUAL"
        reason  = "neither measure degrades across all depths"
    elif out_thr is not None and clu_thr is None:
        outcome = "COMPUTATION_ONLY"
        reason  = f"output degrades at depth {out_thr}; clusters never disrupted"
    elif out_thr is None and clu_thr is not None:
        outcome = "INDETERMINATE"
        reason  = (f"clusters disrupted at depth {clu_thr} without output degrading"
                   " — unexpected pattern")
    else:
        gap = abs(out_thr - clu_thr)
        if gap <= 1:
            outcome = "COUPLED"
            reason  = f"both degrade at depths {out_thr}/{clu_thr} (gap={gap})"
        elif out_thr < clu_thr:
            outcome = "COMPUTATION_ONLY"
            reason  = f"output first at {out_thr}, clusters at {clu_thr}"
        else:
            outcome = "INDETERMINATE"
            reason  = f"clusters before output ({clu_thr} < {out_thr})"

    return {"outcome": outcome, "reason": reason,
            "output_threshold": out_thr, "cluster_threshold": clu_thr}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _extract_plateau_layers(phase1_events: dict, n_layers: int) -> list[int]:
    for key in ("plateau_layers", "metastable_windows", "cluster_plateaus"):
        val = phase1_events.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            return sorted(int(l) for l in val if int(l) < n_layers)
        if isinstance(val, dict):
            layers = []
            for v in val.values():
                if isinstance(v, list):
                    layers.extend(int(l) for l in v if int(l) < n_layers)
            if layers:
                return sorted(set(layers))
    # Fallback: middle third
    lo, hi = n_layers // 3, 2 * n_layers // 3
    return list(range(lo, hi))


def analyze_imaginary_ablation(
    activations:   np.ndarray,
    ov_data:       dict,
    phase1_events: dict,
    thresholds:    Optional[list[int]] = None,
) -> dict:
    """
    Full pipeline: depth sweep → outcome classification → interpretation.

    Returns
    -------
    dict: sweep, classification, plateau_layers, n_layers, interpretation
    """
    plateau_layers = _extract_plateau_layers(phase1_events, activations.shape[0])
    sweep          = depth_sweep(activations, ov_data, plateau_layers, thresholds)
    classification = classify_outcome(sweep)
    return {
        "sweep":          sweep,
        "classification": classification,
        "plateau_layers": plateau_layers,
        "n_layers":       activations.shape[0],
        "interpretation": _ablation_interpretation(classification),
    }


def _ablation_interpretation(c: dict) -> str:
    o = c.get("outcome", "INDETERMINATE")
    if o == "ARTIFACTUAL":
        return ("Imaginary components are inert at all depths. Consistent "
                "with LayerNorm-induced phantom rotation (H2).")
    if o == "COMPUTATION_ONLY":
        return ("Imaginary components carry computation orthogonal to basin "
                "selection. Output degrades without disrupting cluster identity. "
                "Consistent with emergent positional/phase encoding (H1).")
    if o == "COUPLED":
        thr = c.get("output_threshold", "?")
        return (f"Rotation and basin structure are coupled past depth ~{thr}. "
                "Rotation is load-bearing for cluster identity beyond that layer.")
    return f"Indeterminate: {c.get('reason', 'insufficient signal')}."


def ablation_to_json(result: dict) -> dict:
    """JSON-serialisable (drops no data, sanitises non-finite floats)."""
    def _c(v):
        if isinstance(v, (float, np.floating)):
            f = float(v)
            return f if math.isfinite(f) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    def _cd(d):
        if isinstance(d, dict):  return {k: _cd(v) for k, v in d.items()}
        if isinstance(d, list):  return [_cd(x) for x in d]
        return _c(d)
    return _cd(result)


def ablation_summary_lines(result: dict) -> list[str]:
    """LLM-ready lines for phase2i_summary.txt."""
    c     = result.get("classification", {})
    sweep = result.get("sweep", [])
    lines = ["--- Block 3: Imaginary ablation depth sweep ---",
             f"  Outcome:        {c.get('outcome', 'INDETERMINATE')}",
             f"  Reason:         {c.get('reason', 'n/a')}",
             f"  Output deg threshold layer:  {c.get('output_threshold', 'none')}",
             f"  Cluster ARI threshold layer: {c.get('cluster_threshold', 'none')}"]
    if sweep:
        degs = [s["output_degradation"] for s in sweep
                if math.isfinite(s["output_degradation"])]
        aris = [s["mean_ari"] for s in sweep
                if math.isfinite(s.get("mean_ari", float("nan")))]
        if degs:
            lines.append(f"  Output degradation range: "
                         f"[{min(degs):.4f}, {max(degs):.4f}]")
        if aris:
            lines.append(f"  Cluster ARI range:        "
                         f"[{min(aris):.4f}, {max(aris):.4f}]")
    lines.append(f"  Interpretation: {result.get('interpretation', 'n/a')}")
    return lines