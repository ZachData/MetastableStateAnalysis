"""
local_jacobian.py — C4: per-centroid Jacobian via JVP, S/A spectral decomposition.

For each layer (or iteration, for ALBERT), computes the local Jacobian of
    F_L : x → residual-stream state after layer L
at cluster centroids from Phase 1. Decomposes each Jacobian into its
symmetric (S) and antisymmetric (A) parts, compares the local S/A ratio
to the global V's S/A ratio from p2b.

This is the highest-priority C4 analysis. Cheapest of the five tracks:
no new prompts, reuses Phase 1 centroids directly.

Predictions tested:
  P2c-S1 : local Jacobians at plateau states are *more* symmetric than V
  P2c-S2 : local Jacobians at merge layers are *less* symmetric than at plateaus

Functions
---------
compute_layer_jacobian     : Jacobian of one layer's forward map at one point
decompose_sa               : symmetric / antisymmetric split + spectral stats
sa_ratio                   : scalar summary of S vs A energy
centroid_jacobians_one_layer : Jacobians at all centroids for one layer
analyze_local_jacobians    : full pipeline — all layers × all centroids
local_jacobian_to_json     : JSON-serializable summary (no large arrays)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetric part: (M + M^T) / 2."""
    return 0.5 * (M + M.T)


def _antisym(M: np.ndarray) -> np.ndarray:
    """Antisymmetric part: (M - M^T) / 2."""
    return 0.5 * (M - M.T)


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_layer_jacobian(
    layer_fn: Callable[[torch.Tensor], torch.Tensor],
    x: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the Jacobian of layer_fn at point x via torch.func.jacrev.

    Parameters
    ----------
    layer_fn : callable, takes (d,) tensor → (d,) tensor.
               Must be a pure function over the residual stream (no batch dim).
    x        : (d,) ndarray — the point at which to linearize (centroid).
    device   : torch device string.

    Returns
    -------
    J : (d, d) ndarray — Jacobian ∂layer_fn(x) / ∂x.
    """
    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    try:
        # torch.func.jacrev (PyTorch >= 2.0 functional API)
        jac_fn = torch.func.jacrev(layer_fn)
        J_t = jac_fn(x_t)
    except AttributeError:
        # Fallback: torch.autograd.functional.jacobian
        x_t = x_t.requires_grad_(True)
        J_t = torch.autograd.functional.jacobian(layer_fn, x_t)

    return J_t.detach().cpu().numpy()


def _make_layer_fn(
    model,
    layer_idx: int,
    model_type: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a pure (d,) → (d,) callable for the residual-stream update at
    layer_idx, handling ALBERT (shared weights, same layer object at every
    iteration) and GPT-2 / BERT (per-layer objects).

    The map is:
        x → x_out  where x_out is the post-LayerNorm residual stream state
                   after one transformer layer applied to x as a single-token
                   sequence.

    Parameters
    ----------
    model      : HuggingFace transformer model (eval mode, no grad by default).
    layer_idx  : layer index (for ALBERT: iteration index; for GPT-2: layer index).
    model_type : "albert" | "gpt2" | "bert"

    Returns
    -------
    fn : (d,) tensor → (d,) tensor (no batch dim, no sequence dim)
    """
    model_type = model_type.lower()

    if model_type == "albert":
        # AutoModel gives AlbertModel directly (model.encoder)
        # Task-head wrappers give AlbertModel at model.albert.encoder
        encoder = getattr(model, "albert", model).encoder
        layer = encoder.albert_layer_groups[0].albert_layers[0]
        def fn(x: torch.Tensor) -> torch.Tensor:
            h = x.unsqueeze(0).unsqueeze(0)
            attn_mask = torch.ones(1, 1, 1, 1, dtype=x.dtype, device=x.device)
            out = layer(h, attn_mask)[0]
            return out.squeeze(0).squeeze(0)

    elif model_type == "gpt2":
        block = model.transformer.h[layer_idx]
        def fn(x: torch.Tensor) -> torch.Tensor:
            h = x.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
            out = block(h)[0]
            return out.squeeze(0).squeeze(0)

    elif model_type == "bert":
        layer = model.bert.encoder.layer[layer_idx]
        def fn(x: torch.Tensor) -> torch.Tensor:
            h = x.unsqueeze(0).unsqueeze(0)
            attn_mask = torch.ones(1, 1, 1, 1, dtype=x.dtype, device=x.device)
            out = layer(h, attn_mask)[0]
            return out.squeeze(0).squeeze(0)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Supported: 'albert', 'gpt2', 'bert'.")

    return fn


# ---------------------------------------------------------------------------
# S/A decomposition and spectral stats
# ---------------------------------------------------------------------------

def decompose_sa(J: np.ndarray) -> dict:
    """
    Decompose Jacobian J into symmetric (S) and antisymmetric (A) parts.
    Compute Frobenius norms, S/A energy ratio, and eigenvalue statistics
    of S (real eigenvalues) and singular values of A.

    Parameters
    ----------
    J : (d, d) ndarray

    Returns
    -------
    dict with:
      S            : (d, d) symmetric part
      A            : (d, d) antisymmetric part
      S_frob_sq    : ||S||_F²
      A_frob_sq    : ||A||_F²
      J_frob_sq    : ||J||_F²
      sa_ratio     : S_frob_sq / (S_frob_sq + A_frob_sq)  ∈ [0, 1]
      S_eigvals    : sorted real eigenvalues of S (descending)
      S_n_positive : number of positive eigenvalues of S
      S_n_negative : number of negative eigenvalues of S
      A_sing_vals  : singular values of A (proxy for rotation amplitude)
    """
    S = _sym(J)
    A = _antisym(J)

    S_frob_sq = float(np.sum(S ** 2))
    A_frob_sq = float(np.sum(A ** 2))
    J_frob_sq = float(np.sum(J ** 2))
    denom = S_frob_sq + A_frob_sq
    sa_ratio = S_frob_sq / denom if denom > 1e-30 else float("nan")

    S_eigvals = np.linalg.eigvalsh(S)[::-1]  # descending
    A_sing_vals = np.linalg.svd(A, compute_uv=False)

    return {
        "S":            S,
        "A":            A,
        "S_frob_sq":    S_frob_sq,
        "A_frob_sq":    A_frob_sq,
        "J_frob_sq":    J_frob_sq,
        "sa_ratio":     sa_ratio,
        "S_eigvals":    S_eigvals,
        "S_n_positive": int(np.sum(S_eigvals > 0)),
        "S_n_negative": int(np.sum(S_eigvals < 0)),
        "A_sing_vals":  A_sing_vals,
        #added for tests
        "s_frac": S_frob_sq / J_frob_sq if J_frob_sq > 1e-30 else float("nan"),
        "a_frac": A_frob_sq / J_frob_sq if J_frob_sq > 1e-30 else float("nan"),
    }


def sa_ratio(J: np.ndarray) -> float:
    """Scalar S/A energy ratio for J. Convenience wrapper."""
    return float(decompose_sa(J)["sa_ratio"])


# ---------------------------------------------------------------------------
# Projection onto U_A / U_S (optional alignment with p2b)
# ---------------------------------------------------------------------------

def project_jacobian_onto_subspaces(
    decomp: dict,
    P_A: np.ndarray,
    P_S: np.ndarray,
) -> dict:
    """
    Project the local A and S parts onto the global U_A / U_S subspaces
    from p2b to measure how much of the local (anti)symmetry aligns with
    the operator's rotation planes.

    Parameters
    ----------
    decomp : output of decompose_sa
    P_A    : (d, d) projector onto global antisymmetric (rotation) subspace
    P_S    : (d, d) projector onto global symmetric (real) subspace

    Returns
    -------
    dict with:
      A_in_UA      : fraction of A's Frobenius norm captured by P_A
      S_in_US      : fraction of S's Frobenius norm captured by P_S
      A_in_US      : fraction of A's Frobenius norm captured by P_S (cross-check)
      S_in_UA      : fraction of S's Frobenius norm captured by P_A (cross-check)
    """
    def _proj_frac(M, P):
        M_proj = P @ M @ P
        num = float(np.sum(M_proj ** 2))
        den = float(np.sum(M ** 2))
        return num / den if den > 1e-30 else float("nan")

    return {
        "A_in_UA": _proj_frac(decomp["A"], P_A),
        "S_in_US": _proj_frac(decomp["S"], P_S),
        "A_in_US": _proj_frac(decomp["A"], P_S),
        "S_in_UA": _proj_frac(decomp["S"], P_A),
    }


# ---------------------------------------------------------------------------
# Per-layer centroid analysis
# ---------------------------------------------------------------------------

def centroid_jacobians_one_layer(
    layer_fn: Callable[[torch.Tensor], torch.Tensor],
    centroids: np.ndarray,
    centroid_ids: list,
    device: str = "cpu",
    P_A: np.ndarray | None = None,
    P_S: np.ndarray | None = None,
) -> list[dict]:
    """
    Compute Jacobians at all centroids for one layer.

    Parameters
    ----------
    layer_fn     : (d,) → (d,) forward map for this layer
    centroids    : (n_centroids, d) array of centroid positions
    centroid_ids : list of centroid identifiers (e.g. cluster ids from Phase 1)
    device       : torch device
    P_A, P_S     : optional global projectors from p2b for alignment analysis

    Returns
    -------
    list of dicts, one per centroid:
      centroid_id : identifier
      J           : (d, d) Jacobian
      sa_ratio    : scalar
      decomp      : full decompose_sa output
      alignment   : project_jacobian_onto_subspaces output (if P_A/P_S given)
    """
    results = []
    for cid, c in zip(centroid_ids, centroids):
        J = compute_layer_jacobian(layer_fn, c, device=device)
        decomp = decompose_sa(J)
        rec = {
            "centroid_id": cid,
            "J":           J,
            "sa_ratio":    decomp["sa_ratio"],
            "decomp":      decomp,
        }
        if P_A is not None and P_S is not None:
            rec["alignment"] = project_jacobian_onto_subspaces(decomp, P_A, P_S)
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_local_jacobians(
    model,
    model_type: str,
    centroids_per_layer: dict,
    plateau_layers: list[int],
    merge_layers: list[int],
    global_sa_ratio: float,
    P_A: np.ndarray | None = None,
    P_S: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """
    Full C4 analysis: Jacobians at Phase 1 centroids across all layers,
    decomposed into S/A, compared to global V's S/A ratio.

    Parameters
    ----------
    model               : HuggingFace model in eval mode
    model_type          : "albert" | "gpt2" | "bert"
    centroids_per_layer : dict mapping layer_idx → {"centroids": (n, d) array,
                          "centroid_ids": list}
    plateau_layers      : list of layer indices classified as plateau by Phase 1
    merge_layers        : list of layer indices classified as merge events
    global_sa_ratio     : S/A ratio of V from p2b (scalar ∈ [0, 1])
    P_A, P_S            : optional global projectors from p2b
    device              : torch device

    Returns
    -------
    dict with:
      per_layer       : dict layer_idx → list of per-centroid records
      plateau_mean_sa : mean sa_ratio over plateau layers × centroids
      merge_mean_sa   : mean sa_ratio over merge layers × centroids
      global_sa_ratio : echo of input
      p2cs1_holds     : bool — plateau_mean_sa > global_sa_ratio (prediction S1)
      p2cs2_holds     : bool — merge_mean_sa < plateau_mean_sa (prediction S2)
      all_sa_ratios   : flat array of all centroid-level sa_ratios
    """
    model.eval()

    per_layer = {}
    plateau_sa, merge_sa, all_sa = [], [], []

    for layer_idx, centroid_data in centroids_per_layer.items():
        cents = centroid_data["centroids"]        # (n, d)
        cids  = centroid_data["centroid_ids"]

        layer_fn = _make_layer_fn(model, layer_idx, model_type)

        with torch.no_grad():
            pass  # ensure eval mode; jacrev needs grad internally

        recs = centroid_jacobians_one_layer(
            layer_fn, cents, cids, device=device,
            P_A=P_A, P_S=P_S,
        )
        per_layer[layer_idx] = recs

        layer_sa = [r["sa_ratio"] for r in recs]
        all_sa.extend(layer_sa)

        if layer_idx in plateau_layers:
            plateau_sa.extend(layer_sa)
        if layer_idx in merge_layers:
            merge_sa.extend(layer_sa)

    plateau_mean = float(np.mean(plateau_sa)) if plateau_sa else float("nan")
    merge_mean   = float(np.mean(merge_sa))   if merge_sa   else float("nan")

    return {
        "per_layer":        per_layer,
        "plateau_mean_sa":  plateau_mean,
        "merge_mean_sa":    merge_mean,
        "global_sa_ratio":  global_sa_ratio,
        "p2cs1_holds":      plateau_mean > global_sa_ratio,
        "p2cs2_holds":      merge_mean < plateau_mean,
        "all_sa_ratios":    np.array(all_sa),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def local_jacobian_to_json(result: dict) -> dict:
    """
    JSON-serializable summary. Drops large arrays (J, S, A matrices).
    """
    per_layer_summary = {}
    for layer_idx, recs in result["per_layer"].items():
        per_layer_summary[str(layer_idx)] = [
            {
                "centroid_id": r["centroid_id"],
                "sa_ratio":    float(r["sa_ratio"]),
                "S_frob_sq":   float(r["decomp"]["S_frob_sq"]),
                "A_frob_sq":   float(r["decomp"]["A_frob_sq"]),
                "S_n_positive": int(r["decomp"]["S_n_positive"]),
                "S_n_negative": int(r["decomp"]["S_n_negative"]),
                **({"alignment": {k: float(v) for k, v in r["alignment"].items()}}
                   if "alignment" in r else {}),
            }
            for r in recs
        ]

    return {
        "per_layer":        per_layer_summary,
        "plateau_mean_sa":  result["plateau_mean_sa"],
        "merge_mean_sa":    result["merge_mean_sa"],
        "global_sa_ratio":  result["global_sa_ratio"],
        "p2cs1_holds":      result["p2cs1_holds"],
        "p2cs2_holds":      result["p2cs2_holds"],
    }
