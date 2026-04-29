"""
context_selection.py — C5: subspace divergence between matched-context prompt pairs.

For prompt pairs that share the same input content but differ only in
instruction/context (e.g., "answer in English" vs "answer in French"),
measure whether the S-channel trajectory is similar and the A-channel
trajectory diverges — the transformer analog of Mante et al. 2013.

Prediction tested:
  P2c-M3 : S-channel trajectory similar across context pairs;
            A-channel trajectory diverges in a context-selected subspace.
            Failure: reversed, or no divergence.

Pipeline per pair
-----------------
1. Run the model on both prompts; extract per-layer answer-token activations.
2. Project each layer's activation onto P_A and P_S.
3. Compute layer-wise S-channel cosine similarity and A-channel L2 divergence.
4. Aggregate across layers: mean_S_sim, mean_A_div.
5. Verdict: M3 holds if mean_S_sim > mean_A_sim (S more similar than A).

Functions
---------
trajectory_channel_projections  : per-layer S and A projected vectors
layer_cosine_similarity         : cosine between two trajectory channels per layer
layer_l2_divergence             : L2 distance between two trajectory channels per layer
analyze_context_pair            : full analysis for one prompt pair
subspace_divergence_direction   : find the A-channel direction of divergence
analyze_context_selection       : full M3 pipeline over all pairs
print_context_selection         : terminal report
context_selection_to_json       : JSON-serializable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from p2c_churchland.icl_subspace_scaling import extract_answer_token_activations


# ---------------------------------------------------------------------------
# Per-layer channel projections
# ---------------------------------------------------------------------------

def trajectory_channel_projections(
    activations: np.ndarray,
    P_A: np.ndarray,
    P_S: np.ndarray,
) -> dict:
    """
    Project each layer's activation onto P_A and P_S.

    Parameters
    ----------
    activations : (n_layers, d)
    P_A         : (d, d) imaginary-channel projector
    P_S         : (d, d) real-channel projector

    Returns
    -------
    dict with:
      A_vecs : (n_layers, d) — P_A @ x per layer
      S_vecs : (n_layers, d) — P_S @ x per layer
    """
    A_vecs = activations @ P_A.T   # (n_layers, d); P_A symmetric so P_A^T = P_A
    S_vecs = activations @ P_S.T
    return {"A_vecs": A_vecs, "S_vecs": S_vecs}


# ---------------------------------------------------------------------------
# Layer-wise similarity / divergence
# ---------------------------------------------------------------------------

def layer_cosine_similarity(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-layer cosine similarity between two trajectory channel arrays.

    Parameters
    ----------
    vecs_a, vecs_b : (n_layers, d)

    Returns
    -------
    cosines : (n_layers,) ∈ [-1, 1]
    """
    n_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
    n_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    dots = np.sum(vecs_a * vecs_b, axis=1)
    return dots / np.maximum(n_a[:, 0] * n_b[:, 0], eps)


def layer_l2_divergence(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
) -> np.ndarray:
    """
    Per-layer L2 distance between two trajectory channel arrays.

    Parameters
    ----------
    vecs_a, vecs_b : (n_layers, d)

    Returns
    -------
    dists : (n_layers,) ≥ 0
    """
    return np.linalg.norm(vecs_a - vecs_b, axis=1)


def layer_angular_divergence(
    vecs_a: np.ndarray,
    vecs_b: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-layer angular distance (arccos of cosine similarity) in degrees.
    Complement to cosine similarity — more interpretable for subspace work.
    """
    cos = layer_cosine_similarity(vecs_a, vecs_b, eps=eps)
    cos_clipped = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos_clipped))


# ---------------------------------------------------------------------------
# Single pair analysis
# ---------------------------------------------------------------------------

def analyze_context_pair(
    model,
    tokenizer,
    prompt_a: str,
    prompt_b: str,
    P_A: np.ndarray,
    P_S: np.ndarray,
    answer_idx_a: int | None = None,
    answer_idx_b: int | None = None,
    device: str = "cpu",
) -> dict:
    """
    Full M3 analysis for one prompt pair.

    Parameters
    ----------
    model, tokenizer      : HuggingFace model + tokenizer
    prompt_a, prompt_b    : the two prompts (same input, different context)
    P_A, P_S              : (d, d) channel projectors
    answer_idx_a/b        : answer token positions (default: last token)
    device                : torch device

    Returns
    -------
    dict with:
      S_cosine_per_layer   : (n_layers,) S-channel cosine similarity
      A_cosine_per_layer   : (n_layers,) A-channel cosine similarity
      A_l2_per_layer       : (n_layers,) A-channel L2 divergence
      A_angle_per_layer    : (n_layers,) A-channel angular divergence (degrees)
      mean_S_cosine        : float — mean S-channel similarity
      mean_A_cosine        : float — mean A-channel similarity
      mean_A_l2            : float — mean A-channel L2 divergence
      m3_holds             : bool — S more similar than A (mean_S_cosine > mean_A_cosine)
      m3_ratio             : float — mean_S_cosine / (mean_A_cosine + eps)
                             > 1 → M3 pattern, < 1 → reversed
      divergence_layer     : int — layer with peak A-channel angular divergence
    """
    acts_a = extract_answer_token_activations(
        model, tokenizer, prompt_a, answer_token_idx=answer_idx_a, device=device
    )
    acts_b = extract_answer_token_activations(
        model, tokenizer, prompt_b, answer_token_idx=answer_idx_b, device=device
    )

    # Handle mismatched layer counts (shouldn't happen but be safe)
    n_layers = min(acts_a.shape[0], acts_b.shape[0])
    acts_a, acts_b = acts_a[:n_layers], acts_b[:n_layers]

    proj_a = trajectory_channel_projections(acts_a, P_A, P_S)
    proj_b = trajectory_channel_projections(acts_b, P_A, P_S)

    S_cos = layer_cosine_similarity(proj_a["S_vecs"], proj_b["S_vecs"])
    A_cos = layer_cosine_similarity(proj_a["A_vecs"], proj_b["A_vecs"])
    A_l2  = layer_l2_divergence(proj_a["A_vecs"], proj_b["A_vecs"])
    A_ang = layer_angular_divergence(proj_a["A_vecs"], proj_b["A_vecs"])

    mean_S = float(np.mean(S_cos))
    mean_A = float(np.mean(A_cos))
    mean_l2 = float(np.mean(A_l2))

    return {
        "S_cosine_per_layer":  S_cos,
        "A_cosine_per_layer":  A_cos,
        "A_l2_per_layer":      A_l2,
        "A_angle_per_layer":   A_ang,
        "mean_S_cosine":       mean_S,
        "mean_A_cosine":       mean_A,
        "mean_A_l2":           mean_l2,
        "m3_holds":            mean_S > mean_A,
        "m3_ratio":            mean_S / max(mean_A, 1e-12),
        "divergence_layer":    int(np.argmax(A_ang)),
    }


# ---------------------------------------------------------------------------
# Subspace divergence direction
# ---------------------------------------------------------------------------

def subspace_divergence_direction(
    model,
    tokenizer,
    pairs: list[dict],
    P_A: np.ndarray,
    device: str = "cpu",
    divergence_layer: int = -1,
) -> dict:
    """
    Find the direction within U_A along which context pairs diverge most.

    For each pair, the divergence vector at divergence_layer is:
        Δa = (P_A @ x_a) - (P_A @ x_b)
    Stack all Δa vectors and take the top singular vector: the principal
    context-selection direction in the A-channel.

    Parameters
    ----------
    pairs            : list of pair dicts (same format as load_context_pairs output)
    P_A              : (d, d) imaginary-channel projector
    divergence_layer : which layer to extract divergence at (default: last)
    device           : torch device

    Returns
    -------
    dict with:
      divergence_direction : (d,) unit vector — principal A-channel divergence direction
      explained_variance   : float — fraction of total divergence variance on this direction
      delta_vecs           : (n_pairs, d) raw divergence vectors (for downstream use)
    """
    delta_vecs = []
    for pair in pairs:
        acts_a = extract_answer_token_activations(
            model, tokenizer, pair["prompt_a"],
            answer_token_idx=pair.get("answer_idx_a", None), device=device
        )
        acts_b = extract_answer_token_activations(
            model, tokenizer, pair["prompt_b"],
            answer_token_idx=pair.get("answer_idx_b", None), device=device
        )
        n = min(acts_a.shape[0], acts_b.shape[0])
        l_idx = divergence_layer if divergence_layer >= 0 else n + divergence_layer
        l_idx = min(l_idx, n - 1)

        delta = P_A @ (acts_a[l_idx] - acts_b[l_idx])
        delta_vecs.append(delta)

    D = np.stack(delta_vecs, axis=0)    # (n_pairs, d)
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    top_dir = Vt[0]                      # (d,) principal divergence direction
    ev = float(s[0] ** 2) / max(float(np.sum(s ** 2)), 1e-30)

    return {
        "divergence_direction": top_dir,
        "explained_variance":   ev,
        "delta_vecs":           D,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_context_selection(
    model,
    tokenizer,
    pairs: list[dict],
    P_A: np.ndarray,
    P_S: np.ndarray,
    device: str = "cpu",
) -> dict:
    """
    Full M3 analysis over all context pairs.

    Parameters
    ----------
    model, tokenizer : HuggingFace model + tokenizer
    pairs            : list of pair dicts, each with:
                         "prompt_a"    : str
                         "prompt_b"    : str
                         "label"       : str (optional, e.g. "en_vs_fr")
                         "answer_idx_a": int (optional)
                         "answer_idx_b": int (optional)
    P_A, P_S         : (d, d) channel projectors
    device           : torch device

    Returns
    -------
    dict with:
      per_pair         : list of analyze_context_pair outputs
      mean_S_cosine    : float — mean across pairs
      mean_A_cosine    : float — mean across pairs
      mean_m3_ratio    : float
      p2cm3_holds      : bool — mean_S_cosine > mean_A_cosine across pairs
      n_pairs_m3_holds : int — count of pairs where m3_holds individually
      divergence_analysis : subspace_divergence_direction output
    """
    per_pair = []
    for pair in pairs:
        rec = analyze_context_pair(
            model, tokenizer,
            pair["prompt_a"], pair["prompt_b"],
            P_A, P_S,
            answer_idx_a=pair.get("answer_idx_a"),
            answer_idx_b=pair.get("answer_idx_b"),
            device=device,
        )
        rec["label"] = pair.get("label", "")
        per_pair.append(rec)

    mean_S  = float(np.mean([r["mean_S_cosine"] for r in per_pair]))
    mean_A  = float(np.mean([r["mean_A_cosine"] for r in per_pair]))
    mean_r  = float(np.mean([r["m3_ratio"]      for r in per_pair]))
    n_holds = int(sum(r["m3_holds"] for r in per_pair))

    div_analysis = subspace_divergence_direction(
        model, tokenizer, pairs, P_A, device=device
    )

    return {
        "per_pair":            per_pair,
        "mean_S_cosine":       mean_S,
        "mean_A_cosine":       mean_A,
        "mean_m3_ratio":       mean_r,
        "p2cm3_holds":         mean_S > mean_A,
        "n_pairs_m3_holds":    n_holds,
        "n_pairs_total":       len(per_pair),
        "divergence_analysis": div_analysis,
    }


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_context_selection(result: dict) -> None:
    sep = "-" * 60
    print(sep)
    print("C5 — Context Selection (P2c-M3)")
    print(sep)
    print(f"  Pairs analysed : {result['n_pairs_total']}")
    print(f"  Mean S-channel cosine : {result['mean_S_cosine']:.4f}")
    print(f"  Mean A-channel cosine : {result['mean_A_cosine']:.4f}")
    print(f"  Mean M3 ratio (S/A)   : {result['mean_m3_ratio']:.4f}")
    print()
    v = "HOLDS" if result["p2cm3_holds"] else "FAILS"
    print(f"  P2c-M3 {v}: S more similar than A in "
          f"{result['n_pairs_m3_holds']}/{result['n_pairs_total']} pairs")
    da = result["divergence_analysis"]
    print(f"  A-channel principal divergence direction "
          f"explains {da['explained_variance']*100:.1f}% of pair variance")
    print()
    for r in result["per_pair"]:
        tag = "✓" if r["m3_holds"] else "✗"
        print(f"  {tag} [{r['label']:20s}]  "
              f"S_cos={r['mean_S_cosine']:.3f}  "
              f"A_cos={r['mean_A_cosine']:.3f}  "
              f"peak_div_layer={r['divergence_layer']}")
    print(sep)


# ---------------------------------------------------------------------------
# Prompt grid loader
# ---------------------------------------------------------------------------

def load_context_pairs(path: str | Path) -> list[dict]:
    """
    Load context pairs from context_pairs.json.

    Expected format:
    [
      {
        "label":       "en_vs_fr",
        "prompt_a":    "Answer in English. ...",
        "prompt_b":    "Answer in French. ...",
        "answer_idx_a": -1,
        "answer_idx_b": -1
      },
      ...
    ]
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def context_selection_to_json(result: dict) -> dict:
    per_pair_summary = [
        {
            "label":          r["label"],
            "mean_S_cosine":  float(r["mean_S_cosine"]),
            "mean_A_cosine":  float(r["mean_A_cosine"]),
            "m3_holds":       bool(r["m3_holds"]),
            "m3_ratio":       float(r["m3_ratio"]),
            "divergence_layer": int(r["divergence_layer"]),
        }
        for r in result["per_pair"]
    ]
    da = result["divergence_analysis"]
    return {
        "per_pair":             per_pair_summary,
        "mean_S_cosine":        float(result["mean_S_cosine"]),
        "mean_A_cosine":        float(result["mean_A_cosine"]),
        "mean_m3_ratio":        float(result["mean_m3_ratio"]),
        "p2cm3_holds":          bool(result["p2cm3_holds"]),
        "n_pairs_m3_holds":     int(result["n_pairs_m3_holds"]),
        "n_pairs_total":        int(result["n_pairs_total"]),
        "divergence_explained_variance": float(da["explained_variance"]),
    }
