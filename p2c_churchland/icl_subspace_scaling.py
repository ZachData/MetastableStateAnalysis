"""
icl_subspace_scaling.py — C5: A/S projection magnitude vs k-shot count.

For a fixed task, construct a graded prompt series at k ∈ {0,1,2,4,8,16}.
At the answer-position token, extract the per-layer residual-stream activation
and project onto P_A (imaginary/rotation channel) and P_S (real channel).
Sum squared projection magnitudes across layers to get one scalar per channel
per k. Fit monotonicity via Spearman correlation.

Predictions tested:
  P2c-M1 : A-channel magnitude grows monotonically with k;
            S-channel does not (or grows less).
  P2c-M2 : The direction within U_A that scales with k is task-specific —
            different tasks select different directions in the rotation subspace.

Functions
---------
extract_answer_token_activations  : per-layer hidden states at answer position
channel_magnitudes_one_prompt     : squared A/S projection per layer for one prompt
kshot_channel_profile             : run across all k-shot prompts for one task
monotonicity_score                : Spearman ρ of magnitude vs k
task_direction_in_ua              : unit-normalised A-channel vector per k (for M2)
cross_task_direction_agreement    : mean cosine between same-k directions across tasks
analyze_icl_scaling               : full M1 + M2 pipeline
icl_scaling_to_json               : JSON-serializable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from scipy.stats import spearmanr
except ImportError:
    def spearmanr(a, b):                    # minimal fallback
        n = len(a)
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return type("R", (), {"correlation": float(
            np.corrcoef(ra, rb)[0, 1]
        ), "pvalue": float("nan")})()


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_answer_token_activations(
    model,
    tokenizer,
    prompt: str,
    answer_token_idx: int | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run the model on prompt and return the residual-stream activations at
    the answer-position token for every layer.

    Parameters
    ----------
    model            : HuggingFace model with output_hidden_states=True support
    tokenizer        : corresponding tokenizer
    prompt           : text prompt
    answer_token_idx : which token position to extract (default: last token, -1)
    device           : torch device string

    Returns
    -------
    activations : (n_layers, d) array — one vector per layer at answer position.
                  Layer 0 = embedding; layers 1..L = transformer block outputs.
    """
    if answer_token_idx is None:
        answer_token_idx = -1

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple of (1, seq_len, d), one per layer (incl. embedding)
    hs = outputs.hidden_states
    acts = np.stack([
        hs[L][0, answer_token_idx, :].cpu().numpy()
        for L in range(len(hs))
    ], axis=0)          # (n_layers, d)
    return acts


# ---------------------------------------------------------------------------
# Per-prompt channel magnitudes
# ---------------------------------------------------------------------------

def channel_magnitudes_one_prompt(
    activations: np.ndarray,
    P_A: np.ndarray,
    P_S: np.ndarray,
) -> dict:
    """
    Compute squared A- and S-channel projection magnitudes per layer
    for one prompt's answer-token activations.

    Parameters
    ----------
    activations : (n_layers, d) — per-layer activations at answer position
    P_A         : (d, d) imaginary-channel projector
    P_S         : (d, d) real-channel projector

    Returns
    -------
    dict with:
      mag_A_per_layer  : (n_layers,) squared projection norms onto P_A
      mag_S_per_layer  : (n_layers,) squared projection norms onto P_S
      mag_A_total      : float — sum across layers
      mag_S_total      : float — sum across layers
      mag_A_normed     : float — total normalized by ||x||² total (relative)
      mag_S_normed     : float — total normalized by ||x||² total (relative)
    """
    n_layers, d = activations.shape
    mag_A = np.array([float(activations[L] @ P_A @ activations[L])
                      for L in range(n_layers)])
    mag_S = np.array([float(activations[L] @ P_S @ activations[L])
                      for L in range(n_layers)])
    total_sq = np.array([float(activations[L] @ activations[L])
                         for L in range(n_layers)])

    sum_total = float(np.sum(total_sq))
    return {
        "mag_A_per_layer": mag_A,
        "mag_S_per_layer": mag_S,
        "mag_A_total":     float(np.sum(mag_A)),
        "mag_S_total":     float(np.sum(mag_S)),
        "mag_A_normed":    float(np.sum(mag_A)) / max(sum_total, 1e-30),
        "mag_S_normed":    float(np.sum(mag_S)) / max(sum_total, 1e-30),
    }


# ---------------------------------------------------------------------------
# k-shot profile across prompts
# ---------------------------------------------------------------------------

def kshot_channel_profile(
    model,
    tokenizer,
    kshot_prompts: list[dict],
    P_A: np.ndarray,
    P_S: np.ndarray,
    device: str = "cpu",
) -> dict:
    """
    Run channel magnitude analysis across all k-shot prompts for one task.

    Parameters
    ----------
    model          : HuggingFace model
    tokenizer      : tokenizer
    kshot_prompts  : list of dicts, each with:
                       "k"           : int — number of examples in prompt
                       "prompt"      : str — full prompt text
                       "answer_idx"  : int (optional) — answer token position
    P_A, P_S       : (d, d) projectors

    Returns
    -------
    dict with:
      k_vals        : list of k values (sorted)
      mag_A         : (n_k,) total A-channel magnitude per k
      mag_S         : (n_k,) total S-channel magnitude per k
      mag_A_normed  : (n_k,) normalized A-channel magnitude
      mag_S_normed  : (n_k,) normalized S-channel magnitude
      per_k         : list of per-k dicts (full channel_magnitudes output)
      a_directions  : (n_k, n_layers, d) A-channel projected vectors (for M2)
    """
    kshot_prompts_sorted = sorted(kshot_prompts, key=lambda x: x["k"])
    k_vals, mag_A, mag_S, mag_An, mag_Sn, per_k = [], [], [], [], [], []
    a_dirs = []

    for entry in kshot_prompts_sorted:
        k   = entry["k"]
        txt = entry["prompt"]
        ans_idx = entry.get("answer_idx", None)

        acts = extract_answer_token_activations(
            model, tokenizer, txt,
            answer_token_idx=ans_idx,
            device=device,
        )
        mags = channel_magnitudes_one_prompt(acts, P_A, P_S)

        # A-channel direction per layer: P_A @ x (not squared, for M2)
        a_vecs = np.stack([P_A @ acts[L] for L in range(acts.shape[0])], axis=0)
        a_dirs.append(a_vecs)

        k_vals.append(k)
        mag_A.append(mags["mag_A_total"])
        mag_S.append(mags["mag_S_total"])
        mag_An.append(mags["mag_A_normed"])
        mag_Sn.append(mags["mag_S_normed"])
        per_k.append({"k": k, **mags})

    return {
        "k_vals":       k_vals,
        "mag_A":        np.array(mag_A),
        "mag_S":        np.array(mag_S),
        "mag_A_normed": np.array(mag_An),
        "mag_S_normed": np.array(mag_Sn),
        "per_k":        per_k,
        "a_directions": np.stack(a_dirs, axis=0),   # (n_k, n_layers, d)
    }


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def monotonicity_score(
    k_vals: list[int] | np.ndarray,
    magnitudes: np.ndarray,
) -> dict:
    """
    Spearman ρ of magnitude vs k. Tests monotone growth.

    Returns
    -------
    dict with:
      rho    : float — Spearman correlation
      pvalue : float
      monotone_increasing : bool — rho > 0 and pvalue < 0.05
    """
    r = spearmanr(k_vals, magnitudes)
    return {
        "rho":                float(r.correlation),
        "pvalue":             float(r.pvalue),
        "monotone_increasing": float(r.correlation) > 0 and float(r.pvalue) < 0.05,
    }


# ---------------------------------------------------------------------------
# M2 — task-specific direction analysis
# ---------------------------------------------------------------------------

def task_direction_in_ua(
    a_directions: np.ndarray,
    layer_idx: int = -1,
) -> np.ndarray:
    """
    Extract the unit-normalised A-channel direction at a given layer
    for each k-shot count.

    Parameters
    ----------
    a_directions : (n_k, n_layers, d) — A-channel vectors per k per layer
    layer_idx    : which layer to use (default: last layer)

    Returns
    -------
    directions : (n_k, d) unit-normalised A-channel vectors
    """
    vecs = a_directions[:, layer_idx, :]    # (n_k, d)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def cross_task_direction_agreement(
    task_directions: dict[str, np.ndarray],
    k_vals: list[int],
) -> dict:
    """
    For each k, compute mean pairwise cosine similarity between tasks'
    A-channel directions. Low similarity → task-specific (M2 holds).

    Parameters
    ----------
    task_directions : dict task_name → (n_k, d) unit-normalised direction array
    k_vals          : list of k values (must be same order for all tasks)

    Returns
    -------
    dict with:
      per_k_mean_cosine  : (n_k,) mean pairwise cosine per k
      overall_mean_cosine: float
      p2cm2_holds        : bool — overall_mean_cosine < 0.5 (directions are task-specific)
    """
    task_names = list(task_directions.keys())
    n_k = len(k_vals)

    per_k_cosines = []
    for ki in range(n_k):
        cosines = []
        for i in range(len(task_names)):
            for j in range(i + 1, len(task_names)):
                d_i = task_directions[task_names[i]][ki]
                d_j = task_directions[task_names[j]][ki]
                cos = float(np.dot(d_i, d_j) /
                            max(np.linalg.norm(d_i) * np.linalg.norm(d_j), 1e-12))
                cosines.append(abs(cos))   # |cos| — direction sign is arbitrary
        per_k_cosines.append(float(np.mean(cosines)) if cosines else float("nan"))

    overall = float(np.nanmean(per_k_cosines))
    return {
        "per_k_mean_cosine":   per_k_cosines,
        "overall_mean_cosine": overall,
        "p2cm2_holds":         overall < 0.5,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_icl_scaling(
    model,
    tokenizer,
    task_prompt_sets: dict[str, list[dict]],
    P_A: np.ndarray,
    P_S: np.ndarray,
    direction_layer_idx: int = -1,
    device: str = "cpu",
) -> dict:
    """
    Full C5 M1 + M2 analysis across one or more tasks.

    Parameters
    ----------
    model             : HuggingFace model
    tokenizer         : tokenizer
    task_prompt_sets  : dict task_name → list of kshot_prompt dicts
                        Each dict: {"k": int, "prompt": str, "answer_idx": int}
    P_A, P_S          : (d, d) channel projectors
    direction_layer_idx : layer to extract A-channel direction for M2 (default: last)
    device            : torch device

    Returns
    -------
    dict with:
      per_task           : dict task → kshot_channel_profile output
      per_task_m1        : dict task → monotonicity_score for A and S channels
      m2_analysis        : cross_task_direction_agreement output (if >1 task)
      p2cm1_holds        : bool — A-channel monotone in at least one task,
                                  S-channel is not (or less monotone)
      p2cm2_holds        : bool — task directions are task-specific
    """
    per_task, per_task_m1, task_dirs = {}, {}, {}

    for task_name, prompts in task_prompt_sets.items():
        profile = kshot_channel_profile(
            model, tokenizer, prompts, P_A, P_S, device=device
        )
        per_task[task_name] = profile

        k_vals = profile["k_vals"]
        m1_A = monotonicity_score(k_vals, profile["mag_A_normed"])
        m1_S = monotonicity_score(k_vals, profile["mag_S_normed"])
        per_task_m1[task_name] = {"A": m1_A, "S": m1_S}

        dirs = task_direction_in_ua(profile["a_directions"], layer_idx=direction_layer_idx)
        task_dirs[task_name] = dirs

    # P2c-M1: A monotone in ≥1 task, and more monotone than S on average
    any_A_monotone = any(
        v["A"]["monotone_increasing"] for v in per_task_m1.values()
    )
    mean_rho_A = float(np.mean([v["A"]["rho"] for v in per_task_m1.values()]))
    mean_rho_S = float(np.mean([v["S"]["rho"] for v in per_task_m1.values()]))
    p2cm1_holds = any_A_monotone and (mean_rho_A > mean_rho_S)

    # P2c-M2: task directions are task-specific
    m2_analysis = None
    p2cm2_holds = False
    if len(task_dirs) > 1:
        # Use k_vals from first task
        first_k = list(per_task.values())[0]["k_vals"]
        m2_analysis = cross_task_direction_agreement(task_dirs, first_k)
        p2cm2_holds = m2_analysis["p2cm2_holds"]

    return {
        "per_task":     per_task,
        "per_task_m1":  per_task_m1,
        "m2_analysis":  m2_analysis,
        "mean_rho_A":   mean_rho_A,
        "mean_rho_S":   mean_rho_S,
        "p2cm1_holds":  p2cm1_holds,
        "p2cm2_holds":  p2cm2_holds,
    }


# ---------------------------------------------------------------------------
# Prompt grid loader
# ---------------------------------------------------------------------------

def load_kshot_prompts(path: str | Path) -> dict[str, list[dict]]:
    """
    Load k-shot prompt grid from icl_kshot.json.

    Expected format:
    {
      "task_name": [
        {"k": 0, "prompt": "...", "answer_idx": -1},
        {"k": 1, "prompt": "...", "answer_idx": -1},
        ...
      ],
      ...
    }
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def icl_scaling_to_json(result: dict) -> dict:
    per_task_summary = {}
    for task, profile in result["per_task"].items():
        m1 = result["per_task_m1"][task]
        per_task_summary[task] = {
            "k_vals":       profile["k_vals"],
            "mag_A_normed": profile["mag_A_normed"].tolist(),
            "mag_S_normed": profile["mag_S_normed"].tolist(),
            "m1_A_rho":     float(m1["A"]["rho"]),
            "m1_A_pvalue":  float(m1["A"]["pvalue"]),
            "m1_S_rho":     float(m1["S"]["rho"]),
            "m1_S_pvalue":  float(m1["S"]["pvalue"]),
            "m1_A_monotone": bool(m1["A"]["monotone_increasing"]),
        }

    out = {
        "per_task":    per_task_summary,
        "mean_rho_A":  float(result["mean_rho_A"]),
        "mean_rho_S":  float(result["mean_rho_S"]),
        "p2cm1_holds": bool(result["p2cm1_holds"]),
        "p2cm2_holds": bool(result["p2cm2_holds"]),
    }
    if result["m2_analysis"] is not None:
        m2 = result["m2_analysis"]
        out["m2_analysis"] = {
            "per_k_mean_cosine":    m2["per_k_mean_cosine"],
            "overall_mean_cosine":  float(m2["overall_mean_cosine"]),
        }
    return out
