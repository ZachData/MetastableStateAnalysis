"""
head_contributions.py — Group C.1: per-head attention analysis.

Per-head, per-layer: classify each attention head by how it treats the cluster.

Classifications (based on A[cluster, cluster] vs A[cluster, complement] mass):
  - inward       : cluster tokens attend mostly inside cluster
  - outward      : cluster tokens attend outside the cluster
  - ignoring     : head's cluster rows have near-uniform attention
  - positional   : attention concentrates by position rather than cluster
                   (heuristic: strong correlation between |i - j| and A[i, j])

Scalar cohesion score per head = Σ_{i ∈ C} <Δ^(h)_i, x̄_C>
  where Δ^(h)_i is the residual contribution of head h at token i.
  Positive = head pulls the cluster together along its mean direction.
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Attention distribution classification
# ---------------------------------------------------------------------------

def _uniformity_score(row: np.ndarray) -> float:
    """
    1 - (entropy ratio). 1 = uniform, 0 = peaked.
    """
    row = np.clip(row, 1e-12, None)
    row = row / row.sum()
    H = -float((row * np.log(row)).sum())
    H_max = float(np.log(len(row)))
    return 1.0 - (H_max - H) / H_max if H_max > 0 else 0.0


def _positional_score(A_cluster_rows: np.ndarray, cluster_idx: np.ndarray,
                      all_idx: np.ndarray) -> float:
    """
    Pearson correlation between |i - j| and A[i, j] for cluster rows.
    Negative correlation (closer tokens get more attention) → positional head.
    Returns |correlation|.
    """
    if A_cluster_rows.size == 0:
        return 0.0
    i_grid, j_grid = np.meshgrid(cluster_idx, all_idx, indexing="ij")
    dists = np.abs(i_grid - j_grid).astype(np.float64)
    d_flat = dists.ravel()
    a_flat = A_cluster_rows.ravel()
    if np.std(d_flat) < 1e-9 or np.std(a_flat) < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(d_flat, a_flat)[0, 1]))


def classify_head(
    A_h: np.ndarray,       # (n, n) single head's attention for this layer
    cluster_idx: np.ndarray,
    uniform_threshold: float = 0.8,
    positional_threshold: float = 0.5,
) -> dict:
    """
    Classify one head's behavior on this cluster.

    Returns dict with:
      inward_mass    : sum of A[C, C] / sum of A[C, :]
      outward_mass   : 1 - inward_mass
      uniformity     : mean entropy-ratio for cluster rows
      positional     : |corr(|i-j|, A[i,j])| for cluster rows
      classification : string
    """
    n = A_h.shape[0]
    all_idx = np.arange(n)
    A_rows = A_h[cluster_idx]
    row_total = A_rows.sum(axis=1) + 1e-12
    inward = A_rows[:, cluster_idx].sum(axis=1)
    inward_frac = float((inward / row_total).mean())

    unif = float(np.mean([_uniformity_score(A_rows[i]) for i in range(A_rows.shape[0])]))
    positional = _positional_score(A_rows, cluster_idx, all_idx)

    if unif >= uniform_threshold:
        classification = "ignoring"
    elif positional >= positional_threshold:
        classification = "positional"
    elif inward_frac >= 0.6:
        classification = "inward"
    elif inward_frac <= 0.25:
        classification = "outward"
    else:
        classification = "mixed"

    return {
        "inward_mass":    round(inward_frac, 4),
        "outward_mass":   round(1 - inward_frac, 4),
        "uniformity":     round(unif, 4),
        "positional":     round(positional, 4),
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Per-head residual contribution + cohesion scalar
# ---------------------------------------------------------------------------

def head_cohesion_scalar(
    attn_head: np.ndarray,    # (n, n)
    activations: np.ndarray,  # (n, d)  (the layer's input to attention)
    W_V_head: np.ndarray,     # (d, d_head) or (d_head, d); handled adaptively
    W_O_head: np.ndarray,     # (d_head, d) or (d, d_head)
    cluster_idx: np.ndarray,
) -> float:
    """
    Compute Σ_{i ∈ C} <Δ^(h)_i, x̄_C> where
        Δ^(h)_i = (A_h X W_V W_O)_i

    All weights in (d_in, d_out) convention. If the caller passes weights the
    other way round, this function transposes automatically using a shape
    check: W_V must map d → d_head and W_O must map d_head → d.
    """
    d = activations.shape[-1]

    # Normalize shapes
    if W_V_head.shape[0] != d:
        W_V_head = W_V_head.T
    if W_O_head.shape[1] != d:
        W_O_head = W_O_head.T

    V = activations @ W_V_head       # (n, d_head)
    AV = attn_head @ V                # (n, d_head)
    delta = AV @ W_O_head             # (n, d)

    X_C = activations[cluster_idx]
    x_bar = X_C.mean(axis=0)
    n_bar = np.linalg.norm(x_bar)
    if n_bar < 1e-12:
        return 0.0
    x_bar = x_bar / n_bar

    coh = float((delta[cluster_idx] @ x_bar).sum())
    return coh


# ---------------------------------------------------------------------------
# OV × cluster direction
# ---------------------------------------------------------------------------

def head_ov_cluster_alignment(
    W_V_head: np.ndarray,
    W_O_head: np.ndarray,
    cluster_direction: np.ndarray,
    top_k: int = 3,
) -> dict:
    """
    Eigendecompose the head's OV = W_V @ W_O (composed), take the top-k
    eigenvectors by |eigenvalue|, report overlap with the cluster direction.

    Uses the symmetric part for stability (Phase 2 convention).
    """
    d = cluster_direction.shape[0]

    # Normalize shape: we want OV_head as (d, d)
    if W_V_head.shape[0] != d:
        W_V_head = W_V_head.T
    if W_O_head.shape[1] != d:
        W_O_head = W_O_head.T

    OV = W_V_head @ W_O_head          # (d, d)
    OV_sym = 0.5 * (OV + OV.T)
    eigvals, eigvecs = np.linalg.eigh(OV_sym)
    order = np.argsort(-np.abs(eigvals))[:top_k]

    top = []
    for i in order:
        v = eigvecs[:, i]
        top.append({
            "eigval":   round(float(eigvals[i]), 4),
            "overlap":  round(float(abs(v @ cluster_direction)), 4),
            "sign":     "attractive" if eigvals[i] > 0 else "repulsive",
        })
    return {"top_eigenvectors": top}


# ---------------------------------------------------------------------------
# QK pattern analysis within cluster
# ---------------------------------------------------------------------------

def top_qk_pairs(
    A_h: np.ndarray,
    cluster_idx: np.ndarray,
    tokens: list,
    top_k: int = 5,
) -> list:
    """
    Report the top-k (i, j) attention pairs (i, j both in cluster) for one head.
    """
    A_cc = A_h[cluster_idx][:, cluster_idx]
    n = len(cluster_idx)
    flat = A_cc.ravel()
    order = np.argsort(-flat)[:min(top_k * 2, flat.size)]
    pairs = []
    seen = set()
    for f in order:
        i, j = int(f // n), int(f % n)
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        ti = int(cluster_idx[i])
        tj = int(cluster_idx[j])
        pairs.append({
            "i": ti, "j": tj,
            "tok_i": tokens[ti] if ti < len(tokens) else "?",
            "tok_j": tokens[tj] if tj < len(tokens) else "?",
            "attn":  round(float(A_cc[i, j]), 4),
        })
        if len(pairs) >= top_k:
            break
    return pairs


# ---------------------------------------------------------------------------
# Sinkhorn Fiedler restricted to cluster
# ---------------------------------------------------------------------------

def cluster_sinkhorn_fiedler(
    A_h: np.ndarray,
    cluster_idx: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Doubly-stochasticise A[cluster, cluster] via Sinkhorn, return Fiedler value
    (second smallest eigenvalue of the symmetrized Laplacian).
    """
    if len(cluster_idx) < 3:
        return float("nan")
    A = A_h[cluster_idx][:, cluster_idx].astype(np.float64)
    A = A + 1e-12
    for _ in range(max_iter):
        r = A.sum(axis=1, keepdims=True)
        A = A / np.maximum(r, 1e-30)
        c = A.sum(axis=0, keepdims=True)
        A = A / np.maximum(c, 1e-30)
        if abs(A.sum(axis=1).max() - 1.0) < tol:
            break
    A_sym = 0.5 * (A + A.T)
    D = np.diag(A_sym.sum(axis=1))
    L = D - A_sym
    eigs = np.linalg.eigvalsh(L)
    eigs.sort()
    return float(eigs[1]) if len(eigs) >= 2 else float("nan")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_heads(
    activations: np.ndarray,        # (n_layers, n, d)
    attentions: np.ndarray,         # (n_layers, n_heads, n, n)
    hdb_labels: list,
    trajectory: dict,
    tokens: list,
    weights: dict = None,           # {"W_V": ..., "W_O": ...} — per-head slicing
                                    # depends on model.  If missing, OV-based
                                    # metrics are skipped (attention-only output).
    n_heads: int = None,
) -> dict:
    """
    Full Group C.1 analysis.

    Returns
    -------
    dict with:
      per_layer : list of per-layer dicts, each with per_head list
      summary   : top contributing heads by cumulative cohesion across layers
    """
    chain = trajectory["chain"]
    if n_heads is None:
        n_heads = attentions.shape[1]

    # Per-head W_V, W_O slicing
    # ALBERT shared weights: weights["W_V"] shape (d, d). For per-head,
    # (d, n_heads, d_head). We try to auto-detect.
    W_V_all, W_O_all = None, None
    if weights:
        W_V_all = weights.get("W_V")
        W_O_all = weights.get("W_O")

    def _slice_head(W_full, h, split_dim):
        """
        Slice a per-head chunk from the full weight tensor.
        split_dim ∈ {"V", "O"}:
          V: split the second axis (d → n_heads × d_head) for W_V
          O: split the first axis (n_heads × d_head → d) for W_O
        """
        if W_full is None:
            return None
        if W_full.ndim == 3:  # already (d, n_heads, d_head) or similar
            return W_full[:, h] if split_dim == "V" else W_full[h]
        # 2D — infer d_head
        d = activations.shape[-1]
        if split_dim == "V":
            d_head = W_full.shape[1] // n_heads
            return W_full[:, h * d_head:(h + 1) * d_head]
        else:  # O
            d_head = W_full.shape[0] // n_heads
            return W_full[h * d_head:(h + 1) * d_head, :]

    per_layer_out = []
    cumulative_cohesion = np.zeros(n_heads)

    for layer, cid in chain:
        if layer >= attentions.shape[0] or layer >= activations.shape[0]:
            continue
        cluster_idx = np.where(hdb_labels[layer] == cid)[0]
        if len(cluster_idx) < 2:
            continue

        X = activations[layer]
        centroid = X[cluster_idx].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

        per_head = []
        for h in range(n_heads):
            A_h = attentions[layer, h]
            cls = classify_head(A_h, cluster_idx)
            cls["fiedler"] = round(cluster_sinkhorn_fiedler(A_h, cluster_idx), 4)
            cls["top_pairs"] = top_qk_pairs(A_h, cluster_idx, tokens, top_k=3)

            # OV-aware metrics if weights available
            if W_V_all is not None and W_O_all is not None:
                try:
                    WV_h = _slice_head(W_V_all, h, "V")
                    WO_h = _slice_head(W_O_all, h, "O")
                    coh = head_cohesion_scalar(A_h, X, WV_h, WO_h, cluster_idx)
                    cls["cohesion"] = round(coh, 4)
                    cumulative_cohesion[h] += coh
                    ov_align = head_ov_cluster_alignment(WV_h, WO_h, centroid)
                    cls["ov_alignment"] = ov_align
                except Exception as e:
                    cls["cohesion"] = None
                    cls["ov_alignment"] = {"error": str(e)}
            cls["head"] = int(h)
            per_head.append(cls)

        per_layer_out.append({
            "layer":    int(layer),
            "cluster_id": int(cid),
            "per_head": per_head,
        })

    # Rank heads by cumulative cohesion
    top_heads = []
    if cumulative_cohesion.any():
        order = np.argsort(-cumulative_cohesion)
        for h in order[:8]:
            top_heads.append({
                "head":     int(h),
                "cohesion": round(float(cumulative_cohesion[h]), 4),
            })

    return {
        "trajectory_id":       int(trajectory["id"]),
        "per_layer":           per_layer_out,
        "cumulative_cohesion": [round(float(x), 4) for x in cumulative_cohesion],
        "top_attractor_heads": top_heads,
    }


def save_head_contributions(result: dict, out_dir: Path,
                            tag: str = "primary") -> None:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"group_C1_heads_{tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")
