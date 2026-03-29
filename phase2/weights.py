"""
weights.py — Weight extraction and eigendecomposition for Phase 2.

Extracts the composed OV circuit (W_O @ W_V per head, summed) which is
the actual residual-stream-to-residual-stream linear map applied by the
value pathway.  Computes eigendecomposition with sign-separated subspace
projectors for attractive (Re λ > 0) and repulsive (Re λ < 0) directions.

Corrects Phase 1's analyze_value_eigenspectrum, which extracted W_V alone
— a non-square matrix for individual heads whose eigendecomposition is
not meaningful.

Key objects
-----------
OV_h  : per-head composed map  (d_model, d_model)
OV    : sum of OV_h across heads  (d_model, d_model)

Convention: row-vector.  For activation row vector x, the composed value
pathway output is x @ OV.  Eigenvalues are convention-independent.

Functions
---------
extract_ov_circuit      : per-head and total OV from model weights
eigendecompose          : full + symmetric-part eigendecomposition
build_subspace_projectors : orthogonal projectors onto attractive/repulsive
extract_qk_spectrum     : per-head spectral norm of W_Q^T W_K
rescale_matrix          : matrix exponential e^{-OV} for Section 9 coords
save_weight_decomposition / load_weight_decomposition
"""

import json
import numpy as np
from pathlib import Path
from scipy.linalg import schur, eigvals, svdvals, expm

from core.config import MODEL_CONFIGS


# ---------------------------------------------------------------------------
# OV extraction
# ---------------------------------------------------------------------------

def extract_ov_circuit(model, model_name: str) -> dict:
    """
    Extract per-head and total composed OV matrices from model weights.

    The OV matrix for head h is W_V_h @ W_O_h (row-vector convention):
    for a row vector x, the value pathway contribution is x @ OV_h.

    For nn.Linear layers (ALBERT, BERT):
      W_V stored as (d_out, d_in) — so W_V_h = W_V[h*d_h:(h+1)*d_h, :]
      W_O stored as (d_out, d_in) — so W_O_h = W_O[:, h*d_h:(h+1)*d_h]
      OV_h = W_V_h.T @ W_O_h.T  (d_model, d_model)

    For Conv1D layers (GPT-2):
      W_V = c_attn.weight[:, 2d//3:]  — W_V_h = W_V[:, h*d_h:(h+1)*d_h]
      W_O = c_proj.weight              — W_O_h = W_O[h*d_h:(h+1)*d_h, :]
      OV_h = W_V_h @ W_O_h  (d_model, d_model)

    Returns
    -------
    dict with:
      ov_total     : (d_model, d_model) ndarray — sum of per-head OV
      ov_per_head  : list of (d_model, d_model) ndarrays — one per head
      d_model      : int
      d_head       : int
      n_heads      : int
      layer_names  : list of str — "shared" for ALBERT, "layer_0" etc for others
      is_per_layer : bool — True if V varies per layer (GPT-2, BERT)

    For per-layer models, ov_total and ov_per_head are lists (one entry
    per layer); for ALBERT they are single values.
    """
    cfg = MODEL_CONFIGS[model_name]

    if cfg["is_albert"]:
        return _extract_albert_ov(model, model_name)
    elif "bert" in model_name:
        return _extract_bert_ov(model, model_name)
    elif "gpt2" in model_name:
        return _extract_gpt2_ov(model, model_name)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def _extract_albert_ov(model, model_name: str) -> dict:
    attn = model.encoder.albert_layer_groups[0].albert_layers[0].attention

    W_V = attn.value.weight.detach().cpu().float().numpy()   # (d_model, d_model)
    W_O = attn.dense.weight.detach().cpu().float().numpy()   # (d_model, d_model)

    d_model = W_V.shape[0]
    n_heads = attn.num_attention_heads
    d_head  = d_model // n_heads

    ov_per_head = []
    for h in range(n_heads):
        s = h * d_head
        e = s + d_head
        W_V_h = W_V[s:e, :]           # (d_head, d_model) — rows of V weight
        W_O_h = W_O[:, s:e]           # (d_model, d_head) — cols of O weight
        # Row-vector convention: x @ OV_h
        OV_h = W_V_h.T @ W_O_h.T     # (d_model, d_model)
        ov_per_head.append(OV_h)

    ov_total = sum(ov_per_head)

    return {
        "ov_total":     ov_total,
        "ov_per_head":  ov_per_head,
        "d_model":      d_model,
        "d_head":       d_head,
        "n_heads":      n_heads,
        "layer_names":  ["shared"],
        "is_per_layer": False,
    }


def _extract_bert_ov(model, model_name: str) -> dict:
    all_ov_total    = []
    all_ov_per_head = []
    layer_names     = []

    for i, layer in enumerate(model.encoder.layer):
        W_V = layer.attention.self.value.weight.detach().cpu().float().numpy()
        W_O = layer.attention.output.dense.weight.detach().cpu().float().numpy()

        d_model = W_V.shape[0]
        n_heads = layer.attention.self.num_attention_heads
        d_head  = d_model // n_heads

        per_head = []
        for h in range(n_heads):
            s = h * d_head
            e = s + d_head
            W_V_h = W_V[s:e, :]
            W_O_h = W_O[:, s:e]
            OV_h  = W_V_h.T @ W_O_h.T
            per_head.append(OV_h)

        all_ov_total.append(sum(per_head))
        all_ov_per_head.append(per_head)
        layer_names.append(f"layer_{i}")

    return {
        "ov_total":     all_ov_total,
        "ov_per_head":  all_ov_per_head,
        "d_model":      d_model,
        "d_head":       d_head,
        "n_heads":      n_heads,
        "layer_names":  layer_names,
        "is_per_layer": True,
    }


def _extract_gpt2_ov(model, model_name: str) -> dict:
    all_ov_total    = []
    all_ov_per_head = []
    layer_names     = []

    for i, block in enumerate(model.h):
        # Conv1D stores weights as (in_features, out_features)
        c_attn_w = block.attn.c_attn.weight.detach().cpu().float().numpy()
        c_proj_w = block.attn.c_proj.weight.detach().cpu().float().numpy()

        d_total = c_attn_w.shape[1]
        d_model = c_attn_w.shape[0]
        # Q, K, V each get d_total // 3 columns
        W_V = c_attn_w[:, 2 * d_total // 3:]   # (d_model, d_model)
        W_O = c_proj_w                           # (d_model, d_model)

        n_heads = block.attn.num_heads
        d_head  = d_model // n_heads

        per_head = []
        for h in range(n_heads):
            s = h * d_head
            e = s + d_head
            W_V_h = W_V[:, s:e]            # (d_model, d_head) — cols of V
            W_O_h = W_O[s:e, :]            # (d_head, d_model) — rows of O
            # Row-vector convention: x @ OV_h = x @ W_V_h @ W_O_h
            OV_h = W_V_h @ W_O_h           # (d_model, d_model)
            per_head.append(OV_h)

        all_ov_total.append(sum(per_head))
        all_ov_per_head.append(per_head)
        layer_names.append(f"layer_{i}")

    return {
        "ov_total":     all_ov_total,
        "ov_per_head":  all_ov_per_head,
        "d_model":      d_model,
        "d_head":       d_head,
        "n_heads":      n_heads,
        "layer_names":  layer_names,
        "is_per_layer": True,
    }


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def eigendecompose(OV: np.ndarray) -> dict:
    """
    Eigendecompose a (d_model, d_model) OV matrix via two methods.

    Method 1 — Ordered real Schur form.
      Numerically stable invariant subspace decomposition.  The Schur
      vectors spanning the attractive (Re λ > 0) subspace form an
      orthonormal basis regardless of V's normality.

    Method 2 — Symmetric part (OV + OV^T) / 2.
      Always symmetric, orthogonal eigenvectors, numerically clean.
      Discards the antisymmetric (rotational) component of OV.
      When the two methods disagree, rotation matters.

    Returns
    -------
    dict with:
      eigenvalues       : (d,) complex — from full OV
      eig_real          : (d,) float — Re(eigenvalues)
      eig_imag          : (d,) float — Im(eigenvalues)
      frac_attractive   : float — fraction with Re > 0
      frac_repulsive    : float — fraction with Re < 0
      frac_complex      : float — fraction with |Im| > 0.01 * (|Re| + 1e-8)

      schur_T           : (d, d) upper quasi-triangular (real Schur form)
      schur_Z           : (d, d) orthogonal Schur vectors
      schur_n_attractive: int — dimension of attractive invariant subspace
      schur_cond        : float — condition indicator (norm ratio)

      sym_eigenvalues   : (d,) float — eigenvalues of symmetric part
      sym_eigenvectors  : (d, d) float — columns are eigenvectors
      sym_frac_attractive : float
      sym_frac_repulsive  : float

      agree             : bool — do the two methods agree on sign fractions
                          within 10% tolerance?
    """
    d = OV.shape[0]

    # --- Full eigenvalues ---
    eigs     = eigvals(OV)
    eig_real = np.real(eigs)
    eig_imag = np.imag(eigs)

    frac_pos     = float((eig_real > 0).mean())
    frac_neg     = float((eig_real < 0).mean())
    is_complex   = np.abs(eig_imag) > 0.01 * (np.abs(eig_real) + 1e-8)
    frac_complex = float(is_complex.mean())

    # --- Ordered real Schur form ---
    # sort='rhp' puts eigenvalues with Re > 0 in the upper-left block.
    T, Z, sdim = schur(OV, output='real', sort='rhp')
    n_attractive = int(sdim)

    # Condition indicator: ratio of norms of attractive vs repulsive blocks.
    # Not a true condition number but flags degenerate cases.
    if n_attractive > 0 and n_attractive < d:
        norm_a = np.linalg.norm(T[:n_attractive, :n_attractive])
        norm_r = np.linalg.norm(T[n_attractive:, n_attractive:])
        cond = float(norm_a / (norm_r + 1e-12))
    else:
        cond = float("inf")

    # --- Symmetric part ---
    S       = (OV + OV.T) / 2.0
    sym_eig_vals, sym_eig_vecs = np.linalg.eigh(S)  # sorted ascending

    sym_frac_pos = float((sym_eig_vals > 0).mean())
    sym_frac_neg = float((sym_eig_vals < 0).mean())

    # --- Agreement check ---
    agree = (abs(frac_pos - sym_frac_pos) < 0.10 and
             abs(frac_neg - sym_frac_neg) < 0.10)

    return {
        "eigenvalues":         eigs,
        "eig_real":            eig_real,
        "eig_imag":            eig_imag,
        "frac_attractive":     frac_pos,
        "frac_repulsive":      frac_neg,
        "frac_complex":        frac_complex,

        "schur_T":             T,
        "schur_Z":             Z,
        "schur_n_attractive":  n_attractive,
        "schur_cond":          cond,

        "sym_eigenvalues":     sym_eig_vals,
        "sym_eigenvectors":    sym_eig_vecs,
        "sym_frac_attractive": sym_frac_pos,
        "sym_frac_repulsive":  sym_frac_neg,

        "agree":               agree,
    }


def eigendecompose_per_head(ov_per_head: list) -> list:
    """Run eigendecompose on each per-head OV matrix."""
    return [eigendecompose(OV_h) for OV_h in ov_per_head]


# ---------------------------------------------------------------------------
# Subspace projectors
# ---------------------------------------------------------------------------

def build_subspace_projectors(decomp: dict) -> dict:
    """
    Build orthogonal projectors onto attractive and repulsive subspaces.

    Two sets of projectors, one from each decomposition method:

    Schur-based (full OV):
      P_attract = Z_+ @ Z_+^T  where Z_+ are the first sdim Schur vectors
      P_repulse = Z_- @ Z_-^T  where Z_- are the remaining Schur vectors
      These project onto the invariant subspaces of OV.

    Symmetric-part:
      P_attract = U_+ @ U_+^T  where U_+ are eigenvectors with λ > 0
      P_repulse = U_- @ U_-^T  where U_- are eigenvectors with λ < 0
      These are true orthogonal projectors (complementary up to the
      null eigenvalue subspace).

    All projectors are (d_model, d_model) and act in row-vector convention:
      x_projected = x @ P

    Returns
    -------
    dict with keys:
      schur_attract, schur_repulse  : (d, d) ndarrays
      sym_attract, sym_repulse      : (d, d) ndarrays
      schur_dim_attract, schur_dim_repulse : int
      sym_dim_attract, sym_dim_repulse     : int
    """
    Z   = decomp["schur_Z"]
    n_a = decomp["schur_n_attractive"]
    d   = Z.shape[0]

    # Schur projectors
    Z_plus  = Z[:, :n_a]                 # (d, n_a)
    Z_minus = Z[:, n_a:]                 # (d, d - n_a)
    P_schur_a = Z_plus @ Z_plus.T        # (d, d)
    P_schur_r = Z_minus @ Z_minus.T      # (d, d)

    # Symmetric-part projectors
    sym_vals = decomp["sym_eigenvalues"]
    sym_vecs = decomp["sym_eigenvectors"]  # columns are eigenvectors

    pos_mask = sym_vals > 0
    neg_mask = sym_vals < 0
    U_plus   = sym_vecs[:, pos_mask]       # (d, n_pos)
    U_minus  = sym_vecs[:, neg_mask]       # (d, n_neg)
    P_sym_a  = U_plus @ U_plus.T           # (d, d)
    P_sym_r  = U_minus @ U_minus.T         # (d, d)

    return {
        "schur_attract":      P_schur_a,
        "schur_repulse":      P_schur_r,
        "sym_attract":        P_sym_a,
        "sym_repulse":        P_sym_r,
        "schur_dim_attract":  n_a,
        "schur_dim_repulse":  d - n_a,
        "sym_dim_attract":    int(pos_mask.sum()),
        "sym_dim_repulse":    int(neg_mask.sum()),
    }


# ---------------------------------------------------------------------------
# QK spectrum
# ---------------------------------------------------------------------------

def extract_qk_spectrum(model, model_name: str) -> dict:
    """
    Compute the spectral norm of W_Q^T W_K per head per layer.

    This is the effective β — the coupling strength in the softmax
    exponent.  Larger spectral norm means sharper attention, stronger
    attractive dynamics.

    Returns
    -------
    dict with:
      qk_spectral_norms : list (per layer) of lists (per head) of floats
      layer_names        : list of str
    """
    cfg = MODEL_CONFIGS[model_name]

    if cfg["is_albert"]:
        return _qk_albert(model)
    elif "bert" in model_name:
        return _qk_bert(model)
    elif "gpt2" in model_name:
        return _qk_gpt2(model)
    else:
        return {"qk_spectral_norms": [], "layer_names": []}


def _qk_albert(model) -> dict:
    attn = model.encoder.albert_layer_groups[0].albert_layers[0].attention
    W_Q = attn.query.weight.detach().cpu().float().numpy()   # (d, d)
    W_K = attn.key.weight.detach().cpu().float().numpy()     # (d, d)

    d_model = W_Q.shape[0]
    n_heads = attn.num_attention_heads
    d_head  = d_model // n_heads

    norms = []
    for h in range(n_heads):
        s, e = h * d_head, (h + 1) * d_head
        # nn.Linear: map is x @ W.T, so per-head Q_h = W_Q[s:e, :]
        Q_h = W_Q[s:e, :]  # (d_head, d_model)
        K_h = W_K[s:e, :]  # (d_head, d_model)
        # Q^T K in the paper's column convention: Q_h.T @ K_h
        # Spectral norm = largest singular value
        QK = Q_h @ K_h.T   # (d_head, d_head) — this is Q_h @ K_h^T
        norms.append(float(svdvals(QK)[0]))

    return {
        "qk_spectral_norms": [norms],   # single "layer"
        "layer_names":       ["shared"],
    }


def _qk_bert(model) -> dict:
    all_norms   = []
    layer_names = []
    for i, layer in enumerate(model.encoder.layer):
        W_Q = layer.attention.self.query.weight.detach().cpu().float().numpy()
        W_K = layer.attention.self.key.weight.detach().cpu().float().numpy()

        d_model = W_Q.shape[0]
        n_heads = layer.attention.self.num_attention_heads
        d_head  = d_model // n_heads

        norms = []
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            Q_h = W_Q[s:e, :]
            K_h = W_K[s:e, :]
            QK  = Q_h @ K_h.T
            norms.append(float(svdvals(QK)[0]))
        all_norms.append(norms)
        layer_names.append(f"layer_{i}")

    return {"qk_spectral_norms": all_norms, "layer_names": layer_names}


def _qk_gpt2(model) -> dict:
    all_norms   = []
    layer_names = []
    for i, block in enumerate(model.h):
        c_attn_w = block.attn.c_attn.weight.detach().cpu().float().numpy()
        d_total  = c_attn_w.shape[1]
        d_model  = c_attn_w.shape[0]
        d_third  = d_total // 3

        W_Q = c_attn_w[:, :d_third]            # (d_model, d_model)
        W_K = c_attn_w[:, d_third:2*d_third]   # (d_model, d_model)

        n_heads = block.attn.num_heads
        d_head  = d_model // n_heads

        norms = []
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            # Conv1D: map is x @ W, so per-head Q_h = W_Q[:, s:e]
            Q_h = W_Q[:, s:e]   # (d_model, d_head)
            K_h = W_K[:, s:e]   # (d_model, d_head)
            # Effective QK: Q_h^T @ K_h
            QK = Q_h.T @ K_h    # (d_head, d_head)
            norms.append(float(svdvals(QK)[0]))
        all_norms.append(norms)
        layer_names.append(f"layer_{i}")

    return {"qk_spectral_norms": all_norms, "layer_names": layer_names}


# ---------------------------------------------------------------------------
# Rescaled-coordinate matrix
# ---------------------------------------------------------------------------

def rescale_matrix(OV: np.ndarray) -> np.ndarray:
    """
    Compute e^{-OV} for the paper's Section 9 rescaled coordinates.

    In the rescaled frame z_i(t) = e^{-tV} x_i(t), the clustering
    geometry may be cleaner because V's own effect is factored out.

    For ALBERT at iteration t, the rescaled activation is:
      z_i(t) = (e^{-OV})^t @ x_i(t)   (column vector)
    or in row convention:
      z_i(t) = x_i(t) @ ((e^{-OV})^t).T

    Returns
    -------
    (d_model, d_model) ndarray — e^{-OV}
    """
    return expm(-OV)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_weight_decomposition(
    ov_data: dict,
    decomps: dict,
    projectors: dict,
    qk_data: dict,
    save_dir: Path,
    model_name: str,
) -> None:
    """
    Persist weight decomposition to disk for offline Phase 2 analysis.

    Writes:
      ov_weights_{model}.npz   — OV matrices (total + per-head)
      ov_decomp_{model}.npz    — eigenvalues, Schur vectors, sym eigenvectors
      ov_projectors_{model}.npz — subspace projectors
      ov_summary_{model}.json  — scalar summaries for reporting

    The decomps and projectors arguments may be:
      - A single dict (ALBERT shared weights)
      - A list of dicts (per-layer models like GPT-2)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = model_name.replace("/", "_")

    # --- OV weights ---
    ov_arrays = {}
    if ov_data["is_per_layer"]:
        for i, name in enumerate(ov_data["layer_names"]):
            ov_arrays[f"ov_total_{name}"] = ov_data["ov_total"][i]
            for h, ov_h in enumerate(ov_data["ov_per_head"][i]):
                ov_arrays[f"ov_head{h}_{name}"] = ov_h
    else:
        ov_arrays["ov_total_shared"] = ov_data["ov_total"]
        for h, ov_h in enumerate(ov_data["ov_per_head"]):
            ov_arrays[f"ov_head{h}_shared"] = ov_h
    np.savez_compressed(save_dir / f"ov_weights_{stem}.npz", **ov_arrays)

    # --- Decomposition ---
    decomp_arrays = {}
    _save_decomp_arrays(decomps, ov_data, decomp_arrays)
    np.savez_compressed(save_dir / f"ov_decomp_{stem}.npz", **decomp_arrays)

    # --- Projectors ---
    proj_arrays = {}
    _save_projector_arrays(projectors, ov_data, proj_arrays)
    np.savez_compressed(save_dir / f"ov_projectors_{stem}.npz", **proj_arrays)

    # --- JSON summary ---
    summary = _build_summary(ov_data, decomps, projectors, qk_data)
    with open(save_dir / f"ov_summary_{stem}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Phase 2 weights saved to {save_dir}/")


def _save_decomp_arrays(decomps, ov_data, out):
    """Flatten decomposition dicts into arrays for npz."""
    if ov_data["is_per_layer"]:
        for i, name in enumerate(ov_data["layer_names"]):
            d = decomps[i]
            out[f"eig_real_{name}"]     = d["eig_real"]
            out[f"eig_imag_{name}"]     = d["eig_imag"]
            out[f"schur_Z_{name}"]      = d["schur_Z"]
            out[f"sym_evals_{name}"]    = d["sym_eigenvalues"]
            out[f"sym_evecs_{name}"]    = d["sym_eigenvectors"]
    else:
        out["eig_real_shared"]     = decomps["eig_real"]
        out["eig_imag_shared"]     = decomps["eig_imag"]
        out["schur_Z_shared"]      = decomps["schur_Z"]
        out["sym_evals_shared"]    = decomps["sym_eigenvalues"]
        out["sym_evecs_shared"]    = decomps["sym_eigenvectors"]


def _save_projector_arrays(projectors, ov_data, out):
    """Flatten projector dicts into arrays for npz."""
    if ov_data["is_per_layer"]:
        for i, name in enumerate(ov_data["layer_names"]):
            p = projectors[i]
            out[f"schur_attract_{name}"] = p["schur_attract"]
            out[f"schur_repulse_{name}"] = p["schur_repulse"]
            out[f"sym_attract_{name}"]   = p["sym_attract"]
            out[f"sym_repulse_{name}"]   = p["sym_repulse"]
    else:
        out["schur_attract_shared"] = projectors["schur_attract"]
        out["schur_repulse_shared"] = projectors["schur_repulse"]
        out["sym_attract_shared"]   = projectors["sym_attract"]
        out["sym_repulse_shared"]   = projectors["sym_repulse"]


def _build_summary(ov_data, decomps, projectors, qk_data) -> dict:
    """Build JSON-serialisable summary for reporting."""
    summary = {
        "model":       None,  # filled by caller
        "d_model":     ov_data["d_model"],
        "d_head":      ov_data["d_head"],
        "n_heads":     ov_data["n_heads"],
        "is_per_layer": ov_data["is_per_layer"],
        "layers":      {},
    }

    def _layer_summary(decomp, proj, qk_norms=None):
        s = {
            "frac_attractive":     decomp["frac_attractive"],
            "frac_repulsive":      decomp["frac_repulsive"],
            "frac_complex":        decomp["frac_complex"],
            "sym_frac_attractive": decomp["sym_frac_attractive"],
            "sym_frac_repulsive":  decomp["sym_frac_repulsive"],
            "methods_agree":       decomp["agree"],
            "schur_cond":          decomp["schur_cond"],
            "schur_dim_attract":   proj["schur_dim_attract"],
            "schur_dim_repulse":   proj["schur_dim_repulse"],
            "sym_dim_attract":     proj["sym_dim_attract"],
            "sym_dim_repulse":     proj["sym_dim_repulse"],
            "ov_spectral_norm":    float(svdvals(
                # need OV matrix — reconstruct from decomp eigenvalues
                # Actually just use the eigenvalue magnitudes
                np.diag(np.abs(decomp["eigenvalues"]))
            )[0]) if "eigenvalues" in decomp else None,
        }
        if qk_norms is not None:
            s["qk_spectral_norms_per_head"] = qk_norms
            s["qk_spectral_norm_mean"]      = float(np.mean(qk_norms))
        return s

    if ov_data["is_per_layer"]:
        for i, name in enumerate(ov_data["layer_names"]):
            qk = qk_data["qk_spectral_norms"][i] if i < len(qk_data["qk_spectral_norms"]) else None
            summary["layers"][name] = _layer_summary(decomps[i], projectors[i], qk)
    else:
        qk = qk_data["qk_spectral_norms"][0] if qk_data["qk_spectral_norms"] else None
        summary["layers"]["shared"] = _layer_summary(decomps, projectors, qk)

    return summary


def load_weight_decomposition(save_dir: Path, model_name: str) -> dict:
    """
    Load saved weight decomposition for offline analysis.

    Returns
    -------
    dict with:
      summary    : dict from JSON
      ov_total   : ndarray or list of ndarrays
      projectors : dict or list of dicts with schur_attract/repulse, sym_attract/repulse
      decomp     : dict or list of dicts with eig_real, sym_eigenvalues, etc.
    """
    save_dir = Path(save_dir)
    stem     = model_name.replace("/", "_")

    with open(save_dir / f"ov_summary_{stem}.json") as f:
        summary = json.load(f)

    ov_data   = np.load(save_dir / f"ov_weights_{stem}.npz")
    dec_data  = np.load(save_dir / f"ov_decomp_{stem}.npz")
    proj_data = np.load(save_dir / f"ov_projectors_{stem}.npz")

    is_per_layer = summary["is_per_layer"]

    if not is_per_layer:
        return {
            "summary":    summary,
            "ov_total":   ov_data["ov_total_shared"],
            "projectors": {
                "schur_attract": proj_data["schur_attract_shared"],
                "schur_repulse": proj_data["schur_repulse_shared"],
                "sym_attract":   proj_data["sym_attract_shared"],
                "sym_repulse":   proj_data["sym_repulse_shared"],
            },
            "decomp": {
                "eig_real":         dec_data["eig_real_shared"],
                "eig_imag":         dec_data["eig_imag_shared"],
                "sym_eigenvalues":  dec_data["sym_evals_shared"],
                "sym_eigenvectors": dec_data["sym_evecs_shared"],
            },
        }
    else:
        layer_names = [k for k in summary["layers"]]
        ov_totals   = [ov_data[f"ov_total_{n}"] for n in layer_names]
        proj_list   = [{
            "schur_attract": proj_data[f"schur_attract_{n}"],
            "schur_repulse": proj_data[f"schur_repulse_{n}"],
            "sym_attract":   proj_data[f"sym_attract_{n}"],
            "sym_repulse":   proj_data[f"sym_repulse_{n}"],
        } for n in layer_names]
        dec_list = [{
            "eig_real":         dec_data[f"eig_real_{n}"],
            "eig_imag":         dec_data[f"eig_imag_{n}"],
            "sym_eigenvalues":  dec_data[f"sym_evals_{n}"],
            "sym_eigenvectors": dec_data[f"sym_evecs_{n}"],
        } for n in layer_names]

        return {
            "summary":    summary,
            "ov_total":   ov_totals,
            "projectors": proj_list,
            "decomp":     dec_list,
        }


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def analyze_weights(model, model_name: str, save_dir: Path) -> dict:
    """
    Full weight analysis pipeline.  Call once per model.

    1. Extract composed OV circuit
    2. Eigendecompose (full + symmetric)
    3. Build subspace projectors
    4. Extract QK spectrum
    5. Save everything

    Returns the ov_data dict augmented with decomposition and projector
    results for immediate use by trajectory.py.
    """
    print(f"  Phase 2: extracting OV circuit for {model_name}...")
    ov_data = extract_ov_circuit(model, model_name)

    if ov_data["is_per_layer"]:
        decomps    = [eigendecompose(ov) for ov in ov_data["ov_total"]]
        projectors = [build_subspace_projectors(d) for d in decomps]
    else:
        decomps    = eigendecompose(ov_data["ov_total"])
        projectors = build_subspace_projectors(decomps)

    qk_data = extract_qk_spectrum(model, model_name)

    save_weight_decomposition(ov_data, decomps, projectors, qk_data,
                              save_dir, model_name)

    # Augment ov_data for downstream use
    ov_data["decomps"]    = decomps
    ov_data["projectors"] = projectors
    ov_data["qk_data"]    = qk_data

    # Print summary
    _print_weight_summary(ov_data, decomps, model_name)

    return ov_data


def _print_weight_summary(ov_data, decomps, model_name):
    """Print concise terminal summary of weight decomposition."""
    print(f"\n  Weight decomposition: {model_name}")
    print(f"    d_model={ov_data['d_model']}  n_heads={ov_data['n_heads']}  "
          f"d_head={ov_data['d_head']}")

    def _show(name, d):
        agree_str = "✓" if d["agree"] else "✗"
        print(f"    {name:12s}  "
              f"attract={d['frac_attractive']:.2f}  "
              f"repulse={d['frac_repulsive']:.2f}  "
              f"complex={d['frac_complex']:.2f}  "
              f"sym_agree={agree_str}  "
              f"schur_cond={d['schur_cond']:.1f}")

    if ov_data["is_per_layer"]:
        for i, name in enumerate(ov_data["layer_names"]):
            _show(name, decomps[i])
    else:
        _show("shared", decomps)
