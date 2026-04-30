"""
io.py — Phase 5 artifact loaders.

All cross-phase artifact discovery lives here. Each loader returns a single
dict with documented keys; missing artifacts resolve to sensible defaults
(empty dict or None) so downstream modules can gracefully skip groups they
can't run.

Conventions
-----------
- Phase 1 run dirs:  results/phase1/<model_stem>_iter_<prompt>/
                     containing metrics.json, activations.npz, attentions.npz,
                     clusters.npz, centroid_trajectories.npz
- Phase 2 dir:       results/phase2/                (shared across prompts)
                     containing ov_projectors_<model_stem>.npz
- Phase 2i dir:      results/phase2i/
                     containing sym_antisym results per model
- Phase 3 dir:       checkpoints/<model>/final/     (crosscoder ckpt)
                     activation_cache/<model>/eval_prompts/  (PromptActivationStore)
- Phase 4 dir:       results/phase4/<model_stem>_<ts>/
                     containing t1/t2/t3 results
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Phase 1 artifacts — per prompt
# ---------------------------------------------------------------------------

def load_phase1_run(run_dir: Path) -> dict:
    """
    Load one Phase 1 run directory (v2 split-file format).

    v2 layout (written by io_utils.save_run):
      geometry.json         — model, prompt, n_layers, n_tokens, d_model
      trajectory.json       — cluster_tracking (trajectories + events) + plateau_layers
      tokens.txt            — "  i  token\n" per line
      activations.npz       — key "activations": (n_layers, n_tokens, d_model) float32
      attentions.npz        — key "attentions":  (n_layers, n_heads, n_tokens, n_tokens) float32
      clusters.npz          — keys hdbscan_labels_L{i}: (n_tokens,) int32
      centroid_trajectories.npz — keys traj_{id}: (lifespan, d) float32

    Returns
    -------
    dict with keys:
      tokens         : list of str
      prompt_key     : str
      trajectories   : list of trajectory dicts from cluster_tracking
      events         : list of per-transition event dicts
      activations    : (n_layers, n_tokens, d_model) float32, or None
      attentions     : (n_layers, n_heads, n_tokens, n_tokens) float32, or None
      hdbscan_labels : list of (n_tokens,) int32 per layer, or None
      centroid_trajs : dict {trajectory_id (int): (lifespan, d) float32}
      run_dir        : str
    """
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir)}

    # --- geometry.json: prompt name, shape metadata ---
    with open(run_dir / "geometry.json") as f:
        geo = json.load(f)
    out["prompt_key"] = geo.get("prompt", run_dir.name)

    # --- tokens.txt: "  i  token\n" ---
    tokens_path = run_dir / "tokens.txt"
    if tokens_path.exists():
        tokens = []
        with open(tokens_path) as f:
            for line in f:
                parts = line.rstrip("\n").split(None, 1)
                tokens.append(parts[1] if len(parts) == 2 else "")
        out["tokens"] = tokens
    else:
        out["tokens"] = []

    # --- trajectory.json: cluster_tracking ---
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        with open(traj_path) as f:
            traj_data = json.load(f)
        tracking = traj_data.get("cluster_tracking", {})
    else:
        tracking = {}
    out["trajectories"] = tracking.get("trajectories", [])
    out["events"]       = tracking.get("events", [])

    # --- activations.npz ---
    act_path = run_dir / "activations.npz"
    out["activations"] = (
        np.load(act_path)["activations"] if act_path.exists() else None
    )

    # --- attentions.npz ---
    att_path = run_dir / "attentions.npz"
    out["attentions"] = (
        np.load(att_path)["attentions"] if att_path.exists() else None
    )

    # --- clusters.npz: hdbscan_labels_L{i} ---
    clu_path = run_dir / "clusters.npz"
    if clu_path.exists():
        data = np.load(clu_path)
        layer_idxs = sorted(
            int(k.split("_L")[1]) for k in data.files
            if k.startswith("hdbscan_labels_L")
        )
        out["hdbscan_labels"] = [
            data[f"hdbscan_labels_L{i}"] for i in layer_idxs
        ]
    else:
        out["hdbscan_labels"] = None

    # --- centroid_trajectories.npz: traj_{id} ---
    ct_path = run_dir / "centroid_trajectories.npz"
    if ct_path.exists():
        data = np.load(ct_path)
        out["centroid_trajs"] = {
            int(k.split("_")[1]): data[k] for k in data.files
        }
    else:
        out["centroid_trajs"] = {}

    return out


def find_phase1_runs(phase1_dir: Path, model_stem: str) -> dict:
    """
    Enumerate Phase 1 v2 run directories matching a model stem.

    Reads the prompt name from geometry.json (canonical).
    Falls back to inferring it from the directory name.

    Returns
    -------
    dict {prompt_key: run_dir_path}
    """
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        return {}

    out = {}
    for run_dir in phase1_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if model_stem not in run_dir.name:
            continue
        geo_path = run_dir / "geometry.json"
        if geo_path.exists():
            try:
                with open(geo_path) as f:
                    pk = json.load(f).get("prompt")
                if pk:
                    out[pk] = run_dir
                    continue
            except Exception:
                pass
        # Fallback: infer from directory name
        name = run_dir.name
        pk = name.split("iter_", 1)[-1] if "iter_" in name else name
        out[pk] = run_dir
    return out

# ---------------------------------------------------------------------------
# Phase 2 artifacts — V projectors (shared across prompts)
# ---------------------------------------------------------------------------

def load_phase2_projectors(
    phase2_dir: Path,
    model_stem: str,
    k_top: Optional[int] = None,
) -> dict:
    """
    Load V eigenspectrum and build attractive/repulsive projectors.

    If k_top is None, uses all eigenvectors; otherwise keeps top-k by |eigval|.

    Returns
    -------
    dict with keys:
      eigenvalues    : (n,) complex or real array (may include imaginary parts
                       if stored from Schur; we take real parts for subspace
                       construction — Phase 2i confirmed globally rotation-neutral)
      eigenvectors   : (d, n) array
      attractive_P   : (d, d) dense projector onto positive-eigval subspace
      repulsive_P    : (d, d) dense projector onto negative-eigval subspace
      U_att          : (d, n_pos) basis
      U_rep          : (d, n_neg) basis
      eigvals_pos    : positive eigenvalues (sorted desc)
      eigvals_neg    : negative eigenvalues (sorted asc, i.e. most-negative first)
      path           : path the projectors were loaded from, or None if missing
    """
    phase2_dir = Path(phase2_dir)
    out = {
        "eigenvalues": None, "eigenvectors": None,
        "attractive_P": None, "repulsive_P": None,
        "U_att": None, "U_rep": None,
        "eigvals_pos": None, "eigvals_neg": None,
        "path": None,
    }

    candidates = [
        phase2_dir / f"ov_projectors_{model_stem}.npz",
        *phase2_dir.glob(f"*projector*{model_stem}*.npz"),
        *phase2_dir.glob(f"*{model_stem}*projector*.npz"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return out

    data = np.load(path)
    eigvecs = data.get("eigenvectors", data.get("U"))
    eigvals = data.get("eigenvalues", data.get("S"))
    if eigvecs is None or eigvals is None:
        return out

    # Use real parts; Phase 2i confirmed the rotational component is globally
    # negligible for ALBERT-xlarge. Local rotational effects are tested
    # separately by v_alignment.rotational_local_test.
    eigvals = np.asarray(eigvals)
    if np.iscomplexobj(eigvals):
        eigvals = eigvals.real
    eigvecs = np.asarray(eigvecs)
    if np.iscomplexobj(eigvecs):
        eigvecs = eigvecs.real

    if k_top is not None:
        idx = np.argsort(np.abs(eigvals))[::-1][:k_top]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]

    pos_mask = eigvals > 0
    neg_mask = eigvals < 0

    U_att = eigvecs[:, pos_mask].astype(np.float32)
    U_rep = eigvecs[:, neg_mask].astype(np.float32)

    out.update({
        "eigenvalues":  eigvals.astype(np.float32),
        "eigenvectors": eigvecs.astype(np.float32),
        "U_att":        U_att,
        "U_rep":        U_rep,
        "attractive_P": (U_att @ U_att.T) if U_att.size else None,
        "repulsive_P":  (U_rep @ U_rep.T) if U_rep.size else None,
        "eigvals_pos":  np.sort(eigvals[pos_mask])[::-1].astype(np.float32),
        "eigvals_neg":  np.sort(eigvals[neg_mask]).astype(np.float32),
        "path":         str(path),
    })
    return out


def load_phase2_weights(phase2_dir: Path, model_stem: str) -> dict:
    """
    Load per-head OV and QK matrices if available.

    Expected file: <phase2_dir>/weights_<model_stem>.npz with keys
        W_V, W_O, W_Q, W_K (shared for ALBERT, per-layer otherwise).

    Returns dict with whatever is present, or empty dict.
    """
    phase2_dir = Path(phase2_dir)
    for candidate in (
        phase2_dir / f"weights_{model_stem}.npz",
        *phase2_dir.glob(f"*weights*{model_stem}*.npz"),
    ):
        if candidate.exists():
            data = np.load(candidate)
            return {k: data[k] for k in data.files}
    return {}


# ---------------------------------------------------------------------------
# Phase 2i artifacts — symmetric/antisymmetric decomposition
# ---------------------------------------------------------------------------

def load_phase2i(phase2i_dir: Path, model_stem: str) -> dict:
    """
    Load Phase 2i S/A decomposition artifacts for local rotational testing.

    Returns dict with keys (whichever exist):
      V_sym       : (d, d) symmetric part of V_eff
      V_asym      : (d, d) antisymmetric part
      schur_T     : (d, d) real Schur T matrix
      schur_Z     : (d, d) Schur orthogonal basis
      rotational_blocks : list of dicts describing 2D invariant blocks
    """
    phase2i_dir = Path(phase2i_dir)
    out = {}
    if not phase2i_dir.exists():
        return out

    for candidate in phase2i_dir.glob(f"*{model_stem}*.npz"):
        data = np.load(candidate, allow_pickle=True)
        for k in data.files:
            if k not in out:
                out[k] = data[k]
    return out


# ---------------------------------------------------------------------------
# Phase 3 artifacts — crosscoder + prompt store
# ---------------------------------------------------------------------------

def load_phase3(
    checkpoint_dir: Path,
    cache_dir: Path,
    device: str = "cpu",
) -> dict:
    """
    Load the Phase 3 crosscoder and prompt activation store.

    Parameters
    ----------
    checkpoint_dir : typically checkpoints/<model>/final
    cache_dir      : typically activation_cache/<model>
    device         : torch device for the crosscoder

    Returns
    -------
    dict with keys:
      crosscoder     : Crosscoder module or None
      layer_indices  : list of layer indices the crosscoder spans
      prompt_store   : PromptActivationStore or None
      cfg            : crosscoder config dict (if available)
    """
    out = {
        "crosscoder": None, "layer_indices": [],
        "prompt_store": None, "cfg": None,
    }

    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "config.json").exists():
        try:
            from phase3.training import load_trained_crosscoder
            out["crosscoder"] = load_trained_crosscoder(
                checkpoint_dir, device=device,
            )
            with open(checkpoint_dir / "config.json") as f:
                cfg = json.load(f)
            out["cfg"] = cfg
            out["layer_indices"] = cfg.get("layer_indices", [])
        except Exception as e:
            print(f"[phase3] crosscoder load failed: {e}")

    cache_dir = Path(cache_dir)
    eval_dir = cache_dir / "eval_prompts"
    if eval_dir.exists():
        try:
            from phase3.data import PromptActivationStore
            store = PromptActivationStore()
            store.load(eval_dir)
            out["prompt_store"] = store
        except Exception as e:
            print(f"[phase3] prompt_store load failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Phase 4 artifacts — LDA directions, feature-cluster MI, AE bottleneck
# ---------------------------------------------------------------------------

def load_phase4(phase4_dir: Path) -> dict:
    """
    Load Phase 4 outputs. Files are optional — dict contains whichever exist.

    Expected files (any subset):
      t1_feature_cluster_mi.json  — {prompt: {layer: {feature_idx: mi}}}
      t2_lda_directions.npz       — LDA directions per (prompt, layer)
      t3_bottleneck_directions.npz — AE bottleneck basis (d, k)
      verdict.json                — overall Phase 4 verdict
    """
    phase4_dir = Path(phase4_dir)
    out = {}
    if not phase4_dir.exists():
        return out

    # Most recent run subdir
    subdirs = [d for d in phase4_dir.iterdir() if d.is_dir()]
    target = max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else phase4_dir

    for json_name in ("t1_feature_cluster_mi.json", "verdict.json",
                      "track1.json", "track2.json"):
        p = target / json_name
        if p.exists():
            with open(p) as f:
                out[json_name.replace(".json", "")] = json.load(f)

    for npz_name in ("t2_lda_directions.npz", "t3_bottleneck_directions.npz"):
        p = target / npz_name
        if p.exists():
            data = np.load(p)
            out[npz_name.replace(".npz", "")] = {k: data[k] for k in data.files}

    return out
