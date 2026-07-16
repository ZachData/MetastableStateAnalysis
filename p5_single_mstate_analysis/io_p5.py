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

    #  --- centroid_trajectories.npz: traj_{id} ---
    ct_path = run_dir / "centroid_trajectories.npz"
    if ct_path.exists():
        data = np.load(ct_path)
        out["centroid_trajs"] = {
            int(k.split("_")[1]): data[k] for k in data.files
        }
    else:
        out["centroid_trajs"] = {}

    # --- metrics: assemble from split JSON files ---
    # geometry.json is required and already parsed; merge the optional files
    # into a flat per-layer dict, then expose as {"layers": [...]} so that
    # compute_profile / run_sibling_contrast can read per-layer scalars.
    layer_map: dict = {}
    for lr in geo.get("layers", []):
        layer_map[lr["layer"]] = dict(lr)

    def _merge_into_layer_map(fname: str) -> None:
        path = run_dir / fname
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for row in data.get("layers", []):
            li = row.get("layer")
            if li is not None:
                layer_map.setdefault(li, {}).update(row)

    for fname in ("energies.json", "clustering.json", "spectral.json", "sinkhorn.json"):
        _merge_into_layer_map(fname)

    # Rehydrate float-keyed dicts that JSON stringified
    for lr in layer_map.values():
        if "energies" in lr:
            lr["energies"] = {float(k): v for k, v in lr["energies"].items()}
        if "energy_drop_pairs" in lr:
            lr["energy_drop_pairs"] = {float(k): v for k, v in lr["energy_drop_pairs"].items()}

    out["metrics"] = {
        "layers": [layer_map[i] for i in sorted(layer_map)],
    }

    return out



def find_phase1_runs(phase1_dir: Path, model_stem: str) -> dict:
    """
    Enumerate Phase 1 v2 run directories matching a model stem.

    Reads the prompt name from geometry.json (canonical).
    Falls back to inferring it from the directory name.

    Handles the hyphen/underscore naming duality:
      model_stem passed in as "albert_xlarge_v2" (underscores)
      but p1 dirs are named  "albert-xlarge-v2_12iter_paper_excerpt" (hyphens).
    Both forms are checked.

    For ALBERT shared-layer models, multiple iter-depth directories
    (12iter / 24iter / 36iter / 48iter) all resolve to the same prompt_key.
    The directory whose name encodes the highest iteration count is kept so
    the selection is deterministic and maximally informative.

    Returns
    -------
    dict {prompt_key: run_dir_path}
    """
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        return {}

    # Accept both "albert_xlarge_v2" and "albert-xlarge-v2"
    model_stem_hyphen = model_stem.replace("_", "-")

    def _matches(name: str) -> bool:
        if not model_stem:
            return True
        return model_stem in name or model_stem_hyphen in name

    def _iter_depth(run_dir: Path) -> int:
        """Extract numeric iteration depth from names like *_24iter_*; 0 if absent."""
        import re
        m = re.search(r"_(\d+)iter", run_dir.name)
        return int(m.group(1)) if m else 0

    # Collect all candidates; for colliding prompt keys keep highest iter depth.
    # Structure: {prompt_key: (iter_depth, run_dir)}
    best: dict = {}

    for run_dir in phase1_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not _matches(run_dir.name):
            continue

        geo_path = run_dir / "geometry.json"
        pk = None
        if geo_path.exists():
            try:
                with open(geo_path) as f:
                    pk = json.load(f).get("prompt")
            except Exception:
                pk = None

        if not pk:
            # Fallback: infer from directory name
            name = run_dir.name
            pk = name.split("iter_", 1)[-1] if "iter_" in name else name

        depth = _iter_depth(run_dir)
        if pk not in best or depth > best[pk][0]:
            best[pk] = (depth, run_dir)

    return {pk: info[1] for pk, info in best.items()}


# ---------------------------------------------------------------------------
# Phase 2 artifacts — V projectors (shared across prompts)
# ---------------------------------------------------------------------------

def load_phase2_projectors(phase2_dir: Path, model_stem: str, k_top=None) -> dict:
    phase2_dir = Path(phase2_dir)
    out = {
        "eigenvalues": None, "eigenvectors": None,
        "attractive_P": None, "repulsive_P": None,
        "U_att": None, "U_rep": None,
        "eigvals_pos": None, "eigvals_neg": None,
        "path": None,
    }

    stem_h = model_stem.replace("_", "-")  # albert_xlarge_v2 → albert-xlarge-v2
    candidates = [
        phase2_dir / f"ov_projectors_{model_stem}.npz",
        phase2_dir / f"ov_projectors_{stem_h}.npz",
        *phase2_dir.glob(f"*projector*{model_stem}*.npz"),
        *phase2_dir.glob(f"*projector*{stem_h}*.npz"),
        *phase2_dir.glob(f"*{model_stem}*projector*.npz"),
        *phase2_dir.glob(f"*{stem_h}*projector*.npz"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return out

    data = np.load(path)
    keys = set(data.files)

    # --- Path A: eigenvectors + eigenvalues (ideal format) ---
    eigvecs = data.get("eigenvectors", data.get("U"))
    eigvals = data.get("eigenvalues",  data.get("S"))

    if eigvecs is not None and eigvals is not None:
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

    else:
        # --- Path B: subspace_build.py format ---
        # ALBERT shared: schur_attract_shared / sym_attract_shared
        # GPT-2 per-layer: schur_attract_layer0_* etc. — take layer 0 as representative
        def _pick(prefixes, suffix=""):
            for pfx in prefixes:
                k = pfx + suffix
                if k in keys:
                    return data[k].astype(np.float32)
            # glob for any key starting with any prefix
            for pfx in prefixes:
                match = next((k for k in keys if k.startswith(pfx)), None)
                if match:
                    return data[match].astype(np.float32)
            return None

        U_att = _pick(["schur_attract_shared", "sym_attract_shared",
                        "schur_attract", "sym_attract", "U_pos"])
        U_rep = _pick(["schur_repulse_shared", "sym_repulse_shared",
                        "schur_repulse", "sym_repulse", "U_neg"])

        if U_att is None or U_rep is None:
            return out   # genuinely nothing usable

        # Ensure 2-D column matrices (d, k)
        if U_att.ndim == 1:
            U_att = U_att[:, None]
        if U_rep.ndim == 1:
            U_rep = U_rep[:, None]

        out.update({
            "U_att":        U_att,
            "U_rep":        U_rep,
            "attractive_P": U_att @ U_att.T,
            "repulsive_P":  U_rep @ U_rep.T,
            "path":         str(path),
        })

    out["U_attractive"] = out["U_att"]
    out["U_repulsive"]  = out["U_rep"]
    return out


def load_phase2_weights(phase2_dir: Path, model_stem: str) -> dict:
    phase2_dir = Path(phase2_dir)
    stem_h = model_stem.replace("_", "-")  # albert_xlarge_v2 → albert-xlarge-v2
    for candidate in (
        phase2_dir / f"ov_weights_{stem_h}.npz",
        phase2_dir / f"ov_weights_{model_stem}.npz",
        phase2_dir / f"weights_{stem_h}.npz",
        phase2_dir / f"weights_{model_stem}.npz",
        *phase2_dir.glob(f"*weights*{stem_h}*.npz"),
        *phase2_dir.glob(f"*weights*{model_stem}*.npz"),
    ):
        if candidate.exists():
            data = np.load(candidate)
            return {k: data[k] for k in data.files}
    return {}

# ---------------------------------------------------------------------------
# Phase 2i artifacts — symmetric/antisymmetric decomposition
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 2i artifacts — symmetric/antisymmetric decomposition
# ---------------------------------------------------------------------------


def load_phase2i(phase2_dir: Path, model_stem: str) -> dict:
    """
    Compute Phase 2i S/A decomposition artifacts inline via p2b analysis.

    p2b never persisted NPZ files, so this replaces the old disk-based loader
    with an on-demand recompute from the Phase 2 OV weight matrix.

    Reads:  <phase2_dir>/ov_weights_{model}.npz
              Per-head OV matrices stored under keys "wo_head_N", "W_O_head_N",
              or "ov_head_N" (ALBERT-style shared), or a pre-composed "ov_total"
              array of shape (d, d) or (n_layers, d, d).

    Returns dict with keys:
      V_sym            — symmetric part  S = (V + Vᵀ) / 2
      V_asym           — antisymmetric part A = (V - Vᵀ) / 2
      schur_T          — quasi-triangular Schur form T
      schur_Z          — orthogonal Schur vectors Z
      rotational_blocks — list of 2×2 block dicts from extract_schur_blocks
      is_per_layer     — bool; if True, all array-valued keys are lists (one per layer)

    Call sites that previously passed a phase2i_dir must now pass phase2_dir
    (the Phase 2 output directory, e.g. results/phase2).
    """
    from p2b_imaginary.rotational_rescaled import decompose_symmetric_antisymmetric
    from p2b_imaginary.rotational_schur import extract_schur_blocks

    phase2_dir = Path(phase2_dir)
    if not phase2_dir.exists():
        print(f"  [warn] phase2 dir not found: {phase2_dir}")
        return {}

    model_stem_hyphen = model_stem.replace("_", "-")

    # Locate ov_weights file — try hyphen form first (canonical on disk)
    weights_path = None
    for stem_form in (model_stem_hyphen, model_stem):
        candidate = phase2_dir / f"ov_weights_{stem_form}.npz"
        if candidate.exists():
            weights_path = candidate
            break

    if weights_path is None:
        print(f"  [warn] no Phase 2 OV weights found for stem '{model_stem}' "
              f"in {phase2_dir}")
        return {}

    try:
        data = np.load(weights_path, allow_pickle=True)
    except Exception as e:
        print(f"  [warn] could not load {weights_path}: {e}")
        return {}

    keys = list(data.keys())

    # Collect per-head OV matrices.  Convention from run_6._load_ov_weights:
    # each is the composed W_V_h @ W_O_h matrix (not just W_O alone).
    head_keys = sorted(
        k for k in keys
        if k.startswith("wo_head") or k.startswith("W_O_head") or k.startswith("ov_head")
    )

    if head_keys:
        # Shared-weight model (ALBERT): compose V_eff = Σ_h OV_h
        ov_matrices  = [sum(data[k] for k in head_keys)]
        is_per_layer = False
    elif "ov_total" in keys:
        raw = data["ov_total"]
        if raw.ndim == 3:
            # (n_layers, d, d) — one matrix per layer (GPT-2 style)
            ov_matrices  = list(raw)
            is_per_layer = True
        else:
            ov_matrices  = [raw]
            is_per_layer = False
    else:
        print(f"  [warn] no usable OV matrices in {weights_path}")
        return {}

    # Run p2b analysis inline on each composed OV matrix
    sa_list     = [decompose_symmetric_antisymmetric(OV) for OV in ov_matrices]
    blocks_list = [extract_schur_blocks(OV)              for OV in ov_matrices]

    if is_per_layer:
        return {
            "V_sym":             [sa["S"]         for sa in sa_list],
            "V_asym":            [sa["A"]         for sa in sa_list],
            "schur_T":           [b["schur_T"]    for b in blocks_list],
            "schur_Z":           [b["schur_Z"]    for b in blocks_list],
            "rotational_blocks": [b["blocks_2x2"] for b in blocks_list],
            "is_per_layer":      True,
        }
    else:
        sa, blocks = sa_list[0], blocks_list[0]
        return {
            "V_sym":             sa["S"],
            "V_asym":            sa["A"],
            "schur_T":           blocks["schur_T"],
            "schur_Z":           blocks["schur_Z"],
            "rotational_blocks": blocks["blocks_2x2"],
            "is_per_layer":      False,
        }

# # old method, matches tests:
# def load_phase2i(phase2_dir: Path, model_stem: str) -> dict:
#     phase2_dir = Path(phase2_dir)
#     if not phase2_dir.exists():
#         return {}

#     stems = {model_stem, model_stem.replace("_", "-"), model_stem.replace("-", "_")}
#     candidates: list[Path] = []

#     # Nested model subdir (hyphen or underscore form)
#     for s in stems:
#         subdir = phase2_dir / s
#         if subdir.is_dir():
#             candidates.extend(sorted(subdir.glob("*.npz")))

#     # Flat files at top level whose name contains either stem form
#     for s in stems:
#         candidates.extend(sorted(phase2_dir.glob(f"*{s}*.npz")))

#     # Deduplicate, preserve order
#     seen, ordered = set(), []
#     for p in candidates:
#         if p not in seen:
#             seen.add(p); ordered.append(p)

#     if not ordered:
#         print(f"  [warn] no Phase 2i artifacts found for stem '{model_stem}' in {phase2_dir}")
#         return {}

#     merged: dict = {}
#     for p in ordered:
#         try:
#             data = np.load(p, allow_pickle=True)
#         except Exception as e:
#             print(f"  [warn] could not load {p}: {e}")
#             continue
#         for k in data.files:
#             if k not in merged:   # first occurrence wins
#                 merged[k] = data[k]
#     return merged

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
            from p3_crosscoder.training import load_trained_crosscoder
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
            from p3_crosscoder.data import PromptActivationStore
            # PromptActivationStore.load is a classmethod that returns a new
            # instance — it does not mutate an existing one. The previous
            # `store = PromptActivationStore(); store.load(eval_dir)` called
            # the classmethod on an empty instance and discarded its return
            # value, so `store` stayed empty (0 prompts) even on success.
            out["prompt_store"] = PromptActivationStore.load(eval_dir)
        except Exception as e:
            print(f"[phase3] prompt_store load failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Phase 4 artifacts — LDA directions, feature-cluster MI, AE bottleneck
# ---------------------------------------------------------------------------

def load_phase4(phase4_dir: Path, model_stem: str = "") -> dict:
    """
    Load Phase 4 outputs. Files are optional — dict contains whichever exist.

    Selects the most recent run subdirectory whose name matches model_stem
    (both underscore and hyphen forms).  If model_stem is empty, falls back
    to the globally most-recently-modified subdir (legacy behaviour, but
    this is ambiguous when multiple models' results coexist).

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

    model_stem_hyphen = model_stem.replace("_", "-") if model_stem else ""

    def _matches(d: Path) -> bool:
        if not model_stem:
            return True  # no filter requested
        return model_stem in d.name or model_stem_hyphen in d.name

    subdirs = [d for d in phase4_dir.iterdir() if d.is_dir() and _matches(d)]
    if not subdirs:
        available = [d.name for d in phase4_dir.iterdir() if d.is_dir()]
        print(f"  [warn] no phase4 subdirs matching stem '{model_stem}' "
              f"in {phase4_dir}; available: {available}")
        print(f"  [phase4] skipped: run phase4 for '{model_stem}' first")
        return out  # do not fall back to a different model's artifacts

    target = max(subdirs, key=lambda d: d.stat().st_mtime)
    print(f"  [phase4] loading from {target.name}")

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