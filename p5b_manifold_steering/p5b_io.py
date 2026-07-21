"""
p5b_manifold/io.py — Artifact loading for Phase 5b.

Wraps the existing Phase 1 / Phase 5 loaders rather than reimplementing them.
All path conventions match Phase 1 v2 layout (io_utils.save_run).

Phase 1 v2 layout recap (per-run dir = {phase1_dir}/{stem}_{prompt}/):
  trajectory.json   → plateau_layers: [int, ...]
                      cluster_tracking.trajectories: [{id, chain}, ...]
                      cluster_tracking.events: [{layer_from, n_merges, ...}, ...]
  events.json       → merge_layers: [int, ...]
  centroid_trajectories.npz → keys "traj_{id}": (lifespan, d) float32
  activations.npz   → key "activations": (n_layers, n_tokens, d_model)

Phase 2 layout:
  {phase2_dir}/ov_projectors_{stem}.npz → U_pos, U_neg, U_A
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Phase 1 run discovery
# ---------------------------------------------------------------------------

def find_phase1_runs(phase1_dir: Path, stem: str) -> dict[str, Path]:
    """
    Find all Phase 1 run directories for a model stem.

    Phase 1 names dirs as: {stem}_{prompt_key}/
    e.g. gpt2_large_wiki_paragraph/

    Returns {prompt_key: run_dir}.
    """
    phase1_dir = Path(phase1_dir)
    runs: dict[str, Path] = {}

    # A missing phase1_dir is an ordinary "nothing to do" case, not an error:
    # run_5b._run_one treats an empty result as "no Phase 1 runs found" and
    # returns 1. Without this guard, .iterdir() below raises FileNotFoundError
    # and the caller never gets to make that decision.
    if not phase1_dir.is_dir():
        return runs

    # Reuse Phase 5's finder if available; fall back to glob.
    try:
        from p5_single_mstate_analysis.io import find_phase1_runs as _p5_find
        return _p5_find(phase1_dir, stem)
    except (ImportError, Exception):
        pass

    for subdir in sorted(phase1_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if not name.startswith(stem):
            continue
        # Strip stem prefix (+ underscore) to get prompt key
        suffix = name[len(stem):].lstrip("_")
        if suffix:
            runs[suffix] = subdir
    return runs


# ---------------------------------------------------------------------------
# Per-run artifact loading
# ---------------------------------------------------------------------------

def load_phase1_run(run_dir: Path) -> dict:
    """
    Load Phase 1 artifacts from a single run directory.

    Returns
    -------
    dict with guaranteed keys:
      plateau_layers    : list[int]
      merge_layers      : list[int]
      trajectories      : list of trajectory dicts {id, chain}
      centroid_trajs    : {int trajectory_id: (lifespan, d) float32}
      activations       : (n_layers, n_tokens, d) float32, or None
      n_layers          : int
      n_tokens          : int
      d_model           : int
      prompt            : str
      model             : str
    """
    # Reuse Phase 5's loader if available.
    try:
        from p5_single_mstate_analysis.io import load_phase1_run as _p5_load
        return _p5_load(run_dir)
    except (ImportError, Exception):
        pass

    return _load_phase1_run_direct(Path(run_dir))


def _load_phase1_run_direct(run_dir: Path) -> dict:
    """Minimal direct loader matching the v2 split-file format."""
    out: dict = {
        "plateau_layers": [],
        "merge_layers":   [],
        "trajectories":   [],
        "centroid_trajs": {},
        "activations":    None,
        "n_layers": 0, "n_tokens": 0, "d_model": 0,
        "prompt": "", "model": "",
        "run_dir": str(run_dir),
    }

    # --- geometry.json ---
    geo_path = run_dir / "geometry.json"
    if geo_path.exists():
        with open(geo_path) as f:
            geo = json.load(f)
        out["n_layers"] = geo.get("n_layers", 0)
        out["n_tokens"] = geo.get("n_tokens", 0)
        out["d_model"]  = geo.get("d_model", 0)
        out["prompt"]   = geo.get("prompt", "")
        out["model"]    = geo.get("model", "")

    # --- trajectory.json → plateau_layers + trajectories ---
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        with open(traj_path) as f:
            tj = json.load(f)
        out["plateau_layers"] = [int(l) for l in tj.get("plateau_layers", [])]
        ct = tj.get("cluster_tracking", {})
        out["trajectories"] = ct.get("trajectories", [])

    # --- events.json → merge_layers ---
    events_path = run_dir / "events.json"
    if events_path.exists():
        with open(events_path) as f:
            ev = json.load(f)
        out["merge_layers"] = [int(l) for l in ev.get("merge_layers", [])]

    # --- centroid_trajectories.npz → {int_id: (lifespan, d)} ---
    ct_path = run_dir / "centroid_trajectories.npz"
    if ct_path.exists():
        data = np.load(ct_path)
        for key in data.files:
            # Key format: "traj_{id}"
            if key.startswith("traj_"):
                try:
                    tid = int(key[5:])
                    out["centroid_trajs"][tid] = data[key].astype(np.float32)
                except (ValueError, Exception):
                    pass

    # --- activations.npz ---
    acts_path = run_dir / "activations.npz"
    if acts_path.exists():
        data = np.load(acts_path)
        key  = "activations" if "activations" in data else data.files[0]
        out["activations"] = data[key]
        if out["n_layers"] == 0:
            out["n_layers"] = out["activations"].shape[0]
        if out["n_tokens"] == 0:
            out["n_tokens"] = out["activations"].shape[1]
        if out["d_model"] == 0:
            out["d_model"]  = out["activations"].shape[2]

    return out


# ---------------------------------------------------------------------------
# Best-run selection
# ---------------------------------------------------------------------------

def select_best_run(
    runs: dict[str, Path],
    preferred_prompt: str | None = None,
) -> tuple[str, Path] | tuple[None, None]:
    """
    Pick the best Phase 1 run for Phase 5b.

    Priority: explicitly requested prompt → sullivan_ballou → paper_excerpt
              → wiki_paragraph → first available.
    """
    priority = ["sullivan_ballou", "paper_excerpt", "wiki_paragraph",
                "short_heterogeneous"]

    if preferred_prompt and preferred_prompt in runs:
        return preferred_prompt, runs[preferred_prompt]

    for key in priority:
        if key in runs:
            return key, runs[key]

    if runs:
        k = next(iter(runs))
        return k, runs[k]

    return None, None


# ---------------------------------------------------------------------------
# Phase 2 projector loading
# ---------------------------------------------------------------------------

def load_phase2_projectors(
    phase2_dir: Path,
    stem:       str,
) -> dict | None:
    """
    Load OV subspace projectors from Phase 2.

    Returns dict with U_pos, U_neg, U_A as (d, k) arrays, or None if not found.
    """
    phase2_dir = Path(phase2_dir)
    if not phase2_dir.is_dir():
        return None

    # Both directions must be tried. stem arrives in either form depending on
    # the call site (run_5b passes model_name.replace("-", "_"); tests and
    # CLI args pass the raw hyphenated model name), while the file on disk
    # may have been written under either. Previously only "_"→"-" was
    # covered, so a hyphenated stem never found an underscored file.
    # dict.fromkeys keeps first-seen order while de-duplicating when the
    # stem contains only one separator style.
    stem_variants = list(dict.fromkeys([
        stem,
        stem.replace("_", "-"),
        stem.replace("-", "_"),
    ]))

    for sv in stem_variants:
        for pattern in (
            f"ov_projectors_{sv}.npz",
            f"ov_projectors_{sv}_*.npz",
        ):
            candidates = sorted(phase2_dir.glob(pattern))
            if candidates:
                data = np.load(candidates[-1])
                proj: dict = {}
                # U_pos / U_neg (attractive / repulsive symmetric eigenvectors)
                for k_name in ("U_pos", "U_attract", "U_sym_pos"):
                    if k_name in data:
                        proj["U_S"] = data[k_name]
                        break
                for k_name in ("U_neg", "U_repulse", "U_sym_neg"):
                    if k_name in data:
                        proj["U_S_neg"] = data[k_name]
                        break
                for k_name in ("U_A", "U_imag", "U_antisym"):
                    if k_name in data:
                        proj["U_A"] = data[k_name]
                        break
                if "U_S" in proj:
                    # Build full real basis = U_pos ∪ U_neg if both present
                    if "U_S_neg" in proj:
                        proj["U_S_full"] = np.concatenate(
                            [proj["U_S"], proj["U_S_neg"]], axis=1
                        )
                    else:
                        proj["U_S_full"] = proj["U_S"]
                    proj["source"] = str(candidates[-1])
                    return proj

    return None
