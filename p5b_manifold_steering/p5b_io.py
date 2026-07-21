"""
p5b_manifold_steering/p5b_io.py — Artifact loading for Phase 5b.

Wraps p1_mstate_tracking.p1_io rather than reimplementing Phase 1 loading.
All path conventions match Phase 1 v2 layout (p1_io.save_run).

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

    Deliberately its own simple stem-prefix rule rather than delegating to
    p5_io.find_phase1_runs (which does ALBERT iter-depth dedup and reads
    the prompt key from geometry.json) — that's the right behaviour for
    Phase 5's trajectory ranking, but it would silently change which run
    Phase 5b selects. Phase 5b just wants the most direct name match.
    """
    phase1_dir = Path(phase1_dir)
    runs: dict[str, Path] = {}

    # A missing phase1_dir is an ordinary "nothing to do" case, not an error:
    # run_5b._run_one treats an empty result as "no Phase 1 runs found" and
    # returns 1. Without this guard, .iterdir() below raises FileNotFoundError
    # and the caller never gets to make that decision.
    if not phase1_dir.is_dir():
        return runs

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

    Delegates to p1_mstate_tracking.p1_io.load_phase1_run, which already
    returns everything this function needs: plateau_layers, merge_layers,
    trajectories, centroid_trajs, activations, n_layers, n_tokens, d_model,
    prompt, model. Backfills n_layers/n_tokens/d_model from the activations
    array only in the unlikely case p1_io didn't have geometry.json to read
    them from.

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
    from p1_mstate_tracking.p1_io import load_phase1_run as _p1_load

    run_dir = Path(run_dir)
    p1 = _p1_load(run_dir)

    out: dict = {
        "plateau_layers": p1.get("plateau_layers", []),
        "merge_layers":   p1.get("merge_layers", []),
        "trajectories":   p1.get("trajectories", []),
        "centroid_trajs": p1.get("centroid_trajs", {}),
        "activations":    p1.get("activations"),
        "n_layers":       p1.get("n_layers", 0),
        "n_tokens":       p1.get("n_tokens", 0),
        "d_model":        p1.get("d_model", 0),
        "prompt":         p1.get("prompt", ""),
        "model":          p1.get("model", ""),
        "run_dir":        str(run_dir),
    }

    # Backfill from activations shape if geometry.json didn't have them
    if out["activations"] is not None:
        if not out["n_layers"]:
            out["n_layers"] = out["activations"].shape[0]
        if not out["n_tokens"]:
            out["n_tokens"] = out["activations"].shape[1]
        if not out["d_model"]:
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