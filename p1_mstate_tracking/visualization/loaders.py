"""
visualization/loaders.py

All disk reads. Every function here takes a run_dir and returns whatever
JSON/NPZ artifact (or derived array) it names — geometry.json,
clustering.json, trajectory.json, energies.json, sinkhorn.json, HDBSCAN
labels, PCA trajectories, raw activations. Nothing in this module plots
anything; nothing outside it should open these files directly.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Artifact loaders
# ─────────────────────────────────────────────────────────────────────────────

def discover_runs(results_dir: Path) -> Dict[Tuple[str, str], Path]:
    """
    Scan results_dir and return {(model, prompt): run_dir} for every saved run.
    Reads geometry.json from each subdir; skips silently on errors.
    """
    runs: Dict[Tuple[str, str], Path] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        geo_file = d / "geometry.json"
        if not geo_file.exists():
            continue
        try:
            with open(geo_file) as f:
                geo = json.load(f)
            model  = geo.get("model", d.name)
            prompt = geo.get("prompt", "")
            runs[(model, prompt)] = d
        except Exception:
            continue
    return runs

def _geo(run_dir: Path) -> dict:
    with open(run_dir / "geometry.json") as f:
        return json.load(f)

def _clustering(run_dir: Path) -> dict:
    p = run_dir / "clustering.json"
    return json.load(open(p)) if p.exists() else {}

def _trajectory(run_dir: Path) -> dict:
    p = run_dir / "trajectory.json"
    return json.load(open(p)) if p.exists() else {}

def _energies(run_dir: Path) -> dict:
    p = run_dir / "energies.json"
    return json.load(open(p)) if p.exists() else {}

def _available_betas(run_dir: Path) -> List[float]:
    """Every β present in this run's energies.json, sorted ascending."""
    layers = _energies(run_dir).get("layers", [])
    keys = set()
    for lr in layers:
        keys.update(lr.get("energies", {}).keys())
    out = []
    for k in keys:
        try:
            out.append(float(k))
        except ValueError:
            continue
    return sorted(out)

def _energy_series(run_dir: Path, beta: float) -> Optional[List[float]]:
    """E_beta vs. layer for one beta, NaN where missing."""
    layers = _energies(run_dir).get("layers", [])
    if not layers:
        return None
    bstr = str(beta)
    out = [lr.get("energies", {}).get(bstr) for lr in layers]
    return [np.nan if v is None else v for v in out]

def _sinkhorn(run_dir: Path) -> dict:
    p = run_dir / "sinkhorn.json"
    return json.load(open(p)) if p.exists() else {}

def _hdbscan_labels(run_dir: Path) -> Dict[int, List[int]]:
    """Returns {layer_idx: [int labels]} from hdbscan_labels.json."""
    p = run_dir / "hdbscan_labels.json"
    if not p.exists():
        return {}
    raw = json.load(open(p))
    return {int(k): v for k, v in raw.items()}

def _pca_trajs(run_dir: Path) -> Dict[int, np.ndarray]:
    """Returns {layer_idx: (n_tokens, 3)} from pca_trajectories.npz."""
    p = run_dir / "pca_trajectories.npz"
    if not p.exists():
        return {}
    data = np.load(p)
    out  = {}
    for key in data.files:
        parts = key.split("_")
        if len(parts) >= 2:
            try:
                out[int(parts[-1])] = data[key]
            except ValueError:
                pass
    return out

def _load_activations(run_dir: Path) -> Optional[np.ndarray]:
    """
    Returns (n_layers, n_tokens, d) L2-normed hidden states from
    activations.npz, or None if the file is missing. This is the real
    high-dimensional geometry — used for t-SNE/UMAP, which need it raw
    rather than the cached 3-D PCA projection.
    """
    p = run_dir / "activations.npz"
    if not p.exists():
        return None
    data = np.load(p)
    key  = "activations" if "activations" in data.files else data.files[0]
    return data[key]


