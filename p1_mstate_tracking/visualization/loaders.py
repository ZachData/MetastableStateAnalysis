"""
visualization/loaders.py

All disk reads. Every function here takes a run_dir and returns whatever
JSON/NPZ artifact (or derived array) it names — geometry.json,
clustering.json, trajectory.json, energies.json, sinkhorn.json,
spectral.json, HDBSCAN labels, the other three cluster-label families in
clusters.npz (KMeans, agglomerative-at-mid-threshold, and the Fiedler
bipartition), Fiedler eigenvectors, PCA trajectories, raw activations.
Nothing in this module plots anything; nothing outside it should open
these files directly.
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

def _spectral(run_dir: Path) -> dict:
    """spectral.json — k_eigengap, eigenvalues, eigengaps, fiedler_bipartition."""
    p = run_dir / "spectral.json"
    return json.load(open(p)) if p.exists() else {}


# ─────────────────────────────────────────────────────────────────────────────
# Alternative cluster definitions
#
# HDBSCAN is the one every existing figure uses, but run_1 computes and
# persists three more partitions of the same tokens: KMeans at the
# silhouette-selected k, agglomerative average-linkage at the middle
# cosine-distance threshold, and the sign of the Fiedler vector. All three
# are readable without rerunning anything — clusters.npz for the first two,
# spectral.json for the third. Each returns {layer_idx: (n_tokens,) int32},
# the same shape contract as _hdbscan_labels, so they are interchangeable
# anywhere a label dict is consumed.
# ─────────────────────────────────────────────────────────────────────────────

def _npz(path: Path) -> Dict[str, np.ndarray]:
    """Contents of an .npz keyed by array name, or {} if the file is absent."""
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _clusters_npz(run_dir: Path) -> Dict[str, np.ndarray]:
    """Raw contents of clusters.npz keyed by array name, or {} if absent."""
    return _npz(run_dir / "clusters.npz")


def _labels_by_prefix(arrays: Dict[str, np.ndarray], prefix: str) -> Dict[int, np.ndarray]:
    """
    Pull {layer_idx: array} out of an npz dict for keys named
    '{prefix}L{layer}'. Keys that don't parse are skipped silently — a
    stale or partially-written npz degrades to fewer layers, never raises.
    """
    out: Dict[int, np.ndarray] = {}
    for key, arr in arrays.items():
        if not key.startswith(prefix):
            continue
        try:
            out[int(key[len(prefix):].lstrip("L"))] = arr
        except ValueError:
            continue
    return out


def _kmeans_labels(run_dir: Path) -> Dict[int, np.ndarray]:
    """{layer_idx: (n_tokens,)} KMeans labels at the silhouette-selected k."""
    return _labels_by_prefix(_clusters_npz(run_dir), "kmeans_labels_")


def _kmeans_centroids(run_dir: Path) -> Dict[int, np.ndarray]:
    """{layer_idx: (k, d)} KMeans centroids for the winning k."""
    return _labels_by_prefix(_clusters_npz(run_dir), "kmeans_centroids_")


def _agglom_labels(run_dir: Path) -> Dict[int, np.ndarray]:
    """
    {layer_idx: (n_tokens,)} agglomerative labels at the *middle*
    cosine-distance threshold — the only threshold whose labels run_1
    persists. The full sweep survives as counts only (see
    _agglom_threshold_counts).
    """
    return _labels_by_prefix(_clusters_npz(run_dir), "agglom_mid_labels_")


def _fiedler_vecs(run_dir: Path) -> Dict[int, np.ndarray]:
    """
    {layer_idx: (n_tokens,)} second eigenvector of the normalized Laplacian
    of the token Gram matrix, from fiedler_vecs.npz. Note this is the
    *token-geometry* Fiedler vector — unrelated to sinkhorn.json's
    fiedler_mean, which is a property of the attention graph.
    """
    return _labels_by_prefix(_npz(run_dir / "fiedler_vecs.npz"), "fiedler_")


def _fiedler_bipartition(run_dir: Path) -> Dict[int, np.ndarray]:
    """
    {layer_idx: (n_tokens,)} the Fiedler sign bipartition as a 0/1 label
    array, read from spectral.json. Layers where the bipartition is null
    (n < 3, or the eigensolve was skipped) are omitted rather than
    returned as an all-zero partition.
    """
    out: Dict[int, np.ndarray] = {}
    for lr in _spectral(run_dir).get("layers", []):
        bip = lr.get("fiedler_bipartition")
        if not bip:
            continue
        arr = np.asarray(bip, dtype=np.int32)
        out[int(lr["layer"])] = (arr > 0).astype(np.int32)
    return out


def _agglom_threshold_counts(run_dir: Path) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    The full agglomerative distance-threshold sweep as a matrix.

    Returns (thresholds, layers, counts) where thresholds is (n_thresh,)
    ascending, layers is (n_layers,) ascending, and counts is
    (n_thresh, n_layers) float with NaN wherever a threshold is missing
    for a layer.

    clustering.json stores these keyed by threshold, but JSON stringifies
    the float keys — so they come back as e.g. "0.30000000000000004" and
    have to be parsed rather than looked up by value.
    """
    layers_raw = _clustering(run_dir).get("layers", [])
    if not layers_raw:
        return np.array([]), [], np.zeros((0, 0))

    per_layer: List[Tuple[int, Dict[float, float]]] = []
    all_thresh = set()
    for lr in layers_raw:
        agg = lr.get("clustering", {}).get("agglomerative", {})
        parsed: Dict[float, float] = {}
        for k, v in agg.items():
            if k == "mid_labels":
                continue
            try:
                parsed[float(k)] = float(v)
            except (TypeError, ValueError):
                continue
        per_layer.append((int(lr["layer"]), parsed))
        all_thresh.update(parsed.keys())

    if not all_thresh:
        return np.array([]), [], np.zeros((0, 0))

    thresholds = np.array(sorted(all_thresh), dtype=float)
    layers = [li for li, _ in per_layer]
    counts = np.full((len(thresholds), len(layers)), np.nan)
    for col, (_, parsed) in enumerate(per_layer):
        for row, t in enumerate(thresholds):
            if t in parsed:
                counts[row, col] = parsed[t]
    return thresholds, layers, counts


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


