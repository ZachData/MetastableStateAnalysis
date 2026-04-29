"""
p2c_churchland — Phase 2c: Trajectory-Side Dynamical Systems Analysis.

Five analysis tracks:
  C1 — jPCA + U_A alignment + HDR (borderline fallback)
  C2 — Trajectory tangling (Russo et al. 2018 Q metric)
  C3 — Condition-invariant signal / dPCA-style split (Kaufman et al. 2016)
  C4 — Local Jacobians at Phase 1 centroids (slow-point comparison)
  C5 — ICL subspace scaling + context-selection divergence (Mante et al. 2013)

Imports are lazy so that modules with heavy optional deps (torch, scipy)
do not fail at package import time if only a subset is being used.
"""

from __future__ import annotations
from importlib import import_module as _im

_PUBLIC: dict[str, tuple[str, str]] = {
    # C1
    "fit_jpca":                ("p2c_churchland.jpca_fit",             "fit_jpca"),
    "jpca_to_json":            ("p2c_churchland.jpca_fit",             "jpca_to_json"),
    "align_jpca_to_ua":        ("p2c_churchland.jpca_alignment",       "align_jpca_to_ua"),
    "principal_angles":        ("p2c_churchland.jpca_alignment",       "principal_angles"),
    "jpca_alignment_to_json":  ("p2c_churchland.jpca_alignment",       "jpca_alignment_to_json"),
    "fit_hdr":                 ("p2c_churchland.hdr_fit",              "fit_hdr"),
    "hdr_to_json":             ("p2c_churchland.hdr_fit",              "hdr_to_json"),
    # C2
    "analyze_tangling":        ("p2c_churchland.tangling",             "analyze_tangling"),
    "tangling_to_json":        ("p2c_churchland.tangling",             "tangling_to_json"),
    # C3
    "analyze_cis":             ("p2c_churchland.cis_decompose",        "analyze_cis"),
    "cis_to_json":             ("p2c_churchland.cis_decompose",        "cis_to_json"),
    # C4
    "analyze_local_jacobians": ("p2c_churchland.local_jacobian",       "analyze_local_jacobians"),
    "compare_local_global":    ("p2c_churchland.slow_point_compare",   "compare_local_global"),
    "layer_sa_profile":        ("p2c_churchland.slow_point_compare",   "layer_sa_profile"),
    # C5
    "analyze_icl_scaling":     ("p2c_churchland.icl_subspace_scaling", "analyze_icl_scaling"),
    "icl_scaling_to_json":     ("p2c_churchland.icl_subspace_scaling", "icl_scaling_to_json"),
    "analyze_context_pair":    ("p2c_churchland.context_selection",    "analyze_context_pair"),
}

__all__ = list(_PUBLIC)


def __getattr__(name: str):
    if name in _PUBLIC:
        mod_path, attr = _PUBLIC[name]
        return getattr(_im(mod_path), attr)
    raise AttributeError(f"module 'p2c_churchland' has no attribute {name!r}")