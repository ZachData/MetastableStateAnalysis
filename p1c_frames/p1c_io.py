"""
p1c_frames/p1c_io.py — artifact loading and persistence for Phase 1c.

Phase 1c reads Phase 1's outputs and writes its own. Two rules carried over
from p1_mstate_tracking/p1_io.py, both learned the hard way:

1. IF A QUANTITY APPEARS IN A REPORT, IT IS PERSISTED. Phase 1's D2 defect
   was a per-head Fiedler section that existed only in the session that
   produced it, because the save function dropped three keys the reporter
   read. A derived statistic that can only be computed in the session that
   produced it is not a result.

2. EVERY DATA-DEPENDENT FALLBACK RECORDS THE BRANCH IT TOOK. Phase 1c has
   several: which cumulant source was available, whether beta_eff fell back
   to the run median, whether the ODE step converged, whether the hull
   optimizer converged. Each is written next to the number it produced.

Phase 1c also writes a PROVENANCE block per run recording the input
artifact paths and their modification times, because this is a re-analysis
phase — its results are only meaningful relative to a specific version of
the Phase 1 artifacts, and those are being regenerated in the same update
cycle that produced this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict:
    """
    Everything Phase 1c needs from one Phase 1 run directory.

    Missing pieces are returned as None rather than raising, with an
    `available` set recording what was found — sub-experiments have
    different requirements (A needs activations, C can work from
    geometry.json alone) and a run that supports three of five should
    contribute those three rather than being skipped entirely.
    """
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir), "available": set()}

    geo_p = run_dir / "geometry.json"
    if geo_p.exists():
        with open(geo_p) as f:
            out["geometry"] = json.load(f)
        out["available"].add("geometry")
    else:
        out["geometry"] = None

    en_p = run_dir / "energies.json"
    if en_p.exists():
        with open(en_p) as f:
            out["energies"] = json.load(f)
        out["available"].add("energies")
    else:
        out["energies"] = None

    act_p = run_dir / "activations.npz"
    if act_p.exists():
        data = np.load(act_p)
        out["activations"] = data["activations"] if "activations" in data.files else data[data.files[0]]
        # `norms` was added later (p1_io._save_activations); older runs lack
        # it. Without it the RAW residual stream is unrecoverable, and every
        # norm-weighted quantity — the whole sink question, and the
        # denominator of h_l — is unavailable. That is a hard limit on what
        # an old artifact can answer, so it is recorded rather than worked
        # around with a unit-norm stand-in.
        out["norms"] = data["norms"] if "norms" in data.files else None
        out["available"].add("activations")
        if out["norms"] is not None:
            out["available"].add("norms")
    else:
        out["activations"] = None
        out["norms"] = None

    out["provenance"] = {
        p.name: {"exists": p.exists(),
                 "mtime": (p.stat().st_mtime if p.exists() else None),
                 "bytes": (p.stat().st_size if p.exists() else None)}
        for p in (geo_p, en_p, act_p)
    }
    return out


def raw_states(run: dict) -> np.ndarray:
    """
    (n_layers, n_tokens, d) RAW residual stream.

    activations.npz stores UNIT-NORM activations plus the norms that
    projection discarded; raw = norms[..., None] * activations. Sub-
    experiment A's denominator and the entire sink question live in those
    norms, so a run without them cannot answer either — this raises rather
    than silently returning the unit-norm array, which would produce
    plausible numbers meaning something else.
    """
    if run.get("activations") is None:
        raise ValueError(f"{run['run_dir']}: no activations.npz")
    if run.get("norms") is None:
        raise ValueError(
            f"{run['run_dir']}: activations.npz has no `norms` key. This run "
            f"predates p1_io's norm-saving fix; the raw residual stream is "
            f"not recoverable and sub-experiments A and C cannot run on it. "
            f"Re-extract rather than substituting unit-norm activations."
        )
    return np.asarray(run["norms"])[..., None] * np.asarray(run["activations"])


def layer_series(run: dict, key: str) -> np.ndarray:
    """Per-layer scalar series from geometry.json, nan where absent."""
    geo = run.get("geometry") or {}
    layers = geo.get("layers", [])
    return np.array([l.get(key, np.nan) for l in layers], dtype=np.float64)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

class _NpEncoder(json.JSONEncoder):
    """numpy -> json. Nan/inf become null, which json.load reads back as
    None — the alternative is invalid JSON that fails at read time, far
    from where the nan was produced."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return [None if (isinstance(v, float) and not np.isfinite(v)) else v
                    for v in o.ravel().tolist()] if o.ndim == 1 else o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def save_p1c(results: dict, out_dir: Path, name: str = "p1c") -> Path:
    """
    Write one run's Phase 1c results.

    Arrays go to an .npz next to the .json rather than inline: the residual
    curves and margin profiles are per-layer and would dominate the JSON,
    and a JSON that is 95% number arrays is not readable by the human it
    exists for.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays, scalars = {}, {}
    def _split(prefix, d):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, np.ndarray) and v.size > 8:
                arrays[key] = v
            elif isinstance(v, dict):
                _split(f"{key}.", v)
            else:
                scalars.setdefault(prefix.rstrip("."), {})[k] = v
    _split("", results)

    jp = out_dir / f"{name}.json"
    with open(jp, "w") as f:
        json.dump({"scalars": scalars, "array_keys": sorted(arrays)},
                  f, indent=2, cls=_NpEncoder)
    if arrays:
        np.savez_compressed(out_dir / f"{name}_curves.npz", **arrays)
    return jp
