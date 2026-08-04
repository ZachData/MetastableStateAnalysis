"""
p2_eigenspectra/visualization/loaders.py

All disk reads. Mirrors p1_mstate_tracking/visualization/loaders.py's role:
every function here takes a Phase 2 output directory (or one run stem
inside it) and returns the artifact it names. Nothing here plots; nothing
outside this module opens a Phase 2 file directly.

Phase 2's on-disk layout is NOT Phase 1's, which is why this file exists
rather than reusing p1's `discover_runs`. Phase 1 writes one directory per
(model, prompt) with geometry.json inside. Phase 2 writes:

    p2_eigenspectra_<ts>/
      ov_summary_<model>.json        <- per-model, PROMPT-INDEPENDENT
      ov_decomp_<model>.npz          <- per-model eigen/Schur arrays
      ov_weights_<model>.npz         <- per-model OV matrices (large)
      ov_projectors_<model>.npz      <- per-model projectors (large)
      <model>_<prompt>/
        phase2_verdict.json          <- note: NOT 'verdict.json'; run_2.py's
                                        module docstring is stale, the writer
                                        is reporting_p2.save_verdict
        summary.txt
        sub/{trajectory,layer_v_events,head_ov,...}.json

That split is the whole reason the figure set divides in two. The
`ov_summary_*.json` family is one table per checkpoint with no prompt
dependence at all — 27 files for the 410M pilot, a few KB each — and it
carries every eigenspectrum scalar. The per-(model, prompt) verdicts are
9× more numerous and only add the activation-dependent tests. Weight
figures read the first; run figures read the second and get a 9-prompt
spread for free.

Two on-disk facts worth knowing before reading anything downstream:

1. `ov_summary_*.json`'s "model" field is always null. `weights._build_summary`
   creates it with the comment "filled by caller" and no caller ever fills
   it. The model name is therefore recovered from the FILENAME stem, which
   is `model_name.replace("/", "_")`. Harmless for Pythia registry keys
   (no slashes); it would collide for two HF repos differing only by
   organisation.

2. `_jsonify` maps non-finite floats to JSON null, so a missing value and
   a NaN are indistinguishable on disk. Every reader here maps null back
   to NaN rather than dropping the layer, so depth profiles keep their
   length and stay index-aligned with the layer axis.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Filename grammar
# ─────────────────────────────────────────────────────────────────────────────

_OV_SUMMARY_RE = re.compile(r"^ov_summary_(?P<stem>.+)\.json$")
_LAYER_RE = re.compile(r"^layer_(?P<idx>\d+)$")

VERDICT_FILENAME = "phase2_verdict.json"


def _nan(v) -> float:
    """JSON null (which _jsonify writes for NaN/inf) -> NaN, not a dropped point."""
    return float("nan") if v is None else float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Weight-side artifacts (per checkpoint, prompt-independent)
# ─────────────────────────────────────────────────────────────────────────────

def discover_weight_summaries(p2_dir: Path) -> Dict[str, dict]:
    """
    {model_name: ov_summary dict} for every checkpoint in a Phase 2 run dir.

    Model name comes from the filename stem (see module docstring note 1).
    Unreadable files are skipped silently, matching p1 loaders' behaviour —
    a half-written summary from an interrupted sweep shouldn't abort the
    whole figure run.
    """
    out: Dict[str, dict] = {}
    p2_dir = Path(p2_dir)
    if not p2_dir.exists():
        return out
    for f in sorted(p2_dir.glob("ov_summary_*.json")):
        m = _OV_SUMMARY_RE.match(f.name)
        if not m:
            continue
        try:
            with open(f) as fh:
                summary = json.load(fh)
        except Exception:
            continue
        summary.setdefault("model", None)
        if not summary.get("model"):
            summary["model"] = m.group("stem")
        out[m.group("stem")] = summary
    return out


def layer_keys(summary: dict) -> List[str]:
    """
    Layer keys in DEPTH order.

    `weights._extract_gptneox_ov` names layers "layer_0".."layer_23", so a
    lexical sort puts layer_10 before layer_2 and silently scrambles every
    depth profile. Sort on the parsed integer; anything unparseable (e.g.
    ALBERT's single "shared") keeps insertion order after the numbered ones.
    """
    keys = list(summary.get("layers", {}).keys())
    numbered, other = [], []
    for k in keys:
        m = _LAYER_RE.match(k)
        (numbered if m else other).append((int(m.group("idx")) if m else 0, k))
    numbered.sort(key=lambda t: t[0])
    return [k for _, k in numbered] + [k for _, k in other]


def n_layers(summary: dict) -> int:
    return len(summary.get("layers", {}))


def layer_field(summary: dict, field: str) -> np.ndarray:
    """
    (n_layers,) float array of one per-layer scalar from ov_summary.

    Missing field -> all-NaN of the right length, so callers can plot the
    gap rather than branching on absence.
    """
    keys = layer_keys(summary)
    layers = summary.get("layers", {})
    return np.array(
        [_nan(layers.get(k, {}).get(field)) for k in keys], dtype=float
    )


def per_head_field(summary: dict, field: str = "qk_spectral_norms_per_head") -> Optional[np.ndarray]:
    """
    (n_layers, n_heads) array of a per-head list field, or None if absent
    or ragged. Ragged is not padded: a ragged QK table means the extractor
    disagreed with itself about head count across layers, which is a bug to
    surface, not a shape to paper over.
    """
    keys = layer_keys(summary)
    layers = summary.get("layers", {})
    rows = []
    for k in keys:
        v = layers.get(k, {}).get(field)
        if v is None:
            return None
        rows.append([_nan(x) for x in v])
    if not rows:
        return None
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return None
    return np.asarray(rows, dtype=float)


def decomp_path(p2_dir: Path, model: str) -> Optional[Path]:
    p = Path(p2_dir) / f"ov_decomp_{model}.npz"
    return p if p.exists() else None


def eigen_cloud(
    p2_dir: Path, model: str, layer_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    (eig_real, eig_imag, sym_eigenvalues) for one layer of one checkpoint,
    read out of ov_decomp_{model}.npz.

    np.load on an .npz is lazy — only the two or three named arrays are
    decompressed, not the file's Schur/eigenvector matrices, which for
    Pythia-410M are 24 × 1024² each. Reading a whole ov_decomp eagerly
    would be ~200 MB per checkpoint for a scatter plot of 1024 points.
    """
    p = decomp_path(p2_dir, model)
    if p is None:
        return None
    name = f"layer_{layer_idx}"
    with np.load(p) as z:
        rk, ik = f"eig_real_{name}", f"eig_imag_{name}"
        if rk not in z.files or ik not in z.files:
            # ALBERT-style shared-weight decomposition
            rk, ik = "eig_real_shared", "eig_imag_shared"
            if rk not in z.files:
                return None
            sk = "sym_evals_shared"
        else:
            sk = f"sym_evals_{name}"
        re_ = np.asarray(z[rk], dtype=float)
        im_ = np.asarray(z[ik], dtype=float)
        sym = np.asarray(z[sk], dtype=float) if sk in z.files else None
    return re_, im_, sym


# ─────────────────────────────────────────────────────────────────────────────
# Run-side artifacts (per checkpoint × prompt)
# ─────────────────────────────────────────────────────────────────────────────

def discover_p2_runs(p2_dir: Path) -> Dict[Tuple[str, str], Path]:
    """
    {(model, prompt): stem_dir} for every completed (model, prompt) run.

    Keyed off phase2_verdict.json's own "model"/"prompt" fields rather than
    parsing the directory name: the stem is f"{model}_{prompt}" and both
    halves may contain underscores, so the split is genuinely ambiguous
    ("pythia-410m-step0_short_heterogeneous"). The verdict says which is
    which.
    """
    runs: Dict[Tuple[str, str], Path] = {}
    p2_dir = Path(p2_dir)
    if not p2_dir.exists():
        return runs
    for d in sorted(p2_dir.iterdir()):
        if not d.is_dir():
            continue
        vf = d / VERDICT_FILENAME
        if not vf.exists():
            continue
        try:
            with open(vf) as fh:
                v = json.load(fh)
        except Exception:
            continue
        model, prompt = v.get("model"), v.get("prompt")
        if model is None or prompt is None:
            continue
        runs[(model, prompt)] = d
    return runs


def verdict(stem_dir: Path) -> dict:
    p = Path(stem_dir) / VERDICT_FILENAME
    return json.load(open(p)) if p.exists() else {}


def sub(stem_dir: Path, name: str) -> dict:
    """
    One sub-experiment's saved record: {name, applicable, payload,
    verdict_contribution, error}. Missing file -> {} (the sub-experiment
    never ran); present-but-inapplicable is a real record with
    applicable=False, and callers should tell those two apart.
    """
    p = Path(stem_dir) / "sub" / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def sub_payload(stem_dir: Path, name: str) -> Optional[dict]:
    """payload of an APPLICABLE sub-experiment, else None."""
    rec = sub(stem_dir, name)
    if not rec or not rec.get("applicable"):
        return None
    return rec.get("payload")


def prompts_for(runs: Dict[Tuple[str, str], Path]) -> List[str]:
    return sorted({p for (_, p) in runs})


def models_for(runs: Dict[Tuple[str, str], Path]) -> List[str]:
    return sorted({m for (m, _) in runs})


# ─────────────────────────────────────────────────────────────────────────────
# Cross-run table
# ─────────────────────────────────────────────────────────────────────────────

def cross_run(p2_dir: Path) -> List[dict]:
    """The flat list of every verdict, as written by run_2._save_cross_run."""
    p = Path(p2_dir) / "p2_eigenspectra_cross_run.json"
    if not p.exists():
        return []
    try:
        return json.load(open(p))
    except Exception:
        return []
