"""
core/io.py — Phase 1 artifact loader for downstream phases (5, 6).

Provides
--------
  load_phase1_run(run_dir)       → flat dict of arrays/lists (p6 ctx contract)
  find_phase1_run_dir(...)       → resolve phase1 dir under hyphen/flat layout

load_phase1_run contract
------------------------
Returns a dict with the following keys (all optional except where noted):

  activations     : ndarray (n_layers, n_tokens, d)  — REQUIRED when present
  tokens          : list[str] of length n_tokens
  hdbscan_labels  : list[ndarray(n_tokens,)]          — list, NOT dict
                    indexed by layer position 0..n_layers-1
                    gaps filled with np.full(n_tokens, -1, dtype=int32)
  events          : list[dict]  — merge/violation events
  trajectories    : list[dict]  — cluster chain trajectories
  attentions      : ndarray (n_layers, n_heads, n_tokens, n_tokens) or None

Key invariant: hdbscan_labels is always a list (never a string-keyed dict)
so that ctx["labels_per_layer"][L] works with integer L in every Track B/D
sub-experiment.

find_phase1_run_dir resolution order
--------------------------------------
Phase 1 writes flat dirs like:
  phase1_dir/albert-xlarge-v2_48iter_wiki_paragraph/

run_6.build_context previously looked for:
  phase1_dir / stem / *prompt_key*   →  phase1_dir/albert_xlarge_v2/...
which does not match the flat layout.

This function resolves correctly:
  1. phase1_dir / stem / *prompt_key*      (legacy nested layout)
  2. phase1_dir / *{model_any_form}*{prompt}*   (flat layout)
  3. Any dir containing either model fragment or prompt fragment
  4. None if nothing matches
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public: directory resolution
# ---------------------------------------------------------------------------

def find_phase1_run_dir(
    phase1_dir:  Path,
    model_name:  str,
    prompt_key:  str,
) -> Optional[Path]:
    """
    Locate the Phase 1 run directory for (model_name, prompt_key).

    Handles both:
      - nested layout: phase1_dir/{stem}/{prompt_key}/
      - flat layout:   phase1_dir/{model}-{iter}_{prompt_key}/

    Parameters
    ----------
    phase1_dir : root of phase1 results (e.g. results/phase1 or a timestamped subdir)
    model_name : model name in any form, e.g. "albert-xlarge-v2" or "albert_xlarge_v2"
    prompt_key : e.g. "wiki_paragraph"

    Returns
    -------
    Path to run directory, or None if not found.
    """
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        return None

    # Normalise model name into both hyphen and underscore variants
    stem_under  = model_name.replace("-", "_").replace("/", "_")
    stem_hyphen = model_name.replace("_", "-").replace("/", "-")

    # --- 1. Legacy nested layout: phase1_dir / stem_under / *prompt_key* ---
    nested = phase1_dir / stem_under
    if nested.is_dir():
        candidates = sorted(nested.glob(f"*{prompt_key}*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        # Fallback: any subdir under nested
        all_sub = [d for d in nested.iterdir() if d.is_dir()]
        if all_sub:
            return sorted(all_sub, key=lambda p: p.stat().st_mtime)[-1]

    # --- 2. Flat layout: phase1_dir / *model*_*prompt* ---
    for stem_form in (stem_hyphen, stem_under):
        # Both model fragment AND prompt fragment in name
        candidates = [
            d for d in phase1_dir.iterdir()
            if d.is_dir()
            and _stem_matches(d.name, stem_form)
            and prompt_key in d.name
        ]
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    # --- 3. Flat layout fallback: model match only (prompt absent from name) ---
    for stem_form in (stem_hyphen, stem_under):
        candidates = [
            d for d in phase1_dir.iterdir()
            if d.is_dir() and _stem_matches(d.name, stem_form)
        ]
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    return None


def _stem_matches(dirname: str, stem: str) -> bool:
    """
    True if dirname starts with stem (treating - and _ as equivalent).
    Both strings are normalised before comparison.
    """
    d = dirname.replace("-", "_").lower()
    s = stem.replace("-", "_").lower()
    return d.startswith(s)


# ---------------------------------------------------------------------------
# Public: artifact loader
# ---------------------------------------------------------------------------

def load_phase1_run(run_dir: Path) -> dict:
    """
    Load Phase 1 on-disk artifacts for a single (model, prompt) run and
    return a flat dict conforming to the p6 ctx contract.

    Parameters
    ----------
    run_dir : path to the per-prompt run directory written by io_utils.save_run

    Returns
    -------
    dict with keys described in module docstring.  Missing optional files
    produce None / empty-list values; never raises on missing files.
    """
    run_dir = Path(run_dir)
    out: dict = {}

    # ------------------------------------------------------------------
    # activations.npz  →  (n_layers, n_tokens, d)
    # ------------------------------------------------------------------
    acts_path = run_dir / "activations.npz"
    if acts_path.exists():
        data = np.load(acts_path)
        key  = "activations" if "activations" in data else list(data.keys())[0]
        out["activations"] = data[key]
    else:
        out["activations"] = None

    n_layers = out["activations"].shape[0] if out["activations"] is not None else 0
    n_tokens = out["activations"].shape[1] if out["activations"] is not None else 0

    # ------------------------------------------------------------------
    # tokens.txt  →  list[str]
    # ------------------------------------------------------------------
    out["tokens"] = _load_tokens(run_dir, n_tokens)

    # ------------------------------------------------------------------
    # hdbscan_labels.json  →  list[ndarray]  (NOT dict)
    #
    # Phase 1 writes: {"0": [0,1,0,...], "2": [...], ...}
    # We must convert to a list indexed by layer position so that
    # ctx["labels_per_layer"][L] works with integer L.
    # ------------------------------------------------------------------
    out["hdbscan_labels"] = _load_hdbscan_labels(
        run_dir / "hdbscan_labels.json", n_layers, n_tokens
    )

    # ------------------------------------------------------------------
    # events.json  →  list[dict]
    # ------------------------------------------------------------------
    out["events"] = _load_events(run_dir / "events.json")

    # ------------------------------------------------------------------
    # trajectory.json  →  trajectories list + plateau_layers
    # ------------------------------------------------------------------
    traj_path = run_dir / "trajectory.json"
    if traj_path.exists():
        try:
            with open(traj_path) as f:
                tj = json.load(f)
            out["trajectories"]   = tj.get("trajectories", [])
            out["plateau_layers"] = tj.get("plateau_layers", [])
        except Exception:
            out["trajectories"]   = []
            out["plateau_layers"] = []
    else:
        out["trajectories"]   = []
        out["plateau_layers"] = []

    # ------------------------------------------------------------------
    # attentions.npz  →  (n_layers, n_heads, n_tokens, n_tokens) or None
    # ------------------------------------------------------------------
    attn_path = run_dir / "attentions.npz"
    if attn_path.exists():
        try:
            adata = np.load(attn_path)
            akey  = "attentions" if "attentions" in adata else list(adata.keys())[0]
            out["attentions"] = adata[akey]
        except Exception:
            out["attentions"] = None
    else:
        out["attentions"] = None

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tokens(run_dir: Path, n_tokens: int) -> list[str]:
    """
    Load tokens from tokens.txt (tab-separated index\\ttoken per line).
    Falls back to synthetic names if file absent or malformed.
    """
    txt_path = run_dir / "tokens.txt"
    if txt_path.exists():
        try:
            toks = []
            with open(txt_path) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "\t" in line:
                        _, tok = line.split("\t", 1)
                    else:
                        tok = line
                    toks.append(tok)
            if toks:
                return toks
        except Exception:
            pass
    # Fallback: try tokens.json (alternate format)
    json_path = run_dir / "tokens.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                toks = json.load(f)
            if isinstance(toks, list):
                return [str(t) for t in toks]
        except Exception:
            pass
    return [f"<tok_{i}>" for i in range(n_tokens)]


def _load_hdbscan_labels(
    path:     Path,
    n_layers: int,
    n_tokens: int,
) -> list[np.ndarray] | None:
    """
    Convert hdbscan_labels.json from string-keyed dict to list[ndarray].

    Input (on disk): {"0": [0, 1, 0, ...], "1": [...], ...}
    Output:          [ndarray([0,1,0,...]), ndarray([...]), ...]
                      indexed by layer position 0..n_layers-1

    Gaps (layers absent from the JSON) are filled with noise-label arrays
    (all -1) so downstream code never receives None in the list.
    """
    if not path.exists():
        return None

    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as e:
        warnings.warn(f"load_phase1_run: could not parse {path}: {e}")
        return None

    if not isinstance(raw, dict):
        warnings.warn(f"load_phase1_run: expected dict in {path}, got {type(raw)}")
        return None

    # Infer n_layers from the dict if activations weren't loaded
    effective_n = n_layers if n_layers > 0 else (
        max(int(k) for k in raw) + 1 if raw else 0
    )
    if effective_n == 0:
        return None

    noise = np.full(n_tokens, -1, dtype=np.int32)
    result: list[np.ndarray] = []

    for L in range(effective_n):
        if str(L) in raw:
            result.append(np.array(raw[str(L)], dtype=np.int32))
        else:
            result.append(noise.copy())

    return result


def _load_events(path: Path) -> list[dict]:
    """
    Load events.json and return a flat list of event dicts.

    Phase 1 writes events.json as:
      {"merge_layers": [2, 5], "energy_violations": {"1.0": [3, 4]}}

    We normalise this into a list of {"type": ..., "layer": ...} dicts
    matching what build_context and _classify_layer_types expect.
    """
    if not path.exists():
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return []

    events: list[dict] = []

    for layer in raw.get("merge_layers", []):
        events.append({"type": "merge", "layer_name": str(layer), "layer_from": str(layer)})

    for beta_str, layers in raw.get("energy_violations", {}).items():
        for layer in layers:
            events.append({"type": "energy_violation", "layer": layer, "beta": float(beta_str)})

    return events


# ===========================================================================
# Run manifests — core foundations item 1 (transition plan v2).
#
# Cross-checkpoint comparison is only meaningful if every point provably
# came from the same code and prompts. Every run writes a manifest.json
# recording: model, HF revision, checkpoint step, prompt-battery hash, git
# SHA, a config dump, seeds, a timestamp, and wall-time. Every saved figure
# is stamped with the manifest's short id (stamp_figure_name), and
# wall_time_seconds doubles as the compute ledger informing adaptive-
# checkpoint and expensive-tier decisions (see the plan's checkpoint
# schedule section).
#
# Deliberately separate functions rather than one do-everything call:
# git-sha lookup and timing are cross-cutting (RunTimer, get_git_sha) so
# they can be reused or mocked independently of manifest writing itself.
# ===========================================================================

import hashlib
import subprocess
import time
from datetime import datetime, timezone


def compute_manifest_id(
    model: str,
    prompt_battery_hash: str,
    checkpoint_step=None,
    seeds=None,
) -> str:
    """
    Short, deterministic id for a run — used to stamp figures and to name
    manifest.json's own "manifest_id" field. Deterministic in
    (model, prompt_battery_hash, checkpoint_step, seeds) so the same
    logical run always gets the same id even if re-run (unlike a random
    UUID or a timestamp-based id, which would make re-plotting from a
    prior run's figures unable to confirm they came from that run).

    Does NOT include the timestamp or wall-time — those vary run-to-run
    for reasons unrelated to the run's identity (retries, machine load).
    """
    parts = [str(model), str(prompt_battery_hash), str(checkpoint_step), str(seeds)]
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def get_git_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """
    Best-effort `git rev-parse HEAD`. Returns None (never raises) if git
    isn't installed, repo_root isn't a git repo, or anything else goes
    wrong — a missing SHA is a real gap worth showing up as null in the
    manifest, not a reason to crash a run that's otherwise fine.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class RunTimer:
    """
    Context manager measuring wall-clock seconds for a run, for the
    manifest's wall_time_seconds field.

        with RunTimer() as timer:
            ... do the run ...
        write_manifest(..., wall_time_seconds=timer.elapsed)
    """

    def __enter__(self):
        self._start = time.monotonic()
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.monotonic() - self._start
        return False


def write_manifest(
    run_dir: Path,
    *,
    model: str,
    prompt_battery_hash: str,
    wall_time_seconds: float,
    hf_revision: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
    git_sha: Optional[str] = None,
    config: Optional[dict] = None,
    seeds: Optional[dict] = None,
    timestamp: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build and write manifest.json into run_dir. Returns the dict written
    (so callers can immediately use manifest["manifest_id"] to stamp
    figures without re-reading the file).

    Required keys match core.artifacts.MANIFEST.required_keys exactly —
    both are meant to be checked together (validate_artifact(run_dir,
    "manifest", "manifest") should pass on whatever this function writes).

    Parameters
    ----------
    run_dir              : directory to write manifest.json into (created
                            if missing)
    model                 : model name/key as used in core.config.MODEL_CONFIGS
    prompt_battery_hash   : from core.prompts's versioned battery hash
    wall_time_seconds     : e.g. RunTimer().elapsed
    hf_revision           : HF revision string (e.g. "step1000"); None for
                            non-checkpointed models
    checkpoint_step       : integer step; None for non-checkpointed models
    git_sha               : defaults to get_git_sha() if not passed
    config                : arbitrary config dump (must be JSON-serialisable)
    seeds                 : e.g. {"numpy": 42, "torch": 42}
    timestamp             : ISO-8601 UTC; defaults to now
    extra                 : any additional fields a specific phase needs;
                            merged in without overwriting the required keys
    """
    if git_sha is None:
        git_sha = get_git_sha()
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    seeds = seeds or {}
    config = config or {}

    manifest_id = compute_manifest_id(
        model=model,
        prompt_battery_hash=prompt_battery_hash,
        checkpoint_step=checkpoint_step,
        seeds=seeds,
    )

    manifest = {
        "manifest_id": manifest_id,
        "model": model,
        "checkpoint_step": checkpoint_step,
        "hf_revision": hf_revision,
        "prompt_battery_hash": prompt_battery_hash,
        "git_sha": git_sha,
        "config": config,
        "seeds": seeds,
        "timestamp": timestamp,
        "wall_time_seconds": wall_time_seconds,
    }
    if extra:
        for k, v in extra.items():
            manifest.setdefault(k, v)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=_manifest_json_default)

    return manifest


def _manifest_json_default(obj):
    """
    json.dump's `default` for write_manifest: converts the numpy scalar /
    array / Path types that legitimately show up in a config dump (e.g.
    a numpy seed, a Path in `config`), but re-raises TypeError for
    anything else. Deliberately not a blanket str() fallback — silently
    stringifying an arbitrary object (a live model, a class instance)
    would produce a manifest that looks complete but isn't actually
    reconstructable, defeating the point of the manifest.
    """
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(
        f"manifest config contains a non-JSON-serialisable {type(obj).__name__}: {obj!r}. "
        "Convert it to a plain type before calling write_manifest."
    )


def load_manifest(run_dir: Path) -> Optional[dict]:
    """Load a previously written manifest.json, or None if absent/unparseable."""
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def stamp_figure_name(base_name: str, manifest_id: str) -> str:
    """
    "energy_curve.png" + "a1b2c3d4e5f6" -> "energy_curve__a1b2c3d4e5f6.png"

    Every saved figure is stamped with its manifest id so "which code
    produced Figure 11" is answerable later (plan, Visualization section).
    Splits on the last "." so multi-dot names (e.g. "fig.v2.png") keep
    only the true extension after the stamp.
    """
    p = Path(base_name)
    return f"{p.stem}__{manifest_id}{p.suffix}"
