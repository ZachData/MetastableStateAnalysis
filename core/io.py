"""
core/io.py — Run manifests, git SHA, and figure stamping.

Phase 1 artifact loading (load_phase1_run, find_phase1_run_dir) lives in
p1_mstate_tracking/p1_io.py, not here — Phase 1 owns its own on-disk
format, so downstream phases read through the module that writes it rather
than a separate reader guessing at the same files. This file used to hold
both; they were consolidated after three independent load_phase1_run
implementations (this one, p5_single_mstate_analysis/p5_io.py,
p5b_manifold_steering/p5b_io.py) were found to silently disagree on key
names, source files, and key sets — and this one had a real bug on top: it
read `trajectories` from the top level of trajectory.json, but Phase 1
writes them nested under cluster_tracking, so it had been returning []
on every real run.

Naming convention (project-wide): one io module per phase, named
p{phase}_io.py — p1_io.py, p5_io.py, p5b_io.py, p6_io.py. This file's bare
basename io.py is reserved for exactly this cross-cutting content.

Provides
--------
compute_manifest_id : deterministic short id from (model, prompt_battery_hash,
                      checkpoint_step, seeds)
get_git_sha         : best-effort `git rev-parse HEAD`, never raises
RunTimer            : context manager for wall_time_seconds
write_manifest      : build + write manifest.json, return the dict written
load_manifest       : load a previously written manifest.json
stamp_figure_name   : "fig.png" + manifest_id -> "fig__<id>.png"
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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


def compute_manifest_id(
    model: str,
    prompt_battery_hash: str,
    checkpoint_step=None,
    seeds=None,
    prompt_key=None,
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

    `prompt_key` distinguishes runs that differ only in which prompt from
    the battery they used. The battery *hash* covers the whole set, so it
    is identical across those runs — without prompt_key, Phase 1's
    per-(model, prompt) run directories all share one id, and
    stamp_figure_name then produces colliding filenames for figures from
    different runs. It is keyword-only-by-position at the end and defaults
    to None so callers with a genuinely whole-battery scope (a session-level
    manifest) keep the 4-part id they had.
    """
    parts = [str(model), str(prompt_battery_hash), str(checkpoint_step), str(seeds)]
    if prompt_key is not None:
        parts.append(str(prompt_key))
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
    prompt_key: Optional[str] = None,
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
    prompt_key            : which prompt from the battery this run used;
                            required for per-prompt run dirs, or two prompts
                            of the same model collide on manifest_id
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
        prompt_key=prompt_key,
    )

    manifest = {
        "manifest_id": manifest_id,
        "model": model,
        "checkpoint_step": checkpoint_step,
        "prompt_key": prompt_key,
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