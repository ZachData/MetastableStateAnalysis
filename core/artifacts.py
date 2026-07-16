"""
core/artifacts.py — Artifact contract (transition plan v2, core
infrastructure item 2).

Each phase declares what it writes: filenames, the keys/columns inside
each file, and (for arrays) expected shape/dtype constraints. Every
consumer is meant to import the ArtifactSpec constants from here instead
of hardcoding a filename string or a dict key at the call site.

Why this exists (from the plan): two of the six known bugs are exactly
this class of mismatch —
  - Phase 5's OV values in head_contributions.py are always n/a, most
    likely a miskeyed Phase 2 weights load.
  - Phase 5 Group D is universally blocked because p5io.load_phase4()'s
    path/naming doesn't match what Phase 4 actually writes.
A contract does not retroactively fix either instance (that's the "known
bugs" fix, item 4) — it kills the *class*: once a producer and consumer
both import ARTIFACTS["phaseN"]["thing"].filename (or .required_keys) from
one place, they cannot drift apart silently the way two independently
hand-typed strings can.

Scope of what's registered here: Phase 1's on-disk contract, verified
against core/io.py's existing `load_phase1_run` docstring contract and
p1_mstate_tracking/io_utils.py's actual save/load file list — plus the
run-manifest artifact (core/io.py's new write_manifest). Phases 2, 4, 5,
5b, 6 are left as explicit TODO placeholders: their real file lists were
not available to verify against for this pass (see each TODO for exactly
what's missing), and a guessed contract is worse than no contract — it
would look authoritative while being unverified, which is the opposite of
this module's purpose. Fill each in from that phase's actual io module
before any Pythia-checkpoint code depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ArtifactSpec:
    """
    One declared artifact: a file a phase writes and every consumer of it
    is expected to read the same way.

    kind            : "json" | "npz" | "txt"
    filename        : exact on-disk filename within a run directory
    required_keys   : for "json" — required top-level keys.
                       for "npz" — required array keys (np.load(...).files).
                       None / empty for "txt" or free-form files.
    key_shape_hint  : optional {key: shape-description} for documentation
                       and for validate_artifact's shape checks when a
                       fixed ndim is known (e.g. "(n_layers, n_tokens, d)").
    description     : one line, human-readable
    """
    kind: str
    filename: str
    required_keys: tuple = ()
    key_shape_hint: dict = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if self.kind not in ("json", "npz", "txt"):
            raise ValueError(f"ArtifactSpec.kind must be 'json', 'npz', or 'txt'; got {self.kind!r}")


# ---------------------------------------------------------------------------
# Phase 1 — verified against core/io.py's load_phase1_run contract and
# p1_mstate_tracking/io_utils.py's save_run/load_run file list.
# ---------------------------------------------------------------------------

PHASE1 = {
    "geometry": ArtifactSpec(
        kind="json", filename="geometry.json",
        description="Per-layer geometric scalars (ip_mean, ip_std, ip_mass_near_1, ...).",
    ),
    "energies": ArtifactSpec(
        kind="json", filename="energies.json",
        description="Per-layer, per-beta interaction energies.",
    ),
    "clustering": ArtifactSpec(
        kind="json", filename="clustering.json",
        required_keys=("hdbscan",),
        description="Agglomerative/KMeans/HDBSCAN results per layer.",
    ),
    "spectral": ArtifactSpec(
        kind="json", filename="spectral.json",
        description="Eigengap-k / Fiedler results per layer.",
    ),
    "hdbscan_labels": ArtifactSpec(
        kind="json", filename="hdbscan_labels.json",
        description=(
            "String-keyed dict {\"<layer>\": [label, ...]}. core.io.load_phase1_run "
            "converts this to a list indexed by integer layer position — that "
            "conversion is the contract point; don't re-implement it at a new "
            "call site, import load_phase1_run instead."
        ),
    ),
    "events": ArtifactSpec(
        kind="json", filename="events.json",
        description="{'merge_layers': [...], 'energy_violations': {'<beta>': [...]}}.",
    ),
    "trajectory": ArtifactSpec(
        kind="json", filename="trajectory.json",
        required_keys=("trajectories", "plateau_layers"),
        description="Cluster-chain trajectories + plateau_layers.",
    ),
    "activations": ArtifactSpec(
        kind="npz", filename="activations.npz",
        required_keys=("activations",),
        key_shape_hint={"activations": "(n_layers, n_tokens, d_model)"},
        description="Raw per-layer hidden states for one (model, prompt) run.",
    ),
    "attentions": ArtifactSpec(
        kind="npz", filename="attentions.npz",
        required_keys=("attentions",),
        key_shape_hint={"attentions": "(n_layers, n_heads, n_tokens, n_tokens)"},
        description="Raw per-layer attention tensors. Optional — plateau layers only.",
    ),
    "clusters": ArtifactSpec(
        kind="npz", filename="clusters.npz",
        description="Per-layer cluster label arrays (array-form companion to hdbscan_labels.json).",
    ),
    "centroid_trajectories": ArtifactSpec(
        kind="npz", filename="centroid_trajectories.npz",
        description="traj_N -> centroid path array, keyed per cluster trajectory id.",
    ),
    "tokens": ArtifactSpec(
        kind="txt", filename="tokens.txt",
        description="index\\ttoken per line.",
    ),
    "llm_report": ArtifactSpec(
        kind="txt", filename="llm_report.txt",
        description="Per-run human-readable summary.",
    ),
}

# Session-level (not per-run) Phase 1 artifacts.
PHASE1_SESSION = {
    "llm_cross_run_report": ArtifactSpec(
        kind="txt", filename="llm_cross_run_report.txt",
        description="Cross-run comparison report, one per session directory.",
    ),
    "experiment": ArtifactSpec(
        kind="txt", filename="experiment.txt",
        description="Session-level experiment log.",
    ),
}

# ---------------------------------------------------------------------------
# Particle records — the canonical artifact shape (core infrastructure
# item 4, core/particles.py). One per (model, checkpoint, prompt) run;
# cluster- and population-level results elsewhere are aggregations over
# this table, not separate artifacts.
# ---------------------------------------------------------------------------

PARTICLES = {
    "particle_table": ArtifactSpec(
        kind="npz", filename="particle_table.npz",
        required_keys=(
            "model", "checkpoint_step", "prompt_key", "layer", "token_position",
            "cluster_label", "population",
        ),
        description=(
            "Long-format per-particle table: one row per (model, checkpoint, "
            "prompt, layer, token). See core/particles.py:ParticleTable for "
            "the full schema, including optional columns (token_str, "
            "v_attractive_proj, v_repulsive_proj) and the extra__* columns a "
            "future dual-reading primitive adds. Write/read via "
            "ParticleTable.save/load rather than np.savez/np.load directly."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Run manifest — every phase, every run (core/io.py's write_manifest).
# ---------------------------------------------------------------------------

MANIFEST = ArtifactSpec(
    kind="json", filename="manifest.json",
    required_keys=("manifest_id", "model", "checkpoint_step", "hf_revision",
                   "prompt_battery_hash", "git_sha", "config", "seeds",
                   "timestamp", "wall_time_seconds"),
    description=(
        "Cross-checkpoint comparison is only meaningful if every point "
        "provably came from the same code and prompts. See core/io.py "
        "write_manifest / manifest_id."
    ),
)

# ---------------------------------------------------------------------------
# TODO — not yet registered. Each needs that phase's actual writer read
# first (not guessed), same standard PHASE1 was held to above.
# ---------------------------------------------------------------------------

# PHASE2  : p2_eigenspectra/weights.py + run_2.py write eigenspectrum /
#           decompose / OV-analysis artifacts. Registering this is what
#           would have caught the "Phase 5 OV values always n/a" bug at
#           the contract level — read weights.py's actual save calls
#           before filling this in, don't infer from head_contributions.py's
#           read side alone (that's the miskeyed half, not the source of
#           truth).
# PHASE4  : p4_mstate_features/run_4.py + low_rank_ae.py (frozen-for-
#           deletion — low priority to register, and per the freeze policy
#           no new work should happen on it regardless).
# PHASE5  : p5_single_mstate_analysis/io.py — this is the consumer side of
#           the Phase 4 path/naming mismatch (Group D blocker); needs
#           Phase 4's actual writer, not this module's reader, as the
#           source of truth for required_keys.
# PHASE5B : p5b_manifold_steering/p5b_io.py.
# PHASE6  : p6_subspace/p6_io.py.


# ---------------------------------------------------------------------------
# Registry + validation
# ---------------------------------------------------------------------------

REGISTRY = {
    "phase1": PHASE1,
    "phase1_session": PHASE1_SESSION,
    "particles": PARTICLES,
    "manifest": {"manifest": MANIFEST},
}


def get_spec(phase: str, name: str) -> ArtifactSpec:
    """Look up a registered ArtifactSpec. Raises KeyError with the valid
    options listed, rather than a bare KeyError, since a typo'd artifact
    name is exactly the bug class this module exists to catch early."""
    try:
        phase_registry = REGISTRY[phase]
    except KeyError:
        raise KeyError(f"Unknown phase {phase!r}. Known phases: {sorted(REGISTRY)}")
    try:
        return phase_registry[name]
    except KeyError:
        raise KeyError(
            f"Unknown artifact {name!r} for phase {phase!r}. "
            f"Known artifacts: {sorted(phase_registry)}"
        )


def artifact_path(run_dir, phase: str, name: str) -> Path:
    """Expected path for a registered artifact inside a given run directory."""
    spec = get_spec(phase, name)
    return Path(run_dir) / spec.filename


def validate_artifact(run_dir, phase: str, name: str) -> dict:
    """
    Check that a registered artifact exists on disk and (for json/npz)
    carries every required key.

    Returns
    -------
    dict: {ok: bool, path: str, missing_keys: list[str], error: str|None}
    Never raises on a missing file or missing key — that's exactly the
    condition this function exists to report, not crash on. Raises only
    for an unregistered (phase, name) pair — see get_spec.
    """
    spec = get_spec(phase, name)
    path = artifact_path(run_dir, phase, name)

    if not path.exists():
        return {"ok": False, "path": str(path), "missing_keys": list(spec.required_keys),
                "error": f"file does not exist: {path}"}

    if not spec.required_keys:
        return {"ok": True, "path": str(path), "missing_keys": [], "error": None}

    try:
        if spec.kind == "json":
            import json
            with open(path) as f:
                data = json.load(f)
            present = set(data.keys()) if isinstance(data, dict) else set()
        elif spec.kind == "npz":
            import numpy as np
            data = np.load(path)
            present = set(data.files)
        else:
            present = set()
    except Exception as e:
        return {"ok": False, "path": str(path), "missing_keys": list(spec.required_keys),
                "error": f"could not parse {path}: {e}"}

    missing = [k for k in spec.required_keys if k not in present]
    return {"ok": not missing, "path": str(path), "missing_keys": missing, "error": None}