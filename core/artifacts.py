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
        required_keys=("layers",),
        description=(
            "Agglomerative/KMeans/HDBSCAN results per layer, plus nesting "
            "and pair_agreement. Everything is nested under layers[i]; the "
            "HDBSCAN block is at layers[i].clustering.hdbscan. required_keys "
            "is checked against the top level only, so the previous "
            "('hdbscan',) could never be satisfied by any file this "
            "pipeline writes — the same false-failure class as the "
            "trajectory spec above. Label arrays live in clusters.npz, not "
            "here."
        ),
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
        required_keys=("cluster_tracking", "plateau_layers"),
        description=(
            "Cluster-chain trajectories + plateau_layers. The trajectory "
            "list is nested at cluster_tracking.trajectories, not at the top "
            "level: that is what p1_io._save_trajectory writes and what "
            "every reader (loaders.py, cluster_reality.py, "
            "cluster_orthogonality.py, p1_io's own loaders) already reaches "
            "for. This spec previously required a top-level 'trajectories' "
            "key that has never been written by anything, so "
            "validate_artifact reported a failure on every Phase 1 run "
            "while the pipeline was in fact consistent."
        ),
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
# Phase 7 — interaction records (p7_motifs). Written BEFORE the producer
# exists, which is the point: Phase 5's blockers 2 and 3 were both
# producer/consumer mismatches that nobody noticed because neither side
# had declared what it was writing. See p7_motifs/design-7.md.
#
# The edge table is the interaction-graph analogue of PARTICLES' node
# table: one row per directed edge (target particle i <- source particle
# j) at one (model, checkpoint, prompt, layer, head). Motif counts are
# aggregations over it, not a separate producer.
# ---------------------------------------------------------------------------

PHASE7 = {
    "interaction_table": ArtifactSpec(
        kind="npz", filename="interaction_table.npz",
        required_keys=(
            "model", "checkpoint_step", "prompt_key", "layer", "head",
            "target", "source",
            "weight", "force_magnitude",
            "attractive_frac", "repulsive_frac",
            "offset", "pair_type",
        ),
        key_shape_hint={
            "target": "(n_edges,) int64 — target particle's token_position",
            "source": "(n_edges,) int64 — source particle's token_position",
            "weight": "(n_edges,) float64 — post-softmax attention A_ij",
            "force_magnitude": "(n_edges,) float64 — ||A_ij * V x_j||",
            "attractive_frac": "(n_edges,) float64 — force fraction in U_pos",
            "repulsive_frac": "(n_edges,) float64 — force fraction in U_neg",
            "offset": "(n_edges,) int64 — target - source",
            "pair_type": "(n_edges,) <U — induction | strict | same_content | neither",
        },
        description=(
            "Long-format typed interaction edges: one row per (model, "
            "checkpoint, prompt, layer, head, target, source). See "
            "core/interactions.py:InteractionTable for the full schema, "
            "including the optional rotational-channel columns (real_frac, "
            "imag_frac, NaN when Phase 2b projectors were not supplied) and "
            "the extra__* columns. Write/read via InteractionTable.save/load "
            "rather than np.savez/np.load directly. Edge tables are subject "
            "to a top-k-by-force retention cutoff recorded in the manifest; "
            "an absent edge is not a zero-force edge."
        ),
    ),
    "motif_counts": ArtifactSpec(
        kind="json", filename="motif_counts.json",
        required_keys=("motif_alphabet_version", "counts", "nulls", "verdicts",
                       "degenerate_prompts", "force_cutoff"),
        description=(
            "Per-motif counts with their N1/N2/N3 null comparisons and "
            "verdicts. `degenerate_prompts` lists prompts excluded by "
            "core.battery_structure's four degeneracy modes and why — an "
            "empty list is a claim, not an omission. `force_cutoff` records "
            "the retention threshold the counts were computed under, and "
            "whether it was placed or calibrated (standing rule 6)."
        ),
    ),
    "formation_curve": ArtifactSpec(
        kind="json", filename="formation_curve.json",
        required_keys=("checkpoint_steps", "motif_strength",
                       "behavioral_induction_score", "independence_source"),
        description=(
            "Motif strength and behavioral induction score on a shared "
            "checkpoint axis (P-I1). `independence_source` states which of "
            "the three independence sources (two_stage | force_channel | "
            "particle_event) carries the association — see PREDICTIONS.md's "
            "Phase 7 adjudication constraint 2. It is required because a "
            "result that cannot name one has measured the same quantity "
            "twice."
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
# Phase 2 — verified against p2_eigenspectra/weights.py's
# save_weight_decomposition (the writer) and run_2.py's own header/output
# list. Two kinds of artifact: weight-decomposition files written per
# *model* into a shared weights dir, and per-run files written into each
# run directory.
#
# Filename templating: the weight files embed the model stem
# (model_name.replace("/", "_")) — ArtifactSpec.filename holds the
# "{stem}" template; use phase2_weight_path() below rather than
# artifact_path() for these four.
#
# NPZ key templating: keys are per-layer-templated —
#   shared (ALBERT):   ov_total_shared,     ov_head{h}_shared
#   per-layer (GPT-2): ov_total_layer_{i},  ov_head{h}_layer_{i}
# plus (when the model was passed to the writer) raw QK arrays via
# _add_qk_arrays_to_decomposition. Templated keys can't go in
# required_keys (they depend on n_layers/n_heads); key_shape_hint
# documents the pattern, and validate_artifact checks only file
# existence + parseability for these. This registration is what the
# "Phase 5 OV values always n/a" bug needed: run_5.py's
# _load_ov_head_matrices now parses exactly these key patterns — keep
# both in sync through this spec, not through two independent comments.
# ---------------------------------------------------------------------------

PHASE2_WEIGHTS = {
    "ov_weights": ArtifactSpec(
        kind="npz", filename="ov_weights_{stem}.npz",
        key_shape_hint={
            "ov_total_shared | ov_total_layer_{i}": "(d, d)",
            "ov_head{h}_shared | ov_head{h}_layer_{i}": "(d, d) composed W_V@W_O per head",
        },
        description="OV matrices (total + per-head) and raw W_Q/W_K per head.",
    ),
    "ov_decomp": ArtifactSpec(
        kind="npz", filename="ov_decomp_{stem}.npz",
        key_shape_hint={
            "eig_real_* / eig_imag_*": "(d,)",
            "schur_Z_*": "(d, d)",
            "sym_evals_*": "(d,)", "sym_evecs_*": "(d, d)",
        },
        description="Eigenvalues, Schur vectors, symmetric eigenvectors; *_shared or *_layer_{i}.",
    ),
    "ov_projectors": ArtifactSpec(
        kind="npz", filename="ov_projectors_{stem}.npz",
        key_shape_hint={
            "schur_attract_* / schur_repulse_*": "(d, d)",
            "sym_attract_* / sym_repulse_*": "(d, d)",
        },
        description="Subspace projectors; *_shared or *_layer_{i}.",
    ),
    "ov_summary": ArtifactSpec(
        kind="json", filename="ov_summary_{stem}.json",
        required_keys=("d_model", "d_head", "n_heads", "is_per_layer", "layers"),
        description="Scalar summaries per layer (frac_attractive, schur dims, spectral norms...).",
    ),
}

PHASE2 = {
    "attn_deltas_raw": ArtifactSpec(
        kind="npz", filename="attn_deltas_raw.npz",
        description="Per-layer attention deltas (parallel-decomposition input for 2i and crosscoder training).",
    ),
    "ffn_deltas_raw": ArtifactSpec(
        kind="npz", filename="ffn_deltas_raw.npz",
        description="Per-layer FFN deltas.",
    ),
    "verdict": ArtifactSpec(
        kind="json", filename="verdict.json",
        description="Per-run verdict record; cross-run aggregate is p2_eigenspectra_cross_run.json (session level).",
    ),
}

PHASE2_SESSION = {
    "cross_run": ArtifactSpec(
        kind="json", filename="p2_eigenspectra_cross_run.json",
        description="List of per-run verdicts, one file per output dir.",
    ),
}

# Subexperiment JSONs written under each run's subexperiments/ dir
# (run_2.py header): trajectory.json, layer_v_events.json, head_ov.json,
# decomposed_violations.json, ffn_subspace.json, continuous_correlations.json,
# ov_norm_confound.json, zone_comparison.json, attractive_zone_violations.json.
PHASE2_SUBEXPERIMENTS = {
    name: ArtifactSpec(kind="json", filename=f"{name}.json",
                       description=f"Phase 2 subexperiment output: {name}.")
    for name in (
        "trajectory", "layer_v_events", "head_ov", "decomposed_violations",
        "ffn_subspace", "continuous_correlations", "ov_norm_confound",
        "zone_comparison", "attractive_zone_violations",
    )
}

# ---------------------------------------------------------------------------
# Phase 5b — verified against run_5b.py / logit_cache.py's actual writes.
# ---------------------------------------------------------------------------

PHASE5B = {
    "logit_cache": ArtifactSpec(
        kind="npz", filename="logit_cache.npz",
        description="Cached logits for steering readout (logit_cache.py).",
    ),
    "fit_summary": ArtifactSpec(
        kind="json", filename="fit_summary.json",
        description="Manifold fit summary.",
    ),
    "mh_params": ArtifactSpec(
        kind="npz", filename="mh_params.npz",
        description="Manifold-hypothesis fit parameters.",
    ),
    "isometry": ArtifactSpec(
        kind="json", filename="isometry.json",
        description="Isometry test results (flat).",
    ),
    "isometry_mds": ArtifactSpec(
        kind="npz", filename="isometry_mds.npz",
        description="MDS embeddings backing the isometry test.",
    ),
    "merge_teleportation": ArtifactSpec(
        kind="json", filename="merge_teleportation.json",
        description="Merge-teleportation subspace test results.",
    ),
    "subspace_isometry": ArtifactSpec(
        kind="json", filename="subspace_isometry.json",
        description="Subspace isometry results (needs phase 2 ov_projectors).",
    ),
}

# ---------------------------------------------------------------------------
# Phase 6 — verified against p6_subspace/p6_io.py's save_subresult:
# every subexperiment writes {name}.json + {name}.summary.txt into the
# run's subresult dir. Names from p6_io.py's own header plus run_6.py's
# registered subexperiments.
# ---------------------------------------------------------------------------

def _p6_pair(name: str, desc: str) -> dict:
    return {
        name: ArtifactSpec(kind="json", filename=f"{name}.json", description=desc),
        f"{name}_summary": ArtifactSpec(kind="txt", filename=f"{name}.summary.txt",
                                        description=f"Human-readable summary for {name}."),
    }

PHASE6 = {}
for _n, _d in (
    ("subspace_build",  "Projector diagnostics (U_A/U_S/U_neg construction)."),
    ("head_classify",   "Track A head classification."),
    ("qk_decompose",    "Track A QK decomposition."),
    ("dissociation",    "Track C double-dissociation causal test."),
):
    PHASE6.update(_p6_pair(_n, _d))


def phase2_weight_path(weights_dir, name: str, model_name: str) -> Path:
    """Path for a Phase 2 weight-decomposition artifact — these are keyed
    by model stem, not run dir. Mirrors save_weight_decomposition's
    stem = model_name.replace('/', '_')."""
    spec = get_spec("phase2_weights", name)
    stem = model_name.replace("/", "_")
    return Path(weights_dir) / spec.filename.format(stem=stem)


# ---------------------------------------------------------------------------
# Phase 2b — registered as part of the Pythia rerun. Previously unregistered:
# every Phase 2b filename was hand-typed at the producer (`run_2i._write_subresult`)
# and again at any consumer, which is the drift this module exists to end.
#
# Filenames are `phase2b_*`, not `phase2i_*`. INDEX.md left the rename
# unscoped on the grounds that renaming a frozen artifact bought nothing —
# true while the files were frozen, and no longer true now that they are
# regenerated. The old names are listed in
# `p2b_imaginary/p2b_io.LEGACY_COMBINED_NAMES` so a pre-rewrite run directory
# can be RECOGNISED AND REFUSED rather than parsed as current output: the
# counting rule changed underneath them.
# ---------------------------------------------------------------------------

PHASE2B = {
    "block1a_rotational_spectrum": ArtifactSpec(
        kind="json", filename="block1a_rotational_spectrum.json",
        required_keys=("is_per_layer",),
        description=(
            "Schur block statistics per layer: complex/real counts, rotation "
            "angles, spectral-energy fractions, Henrici non-normality. "
            "Weights-only — no activations, no forward pass."
        ),
    ),
    "block1b_rescaled_comparison": ArtifactSpec(
        kind="json", filename="block1b_rescaled_comparison.json",
        required_keys=("frames", "comparison", "counting_rule", "interpretation"),
        key_shape_hint={
            "frames.{frame}.n_valid_layers":
                "int — truncation depth. Phase 2's verification item V1; "
                "dropping it is what made V1 unanswerable from the artifact.",
            "frames.remove_rotation.is_invariance_control":
                "bool — true. e^{-A} is orthogonal, so this frame reproduces "
                "`original` by construction. Never a causal result.",
            "counting_rule":
                "{rel_tol, gate_kind, gate_threshold, criterion} — the exact "
                "rule every count in this file was scored with.",
        },
        description=(
            "S/A rescaled-frame comparison. `comparison` holds elim_full and "
            "elim_signed only; rates are unclipped and may be None (refused)."
        ),
    ),
    "block2_hemispheric": ArtifactSpec(
        kind="json", filename="block2_hemispheric.json",
        description="Fiedler tracking + rotation-hemisphere alignment (conditional).",
    ),
    "block3_imaginary_ablation": ArtifactSpec(
        kind="json", filename="block3_imaginary_ablation.json",
        description=(
            "Depth-swept ablation of the rotational subspace. NOT the col(A) "
            "projector the first implementation used: a real antisymmetric A "
            "in even dimension is generically full rank, so that projector is "
            "the identity and the ablation zeroes every activation."
        ),
    ),
    "block4_layernorm_jacobian": ArtifactSpec(
        kind="json", filename="block4_layernorm_jacobian.json",
        description=(
            "LN-induced change to the effective operator diag(gamma) J_LN V. "
            "The first implementation's curvature regressor was identically 1 "
            "by algebra and its inflation ratio was saturated by a base "
            "fraction of ~0.98; both are redefined."
        ),
    ),
    "ffn_rotation": ArtifactSpec(
        kind="json", filename="ffn_rotation.json",
        description=(
            "FFN displacement projected onto rotation planes. On Pythia the "
            "input is core/sublayer_streams.py's parallel-residual streams "
            "(dx = attn_out + ffn_out, exact), not the frozen GPT-2-only "
            "ffn_deltas_*.npz this block originally required."
        ),
    ),
    "combined_results": ArtifactSpec(
        kind="json", filename="phase2b_results.json",
        description="All checkpoints, keyed by model stem; each entry carries checkpoint_step.",
    ),
    "combined_summary": ArtifactSpec(
        kind="txt", filename="phase2b_summary.txt",
        description="LLM-consumable cross-checkpoint summary.",
    ),
}

# ---------------------------------------------------------------------------
# Registry + validation
# ---------------------------------------------------------------------------

REGISTRY = {
    "phase1": PHASE1,
    "phase1_session": PHASE1_SESSION,
    "phase2": PHASE2,
    "phase2_session": PHASE2_SESSION,
    "phase2_subexperiments": PHASE2_SUBEXPERIMENTS,
    "phase2_weights": PHASE2_WEIGHTS,
    "phase2b": PHASE2B,
    "phase5b": PHASE5B,
    "phase6": PHASE6,
    "particles": PARTICLES,
    "phase7": PHASE7,
    "manifest": {"manifest": MANIFEST},
}

# Still unregistered, deliberately:
# PHASE4 : frozen-for-deletion (plan v2); registering would be new work on
#          code the freeze policy says gets none. The reintroduction
#          trigger, if ever hit, includes writing this contract first.
# PHASE5 : p5_single_mstate_analysis writes report-level outputs through
#          report.py; register from that writer when Phase 5's Pythia
#          rerun (execution-order item 11) touches it.


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
    """Expected path for a registered artifact inside a given run directory.
    Raises for stem-templated filenames (phase2_weights) — those need a
    model name; use phase2_weight_path instead."""
    spec = get_spec(phase, name)
    if "{stem}" in spec.filename:
        raise ValueError(
            f"Artifact {phase}/{name} has a model-stem-templated filename "
            f"({spec.filename!r}); use phase2_weight_path(weights_dir, name, model_name)."
        )
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