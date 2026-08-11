"""
p1b_io.py — Phase 1b's reader/writer, following the project's
one-io-module-per-phase convention (p1_io.py, p5_io.py, p5b_io.py, p6_io.py).

Two jobs.

**1. Read Phase 1 output through Phase 1's own reader.**

`run_1b.py::_load_phase1_xref` hand-built `phase1_dir / stem` and hardcoded
three filenames. Two things were wrong with that, one silently:

  - The stem it built for ALBERT extended runs was
    `{model}_{prompt}_d{depth}`. Phase 1 writes
    `{model}_{depth}iter_{prompt}` (`run_1.py`, `effective_model_name`). So
    the directory never resolved for any ALBERT depth, `hdbscan_labels` was
    never loaded, and Block 2's nesting test silently had nothing to test.
    status-1b.md records that outcome as "Inconclusive for ALBERT". It was a
    path bug, not an inconclusive measurement.
  - Hardcoded filenames are the artifact-contract bug class core/artifacts.py
    exists to kill. `p1_io.find_phase1_run_dir` already resolves both the
    nested and flat layouts and warns when it falls back to a loose match;
    `p1_io.load_phase1_run` already normalises hdbscan_labels to a list
    indexed by layer, which is what Block 2 wants and what the old loader's
    string-keyed dict was not.

So this module resolves and loads through p1_io, and adds only what Phase 1b
needs on top: the Fiedler vectors Phase 1 already saved, and the per-beta
energy-violation layers.

**2. Reuse rather than recompute.**

Phase 1 saves `activations.npz` (unit-norm, plus the `norms` that projection
discards) and `fiedler_vecs.npz` (the second Laplacian eigenvector per
layer). Phase 1b was re-running every forward pass to recompute both, which
costs a model load per run and — worse — guarantees that the Fiedler vector
Phase 1b analyses is not the one Phase 1 clustered on. `load_phase1_context`
returns both, so `run_1b --from-phase1` needs no torch and no GPU, and
analyses exactly the numbers Phase 1 produced.

The saved Fiedler vectors come from Phase 1's graph
(connectivity_floor = 0.0). Phase 1b's own default floor is 1e-4. That
difference is now recorded on every record rather than being invisible; see
bipartition_detect.CONNECTIVITY_FLOOR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p1_mstate_tracking.p1_io import find_phase1_run_dir, load_phase1_run


# ---------------------------------------------------------------------------
# Phase 1 artifacts
# ---------------------------------------------------------------------------

def load_fiedler_vecs(run_dir: Path, n_layers: int | None = None):
    """
    fiedler_vecs.npz -> (n_layers, n_tokens) float array, or None.

    Keys are "fiedler_L{i}". Layers with no saved vector become zero rows,
    matching Block 0's own convention for an invalid layer, and are reported
    in `missing_layers` so a caller can mark them invalid rather than
    silently analysing zeros.
    """
    path = Path(run_dir) / "fiedler_vecs.npz"
    if not path.exists():
        return None, []

    try:
        data = np.load(path)
    except Exception:
        return None, []

    by_layer: dict = {}
    for key in data.files:
        if not key.startswith("fiedler_L"):
            continue
        try:
            by_layer[int(key[len("fiedler_L"):])] = np.asarray(data[key], dtype=np.float64)
        except ValueError:
            continue
    if not by_layer:
        return None, []

    n_layers = (max(by_layer) + 1) if n_layers is None else int(n_layers)
    n_tokens = len(next(iter(by_layer.values())))
    out = np.zeros((n_layers, n_tokens), dtype=np.float64)
    missing = []
    for L in range(n_layers):
        v = by_layer.get(L)
        if v is None or v.shape[0] != n_tokens:
            missing.append(L)
            continue
        out[L] = v
    return out, missing


def load_energy_violations(run_dir: Path) -> dict:
    """
    events.json -> {beta_str: [layer, ...]}, or {}.

    Returned per beta rather than pre-unioned. The old loader unioned across
    every beta and called the result "beta=1.0, falling back to the union" —
    it never actually preferred beta=1.0, so a violation that only occurs at
    beta=9 was indistinguishable from one at the canonical beta. Callers
    choose; `violation_layers_for` implements the documented preference.
    """
    path = Path(run_dir) / "events.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    viols = data.get("energy_violations", {}) or {}
    return {str(k): [int(x) for x in v] for k, v in viols.items()}


def violation_layers_for(violations: dict, beta: float | None = 1.0) -> set:
    """
    Pick one beta's violation layers, falling back to the union.

    beta=None means "union across all betas" explicitly rather than by
    accident.
    """
    if not violations:
        return set()
    if beta is None:
        out: set = set()
        for layers in violations.values():
            out.update(int(x) for x in layers)
        return out

    for key in (f"{beta}", f"{float(beta)}", f"{beta:g}"):
        if key in violations:
            return {int(x) for x in violations[key]}

    out = set()
    for layers in violations.values():
        out.update(int(x) for x in layers)
    return out


def load_phase1_context(
    phase1_dir: Path | str,
    model_name: str,
    prompt_key: str,
    beta: float | None = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Everything Phase 1b wants from a Phase 1 run, resolved by (model, prompt).

    Returns
    -------
    dict with (any of which may be absent):
      run_dir           Path actually used
      activations       (n_layers, n_tokens, d) unit-norm, from Phase 1
      norms             (n_layers, n_tokens) the radii the sphere projection
                        discarded, when the run recorded them
      tokens            list[str]
      hdbscan_labels    {layer: (n_tokens,) int array} — dict form, which is
                        what Block 2 consumes
      merge_indices     set[int] of layer_from indices with merges
      violation_layers  set[int] for the requested beta
      violations_by_beta {beta_str: [layer, ...]}
      plateau_layers    list[int]
      fiedler_vecs      (n_layers, n_tokens) from Phase 1's own graph
      fiedler_missing   layers with no saved vector
      n_layers/n_tokens/d_model
    """
    out: dict = {}
    run_dir = find_phase1_run_dir(Path(phase1_dir), model_name, prompt_key)
    if run_dir is None:
        if verbose:
            print(f"    [xref] no Phase 1 run found for "
                  f"({model_name}, {prompt_key}) under {phase1_dir}")
        return out

    out["run_dir"] = run_dir
    try:
        p1 = load_phase1_run(run_dir)
    except Exception as exc:
        if verbose:
            print(f"    [xref] load_phase1_run failed for {run_dir}: {exc}")
        return out

    if p1.get("activations") is not None:
        out["activations"] = np.asarray(p1["activations"])
    out["tokens"]     = p1.get("tokens") or []
    out["n_layers"]   = int(p1.get("n_layers") or 0)
    out["n_tokens"]   = int(p1.get("n_tokens") or 0)
    out["d_model"]    = int(p1.get("d_model") or 0)

    # p1_io normalises hdbscan_labels to a list indexed by layer; Block 2
    # takes a dict. Convert once, here, rather than at the call site.
    labels = p1.get("hdbscan_labels")
    if labels:
        out["hdbscan_labels"] = {
            L: np.asarray(v, dtype=np.int32) for L, v in enumerate(labels)
            if v is not None
        }

    merges = p1.get("merge_layers") or []
    if merges:
        out["merge_indices"] = {int(x) for x in merges}

    plateaus = p1.get("plateau_layers") or []
    if plateaus:
        out["plateau_layers"] = [int(x) for x in plateaus]

    viols = load_energy_violations(run_dir)
    if viols:
        out["violations_by_beta"] = viols
        out["violation_layers"]   = violation_layers_for(viols, beta=beta)

    fvecs, missing = load_fiedler_vecs(run_dir, n_layers=out.get("n_layers") or None)
    if fvecs is not None:
        out["fiedler_vecs"]    = fvecs
        out["fiedler_missing"] = missing

    # The norms Phase 1 now records alongside the unit vectors. Present only
    # for runs written after that change; absent is not an error.
    acts_path = Path(run_dir) / "activations.npz"
    if acts_path.exists():
        try:
            data = np.load(acts_path)
            if "norms" in data.files:
                out["norms"] = np.asarray(data["norms"])
        except Exception:
            pass

    if verbose:
        loaded = [k for k in ("activations", "norms", "hdbscan_labels",
                              "merge_indices", "violation_layers",
                              "plateau_layers", "fiedler_vecs") if k in out]
        print(f"    [xref] {run_dir.name}: {loaded or 'nothing'}")
    return out


# ---------------------------------------------------------------------------
# Particle records
# ---------------------------------------------------------------------------

def hemisphere_particle_rows(
    model: str,
    prompt_key: str,
    checkpoint_step: int,
    block0: dict,
    block1: dict,
    block2_json: dict,
    tokens: list | None = None,
    cluster_labels: dict | None = None,
) -> tuple:
    """
    Phase 1b's per-(layer, token) output as columns for core.particles.

    `per_token` in the old per-run JSON was already a particle table wearing
    a different shape: one record per token carrying a full per-layer
    trajectory list. Emitting it long instead — one row per (layer, token) —
    is what makes the Phase 5c questions groupbys rather than new code, and
    is the canonical artifact shape the v2 plan names (core infrastructure
    item 4).

    Returns
    -------
    (columns, extra) as ParticleTable takes them. Two dicts, not one: the
    schema columns go in the first, everything Phase 1b adds goes in the
    second, and `save` writes the second with an `extra__` prefix. String
    columns are fixed-width unicode rather than dtype=object because
    ParticleTable.save refuses object arrays (they pickle, and `load` uses
    allow_pickle=False).
    """
    from core.particles import default_population_tag

    n_layers = int(block0["n_layers"])
    n_tokens = int(block0["n_tokens"])
    aligned  = block1["aligned_assignments"]
    fvecs    = block0["fiedler_vecs"]
    valid    = block0["valid"]
    regime   = block0["regime"]
    rel      = block0.get("regime_relative")

    traj_by_idx = {r["token_idx"]: r for r in block2_json.get("per_token", [])}

    layer_ix, token_ix = np.meshgrid(
        np.arange(n_layers), np.arange(n_tokens), indexing="ij")
    layer_ix = layer_ix.ravel()
    token_ix = token_ix.ravel()
    N = layer_ix.size

    cluster = np.full(N, -1, dtype=np.int64)
    if cluster_labels:
        for L in range(n_layers):
            row = cluster_labels.get(L)
            if row is None:
                continue
            row = np.asarray(row, dtype=np.int64)
            k = min(len(row), n_tokens)
            cluster[L * n_tokens:L * n_tokens + k] = row[:k]

    tok_strs = ([str(tokens[t]) if t < len(tokens) else "" for t in token_ix]
                if tokens is not None else [""] * N)

    border = np.full(N, np.nan)
    stabil = np.full(N, np.nan)
    for j in range(N):
        rec = traj_by_idx.get(int(token_ix[j]))
        if not rec:
            continue
        if rec.get("border_index") is not None:
            border[j] = float(rec["border_index"])
        if rec.get("stability_score") is not None:
            stabil[j] = float(rec["stability_score"])

    columns = {
        "model":           np.array([model] * N),
        "checkpoint_step": np.full(N, int(checkpoint_step), dtype=np.int64),
        "prompt_key":      np.array([prompt_key] * N),
        "layer":           layer_ix.astype(np.int64),
        "token_position":  token_ix.astype(np.int64),
        "cluster_label":   cluster,
        "population":      np.asarray(default_population_tag(cluster)),
        "token_str":       np.array(tok_strs),
    }

    extra = {
        "hemisphere":      np.array([int(aligned[L, t])
                                     for L, t in zip(layer_ix, token_ix)],
                                    dtype=np.int64),
        "fiedler_value":   np.array([float(fvecs[L, t])
                                     for L, t in zip(layer_ix, token_ix)],
                                    dtype=np.float64),
        "border_index":    border,
        "stability_score": stabil,
        "layer_valid":     np.array([int(bool(valid[L])) for L in layer_ix],
                                    dtype=np.int64),
        "layer_regime":    np.array([str(regime[L]) for L in layer_ix]),
    }
    if rel is not None:
        extra["layer_regime_relative"] = np.array([str(rel[L]) for L in layer_ix])

    return columns, extra


def write_particle_table(columns: dict, extra: dict, path: Path):
    """
    Save hemisphere particle columns as a ParticleTable npz.

    Returns the path written, or None if core.particles rejects the columns
    — which it should, loudly, if the schema drifts. Swallowing that would
    reproduce the artifact-contract failure this is meant to avoid, so the
    exception is printed rather than hidden.
    """
    from core.particles import ParticleTable
    try:
        table = ParticleTable(columns, extra)
        table.save(Path(path))
        return Path(path)
    except Exception as exc:
        print(f"    [particles] refused to write {path}: {exc}")
        return None
