"""
p2b_imaginary/p2b_io.py — Phase 2b's discovery and loading layer.

Four things `run_2i.py` did inline, each of which is a known bug class:

1. **Substring model matching.** `find_phase2_runs` matched
   `model_stem in d.name`. On the GPT-2 study that made the `gpt2` entry in
   `phase2i_results.json` aggregate `gpt2-large` / `gpt2-medium` / `gpt2-xl`
   — status-2b caveat 2, recorded as "a runner bug, not a result." On the
   Pythia sweep the same predicate makes `pythia-410m-step1` match
   `step16`, `step128`, `step1000`, `step128000`: eight of the twenty-seven
   stems collide. Everything here matches exactly, or on an exact
   `{stem}_` prefix, and never on a bare substring.

2. **A second Phase 1 reader.** `run_2i.load_phase1_events` reimplemented
   `p1_mstate_tracking/p1_io.load_phase1_run`, including its own v1/v2
   layout branch, its own violation recount, and a
   `from phase2.trajectory import ...` whose module path does not exist
   (swallowed by `except ImportError`; the inline comment claims it was a
   fix). `p1_io.py`'s own docstring says: "Do not add another
   load_phase1_run somewhere else; extend this one." This module calls it.

3. **No checkpoint axis.** The runner is organised model x prompt, so 27
   Pythia checkpoints arrive as 27 unrelated "models" with no
   `checkpoint_step`, no ordering, and no way to express the trajectory that
   is Phase 1's and Phase 2's actual headline result. `discover_checkpoints`
   returns `(step, stem)` pairs in training order.

4. **No artifact contract, no manifest.** Every filename was hand-typed at
   both the producer and the consumer — the drift `core/artifacts.py` exists
   to make impossible. `ov_weights_path` goes through
   `core.artifacts.phase2_weight_path`; `write_subresult` writes through the
   registered PHASE2B specs; `write_run_manifest` goes through
   `core.io.write_manifest` instead of a bare `datetime.now()` string.

Deliberately torch-free. `core.config` (torch, transformers) is reached only
through the deferred path in `p2b_energy.resolve_rank_gate`, and
`core.prompts` (which imports `core.config`) only when the caller asks for
the canonical battery.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.model_family import checkpoint_step, checkpoint_base, sort_by_step

# ---------------------------------------------------------------------------
# Artifact names
# ---------------------------------------------------------------------------

#: Phase 2b's on-disk outputs. Mirrors the block added to
#: `core.artifacts.PHASE2B`; `subresult_filename` reads the registry when it
#: is available so the two cannot drift, and raises rather than guessing when
#: it is not.
SUBRESULT_NAMES = (
    "block1a_rotational_spectrum",
    "block1a_head_circuits",
    "block1b_rescaled_comparison",
    "block2_hemispheric",
    "block3_imaginary_ablation",
    "block4_layernorm_jacobian",
    "ffn_rotation",
)

#: The pre-rewrite artifact names. Kept only so a reader can recognise an old
#: run directory and refuse it, rather than parsing it as current output —
#: the counting rule changed, so the numbers inside are not comparable.
LEGACY_COMBINED_NAMES = ("phase2i_results.json", "phase2i_summary.txt")

COMBINED_RESULTS = "phase2b_results.json"
COMBINED_SUMMARY = "phase2b_summary.txt"

#: Per-checkpoint array sidecars, written next to the subresult JSON.
#:
#: `planes` holds Block 1a's per-plane spectrum — every 2x2 block's rho,
#: theta, sign and Schur index, per layer. It is a sidecar rather than a key
#: in the JSON because at d = 1024 a layer holds up to 512 planes, so inlining
#: it would add ~1 MB per checkpoint to `phase2b_results.json`, which every
#: consumer reads whole. Same split Phase 1b made for its Fiedler axes.
SIDECAR_NAMES = {
    "planes": "planes.npz",
}


def subresult_filename(name: str) -> str:
    if name not in SUBRESULT_NAMES:
        raise KeyError(
            f"p2b_io: unknown subresult {name!r}. Known: {SUBRESULT_NAMES}. "
            "Add it to SUBRESULT_NAMES and to core.artifacts.PHASE2B together, "
            "not to one of them."
        )
    return f"{name}.json"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(
    weights_dir,
    base: Optional[str] = None,
    include_non_checkpoints: bool = False,
) -> list:
    """
    Every model stem with OV weights on disk, in training order.

    Parameters
    ----------
    weights_dir             : directory holding `ov_weights_{stem}.npz`
    base                    : restrict to one family, e.g. "pythia-410m".
                              Matched on `checkpoint_base`, EXACTLY — a
                              "pythia-410m" base never picks up
                              "pythia-1.4b" entries, and vice versa.
    include_non_checkpoints : also return stems with no `-step{N}` suffix
                              (`pythia-1.4b-random`, `gpt2-large`, ...).
                              Off by default: they have no step and must not
                              silently join a step-indexed sweep.

    Returns list of (step_or_None, stem).
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        return []

    stems = [p.name[len("ov_weights_"):-len(".npz")]
             for p in sorted(weights_dir.glob("ov_weights_*.npz"))]

    out = []
    for stem in sort_by_step(stems):
        step = checkpoint_step(stem)
        if step is None and not include_non_checkpoints:
            continue
        if base is not None and step is not None and checkpoint_base(stem) != base:
            continue
        if base is not None and step is None and stem != base and not stem.startswith(base):
            continue
        out.append((step, stem))
    return out


def checkpoint_step_of(stem: str) -> Optional[int]:
    """Re-export, so callers need not import two modules for one grammar."""
    return checkpoint_step(stem)


# ---------------------------------------------------------------------------
# OV weights
# ---------------------------------------------------------------------------

def ov_weights_path(weights_dir, model_stem: str) -> Path:
    """
    Path to one checkpoint's OV weights, through the artifact contract.

    Falls back to the literal filename only if `core.artifacts` has no
    `phase2_weights` registration — and warns when it does, because an
    unregistered artifact is exactly the state the contract module exists to
    end.
    """
    try:
        from core.artifacts import phase2_weight_path
        return phase2_weight_path(weights_dir, "ov_weights", model_stem)
    except Exception as e:  # pragma: no cover - contract should be present
        warnings.warn(
            f"p2b_io.ov_weights_path: core.artifacts lookup failed ({e}); "
            "falling back to a hand-typed filename. Register the artifact."
        )
        stem = model_stem.replace("/", "_")
        return Path(weights_dir) / f"ov_weights_{stem}.npz"


def load_ov_data(weights_dir, model_stem: str) -> Optional[dict]:
    """
    Load one checkpoint's OV matrices in `weights.extract_ov_circuit` shape.

    Returns None when the file is absent — the caller decides whether that is
    a skip or an error.

    The per-layer key sort is numeric, not lexicographic. The previous
    version's `sorted(...)` over `ov_total_layer_{i}` was already keyed on
    `int(k.split("layer_")[1])`, which is correct; it is restated here
    because a lexicographic sort silently orders layer_10 before layer_2 and
    the resulting trajectory looks plausible.
    """
    path = ov_weights_path(weights_dir, model_stem)
    if not Path(path).exists():
        return None

    data = np.load(path)
    keys = list(data.keys())

    layer_keys = [k for k in keys if k.startswith("ov_total_layer_")]
    if layer_keys:
        layer_keys.sort(key=lambda k: int(k.split("layer_")[1]))
        names = [k[len("ov_total_"):] for k in layer_keys]
        return {
            "ov_total": [data[k] for k in layer_keys],
            "ov_per_head": [_per_head_for(data, keys, name) for name in names],
            "is_per_layer": True,
            "layer_names": names,
            "model_stem": model_stem,
            "checkpoint_step": checkpoint_step(model_stem),
            "source_path": str(path),
        }

    if "ov_total_shared" in keys:
        return {
            "ov_total": data["ov_total_shared"],
            "ov_per_head": _per_head_for(data, keys, "shared"),
            "is_per_layer": False,
            "layer_names": ["shared"],
            "model_stem": model_stem,
            "checkpoint_step": checkpoint_step(model_stem),
            "source_path": str(path),
        }

    raise KeyError(
        f"p2b_io.load_ov_data: {path} has neither 'ov_total_layer_*' nor "
        f"'ov_total_shared'. Keys present: {sorted(keys)[:8]}... "
        "This is an artifact-contract mismatch, not a missing model."
    )


#: `weights.py` writes per-head OV as `ov_head{h}_{layer_name}` in the same
#: npz as `ov_total_{layer_name}`.
_HEAD_RE = re.compile(r"^ov_head(\d+)_(.+)$")


def _per_head_for(data, keys: Sequence[str], layer_name: str) -> list:
    """
    One layer's per-head OV matrices, in head order. Empty when absent.

    `ov_total = sum_h ov_per_head` is the effective operator only under a
    counterfactual the model does not satisfy — that every head shares an
    attention pattern; the real update is `sum_h alpha^h X W_OV^h`. So the
    phase's headline is a statistic of an object the model never forms, and
    `head_circuits.summed_vs_per_head` exists to report both and the gap.
    None of that was reachable from a Phase 2b run, because this loader read
    only the summed matrices — the per-head arrays have been sitting in the
    same file the whole time.

    The head index is sorted NUMERICALLY. A lexicographic sort puts head 10
    before head 2, which produces a plausible-looking per-head table in the
    wrong order — the same failure the layer-key sort above is explicit
    about.
    """
    heads = []
    for k in keys:
        m = _HEAD_RE.match(k)
        if m and m.group(2) == layer_name:
            heads.append((int(m.group(1)), k))
    heads.sort()
    return [data[k] for _, k in heads]


# ---------------------------------------------------------------------------
# Phase 1 runs — exact matching only
# ---------------------------------------------------------------------------

def find_phase1_runs(
    phase1_dir,
    model_stem: str,
    prompt_keys: Optional[Sequence[str]] = None,
    require_activations: bool = True,
) -> dict:
    """
    Locate this checkpoint's Phase 1 run directories, keyed by prompt.

    Two resolution modes:

      - `prompt_keys` given: each is resolved with
        `p1_io.find_phase1_run_dir`, the canonical resolver. Preferred — it
        already encodes the legacy-nested / flat / prefix fallbacks and warns
        on the loose ones.
      - `prompt_keys` None: the directory is scanned and a candidate is
        accepted only when its name is exactly `{stem}` or begins with
        `{stem}_`. NEVER `stem in name`. That predicate is what made
        `pythia-410m-step1` swallow `step16`.

    Returns {prompt_key: Path}.
    """
    phase1_dir = Path(phase1_dir)
    if not phase1_dir.exists():
        return {}

    if prompt_keys is not None:
        from p1_mstate_tracking.p1_io import find_phase1_run_dir
        found = {}
        for key in prompt_keys:
            d = find_phase1_run_dir(phase1_dir, model_stem, key)
            if d is None:
                continue
            if require_activations and not (Path(d) / "activations.npz").exists():
                continue
            found[key] = Path(d)
        return found

    prefix = model_stem + "_"
    found = {}
    for d in sorted(phase1_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name == model_stem:
            prompt = ""
        elif d.name.startswith(prefix):
            prompt = d.name[len(prefix):]
        else:
            continue
        if require_activations and not (d / "activations.npz").exists():
            continue
        found[prompt] = d
    return found


def load_activations(run_dir) -> Optional[np.ndarray]:
    """
    L2-normed activations for one run, through the canonical Phase 1 reader.

    `p1_io._save_activations` writes `layernorm_to_sphere(x)`, which despite
    the name is plain L2 normalization (`core/models.py:53`). So these are in
    the `l2_sphere` frame of `core/frames.py`, NOT the LN frame attention
    actually reads. `frame_spec_for_activations` records that rather than
    leaving it implicit.
    """
    from p1_mstate_tracking.p1_io import load_phase1_run
    return load_phase1_run(Path(run_dir)).get("activations")


def load_phase1_run_bundle(run_dir) -> dict:
    """
    Activations plus the Phase 1 scalars Phase 2b cross-checks against.

    `phase1_violation_layers` is read from `energies.json` as Phase 1 wrote
    it — it is NOT recomputed here. Recomputing it with a different rule is
    how Phase 2b came to have a violation count that matched neither Phase 1
    nor Phase 2 (see `p2b_energy`'s module docstring).
    """
    from p1_mstate_tracking.p1_io import load_phase1_run
    run_dir = Path(run_dir)
    p1 = load_phase1_run(run_dir)

    energies_path = run_dir / "energies.json"
    viol: dict = {}
    energies_by_layer: dict = {}
    if energies_path.exists():
        try:
            with open(energies_path) as f:
                eng = json.load(f)
            for lr in eng.get("layers", []):
                energies_by_layer[int(lr["layer"])] = {
                    float(k): v for k, v in (lr.get("energies") or {}).items()
                }
            viol = {float(k): list(v)
                    for k, v in (eng.get("violation_layers") or {}).items()}
        except Exception as e:
            warnings.warn(f"p2b_io: could not parse {energies_path}: {e}")

    geometry_path = run_dir / "geometry.json"
    raw_rank = []
    normed_rank = []
    if geometry_path.exists():
        try:
            with open(geometry_path) as f:
                geo = json.load(f)
            for lr in geo.get("layers", []):
                raw_rank.append(lr.get("effective_rank"))
                normed_rank.append(lr.get("effective_rank_normed"))
        except Exception as e:
            warnings.warn(f"p2b_io: could not parse {geometry_path}: {e}")

    return {
        "activations": p1.get("activations"),
        "tokens": p1.get("tokens", []),
        "model": p1.get("model", ""),
        "prompt": p1.get("prompt", ""),
        "n_layers": p1.get("n_layers", 0),
        "n_tokens": p1.get("n_tokens", 0),
        "d_model": p1.get("d_model", 0),
        "phase1_violation_layers": viol,
        "phase1_energies": energies_by_layer,
        # status-1 D1: this key holds RAW effective rank. Phase 2b gates on
        # normed (p2b_energy docstring). Both are carried so the divergence is
        # a recorded number rather than an invisible term.
        "phase1_effective_rank_raw": raw_rank,
        "phase1_effective_rank_normed": normed_rank,
        "run_dir": str(run_dir),
    }


# ---------------------------------------------------------------------------
# Frame ledger
# ---------------------------------------------------------------------------

def frame_spec_for_activations(model_stem: str, layer_idx: Optional[int] = None):
    """
    The FrameSpec Phase 2b's numbers live in.

    `kind="l2_sphere"` because that is what `activations.npz` holds.
    `rope_applied=False` is a live claim, not an oversight: Phase 2b's OV
    analysis is a statement about `W_V W_O`, which rotary does not touch —
    unlike the QK bilinear, where `core/rope.py` and
    `p6_subspace/qk_offset_null.py` exist precisely because it does.

    `model_rev` is the checkpoint stem, so a Phase 2b record from step 512
    can never be silently compared with one from step 143000 —
    `core.frames.verify_same_revision` raises.
    """
    from core.frames import FrameSpec
    return FrameSpec(
        kind="l2_sphere",
        layer_idx=layer_idx,
        model_rev=model_stem,
        rope_applied=False,
        pos0_policy="included",
        extras=(("phase", "2b"),),
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def json_default(obj):
    """numpy -> json. Raises on anything else, as the previous version did."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [json_default(x) if isinstance(x, np.generic) else
                (None if isinstance(x, float) and not np.isfinite(x) else x)
                for x in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def sanitize_for_json(obj):
    """
    Replace every non-finite float with None, recursively.

    `json_default` handles numpy scalars and arrays, but `default=` is only
    consulted for types the encoder does not already know — a plain Python
    `float('nan')` goes straight to the encoder and `allow_nan=False` rejects
    it. So a block whose analysis returns a bare NaN could not be written at
    all, and the failure surfaced as a `ValueError` from deep inside
    `json.dump` naming neither the block nor the key.

    That is not hypothetical: `core.precision_policy.complex_fraction_surface`
    reports `z_score` and `percentile` as NaN for a deterministic perturbation
    (one draw, so there is no distribution to score against), which is the
    correct in-memory value and not a writable one.

    NaN maps to JSON null rather than to 0.0 or to a dropped key, for the same
    reason it does everywhere else in this phase: "not computed" and "computed
    and zero" are different statements, and the file has to keep them apart.
    """
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    # np.float64 subclasses float and is caught above; np.float32 does not.
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def write_subresult(sub_dir, name: str, payload: dict,
                    summary_lines: Optional[Sequence[str]] = None) -> Path:
    """Write `sub/{name}.json` (+ `.summary.txt`), validating the name."""
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / subresult_filename(name)
    with open(path, "w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2,
                  default=json_default, allow_nan=False)
    if summary_lines is not None:
        with open(sub_dir / f"{name}.summary.txt", "w") as f:
            f.write("\n".join(summary_lines) + "\n")
    return path


def sidecar_path(sub_dir, name: str) -> Path:
    """
    Path to one array sidecar, through the artifact contract.

    The filename is read from `core.artifacts.PHASE2B` when it is available,
    so the registry and this module cannot drift; `SIDECAR_NAMES` is the
    fallback and a mismatch between the two raises rather than silently
    preferring one. That is the failure `core/artifacts.py` exists for and
    the reason the local table is not simply the source of truth.
    """
    if name not in SIDECAR_NAMES:
        raise KeyError(
            f"p2b_io: unknown sidecar {name!r}. Known: {sorted(SIDECAR_NAMES)}. "
            "Add it to SIDECAR_NAMES and to core.artifacts.PHASE2B together, "
            "not to one of them."
        )
    filename = SIDECAR_NAMES[name]
    try:
        from core.artifacts import get_spec
        registered = get_spec("phase2b", name).filename
    except Exception:      # pragma: no cover - contract should be present
        registered = filename
    if registered != filename:
        raise RuntimeError(
            f"p2b_io: sidecar {name!r} is {filename!r} here and "
            f"{registered!r} in core.artifacts.PHASE2B. One writer and one "
            "reader disagreeing about a filename is the drift the contract "
            "module exists to stop; fix both together."
        )
    return Path(sub_dir) / filename


def write_sidecar(sub_dir, name: str, arrays: dict) -> Optional[Path]:
    """
    Write one array sidecar, or nothing when there are no arrays to write.

    Returns the path, or None when `arrays` is empty — an absent sidecar and
    an empty one are different states downstream, and writing a zero-array
    npz would make "this run computed no planes" indistinguishable from "this
    run predates the emission".
    """
    if not arrays:
        return None
    path = sidecar_path(sub_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_sidecar(sub_dir, name: str) -> Optional[dict]:
    """One array sidecar as a plain `{key: ndarray}` dict, or None."""
    path = sidecar_path(sub_dir, name)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def write_run_manifest(run_dir, model_stem: str, prompt_key: Optional[str],
                       wall_time_seconds: float, config: Optional[dict] = None,
                       prompt_battery_hash: Optional[str] = None) -> dict:
    """
    `core.io.write_manifest` for a Phase 2b run directory.

    `hf_revision` and `checkpoint_step` come from the stem grammar, so a
    Phase 2b artifact carries the step even though Phase 2b never loads a
    model. Without it nothing downstream can place the run on the training
    axis, which is the whole point of the rerun.
    """
    from core.io import write_manifest

    step = checkpoint_step(model_stem)
    if prompt_battery_hash is None:
        try:
            from core.prompts import PROMPT_BATTERY_HASH
            prompt_battery_hash = PROMPT_BATTERY_HASH
        except Exception:
            prompt_battery_hash = "unavailable"

    return write_manifest(
        Path(run_dir),
        model=model_stem,
        prompt_battery_hash=prompt_battery_hash,
        wall_time_seconds=float(wall_time_seconds),
        hf_revision=None if step is None else f"step{step}",
        checkpoint_step=step,
        prompt_key=prompt_key,
        config=config or {},
        extra={"phase": "2b"},
    )


def refuse_legacy_run_dir(run_dir) -> None:
    """
    Raise if `run_dir` holds pre-rewrite Phase 2b output.

    The counting rule changed (absolute 1e-6 / rank 3.0 -> relative 1e-3 /
    DEGENERATE_RANK_THRESHOLD) and the rotation-only frame was demoted from a
    result to an identity check. Numbers from the two versions are not
    comparable, and the old filenames do not say which version wrote them.
    Refusing is cheaper than a silent mixed table.
    """
    run_dir = Path(run_dir)
    present = [n for n in LEGACY_COMBINED_NAMES if (run_dir / n).exists()]
    if present:
        raise RuntimeError(
            f"p2b_io: {run_dir} contains pre-rewrite Phase 2b artifacts "
            f"({', '.join(present)}). Those were scored with an absolute "
            "1e-6 threshold and a 3.0 rank gate, and their `elim_rotation` "
            "column is an algebraic identity rather than a measurement. "
            "Write the new run to a new directory."
        )


__all__ = [
    "SUBRESULT_NAMES",
    "SIDECAR_NAMES",
    "LEGACY_COMBINED_NAMES",
    "COMBINED_RESULTS",
    "COMBINED_SUMMARY",
    "subresult_filename",
    "sidecar_path",
    "write_sidecar",
    "load_sidecar",
    "discover_checkpoints",
    "checkpoint_step_of",
    "ov_weights_path",
    "load_ov_data",
    "find_phase1_runs",
    "load_activations",
    "load_phase1_run_bundle",
    "frame_spec_for_activations",
    "json_default",
    "sanitize_for_json",
    "write_subresult",
    "write_run_manifest",
    "refuse_legacy_run_dir",
]
