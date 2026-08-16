"""
p2b_imaginary/visualization/_fixture.py — a synthetic Phase 2b output
directory.

A test aid, and the fastest way to see the whole catalogue without a Pythia
checkpoint sweep. **The weights are random and no result should ever be read
off the numbers.** Everything else is real: the filenames come from
`p2b_io.subresult_filename`, the per-checkpoint JSON from
`run_2b.run_block_1a`, the per-prompt JSON from `run_2b.run_block_1b`, the
combined file from `run_2b.sweep_summary_lines` plus the same
`json.dump(..., default=p2b_io.json_default, allow_nan=False)` call the
runner makes. So every key, every verdict string and every refusal status in
this directory is one the phase can really emit, and a figure that breaks
here breaks against a run. A fixture that writes its own JSON tests the
fixture.

WHAT IS REAL

The file layout, every key name, the shapes (24 OV layers, 25 activation
layers — Pythia's embeddings-plus-blocks off-by-one is the real one and is
what `_matrices_for`'s index clamp exists for), the verdict vocabulary, the
refusal statuses, and the whole Block 1a spectrum including its nulls. `d` is
12 rather than 1024 because every quantity in the phase is scale-free in `d`
and a 1024³ Schur decomposition per layer per checkpoint is not a test.

WHAT IS INVENTED

Every number. The OV matrices are a fixed symmetric part plus a fixed
antisymmetric part in a step-dependent mixture, so the trajectories MOVE —
which is what makes the trajectory figures worth looking at and what makes
`flatness` return something other than "no data". The mixture is arbitrary
and its shape means nothing.

WHAT IS DELIBERATELY BROKEN

Three things, because the skip paths need exercising as much as the drawing
paths:

  - step 1000 is in `missing_checkpoints` and carries `status:
    no_ov_weights` — the silent-absence failure `run_sweep`'s
    `expected_steps` exists to surface.
  - one (checkpoint, prompt) pair records `{"status": "failed"}`, the shape
    `--continue-on-error` writes.
  - nulls are run at two checkpoints out of six and the precision surface at
    one, which is what `--with-nulls` / `--with-precision` at a subset look
    like, and is why those classes have to handle a partly-populated sweep
    rather than an all-or-nothing one.

THE DEGENERACY GATE is passed explicitly rather than read from
`core.config.DEGENERATE_RANK_THRESHOLD`, matching `tests/test_phase2b_*.py`:
importing `core.config` pulls in torch, and this fixture is used by a smoke
test that must run without it. `p2b_energy.resolve_rank_gate` raises rather
than falling back to a literal, which is what makes passing it explicit
correct rather than a shortcut.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from p2b_imaginary import p2b_io, run_2b
from p2b_imaginary.p2b_energy import DEFAULT_GATE_KIND

__all__ = ["build_fixture", "build_checkpoint", "FIXTURE_STEPS",
           "FIXTURE_PROMPTS", "GATE", "D_MODEL", "N_OV_LAYERS", "N_HEADS",
           "NULL_STEPS", "PRECISION_STEPS"]

#: `core.config.DEGENERATE_RANK_THRESHOLD`'s value, passed explicitly so this
#: module never imports `core.config` (and so torch). Not a fallback: the
#: phase refuses to guess this, and so does the fixture.
GATE: float = 2.0

D_MODEL: int = 12
N_OV_LAYERS: int = 24
N_TOKENS: int = 20
N_HEADS: int = 4

#: Six real Pythia checkpoint steps spanning the schedule, chosen to bracket
#: three of `p2b_report.KNOWN_TRANSITIONS`' dated spans (8->16, 256->512,
#: 3000->5000 partially) so the alignment figures have something to align to.
FIXTURE_STEPS: tuple = (0, 8, 16, 512, 3000, 143000)

#: The step the fixture pretends Phase 2 never wrote OV weights for.
MISSING_STEP: int = 1000

FIXTURE_PROMPTS: tuple = ("wiki_paragraph", "repeated_tokens")

#: Nulls are expensive (n_draws extra Schur decompositions per layer), so a
#: real sweep runs them at a subset. So does this.
NULL_STEPS: tuple = (0, 143000)

#: The precision surface is dearer still — about ten dense eigendecompositions
#: per layer — so a real sweep runs it at fewer checkpoints than the nulls.
#: One here, which is also the case the figures have to handle: a statistic
#: that exists at a single point on the training axis.
PRECISION_STEPS: tuple = (143000,)

#: Overall scale of the invented OV matrices — see `_ov_matrices`.
OV_SCALE: float = 0.03

#: The one checkpoint whose OV matrices are scaled up until `e^{−S}`
#: overflows the cumulative product. Truncation is the mechanism that makes
#: `elim_signed = 1.0` free (`e^{−A}` is orthogonal and cannot truncate,
#: `e^{−S}` can), so a fixture with no truncated frame anywhere would leave
#: F3, F4 and V5 drawing nothing at all.
OVERFLOW_STEP: int = 3000
OVERFLOW_SCALE: float = 2000.0

#: And one checkpoint scaled just far enough that the two causal frames give
#: DIFFERENT counts while still scoring the same transitions — the only
#: configuration in which `elim_full` and `elim_signed` are both numbers and
#: the verdict is a measurement rather than a refusal. Both rates come out
#: negative, which is the unclipped reading F2 exists to draw and the one
#: `analysis_p2.py:153`'s `max(0, ...)` destroys in Phase 2.
AMPLIFIED_STEP: int = 143000
AMPLIFIED_SCALE: float = 12.0

#: Steps before this one produce a monotone energy trajectory and therefore
#: `n_original == 0` — the `no_violations` refusal. Study B's steps 8-64 are
#: clean on all 9 prompts for reasons that have nothing to do with this
#: arithmetic; the point is that the verdict exists at the early end and must
#: not read as "rescaling did nothing".
FIRST_VIOLATING_STEP: int = 512

BASE: str = "pythia-410m"


# ---------------------------------------------------------------------------
# Invented weights, with a trajectory
# ---------------------------------------------------------------------------

def _mixture(step: int) -> float:
    """
    Fraction of antisymmetric content at this step, on [0.35, 0.9].

    Monotone in log(step+1) with a bump between 8 and 16, so the trajectory
    figures have both a trend and a localized event and the alignment table
    has one span that ranks high and several that do not. The shape is
    arbitrary — it is here so `flatness` and `interval_deltas` return
    something to draw, not because anything is being modelled.
    """
    t = float(np.log10(step + 1.0) / np.log10(143001.0))
    bump = 0.12 if 8 <= step <= 16 else 0.0
    return float(0.35 + 0.55 * t + bump)


def _ov_per_head(step: int, seed: int = 0) -> List[List[np.ndarray]]:
    """
    Per-head OV for one checkpoint: `[layer][head] -> (d, d)`, each of rank
    `d_head`.

    Built the way the real thing is — `W_O W_V` with `W_O: (d, d_head)` — so
    `head_circuits.factor_from_dense` recovers factors of the right rank and
    the per-head spectra are `d_head`-dimensional rather than `d`-dimensional.
    A fixture that wrote full-rank per-head matrices would exercise the code
    and none of its mathematics.
    """
    rng = np.random.default_rng(seed + 991)
    d_head = D_MODEL // N_HEADS
    out = []
    for L in range(N_OV_LAYERS):
        heads = []
        for h in range(N_HEADS):
            W_O = rng.normal(size=(D_MODEL, d_head))
            W_V = rng.normal(size=(d_head, D_MODEL))
            heads.append((W_O @ W_V) / np.sqrt(D_MODEL * d_head))
        out.append(heads)
    return out


def _ov_matrices(step: int, seed: int = 0) -> List[np.ndarray]:
    """24 OV matrices for one checkpoint, sharing a fixed S and A basis."""
    rng = np.random.default_rng(seed)
    a = _mixture(step)
    mats = []
    for L in range(N_OV_LAYERS):
        M = rng.normal(size=(D_MODEL, D_MODEL))
        S = 0.5 * (M + M.T)
        A = 0.5 * (M - M.T)
        # Depth tilt: deeper layers get more rotation, so the layer x step
        # heatmaps are not flat in either direction.
        depth = 0.15 * (L / max(N_OV_LAYERS - 1, 1))
        V = (1.0 - a - depth) * S + (a + depth) * A
        # Kept small so `e^{-V}` stays near the identity and the rescaled
        # frames score the SAME transitions as the original at most
        # checkpoints. At full scale every comparison refuses with
        # `different_transitions_scored` — which is a real Phase 2b failure
        # mode and is exercised deliberately at OVERFLOW_STEP below, but a
        # fixture where every run refuses would leave the verdict palette
        # untested.
        mats.append(OV_SCALE * V)
    return mats


def _ov_data(step: int, stem: str) -> dict:
    """`p2b_io.load_ov_data`'s return shape, without a file on disk."""
    scale = {OVERFLOW_STEP: OVERFLOW_SCALE,
             AMPLIFIED_STEP: AMPLIFIED_SCALE}.get(step, 1.0)
    # The per-head arrays are deliberately NOT a decomposition of `ov_total`:
    # a real `ov_total` is `sum_h ov_head_h`, and reproducing that here would
    # make `summed_vs_per_head`'s gap an artifact of this fixture's arithmetic
    # rather than of the rank-`d_head` structure the figure is about. The
    # fixture's job is shapes and vocabulary; the gap it produces is invented
    # like every other number in this file.
    return {
        "ov_total": [scale * M for M in _ov_matrices(step)],
        "ov_per_head": _ov_per_head(step),
        "is_per_layer": True,
        "layer_names": [f"layer_{i}" for i in range(N_OV_LAYERS)],
        "model_stem": stem,
        "checkpoint_step": step,
        "source_path": f"<fixture>/ov_weights_{stem}.npz",
    }


def _activations(step: int, prompt: str) -> np.ndarray:
    """
    (25, n_tokens, d) invented activations.

    Tokens contract toward two centroids with depth, which makes interaction
    energy rise and most transitions clean; a per-step perturbation at three
    layers reverses the contraction there, which is what produces violations.
    Later checkpoints get a larger perturbation, so the `no_violations`
    verdict appears at the early end — the shape Study B really has, for
    reasons that have nothing to do with this arithmetic.
    """
    # `hash()` on a str is randomized per process (PYTHONHASHSEED), so seeding
    # from it would make the fixture — and every figure drawn against it —
    # different on every run. crc32 is stable across processes and versions.
    rng = np.random.default_rng(
        (int(step) * 1_000_003 + zlib.crc32(prompt.encode())) % (2 ** 31))
    n_layers = N_OV_LAYERS + 1
    centroids = rng.normal(size=(2, D_MODEL))
    assign = rng.integers(0, 2, size=N_TOKENS)
    base = centroids[assign] + 0.9 * rng.normal(size=(N_TOKENS, D_MODEL))

    kick = (0.0 if step < FIRST_VIOLATING_STEP else
            0.15 + 0.55 * float(np.log10(step + 1.0) / np.log10(143001.0)))
    out = np.empty((n_layers, N_TOKENS, D_MODEL))
    for L in range(n_layers):
        pull = 0.85 * (L / (n_layers - 1))
        x = (1.0 - pull) * base + pull * centroids[assign]
        if kick and L in (6, 13, 19):
            x = x + kick * rng.normal(size=x.shape)
        out[L] = x + 0.02 * rng.normal(size=x.shape)
    return out


def _phase1_violation_layers(step: int, betas: Sequence[float]) -> dict:
    """
    A stand-in for Phase 1's own violation layers, for the cross-check.

    Deliberately NOT equal to what Phase 2b will count: the two gate on
    different effective ranks (raw vs normed) and are expected to disagree.
    A fixture where they agreed would hide the one thing F7 is for.
    """
    rng = np.random.default_rng(step + 7)
    return {float(b): sorted(rng.choice(np.arange(1, N_OV_LAYERS),
                                        size=3, replace=False).tolist())
            for b in betas}


# ---------------------------------------------------------------------------
# One checkpoint, through the phase's own code
# ---------------------------------------------------------------------------

def build_checkpoint(out_root: Path, step: int, *,
                     prompts: Sequence[str] = FIXTURE_PROMPTS,
                     betas: Sequence[float] = (1.0,),
                     with_nulls: bool = False,
                     with_precision: bool = False,
                     fail_prompt: Optional[str] = None) -> dict:
    """
    One checkpoint's subdirectory, in `run_2b.run_checkpoint`'s output shape.

    Returns the dict the combined file's `results` is keyed on, so the caller
    assembles the sweep exactly as the runner does.
    """
    stem = f"{BASE}-step{step}"
    ckpt_dir = Path(out_root) / stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ov = _ov_data(step, stem)

    out: dict = {
        "model_stem": stem,
        "checkpoint_step": step,
        "n_ov_layers": N_OV_LAYERS,
        "status": "ok",
        "failures": [],
    }

    # -- Block 1a: the phase's own runner entry point ------------------------
    out["block1a"] = run_2b.run_block_1a(
        ov, ckpt_dir, with_nulls=with_nulls, n_null_draws=4, seed=step,
        with_precision=with_precision,
    )

    # Per-head circuits, from the same weights. Cheap: per-head W_OV has rank
    # d_head, so every spectrum here is a d_head^2 problem.
    heads = run_2b.run_head_circuits(ov, ckpt_dir)
    if heads is not None:
        out["head_circuits"] = heads

    # -- Block 1b: one record per prompt -------------------------------------
    rescaler_cache: dict = {}          # built once per checkpoint, as in run_2b
    per_prompt: dict = {}
    for prompt in prompts:
        if prompt == fail_prompt:
            # The shape `--continue-on-error` writes. A prompt that failed
            # must be visible as a failure downstream, not as an absence.
            per_prompt[prompt] = {"status": "failed"}
            out["failures"].append({"prompt": prompt, "block": "1b",
                                    "traceback": "<fixture: simulated failure>"})
            continue
        bundle = {
            "activations": _activations(step, prompt),
            "phase1_violation_layers": _phase1_violation_layers(step, betas),
        }
        per_prompt[prompt] = run_2b.run_block_1b(
            bundle, ov, ckpt_dir / prompt,
            betas=betas, rescaler_cache=rescaler_cache,
            gate_kind=DEFAULT_GATE_KIND, gate_threshold=GATE,
            model_stem=stem, prompt_key=prompt,
        )
    out["block1b"] = per_prompt
    out["n_phase1_runs"] = len(prompts)

    p2b_io.write_run_manifest(
        ckpt_dir, stem, None, 0.0,
        config={"blocks": ["1a", "1b"], "betas": list(betas),
                "gate_kind": DEFAULT_GATE_KIND, "gate_threshold": GATE,
                "with_nulls": bool(with_nulls),
                "with_precision": bool(with_precision),
                "seed": int(step), "fixture": True},
    )
    # Invented, and used only by the cost figure: real d^3 timings at d = 12
    # would say nothing about a 1024-dimensional sweep, so a plausible curve
    # is drawn instead and X10's caption says the numbers are fictional.
    out["wall_time_seconds"] = float(8.0 + 0.4 * len(prompts) * (1 + step % 5))
    return out


# ---------------------------------------------------------------------------
# The whole directory
# ---------------------------------------------------------------------------

def build_fixture(out_root, steps: Sequence[int] = FIXTURE_STEPS,
                  prompts: Sequence[str] = FIXTURE_PROMPTS,
                  betas: Sequence[float] = (1.0,)) -> Path:
    """
    Build a complete synthetic Phase 2b output directory and return its path.

    Writes exactly what `run_2b.run_sweep` writes: one subdirectory per
    checkpoint holding `block1a_rotational_spectrum.json` and a per-prompt
    `block1b_rescaled_comparison.json`, plus `phase2b_results.json` and
    `phase2b_summary.txt` at the root.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # The checkpoint Phase 2 never wrote weights for, in the runner's own
    # shape: present in `results` with a status, and named in
    # `missing_checkpoints`. Absent from the glob is what the rewrite was
    # meant to stop being silent.
    missing_stem = f"{BASE}-step{MISSING_STEP}"
    results[missing_stem] = {"model_stem": missing_stem,
                             "checkpoint_step": MISSING_STEP,
                             "status": "no_ov_weights"}

    for step in steps:
        results[f"{BASE}-step{step}"] = build_checkpoint(
            out_root, step, prompts=prompts, betas=betas,
            with_nulls=step in NULL_STEPS,
            with_precision=step in PRECISION_STEPS,
            # One failed prompt, at one checkpoint, so the refusal path is
            # exercised without swamping the verdict figures.
            fail_prompt=(prompts[-1] if step == 16 else None),
        )

    combined = {
        "phase": "2b",
        "base": BASE,
        "blocks": ["1a", "1b"],
        "betas": [float(b) for b in betas],
        "counting_rule": {"gate_kind": DEFAULT_GATE_KIND,
                          "gate_threshold": GATE},
        "n_checkpoints": len(steps),
        "n_failed": sum(len(r.get("failures", [])) for r in results.values()),
        "missing_checkpoints": [MISSING_STEP],
        "steps": list(steps),
        "wall_time_seconds": float(sum(r.get("wall_time_seconds") or 0.0
                                       for r in results.values())),
        "results": results,
        "fixture": True,
    }

    # Through the runner's own writer, not a copy of it: this was a copy
    # once, and it broke the moment `run_sweep` gained a NaN sanitizer.
    run_2b.write_combined(out_root, combined)
    return out_root
