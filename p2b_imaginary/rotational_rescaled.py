"""
rotational_rescaled.py — Block 1b. Causal isolation of the signed and
rotational parts of V_eff, and the identity that used to be mistaken for a
result.

THE CORRECTION THIS REWRITE EXISTS FOR
--------------------------------------
The previous version built three rescaled frames from V = S + A:

    remove_full     z = x @ (e^{-V})^T
    remove_signed   z = x @ (e^{-S})^T
    remove_rotation z = x @ (e^{-A})^T

and compared violation counts across them. `elim_rotation = 0.0` in 35/35
runs was reported as the phase's headline: rotation is dynamically neutral.

It is not a finding. `A = (V - V^T)/2` is real antisymmetric, so `e^{-A}` is
ORTHOGONAL, and so is any cumulative product of such matrices. The frame
then re-projects each token to the unit sphere — which an orthogonal map
already preserves. Every quantity Block 1b measures is a function of the
Gram matrix `X X^T`, and

    (X R^T)(X R^T)^T = X R^T R X^T = X X^T   for R R^T = I

exactly. Energies, effective rank, `ip_mean`, `ip_mass_near_1`: all
identical to the unrescaled trajectory. Measured residual over 24
accumulated layers at d=1024 is ~1e-15, against a violation threshold of
1e-3 relative. So `n_remove_rotation == n_original` and
`elim_rotation == 0.0` are forced by construction, in every run, on every
model, at every beta, before any data is read.

Three consequences, all of which belong next to any use of this module:

1. `rotation_neutral` was never a falsifiable claim in this frame. It must
   be withdrawn from status-2b, not re-run.
2. `status-2.md`'s third candidate explanation for Pythia's inert rescaled
   frame ("Phase 2b established that the signed component carries 100% of
   causal weight") loses its evidence. The other two — numerical truncation
   and clipped overcorrection — stand and are testable here.
3. `core/precision_policy.py`'s docstring says Phase 2b's causal conclusion
   is "unaffected by how the complex fraction is counted." True about the
   tolerance, and irrelevant: the conclusion fails for an unrelated reason.

WHAT SURVIVES
-------------
`remove_full` vs `remove_signed`. `e^{-(S+A)} != e^{-S} e^{-A}` unless S and
A commute, so these two frames genuinely differ, and their difference is
exactly the quantity `status-2.md`'s "next experiments" item 2 asks for:

    if signed-only rescaling recovers ~1.0 while full-V rescaling gives the
    2.1% Study B measured, the failure is rotational interference in the
    matrix exponential and V is still causal. If signed-only also fails, the
    mechanism does not transfer.

`remove_rotation` is retained, but demoted: it is now an INVARIANCE CONTROL,
returned with `is_invariance_control=True` and a measured residual, and it
is refused as an input to `interpret_comparison`. Its job is to fail loudly
if the orthogonality it depends on ever stops holding numerically — not to
answer a question.

WHAT WOULD BE A REAL ROTATION TEST
----------------------------------
Something not invariant under a global orthogonal map of the residual
stream. Two are reachable from what already exists:

  - Weight-space ablation through `core/intervention.py`: set W_OV := S per
    layer, re-run the forward pass, recount. The composition with attention
    and the FFN is not orthogonally invariant even though the metric is.
  - Readout-space measurement through `core/functional_distance.py`: the
    decoded next-token distribution depends on `embed_out`, which is fixed,
    so rotating the residual stream does change it. This is the clean
    discriminator between "rotation is inert" and "rotation happens to be
    orthogonal to the metric we chose."

Neither is in this module. Both are the follow-up.

OTHER CHANGES FROM THE PREVIOUS VERSION
---------------------------------------
- Violation counting moved to `p2b_energy` — relative tolerance and the
  project's degeneracy gate, not a local absolute `-1e-6` / `>= 3.0` pair.
  Effective rank now uses squared singular values, matching
  `core.metrics.effective_rank`; the old local version used unsquared ones.
- `n_valid_layers` is returned PER FRAME and is a first-class output rather
  than being computed and then dropped by the serializer. This is Phase 2's
  verification item V1. It matters more here than there: `e^{-A}` is
  orthogonal and cannot overflow, while `e^{-S}` with positive eigenvalues
  can, so an early-truncating signed frame produces `elim_signed = 1.0` for
  free.
- `expm` is computed once per matrix set and reused across prompts
  (`build_rescalers`). It was previously recomputed inside every
  (checkpoint, prompt) pair, for a prompt-independent quantity.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm

from core.metrics import ENERGY_VIOLATION_REL_TOL

from p2b_imaginary.p2b_energy import (
    DEFAULT_GATE_KIND,
    count_violations_all_betas,
    elimination_rate,
    sphere_project,
    trajectory_scalars,
)

#: Magnitude at which the cumulative rescaling is declared diverged. Kept
#: identical to the value `p2_eigenspectra/trajectory_perlayer.py` uses, so
#: a truncation here and a truncation there mean the same thing.
RESCALE_OVERFLOW_LIMIT: float = 1e15

#: Row norm below which a rescaled token is no longer a direction.
#:
#: The mirror failure to overflow, and the one that is silent. `e^{-S}` for a
#: POSITIVE-definite S contracts; after enough layers the rescaled rows
#: underflow toward zero. `core.metrics.l2_normalize` leaves rows with norm
#: < 1e-12 unnormalized rather than dividing by zero, so the Gram matrix goes
#: to ~0, every energy goes to the constant 1/(2*beta), and the frame reports
#: ZERO VIOLATIONS. An `elim = 1.0` produced this way is indistinguishable
#: from a real one unless the collapse is detected here.
RESCALE_UNDERFLOW_LIMIT: float = 1e-12

#: How far the cumulative rotation-only rescaler may drift from orthogonality
#: before the invariance control is reported as failed rather than passed.
#: Set well above float64 accumulation over ~48 layers (~1e-14 observed) and
#: well below anything that could move a relative energy drop past 1e-3.
ORTHOGONALITY_TOL: float = 1e-8

FRAME_KEYS = ("original", "remove_full", "remove_signed", "remove_rotation")

#: Old key -> new key, for reading pre-rewrite `phase2i_results.json`.
#: The old names described what was KEPT; the new ones describe what is
#: REMOVED, which is what the rescaling actually does.
LEGACY_FRAME_KEYS = {
    "n_original": "original",
    "n_full_rescaled": "remove_full",
    "n_signed_only": "remove_signed",
    "n_rotation_only": "remove_rotation",
}


# ---------------------------------------------------------------------------
# S / A decomposition
# ---------------------------------------------------------------------------

def decompose_symmetric_antisymmetric(OV: np.ndarray) -> dict:
    """
    Split V_eff into symmetric (signed) and antisymmetric (rotational) parts.

      S = (V + V^T)/2   real eigenvalues; attraction/repulsion
      A = (V - V^T)/2   purely imaginary eigenvalues; pure rotation

    `rotation_ratio` = ||A||_F / ||S||_F. Note this is a FROBENIUS ratio and
    is not the same quantity as `rotational_schur.rotation_energy_fractions`'
    spectral-energy fraction; the two answer different questions and Phase
    2b previously reported both as "how rotational V is". Both are returned
    here with distinct names so they cannot be conflated.
    """
    OV = np.asarray(OV, dtype=np.float64)
    S = (OV + OV.T) / 2.0
    A = (OV - OV.T) / 2.0

    s_norm = float(np.linalg.norm(S, "fro"))
    a_norm = float(np.linalg.norm(A, "fro"))
    v_norm = float(np.linalg.norm(OV, "fro"))

    return {
        "S": S,
        "A": A,
        "S_frob": s_norm,
        "A_frob": a_norm,
        "V_frob": v_norm,
        "rotation_ratio_frobenius": float(a_norm / max(s_norm, 1e-12)),
        "rotational_frobenius_fraction": float(a_norm ** 2 / max(v_norm ** 2, 1e-12)),
    }


# ---------------------------------------------------------------------------
# Rescalers
# ---------------------------------------------------------------------------

def build_rescalers(matrices: Sequence[np.ndarray]) -> list:
    """
    `[expm(-M) for M in matrices]`, in float64.

    Separated from the trajectory application so it can be cached per
    checkpoint. The matrices are OV weights: prompt-independent. The
    previous version recomputed them inside every (checkpoint, prompt) pair,
    which on the Study B sweep is 27 x 9 x 3 x 24 exponentials of a
    1024x1024 matrix where 27 x 3 x 24 suffice.
    """
    return [expm(-np.asarray(M, dtype=np.float64)) for M in matrices]


def orthogonality_residual(rescalers: Sequence[np.ndarray]) -> dict:
    """
    How far the cumulative product of `rescalers` is from orthogonal.

    Used on the rotation-only rescalers, where `e^{-A}` is orthogonal by
    construction. A residual above ORTHOGONALITY_TOL means the invariance
    the control depends on has broken numerically and the control's result
    is no longer an identity check.
    """
    if not len(rescalers):
        return {"max_residual": 0.0, "n_matrices": 0, "orthogonal": True}
    d = np.asarray(rescalers[0]).shape[0]
    R = np.eye(d, dtype=np.float64)
    worst = 0.0
    for M in rescalers:
        R = R @ np.asarray(M, dtype=np.float64)
        worst = max(worst, float(np.abs(R @ R.T - np.eye(d)).max()))
    return {
        "max_residual": worst,
        "n_matrices": int(len(rescalers)),
        "orthogonal": bool(worst <= ORTHOGONALITY_TOL),
    }


# ---------------------------------------------------------------------------
# Rescaled trajectory
# ---------------------------------------------------------------------------

def rescaled_trajectory(
    activations: np.ndarray,
    rescalers: Optional[Sequence[np.ndarray]],
) -> dict:
    """
    Apply the cumulative rescaling z_i(L) = normalize( x_i(L) @ R_cum(L)^T ),
    with R_cum(L) = prod_{l<L} rescalers[l].

    `rescalers=None` returns the unrescaled trajectory (the `original`
    frame) so every frame goes through one code path.

    Returns
    -------
    dict with:
      normed           : (n_layers, n_tokens, d), NaN at and after truncation
      n_valid_layers   : int — layers actually produced. THE number Phase 2's
                         verification item V1 asks for. A frame that stops at
                         layer 3 has three transitions to score and cannot be
                         compared with one that scored twenty-three.
      truncated        : bool
      truncation_reason: "nonfinite_activations" | "rescaler_overflow"
                         | "rescaler_underflow" | None
      r_cum_max_abs    : (n_layers,) — the growth curve, recorded even when
                         truncation does not fire, so "the rescaling was fine"
                         is a measurement rather than the absence of a flag.

    Truncation semantics differ from the previous version. That one set
    `max_valid = L + 1` BEFORE the overflow check, so the layer computed
    from an already-diverged R_cum was counted as valid. Here a layer is
    valid only if the R_cum used to produce it was itself within limits.
    """
    acts = np.asarray(activations)
    n_layers, n_tokens, d = acts.shape

    out = np.full((n_layers, n_tokens, d), np.nan, dtype=np.float64)
    r_cum_max = np.full(n_layers, np.nan)

    if rescalers is None:
        out = sphere_project(acts.astype(np.float64))
        r_cum_max[:] = 1.0
        return {
            "normed": out,
            "n_valid_layers": int(n_layers),
            "truncated": False,
            "truncation_reason": None,
            "r_cum_max_abs": r_cum_max,
        }

    R_list = [np.asarray(M, dtype=np.float64) for M in rescalers]
    R_cum = np.eye(d, dtype=np.float64)
    n_valid = 0
    reason = None

    for L in range(n_layers):
        cur_max = float(np.abs(R_cum).max())
        r_cum_max[L] = cur_max
        if not np.isfinite(cur_max) or cur_max > RESCALE_OVERFLOW_LIMIT:
            reason = "rescaler_overflow"
            break

        raw = acts[L].astype(np.float64) @ R_cum.T
        if not np.all(np.isfinite(raw)):
            reason = "nonfinite_activations"
            break
        if float(np.linalg.norm(raw, axis=-1).min()) < RESCALE_UNDERFLOW_LIMIT:
            reason = "rescaler_underflow"
            break

        out[L] = sphere_project(raw)
        n_valid = L + 1

        idx = min(L, len(R_list) - 1)
        R_cum = R_cum @ R_list[idx]

    return {
        "normed": out,
        "n_valid_layers": int(n_valid),
        "truncated": bool(reason is not None),
        "truncation_reason": reason,
        "r_cum_max_abs": r_cum_max,
    }


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def _matrices_for(ov_data: dict, n_layers: int) -> dict:
    """
    Per-layer V, S and A lists, one entry per analysed layer.

    Shared-weight models (ALBERT) repeat a single matrix; per-layer models
    (GPT-2, BERT, Pythia) use their own. `n_layers` is the ACTIVATION depth,
    which for Pythia is 25 (embeddings + 24 blocks) against 24 OV matrices —
    `rescaled_trajectory` clamps the index, so the last OV is reused for the
    trailing hidden state rather than raising.
    """
    if ov_data["is_per_layer"]:
        Vs = [np.asarray(M, dtype=np.float64) for M in ov_data["ov_total"]]
    else:
        Vs = [np.asarray(ov_data["ov_total"], dtype=np.float64)] * n_layers

    sa = [decompose_symmetric_antisymmetric(V) for V in Vs]
    return {
        "V": Vs,
        "S": [x["S"] for x in sa],
        "A": [x["A"] for x in sa],
        "sa": sa,
    }


def compare_rescaled_frames(
    activations: np.ndarray,
    ov_data: dict,
    beta_values: Sequence[float],
    *,
    rescaler_cache: Optional[dict] = None,
    gate_kind: str = DEFAULT_GATE_KIND,
    gate_threshold: Optional[float] = None,
    rel_tol: float = ENERGY_VIOLATION_REL_TOL,
    include_invariance_control: bool = True,
) -> dict:
    """
    Build the frames, score each one, and compare only the pairs that are
    comparable.

    `rescaler_cache` is an optional dict, keyed "V"/"S"/"A", holding the
    output of `build_rescalers` for this checkpoint. Pass the same dict
    across every prompt of a checkpoint; it is populated on first use.

    Returns
    -------
    dict with:
      frames        : {frame_key: {scalars, counts, n_valid_layers,
                                   truncated, truncation_reason}}
      comparison    : {beta: {pair_key: elimination_rate(...) result}}
      sa_decomp     : Frobenius summaries per layer
      invariance    : the rotation-only control's audit, or None
      counting_rule : the rule every count above was scored with
    """
    acts = np.asarray(activations)
    n_layers = acts.shape[0]
    mats = _matrices_for(ov_data, n_layers)

    cache = rescaler_cache if rescaler_cache is not None else {}
    for key in ("V", "S", "A"):
        if key not in cache:
            cache[key] = build_rescalers(mats[key])

    frame_rescalers = {
        "original": None,
        "remove_full": cache["V"],
        "remove_signed": cache["S"],
    }
    if include_invariance_control:
        frame_rescalers["remove_rotation"] = cache["A"]

    frames: dict = {}
    for key, R in frame_rescalers.items():
        traj = rescaled_trajectory(acts, R)
        scal = trajectory_scalars(
            traj["normed"], beta_values, n_valid_layers=traj["n_valid_layers"],
        )
        counts = count_violations_all_betas(
            scal, rel_tol=rel_tol,
            gate_kind=gate_kind, gate_threshold=gate_threshold,
        )
        frames[key] = {
            "scalars": scal,
            "counts": counts,
            "n_valid_layers": traj["n_valid_layers"],
            "truncated": traj["truncated"],
            "truncation_reason": traj["truncation_reason"],
            "r_cum_max_abs": traj["r_cum_max_abs"],
            # The one frame whose result is an identity, not a measurement.
            "is_invariance_control": key == "remove_rotation",
        }

    # --- comparisons -------------------------------------------------------
    # remove_rotation is deliberately absent from the causal pairs. Including
    # it would put an algebraic identity next to two measurements in the same
    # table, which is how it came to be read as a result.
    causal_pairs = {
        "elim_full": ("original", "remove_full"),
        "elim_signed": ("original", "remove_signed"),
    }

    comparison: dict = {}
    for beta in frames["original"]["counts"]:
        row = {}
        for name, (a_key, b_key) in causal_pairs.items():
            row[name] = elimination_rate(
                frames[a_key]["counts"][beta], frames[b_key]["counts"][beta],
            )
        comparison[float(beta)] = row

    # --- invariance control ------------------------------------------------
    invariance = None
    if include_invariance_control:
        invariance = _audit_invariance(frames, cache["A"], beta_values)

    return {
        "frames": frames,
        "comparison": comparison,
        "sa_decomp": {
            "per_layer_rotation_ratio_frobenius": [
                x["rotation_ratio_frobenius"] for x in mats["sa"]
            ],
            "mean_rotation_ratio_frobenius": float(np.mean(
                [x["rotation_ratio_frobenius"] for x in mats["sa"]]
            )),
            "per_layer_S_frob": [x["S_frob"] for x in mats["sa"]],
            "per_layer_A_frob": [x["A_frob"] for x in mats["sa"]],
            "layer_names": ov_data.get("layer_names", []),
        },
        "invariance": invariance,
        "counting_rule": frames["original"]["counts"][
            float(list(frames["original"]["counts"])[0])
        ]["rule"],
    }


def _audit_invariance(frames: dict, a_rescalers: Sequence[np.ndarray],
                      beta_values: Sequence[float]) -> dict:
    """
    Confirm the rotation-only frame reproduces the original frame exactly,
    and report by how much it fails to.

    `status` is:
      "identity_holds"  — as predicted by orthogonality. This is the expected
                          outcome and is NOT evidence about rotation.
      "identity_broken" — the frames differ. That is a numerical-stability
                          finding about `expm` or the accumulation, not a
                          dynamical one, and it invalidates the control.
    """
    orig = frames["original"]
    rot = frames["remove_rotation"]

    ortho = orthogonality_residual(a_rescalers)

    n_common = min(orig["n_valid_layers"], rot["n_valid_layers"])
    worst_energy = 0.0
    for beta in beta_values:
        b = float(beta)
        Eo = np.asarray(orig["scalars"]["energies"][b][:n_common])
        Er = np.asarray(rot["scalars"]["energies"][b][:n_common])
        m = np.isfinite(Eo) & np.isfinite(Er)
        if m.any():
            ref = np.maximum(np.abs(Eo[m]), 1e-12)
            worst_energy = max(worst_energy,
                               float(np.abs(Er[m] - Eo[m]).max() / ref.max()))

    counts_match = all(
        orig["counts"][b]["n_violations"] == rot["counts"][b]["n_violations"]
        for b in orig["counts"]
    )
    holds = bool(ortho["orthogonal"] and counts_match)

    return {
        "status": "identity_holds" if holds else "identity_broken",
        "orthogonality": ortho,
        "max_relative_energy_difference": worst_energy,
        "violation_counts_match": bool(counts_match),
        "note": (
            "e^{-A} is orthogonal for antisymmetric A, so this frame cannot "
            "change any Gram-derived quantity. A match here is an arithmetic "
            "check, not evidence that rotation is dynamically neutral."
        ),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

#: What Block 1b can now conclude. Deliberately does NOT include
#: `rotation_neutral`, `rotation_contributes`, or `rotation_dominant` — those
#: were verdicts about the rotation-only frame, which cannot support any of
#: them.
VERDICTS = (
    "signed_carries_full_v",       # remove_signed matches remove_full
    "signed_exceeds_full_v",       # removing S alone beats removing all of V
    "full_v_exceeds_signed",       # S and A interact; the product matters
    "both_frames_inert",           # neither rescaling moves the count
    "no_violations",               # nothing to eliminate at this checkpoint
    "not_comparable",              # truncation or gate divergence
)

#: Elimination-rate difference below which two frames are called equivalent.
EQUIVALENCE_BAND: float = 0.1


def interpret_comparison(comparison: dict, band: float = EQUIVALENCE_BAND) -> dict:
    """
    Classify the signed-vs-full contrast, per beta and overall.

    No majority vote across beta. Phase 1 found violation counts are
    beta-independent after step 512 and Phase 2 Study B ran beta=1.0 only, so
    a vote over four betas is a vote over four near-copies at trained
    checkpoints and over a real gradient only at steps 128-256. `overall` is
    therefore taken at beta=1.0 (the Study B beta) when present, with the
    full per-beta table always returned alongside; `beta_dispersion` records
    whether the betas actually disagreed.
    """
    per_beta: dict = {}

    for beta, row in comparison.items():
        full = row["elim_full"]
        signed = row["elim_signed"]

        if full["status"] == "no_violations_to_eliminate" or \
           signed["status"] == "no_violations_to_eliminate":
            verdict = "no_violations"
        elif full["status"] != "ok" or signed["status"] != "ok":
            verdict = "not_comparable"
        else:
            ef, es = full["rate"], signed["rate"]
            if abs(ef) < band and abs(es) < band:
                verdict = "both_frames_inert"
            elif abs(es - ef) <= band:
                verdict = "signed_carries_full_v"
            elif es > ef:
                verdict = "signed_exceeds_full_v"
            else:
                verdict = "full_v_exceeds_signed"

        per_beta[float(beta)] = {
            "verdict": verdict,
            "elim_full": full["rate"],
            "elim_signed": signed["rate"],
            "elim_full_status": full["status"],
            "elim_signed_status": signed["status"],
            "n_original": full["n_original"],
        }

    betas = sorted(per_beta)
    ref_beta = 1.0 if 1.0 in per_beta else (betas[0] if betas else None)
    overall = per_beta[ref_beta]["verdict"] if ref_beta is not None else "no_violations"

    distinct = {v["verdict"] for v in per_beta.values()}

    return {
        "per_beta": per_beta,
        "overall": overall,
        "reference_beta": ref_beta,
        "beta_dispersion": {
            "n_distinct_verdicts": len(distinct),
            "verdicts": sorted(distinct),
            "beta_independent": len(distinct) <= 1,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analyze_rotational_rescaling(
    activations: np.ndarray,
    ov_data: dict,
    beta_values: Optional[Sequence[float]] = None,
    *,
    rescaler_cache: Optional[dict] = None,
    gate_kind: str = DEFAULT_GATE_KIND,
    gate_threshold: Optional[float] = None,
    rel_tol: float = ENERGY_VIOLATION_REL_TOL,
    include_invariance_control: bool = True,
) -> dict:
    """Block 1b for one (checkpoint, prompt): frames, comparison, verdict."""
    if beta_values is None:
        beta_values = [1.0]

    frames = compare_rescaled_frames(
        activations, ov_data, beta_values,
        rescaler_cache=rescaler_cache,
        gate_kind=gate_kind, gate_threshold=gate_threshold, rel_tol=rel_tol,
        include_invariance_control=include_invariance_control,
    )
    interp = interpret_comparison(frames["comparison"])
    return {"frames": frames, "interpretation": interp}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _series(values) -> list:
    """
    A per-layer array as a JSON list, non-finite -> None.

    NaN is MEANINGFUL in every series this writes: a rescaled frame's
    energies are NaN at and after truncation, and `rel_drops` is NaN at
    every transition the gate rejected. So the length is preserved and the
    NaN becomes an explicit JSON null rather than the row being dropped —
    dropping it would silently shift every layer index by one, and only at
    the end of the depth axis where nobody looks.

    `p2b_io.json_default` would map these on the way out anyway; doing it
    here means the returned dict is plain-JSON before it reaches a writer,
    so a caller that dumps it with a different `default=` gets the same
    file.
    """
    return [None if v is None or not np.isfinite(v) else float(v)
            for v in np.asarray(values, dtype=np.float64).tolist()]


def comparison_to_json(result: dict) -> dict:
    """
    JSON-serializable summary.

    Unlike the previous version this KEEPS `n_valid_layers`, `truncated` and
    `truncation_reason` per frame. Dropping them is what made Phase 2's
    verification item V1 unanswerable from the artifact.

    It also keeps three per-layer series that were computed and discarded,
    all small and all load-bearing for reading a count:

      `per_layer.energies` / `.effective_rank` / `.ip_mean` /
      `.ip_mass_near_1`
          `trajectory_scalars` computes these for every frame and the first
          version kept only the derived counts — so the ENERGY CURVE, the
          object a violation is a feature of, could not be drawn, and
          neither could the gate quantity that decides which transitions are
          scored at all. A count says four transitions violated; only the
          curve says whether the frame changed the trajectory's shape or
          moved four numbers across a threshold.

      `r_cum_max_abs`
          `rescaled_trajectory` records the growth curve "even when
          truncation does not fire, so 'the rescaling was fine' is a
          measurement rather than the absence of a flag" — and then only its
          maximum survived, which is the flag again. The curve says WHERE
          the cumulative product started to diverge and how fast.

      `counts.rel_drops`
          The relative energy drop at every transition, NaN where unscored.
          `sum_severity` and `max_severity` are aggregates of it. Four
          violations all marginally over `rel_tol` and one catastrophic
          violation are the same count and not the same result.

    Cost at Study B's shape: 4 frames x 1 beta x 25 layers of float, about
    2 kB per (checkpoint, prompt) record. There was never a size argument
    for dropping them.
    """
    frames = result["frames"]

    out = {
        "sa_decomp": frames["sa_decomp"],
        "counting_rule": frames["counting_rule"],
        "invariance": frames["invariance"],
        "interpretation": result["interpretation"],
        "frames": {},
        "comparison": {},
    }

    for key, fr in frames["frames"].items():
        scal = fr["scalars"]
        out["frames"][key] = {
            "n_valid_layers": int(fr["n_valid_layers"]),
            "truncated": bool(fr["truncated"]),
            "truncation_reason": fr["truncation_reason"],
            "is_invariance_control": bool(fr["is_invariance_control"]),
            "r_cum_max_abs_final": (
                None if not np.isfinite(np.nanmax(fr["r_cum_max_abs"]))
                else float(np.nanmax(fr["r_cum_max_abs"]))
            ),
            # The growth CURVE, not just its maximum. NaN past truncation,
            # which is where the divergence is.
            "r_cum_max_abs": _series(fr["r_cum_max_abs"]),
            "per_layer": {
                # Keyed by str(beta) to match `counts` below — the whole file
                # uses one convention for a beta key, so a reader resolves it
                # once.
                "energies": {str(beta): _series(E)
                             for beta, E in scal["energies"].items()},
                "effective_rank": _series(scal["effective_rank"]),
                "ip_mean": _series(scal["ip_mean"]),
                "ip_mass_near_1": _series(scal["ip_mass_near_1"]),
                "n_layers": int(scal["n_layers"]),
                "gate_quantity": "effective_rank",
            },
            "counts": {
                str(beta): {
                    "n_violations": c["n_violations"],
                    "n_transitions_scored": c["n_transitions_scored"],
                    "n_transitions_gated": c["n_transitions_gated"],
                    "n_transitions_nan": c["n_transitions_nan"],
                    "violation_layers": c["violation_layers"],
                    "sum_severity": c["sum_severity"],
                    "max_severity": c["max_severity"],
                    # Per-transition severity, NaN (JSON null) where the
                    # transition was not scored. Length n_layers - 1, indexed
                    # by the transition L-1 -> L at position L-1.
                    "rel_drops": list(c["rel_drops"]),
                }
                for beta, c in fr["counts"].items()
            },
        }

    for beta, row in frames["comparison"].items():
        out["comparison"][str(beta)] = {
            name: {
                "rate": res["rate"],
                "status": res["status"],
                "n_original": res["n_original"],
                "n_rescaled": res["n_rescaled"],
                "n_scored_a": res["n_scored_a"],
                "n_scored_b": res["n_scored_b"],
            }
            for name, res in row.items()
        }

    return out


__all__ = [
    "RESCALE_OVERFLOW_LIMIT",
    "ORTHOGONALITY_TOL",
    "FRAME_KEYS",
    "LEGACY_FRAME_KEYS",
    "VERDICTS",
    "EQUIVALENCE_BAND",
    "decompose_symmetric_antisymmetric",
    "build_rescalers",
    "orthogonality_residual",
    "rescaled_trajectory",
    "compare_rescaled_frames",
    "interpret_comparison",
    "analyze_rotational_rescaling",
    "comparison_to_json",
]
