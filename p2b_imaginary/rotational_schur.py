"""
rotational_schur.py — Block 1a. The rotational structure of V_eff.

Weights only: no activations, no forward passes, no checkpoints beyond what
Phase 2 already caches. That makes this the cheapest thing in the phase and
the one item that can be run across all 27 Pythia checkpoints immediately —
which is also the question it should be answering. See "What this is for".

WHAT THIS IS FOR, AFTER THE BLOCK 1B WITHDRAWAL
-----------------------------------------------
Block 1b's `rotation_neutral` result is withdrawn (it was an orthogonal-
invariance identity; see `rotational_rescaled`'s docstring). Block 1a's
"84-97.5% of OV's spectral energy is rotational" survives as a DESCRIPTION,
and it was never a causal claim. But two things now have to be attached to
it, because without them it does not distinguish a trained network from a
random matrix:

  - **A null.** A real matrix with iid Gaussian entries has essentially all
    of its eigenvalues in complex conjugate pairs. If trained OV is 98%
    complex and a norm-matched Gaussian is 100%, the headline is a fact
    about square matrices. `null_comparison` runs that comparison through
    `core.nulls.sigma_from_null`, so a Block 1a claim lands in the same
    shape as every other claim in the project.
  - **A trajectory.** The number is only interesting if it MOVES. Phase 1
    dates four transitions and Phase 2 has an unexplained `frac_repulsive`
    decay (1.00 -> 0.50 -> 0.80 across ~90k steps with violation count flat).
    Whether the complex fraction or the Henrici non-normality moves on either
    schedule is the first candidate mechanism for either, and it costs one
    Schur decomposition per layer per checkpoint.

FOUR DEFECTS FIXED FROM THE PREVIOUS VERSION
--------------------------------------------
1. **Memory.** `build_rotation_plane_projectors` materialized `top_k=32`
   dense `(d, d)` projectors PLUS `combined_rotation` and `real_subspace`,
   per layer, and `analyze_rotational_spectrum` retained `schur_T` and
   `schur_Z` per layer on top. At d=1024 x 24 layers that is ~7 GB resident;
   at d=2048 (pythia-1.4b) ~27 GB. Nothing here stores a `(d, d)` projector:
   planes are kept as their `(d, 2)` orthonormal bases and
   `project_onto_planes` contracts through the basis, which is also O(n d k)
   instead of O(n d^2 k). Schur factors are dropped unless asked for.

2. **Two energy conventions in one file.** `rotation_energy_fractions`
   counted `rho^2` ONCE per 2x2 block while `henrici_nonnormality` counted
   `2*rho^2` for the same block — a 2x2 block holds two eigenvalues and
   occupies two dimensions, so the first understates rotational energy by
   about a factor of two relative to the second. The reported 84-97.5% uses
   the first. Here `complex_energy_fraction` is per-eigenvalue throughout
   (the Henrici convention), and `rotational_fraction_per_block` reproduces
   the old number under a name that says which one it is, so the historical
   figure stays checkable rather than silently restated.

3. **Rotation angle folded onto [0, pi/2].** `theta = arctan2(sqrt(-bc),
   abs(a))` uses `abs(a)`, so a repulsive rotation (Re lambda < 0, theta near
   pi) was reported as its reflection. The sign survived in a separate field,
   so nothing was lost — but `theta_mean` was not the mean rotation angle,
   which matters the moment theta is regressed against depth or against step.
   Now `arctan2(sqrt(-bc), a)` on [0, pi].

4. **Absolute subdiagonal threshold.** The 2x2 test was
   `abs(T[i+1, i]) > 1e-10` on a matrix whose scale varies by orders of
   magnitude across layers and checkpoints. Now relative to `||T||_F`.

A THIRD DEFINITION OF "ROTATIONAL FRACTION"
-------------------------------------------
`p2b_imaginary/layernorm_jacobian.rotational_fraction` counts a dimension as
complex when `|Im lambda| > tol * (|Re lambda| + eps)` with `tol = 0.01` — a
relative criterion on eigenvalues, not a Schur-block partition.
`core/precision_policy.py` imports THAT one as its default `frac_fn` and
flags it (item P2) as possibly an fp16-storage artifact: an exactly-real
eigenvalue pair perturbed at fp16 epsilon splits into a complex pair with a
tiny imaginary part, which a relative criterion counts as rotation.

So the phase has had three definitions and reported all of them as "how
rotational V is". `complex_energy_fraction_relative` below is that third
definition, given a home next to the other two, with the same signature
`precision_policy` expects. `precision_policy._default_frac_fn` should be
repointed here (plan item 16); until it is, both call sites compute the same
thing from different files.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.linalg import schur

#: Subdiagonal entries below this fraction of ||T||_F are treated as zero,
#: i.e. the block is 1x1. Relative, because OV norms vary by orders of
#: magnitude across layers and checkpoints.
SUBDIAGONAL_REL_TOL: float = 1e-12

#: Default relative tolerance for the eigenvalue-based complex criterion.
#: Matches the shipped value in layernorm_jacobian and
#: core.precision_policy.SHIPPED_TOL.
COMPLEX_REL_TOL: float = 0.01


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def extract_schur_blocks(OV: np.ndarray, keep_factors: bool = False) -> dict:
    """
    Real Schur decomposition of OV, parsed into 1x1 and 2x2 blocks.

    Each 2x2 block `[[a, b], [c, a']]` with `bc < 0` encodes a complex
    conjugate pair: rotation angle `theta = atan2(sqrt(-bc), a)` on [0, pi],
    modulus `rho = sqrt(a^2 - bc)`, and radial direction `sign(a)`.

    Parameters
    ----------
    keep_factors : retain `schur_T` and `schur_Z`, two (d, d) float64 arrays
                   per call. Off by default — retaining them per layer is
                   most of the previous version's memory footprint, and
                   nothing downstream reads them except tests.

    Returns
    -------
    dict with `blocks_1x1`, `blocks_2x2`, `d`, `n_real`, `n_complex`,
    `t_frob_sq`, `subdiagonal_tol`, and (optionally) `schur_T` / `schur_Z`.

    Each 2x2 block carries `plane` as a (d, 2) BASIS, not a (d, d) projector.
    Each 1x1 block carries `schur_vec` as (d,).

    Degenerate 2x2 blocks (`bc >= 0`: two real eigenvalues that LAPACK
    happened to leave in a 2x2) are split into two 1x1 entries with values
    `a +/- sqrt(bc)` rather than being recorded as a rotation of angle 0.
    The previous version kept them as 2x2 with `theta = 0` and
    `rho = |a|`, which is the wrong modulus and inflates `n_complex`.
    """
    OV = np.asarray(OV, dtype=np.float64)
    d = OV.shape[0]

    T, Z = schur(OV, output="real")
    t_frob = float(np.linalg.norm(T, "fro"))
    tol = max(SUBDIAGONAL_REL_TOL * t_frob, np.finfo(np.float64).tiny)

    blocks_1x1: list = []
    blocks_2x2: list = []

    def _add_real(idx, value, vec):
        blocks_1x1.append({
            "idx": int(idx),
            "value": float(value),
            "sign": 1 if value >= 0 else -1,
            "schur_vec": np.ascontiguousarray(vec),
        })

    i = 0
    while i < d:
        if i == d - 1 or abs(T[i + 1, i]) <= tol:
            _add_real(i, T[i, i], Z[:, i])
            i += 1
            continue

        a = float(T[i, i])
        a2 = float(T[i + 1, i + 1])
        b = float(T[i, i + 1])
        c = float(T[i + 1, i])
        bc = b * c

        if bc >= 0.0:
            # Not a conjugate pair. Two real eigenvalues of the 2x2.
            disc = np.sqrt(max(((a - a2) / 2.0) ** 2 + bc, 0.0))
            mid = (a + a2) / 2.0
            _add_real(i, mid + disc, Z[:, i])
            _add_real(i + 1, mid - disc, Z[:, i + 1])
            i += 2
            continue

        re = (a + a2) / 2.0
        im = float(np.sqrt(-bc))
        blocks_2x2.append({
            "idx": int(i),
            # [0, pi]. The previous version passed abs(re), folding a
            # repulsive rotation onto its reflection in [0, pi/2].
            "theta": float(np.arctan2(im, re)),
            "rho": float(np.sqrt(re * re + im * im)),
            "re": re,
            "im": im,
            "bc": float(bc),
            "sign": 1 if re >= 0 else -1,
            "plane": np.ascontiguousarray(Z[:, i:i + 2]),   # (d, 2) BASIS
        })
        i += 2

    out = {
        "blocks_1x1": blocks_1x1,
        "blocks_2x2": blocks_2x2,
        "d": int(d),
        "n_real": len(blocks_1x1),
        "n_complex": len(blocks_2x2),
        "t_frob_sq": float(t_frob ** 2),
        "subdiagonal_tol": float(tol),
    }
    if keep_factors:
        out["schur_T"] = T
        out["schur_Z"] = Z
    return out


# ---------------------------------------------------------------------------
# Energy fractions — ONE convention, plus the legacy one under its own name
# ---------------------------------------------------------------------------

def complex_energy_fraction(block_data: dict) -> dict:
    """
    Fraction of eigenvalue energy in complex pairs, counted PER EIGENVALUE.

    A 1x1 block contributes `lambda^2`. A 2x2 block holds two eigenvalues of
    modulus `rho`, so it contributes `2*rho^2` — the convention
    `henrici_nonnormality` already used, and the one that makes
    `complex_energy + real_energy == eigenvalue_energy` an identity rather
    than an approximation.

    `dim_complex_fraction` (`2*n_complex/d`) is reported alongside because it
    is a different question — how many DIMENSIONS rotate, versus how much
    ENERGY is in rotation — and the two were previously reported adjacent
    without a name distinguishing them.
    """
    real_e = sum(b["value"] ** 2 for b in block_data["blocks_1x1"])
    cplx_e = 2.0 * sum(b["rho"] ** 2 for b in block_data["blocks_2x2"])
    total = real_e + cplx_e
    d = block_data["d"]

    return {
        "convention": "per_eigenvalue",
        "complex_energy": float(cplx_e),
        "real_energy": float(real_e),
        "eigenvalue_energy": float(total),
        "complex_energy_fraction": float(cplx_e / max(total, 1e-300)),
        "real_energy_fraction": float(real_e / max(total, 1e-300)),
        "n_real": int(block_data["n_real"]),
        "n_complex": int(block_data["n_complex"]),
        "dim_real_fraction": float(block_data["n_real"] / d),
        "dim_complex_fraction": float(2 * block_data["n_complex"] / d),
    }


def rotational_fraction_per_block(block_data: dict) -> float:
    """
    The PREVIOUS version's number, reproduced so the historical 84-97.5%
    figure stays checkable.

    Counts `rho^2` once per 2x2 block against `lambda^2` once per 1x1 block,
    which mixes a per-pair total with a per-eigenvalue one and understates
    rotational energy by roughly a factor of two. Do not use it for new
    results; it exists so a re-run can say "the old convention gives X, the
    corrected one gives Y" instead of quietly reporting a different number
    under the same name.
    """
    real_e = sum(b["value"] ** 2 for b in block_data["blocks_1x1"])
    rot_e = sum(b["rho"] ** 2 for b in block_data["blocks_2x2"])
    return float(rot_e / max(real_e + rot_e, 1e-300))


def complex_energy_fraction_relative(M, tol: float = COMPLEX_REL_TOL) -> float:
    """
    The eigenvalue-based criterion: a dimension counts as complex when
    `|Im lambda| > tol * (|Re lambda| + eps)`.

    This is the third definition of "rotational fraction" in the phase
    (see the module docstring). It is here so there is one home for it, with
    the signature `core.precision_policy.complex_fraction_surface` expects
    for `frac_fn`; `p2b_imaginary.layernorm_jacobian.rotational_fraction` is
    the same function and should become a re-export of this one.

    `core/precision_policy.py` item P2 is about exactly this criterion: it is
    RELATIVE, and a relative criterion is what an fp16-epsilon split of a
    genuinely real eigenvalue pair defeats — the split is small in absolute
    terms and unbounded in ratio when `|Re lambda|` is also small. Which is
    why the answer to "how complex is OV" is a surface over (tol,
    perturbation), not a scalar. See `precision_surface` below.
    """
    eigs = np.linalg.eigvals(np.asarray(M, dtype=np.float64))
    is_cx = np.abs(np.imag(eigs)) > tol * (np.abs(np.real(eigs)) + 1e-12)
    total = float(np.sum(np.abs(eigs) ** 2))
    if total < 1e-300:
        return 0.0
    return float(np.sum(np.abs(eigs[is_cx]) ** 2) / total)


# ---------------------------------------------------------------------------
# Angle / modulus statistics
# ---------------------------------------------------------------------------

def rotation_angle_stats(block_data: dict) -> dict:
    """
    Distribution of rotation angles and moduli over the 2x2 blocks.

    `frac_repulsive_real_part` is the quantity the rescaled frame actually
    responds to: `e^{-V}` grows in the directions where `Re lambda < 0`. The
    previous version reported `frac_expanding` = fraction with `rho > 1`,
    which is a threshold on a scale convention (how OV was normalized), not
    on a dynamical property. Both are returned, with `rho > 1` named so it
    cannot be mistaken for the dynamical one.
    """
    blocks = block_data["blocks_2x2"]
    if not blocks:
        return {
            "theta_mean": float("nan"), "theta_std": float("nan"),
            "theta_median": float("nan"), "theta_min": float("nan"),
            "theta_max": float("nan"),
            "rho_mean": float("nan"), "rho_std": float("nan"),
            "frac_rho_above_one": float("nan"),
            "frac_repulsive_real_part": float("nan"),
            "frac_attractive_real_part": float("nan"),
            "n_complex": 0,
        }

    thetas = np.array([b["theta"] for b in blocks], dtype=np.float64)
    rhos = np.array([b["rho"] for b in blocks], dtype=np.float64)
    signs = np.array([b["sign"] for b in blocks], dtype=np.int64)
    n = len(blocks)

    return {
        "theta_mean": float(thetas.mean()),
        "theta_std": float(thetas.std()),
        "theta_median": float(np.median(thetas)),
        "theta_min": float(thetas.min()),
        "theta_max": float(thetas.max()),
        "rho_mean": float(rhos.mean()),
        "rho_std": float(rhos.std()),
        "frac_rho_above_one": float((rhos > 1.0).sum() / n),
        "frac_repulsive_real_part": float((signs < 0).sum() / n),
        "frac_attractive_real_part": float((signs > 0).sum() / n),
        "n_complex": int(n),
    }


# ---------------------------------------------------------------------------
# Non-normality
# ---------------------------------------------------------------------------

def henrici_nonnormality(block_data: dict) -> dict:
    """
    Henrici departure from normality: `||T||_F^2 - sum |lambda_i|^2`.

    Zero for a normal matrix. Otherwise it is the squared Frobenius norm of
    T's strict upper triangle, i.e. how much the Schur blocks interact — the
    scalar that says whether the S/A split is informative or decorative.

    Phase 2's open item 5 is the reason this is worth a trajectory:
    `frac_repulsive` moves 1.00 -> 0.50 -> 0.80 over ~90k steps while the
    violation count stays flat, so something reorganizes WHICH subspace the
    violations occupy without changing how many there are. This is a
    weights-only per-layer scalar measuring exactly that interaction.

    `henrici_absolute_unclamped` is returned next to the clamped value. The
    previous version clamped at zero silently; a materially negative
    unclamped value means the block parse disagrees with T, which is a bug
    signal rather than numerical noise.
    """
    eig_energy = complex_energy_fraction(block_data)["eigenvalue_energy"]
    t_frob_sq = float(block_data["t_frob_sq"])
    raw = t_frob_sq - eig_energy

    return {
        "henrici_absolute": float(max(raw, 0.0)),
        "henrici_absolute_unclamped": float(raw),
        "henrici_relative": float(max(raw, 0.0) / max(t_frob_sq, 1e-300)),
        "t_frob_sq": t_frob_sq,
        "eigenvalue_energy": float(eig_energy),
    }


# ---------------------------------------------------------------------------
# Rotation planes — bases, never (d, d) projectors
# ---------------------------------------------------------------------------

def top_rotation_planes(block_data: dict, top_k: int = 32) -> dict:
    """
    The `top_k` rotation planes by modulus, as `(d, 2)` orthonormal bases.

    Returns `bases` (list of (d, 2)), `rhos`, `thetas`, `signs`, `indices`,
    and `dim_rotation` / `dim_real`.

    NOT projectors. `P = plane @ plane.T` is (d, d); the previous version
    built `top_k` of them plus two combined ones PER LAYER, which is ~7 GB at
    d=1024 x 24 layers and ~27 GB at d=2048. Every downstream use is
    `X @ P` or `trace(P M)`, both of which factor through the basis —
    see `project_onto_planes` and `plane_energy`.
    """
    blocks = sorted(block_data["blocks_2x2"], key=lambda b: b["rho"], reverse=True)
    k = min(int(top_k), len(blocks))
    sel = blocks[:k]
    return {
        "bases": [b["plane"] for b in sel],
        "rhos": [float(b["rho"]) for b in sel],
        "thetas": [float(b["theta"]) for b in sel],
        "signs": [int(b["sign"]) for b in sel],
        "indices": [int(b["idx"]) for b in sel],
        "dim_rotation": int(2 * block_data["n_complex"]),
        "dim_real": int(block_data["n_real"]),
        "d": int(block_data["d"]),
    }


def project_onto_planes(X: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    """
    Squared norm of each row of X inside each plane: `(n_tokens, k)`.

    `||P_j x||^2 = ||B_j^T x||^2` for an orthonormal `B_j` of shape (d, 2),
    so this never forms `P_j`. O(n d k) rather than O(n d^2 k).
    """
    X = np.asarray(X, dtype=np.float64)
    if not len(bases):
        return np.zeros((X.shape[0], 0))
    return np.stack([np.sum((X @ B) ** 2, axis=1) for B in bases], axis=1)


def plane_energy(M: np.ndarray, bases: Sequence[np.ndarray]) -> np.ndarray:
    """
    `||B_j^T M B_j||_F^2` per plane — how much of an operator M acts within
    each rotation plane. Again without forming a (d, d) projector.
    """
    M = np.asarray(M, dtype=np.float64)
    if not len(bases):
        return np.zeros(0)
    return np.array([float(np.sum((B.T @ M @ B) ** 2)) for B in bases])


def rotation_subspace_fraction(X: np.ndarray, block_data: dict) -> float:
    """
    Fraction of ||X||_F^2 lying in the span of ALL rotation planes.

    Uses the stacked (d, 2*n_complex) basis directly. The combined projector
    the previous version built for this is (d, d) and unnecessary.
    """
    blocks = block_data["blocks_2x2"]
    if not blocks:
        return 0.0
    B = np.concatenate([b["plane"] for b in blocks], axis=1)   # (d, 2*n_complex)
    X = np.asarray(X, dtype=np.float64)
    total = float(np.sum(X ** 2))
    if total < 1e-300:
        return 0.0
    return float(np.sum((X @ B) ** 2) / total)


# ---------------------------------------------------------------------------
# Per-layer scalars
# ---------------------------------------------------------------------------

def layer_scalars(OV: np.ndarray, layer_name: str = "",
                  top_k: int = 0) -> dict:
    """
    Every Block 1a scalar for one OV matrix, with no (d, d) array retained.

    `top_k > 0` additionally returns the top rotation-plane bases. Left at 0
    by default because a checkpoint sweep wants the scalars, not 24 x 27 sets
    of plane bases.
    """
    blocks = extract_schur_blocks(OV)
    energy = complex_energy_fraction(blocks)
    angles = rotation_angle_stats(blocks)
    henrici = henrici_nonnormality(blocks)

    out = {
        "layer": layer_name,
        "d": blocks["d"],
        "ov_frob": float(np.linalg.norm(np.asarray(OV, dtype=np.float64), "fro")),
        **energy,
        **angles,
        **henrici,
        "complex_energy_fraction_legacy_per_block":
            rotational_fraction_per_block(blocks),
    }
    if top_k:
        out["planes"] = top_rotation_planes(blocks, top_k)
    return out


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

def gaussian_null_matrices(OV: np.ndarray, n_draws: int = 16, rng=None) -> list:
    """
    Norm-matched iid Gaussian matrices of the same shape as OV.

    The construction matches `core/pythia_registry.build_pythia_random_baseline`'s
    scheme: structure destroyed, Frobenius norm preserved. That is the
    project's continuity control, so Block 1a's null uses the same object
    rather than inventing a second notion of "random".
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    OV = np.asarray(OV, dtype=np.float64)
    target = float(np.linalg.norm(OV, "fro"))
    out = []
    for _ in range(int(n_draws)):
        M = rng.normal(size=OV.shape)
        n = float(np.linalg.norm(M, "fro"))
        out.append(M * (target / max(n, 1e-300)))
    return out


def null_comparison(OV: np.ndarray, statistic: str = "complex_energy_fraction",
                    n_draws: int = 16, rng=None) -> dict:
    """
    Observed Block 1a statistic against the norm-matched Gaussian null,
    reported through `core.nulls.sigma_from_null`.

    This is the missing control on the phase's headline. A Gaussian matrix is
    essentially all complex pairs, so if the observed fraction sits inside
    the null the statement "OV is 84-97% rotational" is a statement about
    square matrices and not about the trained network. A `z_score` near zero is
    therefore the EXPECTED result for the fraction, and it is worth
    reporting rather than assuming: the interesting nulls are on `theta`
    (a Gaussian's angles are near-uniform on [0, pi]) and on
    `henrici_relative` (a Gaussian is strongly non-normal, so a trained
    matrix sitting BELOW the null means training has made V more normal).

    `statistic` may name any scalar returned by `layer_scalars`.
    """
    from core.nulls import sigma_from_null

    observed = layer_scalars(OV)[statistic]
    null_vals = np.array([
        layer_scalars(M)[statistic]
        for M in gaussian_null_matrices(OV, n_draws=n_draws, rng=rng)
    ], dtype=np.float64)

    # sigma_from_null already returns observed / null_mean / null_std /
    # z_score / percentile / n_null. Nothing is recomputed here — the
    # provenance fields are added, not the statistics.
    res = dict(sigma_from_null(float(observed), null_vals))
    res.update({
        "statistic": statistic,
        "n_draws": int(n_draws),
        "null_construction": "norm_matched_gaussian",
    })
    return res


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def precision_surface(ov_list: Sequence[np.ndarray],
                      layer_names: Optional[Sequence[str]] = None,
                      **kwargs) -> dict:
    """
    `core.precision_policy.analyze_ov_precision` over this model's OV list.

    Wired in because it was written specifically against this block (its
    docstring names `p2b_imaginary.layernorm_jacobian.rotational_fraction`)
    and was never called from the runner. It answers whether "84-97% complex"
    survives the fp16 round-trip the checkpoints actually went through, over
    a sweep of the relative tolerance rather than at the single shipped 0.01.

    `frac_fn` defaults to `complex_energy_fraction_relative` here, which is
    the same function `precision_policy` would import — passed explicitly so
    the dependency is visible at the call site rather than resolved by a
    lazy import inside `core`.
    """
    from core.precision_policy import analyze_ov_precision
    kwargs.setdefault("frac_fn", complex_energy_fraction_relative)
    return analyze_ov_precision(ov_list, layer_names, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analyze_rotational_spectrum(
    ov_data: dict,
    top_k_planes: int = 0,
    with_nulls: bool = False,
    null_statistics: Sequence[str] = ("complex_energy_fraction",
                                      "theta_mean", "henrici_relative"),
    n_null_draws: int = 16,
    rng=None,
) -> dict:
    """
    Block 1a for one checkpoint.

    Parameters
    ----------
    top_k_planes  : plane bases to retain per layer. 0 (default) keeps none —
                    a checkpoint sweep wants scalars. Consumers that need
                    planes (the FFN-rotation block, Block 3) ask for them.
    with_nulls    : run `null_comparison` per layer. Costs
                    `n_null_draws` extra Schur decompositions per layer, so
                    it is off by default and worth doing at a subset of
                    checkpoints rather than all 27.

    Returns per-layer scalars plus a cross-layer summary. Carries
    `checkpoint_step` and `model_stem` straight through from `ov_data` so a
    result can be placed on the training axis without re-parsing a filename.
    """
    is_per_layer = bool(ov_data["is_per_layer"])
    ov_list = (list(ov_data["ov_total"]) if is_per_layer
               else [ov_data["ov_total"]])
    layer_names = list(ov_data.get("layer_names") or
                       [f"layer_{i}" for i in range(len(ov_list))])

    per_layer = [
        layer_scalars(OV, name, top_k=top_k_planes)
        for OV, name in zip(ov_list, layer_names)
    ]

    if with_nulls:
        for OV, rec in zip(ov_list, per_layer):
            rec["nulls"] = {
                stat: null_comparison(OV, stat, n_draws=n_null_draws, rng=rng)
                for stat in null_statistics
            }

    return {
        "is_per_layer": is_per_layer,
        "layer_names": layer_names,
        "model_stem": ov_data.get("model_stem"),
        "checkpoint_step": ov_data.get("checkpoint_step"),
        "per_layer": per_layer,
        "summary": _cross_layer_summary(per_layer, layer_names),
    }


def _cross_layer_summary(per_layer: Sequence[dict],
                         layer_names: Sequence[str]) -> dict:
    """
    Depth profile reduced to the scalars a checkpoint trajectory plots.

    `*_argmax_layer` is the layer NAME, not the index, so a summary read back
    from JSON does not need the layer list to be interpretable.
    """
    if not per_layer:
        return {}

    def col(key):
        return np.array([r.get(key, np.nan) for r in per_layer], dtype=np.float64)

    cef = col("complex_energy_fraction")
    hen = col("henrici_relative")
    th = col("theta_mean")
    rep = col("frac_repulsive_real_part")

    def argmax_name(a):
        if not np.isfinite(a).any():
            return None
        return layer_names[int(np.nanargmax(a))]

    return {
        "n_layers": len(per_layer),
        "complex_energy_fraction_mean": float(np.nanmean(cef)),
        "complex_energy_fraction_min": float(np.nanmin(cef)),
        "complex_energy_fraction_max": float(np.nanmax(cef)),
        "henrici_relative_mean": float(np.nanmean(hen)),
        "henrici_relative_max": float(np.nanmax(hen)),
        "henrici_argmax_layer": argmax_name(hen),
        "theta_mean_across_layers": float(np.nanmean(th)),
        "theta_std_across_layers": float(np.nanstd(th)),
        "frac_repulsive_real_part_mean": float(np.nanmean(rep)),
        "dim_complex_fraction_mean": float(np.nanmean(col("dim_complex_fraction"))),
        "complex_energy_fraction_legacy_mean": float(
            np.nanmean(col("complex_energy_fraction_legacy_per_block"))),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("planes",)


def summary_to_json(result: dict) -> dict:
    """
    JSON-serializable Block 1a output. Drops plane bases; every other value
    is already a scalar, because nothing (d, d) is retained in the first
    place.
    """
    def clean(rec):
        return {k: v for k, v in rec.items() if k not in _ARRAY_KEYS}

    return {
        "is_per_layer": bool(result["is_per_layer"]),
        "model_stem": result.get("model_stem"),
        "checkpoint_step": result.get("checkpoint_step"),
        "layer_names": list(result["layer_names"]),
        "per_layer": [clean(r) for r in result["per_layer"]],
        "summary": result["summary"],
    }


def summary_lines(js: dict) -> list:
    """LLM-consumable summary block."""
    s = js.get("summary", {})
    step = js.get("checkpoint_step")
    lines = [
        "--- Block 1a: rotational spectrum ---",
        f"  Model: {js.get('model_stem')}" + (f"  step {step}" if step is not None else ""),
        f"  Layers: {s.get('n_layers', 0)}",
        f"  Complex energy fraction (per-eigenvalue): "
        f"mean {s.get('complex_energy_fraction_mean', float('nan')):.4f} "
        f"[{s.get('complex_energy_fraction_min', float('nan')):.4f}, "
        f"{s.get('complex_energy_fraction_max', float('nan')):.4f}]",
        f"  Same, legacy per-block convention: "
        f"{s.get('complex_energy_fraction_legacy_mean', float('nan')):.4f}",
        f"  Rotating dimensions: {s.get('dim_complex_fraction_mean', float('nan')):.4f}",
        f"  Mean theta: {s.get('theta_mean_across_layers', float('nan')):.4f} rad "
        f"(sd {s.get('theta_std_across_layers', float('nan')):.4f})",
        f"  Repulsive real part: {s.get('frac_repulsive_real_part_mean', float('nan')):.4f}",
        f"  Henrici (relative): mean {s.get('henrici_relative_mean', float('nan')):.4f}, "
        f"max {s.get('henrici_relative_max', float('nan')):.4f} "
        f"at {s.get('henrici_argmax_layer')}",
    ]
    nulls = (js.get("per_layer") or [{}])[0].get("nulls")
    if nulls:
        lines.append("  Nulls (norm-matched Gaussian, layer 0):")
        for stat, res in nulls.items():
            lines.append(
                f"    {stat}: observed {res['observed']:.4f} vs null "
                f"{res['null_mean']:.4f} +/- {res['null_std']:.4f} "
                f"(z {res.get('z_score', float('nan')):.2f}, "
                f"pct {res.get('percentile', float('nan')):.1f})"
            )
    return lines


__all__ = [
    "SUBDIAGONAL_REL_TOL",
    "COMPLEX_REL_TOL",
    "extract_schur_blocks",
    "complex_energy_fraction",
    "rotational_fraction_per_block",
    "complex_energy_fraction_relative",
    "rotation_angle_stats",
    "henrici_nonnormality",
    "top_rotation_planes",
    "project_onto_planes",
    "plane_energy",
    "rotation_subspace_fraction",
    "layer_scalars",
    "gaussian_null_matrices",
    "null_comparison",
    "precision_surface",
    "analyze_rotational_spectrum",
    "summary_to_json",
    "summary_lines",
]
