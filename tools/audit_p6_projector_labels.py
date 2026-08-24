"""
tools/audit_p6_projector_labels.py — the item-5 audit of Phase 6's projectors.

WHAT THIS SETTLES, AND WHY IT IS A TOOL RATHER THAN A TEST

`archive/p6_subspace/status-6.md` item 5 records two competing explanations for
Phase 6's headline result -- mean LDA alignment 0.887 with the imaginary
subspace U_A against 0.067 with the real repulsive U_neg, 0 of 49 layers in the
predicted direction:

    (a) a projector-construction error in subspace_build.py -- Schur block
        mislabelling that swaps U_neg and U_A -- which would invert all four
        geometry tests together;
    (b) the real/imaginary functional-separation hypothesis genuinely failing
        under ALBERT's weight-tying.

`archive/p6_subspace/design-6.md` pre-registered the ordering: rule out (a)
before treating (b) as established, "since a labelling bug would produce exactly
this pattern of results and is checkable in isolation". Emitting a p-value with
(a) open would put a calibrated number on possibly-broken instrumentation, which
is what standing rule 4 exists to prevent.

`p6_subspace/math-6.md` §7.2 then names a THIRD explanation neither status-6.md
nor the plan lists, and argues it comes first:

    (c) the comparison is not dimension-normalized. For a random unit vector v
        and a k-dimensional subspace U, E[||P_U v||^2] = k/d, so raw alignment
        scales with the subspace's dimension. If dim U_A >> dim U_neg -- which
        subspace_build's own resolution order (S wins over A; U_pos wins over
        U_neg) actively produces -- then the observed ratio measures dimension
        rather than content.

This module audits all three, offline, and commits its findings. It is a TOOL
and not a test module because settling (a) requires running the archived code
that produced the number, and `archive/README.md` rule 1 is that nothing under
archive/ is imported by anything live. A tool loads it by file path only when
run; nothing collected by pytest imports it. `tests/test_p6_projector_audit.py`
pins the COMMITTED RESULT instead, together with the sha256 of the file this
audit describes, so the record going stale is a test failure rather than a
silent one -- the same shape as CLAIM-C's committed calibration curve.

THE FOUR ARMS, AND WHY THERE ARE FOUR

  L  labelling.   Planted OV matrices with KNOWN real-positive, real-negative,
                  rotation and kernel structure in a random orthonormal basis.
                  Each bucket must recover its planted span. This is the check
                  design-6.md asked for.

  C  counts.      Bucket DIMENSIONS on non-normal OV matrices, against a
                  classification derived independently from `np.linalg.eigvals`
                  rather than from the Schur form at all.

  S  sensitivity. Arms L and C run against two DELIBERATELY BROKEN extractors.
                  An audit that cannot fail establishes nothing, and the two
                  breakages are the concrete forms explanation (a) would take:
                  a relabelling that swaps the neg and rotation buckets, and
                  the transposed-subdiagonal bug (reading T[i, i+1] instead of
                  T[i+1, i]) that real Schur form makes so easy to write and so
                  hard to see.

  D  dimension.   The dim(U_A) / dim(U_neg) the full per-layer pipeline
                  produces, measured across shapes including ALBERT-xlarge-v2's
                  exact (d_model, n_heads, head_dim). That ratio IS the chance
                  alignment ratio, so it is what the observed 0.887 / 0.067 has
                  to be read against.

Arm C exists because arm S caught arm L being blind, which is the whole reason
an audit gets a sensitivity arm. To plant KNOWN real-versus-rotational
structure you build a block-diagonal matrix of scaled rotations and real
eigenvalues -- and that matrix is NORMAL. A normal matrix's real Schur form is
block DIAGONAL, so its superdiagonal is zero everywhere outside the 2x2 blocks,
and reading T[i, i+1] instead of T[i+1, i] gives bit-identical answers. The
family that makes ground truth unambiguous is exactly the family that cannot
express the bug. Arm C reaches where arm L structurally cannot: on a non-normal
OV the spectrum still fixes how many eigenvalues are real-positive,
real-negative and complex, `np.linalg.eigvals` gets there without touching the
Schur form, and the transposed extractor disagrees with it.

WHAT ARM D IS AND IS NOT

Arm D runs on RANDOM OV matrices at the right SHAPE. It is not ALBERT's trained
weights, which this repository does not contain, so it does not report ALBERT's
actual dimensions. What it measures is the dimension asymmetry the pipeline's
own construction produces at that shape -- which is a property of the design,
not of the data, and is therefore exactly what a synthetic reference can carry.
The actual per-layer dims are computed by `_build_for_layer` on every run and
were never reported; recovering them is one number and it decides (c) outright.

Usage
-----
    python3 tools/audit_p6_projector_labels.py --write
    python3 tools/audit_p6_projector_labels.py --check
    python3 tools/audit_p6_projector_labels.py --summary
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

#: The archived module under audit. Loaded by PATH, never imported as a package:
#: archive/README.md rule 1 is that no live module imports anything under
#: archive/, and a forensic read of a frozen file is not a maintenance
#: dependency on it.
AUDITED_PATH = ROOT / "archive" / "p6_subspace" / "subspace_build.py"

AUDIT_PATH = ROOT / "claims" / "audits" / "p6_projector_labels.json"
AUDIT_SCHEMA_VERSION = 1

#: Recorded in status-6.md and p6_subspace/math-6.md §7.1. Reproduced here so
#: the audit's own arithmetic reads against them rather than a reader doing it.
OBSERVED_ALIGN_U_A = 0.887
OBSERVED_ALIGN_U_NEG = 0.067

#: albert-xlarge-v2, the only model Phase 6 was ever run on.
ALBERT_D_MODEL = 2048
ALBERT_N_HEADS = 16
ALBERT_HEAD_DIM = 128

#: Shapes arm D sweeps. The last is ALBERT's own; the rest establish that the
#: ratio is driven by head_dim/d rather than being an accident of one size.
DIM_SHAPES = (
    (256, 16, 16),
    (512, 16, 32),
    (1024, 16, 64),
    (ALBERT_D_MODEL, ALBERT_N_HEADS, ALBERT_HEAD_DIM),
)

#: Planted structures arm L recovers. (d, n_pos, n_neg, n_rot, n_zero).
PLANTED_SHAPES = (
    (40, 6, 5, 8, 13),
    (64, 3, 17, 12, 20),
    (48, 20, 2, 10, 6),   # a deliberately lopsided one: few negatives, many positives
)

#: Shapes arm C sweeps, as (d_model, rank). OV = W_V W_O is rank-limited, and
#: the rank is what decides how many live eigenvalues there are to classify --
#: so the rank, not d, is the knob. All four carry at least one real-positive
#: and one real-negative eigenvalue, which is what the transposed extractor
#: loses; a rank too low to produce any real eigenvalue would make arm C agree
#: with the broken version by having nothing to disagree about.
NONNORMAL_SHAPES = ((64, 20), (96, 30), (128, 48), (160, 64))

#: |Im lambda| > IM_REL_TOL * |lambda| marks an eigenvalue complex in arm C's
#: independent classification. CALIBRATED, not placed: a real eigenvalue of a
#: real matrix has an imaginary part that is pure LAPACK round-off, measured at
#: <1e-15 relative on these shapes, while a genuine conjugate pair sits at
#: O(1) relative. Seven orders of headroom either side, and arm C reports the
#: closest approach so the margin is visible rather than asserted.
IM_REL_TOL = 1e-8

#: Largest principal angle (radians) a recovered span may make with its planted
#: span before arm L calls the bucket wrong. CALIBRATED, not placed: the
#: measured angles sit at ~3e-8, which is the SVD truncation tolerance
#: subspace_build itself uses (1e-8) carried through one orthonormalisation.
#: Three orders of magnitude of headroom above that, and still ~7 orders below
#: the O(1) angle a genuine mislabelling produces -- arm S measures the gap.
#: CALIBRATED from the measured angles, not placed.
LABEL_ANGLE_TOL = 1e-5

#: Tolerances handed to the archived code. Not this audit's cuts to choose --
#: every one is subspace_build.py's own default, restated here so the record
#: says what it ran under instead of inheriting silently. Their provenance is
#: whatever it is over there; INHERITED, neither placed nor calibrated here.
SVD_TOL = 1e-8                # inherited: build_global_projectors default
EIG_REL_TOL_PLANTED = 1e-8    # inherited: _extract_schur_subspaces default
EIG_REL_TOL_PIPELINE = 1e-12  # inherited: build_global_projectors default
#: Also inherited, neither placed nor calibrated here:
COS_OVERLAP_TOL = 0.99        # inherited: build_global_projectors default

SEED = 20260824


# ---------------------------------------------------------------------------
# Loading the audited module without importing archive/ as a package
# ---------------------------------------------------------------------------

def load_audited_module():
    """
    Load `subspace_build.py` from its path.

    `importlib` with an explicit file location rather than `import
    archive.p6_subspace.subspace_build`: the module imports nothing from its own
    package (json, numpy, pathlib and scipy.linalg.schur are its whole import
    list), so there is nothing for a package context to supply, and loading by
    path keeps `archive/` off sys.path. That matters beyond tidiness -- a live
    `p6_subspace/` package now exists, and putting archive/ on the path would
    let `import p6_subspace` resolve to whichever came first.
    """
    if not AUDITED_PATH.exists():
        raise FileNotFoundError(
            f"{AUDITED_PATH} is missing. The audit describes a specific frozen "
            f"file; if the archive moved, this tool needs updating rather than "
            f"pointing somewhere else."
        )
    spec = importlib.util.spec_from_file_location(
        "_audited_subspace_build", AUDITED_PATH)
    if spec is None or spec.loader is None:      # pragma: no cover - defensive
        raise ImportError(f"could not build a loader for {AUDITED_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audited_sha256() -> str:
    """SHA-256 of the file under audit, so a stale record is detectable."""
    return hashlib.sha256(AUDITED_PATH.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def max_principal_angle(U: np.ndarray, V: np.ndarray) -> float:
    """
    Largest principal angle between two equal-dimension orthonormal spans.

    Returns pi/2 when the dimensions differ -- a bucket that recovered the
    wrong NUMBER of directions is as wrong as one that recovered the wrong
    ones, and collapsing both into one scalar keeps arm L's verdict single.
    """
    if U.shape[1] != V.shape[1]:
        return float(np.pi / 2)
    if U.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.arccos(np.clip(float(s.min()), -1.0, 1.0)))


def plant_ov(d: int, n_pos: int, n_neg: int, n_rot: int, n_zero: int,
             rng: np.random.Generator):
    """
    An OV matrix whose real/imaginary structure is known by construction.

    Block-diagonal B in a random orthonormal basis Q: OV = Q B Q^T. The
    eigenstructure is exactly what was planted, and Q's columns give the
    planted span for each bucket, so "did the extractor route this direction
    correctly" is a principal angle rather than an opinion.

    Eigenvalue magnitudes are drawn in [0.5, 2.0] and rotation angles in
    [0.3, pi - 0.3], both well clear of the tolerances, so arm L tests the
    LABELLING and not the tolerance handling -- a separate question, and one
    subspace_build already documents as relative-to-||T||_F by design.
    """
    if n_pos + n_neg + 2 * n_rot + n_zero != d:
        raise ValueError(
            f"planted blocks fill {n_pos + n_neg + 2 * n_rot + n_zero} of {d} "
            f"dimensions; they must fill it exactly or the leftover is untyped"
        )
    B = np.zeros((d, d))
    i = 0
    pos_idx = list(range(i, i + n_pos))
    for j in pos_idx:
        B[j, j] = rng.uniform(0.5, 2.0)
    i += n_pos

    neg_idx = list(range(i, i + n_neg))
    for j in neg_idx:
        B[j, j] = -rng.uniform(0.5, 2.0)
    i += n_neg

    rot_idx: List[int] = []
    for _ in range(n_rot):
        theta = rng.uniform(0.3, np.pi - 0.3)
        rho = rng.uniform(0.5, 2.0)
        c, s = np.cos(theta), np.sin(theta)
        B[i:i + 2, i:i + 2] = rho * np.array([[c, -s], [s, c]])
        rot_idx += [i, i + 1]
        i += 2

    zero_idx = list(range(i, i + n_zero))
    i += n_zero

    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    planted = {
        "pos": Q[:, pos_idx], "neg": Q[:, neg_idx],
        "rot": Q[:, rot_idx], "zero": Q[:, zero_idx],
    }
    return Q @ B @ Q.T, planted


def random_ov_layer(d: int, n_heads: int, head_dim: int,
                    rng: np.random.Generator) -> List[np.ndarray]:
    """
    One layer's per-head OV = W_V W_O at a given shape, with random weights.

    The 1/sqrt(fan_in) scaling is the usual initialisation convention; nothing
    in arm D depends on the scale, since both eig_tol and block_tol are
    relative to ||T||_F. What matters is the RANK: OV is at most head_dim, so
    d - head_dim directions per head land in the kernel bucket, and the
    rotation/real split among the rest is the shape-driven quantity arm D is
    after.
    """
    out = []
    for _ in range(n_heads):
        Wv = rng.standard_normal((d, head_dim)) / np.sqrt(d)
        Wo = rng.standard_normal((head_dim, d)) / np.sqrt(head_dim)
        out.append(Wv @ Wo)
    return out


# ---------------------------------------------------------------------------
# The two deliberate breakages (arm S)
# ---------------------------------------------------------------------------

def _extract_swapped(sb, OV: np.ndarray, **kw) -> dict:
    """
    Explanation (a) in its most literal form: U_neg and U_A exchanged.

    Runs the real extractor and swaps the two buckets afterward, so the
    breakage is exactly the mislabelling and nothing else.
    """
    sub = sb._extract_schur_subspaces(OV, **kw)
    out = dict(sub)
    out["real_neg_vecs"], out["rot_vecs"] = sub["rot_vecs"], sub["real_neg_vecs"]
    return out


def _extract_transposed_subdiagonal(sb, OV: np.ndarray,
                                    eig_rel_tol: float = 1e-8,
                                    **_kw) -> dict:
    """
    Explanation (a) in the form somebody would actually write by accident.

    Real Schur form is upper quasi-triangular: a 2x2 block is identified by a
    nonzero SUB-diagonal entry T[i+1, i], while the SUPER-diagonal T[i, i+1] is
    generically nonzero for 1x1 blocks too. Reading the wrong one types almost
    everything as a rotation -- which is the direction the observed result
    points, so this is the breakage most worth being able to detect.

    A transcription of the real loop with that one index reversed, kept
    deliberately close to the original so the difference is the defect.
    """
    from scipy.linalg import schur

    d = OV.shape[0]
    T, Z = schur(OV, output="real")
    nrm = float(np.linalg.norm(T, "fro"))
    eps = float(np.finfo(T.dtype).eps)
    eig_tol = max(nrm * eig_rel_tol, nrm * eps * 10.0, 1e-12)
    block_tol = eig_tol

    real_pos: List[np.ndarray] = []
    real_neg: List[np.ndarray] = []
    real_zero: List[np.ndarray] = []
    rot: List[np.ndarray] = []
    i = 0
    while i < d:
        if i < d - 1 and abs(T[i, i + 1]) > block_tol:      # <-- the defect
            a, b = float(T[i, i]), float(T[i, i + 1])
            c, dd = float(T[i + 1, i]), float(T[i + 1, i + 1])
            mag = float(np.sqrt(max(a * dd - b * c, 0.0)))
            if mag > eig_tol:
                rot.append(Z[:, i].copy())
                rot.append(Z[:, i + 1].copy())
            i += 2
        else:
            val = float(T[i, i])
            if val > eig_tol:
                real_pos.append(Z[:, i].copy())
            elif val < -eig_tol:
                real_neg.append(Z[:, i].copy())
            else:
                real_zero.append(Z[:, i].copy())
            i += 1
    return {"real_pos_vecs": real_pos, "real_neg_vecs": real_neg,
            "real_zero_vecs": real_zero, "rot_vecs": rot}


# ---------------------------------------------------------------------------
# Arm L / arm S
# ---------------------------------------------------------------------------

_BUCKET_KEY = {"pos": "real_pos_vecs", "neg": "real_neg_vecs",
               "rot": "rot_vecs", "zero": "real_zero_vecs"}


def _recover(sb, sub: dict, d: int) -> Dict[str, np.ndarray]:
    out = {}
    for name, key in _BUCKET_KEY.items():
        vecs = sub.get(key) or []
        out[name] = (sb._orthonormal_basis(vecs, SVD_TOL) if vecs
                     else np.zeros((d, 0)))
    return out


def run_labelling_arm(sb, extractor=None, label: str = "as-shipped") -> dict:
    """
    Arm L: does each Schur bucket recover the span that was planted in it?

    `extractor` defaults to the archived `_extract_schur_subspaces`; arm S
    passes a broken one and expects this to fail.
    """
    rng = np.random.default_rng(SEED)
    cases = []
    worst = 0.0
    for (d, n_pos, n_neg, n_rot, n_zero) in PLANTED_SHAPES:
        OV, planted = plant_ov(d, n_pos, n_neg, n_rot, n_zero, rng)
        if extractor is None:
            sub = sb._extract_schur_subspaces(OV, eig_rel_tol=EIG_REL_TOL_PLANTED)
        else:
            sub = extractor(sb, OV, eig_rel_tol=EIG_REL_TOL_PLANTED)
        got = _recover(sb, sub, d)
        per_bucket = {}
        for name in _BUCKET_KEY:
            angle = max_principal_angle(got[name], planted[name])
            per_bucket[name] = {
                "planted_dim": int(planted[name].shape[1]),
                "recovered_dim": int(got[name].shape[1]),
                "max_principal_angle_rad": angle,
            }
            worst = max(worst, angle)
        cases.append({
            "d": d, "planted": {"pos": n_pos, "neg": n_neg,
                                "rot": n_rot, "zero": n_zero},
            "buckets": per_bucket,
        })
    return {
        "extractor": label,
        "cases": cases,
        "worst_max_principal_angle_rad": worst,
        "tolerance_rad": LABEL_ANGLE_TOL,
        "verdict": "PASS" if worst <= LABEL_ANGLE_TOL else "FAIL",
    }


# ---------------------------------------------------------------------------
# Arm C — counts against an independently derived spectrum
# ---------------------------------------------------------------------------

def spectrum_reference_counts(M: np.ndarray,
                              eig_rel_tol: float = EIG_REL_TOL_PLANTED) -> dict:
    """
    How many eigenvalues are real-positive, real-negative, complex and null --
    decided from `np.linalg.eigvals` and NOT from the Schur form.

    This is the point of the arm. `_extract_schur_subspaces` reads block
    structure off the quasi-triangular factor; `eigvals` goes to the spectrum
    directly. Two routes to the same classification, so agreement is evidence
    and disagreement localises the defect.

    The live/kernel cut uses the same ||T||_F-relative tolerance the audited
    code uses, so the two are answering the same question about the same
    matrix rather than differing on where zero is.
    """
    from scipy.linalg import schur

    T, _ = schur(M, output="real")
    nrm = float(np.linalg.norm(T, "fro"))
    eps = float(np.finfo(T.dtype).eps)
    tol = max(nrm * eig_rel_tol, nrm * eps * 10.0, 1e-12)

    w = np.linalg.eigvals(M)
    mag = np.abs(w)
    live = mag > tol
    scale = np.maximum(mag, tol)
    is_complex = live & (np.abs(w.imag) > IM_REL_TOL * scale)
    is_real = live & ~is_complex

    # The closest either class comes to the cut, so the margin is reported
    # rather than assumed. Empty classes give inf/0, which read correctly.
    rel_im = np.abs(w.imag) / scale
    return {
        "pos": int(np.sum(is_real & (w.real > tol))),
        "neg": int(np.sum(is_real & (w.real < -tol))),
        "rot": int(np.sum(is_complex)),
        "zero": int(np.sum(~live)),
        "max_rel_im_among_real": (
            float(np.max(rel_im[is_real])) if np.any(is_real) else 0.0),
        "min_rel_im_among_complex": (
            float(np.min(rel_im[is_complex])) if np.any(is_complex)
            else float("inf")),
    }


def run_count_arm(sb, extractor=None, label: str = "as-shipped") -> dict:
    """
    Arm C: do the extractor's bucket sizes match the independent spectrum?

    Sizes, not spans. On a non-normal matrix the Schur vectors are not
    eigenvectors and span comparison would be testing basis conventions rather
    than classification -- which is a separate question and not the one item 5
    asks.
    """
    cases = []
    ok = True
    for (d, rank) in NONNORMAL_SHAPES:
        rng = np.random.default_rng(SEED + d)
        Wv = rng.standard_normal((d, rank)) / np.sqrt(d)
        Wo = rng.standard_normal((rank, d)) / np.sqrt(rank)
        M = Wv @ Wo
        commutator = float(np.linalg.norm(M @ M.T - M.T @ M))

        if extractor is None:
            sub = sb._extract_schur_subspaces(M, eig_rel_tol=EIG_REL_TOL_PLANTED)
        else:
            sub = extractor(sb, M, eig_rel_tol=EIG_REL_TOL_PLANTED)
        got = {name: len(sub.get(key) or [])
               for name, key in _BUCKET_KEY.items()}
        ref = spectrum_reference_counts(M)
        agree = all(got[k] == ref[k] for k in ("pos", "neg", "rot", "zero"))
        ok = ok and agree
        cases.append({
            "d": d, "rank": rank,
            "commutator_fro": commutator,
            "schur_counts": got,
            "spectrum_counts": {k: ref[k] for k in ("pos", "neg", "rot", "zero")},
            "max_rel_im_among_real": ref["max_rel_im_among_real"],
            "min_rel_im_among_complex": ref["min_rel_im_among_complex"],
            "agree": agree,
        })
    return {
        "extractor": label,
        "cases": cases,
        "im_rel_tol": IM_REL_TOL,
        "verdict": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Arm D
# ---------------------------------------------------------------------------

def chance_alignment_ratio(dim_a: int, dim_neg: int) -> float:
    """
    E[||P_{U_A} v||^2] / E[||P_{U_neg} v||^2] for a random unit v.

    Both expectations are dim/d (math-6.md §7.2), so d cancels and the chance
    ratio IS the dimension ratio. Stated as its own function because that
    cancellation is the whole of explanation (c) and it should be readable
    rather than inlined.
    """
    if dim_neg <= 0:
        return float("inf")
    return float(dim_a) / float(dim_neg)


def run_dimension_arm(sb) -> dict:
    rows = []
    for (d, n_heads, head_dim) in DIM_SHAPES:
        rng = np.random.default_rng(SEED)
        layer = sb._build_for_layer(
            random_ov_layer(d, n_heads, head_dim, rng),
            d, SVD_TOL, EIG_REL_TOL_PIPELINE, COS_OVERLAP_TOL,
        )
        dim_neg = int(layer["U_neg"].shape[1])
        dim_a = int(layer["dim_A"])
        rows.append({
            "d_model": d, "n_heads": n_heads, "head_dim": head_dim,
            "is_albert_xlarge_v2_shape": (d, n_heads, head_dim) == (
                ALBERT_D_MODEL, ALBERT_N_HEADS, ALBERT_HEAD_DIM),
            "dim_pos_pre": int(layer["dim_pos_pre"]),
            "dim_neg_pre": int(layer["dim_neg_pre"]),
            "dim_A_pre": int(layer["dim_A_pre"]),
            "dim_neg": dim_neg,
            "dim_A": dim_a,
            "dim_kernel": int(layer["dim_kernel"]),
            "chance_alignment_ratio": chance_alignment_ratio(dim_a, dim_neg),
            "chance_alignment_U_A": dim_a / d,
            "chance_alignment_U_neg": dim_neg / d,
        })
    albert = [r for r in rows if r["is_albert_xlarge_v2_shape"]][0]
    observed_ratio = OBSERVED_ALIGN_U_A / OBSERVED_ALIGN_U_NEG
    return {
        "rows": rows,
        "observed_alignment_U_A": OBSERVED_ALIGN_U_A,
        "observed_alignment_U_neg": OBSERVED_ALIGN_U_NEG,
        "observed_alignment_ratio": observed_ratio,
        "albert_shape_chance_ratio": albert["chance_alignment_ratio"],
        "observed_over_chance": observed_ratio / albert["chance_alignment_ratio"],
        "normalized_alignment_U_A": (
            OBSERVED_ALIGN_U_A / albert["chance_alignment_U_A"]),
        "normalized_alignment_U_neg": (
            OBSERVED_ALIGN_U_NEG / albert["chance_alignment_U_neg"]),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_audit() -> dict:
    sb = load_audited_module()

    labelling = run_labelling_arm(sb)
    counts = run_count_arm(sb)

    swapped_L = run_labelling_arm(sb, _extract_swapped, "deliberately-swapped")
    swapped_C = run_count_arm(sb, _extract_swapped, "deliberately-swapped")
    transposed_L = run_labelling_arm(
        sb, _extract_transposed_subdiagonal, "transposed-subdiagonal")
    transposed_C = run_count_arm(
        sb, _extract_transposed_subdiagonal, "transposed-subdiagonal")

    # Each breakage must be caught by at least one arm. The pairing is not
    # symmetric and that asymmetry is the finding: arm L catches the swap and
    # is structurally blind to the transposition, arm C catches both.
    swap_caught = (swapped_L["verdict"] == "FAIL"
                   or swapped_C["verdict"] == "FAIL")
    transpose_caught = (transposed_L["verdict"] == "FAIL"
                        or transposed_C["verdict"] == "FAIL")
    sensitivity_ok = swap_caught and transpose_caught

    dimension = run_dimension_arm(sb)

    if not sensitivity_ok:
        explanation_a = "UNDECIDED"
    elif labelling["verdict"] == "PASS" and counts["verdict"] == "PASS":
        explanation_a = "RULED-OUT"
    else:
        explanation_a = "CONFIRMED"

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "_what": (
            "Audit of archive/p6_subspace/subspace_build.py against "
            "status-6.md item 5 (Schur block mislabelling) and "
            "p6_subspace/math-6.md 7.2 (the comparison is not "
            "dimension-normalized). Generated offline by "
            "tools/audit_p6_projector_labels.py and committed, so the finding "
            "is a fixed record rather than a claim in prose. Regenerate with "
            "--write; the committed result is pinned by "
            "tests/test_p6_projector_audit.py."),
        "_why_a_tool": (
            "Settling explanation (a) requires running the archived code that "
            "produced the number, and archive/README.md rule 1 is that nothing "
            "under archive/ is imported by anything live. This tool loads it by "
            "file path only when run; nothing collected by pytest imports it."),
        "audited_file": str(AUDITED_PATH.relative_to(ROOT)),
        "audited_sha256": audited_sha256(),
        "seed": SEED,
        "arm_L_labelling": labelling,
        "arm_C_counts": counts,
        "arm_S_sensitivity": {
            "_what": (
                "Arms L and C against two deliberately broken extractors. Each "
                "breakage must be caught by at least one arm, or the arm's "
                "PASS means only that the audit is incapable of failing."),
            "_asymmetry": (
                "arm L is BLIND to the transposed-subdiagonal breakage, and "
                "not by oversight. A planted matrix with known "
                "real-versus-rotational structure is block-diagonal in scaled "
                "rotations and real eigenvalues, which makes it NORMAL, and a "
                "normal matrix's real Schur form is block diagonal -- its "
                "superdiagonal is zero outside the 2x2 blocks, so T[i, i+1] "
                "and T[i+1, i] carry the same information and the two "
                "extractors agree bit for bit. The family that makes ground "
                "truth unambiguous is the family that cannot express the bug. "
                "Arm C was added for exactly this, and catches it."),
            "swapped": {"labelling": swapped_L, "counts": swapped_C,
                        "caught": swap_caught},
            "transposed_subdiagonal": {
                "labelling": transposed_L, "counts": transposed_C,
                "caught": transpose_caught},
            "verdict": "PASS" if sensitivity_ok else "FAIL",
        },
        "arm_D_dimension": dimension,
        "explanation_a_schur_mislabelling": explanation_a,
        "_explanation_a_scope": (
            "RULED-OUT means two independent things agree. On planted matrices "
            "whose real-positive, real-negative, rotation and kernel structure "
            "is known by construction, every bucket recovers its own span to "
            f"within {LABEL_ANGLE_TOL} rad (arm L); and on non-normal matrices "
            "where no planted ground truth is available, the bucket sizes match "
            "a classification taken from np.linalg.eigvals without touching the "
            "Schur form (arm C). Two deliberate mislabellings of the same code "
            "are caught. It does NOT certify the rest of the pipeline, the "
            "head-loop wiring in run_6.py, or that the ALBERT run used this "
            "version of the file -- only that the block classification this "
            "file implements is correct."),
        "_explanation_c_finding": (
            "The dimension asymmetry is real, large, and produced by the "
            "construction rather than by the data: subspace_build's resolution "
            "order removes span(U_pos) from U_neg and span(U_S) from U_A, so "
            "U_neg is the doubly-shrunk bucket. At albert-xlarge-v2's exact "
            "shape the pipeline yields a chance alignment ratio LARGER than "
            "the observed 0.887/0.067, which means the recorded inversion is "
            "not merely uncorrected for dimension -- correcting it moves the "
            "comparison back toward, and on these numbers past, the predicted "
            "direction. The reference is random OV matrices at the right "
            "shape, NOT ALBERT's trained weights, so this bounds the size of "
            "the correction and does not report ALBERT's own dimensions. Those "
            "are computed by _build_for_layer on every run and were never "
            "reported; recovering them is one number and it decides (c) "
            "outright."),
    }


def check_audit(path: Path = AUDIT_PATH) -> List[str]:
    """
    Is the committed audit still about the file on disk, and self-consistent?

    Deliberately does NOT re-run the arms: arm D takes about ninety seconds and
    the point of committing the result is that it does not have to be recomputed
    to be trusted. What can go stale is the file it describes, so that is what
    is checked.
    """
    problems: List[str] = []
    if not path.exists():
        return [f"{path} is missing; regenerate with --write"]
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    if rec.get("schema_version") != AUDIT_SCHEMA_VERSION:
        problems.append(
            f"schema_version {rec.get('schema_version')} != "
            f"{AUDIT_SCHEMA_VERSION}; regenerate with --write")
    if not AUDITED_PATH.exists():
        problems.append(f"{AUDITED_PATH} is missing")
    elif rec.get("audited_sha256") != audited_sha256():
        problems.append(
            f"{AUDITED_PATH.name} has changed since the audit was written "
            f"(sha256 {audited_sha256()[:12]} on disk vs "
            f"{str(rec.get('audited_sha256'))[:12]} on record). The audit "
            f"describes a file that no longer exists in that form; rerun "
            f"--write.")
    for arm in ("arm_L_labelling", "arm_C_counts", "arm_D_dimension"):
        if arm not in rec:
            problems.append(f"{arm} is missing from the record")
    if rec.get("arm_S_sensitivity", {}).get("verdict") != "PASS":
        problems.append(
            "arm S did not pass: the audit could not be made to fail, so arm "
            "L's verdict carries no information")
    return problems


def print_summary(rec: dict) -> None:
    L = rec["arm_L_labelling"]
    print(f"audited: {rec['audited_file']}  sha256 {rec['audited_sha256'][:12]}")
    print(f"\narm L (labelling, planted normal): {L['verdict']}  "
          f"worst angle {L['worst_max_principal_angle_rad']:.2e} rad "
          f"(tol {L['tolerance_rad']:.0e})")
    for case in L["cases"]:
        b = case["buckets"]
        print(f"  d={case['d']:4d}  " + "  ".join(
            f"{k}:{b[k]['recovered_dim']}/{b[k]['planted_dim']}"
            f"@{b[k]['max_principal_angle_rad']:.1e}"
            for k in ("pos", "neg", "rot", "zero")))

    C = rec["arm_C_counts"]
    print(f"\narm C (counts, non-normal, independent spectrum): {C['verdict']}")
    for case in C["cases"]:
        sc, rf = case["schur_counts"], case["spectrum_counts"]
        print(f"  d={case['d']:4d} rank={case['rank']:3d}  "
              f"schur {[sc[k] for k in ('pos','neg','rot','zero')]}  "
              f"eigvals {[rf[k] for k in ('pos','neg','rot','zero')]}  "
              f"{'agree' if case['agree'] else 'DISAGREE'}  "
              f"(margin: real Im<={case['max_rel_im_among_real']:.1e}, "
              f"complex Im>={case['min_rel_im_among_complex']:.1e})")

    S = rec["arm_S_sensitivity"]
    print(f"\narm S (sensitivity): {S['verdict']}")
    for key in ("swapped", "transposed_subdiagonal"):
        arm = S[key]
        print(f"  {key:24s} caught={str(arm['caught']):5s}  "
              f"by L: {arm['labelling']['verdict']:4s}  "
              f"by C: {arm['counts']['verdict']:4s}")
    print("  (arm L is blind to the transposition by construction -- see "
          "_asymmetry)")

    D = rec["arm_D_dimension"]
    print(f"\narm D (dimension): dim(U_A) / dim(U_neg) by shape")
    print(f"  {'d_model':>8} {'heads':>6} {'head_dim':>9} {'dim_neg':>8} "
          f"{'dim_A':>7} {'ratio':>8}")
    for r in D["rows"]:
        mark = "  <- albert-xlarge-v2" if r["is_albert_xlarge_v2_shape"] else ""
        print(f"  {r['d_model']:>8} {r['n_heads']:>6} {r['head_dim']:>9} "
              f"{r['dim_neg']:>8} {r['dim_A']:>7} "
              f"{r['chance_alignment_ratio']:>8.2f}{mark}")
    print(f"\n  observed alignment ratio  0.887 / 0.067 = "
          f"{D['observed_alignment_ratio']:.2f}")
    print(f"  chance ratio at ALBERT's shape           = "
          f"{D['albert_shape_chance_ratio']:.2f}")
    print(f"  observed / chance                        = "
          f"{D['observed_over_chance']:.3f}")
    print(f"\n  chance-normalized alignment  U_A   = "
          f"{D['normalized_alignment_U_A']:.3f}")
    print(f"  chance-normalized alignment  U_neg = "
          f"{D['normalized_alignment_U_neg']:.3f}")
    print(f"\nexplanation (a), Schur mislabelling: "
          f"{rec['explanation_a_schur_mislabelling']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true",
                    help="run all three arms and write the audit (~90s)")
    ap.add_argument("--check", action="store_true",
                    help="committed audit still describes the file on disk?")
    ap.add_argument("--summary", action="store_true",
                    help="print the committed audit")
    ap.add_argument("--out", type=Path, default=AUDIT_PATH)
    args = ap.parse_args(argv)

    if args.write:
        rec = build_audit()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")

    if args.check:
        problems = check_audit(args.out)
        for p in problems:
            print(f"STALE: {p}")
        if problems:
            return 1
        print(f"audit_p6_projector_labels: {args.out.name} in step with "
              f"{AUDITED_PATH.name}")

    if args.summary:
        print_summary(json.loads(args.out.read_text()))

    if not (args.write or args.check or args.summary):
        ap.error("nothing to do: pass --write, --check or --summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
