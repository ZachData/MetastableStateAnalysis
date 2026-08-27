"""
p6_subspace/subspace_geometry.py — the S/A channel decomposition, rebuilt.

WHAT THIS IS A REBUILD OF, AND WHY IT IS NOT A COPY

`archive/p6_subspace/subspace_build.py` partitions d_model using the real Schur
decomposition of each head's OV = W_V W_O, into an attractive real channel, a
repulsive real channel, a rotational (imaginary) channel, and a kernel. That
construction is correct -- `claims/audits/p6_projector_labels.json` establishes
it on planted structure and against an independently derived spectrum, which is
`status-6.md` item 5 settled.

Correct is not the same as reusable. `archive/README.md` rule 2: a capability
that lives in the archive is rebuilt against the particle schema, not lifted,
because every archived module keys its own bespoke structures and copying one
forward reintroduces the producer/consumer mismatches `core/artifacts.py` exists
to kill. So this module implements the same mathematics (`math-6.md` §2) with a
typed result, and one deliberate difference in what it exposes.

THE DELIBERATE DIFFERENCE: ALIGNMENT IS REPORTED RELATIVE TO CHANCE

`math-6.md` §7.2, on the phase's only substantive result:

    For a random unit vector v and a k-dimensional subspace U,
    E[||P_U v||^2] = k/d. So alignment with a subspace scales with that
    subspace's dimension, and comparing raw alignment against U_A versus
    U_neg measures dimension at least as much as content.

    "Until one of these is done, the inversion is not evidence for or
     against the hypothesis."

It is not a small correction here. The resolution order makes U_neg the
doubly-shrunk bucket -- span(U_pos) is removed from U_neg and span(U_S) from
U_A -- and at albert-xlarge-v2's shape the audit measures dim(U_A)/dim(U_neg)
= 24.9, against an observed alignment ratio of 13.2. The correction is nearly
twice the effect it would explain.

So `raw_alignment` exists but nothing in this package compares two raw
alignments. The comparable quantity is `normalized_alignment`, and the null
`p6_subspace.r2_r4_null` actually uses is stronger still: a random subspace of
MATCHED DIMENSION, which holds dimension fixed by construction rather than by
trusting the k/d identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

#: Relative eigenvalue cutoff: |lambda| <= EIG_REL_TOL * ||T||_F is kernel.
#: Relative and not absolute, which is the recurring defect this codebase keeps
#: finding (math-2b.md §2.2, and fix #3 in the archived module). INHERITED from
#: archive/p6_subspace/subspace_build.py's own default so the rebuild answers
#: the same question about the same matrix; neither placed nor calibrated here.
EIG_REL_TOL = 1e-8

#: Rank-truncation tolerance for the orthonormal bases.
#: INHERITED likewise: neither placed nor calibrated here.
SVD_REL_TOL = 1e-8


@dataclass(frozen=True)
class ChannelBasis:
    """
    One head's OV circuit, partitioned into channels.

    Columns are orthonormal within each channel. `rotation` always has an even
    number of columns: a 2x2 real Schur block is a complex-conjugate eigenvalue
    pair and its two Schur vectors span one rotation plane, so splitting them
    is meaningless.
    """
    real_pos: np.ndarray
    real_neg: np.ndarray
    rotation: np.ndarray
    kernel: np.ndarray
    d_model: int

    def dims(self) -> dict:
        return {
            "real_pos": int(self.real_pos.shape[1]),
            "real_neg": int(self.real_neg.shape[1]),
            "rotation": int(self.rotation.shape[1]),
            "kernel": int(self.kernel.shape[1]),
        }


@dataclass(frozen=True)
class LayerChannels:
    """
    One layer's channels after the cross-head union and the resolution order.

    `u_neg` and `u_a` are the two subspaces P6-R2 compares and P6-R4 projects
    onto. `dims` is carried because every statistic in this package is
    dimension-aware and a consumer that has the bases but not the dimensions
    would be one step from the defect math-6.md §7.2 names.
    """
    u_pos: np.ndarray
    u_neg: np.ndarray
    u_a: np.ndarray
    u_s: np.ndarray
    d_model: int
    n_heads: int

    def dims(self) -> dict:
        return {
            "u_pos": int(self.u_pos.shape[1]),
            "u_neg": int(self.u_neg.shape[1]),
            "u_a": int(self.u_a.shape[1]),
            "u_s": int(self.u_s.shape[1]),
            "d_model": int(self.d_model),
        }


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

def orthonormal_basis(vectors: Sequence[np.ndarray], d: int,
                      rel_tol: float = SVD_REL_TOL) -> np.ndarray:
    """
    Orthonormal basis for the span, with rank truncated relative to the leading
    singular value. Returns (d, 0) on empty input rather than raising: an empty
    channel is a real and common answer (a head with no negative real
    eigenvalue), and callers that treat it as an error end up guarding every
    call site.
    """
    if len(vectors) == 0:
        return np.zeros((d, 0))
    V = np.column_stack(vectors).astype(np.float64)
    U, s, _ = np.linalg.svd(V, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((d, 0))
    r = int(np.sum(s > rel_tol * s[0]))
    return U[:, :r]


def project_out(target: np.ndarray, remove: np.ndarray,
                rel_tol: float = SVD_REL_TOL) -> np.ndarray:
    """Orthonormal basis for span(target) intersected with span(remove)^perp."""
    d = target.shape[0]
    if target.shape[1] == 0:
        return np.zeros((d, 0))
    if remove.shape[1] == 0:
        return target.copy()
    residual = target - remove @ (remove.T @ target)
    return orthonormal_basis([residual[:, i] for i in range(residual.shape[1])],
                             d, rel_tol)


# ---------------------------------------------------------------------------
# The Schur partition
# ---------------------------------------------------------------------------

def schur_channels(ov: np.ndarray, eig_rel_tol: float = EIG_REL_TOL) -> ChannelBasis:
    """
    Partition one head's OV circuit by the real Schur decomposition.

    2x2 block with a nonzero SUBdiagonal      -> rotation (a conjugate pair)
    1x1 block, lambda >  tol                  -> real_pos (attractive)
    1x1 block, lambda < -tol                  -> real_neg (repulsive)
    otherwise                                 -> kernel

    Both cuts are relative to ||T||_F. The SUBdiagonal T[i+1, i] is what
    identifies a 2x2 block; the SUPERdiagonal T[i, i+1] is generically nonzero
    for 1x1 blocks too, so reading it instead types nearly everything as a
    rotation. That is not a hypothetical -- it is one of the two breakages
    `tools/audit_p6_projector_labels.py` plants, and the one a planted normal
    matrix cannot see.
    """
    from scipy.linalg import schur

    ov = np.asarray(ov, dtype=np.float64)
    if ov.ndim != 2 or ov.shape[0] != ov.shape[1]:
        raise ValueError(f"OV must be square; got shape {ov.shape}")
    d = ov.shape[0]

    T, Z = schur(ov, output="real")
    nrm = float(np.linalg.norm(T, "fro"))
    eps = float(np.finfo(T.dtype).eps)
    tol = max(nrm * eig_rel_tol, nrm * eps * 10.0, 1e-12)

    pos: List[np.ndarray] = []
    neg: List[np.ndarray] = []
    rot: List[np.ndarray] = []
    ker: List[np.ndarray] = []

    i = 0
    while i < d:
        if i < d - 1 and abs(T[i + 1, i]) > tol:
            a, b = float(T[i, i]), float(T[i, i + 1])
            c, dd = float(T[i + 1, i]), float(T[i + 1, i + 1])
            magnitude = float(np.sqrt(max(a * dd - b * c, 0.0)))
            target = rot if magnitude > tol else ker
            target.append(Z[:, i].copy())
            target.append(Z[:, i + 1].copy())
            i += 2
        else:
            lam = float(T[i, i])
            if lam > tol:
                pos.append(Z[:, i].copy())
            elif lam < -tol:
                neg.append(Z[:, i].copy())
            else:
                ker.append(Z[:, i].copy())
            i += 1

    return ChannelBasis(
        real_pos=orthonormal_basis(pos, d),
        real_neg=orthonormal_basis(neg, d),
        rotation=orthonormal_basis(rot, d),
        kernel=orthonormal_basis(ker, d),
        d_model=d,
    )


def layer_channels(head_ovs: Sequence[np.ndarray],
                   eig_rel_tol: float = EIG_REL_TOL) -> LayerChannels:
    """
    Cross-head union for one layer, with the resolution order applied.

    The order is `math-6.md` §2's and is load-bearing rather than incidental:

        U_A   := U_A_raw   intersect span(U_S_raw)^perp     (S wins over A)
        U_neg := U_neg_raw intersect span(U_pos)^perp       (pos wins over neg)

    Both subtractions shrink U_neg relative to U_A -- U_neg loses whatever it
    shares with U_pos, and U_A loses only whatever it shares with the union.
    That asymmetry is the mechanism behind the dimension gap the audit
    measures, so it is stated here rather than left for a reader to derive from
    two lines of code.
    """
    head_ovs = list(head_ovs)
    if not head_ovs:
        raise ValueError(
            "layer_channels needs at least one head's OV matrix; an empty layer "
            "is a wiring bug upstream, not a layer with no channels")
    per_head = [schur_channels(ov, eig_rel_tol) for ov in head_ovs]
    d = per_head[0].d_model
    if any(h.d_model != d for h in per_head):
        raise ValueError(
            f"heads disagree on d_model: {sorted({h.d_model for h in per_head})}")

    def union(attr: str) -> np.ndarray:
        cols: List[np.ndarray] = []
        for h in per_head:
            B = getattr(h, attr)
            cols.extend(B[:, j] for j in range(B.shape[1]))
        return orthonormal_basis(cols, d)

    u_pos_raw = union("real_pos")
    u_neg_raw = union("real_neg")
    u_a_raw = union("rotation")

    u_s_raw = orthonormal_basis(
        [u_pos_raw[:, j] for j in range(u_pos_raw.shape[1])]
        + [u_neg_raw[:, j] for j in range(u_neg_raw.shape[1])], d)

    u_a = project_out(u_a_raw, u_s_raw)
    u_pos = u_pos_raw
    u_neg = project_out(u_neg_raw, u_pos)
    u_s = (np.column_stack([u_pos, u_neg])
           if u_pos.shape[1] + u_neg.shape[1] > 0 else np.zeros((d, 0)))

    return LayerChannels(u_pos=u_pos, u_neg=u_neg, u_a=u_a, u_s=u_s,
                         d_model=d, n_heads=len(head_ovs))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def raw_alignment(v: np.ndarray, U: np.ndarray) -> float:
    """
    ||P_U v||^2 for a unit v. The archived statistic.

    Provided so the two can be compared and so a reader can reproduce what the
    2026-04 run reported -- NOT for comparing two subspaces of different
    dimension against each other, which is the defect. Use
    `normalized_alignment`, or the matched-dimension null in `r2_r4_null`.
    """
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        raise ValueError("alignment needs a nonzero finite vector")
    v = v / n
    if U.shape[1] == 0:
        return 0.0
    return float(np.sum((U.T @ v) ** 2))


def chance_alignment(dim_u: int, d_model: int) -> float:
    """
    E[||P_U v||^2] for a random unit v and a dim_u-dimensional U: dim_u/d_model.

    One line, given its own name, because it is the entire content of
    explanation (c) and inlining it is how it got left out the first time.
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive; got {d_model}")
    return float(dim_u) / float(d_model)


def normalized_alignment(v: np.ndarray, U: np.ndarray, d_model: int) -> float:
    """
    `raw_alignment` divided by what chance alone would give: 1.0 means a random
    direction would have done as well.

    Refuses on an empty subspace rather than returning 0/0 or a large number
    from a small one: a channel with no dimensions has no alignment to report,
    and 0.0 would read as "orthogonal" when it means "absent".
    """
    if U.shape[1] == 0:
        raise ValueError(
            "normalized_alignment on an empty subspace: chance alignment is "
            "zero and the ratio is undefined. An empty channel is a finding "
            "about the OV circuit, not a zero alignment.")
    return raw_alignment(v, U) / chance_alignment(U.shape[1], d_model)


def random_subspace(d_model: int, dim_u: int,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    A uniformly random dim_u-dimensional subspace of R^d_model, orthonormal
    columns. The matched-dimension control `dissociation.py`'s arm 3 already
    used and `EVALUABILITY.md` prescribes for P6-R1 and P6-C1.
    """
    if not 0 <= dim_u <= d_model:
        raise ValueError(
            f"cannot draw a {dim_u}-dimensional subspace of R^{d_model}")
    rng = np.random.default_rng() if rng is None else rng
    if dim_u == 0:
        return np.zeros((d_model, 0))
    Q, _ = np.linalg.qr(rng.standard_normal((d_model, dim_u)))
    return Q[:, :dim_u]


def random_orthogonal_subspace_pair(d_model: int, dim_a: int, dim_b: int,
                                    rng: Optional[np.random.Generator] = None):
    """
    Two random subspaces of the given dimensions that are MUTUALLY ORTHOGONAL.

    Needed, and not a refinement. `U_neg` and `U_A` are orthogonal by
    construction -- `layer_channels` removes span(U_S) from U_A, and U_neg is
    inside U_S -- so a null that drew them independently would compare an
    orthogonal observed pair against overlapping null pairs. That difference
    has nothing to do with operator content and it does not wash out: measured,
    it put the H0 rejection rate at 0.0875 against a nominal 0.05, in the
    anticonservative direction and invisible in any single result. Same shape as
    the P-S1 defect POPPER_PLAN.md §6d found by simulating rather than
    reasoning.

    Drawn as one orthonormal basis of dimension dim_a + dim_b, split. Uniform
    on the Stiefel manifold, so each half is marginally a uniform random
    subspace of its own dimension -- the matched-dimension property survives
    intact and only the cross term is fixed.
    """
    total = int(dim_a) + int(dim_b)
    if total > d_model:
        raise ValueError(
            f"cannot draw orthogonal subspaces of dimensions {dim_a} and "
            f"{dim_b} in R^{d_model}: they need {total} dimensions. The "
            f"observed U_neg and U_A are orthogonal and fit, so a caller "
            f"hitting this is passing dimensions that did not come from one "
            f"LayerChannels.")
    Q = random_subspace(d_model, total, rng)
    return Q[:, :dim_a], Q[:, dim_a:]


def subspace_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    An orthonormal basis for span(a + b), of dimension dim_a + dim_b.

    REFUSES if the union's numerical rank falls short of that, which happens
    when the two subspaces overlap or their dimensions together exceed
    d_model. The resolution order makes the observed channels orthogonal, so a
    caller hitting this is not passing channels from one `LayerChannels`.
    """
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"subspaces live in R^{A.shape[0]} and R^{B.shape[0]}; these must "
            f"match (frame mismatch)")
    want = A.shape[1] + B.shape[1]
    M = np.hstack([A, B])
    s = np.linalg.svd(M, compute_uv=False)
    rank = int((s > max(s[0] if s.size else 1.0, 1.0) * 1e-10).sum())
    if rank < want:
        raise ValueError(
            f"span of the two channels has rank {rank}, not their dimension "
            f"sum {want} (d_model = {A.shape[0]}). They overlap, or together "
            f"exceed d_model; either way no re-split of the union reproduces "
            f"the observed geometry.")
    return np.linalg.qr(M)[0][:, :want]


def resplit_union(union: np.ndarray, dim_a: int,
                  rng: np.random.Generator) -> tuple:
    """
    A uniformly random (dim_a, k - dim_a) split of `union`, as two bases.

    The pair spans exactly what the union spans, is orthogonal, and has the
    requested dimensions -- so everything about the pair AS A PAIR is held
    fixed and only the assignment moves. That is what
    `random_orthogonal_subspace_pair` does not do: it randomises the union and
    the assignment together, so a union that sits above chance against whatever
    the statistic reads is not reproduced in the null. See POPPER_PLAN.md 6n.
    """
    S = np.asarray(union, dtype=np.float64)
    k = S.shape[1]
    ka = int(dim_a)
    if not 0 < ka < k:
        raise ValueError(
            f"a split needs 0 < dim_a < k; got dim_a={ka}, k={k}. A union "
            f"assigned entirely to one channel has no second channel.")
    R = np.linalg.qr(rng.normal(size=(k, k)))[0]
    Z = S @ R
    return Z[:, :ka], Z[:, ka:]
