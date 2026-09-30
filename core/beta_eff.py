"""
beta_eff.py — Effective inverse temperature of the attention softmax
(frames item 6).

Why this module exists
----------------------
`v_alignment.estimate_effective_beta` regresses `log A_ij` on `<x_i, x_j>`
over the pairs selected by `np.triu_indices(n, k=1)` — that is, pairs with
query index BELOW key index.

Causal attention masks exactly those entries. `A[i, j] = 0` for `j > i`, the
clip at 1e-12 turns every one of them into `log(1e-12) = -27.63`, and the
regression fits a varying x against a constant y. On a synthetic softmax
with a known beta of 6.0 the estimator returns **-1.8e-14**. It has been
reporting approximately zero for every head, on every model, independent of
the data.

Three further problems, each of which survives the indexing fix:

1. **Row-varying normaliser.** `log A_ij = beta * s_ij - log Z_i`. The
   denominator is per query row, and an intercept cannot absorb a per-row
   term. Pooling rows biases the slope, and because later rows attend over
   more keys, `log Z_i` correlates with position — and therefore with
   offset. Corrected by within-row demeaning (a fixed-effects estimator).
   On the same synthetic data: pooled 5.937, row-demeaned 6.000.

2. **Wrong frame.** `<x_i, x_j>` on L2-normalized residuals is not
   `q_i . k_j`. The head reads LN1(x), then projects. The Gram matrix is now
   an argument rather than something this function computes, so the frame is
   the caller's explicit, recorded choice (core/frames.py).

3. **Rotary and scale.** On Pythia the logit carries `R(Delta)`, so offset
   structure loads onto the slope unless Delta is controlled. And the model
   divides logits by `sqrt(head_size)`, which differs across architectures
   (64 on gpt2-large, 128 on pythia-1.4b) — so an uncorrected beta is not
   comparable between them even once everything else is right.

See DESIGN_pythia_frames.md item 6.
"""

from __future__ import annotations

import numpy as np


MIN_PAIRS = 6
LOG_FLOOR = 1e-12
#: PLACED. Smallest singular-value ratio of the column-normalised design
#: (similarity, offset tail) accepted by `estimate_beta_offset_fe`.
COLLINEAR_SV = 1e-6


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def causal_pairs(indices, include_diagonal: bool = False) -> tuple:
    """
    Query/key index arrays for pairs the softmax actually sees.

    `indices` are positions in the original sequence (e.g. a cluster's
    members); order is not assumed. A pair is kept when key <= query in
    ORIGINAL position, not in submatrix order — sorting a cluster's indices
    would otherwise silently change which pairs are causal.

    Returns (rows, cols) as indices INTO `indices`, so they can address a
    submatrix directly.
    """
    idx = np.asarray(indices, dtype=np.int64)
    n = idx.size
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pos_q, pos_k = idx[ii], idx[jj]
    keep = pos_k < pos_q if not include_diagonal else pos_k <= pos_q
    return ii[keep], jj[keep]


def structural_zero_fraction(A) -> float:
    """
    Fraction of a submatrix that is exactly zero.

    Reported so that a caller feeding masked entries into the regression sees
    it as a number rather than as a slope of zero.
    """
    a = np.asarray(A, dtype=np.float64)
    return float(np.mean(a == 0.0)) if a.size else float("nan")


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def _within_row_demean(values, rows) -> np.ndarray:
    """Subtract each query row's mean. The fixed-effects transform."""
    # One pass (bincount), not a mask per row: that was O(rows x pairs), and
    # at 2048 tokens it made one layer's fit take minutes (`status-1d.md`).
    v = np.asarray(values, dtype=np.float64)
    _, g = np.unique(rows, return_inverse=True)
    return _group_demean(v, g, int(g.max()) + 1 if g.size else 0)


def estimate_beta_from_gram(
    attn_head,
    gram,
    indices,
    offsets=None,
    attn_scale: float | None = None,
    row_fixed_effects: bool = True,
    control_offset: bool = True,
    max_offset: int | None = None,
) -> dict:
    """
    Effective beta for one head.

    Parameters
    ----------
    attn_head : (n_seq, n_seq) post-softmax attention, [query, key]
    gram      : (n_seq, n_seq) pairwise similarity IN THE READER'S FRAME.
                Not computed here on purpose — see module docstring. Build it
                with core.frames.frame_gram and record the FrameSpec.
    indices   : positions to restrict to (a cluster, or all tokens)
    offsets   : (n_seq, n_seq) key - query, or None to derive from `indices`
    attn_scale: 1/sqrt(head_size). When given, the returned beta is divided
                by it, making the value comparable across architectures with
                different head widths. When None the raw slope is returned
                and `scale_applied` is False.
    max_offset: keep only pairs at most this many positions apart (None:
                all), e.g. to fit a long prompt on a short prompt's offsets.

    Returns dict with beta, beta_raw, n_pairs, r2, offset_coeff,
    structural_zero_fraction, scale_applied, note.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 3:
        return _empty("cluster too small (<3) for regression")

    A = np.asarray(attn_head, dtype=np.float64)[np.ix_(idx, idx)]
    G = np.asarray(gram, dtype=np.float64)[np.ix_(idx, idx)]

    rows, cols = causal_pairs(idx)
    if rows.size < MIN_PAIRS:
        return _empty(f"only {rows.size} causal pairs (<{MIN_PAIRS})")

    a = A[rows, cols]
    s = G[rows, cols]
    # Two different diagnostics, both worth having:
    #   submatrix_zero_frac — how much of the submatrix causal masking removed.
    #     A regression run over the whole submatrix is fitting mostly this.
    #   zero_among_causal   — zeros among the pairs actually selected. Should
    #     be ~0; anything else means masked entries reached the fit.
    submatrix_zero_frac = structural_zero_fraction(A)
    zero_among_causal = float(np.mean(a == 0.0))

    keep = a > 0.0
    if keep.sum() < MIN_PAIRS:
        return _empty(
            f"only {int(keep.sum())} non-zero attention entries among causal "
            f"pairs; the softmax gives these pairs no mass"
        )
    rows, cols, a, s = rows[keep], cols[keep], a[keep], s[keep]
    y = np.log(np.clip(a, LOG_FLOOR, None))

    if offsets is None:
        d = (idx[cols] - idx[rows]).astype(np.float64)
    else:
        d = np.asarray(offsets, dtype=np.float64)[np.ix_(idx, idx)][rows, cols]
    if max_offset is not None:
        m = np.abs(d) <= max_offset
        if m.sum() < MIN_PAIRS:
            return _empty(f"only {int(m.sum())} pairs within max_offset {max_offset}")
        rows, y, s, d = rows[m], y[m], s[m], d[m]

    # Design matrix. Row fixed effects are applied by demeaning rather than by
    # dummy columns: a cluster can have hundreds of rows, and the demeaned
    # form is numerically identical with two columns instead of hundreds.
    cols_list = [s]
    if control_offset and np.std(d) > 1e-9:
        cols_list.append(d)
    Xd = np.column_stack(cols_list)

    if row_fixed_effects:
        y_f = _within_row_demean(y, rows)
        Xd = np.column_stack([_within_row_demean(Xd[:, k], rows)
                              for k in range(Xd.shape[1])])
        # Each row loses one degree of freedom to its own mean.
        dof_lost = int(np.unique(rows).size)
    else:
        y_f = y - y.mean()
        Xd = Xd - Xd.mean(axis=0, keepdims=True)
        dof_lost = 1

    if np.std(Xd[:, 0]) < 1e-9:
        return _empty("similarity has no variance after demeaning")
    if y_f.size - dof_lost <= Xd.shape[1]:
        return _empty("too few effective degrees of freedom after fixed effects")

    coef, *_ = np.linalg.lstsq(Xd, y_f, rcond=None)
    fit = Xd @ coef
    ss_tot = float(np.sum(y_f ** 2))
    r2 = 1.0 - float(np.sum((y_f - fit) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    beta_raw = float(coef[0])
    scaled = attn_scale is not None and attn_scale > 0
    return {
        "beta": beta_raw / attn_scale if scaled else beta_raw,
        "beta_raw": beta_raw,
        "offset_coeff": float(coef[1]) if Xd.shape[1] > 1 else None,
        "n_pairs": int(y_f.size),
        "r2": r2,
        "structural_zero_fraction": submatrix_zero_frac,
        "zero_among_causal_pairs": zero_among_causal,
        "scale_applied": bool(scaled),
        "row_fixed_effects": bool(row_fixed_effects),
        "note": "",
    }


def _group_demean(values, groups, n_groups) -> np.ndarray:
    """Subtract each group's mean (groups are 0..n_groups-1)."""
    cnt = np.bincount(groups, minlength=n_groups)
    s = np.bincount(groups, weights=values, minlength=n_groups)
    return values - (s / np.maximum(cnt, 1))[groups]


def _two_way_demean(v, rows, bins, n_rows, n_bins, tol=1e-10, max_iter=5000) -> np.ndarray:
    """
    Project out row and offset-bin fixed effects by alternating projections
    (the method behind reghdfe). Converges to the residual of an OLS on both
    sets of dummies; refuses rather than return a half-converged answer.
    """
    out = np.asarray(v, dtype=np.float64).copy()
    scale = max(float(np.abs(out).max()), 1e-300)
    for _ in range(max_iter):
        prev = out
        out = _group_demean(_group_demean(out, rows, n_rows), bins, n_bins)
        if float(np.abs(out - prev).max()) <= tol * scale:
            return out
    raise RuntimeError(f"two-way demeaning did not converge in {max_iter} iterations")


def _two_way_demean_exact(v, rows, bins, n_rows, n_bins) -> np.ndarray:
    """
    The same projection as `_two_way_demean`, solved directly. On a causal
    design (row i sees offsets 1..i) alternating projections converge slowly:
    at n ~ 2048 one head took minutes and risked the iteration cap (the long
    prompts, `status-1d.md`). Here the row effects are eliminated in closed
    form and the offset effects solve the ``n_bins``-square Schur complement,
    with one bin pinned to 0 (the constant is shared by both sets of dummies).
    ``v`` may be (m,) or (m, k); every column is projected.
    """
    from scipy.linalg import cho_factor, cho_solve, LinAlgError
    V = np.asarray(v, dtype=np.float64)
    one = V.ndim == 1
    V = V[:, None] if one else V
    n_r = np.bincount(rows, minlength=n_rows).astype(np.float64)
    n_b = np.bincount(bins, minlength=n_bins).astype(np.float64)
    C = np.bincount(rows * n_bins + bins, minlength=n_rows * n_bins).reshape(n_rows, n_bins)
    C = C.astype(np.float64)
    inv_r = 1.0 / np.maximum(n_r, 1.0)
    sr = np.stack([np.bincount(rows, weights=V[:, j], minlength=n_rows) for j in range(V.shape[1])], 1)
    sb = np.stack([np.bincount(bins, weights=V[:, j], minlength=n_bins) for j in range(V.shape[1])], 1)
    S = np.diag(n_b) - C.T @ (inv_r[:, None] * C)
    rhs = sb - C.T @ (inv_r[:, None] * sr)
    theta_b = np.zeros((n_bins, V.shape[1]))
    if n_bins > 1:
        try:
            theta_b[:-1] = cho_solve(cho_factor(S[:-1, :-1]), rhs[:-1])
        except LinAlgError:              # a disconnected design: fall back to least squares
            theta_b = np.linalg.lstsq(S, rhs, rcond=None)[0]
    theta_r = inv_r[:, None] * (sr - C @ theta_b)
    out = V - theta_r[rows] - theta_b[bins]
    return out[:, 0] if one else out


def estimate_beta_offset_fe(
    attn_head,
    gram,
    indices,
    positions=None,
    offset_window: int | None = None,
    max_offset: int | None = None,
) -> dict:
    """
    ``beta_raw`` with per-offset fixed effects instead of a linear offset term.

    Model: ``log A_ij = beta * s_ij + a_i + g(i - j) + e_ij``, with ``a_i``
    the row's normaliser (as in `estimate_beta_from_gram`) and ``g`` free:
    one dummy per offset ``1 .. offset_window - 1``, then one pooled bin with
    a linear slope inside it for offsets ``>= offset_window``. ``None`` gives
    every offset its own dummy. A recency head's log-attention is not linear
    in offset; a linear control leaves the curvature on the slope whenever
    similarity also varies with offset.

    ``indices`` select rows/columns of ``attn_head`` and ``gram``;
    ``positions`` (same length as the matrices) maps them to sequence
    positions, which set causality and offset (default: the index itself).
    ``max_offset`` keeps only pairs at most that many positions apart.

    Returns beta_raw; r2, within-row R² of the whole model (comparable to
    `estimate_beta_from_gram`'s); partial_r2, the share of what the row and
    offset effects leave that similarity explains; n_pairs; n_offset_bins;
    design_sv_ratio, the smallest over largest singular value of the
    column-normalised (similarity, tail) design (NaN with no tail column).
    """
    idx = np.asarray(indices, dtype=np.int64)
    pos = idx if positions is None else np.asarray(positions, dtype=np.int64)[idx]
    empty = {"beta_raw": float("nan"), "r2": float("nan"), "partial_r2": float("nan"),
             "n_pairs": 0, "n_offset_bins": 0, "design_sv_ratio": float("nan")}
    if idx.size < 3:
        return {**empty, "note": "too few tokens (<3)"}
    A = np.asarray(attn_head, dtype=np.float64)[np.ix_(idx, idx)]
    G = np.asarray(gram, dtype=np.float64)[np.ix_(idx, idx)]
    rows, cols = causal_pairs(pos)
    if max_offset is not None:
        near = (pos[rows] - pos[cols]) <= max_offset
        rows, cols = rows[near], cols[near]
    a = A[rows, cols]
    keep = a > 0.0
    if keep.sum() < MIN_PAIRS:
        return {**empty, "note": "too few non-zero causal pairs"}
    rows, cols, a = rows[keep], cols[keep], a[keep]
    y = np.log(np.clip(a, LOG_FLOOR, None))
    s = G[rows, cols]
    d = (pos[rows] - pos[cols]).astype(np.int64)          # >= 1
    W = int(d.max()) + 1 if offset_window is None else int(offset_window)
    _, bins = np.unique(np.minimum(d, W), return_inverse=True)
    _, rws = np.unique(rows, return_inverse=True)
    n_r, n_b = int(rws.max()) + 1, int(bins.max()) + 1

    tail = np.where(d >= W, d, 0).astype(np.float64)
    y_t, s_t, t_t = _two_way_demean_exact(np.column_stack([y, s, tail]), rws, bins, n_r, n_b).T
    if np.std(s_t) < 1e-9:
        return {**empty, "note": "similarity has no variance after fixed effects"}
    X = s_t[:, None]
    if np.std(tail) > 1e-9:
        if np.std(t_t) > 1e-9:
            X = np.column_stack([s_t, t_t])
    # Similarity collinear with the tail slope leaves beta unidentified;
    # lstsq would return the minimum-norm split instead of refusing.
    sv_ratio = float("nan")
    if X.shape[1] > 1:
        sv = np.linalg.svd(X / np.linalg.norm(X, axis=0), compute_uv=False)
        sv_ratio = float(sv[-1] / sv[0])
        if sv_ratio < COLLINEAR_SV:
            return {**empty, "design_sv_ratio": sv_ratio,
                    "note": "similarity collinear with the offset tail; beta unidentified"}
    coef, *_ = np.linalg.lstsq(X, y_t, rcond=None)
    ssr_full = float(np.sum((y_t - X @ coef) ** 2))
    if X.shape[1] > 1:
        c0, *_ = np.linalg.lstsq(X[:, 1:], y_t, rcond=None)
        ssr_red = float(np.sum((y_t - X[:, 1:] @ c0) ** 2))
    else:
        ssr_red = float(np.sum(y_t ** 2))
    ss_row = float(np.sum(_within_row_demean(y, rows) ** 2))
    return {"beta_raw": float(coef[0]),
            "r2": 1.0 - ssr_full / ss_row if ss_row > 0 else float("nan"),
            "partial_r2": 1.0 - ssr_full / ssr_red if ssr_red > 0 else float("nan"),
            "n_pairs": int(y.size), "n_offset_bins": n_b,
            "design_sv_ratio": sv_ratio, "note": ""}


def _empty(note: str) -> dict:
    return {"beta": float("nan"), "beta_raw": float("nan"), "offset_coeff": None,
            "n_pairs": 0, "r2": float("nan"),
            "structural_zero_fraction": float("nan"),
            "zero_among_causal_pairs": float("nan"),
            "scale_applied": False, "row_fixed_effects": False, "note": note}


def estimate_beta_all_heads(attentions_layer, gram, indices, **kw) -> dict:
    """
    Per-head beta plus cluster summaries, matching the old return keys so
    existing report code keeps working.

    Adds `frame_required`: a standing reminder in the record itself that the
    number is only meaningful relative to the Gram matrix's frame, which this
    function cannot see and therefore cannot record. The caller attaches the
    FrameSpec.
    """
    A = np.asarray(attentions_layer, dtype=np.float64)
    per = [estimate_beta_from_gram(A[h], gram, indices, **kw)
           for h in range(A.shape[0])]
    betas = np.array([p["beta"] for p in per], dtype=np.float64)
    valid = betas[~np.isnan(betas)]
    return {
        "per_head": per,
        "per_head_beta": [None if np.isnan(b) else round(float(b), 3) for b in betas],
        "cluster_mean_beta": float(valid.mean()) if valid.size else float("nan"),
        "cluster_median_beta": float(np.median(valid)) if valid.size else float("nan"),
        "n_valid_heads": int(valid.size),
        "frame_required": True,
    }


# ---------------------------------------------------------------------------
# The legacy estimator, for the diff
# ---------------------------------------------------------------------------

def legacy_beta(attn_head, activations, indices) -> float:
    """
    The shipping computation, preserved verbatim so the correction can be
    measured against what actually ran rather than a reconstruction.

    Regresses on `triu_indices(k=1)` — the causally masked half.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < 3:
        return float("nan")
    X = np.asarray(activations, dtype=np.float64)[idx]
    G = X @ X.T
    iu = np.triu_indices(idx.size, k=1)
    ips = G[iu]
    A = np.asarray(attn_head, dtype=np.float64)[np.ix_(idx, idx)]
    log_A = np.log(np.clip(A, LOG_FLOOR, None))[iu]
    if np.std(ips) < 1e-6:
        return float("nan")
    return float(np.polyfit(ips, log_A, 1)[0])


def beta_summary_lines(result: dict) -> list:
    per = result.get("per_head", [])
    lines = [
        "Effective beta:",
        f"  heads valid   {result.get('n_valid_heads', 0)} of {len(per)}",
        f"  mean / median {result.get('cluster_mean_beta', float('nan')):.3f}"
        f" / {result.get('cluster_median_beta', float('nan')):.3f}",
    ]
    if per:
        p0 = per[0]
        lines.append(
            f"  pairs/head    {p0['n_pairs']} causal, "
            f"{p0['structural_zero_fraction']:.2f} of submatrix structurally zero"
        )
        lines.append(
            f"  scale         {'divided out' if p0['scale_applied'] else 'NOT applied — not cross-model comparable'}"
        )
        if p0.get("offset_coeff") is not None:
            lines.append(f"  offset coeff  {p0['offset_coeff']:+.4f} per position")
    return lines
