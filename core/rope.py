"""
core/rope.py — Rotary position embeddings in the attention bilinear
(frames item 5).

Why this module exists
----------------------
Every weight-space QK quantity in this project computes

    logit(i, j) = x_i^T (W_Q W_K^T) x_j

That identity holds for GPT-2. It does not hold for Pythia. GPT-NeoX
applies a rotary position embedding to the first `rotary_ndims` dims of
each head's query and key *after* projection, so the true bilinear is

    logit(i, j) = q_i^T R(j - i) k_j ,   q_i = W_Q^T LN1(x_i)
                                         k_j = W_K^T LN1(x_j)

with R(Δ) block-diagonal and orthogonal. W_Q W_K^T is M(Δ=0) only. This
is the same class of error as the distance measurement: correct code for
the wrong object, invisible because nothing recorded which object it was.

Layout: GPT-NeoX uses the HALF-SPLIT convention, not interleaved. Within
the rotary block of width `n_rot`, dim t pairs with dim t + n_rot/2 (this
is what HF's `rotate_half` implements). Getting this wrong produces
plausible, wrong numbers — pairing t with t+1 still yields an orthogonal
matrix with the right Frobenius norm.

    rotary_ndims = int(head_size * config.rotary_pct)      # 0.25 on Pythia
    inv_freq[t]  = base ** (-2t / rotary_ndims)            # base = 10000
    angle_t(m)   = m * inv_freq[t]

Dims from `rotary_ndims` to `head_size` pass through unrotated.

Split pure/extraction, following core/pythia_weights.py and
core/ln_frame.py: everything below the extraction section is pure numpy
and oracle-testable torch-free. `rope_config_from_model` is duck-typed on
the HF config object and never imports torch.

Cost discipline
---------------
Two paths, deliberately:

  * Logits — project into head space and rotate the vectors, exactly as
    the model does. O(n·d_model·d_head + n²·d_head). Never materialize
    M(Δ).
  * S/A fractions — closed form via trace identities on d_head × d_head
    operands (see qk_sa_fractions_at_offset). Materializing M(Δ) would be
    (2048, 2048) per head per offset, which is what makes the naive
    version of this analysis unaffordable rather than merely wrong.

See DESIGN_pythia_frames.md, item 5.
"""

from __future__ import annotations

import numpy as np

from core.metrics import _as_numpy


DEFAULT_ROPE_BASE: float = 10000.0

#: Model-name substrings whose attention bilinear carries rotary. Used to
#: gate every rotary branch so GPT-2 code paths stay bit-identical.
_ROPE_MODEL_MARKERS = ("pythia", "neox", "gpt-neox", "gptneox")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def model_uses_rope(model_or_name) -> bool:
    """
    True when the attention bilinear carries a rotary factor.

    Accepts a model object or a model-name string. Duck-typed: a config
    exposing `rotary_pct` or `rotary_emb_base` is authoritative; otherwise
    fall back to the name. Callers gate on this so no GPT-2 result moves.
    """
    if isinstance(model_or_name, str):
        low = model_or_name.lower()
        return any(m in low for m in _ROPE_MODEL_MARKERS)
    cfg = getattr(model_or_name, "config", None)
    if cfg is not None and (
        hasattr(cfg, "rotary_pct") or hasattr(cfg, "rotary_emb_base")
    ):
        return True
    name = getattr(getattr(model_or_name, "config", None), "_name_or_path", "")
    return any(m in str(name).lower() for m in _ROPE_MODEL_MARKERS)


def rope_config_from_model(model) -> dict:
    """
    Extract the rotary geometry from a GPT-NeoX/Pythia model.

    Never assume rotary_pct == 1.0. Pythia uses 0.25, so three quarters of
    every head passes through unrotated; assuming full rotary silently
    changes every downstream number.

    Returns
    -------
    dict with head_size, rotary_ndims, base, n_heads, d_model, scale
    where `scale` is the 1/sqrt(head_size) applied to logits before the
    softmax. head_size differs across architectures (64 on gpt2-large,
    128 on pythia-1.4b), so any cross-model comparison of a
    logit-magnitude quantity must divide it out.
    """
    cfg = getattr(model, "config", model)
    d_model = int(getattr(cfg, "hidden_size"))
    n_heads = int(getattr(cfg, "num_attention_heads"))
    head_size = d_model // n_heads
    rotary_pct = float(getattr(cfg, "rotary_pct", 1.0))
    base = float(getattr(cfg, "rotary_emb_base", DEFAULT_ROPE_BASE))

    rotary_ndims = int(head_size * rotary_pct)
    if rotary_ndims % 2 != 0:
        raise ValueError(
            f"rope_config_from_model: rotary_ndims={rotary_ndims} is odd "
            f"(head_size={head_size}, rotary_pct={rotary_pct}). The "
            f"half-split layout requires an even width."
        )
    return dict(
        d_model=d_model,
        n_heads=n_heads,
        head_size=head_size,
        rotary_ndims=rotary_ndims,
        rotary_pct=rotary_pct,
        base=base,
        scale=1.0 / np.sqrt(head_size),
    )


# ---------------------------------------------------------------------------
# Pure: frequencies and angles
# ---------------------------------------------------------------------------

def rope_frequencies(rotary_ndims: int, base: float = DEFAULT_ROPE_BASE) -> np.ndarray:
    """
    inv_freq[t] = base ** (-2t / rotary_ndims), t = 0 .. rotary_ndims/2 - 1.

    Returns (rotary_ndims // 2,) float64. Matches HF's
    `1.0 / (base ** (arange(0, dim, 2) / dim))`.
    """
    if rotary_ndims <= 0:
        return np.zeros(0, dtype=np.float64)
    if rotary_ndims % 2 != 0:
        raise ValueError(f"rope_frequencies: rotary_ndims must be even, got {rotary_ndims}")
    idx = np.arange(0, rotary_ndims, 2, dtype=np.float64)
    return 1.0 / (float(base) ** (idx / float(rotary_ndims)))


def rope_angles(
    positions,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
) -> np.ndarray:
    """
    Angles θ[m, t] = position[m] * inv_freq[t]. Returns (n, rotary_ndims//2).

    `positions` may be any integer or float offsets, including negative —
    relative offsets Δ = j - i are the natural argument for the S/A work.
    """
    pos = np.atleast_1d(_as_numpy(positions)).astype(np.float64, copy=False)
    return pos[:, None] * rope_frequencies(rotary_ndims, base)[None, :]


# ---------------------------------------------------------------------------
# Pure: the rotation applied to vectors
# ---------------------------------------------------------------------------

def apply_rope(
    vecs,
    positions,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
) -> np.ndarray:
    """
    Apply GPT-NeoX rotary to per-head vectors. This is the forward map the
    model actually runs; the fidelity oracle checks it against a hooked
    forward pass.

        y[:, t]        = x[:, t]   * cos - x[:, t+h] * sin
        y[:, t+h]      = x[:, t+h] * cos + x[:, t]   * sin        t < h
        y[:, n_rot:]   = x[:, n_rot:]                             (pass-through)

    with h = rotary_ndims // 2 and angles from `positions`. Equivalent to
    HF's `x*cos + rotate_half(x)*sin` where
    rotate_half(x) = cat(-x[..., h:], x[..., :h]).

    Parameters
    ----------
    vecs      : (n, head_size) — per-head q or k, already projected
    positions : (n,) — absolute positions, or relative offsets

    Returns (n, head_size) float64.
    """
    X = _as_numpy(vecs).astype(np.float64, copy=False)
    if X.ndim == 1:
        X = X[None, :]
    n, head_size = X.shape
    if rotary_ndims > head_size:
        raise ValueError(
            f"apply_rope: rotary_ndims={rotary_ndims} exceeds head_size={head_size}"
        )
    pos = np.atleast_1d(_as_numpy(positions)).astype(np.float64, copy=False)
    if pos.shape[0] != n:
        raise ValueError(
            f"apply_rope: positions length {pos.shape[0]} != n_vectors {n}"
        )
    if rotary_ndims == 0:
        return X.copy()

    h = rotary_ndims // 2
    theta = pos[:, None] * rope_frequencies(rotary_ndims, base)[None, :]  # (n, h)
    cos, sin = np.cos(theta), np.sin(theta)

    x1 = X[:, :h]
    x2 = X[:, h:rotary_ndims]
    out = np.empty_like(X)
    out[:, :h] = x1 * cos - x2 * sin
    out[:, h:rotary_ndims] = x2 * cos + x1 * sin
    out[:, rotary_ndims:] = X[:, rotary_ndims:]
    return out


def rope_rotation(
    delta,
    head_size: int,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
) -> np.ndarray:
    """
    The (head_size, head_size) orthogonal matrix R(Δ) such that

        ⟨apply_rope(q, m), apply_rope(k, n)⟩ == q^T R(n - m) k

    i.e. for causal attention, logit(i, j) = q_i^T R(j - i) k_j. Note
    Δ < 0 for the only pairs with non-trivial post-softmax weight.

    Block-diagonal: a standard 2×2 rotation on each plane (e_t, e_{t+h}),
    identity on the pass-through dims. Materializing this is cheap
    (head_size ≤ 128); materializing the d_model-space M(Δ) is not, which
    is why qk_sa_fractions_at_offset takes R and never forms M.
    """
    R = np.eye(head_size, dtype=np.float64)
    if rotary_ndims == 0:
        return R
    h = rotary_ndims // 2
    theta = float(delta) * rope_frequencies(rotary_ndims, base)
    c, s = np.cos(theta), np.sin(theta)
    t = np.arange(h)
    R[t, t] = c
    R[t, t + h] = -s
    R[t + h, t] = s
    R[t + h, t + h] = c
    return R


# ---------------------------------------------------------------------------
# Pure: symmetric / antisymmetric structure
# ---------------------------------------------------------------------------

def rope_sa_fractions(
    delta,
    head_size: int,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
) -> dict:
    """
    Closed-form S/A split of R(Δ) itself.

    On each rotary plane R(Δ) decomposes exactly as cos(Δω_t)·I +
    sin(Δω_t)·J, so

        ||A||_F^2 = 2 Σ_t sin²(Δ ω_t)
        ||R||_F^2 = head_size                      (R is orthogonal)
        a_frac    = 2 Σ_t sin²(Δ ω_t) / head_size

    a_frac(0) == 0, rising with |Δ|. This is why P6-I2 needs a new null
    model: rotary supplies offset-dependent antisymmetry *by
    construction*, so "a_frac is elevated for induction pairs" is not
    evidence of anything until it is measured against this baseline. The
    live question is whether the content bilinear W_Q W_K^T carries
    antisymmetry beyond what rotary contributes at the same offsets.

    Returns dict(s_frac, a_frac, a_norm2, norm2).
    """
    if rotary_ndims == 0:
        return dict(s_frac=1.0, a_frac=0.0, a_norm2=0.0, norm2=float(head_size))
    theta = float(delta) * rope_frequencies(rotary_ndims, base)
    a_norm2 = float(2.0 * np.sum(np.sin(theta) ** 2))
    norm2 = float(head_size)
    return dict(
        s_frac=(norm2 - a_norm2) / norm2,
        a_frac=a_norm2 / norm2,
        a_norm2=a_norm2,
        norm2=norm2,
    )


def qk_sa_fractions_at_offset(WQ, WK, R=None) -> dict:
    """
    Exact S/A fractions of M(Δ) = W_Q R(Δ) W_K^T without forming M.

    With G_Q = W_Q^T W_Q, G_K = W_K^T W_K, C = W_K^T W_Q (all d_head ×
    d_head):

        ||M||_F^2 = tr(R^T G_Q R G_K)
        tr(M^2)   = tr(R C R C)
        ||S||^2   = (||M||^2 + tr(M^2)) / 2
        ||A||^2   = (||M||^2 - tr(M^2)) / 2

    O(d_model·d_head²) once for the Grams, then O(d_head³) per offset —
    versus O(d_model²) storage per head per offset for the naive route.

    Parameters
    ----------
    WQ, WK : (d_model, d_head), canonical orientation as produced by
             weights.extract_qk_per_head. Never transpose downstream.
    R      : (d_head, d_head) from rope_rotation, or None for identity
             (Δ = 0, or a non-rotary model — in which case this reduces
             exactly to qk_decompose.decompose_qk_matrix's fractions).

    Returns dict(s_frac, a_frac, norm2, s_norm2, a_norm2).
    """
    Q = _as_numpy(WQ).astype(np.float64, copy=False)
    K = _as_numpy(WK).astype(np.float64, copy=False)
    if Q.ndim != 2 or K.ndim != 2 or Q.shape != K.shape:
        raise ValueError(
            f"qk_sa_fractions_at_offset: WQ/WK must be 2D and same shape; "
            f"got {Q.shape} and {K.shape}"
        )
    if Q.shape[0] < Q.shape[1]:
        raise ValueError(
            f"qk_sa_fractions_at_offset: expected canonical (d_model, d_head) "
            f"with d_model >= d_head, got {Q.shape}. Transpose upstream, not here."
        )
    d_head = Q.shape[1]
    Rm = np.eye(d_head) if R is None else _as_numpy(R).astype(np.float64, copy=False)
    if Rm.shape != (d_head, d_head):
        raise ValueError(
            f"qk_sa_fractions_at_offset: R must be (d_head, d_head)=({d_head},{d_head}), "
            f"got {Rm.shape}"
        )

    G_Q = Q.T @ Q
    G_K = K.T @ K
    C = K.T @ Q

    norm2 = float(np.trace(Rm.T @ G_Q @ Rm @ G_K))
    tr_M2 = float(np.trace(Rm @ C @ Rm @ C))

    s_norm2 = (norm2 + tr_M2) / 2.0
    a_norm2 = (norm2 - tr_M2) / 2.0
    denom = max(norm2, 1e-24)
    return dict(
        s_frac=s_norm2 / denom,
        a_frac=a_norm2 / denom,
        norm2=norm2,
        s_norm2=s_norm2,
        a_norm2=a_norm2,
    )


def qk_matrix_at_offset(WQ, WK, R=None) -> np.ndarray:
    """
    Explicit M(Δ) = W_Q R(Δ) W_K^T, (d_model, d_model).

    Reference implementation for tests and small models only. Production
    code paths use qk_sa_fractions_at_offset (fractions) or
    qk_logits_with_rope (logits) and never materialize this.
    """
    Q = _as_numpy(WQ).astype(np.float64, copy=False)
    K = _as_numpy(WK).astype(np.float64, copy=False)
    Rm = np.eye(Q.shape[1]) if R is None else _as_numpy(R).astype(np.float64, copy=False)
    return Q @ Rm @ K.T


# ---------------------------------------------------------------------------
# Pure: the logit path
# ---------------------------------------------------------------------------

def qk_logits_with_rope(
    X,
    WQ,
    WK,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
    positions=None,
    scale: float | None = None,
    bq=None,
    bk=None,
) -> np.ndarray:
    """
    Per-head pre-softmax logits, rotary included.

        q_i = W_Q^T x_i + b_q ;  k_j = W_K^T x_j + b_k
        logits[i, j] = ⟨rope(q_i, i), rope(k_j, j)⟩ * scale

    Frame warning: X must already be in the frame the head reads —
    LN1(x) for Pythia, via core.ln_frame / core.frame_card. Passing raw or
    L2-normalized activations here is the same category of error this
    module exists to fix, and it will not raise.

    Bias warning: `bq`/`bk` default to None, matching the weight-only
    convention the repo currently uses everywhere. That convention is not
    free. Expanding the bilinear,

        logit = x_i^T W_Q R W_K^T x_j          (the only term weight-only keeps)
              + b_q^T R W_K^T x_j              (per-key, query-independent)
              + x_i^T W_Q R b_k                (per-query, key-independent)
              + b_q^T R b_k                    (constant)

    The second term is a per-key logit offset — i.e. a learned bias toward
    particular keys regardless of what is querying them, which is exactly
    the shape of attention-sink behaviour (policy P1). Omitting it is a
    first-order error, not a rounding one, and it is NOT rotary-specific:
    GPT-2's c_attn carries a bias too, so the frozen reference inherits it.
    Pass the biases whenever they are available.

    Parameters
    ----------
    X         : (n, d_model) activations IN THE READER'S FRAME
    WQ, WK    : (d_model, d_head) canonical orientation
    bq, bk    : (d_head,) per-head query/key biases, or None
    positions : (n,) absolute positions; defaults to arange(n)
    scale     : 1/sqrt(head_size) if the caller wants the model's actual
                pre-softmax value; None leaves logits unscaled. Cross-model
                comparisons must scale.

    Returns (n, n) float64, logits[i, j] = query i attending to key j.
    """
    Xa = _as_numpy(X).astype(np.float64, copy=False)
    Q = Xa @ _as_numpy(WQ).astype(np.float64, copy=False)   # (n, d_head)
    K = Xa @ _as_numpy(WK).astype(np.float64, copy=False)
    if bq is not None:
        Q = Q + _as_numpy(bq).astype(np.float64, copy=False).reshape(1, -1)
    if bk is not None:
        K = K + _as_numpy(bk).astype(np.float64, copy=False).reshape(1, -1)
    n = Xa.shape[0]
    pos = np.arange(n, dtype=np.float64) if positions is None else \
        np.atleast_1d(_as_numpy(positions)).astype(np.float64, copy=False)

    if rotary_ndims > 0:
        Q = apply_rope(Q, pos, rotary_ndims, base)
        K = apply_rope(K, pos, rotary_ndims, base)

    out = Q @ K.T
    if scale is not None:
        out = out * float(scale)
    return out


def qk_prediction_fidelity(predicted, actual, mask=None) -> dict:
    """
    How well a weight-space logit prediction matches the real thing.

    One number per head that says whether the weight-space picture is
    trustworthy. Rotary-dominated heads degrade; pass-through-dominated
    heads stay near 1. Intended to be reported alongside every weight-space
    QK claim, so "this is a proxy" is a measurement rather than a caveat.

    `mask` (bool, same shape) restricts to the pairs that matter — for
    causal attention, the lower triangle; the upper triangle is never
    softmaxed and its disagreement is meaningless.

    Returns dict(pearson, max_abs_err, rel_fro_err, n_pairs).
    """
    P = _as_numpy(predicted).astype(np.float64, copy=False)
    A = _as_numpy(actual).astype(np.float64, copy=False)
    if P.shape != A.shape:
        raise ValueError(
            f"qk_prediction_fidelity: shape mismatch {P.shape} vs {A.shape}"
        )
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p, a = P[m], A[m]
    else:
        p, a = P.ravel(), A.ravel()

    if p.size < 2:
        raise ValueError("qk_prediction_fidelity: need at least 2 pairs")
    denom = np.linalg.norm(a)
    ps, as_ = p - p.mean(), a - a.mean()
    sp, sa = np.linalg.norm(ps), np.linalg.norm(as_)
    pearson = float(ps @ as_ / (sp * sa)) if sp > 0 and sa > 0 else float("nan")
    return dict(
        pearson=pearson,
        max_abs_err=float(np.max(np.abs(p - a))),
        rel_fro_err=float(np.linalg.norm(p - a) / denom) if denom > 0 else float("nan"),
        n_pairs=int(p.size),
    )


def causal_pair_mask(n: int, include_diagonal: bool = True) -> np.ndarray:
    """
    Lower-triangular bool mask over (query, key) pairs.

    Only these pairs survive the causal mask, so only these should enter a
    fidelity score or an offset-resolved statistic. Note every such pair has
    Δ = j - i <= 0: the rotary offsets that actually matter are non-positive.
    """
    return np.tril(np.ones((n, n), dtype=bool), k=0 if include_diagonal else -1)


def rope_energy_fraction(head_size: int, rotary_ndims: int) -> float:
    """
    Fraction of each head's dims that carry position. 0.25 on Pythia.

    Reported alongside any a_frac so a reader can see immediately how much
    of the head the rotary claim can possibly be about.
    """
    return float(rotary_ndims) / float(head_size) if head_size else 0.0
