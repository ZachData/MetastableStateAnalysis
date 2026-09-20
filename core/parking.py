"""
core/parking.py — the numeric instruments for Phase 10's free rows.

`p10_cluster_function/notes-10.md` §8 lists thirteen experiments; four of them
(F0, F1, F11, F12) need no forward pass because the artifacts they read are
already on disk. This module holds the *pure* half of those four: the
statistics and their content-free baselines, with no I/O, no model and no
sklearn, so they run in `scripts/check.sh pure` and can be tested against
closed forms rather than against a sweep.

The runners that point these at real directories live in `tools/run/`.

WHAT IS HERE, AND WHICH ROW EACH PIECE SERVES
---------------------------------------------
**Row A0 / F11 — the causal-mask baseline.** `math-10.md` §1 derives, in closed
form, what `received[key]` reports when the network does nothing: under a
causal mask token `j` is visible to only `n - j` queries, so the statistic has
a tilt toward early positions of about 1 600x at the battery's n = 264 before
content enters. `attention-10.md` row A0 gates every other attention row on
dividing it out. `received_attention` reproduces the project's existing
statistic exactly (`p1_mstate_tracking/visualization/noise_importance_proxy.py`
`_received_attention`); `received_baseline` is the closed form to divide by.

**F0 — the anchor test.** `2411.04990`'s mechanism is that early tokens act as
nuclei for cluster formation. That is a per-token, position-indexed prediction
checkable against a label vector and nothing else — no beta, no unit convention
and no reading of the paper (`notes-10.md` §3.2, as corrected).

**F12 — parked versus pinned.** `math-10.md` §2: under a causal mask
`Z_{beta,i}` is linear in position in the concentration regime, so the raw
partition function is mostly position and `math-1.md` §1A.6's "high Z = sink"
identification is an unmasked-model statement. `position_corrected_partition_function`
divides the structural `(i + 1)` out.

WHAT THIS MODULE DOES NOT DO
----------------------------
It computes no p-values and no e-values. Statistics and baselines only; the
nulls are `core/nulls.py`'s and the calibration is `core/evalues.py`'s, and
keeping the three apart is what lets each be tested without the others.

It also takes no view on thresholds. `notes-10.md` §10.2 hazard 3 is that four
binarised signatures with four thresholds invite threshold-shopping, so nothing
here binarises anything -- every function returns the continuous quantity and
the caller owns the cut.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "harmonic",
    "uniform_causal_attention",
    "received_attention",
    "received_baseline",
    "relative_to_layer_mean",
    "mask_corrected_received",
    "population_enrichment",
    "cluster_nuclei",
    "mean_nucleus_position",
    "clustered_position_bias",
    "partition_function",
    "log_partition_function",
    "position_corrected_partition_function",
    "log_position_corrected_partition_function",
]


# ---------------------------------------------------------------------------
# Row A0 / F11 — the causal-mask baseline
# ---------------------------------------------------------------------------

def harmonic(k: int) -> float:
    """``H_k = sum_{m=1}^{k} 1/m``, with ``H_0 = 0``.

    Computed by summation rather than by digamma so the module stays numpy-only
    and so the test can compare it against an explicit loop.
    """
    if k < 0:
        raise ValueError(f"harmonic number needs k >= 0; got {k}")
    return float(np.sum(1.0 / np.arange(1, k + 1))) if k > 0 else 0.0


def uniform_causal_attention(n: int) -> np.ndarray:
    """
    The content-free causal attention matrix: ``a[i, j] = 1 / (i + 1)`` for
    ``j <= i`` and 0 above the diagonal.

    This is the null `math-10.md` §1 works in -- attention uniform within the
    causal triangle. It exists so `received_baseline`'s closed form can be
    checked against a matrix the same code path consumes, rather than asserted.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    a = np.tril(np.ones((n, n), dtype=np.float64))
    return a / a.sum(axis=1, keepdims=True)


def received_attention(
    attn_layer: np.ndarray,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    Attention paid TO each token, summed over heads and over queries.

    Reproduces `noise_importance_proxy._received_attention` exactly, including
    its diagonal handling, so a baseline-corrected number is comparable with
    the flip as this project has always reported it.

    Parameters
    ----------
    attn_layer : (n_heads, n, n) array, ``attn[h, query, key]``.
    zero_diagonal : drop self-attention before summing, as the existing
        statistic does.

    Returns
    -------
    (n,) array, ``received[key]``.
    """
    a = np.asarray(attn_layer, dtype=np.float64)
    if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            f"attn_layer must be (n_heads, n, n); got shape {a.shape}"
        )
    a = a.copy()
    if zero_diagonal:
        idx = np.arange(a.shape[-1])
        a[:, idx, idx] = 0.0
    return a.sum(axis=(0, 1))


def received_baseline(
    n: int,
    n_heads: int = 1,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """
    `received_attention` evaluated on `uniform_causal_attention`, in closed
    form (`math-10.md` §1).

    With the diagonal kept::

        received(j) = sum_{i=j}^{n-1} 1/(i+1) = H_n - H_j

    and ``sum_j received(j) = n`` exactly, so the layer mean is exactly 1 and
    `received(j)` IS the "x layer average" quantity the flip reports -- the
    baseline and the measurement are directly comparable with no rescaling.

    With the diagonal zeroed, the self term ``1/(j+1)`` is removed::

        received(j) = H_n - H_{j+1}

    which is **exactly 0 at the last position**, because the last token is
    attended to by nobody but itself. That zero is why `mask_corrected_received`
    refuses to divide by this variant.

    Multiplied by `n_heads`, since the project's statistic sums over heads and
    every head has the same content-free baseline.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if n_heads < 1:
        raise ValueError(f"n_heads must be >= 1; got {n_heads}")
    H = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, n + 1))))  # H[0..n]
    j = np.arange(n)
    out = H[n] - (H[j + 1] if zero_diagonal else H[j])
    return out * float(n_heads)


def relative_to_layer_mean(values: np.ndarray) -> np.ndarray:
    """``values / values.mean()`` -- the "x layer average" normalisation the
    flip is reported in. Refuses a non-positive mean rather than returning
    infinities that read like measurements."""
    v = np.asarray(values, dtype=np.float64)
    m = float(v.mean())
    if not np.isfinite(m) or m <= 1e-12:
        raise ValueError(f"layer mean is not usable for normalisation: {m!r}")
    return v / m


def mask_corrected_received(
    attn_layer: np.ndarray,
    renormalise: bool = True,
) -> np.ndarray:
    """
    Observed received attention divided by its content-free causal-mask
    baseline -- what row A0 asks for.

    A value of 1 means "exactly what an empty network routes here". Above 1 is
    content routing attention TO this token beyond the structural tilt; below 1
    is content routing it away.

    **The diagonal is kept here, deliberately**, though the uncorrected
    statistic zeroes it: the diagonal-zeroed baseline is exactly 0 at the last
    position (`received_baseline`), and dividing by it manufactures an infinity
    at one end of the very axis this correction exists to make readable. Self-
    attention is part of the mask's structure and is divided out with the rest.

    Parameters
    ----------
    attn_layer : (n_heads, n, n).
    renormalise : rescale the result to mean 1, so it stays in the same "x
        layer average" units as the uncorrected flip and the two can be plotted
        on one axis.
    """
    a = np.asarray(attn_layer, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"attn_layer must be (n_heads, n, n); got {a.shape}")
    n_heads, n = a.shape[0], a.shape[-1]
    obs = received_attention(a, zero_diagonal=False)
    base = received_baseline(n, n_heads=n_heads, zero_diagonal=False)
    out = obs / base
    return relative_to_layer_mean(out) if renormalise else out


def population_enrichment(values: np.ndarray, mask: np.ndarray) -> float:
    """
    ``values[mask].mean() / values.mean()`` -- the enrichment ratio the flip is
    reported as (1.6x unclustered, 0.5x clustered).

    Returns ``nan`` when the population is empty, which is a real outcome at
    layers where HDBSCAN finds nothing, and is distinguishable downstream from
    a measured 0.
    """
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if v.shape != m.shape:
        raise ValueError(f"values {v.shape} and mask {m.shape} must match")
    if not m.any():
        return float("nan")
    denom = float(v.mean())
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float(v[m].mean() / denom)


# ---------------------------------------------------------------------------
# F0 — the anchor test
# ---------------------------------------------------------------------------

def cluster_nuclei(labels: Sequence[int]) -> dict:
    """
    Per-cluster position summary. ``{cluster_id: {"first", "last", "mean",
    "size"}}``, positions being indices into `labels`.

    "Nucleus" is read as **the earliest member**, which is the only reading the
    causal mask permits: a token at position p cannot have been drawn toward
    anything later than p, so whatever seeded a cluster is at or before its
    first member. Noise (-1) is excluded and is not a cluster.
    """
    lab = np.asarray(labels)
    if lab.ndim != 1:
        raise ValueError(f"labels must be 1-D; got shape {lab.shape}")
    out = {}
    for cid in sorted(set(lab.tolist()) - {-1}):
        pos = np.flatnonzero(lab == cid)
        out[int(cid)] = {
            "first": int(pos[0]),
            "last": int(pos[-1]),
            "mean": float(pos.mean()),
            "size": int(pos.size),
        }
    return out


def mean_nucleus_position(positions: np.ndarray, labels: Sequence[int]) -> float:
    """
    F0's statistic: the mean, over clusters, of the **normalised position of
    each cluster's earliest member**.

    Signature is ``(positions, labels)`` so it drops straight into
    `core.nulls.label_permutation_null`, which holds the first argument fixed
    and permutes the second -- exactly the null this needs ("cluster membership
    is independent of position", with the size profile preserved exactly).

    Range is [0, 1]. **Small is the parking/nuclei reading**, so the
    alternative is ``"less"`` and that direction is fixed here, in the code,
    before any sweep is read.

    Returns ``nan`` when there are no clusters -- a real outcome, not an error.

    WHAT IT DOES NOT CONTROL FOR, and the permutation null does
    -----------------------------------------------------------
    The minimum of a size-s subset of n positions is small by construction, and
    the more clusters there are the smaller the mean of their minima. Neither
    is evidence of anything. Both are held fixed by permuting labels among
    tokens, which preserves the cluster count and every cluster's size exactly;
    the null is therefore the right chance baseline and the raw value of this
    statistic means nothing on its own.
    """
    pos = np.asarray(positions, dtype=np.float64)
    lab = np.asarray(labels)
    if pos.shape != lab.shape:
        raise ValueError(f"positions {pos.shape} and labels {lab.shape} must match")
    n = pos.size
    if n < 2:
        return float("nan")
    firsts = [pos[np.flatnonzero(lab == cid)[0]] for cid in sorted(set(lab.tolist()) - {-1})]
    if not firsts:
        return float("nan")
    return float(np.mean(firsts) / (n - 1))


def clustered_position_bias(positions: np.ndarray, labels: Sequence[int]) -> float:
    """
    The companion statistic, and the one that bears directly on row A0: the
    mean normalised position of **clustered** tokens minus that of **noise**
    tokens.

    `math-10.md` §1 shows a partition whose unclustered members average early
    and whose clustered members average late reproduces the observed 1.6x/0.5x
    attention flip with no learned behaviour anywhere. This measures that
    partition property directly, so the confound can be read off rather than
    argued about.

    Positive means clustered tokens sit later than noise tokens. Same
    ``(positions, labels)`` signature and the same permutation null; the
    direction is two-sided, because both signs are informative here and
    neither was predicted in advance.

    Returns ``nan`` when either population is empty.
    """
    pos = np.asarray(positions, dtype=np.float64)
    lab = np.asarray(labels)
    if pos.shape != lab.shape:
        raise ValueError(f"positions {pos.shape} and labels {lab.shape} must match")
    n = pos.size
    if n < 2:
        return float("nan")
    noise = lab == -1
    if not noise.any() or noise.all():
        return float("nan")
    scale = n - 1
    return float(pos[~noise].mean() / scale - pos[noise].mean() / scale)


# ---------------------------------------------------------------------------
# F12 — parked versus pinned
# ---------------------------------------------------------------------------

def partition_function(
    X: np.ndarray,
    beta: float,
    causal: bool = True,
) -> np.ndarray:
    """
    ``Z_{beta,i} = sum_j exp(beta <x_i, x_j>)``, over ``j <= i`` when `causal`.

    `math-1.md` §1A.6 reads a high-`Z` token as one the metric makes expensive
    to move -- a sink. `math-10.md` §2 shows that identification is an
    **unmasked-model statement**: in the concentration regime the unmasked
    ``Z_i = n e^{beta gamma}`` is position-independent, so its spread is
    content, while the masked ``Z_i = (i+1) e^{beta gamma}`` is linear in
    position and puts its MINIMUM at position 0 -- the sink. Pythia is masked.

    Computed in a numerically stable way: the row maximum is factored out
    before exponentiating and multiplied back in log space, so large `beta`
    does not overflow to inf and silently produce a uniform answer.

    Parameters
    ----------
    X : (n, d) array. Unit-norm rows are the project's convention
        (`activations.npz` is sphere-projected), but nothing here requires it.
    beta : inverse temperature.
    causal : restrict the sum to ``j <= i``.
    """
    return np.exp(log_partition_function(X, beta, causal=causal))


def log_partition_function(
    X: np.ndarray,
    beta: float,
    causal: bool = True,
) -> np.ndarray:
    """
    ``log Z_{beta,i}`` by the log-sum-exp identity -- and the form to prefer.

    `partition_function` exponentiates this, and at large `beta` the linear
    value is **not representable** however it is computed: on the sphere the
    diagonal term alone is ``e^{beta}``, which overflows float64 past
    ``beta ~ 709``. That is a property of the quantity, not of the algorithm,
    so the stable API is the log one and the exponential is the convenience.
    Everything the project asks of `Z` -- the position correction, ratios
    between tokens, the sink comparison -- is a difference in log space.
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {Xa.shape}")
    S = float(beta) * (Xa @ Xa.T)
    if causal:
        S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
    row_max = np.max(S, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(S - row_max), axis=1)) + row_max[:, 0]


def position_corrected_partition_function(Z: np.ndarray) -> np.ndarray:
    """
    ``Z_i / (i + 1)`` -- `math-10.md` §2's instruction, stated there as
    "measure ``Z_i/(i+1)``, not ``Z_i``".

    Under a causal mask row `i` sums over exactly ``i + 1`` terms, so ``(i+1)``
    is the count the mask contributes before any geometry does. What is left is
    the per-visible-token mean of ``exp(beta <x_i, x_j>)``, which is the
    quantity the metric reading of §1A.6 was about, and it is what separates a
    **pinned** particle (expensive to move) from a merely **parked** one (still
    because nothing is pushing it).
    """
    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 1:
        raise ValueError(f"Z must be 1-D; got shape {Za.shape}")
    return Za / np.arange(1, Za.size + 1, dtype=np.float64)


def log_position_corrected_partition_function(logZ: np.ndarray) -> np.ndarray:
    """``log Z_i - log(i + 1)`` -- the correction in the space that survives
    large `beta`. See `log_partition_function` for why that is the default
    form and not a variant."""
    L = np.asarray(logZ, dtype=np.float64)
    if L.ndim != 1:
        raise ValueError(f"logZ must be 1-D; got shape {L.shape}")
    return L - np.log(np.arange(1, L.size + 1, dtype=np.float64))
