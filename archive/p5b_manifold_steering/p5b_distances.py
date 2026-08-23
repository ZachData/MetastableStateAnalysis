"""
p5b_manifold_steering/p5b_distances.py — the metric layer for Sub-exp B.

Sub-exp B is a correlation between two pairwise-distance vectors. Which
distance is used on each side is a free parameter that was, until now,
silently fixed: the activation side to L2-sphere chordal, the behavior side
to arc length along a fitted Hellinger spline. Neither choice was recorded
in the output, and neither was the one the model actually operates in.

This module makes both choices explicit, named, and swappable, with NO
verdict logic — it computes distance matrices and nothing else. Verdicts
stay in isometry_test.py, per the project's standing separation of readings
from verdicts (DESIGN_dual_reading.md, "No decision logic").

WHAT THIS BUYS, CONCRETELY
--------------------------
1. Cross-architecture comparability. The model never reads the raw residual
   stream; every sub-layer reads LN(x). GPT-2 is sequential; Pythia/GPT-NeoX
   is parallel-residual with TWO LayerNorms per block, of which
   `input_layernorm` — not `post_attention_layernorm`, despite the name —
   is what attention reads. core/ln_frame.py::frame_for_hidden_state
   already resolves that plus this project's embedding-stripped off-by-one
   in one place. Without routing through it, "distance between cluster
   centroids" denotes a different operation per architecture and any
   cross-model Phase 5b comparison is confounded by extraction convention.

2. An arbiter for the behavior side. core/functional_distance.py's
   symmetrized KL is the divergence the readout actually implies, computed
   directly from cached probabilities with no spline and no sphere
   embedding in the way. Hellinger stays primary (it is what Wurgaft used,
   and P5b-B2's 0.7 threshold was calibrated against his numbers), but if
   r_manifold is strong under one and collapses under the other, that is
   information about the fit rather than about the model.

DOCUMENTED LIMITATION, carried from core/functional_distance.py: symmetrized
KL is NOT a metric — the triangle inequality fails. That is fine here
because Sub-exp B only ever correlates distance vectors. Do not reuse
"sym_kl" anywhere that assumes metricity (Sub-exp D's subspace scoring, any
MDS embedding meant to be trusted metrically, anything calling
scipy.spatial).

LN IS NOT LINEAR, so LN(mean of tokens) != mean of LN(tokens). Frames must
therefore be applied to token activations BEFORE masking and averaging, not
post-hoc to an already-built centroid. That is why frame_centroids exists
and why it takes raw activations rather than the output of
load_plateau_centroids.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.metrics import l2_normalize


ACTIVATION_FRAMES  = ("sphere", "ln", "raw")
BEHAVIOR_METRICS   = ("hellinger", "sym_kl")


# ---------------------------------------------------------------------------
# LN parameter resolution
# ---------------------------------------------------------------------------

def ln_params_for_layers(
    model,
    layer_idxs,
    n_hidden_states: int,
    which: str = "attn",
    embedding_stripped: bool = True,
    last_is_post_final_ln: bool = False,
) -> dict:
    """
    Resolve the LN frame for each requested hidden-state index.

    Thin wrapper over core.ln_frame.frame_for_hidden_state so call sites
    never re-derive the off-by-one. Returns
    {layer_idx: {"frame": str, "block_idx": int|None, "params": dict|None}}.

    Layers that raise IndexError (inconsistent extraction conventions) are
    recorded with frame="unavailable" and params=None rather than taking
    the whole run down — a frame that cannot be resolved for one layer is a
    reason to skip that layer's LN reading, not to lose the sphere reading
    for every layer.

    ARCHITECTURE SCOPE: core/ln_frame.py's extraction half (_neox_inner,
    _WHICH_TO_ATTR) is written against GPT-NeoX/Pythia module structure. It
    duck-types, but there is no GPT-2 path. On GPT-2 this will return
    frame="unavailable" for every layer, and the caller should fall back to
    the sphere frame AND RECORD THAT IT DID — see `frame` in isometry.json.
    Silently reporting a sphere number under an "ln" label is the failure
    mode to avoid here.
    """
    from core.ln_frame import frame_for_hidden_state

    out: dict = {}
    for li in layer_idxs:
        li = int(li)
        try:
            out[li] = frame_for_hidden_state(
                model, li, n_hidden_states,
                which=which,
                embedding_stripped=embedding_stripped,
                last_is_post_final_ln=last_is_post_final_ln,
            )
        except Exception as e:  # IndexError, or a model without the attrs
            out[li] = {"frame": "unavailable", "block_idx": None,
                       "params": None, "error": str(e)}
    return out


def _apply_frame(X, frame: str, ln_params: Optional[dict]):
    """
    Map one layer's (n_tokens, d) activations into the requested frame.

    "raw"    — untouched. The ambient residual stream.
    "sphere" — L2-normalized rows. Phase 1's clustering frame.
    "ln"     — core.ln_frame.ln_transform with this layer's learned
               gamma/beta, then L2-normalized. The composition (rather than
               raw LN inner products) is deliberate and matches
               ln_frame_gram, so LN-frame numbers stay directly comparable
               to sphere-frame ones instead of differing by an overall
               scale nobody tracked.

    beta is included, not dropped: the network reads gamma*xhat + beta, and
    that shared offset changes pairwise angles. Excluding it would measure
    a frame nothing in the model uses.
    """
    arr = np.asarray(X, dtype=np.float64)
    if frame == "raw":
        return arr
    if frame == "sphere":
        return np.asarray(l2_normalize(arr), dtype=np.float64)
    if frame == "ln":
        from core.ln_frame import ln_transform
        if ln_params is None:
            raise ValueError(
                "_apply_frame: frame='ln' requires ln_params for this layer. "
                "Resolve them with ln_params_for_layers, and fall back to "
                "frame='sphere' (recording the fallback) when they are "
                "unavailable."
            )
        Y = ln_transform(arr, **ln_params)
        return np.asarray(l2_normalize(Y), dtype=np.float64)
    raise ValueError(
        f"_apply_frame: unknown frame {frame!r}; expected one of "
        f"{ACTIVATION_FRAMES}"
    )


# ---------------------------------------------------------------------------
# Frame-aware centroid construction
# ---------------------------------------------------------------------------

def frame_centroids(
    activations,
    label_arrays,
    trajectories,
    traj_ids,
    frame: str = "sphere",
    ln_frames: Optional[dict] = None,
    renormalize: bool = True,
) -> tuple:
    """
    Build one centroid per trajectory, in a chosen frame, from raw
    activations.

    Mirrors compute_centroid_trajectories' chain-walking exactly, but
    applies the frame transform to the LAYER's tokens before masking, which
    is the only correct order for a non-linear frame (see module docstring).

    Parameters
    ----------
    activations  : (n_layers, n_tokens, d) — load_phase1_run["activations"]
    label_arrays : {layer_idx: (n_tokens,)} or list — HDBSCAN labels
    trajectories : list of {"id", "chain", ...}
    traj_ids     : the identity list from load_plateau_centroids. Output
                   rows follow THIS order.
    frame        : one of ACTIVATION_FRAMES
    ln_frames    : ln_params_for_layers output; required when frame="ln"
    renormalize  : L2-normalize the per-trajectory mean, matching
                   compute_centroid_trajectories / load_plateau_centroids.
                   Leave True unless you specifically want norm information
                   preserved (frame="raw" with renormalize=False is the
                   only combination that does).

    Returns
    -------
    centroids : (n_kept, d) float32, rows ordered by `kept`
    kept      : list[int] — subset of traj_ids that had at least one usable
                chain step. Callers MUST re-index every other per-trajectory
                array by this list rather than assuming it equals traj_ids.
    """
    if frame not in ACTIVATION_FRAMES:
        raise ValueError(f"frame_centroids: unknown frame {frame!r}")

    acts = np.asarray(activations)
    by_id = {int(t["id"]): t for t in trajectories}

    # Cache framed layers — a trajectory chain revisits layers across
    # trajectories, and ln_transform over (n_tokens, d) is not free.
    framed_cache: dict = {}

    def _framed(layer_idx: int):
        if layer_idx not in framed_cache:
            if not (0 <= layer_idx < acts.shape[0]):
                framed_cache[layer_idx] = None
            else:
                params = None
                if frame == "ln":
                    fr = (ln_frames or {}).get(layer_idx)
                    if fr is None or fr.get("params") is None:
                        framed_cache[layer_idx] = None
                        return framed_cache[layer_idx]
                    params = fr["params"]
                framed_cache[layer_idx] = _apply_frame(
                    acts[layer_idx], frame, params
                )
        return framed_cache[layer_idx]

    centroids, kept = [], []
    for tid in traj_ids:
        tid = int(tid)
        traj = by_id.get(tid)
        if traj is None:
            continue
        vecs = []
        for layer_idx, cluster_id in traj.get("chain", []):
            layer_idx = int(layer_idx)
            H = _framed(layer_idx)
            if H is None:
                continue
            labels = label_arrays.get(layer_idx) if isinstance(label_arrays, dict) \
                else (label_arrays[layer_idx] if 0 <= layer_idx < len(label_arrays) else None)
            if labels is None:
                continue
            labels = np.asarray(labels)
            if labels.shape[0] != H.shape[0]:
                raise ValueError(
                    f"frame_centroids: layer {layer_idx} has {labels.shape[0]} "
                    f"labels but {H.shape[0]} activation rows"
                )
            mask = labels == cluster_id
            if mask.any():
                vecs.append(H[mask].mean(axis=0))
        if not vecs:
            continue
        c = np.mean(vecs, axis=0)
        if renormalize:
            c = c / max(float(np.linalg.norm(c)), 1e-12)
        centroids.append(c)
        kept.append(tid)

    if not centroids:
        raise ValueError(
            f"frame_centroids: no trajectory in traj_ids produced a centroid "
            f"in frame {frame!r}. If frame='ln', check ln_frames — an "
            f"unavailable LN frame at every layer yields this."
        )

    return np.stack(centroids, axis=0).astype(np.float32), kept


# ---------------------------------------------------------------------------
# Distance matrices
# ---------------------------------------------------------------------------

def activation_distance_matrix(centroids, metric: str = "chordal") -> np.ndarray:
    """
    (n, n) pairwise distances between already-framed centroids.

    metric
    ------
    "chordal"  — Euclidean. On unit-norm input this is the chord of the
                 great-circle arc. The default because it is what the
                 existing d_linear control computes, so the two stay on the
                 same footing.
    "angular"  — arccos of the cosine, i.e. the great-circle arc itself.
                 Monotone in chordal, so Spearman-based readings are
                 identical and only Pearson-based ones shift. Requires
                 unit-norm rows.

    The frame is NOT an argument here — it is already baked into
    `centroids` by frame_centroids. Applying a frame at this stage would be
    the LN-nonlinearity mistake described in the module docstring.
    """
    X = np.asarray(centroids, dtype=np.float64)
    if metric == "chordal":
        diff = X[:, None, :] - X[None, :, :]
        D = np.linalg.norm(diff, axis=-1)
    elif metric == "angular":
        Xn = np.asarray(l2_normalize(X), dtype=np.float64)
        G = np.clip(Xn @ Xn.T, -1.0, 1.0)
        D = np.arccos(G)
    else:
        raise ValueError(
            f"activation_distance_matrix: unknown metric {metric!r}; "
            f"expected 'chordal' or 'angular'"
        )
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def behavior_distance_matrix(distributions, metric: str = "hellinger") -> np.ndarray:
    """
    (n, n) pairwise distances between per-trajectory output distributions.

    metric
    ------
    "hellinger" — (1/√2)·‖√p − √q‖. A true metric, bounded in [0, 1], and
                  the geometry My's spline is fitted in. PRIMARY: this is
                  the reading P5b-B1/B2/B3 are scored on, because P5b-B2's
                  0.7 threshold was calibrated against Wurgaft's
                  Hellinger-space numbers and swapping the metric under a
                  threshold calibrated for a different one is meaningless.
    "sym_kl"    — (KL(p‖q) + KL(q‖p))/2 via core.functional_distance's
                  matmul identity. SECONDARY / arbiter only. Not a metric.

    Both are computed from probabilities with no spline and no
    parameterization, which is the point: the old geodesic route stacked a
    sphere embedding and a spline fit before producing a number, and a
    disagreement between the two could not be attributed to either.
    """
    P = np.asarray(distributions, dtype=np.float64)
    if metric == "hellinger":
        sq = np.sqrt(np.clip(P, 0.0, None))
        sq = sq / np.maximum(np.linalg.norm(sq, axis=1, keepdims=True), 1e-12)
        diff = sq[:, None, :] - sq[None, :, :]
        D = np.linalg.norm(diff, axis=-1) / np.sqrt(2.0)
    elif metric == "sym_kl":
        from core.functional_distance import kl_matrix_from_probs, sym_kl
        D = sym_kl(kl_matrix_from_probs(P))
    else:
        raise ValueError(
            f"behavior_distance_matrix: unknown metric {metric!r}; "
            f"expected one of {BEHAVIOR_METRICS}"
        )
    D = np.asarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def upper_triangle(D: np.ndarray) -> np.ndarray:
    """Flat upper-triangle (excluding diagonal) of a square matrix."""
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"upper_triangle: expected square matrix, got {D.shape}")
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def pair_indices(n: int) -> np.ndarray:
    """(n_pairs, 2) index pairs matching upper_triangle's ordering."""
    iu = np.triu_indices(n, k=1)
    return np.stack(iu, axis=1)
