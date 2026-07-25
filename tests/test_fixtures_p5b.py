"""
tests/fixtures_p5b.py — a Phase 1 run fixture with ground truth.

WHY THIS EXISTS
---------------
The previous integration fixture (`test_phase5b_io._make_run_dir`) is
structurally valid but semantically empty, and that is why the Phase 5b
integration tests could not tell a working pipeline from a broken one:

  * `activations.npz` was `rng.standard_normal(...)` — no cluster structure
  * `centroid_trajectories.npz` was *separately* drawn random data, with no
    relationship to those activations
  * the synthetic logit cache assigned distributions by TOKEN INDEX,
    `(t + layer) % vocab`, with no relationship to cluster membership
  * there was no `hdbscan_labels.json` at all

So nothing in the fixture connected activations to clusters to behavior.
The only assertion the integration tests could make was "a file appeared
and the number in it is finite," which is satisfied equally well by a
correct pipeline and by one computing a meaningless correlation. Adding
random labels to that fixture would turn the tests green while preserving
exactly that blindness. This module builds the fixture the tests actually
need instead.

THE GROUND TRUTH
----------------
A latent coordinate θ_i ∈ [0, 1] is assigned to each cluster, and BOTH
sides are generated from it:

  activation : cluster i sits at direction v_i on a quarter-circle arc
               INSIDE span(U_S) (see "SUBSPACE STRUCTURE" below), so
               sphere-frame distance is monotone in |θ_i − θ_j|
  behavior   : cluster i emits a wide Gaussian bump over the vocabulary
               centred at a position linear in θ_i, so Hellinger distance
               is also monotone in |θ_i − θ_j|

An isometry therefore genuinely holds, and a correct pipeline must find it.

Four properties are engineered in deliberately, each making one specific
failure detectable:

1. NORM SPREAD (`norm_spread`, default 3.0). Each cluster's activations are
   scaled by a fixed factor drawn from a geometric sequence and assigned in
   a permutation that scrambles the θ ordering. Raw-frame distances are
   then dominated by scale rather than by θ. This makes the frame-vs-raw
   control (P5b-B1) falsifiable in the fixture instead of vacuous. A 3×
   norm spread across clusters is modest and physically ordinary for a
   residual stream.

2. NOISE TOKENS (label −1). Present so that a masked mean and a global mean
   differ. Under the original bug — global mean over every token at a layer
   — all clusters receive the SAME distribution, every pairwise behavior
   distance collapses to zero, the correlation becomes undefined and every
   verdict returns False. The fixture detects the bug it was previously
   blind to.

3. A NEGATIVE MODE (`behavior="shuffled"`). The θ→vocabulary-position map
   is composed with a FIXED permutation, breaking the correspondence while
   leaving every other property intact. A correct pipeline must return a
   low r here.

4. SUBSPACE STRUCTURE (Sub-exp D). The latent arc lives entirely inside
   span(U_S) (the first two columns of a fixed orthonormal frame Q), and
   each cluster additionally carries a FIXED-NORM component inside
   span(U_A) — a different, disjoint block of Q — with a direction drawn
   independently of θ. So S-projected distances recover the arc; A-projected
   distances are structured (not merely noise) but carry no relationship to
   θ; a correct pipeline must show r_S high and r_A low. This property did
   not exist in the first version of this fixture: measured there,
   r_S=+0.993, r_A=+0.989, r_full=+0.996 — P5b-D1 (r_S > r_full >= r_A)
   FAILED even though the isometry was near-perfect, because the arc was
   spread across random directions and no subspace was privileged. Fixed
   by confining the arc to span(U_S) explicitly. `make_orthogonal_projectors`
   below must be used to generate the matching S/A projector file — it
   shares Q's fixed seed with this function, so the projectors on disk
   describe the same subspaces the data was generated in.

MEASURED, not assumed:

  B (40 seeds, defaults, i.e. WITH the Sub-exp D subspace structure active):
    aligned    r_sphere min +0.852  mean +0.911    r_raw mean ≈ −0.14
    shuffled   r_sphere stays negative, well under the 0.70 threshold
    at seed=0 (the fixed seed the integration tests run):
      aligned : r_full=+0.885  r_raw=−0.161  delta=+1.046
      shuffled: r_full=−0.290

  D (30-40 seeds, defaults, dim_S == dim_A == 8):
    r_S    min  +0.982
    r_full mean ≈ +0.91 (varies with alpha — see DEFAULT_ALPHA below)
    r_A    mean ≈ −0.01, uncorrelated with theta as intended
    P5b-D1 (r_S > r_full >= r_A) holds on every seed tried

These are joint numbers: B and D share one generative function, so a
constant tuned for D in isolation can silently break B. That happened once
already — see DEFAULT_ALPHA's own comment — which is why both are measured
together here rather than separately.

Do NOT switch the negative mode to a per-seed random permutation: measured
over 50 seeds it reaches r = +0.77 by chance, which is above the P5b-B2
threshold of 0.70. With only 6 control points there are 15 pairs, and a
random ordering is not reliably uncorrelated. The fixed permutation is
deterministic and stays well clear of that threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Kept compatible with tests/test_phase5b_io.py's constants so the Phase 2
# projector fixture (_make_p2_projectors) can be reused unchanged for the
# structural (regime A) tests, which never reach Sub-exp D.
D_MODEL = 32

DEFAULT_N_CLUSTERS   = 6
DEFAULT_N_LAYERS     = 6
DEFAULT_CHAIN_LAYERS = (1, 2, 3, 4)
DEFAULT_PLATEAU      = (1, 2, 3, 4)
DEFAULT_MERGE        = (3,)
DEFAULT_VOCAB        = 128
DEFAULT_TOK_PER      = 8
DEFAULT_NOISE_TOK    = 6

# Sub-exp D — subspace block size, fixed-norm A-component magnitude, and
# residual noise outside both blocks. See "SUBSPACE STRUCTURE" above.
#
# ALPHA IS LOAD-BEARING FOR BOTH B AND D — do not raise it without
# re-sweeping both. It was originally set to 0.9 to make D's separation
# generous, without checking what a large, cluster-specific, theta-
# independent A-component does to the FULL (unprojected) sphere-frame
# centroid, which is what Sub-exp B actually measures. At 0.9 it dragged
# the swept min of r_full down to +0.368 (mean +0.662) — below the 0.70
# threshold P5b-B2 requires, and below it even at the fixed seed=0 the
# integration tests run (+0.507). 0.5 satisfies both: r_S stays >= +0.982
# (D holds every seed at every alpha tried) while r_full recovers to a
# swept min of +0.852. See fixtures_p5b's "MEASURED" block below for the
# combined numbers this constant was chosen against.
DEFAULT_SUBSPACE_BLOCK = 4
DEFAULT_ALPHA          = 0.5
DEFAULT_AMBIENT        = 0.25

# Fixed, not random — see module docstring.
SCALE_PERMUTATION    = (3, 0, 5, 1, 4, 2)
BEHAVIOR_SHUFFLE     = (2, 5, 0, 4, 1, 3)

# Fixed seed for the orthonormal frame Q that defines S/A/rest. Shared
# between _subspace_basis (data generation) and make_orthogonal_projectors
# (the Phase 2 projector file). If you change one, change the other, or the
# projectors on disk will not describe the subspaces the data lives in.
SUBSPACE_FRAME_SEED = 99


# ---------------------------------------------------------------------------
# Subspace geometry (Sub-exp D)
# ---------------------------------------------------------------------------

def _subspace_basis(rng, d_model, n_clusters, theta, subspace_block,
                    alpha, seed_basis: int = SUBSPACE_FRAME_SEED):
    """
    Build S and A subspaces and place the latent arc INSIDE S.

    Returns (V, A_component, U_S, U_A, U_rest).

      V           : (n_clusters, d) unit vectors on the arc, spanning
                    e1, e2 = the first two columns of U_S
      A_component : (n_clusters, k_a) fixed-norm, θ-independent offset in
                    span(U_A) — see class docstring point 4
      U_S         : (d, 2*subspace_block) — U_pos ∪ U_neg block
      U_A         : (d, 2*subspace_block) — disjoint from U_S
      U_rest      : (d, d - 4*subspace_block) — ambient noise subspace

    Fixed norm on A_component matters: centroids are L2-normalized
    downstream, and a varying ‖v_i + a_i‖ would inject a per-cluster scalar
    into the S-projection and degrade r_S for a reason unrelated to the
    hypothesis under test.

    `seed_basis` is independent of the data-generation `rng` so the
    subspaces are identical across data seeds — only the noise and A-
    component directions vary seed to seed, not the geometry being tested.
    """
    Qrng = np.random.default_rng(seed_basis)
    Q, _ = np.linalg.qr(Qrng.standard_normal((d_model, d_model)))

    kb     = subspace_block
    U_S    = Q[:, 0:2 * kb]
    U_A    = Q[:, 2 * kb:4 * kb]
    U_rest = Q[:, 4 * kb:]

    ang = (np.pi / 2.0) * theta
    e1, e2 = Q[:, 0], Q[:, 1]              # inside span(U_S) by construction
    V = np.cos(ang)[:, None] * e1[None, :] + np.sin(ang)[:, None] * e2[None, :]
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    A_dir = rng.standard_normal((n_clusters, U_A.shape[1]))
    A_dir /= np.linalg.norm(A_dir, axis=1, keepdims=True)
    A_component = alpha * (A_dir @ U_A.T)

    return V, A_component, U_S, U_A, U_rest


def make_orthogonal_projectors(
    base:           Path,
    stem:           str = "gpt2_large",
    d:              int = D_MODEL,
    subspace_block: int = DEFAULT_SUBSPACE_BLOCK,
    seed_basis:     int = SUBSPACE_FRAME_SEED,
) -> Path:
    """
    Write ov_projectors_{stem}.npz with DISJOINT, EQUAL-DIMENSION S and A.

    Replaces tests/test_phase5b_io._make_p2_projectors for Sub-exp D tests.
    That builder writes `U_pos = U[:, :k]` AND `U_A = U[:, :k]` — the same
    columns — so U_A sits inside span(U_S_full) and "S versus A" compares a
    space with a subspace of itself, at twice A's dimension. Regime A
    (structural fixture, never reaches Sub-exp D) still uses the original
    `_make_p2_projectors`; only the ground-truth regime needs this one.

    U_pos = Q[:, 0:k], U_neg = Q[:, k:2k], U_A = Q[:, 2k:4k]: U_S_full is
    2k, U_A is 2k, and the two are orthogonal.

    `seed_basis` MUST match the value `_subspace_basis` was called with, or
    these projectors will not describe the subspaces the activations were
    actually generated in — the default matches `make_ground_truth`'s
    default and there is no reason to pass a different one unless you are
    also passing a different `seed_basis` there.
    """
    Qrng = np.random.default_rng(seed_basis)
    Q, _ = np.linalg.qr(Qrng.standard_normal((d, d)))
    kb = subspace_block
    path = Path(base) / f"ov_projectors_{stem}.npz"
    np.savez_compressed(
        path,
        U_pos=Q[:, 0:kb].astype(np.float32),
        U_neg=Q[:, kb:2 * kb].astype(np.float32),
        U_A=Q[:, 2 * kb:4 * kb].astype(np.float32),
    )
    return path


# ---------------------------------------------------------------------------
# Generative core
# ---------------------------------------------------------------------------

def make_ground_truth(
    n_clusters:     int   = DEFAULT_N_CLUSTERS,
    d_model:        int   = D_MODEL,
    vocab:          int   = DEFAULT_VOCAB,
    n_tok_per:      int   = DEFAULT_TOK_PER,
    n_noise_tok:    int   = DEFAULT_NOISE_TOK,
    norm_spread:    float = 3.0,
    noise:          float = 0.05,
    sigma_frac:     float = 0.22,
    subspace_block: int   = DEFAULT_SUBSPACE_BLOCK,
    alpha:          float = DEFAULT_ALPHA,
    ambient:        float = DEFAULT_AMBIENT,
    behavior:       str   = "aligned",
    seed:           int   = 0,
) -> dict:
    """
    Build the latent world. Returns everything the on-disk writers need.

    Returns
    -------
    dict with:
      theta       : (n_clusters,) latent coordinate
      directions  : (n_clusters, d) unit vectors on a quarter-circle arc,
                    lying inside span(U_S)
      A_component : (n_clusters, k_a) fixed-norm offset inside span(U_A),
                    independent of theta — see "SUBSPACE STRUCTURE"
      U_S         : (d, 2*subspace_block) — Sub-exp D signal subspace
      U_A         : (d, 2*subspace_block) — Sub-exp D null subspace
      scales      : (n_clusters,) per-cluster norm, permuted w.r.t. theta
      labels      : (n_tokens,) int, -1 for noise tokens
      token_acts  : (n_tokens, d) RAW (un-normalized) activations
      cluster_p   : (n_clusters, vocab) per-cluster output distribution
      noise_p     : (vocab,) distribution for noise tokens
      n_tokens    : int

    `behavior`: "aligned" (isometry holds) | "shuffled" (it does not).
    """
    if behavior not in ("aligned", "shuffled"):
        raise ValueError(f"make_ground_truth: behavior must be 'aligned' or "
                         f"'shuffled', got {behavior!r}")
    if n_clusters != len(SCALE_PERMUTATION):
        raise ValueError(
            f"make_ground_truth: SCALE_PERMUTATION and BEHAVIOR_SHUFFLE are "
            f"fixed for n_clusters={len(SCALE_PERMUTATION)}. Change both "
            f"deliberately (and re-measure the separation) rather than "
            f"generating them randomly — see module docstring."
        )
    if 4 * subspace_block > d_model:
        raise ValueError(
            f"make_ground_truth: subspace_block={subspace_block} needs "
            f"4*subspace_block={4 * subspace_block} <= d_model={d_model} "
            f"(U_S, U_A, and a nonempty U_rest all draw from d_model)"
        )

    rng   = np.random.default_rng(seed)
    theta = np.linspace(0.0, 1.0, n_clusters)

    V, A_component, U_S, U_A, U_rest = _subspace_basis(
        rng, d_model, n_clusters, theta, subspace_block, alpha,
    )

    scales = np.geomspace(1.0 / norm_spread, norm_spread, n_clusters)[
        list(SCALE_PERMUTATION)
    ]

    acts, labels = [], []
    for i in range(n_clusters):
        base = V[i] + A_component[i]
        for _ in range(n_tok_per):
            amb = (rng.standard_normal(U_rest.shape[1]) @ U_rest.T) * ambient
            acts.append(scales[i] * (
                base + rng.standard_normal(d_model) * noise + amb
            ))
            labels.append(i)
    for _ in range(n_noise_tok):
        acts.append(rng.standard_normal(d_model) * 0.5)
        labels.append(-1)

    acts   = np.asarray(acts, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    order = (np.arange(n_clusters) if behavior == "aligned"
             else np.asarray(BEHAVIOR_SHUFFLE))
    centers = vocab * (0.1 + 0.8 * theta[order])
    sigma   = vocab * sigma_frac
    idx     = np.arange(vocab)
    P = np.exp(-0.5 * ((idx[None, :] - centers[:, None]) / sigma) ** 2) + 1e-9
    P /= P.sum(axis=1, keepdims=True)

    # Noise tokens get a distribution unlike any cluster's, so that a global
    # mean is visibly wrong rather than merely slightly off.
    noise_p = np.full(vocab, 1e-9)
    noise_p[vocab // 2] = 1.0
    noise_p /= noise_p.sum()

    return {
        "theta":       theta,
        "directions":  V,
        "A_component": A_component,
        "U_S":         U_S,
        "U_A":         U_A,
        "scales":      scales,
        "labels":      labels,
        "token_acts":  acts,
        "cluster_p":   P.astype(np.float32),
        "noise_p":     noise_p.astype(np.float32),
        "n_tokens":    int(acts.shape[0]),
        "n_clusters":  int(n_clusters),
        "vocab":       int(vocab),
        "behavior":    behavior,
    }


# ---------------------------------------------------------------------------
# On-disk Phase 1 run
# ---------------------------------------------------------------------------

def make_coherent_run_dir(
    base:         Path,
    stem:         str = "gpt2_large",
    prompt:       str = "wiki_paragraph",
    n_layers:     int = DEFAULT_N_LAYERS,
    chain_layers      = DEFAULT_CHAIN_LAYERS,
    plateau_layers    = DEFAULT_PLATEAU,
    merge_layers      = DEFAULT_MERGE,
    layer_jitter: float = 0.02,
    gt:           dict | None = None,
    **gt_kwargs,
) -> tuple[Path, dict]:
    """
    Write a Phase 1 v2 run directory whose artifacts are mutually consistent.

    Unlike `_make_run_dir`, `centroid_trajectories.npz` is COMPUTED from the
    activations and labels actually written to disk, exactly the way
    cluster_tracking.compute_centroid_trajectories would compute it
    (L2-normalize the layer's tokens, mask by cluster, mean). Previously it
    was independent random data, so the centroids the pipeline loaded
    described nothing that was in `activations.npz`.

    Returns (run_dir, ground_truth_dict).
    """
    gt = gt or make_ground_truth(**gt_kwargs)
    base = Path(base)
    run_dir = base / f"{stem}_{prompt}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = gt["n_clusters"]
    n_tokens   = gt["n_tokens"]
    d_model    = gt["token_acts"].shape[1]
    labels     = gt["labels"]
    chain_layers   = list(chain_layers)
    plateau_layers = list(plateau_layers)
    merge_layers   = list(merge_layers)

    rng = np.random.default_rng(12345)

    # --- activations.npz: RAW hidden states (per artifacts.py ArtifactSpec)
    acts = np.zeros((n_layers, n_tokens, d_model), dtype=np.float32)
    for L in range(n_layers):
        if L in chain_layers:
            jitter = rng.standard_normal((n_tokens, d_model)) * layer_jitter
            acts[L] = gt["token_acts"] + jitter
        else:
            # Layers outside the chain carry no cluster structure. Anything
            # that reads them expecting structure is reading the wrong layer.
            acts[L] = rng.standard_normal((n_tokens, d_model)) * 0.5
    np.savez_compressed(run_dir / "activations.npz", activations=acts)

    # --- hdbscan_labels.json: {layer_idx: [labels]}
    label_map = {}
    for L in range(n_layers):
        label_map[str(L)] = (labels.tolist() if L in chain_layers
                             else [-1] * n_tokens)
    (run_dir / "hdbscan_labels.json").write_text(json.dumps(label_map))

    # --- trajectory.json
    trajectories = [
        {"id": i, "chain": [[L, i] for L in chain_layers]}
        for i in range(n_clusters)
    ]
    (run_dir / "trajectory.json").write_text(json.dumps({
        "plateau_layers": plateau_layers,
        "cluster_tracking": {
            "trajectories": trajectories,
            "events": [{"layer_from": merge_layers[0] if merge_layers else 3,
                        "n_merges": len(merge_layers)}],
            "summary": {"n_trajectories": n_clusters},
        },
    }))

    # --- events.json
    (run_dir / "events.json").write_text(json.dumps({
        "merge_layers": merge_layers,
        "energy_violations": {"1.0": [2]},
    }))

    # --- geometry.json
    (run_dir / "geometry.json").write_text(json.dumps({
        "model":    stem.replace("_", "-"),
        "prompt":   prompt,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "d_model":  d_model,
    }))

    # --- centroid_trajectories.npz, derived from what is on disk above
    arrays = {}
    for i in range(n_clusters):
        rows = []
        for L in chain_layers:
            H = acts[L]
            Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
            mask = labels == i
            rows.append(Hn[mask].mean(axis=0))
        C = np.asarray(rows, dtype=np.float32)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        arrays[f"traj_{i}"] = C
    np.savez_compressed(run_dir / "centroid_trajectories.npz", **arrays)

    return run_dir, gt


def make_aligned_logit_cache(
    gt: dict,
    layer_idxs,
    jitter: float = 0.0,
    seed:   int = 7,
) -> dict:
    """
    Per-token output distributions consistent with the ground truth.

    Token in cluster i receives cluster i's distribution; noise tokens
    receive `noise_p`. A masked mean therefore recovers the cluster's
    distribution exactly, while a global mean over all tokens at a layer
    yields ONE vector shared by every cluster — which collapses all pairwise
    behavior distances to zero and makes the correlation undefined. That is
    the original bug, and it is what this fixture is built to expose.
    """
    rng = np.random.default_rng(seed)
    labels = gt["labels"]
    P      = gt["cluster_p"]
    noise_p = gt["noise_p"]
    vocab  = gt["vocab"]

    out = {}
    for L in layer_idxs:
        rows = np.zeros((len(labels), vocab), dtype=np.float64)
        for t, c in enumerate(labels):
            rows[t] = noise_p if c < 0 else P[c]
        if jitter > 0:
            rows = np.clip(rows + rng.standard_normal(rows.shape) * jitter, 1e-12, None)
        rows /= rows.sum(axis=1, keepdims=True)
        out[int(L)] = rows.astype(np.float32)
    return out
