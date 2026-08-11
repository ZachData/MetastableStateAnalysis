"""
p1c_frames/frame_table.py — sub-experiment D: is the sphere the right
manifold, and what does the learned LN affine do to every metric on it?

THE PAPER'S OWN CRITERION

Section 2.2 does not assume the sphere. It MEASURES it. RMS-norm multiplies
by a trained diagonal matrix, so the true state space is a time-varying
axis-aligned ellipsoid; the paper sets that matrix to I and justifies it
empirically — in ALBERT XLarge v2 the diagonal is essentially constant
across layers, mean 0.44, sd 0.008.

That is a reproducible measurement, and this module runs it on Pythia. Which
reframes what core/ln_frame.py is for: it has been described in this project
as a departure from the paper's frame. It is the opposite. It is the paper's
own licensing check, run on a model where it may fail.

If Pythia's gamma has wide dynamic range per layer, the correct manifold is
the ellipsoid and EVERY sphere-frame metric in Phase 1 inherits a
distortion — ip_mean, ip_mass_near_1, effective rank, and the interaction
energy alike.

TWO SPECIFIC QUESTIONS

1. What is the dynamic range of Pythia's gamma per layer, against ALBERT's
   mean 0.44 / sd 0.008? `gamma_dynamic_range` answers it directly and
   `sphere_license` adjudicates against the paper's own benchmark.

2. Does zeroing the learned LN bias remove the energy floor? The bias adds
   a FIXED vector to every token — pure common mode. It inflates <G> by
   roughly ||beta_LN||^2 regardless of input, and <G>/2 is the dominant
   term in the small-beta expansion of E_beta. So the learned LN bias puts
   a floor under the interaction energy that has nothing to do with the
   tokens, and every absolute energy number Phase 1 reports sits on top of
   it. `bias_energy_floor` isolates it by recomputing with beta_LN = 0.

FOUR FRAMES

    l2          the existing sphere frame: x / ||x||
    ln_plain    LayerNorm with gamma=1, beta=0. EXACTLY sphere projection
                in the mean-zero subspace: LN(x) = sqrt(d) * P_1 x / ||P_1 x||,
                constant norm sqrt(d). So this frame structurally restores
                uniform token weights and removes the sink domination that
                status-1 defect D10 identifies in raw effective rank — a
                reason to prefer it that has nothing to do with fidelity to
                the paper.
    ln_learned  gamma and beta_LN as trained. What attention actually reads.
    functional  Torgerson double-centering of the symmetrized-KL matrix,
                G = -1/2 J D^2 J. The functional frame has no Gram of its
                own; double-centering produces one, after which every
                moment identity applies unchanged.

COST: [W]. Needs LN weights, no forward pass. The functional frame
additionally needs saved logits/probabilities.
"""

from __future__ import annotations

import numpy as np

from core.ln_frame import ln_frame_gram, ln_transform
from core.metrics import (
    l2_normalize, gram_cumulants, interaction_energies_batched,
    pairwise_upper, effective_rank,
)


# ALBERT XLarge v2, from the paper's sec. 2.2. The benchmark the sphere
# assumption is licensed against.
ALBERT_GAMMA_MEAN = 0.44
ALBERT_GAMMA_SD = 0.008


# ---------------------------------------------------------------------------
# Question 1: is the sphere licensed?
# ---------------------------------------------------------------------------

def gamma_dynamic_range(gamma) -> dict:
    """
    Dispersion statistics for one layer's LN gain vector.

    The paper's criterion is dispersion ACROSS CHANNELS within a layer —
    a constant diagonal is a uniform rescaling of the sphere and changes
    nothing, while a spread-out one is a genuine ellipsoid. So the number
    that matters is the coefficient of variation, not the mean: a gamma of
    all 0.44 and a gamma of all 4.4 both leave the manifold a sphere.

    `condition_number` (max/min) is reported alongside because it is what
    actually bounds the metric distortion: an ellipsoid with axis ratio k
    can move a cosine by up to a factor of k^2 in the worst case, so a
    condition number near 1 licenses the sphere and a large one does not,
    regardless of how small the sd looks next to a mean of 0.44.
    """
    g = np.asarray(gamma, dtype=np.float64).ravel()
    mean = float(g.mean())
    sd = float(g.std())
    amin, amax = float(np.abs(g).min()), float(np.abs(g).max())
    return {
        "mean": mean,
        "sd": sd,
        "cv": float(sd / abs(mean)) if abs(mean) > 1e-12 else float("inf"),
        "min": float(g.min()),
        "max": float(g.max()),
        "abs_min": amin,
        "abs_max": amax,
        "condition_number": float(amax / amin) if amin > 1e-12 else float("inf"),
        "n_channels": int(g.size),
        "n_negative": int((g < 0).sum()),
    }


def sphere_license(gamma_stats_by_layer: list) -> dict:
    """
    Adjudicate the sphere assumption against the paper's ALBERT benchmark.

    gamma_stats_by_layer : list of gamma_dynamic_range outputs, one per block

    ALBERT's cv is 0.008 / 0.44 = 0.018 — under two percent. That is what
    "essentially constant" means and what licenses setting the diagonal to
    I. The verdict below compares Pythia's worst layer against it.

    Also checks CONSTANCY ACROSS LAYERS, which is the second half of the
    paper's observation ("constant across layers") and a separate
    condition: a model whose gamma is uniform within each layer but
    different between layers is on a sphere at every depth, but on a
    DIFFERENT sphere at each, so cross-layer trajectory metrics still
    inherit a rescaling.
    """
    if not gamma_stats_by_layer:
        return {"verdict": "no layers supplied"}
    cvs = np.array([s["cv"] for s in gamma_stats_by_layer])
    conds = np.array([s["condition_number"] for s in gamma_stats_by_layer])
    means = np.array([s["mean"] for s in gamma_stats_by_layer])

    albert_cv = ALBERT_GAMMA_SD / ALBERT_GAMMA_MEAN
    worst_cv = float(np.nanmax(cvs))
    ratio = worst_cv / albert_cv

    if ratio < 2.0:
        verdict = ("SPHERE LICENSED — gamma dispersion is within 2x ALBERT's, "
                   "so the paper's own justification transfers and the "
                   "sphere-frame metrics stand as reported.")
    elif ratio < 10.0:
        verdict = (f"MARGINAL — gamma dispersion is {ratio:.1f}x ALBERT's. The "
                   f"sphere is an approximation here, not a licensed one. "
                   f"Report sphere-frame and LN-frame metrics side by side "
                   f"rather than choosing.")
    else:
        verdict = (f"SPHERE NOT LICENSED — gamma dispersion is {ratio:.1f}x "
                   f"ALBERT's. The correct manifold is the ellipsoid, and "
                   f"every sphere-frame metric in Phase 1 inherits a "
                   f"distortion. The LN frame is not an alternative reading; "
                   f"it is the correct one.")

    return {
        "verdict": verdict,
        "worst_layer_cv": worst_cv,
        "worst_layer": int(np.nanargmax(cvs)),
        "median_cv": float(np.nanmedian(cvs)),
        "albert_cv": float(albert_cv),
        "cv_ratio_to_albert": float(ratio),
        "max_condition_number": float(np.nanmax(conds)),
        # Second half of the paper's observation: constant ACROSS layers.
        "cross_layer_mean_cv": float(np.std(means) / abs(np.mean(means)))
                               if abs(np.mean(means)) > 1e-12 else float("inf"),
        "albert_reference": {"mean": ALBERT_GAMMA_MEAN, "sd": ALBERT_GAMMA_SD},
        "n_layers": len(gamma_stats_by_layer),
    }


# ---------------------------------------------------------------------------
# Question 2: the LN bias energy floor
# ---------------------------------------------------------------------------

def bias_energy_floor(X, gamma, beta_ln, betas=(0.1, 1.0, 2.0, 5.0),
                      eps: float = 1e-5) -> dict:
    """
    How much of the measured interaction energy is the learned LN bias?

    Recomputes the LN-frame Gram with beta_LN as trained and with it
    zeroed, and reports both energies and both first cumulants. The gap is
    the floor.

    Two diagnostics that separate the mechanism from the magnitude:

      bias_norm_ratio   ||beta_LN|| / mean ||gamma * xhat||. The bias is a
                        common mode; how much it moves the geometry depends
                        entirely on its size relative to the signal it is
                        added to, and a small-norm bias on a small-norm
                        signal is a large effect.
      kappa1_shift      the change in <G>. This is the actual mechanism —
                        <G>/2 is the dominant term of E_beta at small beta,
                        so an energy shift with no kappa1 shift would mean
                        something other than common mode is responsible.
    """
    X = np.asarray(X, dtype=np.float64)
    betas = [float(b) for b in betas]

    G_with = ln_frame_gram(X, gamma=gamma, beta=beta_ln, eps=eps)
    G_without = ln_frame_gram(X, gamma=gamma, beta=None, eps=eps)

    E_with = interaction_energies_batched(G_with, betas)
    E_without = interaction_energies_batched(G_without, betas)
    c_with = gram_cumulants(G_with)
    c_without = gram_cumulants(G_without)

    Y_nobias = ln_transform(X, gamma=gamma, beta=None, eps=eps)
    sig = float(np.linalg.norm(Y_nobias, axis=1).mean())
    bn = float(np.linalg.norm(np.asarray(beta_ln, dtype=np.float64))) if beta_ln is not None else 0.0

    return {
        "energies_with_bias": E_with,
        "energies_without_bias": E_without,
        "energy_floor": {b: float(E_with[b] - E_without[b]) for b in betas},
        "energy_floor_frac": {
            b: float((E_with[b] - E_without[b]) / E_with[b])
            if abs(E_with[b]) > 1e-15 else float("nan") for b in betas
        },
        "kappa1_with": c_with["kappa1"],
        "kappa1_without": c_without["kappa1"],
        "kappa1_shift": float(c_with["kappa1"] - c_without["kappa1"]),
        "bias_norm": bn,
        "signal_norm": sig,
        "bias_norm_ratio": float(bn / sig) if sig > 1e-12 else float("inf"),
        "ip_mean_with": float(pairwise_upper(G_with).mean()),
        "ip_mean_without": float(pairwise_upper(G_without).mean()),
    }


# ---------------------------------------------------------------------------
# The four-frame table
# ---------------------------------------------------------------------------

def torgerson_gram(D: np.ndarray) -> np.ndarray:
    """
    G = -1/2 J D^2 J, the classical (Torgerson) double-centering that turns
    a distance matrix into a Gram matrix.

    D is the symmetrized-KL distance matrix from
    core/functional_distance.py. Symmetric KL is not a metric — it violates
    the triangle inequality — so the resulting G is not guaranteed PSD, and
    negative eigenvalues are expected rather than a bug. `frame_moments`
    reports the negative-eigenvalue mass so the size of the violation is
    visible; a frame whose Gram is 40% negative mass is not one in which
    "effective rank" means what it means elsewhere.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D ** 2) @ J


def frame_moments(G: np.ndarray, betas=(0.1, 1.0, 2.0, 5.0),
                  normalize: bool = True) -> dict:
    """
    The moment ladder for one frame's Gram, so all four are on one axis.

    normalize : rescale to unit diagonal first. Required for the l2 and LN
                frames to be comparable (both are already unit-diagonal);
                for the Torgerson frame the diagonal is not 1 and rescaling
                is what makes <G> a cosine rather than an inner product.
                Rows with non-positive diagonal (possible under the
                non-PSD Torgerson Gram) are excluded and counted.
    """
    G = np.asarray(G, dtype=np.float64)
    dropped = 0
    if normalize:
        dg = np.diag(G).copy()
        keep = dg > 1e-12
        dropped = int((~keep).sum())
        G = G[np.ix_(keep, keep)]
        s = np.sqrt(np.diag(G))
        G = G / np.outer(s, s)
        G = np.clip(G, -1.0, 1.0)

    evals = np.linalg.eigvalsh(G)
    neg_mass = float(np.abs(evals[evals < 0]).sum() / max(np.abs(evals).sum(), 1e-30))

    c = gram_cumulants(G)
    return {
        "kappa1": c["kappa1"], "kappa2": c["kappa2"], "kappa3": c["kappa3"],
        "pr_rank": c["pr_rank"],
        "ip_mean": float(pairwise_upper(G).mean()),
        "energies": interaction_energies_batched(G, [float(b) for b in betas]),
        "neg_eigen_mass": neg_mass,
        "n_dropped_rows": dropped,
        "n": int(G.shape[0]),
    }


def frame_table(X, gamma=None, beta_ln=None, kl_matrix=None,
                betas=(0.1, 1.0, 2.0, 5.0), eps: float = 1e-5) -> dict:
    """
    The moment ladder in all four frames for one layer.

    X         : (n, d) raw activations for this layer
    gamma     : learned LN gain, or None to skip the ln_learned frame
    beta_ln   : learned LN bias
    kl_matrix : (n, n) symmetric-KL distances, or None to skip `functional`

    Reported side by side rather than reduced to a preferred frame. Which
    frame is correct is the question sub-experiment D exists to answer; a
    function that answered it by picking one would be assuming its
    conclusion.
    """
    X = np.asarray(X, dtype=np.float64)
    out = {}

    Xn = l2_normalize(X)
    out["l2"] = frame_moments(Xn @ Xn.T, betas=betas)
    out["l2"]["raw_effective_rank"] = float(effective_rank(X, mode="raw"))

    out["ln_plain"] = frame_moments(
        ln_frame_gram(X, gamma=None, beta=None, eps=eps), betas=betas)
    # Plain LN gives every token norm sqrt(d) exactly, which is the
    # structural claim in the module docstring. Verified rather than
    # asserted, since it is the reason this frame removes sink domination.
    Y = ln_transform(X, gamma=None, beta=None, eps=eps)
    norms = np.linalg.norm(Y, axis=1)
    out["ln_plain"]["norm_cv"] = float(norms.std() / max(norms.mean(), 1e-12))
    out["ln_plain"]["norm_mean_over_sqrt_d"] = float(
        norms.mean() / np.sqrt(X.shape[1]))

    if gamma is not None:
        out["ln_learned"] = frame_moments(
            ln_frame_gram(X, gamma=gamma, beta=beta_ln, eps=eps), betas=betas)
        Yl = ln_transform(X, gamma=gamma, beta=beta_ln, eps=eps)
        nl = np.linalg.norm(Yl, axis=1)
        out["ln_learned"]["norm_cv"] = float(nl.std() / max(nl.mean(), 1e-12))
        out["gamma_stats"] = gamma_dynamic_range(gamma)
        if beta_ln is not None:
            out["bias_floor"] = bias_energy_floor(X, gamma, beta_ln,
                                                  betas=betas, eps=eps)

    if kl_matrix is not None:
        out["functional"] = frame_moments(torgerson_gram(kl_matrix), betas=betas)

    return out


def frame_disagreement(table: dict, key: str = "ip_mean") -> dict:
    """
    How much does the choice of frame move a reported quantity?

    This is the number that decides whether frame choice is a footnote or a
    confound. If ip_mean varies by 0.02 across frames it is a footnote; if
    it varies by 0.4 then every Phase 1 statement about clustering is a
    statement about the l2 frame specifically and must be labelled as one.
    """
    vals = {f: t[key] for f, t in table.items()
            if isinstance(t, dict) and key in t}
    if len(vals) < 2:
        return {"key": key, "n_frames": len(vals), "spread": float("nan"),
                "values": vals}
    arr = np.array(list(vals.values()), dtype=np.float64)
    return {
        "key": key,
        "values": vals,
        "spread": float(arr.max() - arr.min()),
        "max_frame": max(vals, key=vals.get),
        "min_frame": min(vals, key=vals.get),
        "n_frames": len(vals),
    }
