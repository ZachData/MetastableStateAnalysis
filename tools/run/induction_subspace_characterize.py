"""Stage 2 of the bottom-up induction programme (PROJECT.md sec 3.11):
characterise the r* subspace of induction head L7H8's OV, and calibrate every
number against control heads so "L7H8's rank-1 mode carries X%" has a
reference distribution.

This REPLACES an earlier inline Stage 2 that was invalid: its copying /
token-alignment readout was a direct logit attribution computed WITHOUT the
final layer norm, and it had no control-head calibration. Both are fixed here.

WHAT IS AND IS NOT LN-SENSITIVE
------------------------------
- The pure-weights spectral part (Schur sign of the top eigenvalues, phi =
  ||A||_F^2 / ||OV||_F^2, Henrici departure from normality, the rank-1 SVD
  mode's self-eigenvalue) has NO layer-norm dependence. Reported directly.
- The COPYING effect is read the only trustworthy way: causally, through the
  full forward pass, where `gpt_neox.final_layer_norm` is present by
  construction -- identical readout to Stage 1's sweep. We ablate the head's
  OV to {full, rank-1 SVD, rank-1 Schur} and measure the change in
  second-copy NLL and the KL from the unablated model.
- The DIRECT token-subspace-alignment number (Olsson-style copy score of the
  mode) is descriptive only and is computed with the mean per-position LN
  scale folded into both the head-input LN and the final LN, measured on the
  real second-copy positions. Marked `approx_` and never used for a claim.

CALIBRATION SET
---------------
- all 16 heads of layer 7 (the target's own layer), and
- N_RANDOM matched-Frobenius-norm random rank-64 OV operators at the real
  1024/64 geometry.
L7H8's statistics are reported as a rank within the layer-7 set and a z-score
against the random set.

Weights are only ever touched through a save/restore around each measurement;
the baseline is re-measured at the end and asserted unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np
import scipy
import scipy.linalg as sla
import torch
import transformers

from core.lm_loading import load_causal_lm
from p2b_imaginary import head_circuits as hc
from tools.run.induction_rank_sweep import (
    D_HEAD, D_MODEL, N_HEADS, LAYER, HEAD, PREV_LAYER, PREV_HEAD, STEP,
    N_REP, N_SEQS, EVAL_SEED, VOCAB_LO, VOCAB_HI,
    induction_batch, ov_factors, write_ov, truncate,
)

N_RANDOM = 24
VOCAB_SAMPLE = 4000           # tokens sampled for the direct copy score
R_STAR_SCHUR = 16            # Stage 1: rank at which Schur matches rank-1 SVD


# --------------------------------------------------------------------------
# pure-weights spectral characterisation (no layer norm anywhere)
# --------------------------------------------------------------------------

def spectral_stats(A: np.ndarray, B: np.ndarray) -> dict:
    """`A (d,64)`, `B (64,d)`, `OV = A @ B`. Core `C = B @ A` (64x64)."""
    C = B @ A
    ev = np.linalg.eigvals(C)
    order = np.argsort(-np.abs(ev))
    ev = ev[order]
    fro2 = float(np.sum(np.abs(C) ** 2))
    henrici = float(np.sqrt(max(fro2 - float(np.sum(np.abs(ev) ** 2)), 0.0))
                    / np.sqrt(fro2))
    hs = hc.head_spectrum(A, B)

    # rank-1 SVD mode and its only nonzero eigenvalue, s1 * (v1 . u1).
    U, s, Vt = np.linalg.svd(A @ B, full_matrices=False)
    u1, v1 = U[:, 0], Vt[0]
    lam1_svd = complex(s[0] * (v1 @ u1))

    # top-r*_schur invariant subspace: its eigenvalues are exactly C's top r*.
    ev_schur_top = ev[:R_STAR_SCHUR]

    return {
        "ov_fro": float(np.sqrt(np.sum((A @ B) ** 2))),
        "henrici": henrici,
        "phi_antisym_fro_fraction": float(hs["rotational_frobenius_fraction"]),
        "attractive_energy_fraction_core": float(hs["attractive_energy_fraction_core"]),
        "repulsive_energy_fraction_core": float(hs["repulsive_energy_fraction_core"]),
        "complex_energy_fraction_core": float(hs["complex_energy_fraction_core"]),
        "spectral_radius": float(np.abs(ev[0])),
        "lambda_top_re": float(ev[0].real),
        "lambda_top_abs": float(np.abs(ev[0])),
        "lambda_top_sign": ("repulsive" if ev[0].real < 0
                            else "attractive" if ev[0].real > 0 else "neutral"),
        "svd_s1": float(s[0]),
        "svd_s1_share_of_fro": float(s[0] ** 2 / max(np.sum(s ** 2), 1e-300)),
        "svd_mode_lambda_re": float(lam1_svd.real),
        "svd_mode_lambda_abs": float(np.abs(lam1_svd)),
        "svd_mode_sign": ("repulsive" if lam1_svd.real < 0
                          else "attractive" if lam1_svd.real > 0 else "neutral"),
        "schur_rstar_re_mean": float(np.mean(ev_schur_top.real)),
        "schur_rstar_frac_repulsive": float(np.mean(ev_schur_top.real < 0)),
        "schur_rstar_energy_repulsive": float(
            np.sum(np.abs(ev_schur_top[ev_schur_top.real < 0]) ** 2)
            / max(np.sum(np.abs(ev_schur_top) ** 2), 1e-300)),
    }


# --------------------------------------------------------------------------
# LN scales, measured on the real second-copy positions
# --------------------------------------------------------------------------

def _ln_inv_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm's 1/sqrt(var+eps) factor, per position (last dim reduced)."""
    v = x.float().var(dim=-1, unbiased=False, keepdim=True)
    return (v + eps).rsqrt().squeeze(-1)


@torch.no_grad()
def measure_ln_scales(model, ids) -> dict:
    """Mean of LN's per-position inverse-RMS scale over the second-copy
    positions, for the head's input LN (layer LAYER) and the final LN."""
    caught = {}
    h1 = model.gpt_neox.layers[LAYER].input_layernorm.register_forward_pre_hook(
        lambda m, a: caught.__setitem__("in", a[0].detach()))
    h2 = model.gpt_neox.final_layer_norm.register_forward_pre_hook(
        lambda m, a: caught.__setitem__("final", a[0].detach()))
    try:
        model(ids)
    finally:
        h1.remove(); h2.remove()
    eps = model.gpt_neox.final_layer_norm.eps
    sl = slice(N_REP - 1, None)               # second-copy positions
    s_in = _ln_inv_scale(caught["in"][:, sl, :], eps)
    s_final = _ln_inv_scale(caught["final"][:, sl, :], eps)
    g_in = model.gpt_neox.layers[LAYER].input_layernorm.weight.detach().float().numpy()
    g_final = model.gpt_neox.final_layer_norm.weight.detach().float().numpy()
    return {
        "s_in_mean": float(s_in.mean()), "s_in_sd": float(s_in.std()),
        "s_final_mean": float(s_final.mean()), "s_final_sd": float(s_final.std()),
        "gamma_in": g_in, "gamma_final": g_final,
    }


def approx_copy_score(A, B, r, basis, lns, W_E, W_U, tok_ids, rng) -> dict:
    """Descriptive Olsson-style copy score of the rank-r mode, LN folded as
    mean scales. `M[t,t'] = logit on t' from feeding token t through
    LN_in -> OV_mode -> LN_final -> W_U`. Copier => diagonal dominant.

    Approximate: LN mean-subtraction is applied as row/col centring, the
    RMS factor as the measured mean scale. Never used for a claim.
    """
    A_r, B_r = truncate(A, B, r, basis, rng)
    OV_mode = (A_r @ B_r).astype(np.float64)             # (d, d)
    g_in, g_f = lns["gamma_in"], lns["gamma_final"]
    Ein = (W_E[tok_ids] - W_E[tok_ids].mean(1, keepdims=True))
    Ein = Ein * lns["s_in_mean"] * g_in                  # (V, d), post-LN_in
    w = Ein @ OV_mode                                    # (V, d), head write
    w = (w - w.mean(1, keepdims=True)) * lns["s_final_mean"] * g_f
    M = w @ W_U[tok_ids].T                               # (V, V) logit table
    diag = np.diag(M)
    off_mean = (M.sum(1) - diag) / (M.shape[1] - 1)
    off_sd = M.std(1)
    z = (diag - off_mean) / np.where(off_sd > 0, off_sd, 1.0)
    return {
        "approx_copy_diag_z_mean": float(np.mean(z)),
        "approx_copy_frac_diag_is_rowmax": float(np.mean(diag == M.max(1))),
        "approx_copy_frac_diag_positive": float(np.mean(diag > 0)),
    }


# --------------------------------------------------------------------------
# causal copying readout (full forward, LN present) -- identical to Stage 1
# --------------------------------------------------------------------------

@torch.no_grad()
def second_copy_nll_and_lp(model, ids):
    out = model(ids)
    logits = out.logits[:, :-1, :].float()
    lp = torch.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(-tok_lp[:, N_REP - 1:].mean()), lp


def causal_readout(model, layer, head, A0, B0, ids, base_lp, rng) -> dict:
    """Fraction of the head's own OV copy-effect carried by its rank-1 mode,
    in each basis. `frac = (nll_r0 - nll_r1) / (nll_r0 - nll_base)`."""
    base_nll = float(-base_lp.gather(
        -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, N_REP - 1:].mean())

    def _abl(A, B):
        write_ov(model, layer, head, A, B)
        nll, lp = second_copy_nll_and_lp(model, ids)
        write_ov(model, layer, head, A0, B0)
        kl = float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())
        return nll, kl

    nll0, kl0 = _abl(*truncate(A0, B0, 0, "svd", rng))
    nll_s1, kl_s1 = _abl(*truncate(A0, B0, 1, "svd", rng))
    nll_h1, kl_h1 = _abl(*truncate(A0, B0, 1, "schur", rng))
    denom = nll0 - base_nll
    return {
        "base_second_copy_nll": base_nll,
        "full_ablation_nll": nll0, "full_ablation_kl": kl0,
        "full_ablation_delta_nll": nll0 - base_nll,
        "svd_r1_frac_of_effect": float((nll0 - nll_s1) / denom) if abs(denom) > 1e-9 else float("nan"),
        "schur_r1_frac_of_effect": float((nll0 - nll_h1) / denom) if abs(denom) > 1e-9 else float("nan"),
        "svd_r1_kl": kl_s1, "schur_r1_kl": kl_h1,
        "effect_denom": denom,
    }


def random_matched_ov(fro: float, rng) -> tuple:
    """A random rank-64 OV at the 1024/64 geometry with Frobenius norm `fro`."""
    A = rng.normal(size=(D_MODEL, D_HEAD))
    B = rng.normal(size=(D_HEAD, D_MODEL))
    cur = np.sqrt(np.sum((A @ B) ** 2))
    return A * np.sqrt(fro / cur), B * np.sqrt(fro / cur)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    args = ap.parse_args()

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    rng = np.random.default_rng(EVAL_SEED)

    model, tok = load_causal_lm(f"pythia-410m-step{STEP}")
    model.eval()
    ids = induction_batch(np.random.default_rng(EVAL_SEED))

    lns = measure_ln_scales(model, ids)
    print(f"  LN scales  s_in {lns['s_in_mean']:.3f}+-{lns['s_in_sd']:.3f}   "
          f"s_final {lns['s_final_mean']:.4f}+-{lns['s_final_sd']:.4f}", flush=True)

    W_E = model.gpt_neox.embed_in.weight.detach().float().numpy()
    W_U = model.embed_out.weight.detach().float().numpy()
    tok_ids = rng.integers(VOCAB_LO, VOCAB_HI, size=VOCAB_SAMPLE)

    _, base_lp = second_copy_nll_and_lp(model, ids)

    out = {
        "_what_this_is":
            "Stage 2 of PROJECT.md sec 3.11: characterisation of induction "
            "head L7H8's OV r* subspace (Schur sign, phi, Henrici, rank-1 "
            "mode), calibrated against all 16 layer-7 heads and matched-norm "
            "random heads. Copying effect is read causally through the full "
            "forward (final layer norm present); the direct copy score folds "
            "the mean LN scale and is descriptive only. Replaces an earlier "
            "inline Stage 2 that omitted the final layer norm and had no "
            "control heads. Exploratory: nothing registered.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
        "target": {"layer": LAYER, "head": HEAD,
                   "prev_token_partner": [PREV_LAYER, PREV_HEAD], "step": STEP},
        "params": {"r_star_schur": R_STAR_SCHUR, "n_random": args.n_random,
                   "vocab_sample": VOCAB_SAMPLE, "eval_seed": EVAL_SEED,
                   "n_rep": N_REP, "n_seqs": N_SEQS},
        "ln_scales": {k: v for k, v in lns.items() if not k.startswith("gamma")},
        "layer7_heads": {},
        "random_heads": [],
    }

    for h in range(N_HEADS):
        A, B = ov_factors(model, LAYER, h)
        rec = spectral_stats(A, B)
        rec.update(approx_copy_score(A, B, 1, "svd", lns, W_E, W_U, tok_ids, rng))
        rec.update(causal_readout(model, LAYER, h, A, B, ids, base_lp, rng))
        out["layer7_heads"][str(h)] = rec
        tag = "  <-- TARGET" if h == HEAD else ""
        print(f"  L7H{h:<2} top-lambda {rec['lambda_top_sign']:>10} "
              f"(Re {rec['lambda_top_re']:+.3f})  henrici {rec['henrici']:.3f}  "
              f"phi {rec['phi_antisym_fro_fraction']:.3f}  "
              f"svd-r1 {rec['svd_r1_frac_of_effect']:+.2f}  "
              f"schur-r1 {rec['schur_r1_frac_of_effect']:+.2f}  "
              f"|full dNLL| {rec['full_ablation_delta_nll']:+.4f}{tag}", flush=True)

    tgt_fro = out["layer7_heads"][str(HEAD)]["ov_fro"]
    for i in range(args.n_random):
        A, B = random_matched_ov(tgt_fro, rng)
        # random OV is calibrated by weights only -- phi / henrici / sign
        out["random_heads"].append(spectral_stats(A, B))

    # --- L7H8 vs the two reference sets -----------------------------------
    tgt = out["layer7_heads"][str(HEAD)]
    def _rank_in_layer(key, want_high=True):
        vals = [out["layer7_heads"][str(h)][key] for h in range(N_HEADS)]
        r = int(sum(v > tgt[key] for v in vals)) if want_high else \
            int(sum(v < tgt[key] for v in vals))
        return {"value": tgt[key], "rank_in_layer7": r, "of": N_HEADS,
                "layer7_mean": float(np.mean(vals)), "layer7_sd": float(np.std(vals))}
    def _z_vs_random(key):
        vals = np.array([r[key] for r in out["random_heads"]])
        return {"value": tgt[key], "random_mean": float(vals.mean()),
                "random_sd": float(vals.std()),
                "z": float((tgt[key] - vals.mean()) / (vals.std() or 1.0))}
    out["target_vs_reference"] = {
        "phi_antisym_fro_fraction": {**_rank_in_layer("phi_antisym_fro_fraction"),
                                     **{"vs_random": _z_vs_random("phi_antisym_fro_fraction")}},
        "henrici": {**_rank_in_layer("henrici"),
                    **{"vs_random": _z_vs_random("henrici")}},
        "attractive_energy_fraction_core": {
            **_rank_in_layer("attractive_energy_fraction_core"),
            **{"vs_random": _z_vs_random("attractive_energy_fraction_core")}},
        "svd_s1_share_of_fro": _rank_in_layer("svd_s1_share_of_fro"),
        "svd_r1_frac_of_effect": _rank_in_layer("svd_r1_frac_of_effect"),
        "schur_r1_frac_of_effect": _rank_in_layer("schur_r1_frac_of_effect"),
        "full_ablation_delta_nll": _rank_in_layer("full_ablation_delta_nll"),
        "approx_copy_diag_z_mean": _rank_in_layer("approx_copy_diag_z_mean"),
    }

    final_nll, _ = second_copy_nll_and_lp(model, ids)
    base_nll = tgt["base_second_copy_nll"]
    out["restore_check"] = {"baseline": base_nll, "after": final_nll,
                            "abs_diff": abs(base_nll - final_nll)}
    print(f"  restore check: {out['restore_check']['abs_diff']:.2e}")

    dest = Path(args.out) if args.out else DATA / "analysis" / "induction_subspace_characterize.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
