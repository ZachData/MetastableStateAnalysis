"""The two §3.12 checks that need a model: K-composition L5H2 -> L7H8 across the
axis, and whether the top OV direction survives input whitening.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-E1 and §3.12-C3. Both are preconditions on the Stage 3
entry and neither produces a p-value.

**Composition (§3.12-E1).** Induction is a two-stage circuit `L5H2 -> L7H8` and
only the ENDPOINTS have been characterised. A grep of the repo confirms no
K/Q/V-composition score exists anywhere -- the only "composition" hits are the
retired `relay` motif. The circuit-level object is how much of `L7H8`'s
key-side input arrives through `L5H2`'s OV:

    K-composition(A -> B) = ||W_K^B W_OV^A||_F / (||W_K^B||_F ||W_OV^A||_F)

Q- and V-composition are computed as the WITHIN-HEAD controls: induction is a
K-composition story, so a circuit that is real should show K elevated and Q, V
not. And every head in layers 0..6 is scored into `L7H8` as the POPULATION
control, so "elevated" is against the model's own distribution rather than
against a placed number.

Reported across the whole 19-step axis, which is the developmental question
§3.12-E1 actually asks: does the COMPOSITION form when the two halves form, or
before, or after? `PROJECT.md` §3.12-F2 puts a violent rotation of `L7H8`'s OV
plane at 512-2000 and §3.12-A puts the population repulsive collapse in the
same window; whether the composition also moves there is the point.

**Whitening (§3.12-C3).** The top singular direction of `W_OV` is identified
from the WEIGHTS ALONE and ignores the residual-stream distribution entirely.
Stage 3 perturbs that object, so if it is not also the top direction in the
metric the data actually occupies, the 82% is about a weight artifact. The
whitened operator is `M Sigma^{1/2}` for `Sigma` the residual covariance at the
head's input; a unit direction `w` there pulls back to input direction
`Sigma^{1/2} w`. §3.12-F2 found the stable object is the top-2 SUBSPACE, so
both the top-1 overlap and the top-2 principal angles are reported.

There is no cached pythia residual stream: `/run/media/system/HDD_1TB/
activation_cache` holds 355 GB of `gpt2-large` and `albert-xlarge-v2` only
(Phase 6 / Blog-1). Hence the forward pass here.

COST. One model load per step for composition (weights only after that), plus
one forward pass at the whitening step. `--steps` restricts the axis.
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
_want = str(REPO / ".venv")
if not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_HEAD, D_MODEL, N_HEADS, EVAL_SEED, induction_batch,
)

ROTARY_NDIMS = 16
OUT = DATA / "analysis" / "induction_composition_whitening.json"

SRC_LAYER, SRC_HEAD = 5, 2          # prev-token head
DST_LAYER, DST_HEAD = 7, 8          # matcher / copier
ALL_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000,
             4000, 8000, 16000, 32000, 54000, 143000]


def _ov_npz(step: int) -> Path:
    hits = sorted(DATA.glob(f"phase12/**/ov_weights_pythia-410m-step{step}.npz"))
    if not hits:
        raise FileNotFoundError(f"no ov_weights npz for step {step}")
    return hits[-1]


def _load_ov(step: int, layer: int, head: int) -> np.ndarray:
    with np.load(_ov_npz(step)) as z:
        return np.asarray(z[f"ov_head{head}_layer_{layer}"], dtype=np.float64)


def qkv_head(model, layer: int, head: int):
    """(W_Q, W_K, W_V) each (64, 1024) for one head."""
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    q3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    return (q3[head, 0:D_HEAD, :], q3[head, D_HEAD:2 * D_HEAD, :],
            q3[head, 2 * D_HEAD:3 * D_HEAD, :])


def comp_score(W_read: np.ndarray, OV: np.ndarray) -> float:
    """||W_read @ OV||_F / (||W_read||_F ||OV||_F). Elhage et al. composition."""
    num = np.linalg.norm(W_read @ OV)
    den = np.linalg.norm(W_read) * np.linalg.norm(OV)
    return float(num / max(den, 1e-300))


def composition_at_step(model, step: int) -> dict:
    OV_src = _load_ov(step, SRC_LAYER, SRC_HEAD)
    W_Q, W_K, W_V = qkv_head(model, DST_LAYER, DST_HEAD)

    row = {
        "step": step,
        "K": comp_score(W_K, OV_src),
        "Q": comp_score(W_Q, OV_src),
        "V": comp_score(W_V, OV_src),
        "K_static": comp_score(W_K[ROTARY_NDIMS:, :], OV_src),
    }

    # Population control for ALL THREE read paths, over every head in layers
    # 0..DST_LAYER-1. H1's caveat was that K and Q rise together with no control
    # on Q or V; each arm now gets its own 112-head distribution, so "elevated"
    # is judged per path rather than only for K.
    popK, popQ, popV = [], [], []
    for L in range(DST_LAYER):
        for h in range(N_HEADS):
            ov = _load_ov(step, L, h)
            popK.append(comp_score(W_K, ov))
            popQ.append(comp_score(W_Q, ov))
            popV.append(comp_score(W_V, ov))
    for name, pop in (("K", popK), ("Q", popQ), ("V", popV)):
        pop = np.asarray(pop)
        row[f"{name}_pop_median"] = float(np.median(pop))
        row[f"{name}_pop_max"] = float(pop.max())
        row[f"{name}_pop_n"] = int(pop.size)
        # where L5H2 sits in the model's own distribution -- no placed threshold
        row[f"{name}_pop_rank"] = int((pop > row[name]).sum())   # 0 == largest
        row[f"{name}_pop_z"] = float(
            (row[name] - pop.mean()) / max(pop.std(), 1e-300))
    return row


@torch.no_grad()
def _resid_cov(model, batches) -> tuple:
    """Chunked (n, mean, Sigma) of the residual stream at DST_LAYER's input.
    Accumulates X^T X rather than holding every activation, so the token count
    is limited by time and not by memory."""
    n = 0
    s = np.zeros(D_MODEL, dtype=np.float64)
    C = np.zeros((D_MODEL, D_MODEL), dtype=np.float64)
    for ids in batches:
        out = model(ids, output_hidden_states=True)
        X = out.hidden_states[DST_LAYER].float().reshape(-1, D_MODEL)
        X = X.numpy().astype(np.float64)
        n += len(X); s += X.sum(0); C += X.T @ X
        del out, X
    mean = s / max(n, 1)
    Sigma = (C - np.outer(s, s) / max(n, 1)) / max(n - 1, 1)
    return n, mean, Sigma


def _sqrtm_psd(Sigma):
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T, w


def _plane_compare(M, S_half, k=2):
    """Top-k right-singular plane of M, against that of M Sigma^(1/2) pulled
    back to input space. Returns (top1 overlap, principal cosines)."""
    _, _, Vt_raw = np.linalg.svd(M, full_matrices=False)
    _, _, Vt_wht = np.linalg.svd(M @ S_half, full_matrices=False)

    def pull(v):
        u = S_half @ v
        nn = np.linalg.norm(u)
        return u / nn if nn > 0 else u

    top1 = float(abs(np.dot(Vt_raw[0], pull(Vt_wht[0]))))
    P = np.stack([pull(Vt_wht[i]) for i in range(k)])
    Q, _ = np.linalg.qr(P.T)
    cos = np.clip(np.linalg.svd(Vt_raw[:k] @ Q, compute_uv=False), -1.0, 1.0)
    return top1, [float(x) for x in cos], Vt_raw, Vt_wht


@torch.no_grad()
def whitening_at_step(model, step: int, n_seqs: int, tok=None) -> dict:
    """Does the top OV plane survive the metric the data actually occupies?

    Three arms, because H2's first pass could not separate a real effect from
    covariance estimation noise at 1536 tokens:
      battery      the repeated-token battery at `n_seqs` sequences
      natural      core.config.PROMPTS, the model's own operating distribution
      split-half   the battery split in two, each half whitening independently.
                   If the two halves AGREE with each other while both DISAGREE
                   with raw, the low overlap is signal and not noise. This is
                   the control, and it needs no model of the noise.
    """
    M = _load_ov(step, DST_LAYER, DST_HEAD)
    rng = np.random.default_rng(EVAL_SEED)

    # ---- battery, chunked so the token count is bounded by time not memory --
    CH = 8
    seqs = [induction_batch(rng) for _ in range(max(n_seqs // CH, 1))]
    n_b, _, Sig_b = _resid_cov(model, seqs)
    Sh_b, w_b = _sqrtm_psd(Sig_b)

    # ---- split half of the SAME battery, for the noise control --------------
    half = max(len(seqs) // 2, 1)
    n_h1, _, Sig_h1 = _resid_cov(model, seqs[:half])
    n_h2, _, Sig_h2 = _resid_cov(model, seqs[half:] or seqs[:half])
    Sh_h1, _ = _sqrtm_psd(Sig_h1)
    Sh_h2, _ = _sqrtm_psd(Sig_h2)

    out = {"step": step, "arms": {}}

    def arm(name, n, Sh, w):
        top1, cos, Vt_raw, Vt_wht = _plane_compare(M, Sh)
        e_r = np.linalg.svd(M, compute_uv=False) ** 2
        e_w = np.linalg.svd(M @ Sh, compute_uv=False) ** 2
        out["arms"][name] = {
            "n_tokens": int(n),
            "top1_overlap_raw_vs_whitened": top1,
            "top2_principal_cos": cos,
            "top2_mean_principal_cos": float(np.mean(cos)),
            "sv1_energy_share_raw": float(e_r[0] / e_r.sum()),
            "sv1_energy_share_whitened": float(e_w[0] / e_w.sum()),
            "sv12_energy_share_raw": float(e_r[:2].sum() / e_r.sum()),
            "sv12_energy_share_whitened": float(e_w[:2].sum() / e_w.sum()),
            "sigma_effective_rank": float(
                np.exp(-((w / w.sum()) * np.log(w / w.sum() + 1e-300)).sum()))
            if w is not None else None,
        }
        return Vt_wht

    arm("battery", n_b, Sh_b, w_b)
    V1 = arm("battery_half1", n_h1, Sh_h1, None)
    V2 = arm("battery_half2", n_h2, Sh_h2, None)

    # ---- THE CONTROL: do the two halves agree with EACH OTHER? --------------
    def pull(Sh, v):
        u = Sh @ v
        nn = np.linalg.norm(u)
        return u / nn if nn > 0 else u
    P1 = np.stack([pull(Sh_h1, V1[i]) for i in range(2)])
    P2 = np.stack([pull(Sh_h2, V2[i]) for i in range(2)])
    Q2, _ = np.linalg.qr(P2.T)
    cos_hh = np.clip(np.linalg.svd(P1 @ Q2, compute_uv=False), -1.0, 1.0)
    out["split_half_agreement"] = {
        "top1_overlap": float(abs(np.dot(P1[0], P2[0]))),
        "top2_principal_cos": [float(x) for x in cos_hh],
        "reading": ("halves agreeing with each other while both disagree with "
                    "raw => the low raw-vs-whitened overlap is signal; halves "
                    "disagreeing with each other too => it is estimation noise"),
    }

    # ---- natural text, the model's operating distribution -------------------
    if tok is not None:
        from core.config import PROMPTS
        batches = []
        for v in PROMPTS.values():
            if not isinstance(v, str):
                continue
            ids = tok(v, return_tensors="pt", truncation=True,
                      max_length=512)["input_ids"]
            if ids.shape[1] >= 8:
                batches.append(ids)
        if batches:
            n_n, _, Sig_n = _resid_cov(model, batches)
            Sh_n, w_n = _sqrtm_psd(Sig_n)
            arm("natural_text", n_n, Sh_n, w_n)
            # and how different are the two metrics themselves?
            fb = np.linalg.norm(Sig_b) * np.linalg.norm(Sig_n)
            out["sigma_battery_vs_natural_cos"] = float(
                np.sum(Sig_b * Sig_n) / max(fb, 1e-300))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in ALL_STEPS))
    ap.add_argument("--whiten-step", type=int, default=4000)
    ap.add_argument("--skip-composition", action="store_true")
    ap.add_argument("--whiten-seqs", type=int, default=128)
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "git_sha": git_sha,
           "src": f"L{SRC_LAYER}H{SRC_HEAD}", "dst": f"L{DST_LAYER}H{DST_HEAD}",
           "composition": [], "whitening": None}

    if not args.skip_composition:
        for s in steps:
            model, _ = load_causal_lm(f"pythia-410m-step{s}")
            model.eval()
            row = composition_at_step(model, s)
            res["composition"].append(row)
            print(f"  step {s:>6}: K {row['K']:.4f} (pop med {row['K_pop_median']:.4f}, "
                  f"rank {row['K_pop_rank']}/{row['K_pop_n']}, z {row['K_pop_z']:+.2f})  "
                  f"Q {row['Q']:.4f}  V {row['V']:.4f}", flush=True)
            del model

    model, tok = load_causal_lm(f"pythia-410m-step{args.whiten_step}")
    model.eval()
    res["whitening"] = whitening_at_step(model, args.whiten_step,
                                         args.whiten_seqs, tok)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    if res["composition"]:
        print("\n=== K-composition L5H2 -> L7H8, against the model's own population ===")
        print(f"{'step':>7} {'K':>8} {'K_static':>9} {'Q':>8} {'V':>8} "
              f"{'pop_med':>8} {'rank':>6} {'z':>7}")
        for r in res["composition"]:
            print(f"{r['step']:>7} {r['K']:>8.4f} {r['K_static']:>9.4f} "
                  f"{r['Q']:>8.4f} {r['V']:>8.4f} {r['K_pop_median']:>8.4f} "
                  f"{r['K_pop_rank']:>6} {r['K_pop_z']:>+7.2f}")

    w = res["whitening"]
    print(f"\n=== whitening at step {w['step']} ===")
    print(f"{'arm':>16} {'tokens':>8} {'top1':>7} {'cos1':>7} {'cos2':>7} "
          f"{'sv12 raw':>9} {'sv12 wht':>9} {'eff rank':>9}")
    for name, a in w["arms"].items():
        er = a["sigma_effective_rank"]
        print(f"{name:>16} {a['n_tokens']:>8} "
              f"{a['top1_overlap_raw_vs_whitened']:>7.3f} "
              f"{a['top2_principal_cos'][0]:>7.3f} {a['top2_principal_cos'][1]:>7.3f} "
              f"{a['sv12_energy_share_raw']:>9.3f} "
              f"{a['sv12_energy_share_whitened']:>9.3f} "
              f"{(f'{er:.1f}' if er else '-'):>9}")
    sh = w["split_half_agreement"]
    print(f"\n  SPLIT-HALF CONTROL: half1 vs half2 top1 {sh['top1_overlap']:.3f}, "
          f"plane cos [{sh['top2_principal_cos'][0]:.3f}, "
          f"{sh['top2_principal_cos'][1]:.3f}]")
    if "sigma_battery_vs_natural_cos" in w:
        print(f"  Sigma(battery) vs Sigma(natural) cosine: "
              f"{w['sigma_battery_vs_natural_cos']:.3f}")


if __name__ == "__main__":
    main()
