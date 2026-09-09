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

    # population control: every head in layers 0..DST_LAYER-1 into L7H8's K
    pop = []
    for L in range(DST_LAYER):
        for h in range(N_HEADS):
            pop.append(comp_score(W_K, _load_ov(step, L, h)))
    pop = np.asarray(pop)
    row["K_pop_median"] = float(np.median(pop))
    row["K_pop_max"] = float(pop.max())
    row["K_pop_n"] = int(pop.size)
    # where L5H2 sits in the model's own distribution -- no placed threshold
    row["K_pop_rank"] = int((pop > row["K"]).sum())          # 0 == the largest
    row["K_pop_z"] = float((row["K"] - pop.mean()) / max(pop.std(), 1e-300))
    return row


@torch.no_grad()
def whitening_at_step(model, step: int) -> dict:
    """Residual covariance at layer DST_LAYER's input, and whether the top OV
    directions survive it."""
    ids = induction_batch(np.random.default_rng(EVAL_SEED))
    out = model(ids, output_hidden_states=True)
    # hidden_states[i] is the INPUT to block i (hidden_states[0] = embeddings)
    X = out.hidden_states[DST_LAYER].float().reshape(-1, D_MODEL).numpy().astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    Sigma = (Xc.T @ Xc) / max(len(Xc) - 1, 1)

    w, Vs = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    S_half = (Vs * np.sqrt(w)) @ Vs.T

    M = _load_ov(step, DST_LAYER, DST_HEAD)
    _, sv_raw, Vt_raw = np.linalg.svd(M, full_matrices=False)
    _, sv_wht, Vt_wht = np.linalg.svd(M @ S_half, full_matrices=False)

    # pull whitened read directions back to input space
    def pullback(v):
        u = S_half @ v
        n = np.linalg.norm(u)
        return u / n if n > 0 else u

    v_raw1 = Vt_raw[0]
    v_wht1 = pullback(Vt_wht[0])
    P_raw = Vt_raw[:2]
    P_wht = np.stack([pullback(Vt_wht[0]), pullback(Vt_wht[1])])
    # orthonormalise the pulled-back plane before taking principal angles
    Qw, _ = np.linalg.qr(P_wht.T)
    cos = np.clip(np.linalg.svd(P_raw @ Qw, compute_uv=False), -1.0, 1.0)

    e_raw, e_wht = sv_raw ** 2, sv_wht ** 2
    return {
        "step": step,
        "n_tokens": int(len(X)),
        "top1_overlap_raw_vs_whitened": float(abs(np.dot(v_raw1, v_wht1))),
        "top2_principal_cos": [float(x) for x in cos],
        "top2_mean_principal_cos": float(np.mean(cos)),
        "sv1_energy_share_raw": float(e_raw[0] / e_raw.sum()),
        "sv1_energy_share_whitened": float(e_wht[0] / e_wht.sum()),
        "sv12_energy_share_raw": float(e_raw[:2].sum() / e_raw.sum()),
        "sv12_energy_share_whitened": float(e_wht[:2].sum() / e_wht.sum()),
        "sigma_effective_rank": float(
            np.exp(-(lambda p: (p * np.log(p + 1e-300)).sum())(w / w.sum()))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in ALL_STEPS))
    ap.add_argument("--whiten-step", type=int, default=4000)
    ap.add_argument("--skip-composition", action="store_true")
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

    model, _ = load_causal_lm(f"pythia-410m-step{args.whiten_step}")
    model.eval()
    res["whitening"] = whitening_at_step(model, args.whiten_step)

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
    print(f"\n=== whitening at step {w['step']} ({w['n_tokens']} tokens, "
          f"Sigma eff. rank {w['sigma_effective_rank']:.1f}) ===")
    print(f"  top-1 overlap raw vs whitened : {w['top1_overlap_raw_vs_whitened']:.3f}")
    print(f"  top-2 principal cosines       : "
          f"[{w['top2_principal_cos'][0]:.3f}, {w['top2_principal_cos'][1]:.3f}]")
    print(f"  sigma1 energy share  raw {w['sv1_energy_share_raw']:.3f} -> "
          f"whitened {w['sv1_energy_share_whitened']:.3f}")
    print(f"  sigma12 energy share raw {w['sv12_energy_share_raw']:.3f} -> "
          f"whitened {w['sv12_energy_share_whitened']:.3f}")


if __name__ == "__main__":
    main()
