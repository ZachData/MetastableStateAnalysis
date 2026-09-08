"""Stage 1, QK half (PROJECT.md sec 3.11): the minimal rank of the STATIC
(non-positional) QK operator of induction head L7H8.

WHY THE STATIC PART, AND WHY THAT IS THE RIGHT CUT. Pythia applies rotary to
the first `rotary_ndims = 16` of each head's 64 query/key dims; dims 16..63
are static. The attention score splits exactly:

    score_ij = <rot(q_i)[:16], rot(k_j)[:16]>  +  <q_i[16:], k_j[16:]>
                                                  [ ... static ... ]

Induction's prefix match is a CONTENT test ("this token equals that token");
RoPE carries the positional "look one back" part. So the static 48-dim
bilinear form is where the matcher's content mechanism lives, and truncating
it -- leaving the 16 rotary dims untouched -- asks how many dimensions that
mechanism needs without breaking what the model computes. It is the QK analog
of Stage 1's OV question and comparable to r*_OV.

    M_static = W_Q_h[16:, :].T @ W_K_h[16:, :]        (1024, 1024, rank <= 48)

truncated as A @ B with A = W_Q_h[16:].T (1024,48), B = W_K_h[16:] (48,1024),
in the SVD, Schur and matched-norm random bases (`truncate` from
induction_rank_sweep), written back into rows 16..63 of Q and K.

READOUT (decision 3): the behavioural induction score itself -- mean
post-softmax attention, at L7H8, from a second-copy position to the token that
followed the first-copy occurrence of the same token. Pure QK; this is the
quantity OV ablation could not move and QK ablation should.

r* = where the score-vs-r curve leaves the matched-norm random control band.
Weights save/restore around every measurement; baseline re-checked at the end.
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
import scipy
import torch
import transformers

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_HEAD, D_MODEL, N_HEADS, LAYER, HEAD, STEP,
    N_REP, N_SEQS, EVAL_SEED, VOCAB_LO, VOCAB_HI, induction_batch, truncate,
)

ROTARY_NDIMS = 16
N_STATIC = D_HEAD - ROTARY_NDIMS          # 48


def qk_static_factors(model, layer, head):
    """A (1024,48) = W_Q_h[16:].T, B (48,1024) = W_K_h[16:].  M = A @ B."""
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    qkv3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    W_Q = qkv3[head, 0:D_HEAD, :]                       # (64, 1024)
    W_K = qkv3[head, D_HEAD:2 * D_HEAD, :]              # (64, 1024)
    return W_Q[ROTARY_NDIMS:, :].T.copy(), W_K[ROTARY_NDIMS:, :].copy()


def write_qk_static(model, layer, head, A, B):
    """Write truncated static factors back into rows 16..63 of Q and K,
    zero-padding rank r -> 48. Rotary rows 0..15 untouched."""
    r = A.shape[1]
    A_pad = np.zeros((D_MODEL, N_STATIC)); A_pad[:, :r] = A
    B_pad = np.zeros((N_STATIC, D_MODEL)); B_pad[:r, :] = B
    with torch.no_grad():
        qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
        v = qkv.view(N_HEADS, 3 * D_HEAD, D_MODEL)
        v[head, ROTARY_NDIMS:D_HEAD, :].copy_(torch.tensor(A_pad.T, dtype=qkv.dtype))
        v[head, D_HEAD + ROTARY_NDIMS:2 * D_HEAD, :].copy_(torch.tensor(B_pad, dtype=qkv.dtype))


@torch.no_grad()
def induction_attention(model, ids, layer, head):
    """Mean post-softmax attention at (layer, head) on the repo's induction
    pairs for a repeated sequence s+s: query = N_REP + t (second copy),
    key = t (the first-copy occurrence of the same token). This is
    `core.battery_structure.induction_candidates(strict=False)`'s condition
    `ids[key-1] == ids[query-1]` specialised to s+s, and the pair
    `formation_curve.behavioural_induction_score` reads."""
    out = model(ids, output_attentions=True)
    att = out.attentions[layer][:, head, :, :].float()      # (B, T, T)
    q = torch.arange(N_REP, 2 * N_REP)                       # second-copy queries
    k = torch.arange(0, N_REP)                               # prev identical token
    picked = att[:, q, k]                                    # (B, N_REP)
    return {"induction_attn": float(picked.mean()),
            "induction_attn_sd": float(picked.mean(-1).std()),
            "row_sum_mean": float(att[:, q, :].sum(-1).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="1,2,3,4,6,8,12,16,24,32,40,48")
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--head", type=int, default=HEAD)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    step, layer, head = args.step, args.layer, args.head
    ranks = sorted({0, *(int(x) for x in args.ranks.split(","))})

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    rng = np.random.default_rng(EVAL_SEED)

    model, tok = load_causal_lm(f"pythia-410m-step{step}")   # eager (from_pretrained_eager)
    model.eval()
    ids = induction_batch(np.random.default_rng(EVAL_SEED))

    A0, B0 = qk_static_factors(model, layer, head)
    base = induction_attention(model, ids, layer, head)
    print(f"  baseline induction attn {base['induction_attn']:.4f} "
          f"(row-sum {base['row_sum_mean']:.3f})", flush=True)

    out = {
        "_what_this_is": __doc__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
        "target": {"layer": layer, "head": head, "step": step,
                   "rotary_ndims": ROTARY_NDIMS, "n_static": N_STATIC},
        "eval": {"n_rep": N_REP, "n_seqs": N_SEQS, "seed": EVAL_SEED},
        "baseline": base, "ranks": ranks, "curves": {},
    }

    for basis in ("svd", "schur", "random"):
        n_draws = args.controls if basis == "random" else 1
        rows = []
        for r in ranks:
            vals = []
            for _ in range(n_draws):
                if r == 0:
                    A, B = np.zeros((D_MODEL, 0)), np.zeros((0, D_MODEL))
                elif basis == "random":
                    # matched-norm random rank-r projector in the 48-dim static
                    # core (induction_rank_sweep.truncate's 'random' hardcodes
                    # d_head=64, wrong here).
                    Qr, _ = np.linalg.qr(rng.normal(size=(N_STATIC, r)))
                    A, B = A0 @ (Qr @ Qr.T), B0
                else:
                    A, B = truncate(A0, B0, r, basis, rng)
                write_qk_static(model, layer, head, A, B)
                vals.append(induction_attention(model, ids, layer, head))
            write_qk_static(model, layer, head, A0, B0)     # restore
            agg = {"induction_attn": float(np.mean([v["induction_attn"] for v in vals])),
                   "sd_over_draws": float(np.std([v["induction_attn"] for v in vals])),
                   "rank": r}
            rows.append(agg)
            print(f"  {basis:>6} r={r:<3} induction attn {agg['induction_attn']:.4f}"
                  + (f"  (sd {agg['sd_over_draws']:.4f})" if n_draws > 1 else ""), flush=True)
        out["curves"][basis] = rows

    final = induction_attention(model, ids, layer, head)
    out["restore_check"] = {"baseline": base["induction_attn"],
                            "after": final["induction_attn"],
                            "abs_diff": abs(base["induction_attn"] - final["induction_attn"])}
    print(f"  restore check: {out['restore_check']['abs_diff']:.2e}")

    _def = ("induction_qk_sweep.json" if (step, layer, head) == (STEP, LAYER, HEAD)
            else f"induction_qk_sweep_s{step}_L{layer}H{head}.json")
    dest = Path(args.out) if args.out else DATA / "analysis" / _def
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
