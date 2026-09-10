"""Elhage's token-basis copying score, which this project has never computed.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-N4. The field-standard test for whether a head's OV circuit
copies takes the eigenvalues of the **token-to-token** circuit and summarises
their positiveness (Elhage et al., *A Mathematical Framework for Transformer
Circuits*):

    C = W_U (W_O W_V) W_E^T          (vocab x vocab)
    copying score = sum(lambda) / sum(|lambda|)

A copier has **positive** eigenvalues: attending to a token promotes that same
token. This project has instead been reading
`p2b_imaginary/head_circuits.head_core` = `W_V W_O`, the `(64,64)` core in the
**residual** basis, and *every* "100 % repulsive" statement in §3.11 and
§3.12-A is about that matrix. The two differ by the vocabulary round-trip
`G = W_E^T W_U` sandwiched between the factors, and nothing guarantees they
share a sign.

THE REDUCTION, which is why this is cheap. `C = P Q` with `P = W_U W_O`
(vocab,64) and `Q = W_V W_E^T` (64,vocab), so the nonzero spectrum of `C` is the
spectrum of

    Q P = W_V (W_E^T W_U) W_O = W_V G W_O          (64 x 64)

`G` is computed **once per checkpoint** and reused for all 384 heads, so the
per-head cost is one `(64,1024)x(1024,1024)x(1024,64)` product and one `64x64`
eigendecomposition -- the same price as the core already computed.

A CONVENTION TRAP, verified rather than assumed. `tools/run/induction_rank_sweep
.ov_factors` returns `OV_h = W_V^T W_O^T = (W_O W_V)^T` -- the **TRANSPOSE** of
the residual-stream operator (checked exactly, relative error 0.0). That is
harmless for every quantity the project has read off it, since eigenvalues,
singular values, Frobenius norm and the symmetric fraction are all
transpose-invariant. It is **not** harmless here: the copying score is
directional, with `W_E` on one side and `W_U` on the other, so this runner uses
`W_O W_V` directly from the model rather than the repo's `OV_h`.

LAYER NORM. The primary arm folds none. A positive scalar rescaling cannot
change the SIGN of an eigenvalue, so the score's sign structure is robust to the
scale part of LN; the per-dimension gain and the mean-centering are not scalars,
so both are folded in a sensitivity arm (`W_U diag(g) (I - 11^T/d)`) and the two
are reported side by side. §3.11's own copy-score note flags the LN-folded
reading as its softest half; this makes the sensitivity explicit instead.

NO FORWARD PASS. Weights only.
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

from core.lm_loading import load_causal_lm

OUT = DATA / "analysis" / "copying_score_sweep.json"
D_HEAD, N_HEADS, D_MODEL = 64, 16, 1024

NAMED = {"L7H8": (7, 8), "L9H9": (9, 9), "L6H0": (6, 0), "L1H15": (1, 15),
         "L5H2_prevtok": (5, 2), "L2H10_anticopier": (2, 10),
         "L9H8_anticopier": (9, 8)}


def head_WV_WO(model, layer, head):
    """The RESIDUAL-stream factors: out = W_O @ (W_V @ x)."""
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    q3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    W_V = q3[head, 2 * D_HEAD:, :]                                   # (64,1024)
    dense = model.gpt_neox.layers[layer].attention.dense.weight
    W_O = dense.detach().numpy().astype(np.float64)[:, head * D_HEAD:(head + 1) * D_HEAD]
    return W_V, W_O                                                  # (64,d),(d,64)


def score_from(K):
    """Elhage's summary plus the two obvious companions."""
    ev = np.linalg.eigvals(K)
    s_abs = float(np.sum(np.abs(ev)))
    return {
        "copying_score": float(np.real(np.sum(ev)) / max(s_abs, 1e-300)),
        "frac_eig_positive_re": float(np.mean(ev.real > 0)),
        "pos_energy_fraction": float(
            (np.abs(ev[ev.real > 0]) ** 2).sum() / max((np.abs(ev) ** 2).sum(), 1e-300)),
        "trace": float(np.real(np.trace(K))),
        "sum_abs_eig": s_abs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1000,2000,4000,8000,16000,143000")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "per_step": {}}

    for step in steps:
        model, _ = load_causal_lm(f"pythia-410m-step{step}")
        model.eval()
        W_E = model.gpt_neox.embed_in.weight.detach().numpy().astype(np.float64)
        W_U = model.embed_out.weight.detach().numpy().astype(np.float64)
        g = model.gpt_neox.final_layer_norm.weight.detach().numpy().astype(np.float64)

        G = W_E.T @ W_U                                              # (1024,1024)
        # sensitivity arm: final-LN gain and mean-centring folded into W_U
        Wu_ln = W_U * g[None, :]
        Wu_ln = Wu_ln - Wu_ln.mean(axis=1, keepdims=True)            # (I - 11^T/d)
        G_ln = W_E.T @ Wu_ln

        n_layers = len(model.gpt_neox.layers)
        raw = np.full((n_layers, N_HEADS), np.nan)
        lnf = np.full((n_layers, N_HEADS), np.nan)
        rows = {}
        for L in range(n_layers):
            for h in range(N_HEADS):
                W_V, W_O = head_WV_WO(model, L, h)
                s = score_from(W_V @ G @ W_O)
                s_ln = score_from(W_V @ G_ln @ W_O)
                raw[L, h] = s["copying_score"]
                lnf[L, h] = s_ln["copying_score"]
                rows[f"L{L}H{h}"] = {"raw": s, "ln_folded": s_ln}
        del model

        flat = raw.ravel()
        row = {"step": step,
               "median": float(np.nanmedian(flat)),
               "p99": float(np.nanpercentile(flat, 99)),
               "max": float(np.nanmax(flat)),
               "min": float(np.nanmin(flat)),
               "n_above_0.2": int(np.nansum(flat > 0.2)),
               "n_above_0.4": int(np.nansum(flat > 0.4)),
               "median_ln_folded": float(np.nanmedian(lnf.ravel())),
               "named": {}, "top10": []}
        order = np.argsort(-flat)
        for j in order[:10]:
            L, h = divmod(int(j), N_HEADS)
            row["top10"].append({"head": f"L{L}H{h}", "score": float(flat[j])})
        rank_of = {int(j): r for r, j in enumerate(order)}
        for name, (L, h) in NAMED.items():
            j = L * N_HEADS + h
            row["named"][name] = {
                "copying_score": float(raw[L, h]),
                "copying_score_ln_folded": float(lnf[L, h]),
                "rank_of_384": rank_of[j],
                **{k: v for k, v in rows[f"L{L}H{h}"]["raw"].items()
                   if k in ("frac_eig_positive_re", "pos_energy_fraction")}}
        res["per_step"][str(step)] = row
        print(f"  step {step:>6}: median {row['median']:+.4f}  max {row['max']:+.4f}  "
              f"n>0.2 {row['n_above_0.2']:>3}  |  L7H8 "
              f"{row['named']['L7H8']['copying_score']:+.4f} "
              f"(rank {row['named']['L7H8']['rank_of_384']})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    print("\n=== token-basis copying score, population of 384 heads ===")
    print(f"{'step':>7} {'median':>8} {'p99':>8} {'max':>8} {'min':>8} "
          f"{'n>0.2':>6} {'n>0.4':>6}")
    for s in steps:
        r = res["per_step"][str(s)]
        print(f"{s:>7} {r['median']:>+8.4f} {r['p99']:>+8.4f} {r['max']:>+8.4f} "
              f"{r['min']:>+8.4f} {r['n_above_0.2']:>6} {r['n_above_0.4']:>6}")

    print("\n=== named heads: copying score (rank of 384) ===")
    hdr = " ".join(f"{n[:11]:>15}" for n in NAMED)
    print(f"{'step':>7} {hdr}")
    for s in steps:
        n = res["per_step"][str(s)]["named"]
        cells = " ".join(f"{n[k]['copying_score']:+.3f}({n[k]['rank_of_384']:>3})"
                         for k in NAMED)
        print(f"{s:>7} {cells}")

    last = str(steps[-1])
    print(f"\n=== top-10 copying heads at step {last} ===")
    for e in res["per_step"][last]["top10"]:
        print(f"    {e['head']:>8}  {e['score']:+.4f}")

    print(f"\n=== LN sensitivity (median raw vs LN-folded) ===")
    for s in steps:
        r = res["per_step"][str(s)]
        print(f"    step {s:>6}: raw {r['median']:+.4f}   "
              f"ln-folded {r['median_ln_folded']:+.4f}")


if __name__ == "__main__":
    main()
