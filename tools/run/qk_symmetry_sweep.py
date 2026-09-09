"""Does QK symmetry track INDUCTION, or merely track MATCHING?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-I found `L7H8`'s static QK operator becoming a similarity
kernel -- symmetric energy fraction 0.50 (the random-matrix baseline every head
starts at) rising to **0.956**, while the layer-7 median stays at 0.520 and the
head sits at rank 0 of 16 from step 2000 on. That is the first weights-only
quantity in the programme that cleanly identifies the head.

Block I also names the reading that has to be ruled out before any of it is
registered: **a symmetric content operator is what one would expect
architecturally.** RoPE carries positional asymmetry in the 16 excluded dims and
the causal mask carries the rest, so "symmetric content part plus positional
part" is the ordinary way to build *any* content matcher. If every matcher is
symmetric, block I is architecture. If only induction heads are, it is a finding.

This runner asks that over the whole model rather than one layer, and joins the
answer to the behavioural induction score already on disk
(`data/analysis/behavioural_series.json`, 384 heads x 19 steps, keyed
"layer,head").

THE DISCRIMINATOR is the confusion matrix, not the correlation. Per §3.13's own
rule -- exploratory work reports the mean view AND the extremum view, and the
choice between them is never made after seeing the data -- both are emitted:

  * Spearman rho of symmetry against induction score over all 384 heads
    (the "mean" instrument, which §3.12-G6 showed can average a concentrated
    signal away);
  * the RANK of the top induction heads on symmetry, and the induction scores
    of the top-symmetry heads (the extremum instrument).

**High-symmetry heads that are NOT induction heads are the decisive cell.** If
they exist in number, symmetry is a property of matching or of architecture and
not of induction.

THE ARITHMETIC, which is why 384 heads x 19 steps is nearly free. For
`M = A B` with `A = W_Q[16:]^T` (1024,48) and `B = W_K[16:]` (48,1024):

    ||M||_F^2 = tr(M^T M) = tr( (A^T A)(B B^T) )        48x48
    tr(M^2)   = tr(ABAB)  = tr( (BA)^2 )                48x48
    ||S||_F^2 = ( ||M||_F^2 + tr(M^2) ) / 2

so the symmetric fraction needs no (1024,1024) matrix at all. The identity is
asserted against the direct computation on the first head of every run rather
than trusted.

COST: one model load per step, then 48x48 algebra. NO FORWARD PASS.
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
from scipy import stats

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import D_HEAD, D_MODEL, N_HEADS

ROTARY_NDIMS = 16
OUT = DATA / "analysis" / "qk_symmetry_sweep.json"
BEHAV = DATA / "analysis" / "behavioural_series.json"
ALL_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000,
             4000, 8000, 16000, 32000, 54000, 143000]

#: The genuine induction heads of PROJECT.md §3.11 block C -- match on the
#: repo-convention pairs AND have an OV copy effect -- plus the prev-token
#: partner, which is a POSITIONAL matcher and so the nearest thing the weights
#: alone offer to "matches, but not by induction".
NAMED = {"L7H8": (7, 8), "L9H9": (9, 9), "L6H0": (6, 0), "L1H15": (1, 15),
         "L5H2_prevtok": (5, 2), "L2H10_anticopier": (2, 10)}


def sym_fraction(A: np.ndarray, B: np.ndarray) -> tuple:
    """(symmetric energy fraction, ||M||_F) for M = A @ B, via 48x48 algebra."""
    AtA = A.T @ A                      # (48,48)
    BBt = B @ B.T                      # (48,48)
    BA = B @ A                         # (48,48)
    fro2 = float(np.sum(AtA * BBt.T))  # tr(AtA @ BBt)
    tr_m2 = float(np.trace(BA @ BA))
    sym2 = (fro2 + tr_m2) / 2.0
    return float(sym2 / max(fro2, 1e-300)), float(np.sqrt(max(fro2, 0.0)))


def head_factors(q3, layer_block, head):
    W_Q = q3[head, 0:D_HEAD, :]
    W_K = q3[head, D_HEAD:2 * D_HEAD, :]
    return W_Q[ROTARY_NDIMS:, :].T.copy(), W_K[ROTARY_NDIMS:, :].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in ALL_STEPS))
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x]

    with open(BEHAV) as fh:
        bd = json.load(fh)
    b_steps = bd["steps"]
    b_series = bd["series_excl_repeated"]          # {"L,H": [19 values]}

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "git_sha": git_sha, "steps": steps, "per_step": {}}

    checked = False
    for s in steps:
        model, _ = load_causal_lm(f"pythia-410m-step{s}")
        model.eval()
        sym = np.full((24, N_HEADS), np.nan)
        nrm = np.full((24, N_HEADS), np.nan)
        for L in range(len(model.gpt_neox.layers)):
            qkv = model.gpt_neox.layers[L].attention.query_key_value.weight
            q3 = qkv.detach().numpy().astype(np.float64).reshape(
                N_HEADS, 3 * D_HEAD, D_MODEL)
            for h in range(N_HEADS):
                A, B = head_factors(q3, L, h)
                sym[L, h], nrm[L, h] = sym_fraction(A, B)
                if not checked:
                    M = A @ B
                    S = (M + M.T) / 2.0
                    direct = float(np.sum(S * S) / np.sum(M * M))
                    assert abs(direct - sym[L, h]) < 1e-9, (direct, sym[L, h])
                    checked = True
        del model

        bi = b_steps.index(s) if s in b_steps else None
        ind = np.full((24, N_HEADS), np.nan)
        if bi is not None:
            for k, v in b_series.items():
                L, h = (int(x) for x in k.split(","))
                if L < 24:
                    ind[L, h] = v[bi]

        flat_s, flat_i = sym.ravel(), ind.ravel()
        ok = np.isfinite(flat_s) & np.isfinite(flat_i)
        rho = float(stats.spearmanr(flat_s[ok], flat_i[ok]).statistic) if ok.sum() > 8 else float("nan")

        order_s = np.argsort(-flat_s)      # highest symmetry first
        order_i = np.argsort(-np.where(np.isfinite(flat_i), flat_i, -np.inf))
        rank_of = {int(j): r for r, j in enumerate(order_s)}

        row = {
            "step": s, "spearman_sym_vs_induction": rho,
            "median_sym": float(np.nanmedian(flat_s)),
            "p99_sym": float(np.nanpercentile(flat_s, 99)),
            "max_sym": float(np.nanmax(flat_s)),
            "n_heads_sym_above_0.7": int(np.nansum(flat_s > 0.7)),
            "n_heads_sym_above_0.9": int(np.nansum(flat_s > 0.9)),
            "top10_symmetry": [], "top10_induction": [], "named": {},
        }
        for j in order_s[:10]:
            L, h = divmod(int(j), N_HEADS)
            row["top10_symmetry"].append(
                {"head": f"L{L}H{h}", "sym": float(flat_s[j]),
                 "induction": (float(flat_i[j]) if np.isfinite(flat_i[j]) else None),
                 "induction_rank": int(np.where(order_i == j)[0][0])})
        for j in order_i[:10]:
            if not np.isfinite(flat_i[j]):
                continue
            L, h = divmod(int(j), N_HEADS)
            row["top10_induction"].append(
                {"head": f"L{L}H{h}", "induction": float(flat_i[j]),
                 "sym": float(flat_s[j]), "sym_rank": rank_of[int(j)]})
        for name, (L, h) in NAMED.items():
            j = L * N_HEADS + h
            row["named"][name] = {
                "sym": float(flat_s[j]), "sym_rank_of_384": rank_of[j],
                "qk_norm": float(nrm[L, h]),
                "induction": (float(flat_i[j]) if np.isfinite(flat_i[j]) else None)}
        res["per_step"][str(s)] = row
        print(f"  step {s:>6}: median sym {row['median_sym']:.4f}  "
              f"max {row['max_sym']:.4f}  n>0.7 {row['n_heads_sym_above_0.7']:>3}  "
              f"n>0.9 {row['n_heads_sym_above_0.9']:>3}  rho {rho:+.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")

    print("\n=== population of QK symmetry over all 384 heads ===")
    print(f"{'step':>7} {'median':>8} {'p99':>8} {'max':>8} {'n>0.7':>6} "
          f"{'n>0.9':>6} {'rho vs induction':>17}")
    for s in steps:
        r = res["per_step"][str(s)]
        print(f"{s:>7} {r['median_sym']:>8.4f} {r['p99_sym']:>8.4f} "
              f"{r['max_sym']:>8.4f} {r['n_heads_sym_above_0.7']:>6} "
              f"{r['n_heads_sym_above_0.9']:>6} "
              f"{r['spearman_sym_vs_induction']:>+17.3f}")

    last = str(steps[-1])
    print(f"\n=== THE DECISIVE CELL: top-10 SYMMETRY heads at step {last} ===")
    print(f"{'head':>8} {'sym':>8} {'induction':>10} {'induction rank/384':>19}")
    for e in res["per_step"][last]["top10_symmetry"]:
        ind = "-" if e["induction"] is None else f"{e['induction']:.4f}"
        print(f"{e['head']:>8} {e['sym']:>8.4f} {ind:>10} {e['induction_rank']:>19}")

    print(f"\n=== top-10 INDUCTION heads at step {last}, and their symmetry ===")
    print(f"{'head':>8} {'induction':>10} {'sym':>8} {'sym rank/384':>13}")
    for e in res["per_step"][last]["top10_induction"]:
        print(f"{e['head']:>8} {e['induction']:>10.4f} {e['sym']:>8.4f} "
              f"{e['sym_rank']:>13}")

    print("\n=== named heads across the axis: symmetry (rank of 384) ===")
    hdr = " ".join(f"{n[:9]:>11}" for n in NAMED)
    print(f"{'step':>7} {hdr}")
    for s in steps:
        r = res["per_step"][str(s)]["named"]
        cells = " ".join(f"{r[n]['sym']:.3f}({r[n]['sym_rank_of_384']:>3})" for n in NAMED)
        print(f"{s:>7} {cells}")


if __name__ == "__main__":
    main()
