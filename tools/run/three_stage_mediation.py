"""Is `L7H8` UPSTREAM of the copiers, or a copier itself?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-O. `L7H8` has the largest OV causal effect in its layer
(~10x the layer mean, growing ~6x across training) **and** is an anti-copier by
Elhage's token-identity test (copying score -0.094, rank 328 of 384 at 143000,
falling as its causal effect rises). The real copiers -- scores to +0.72 -- sit
in **layers 9-20, downstream**. So the natural reading is three stages:

    L5H2  (layer 5)   positional matcher, prev-token
    L7H8  (layer 7)   content matcher, writes something that is NOT token identity
    L9-L20            the actual token-identity copiers

and §3.11 may have been calling `L7H8` "the copier" because ablating its OV moves
second-copy NLL, when it is upstream and the effect is mediated.

TEST 1 -- CAUSAL MEDIATION, a 2x2 ablation interaction. For `L7H8` and each
downstream copier `C`, measure the second-copy NLL under {intact, ablate L7H8,
ablate C, ablate both} and form

    I = dNLL(both) - dNLL(L7H8) - dNLL(C)

    I < 0  SUB-additive: C's contribution depends on L7H8's -- once L7H8 is
           gone there is less for C to do. That is what a SERIAL circuit looks
           like and is the three-stage prediction.
    I ~ 0  independent contributions, two parallel paths.
    I > 0  SUPER-additive: redundancy, either one can carry the job alone.

The sign is the whole readout, so no magnitude threshold is placed. NON-COPIER
CONTROLS from the same layers are run identically, because "any two heads in
layers 9-20 interact sub-additively" would explain a negative `I` without any
three-stage content.

TEST 2 -- COMPOSITION, weights only. Where does `L7H8`'s OV output land in each
copier's input? Q-, K- and V-composition, each scored against **every head in
layers below the copier** so "elevated" is against the model's own distribution
(the §3.12-H1 lesson: a per-path population control, not one control for K
alone). A three-stage circuit predicts elevated **V**-composition -- the copier
copies what `L7H8` wrote -- or elevated K/Q if `L7H8` instead supplies the
copier's *addressing*.

ARITHMETIC. Composition needs no (1024,1024) matrix:
`W_read @ (W_O W_V) = (W_read W_O) W_V` is (64,64)x(64,1024), and
`||W_O W_V||_F^2 = tr((W_O^T W_O)(W_V W_V^T))` is two 64x64 products.

COST: one model load, then 2 + 2N forward passes.
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
    D_HEAD, D_MODEL, N_HEADS, N_REP, VOCAB_LO, VOCAB_HI, EVAL_SEED,
    ov_factors, write_ov, measure,
)

OUT = DATA / "analysis" / "three_stage_mediation.json"
SRC_LAYER, SRC_HEAD = 7, 8


def batch(rng, n):
    seqs = []
    for _ in range(n):
        s = rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP)
        seqs.append(np.concatenate([s, s]))
    return torch.tensor(np.stack(seqs), dtype=torch.long)


def wv_wo(model, layer, head):
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    q3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    dense = model.gpt_neox.layers[layer].attention.dense.weight
    W_O = dense.detach().numpy().astype(np.float64)[:, head * D_HEAD:(head + 1) * D_HEAD]
    return q3[head, 2 * D_HEAD:, :], W_O                       # (64,d), (d,64)


def qkv_reads(model, layer, head):
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    q3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    return (q3[head, 0:D_HEAD, :], q3[head, D_HEAD:2 * D_HEAD, :],
            q3[head, 2 * D_HEAD:3 * D_HEAD, :])                # W_Q, W_K, W_V


def ov_fro(W_V, W_O):
    return float(np.sqrt(max(np.sum((W_O.T @ W_O) * (W_V @ W_V.T).T), 0.0)))


def comp(W_read, W_V_src, W_O_src, fro_src):
    num = float(np.linalg.norm((W_read @ W_O_src) @ W_V_src))
    return num / max(np.linalg.norm(W_read) * fro_src, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--copiers", default="")
    ap.add_argument("--controls", default="")
    args = ap.parse_args()

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()

    # pick the copiers and their non-copier controls from the copying sweep
    cs_path = DATA / "analysis" / "copying_score_sweep.json"
    with open(cs_path) as fh:
        cs = json.load(fh)["per_step"][str(args.step)]
    top = [e["head"] for e in cs["top10"] if int(e["head"].split("H")[0][1:]) > SRC_LAYER][:5]
    copiers = args.copiers.split(",") if args.copiers else top
    controls = args.controls.split(",") if args.controls else None

    model, _ = load_causal_lm(f"pythia-410m-step{args.step}")
    model.eval()
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)

    if controls is None:
        # Controls are NON-copiers in the SAME layers as the copiers, chosen by
        # measured copying score rather than by position, so "any two heads in
        # layers 9-20 interact sub-additively" is testable rather than assumed.
        W_E = model.gpt_neox.embed_in.weight.detach().numpy().astype(np.float64)
        W_U = model.embed_out.weight.detach().numpy().astype(np.float64)
        G = W_E.T @ W_U
        layers = sorted({int(h.split("H")[0][1:]) for h in copiers})
        cand = []
        for L2 in layers:
            for h2 in range(N_HEADS):
                if f"L{L2}H{h2}" in copiers:
                    continue
                v2, o2 = wv_wo(model, L2, h2)
                ev = np.linalg.eigvals(v2 @ G @ o2)
                sc = float(np.real(np.sum(ev)) / max(np.sum(np.abs(ev)), 1e-300))
                cand.append((abs(sc), sc, f"L{L2}H{h2}"))
        cand.sort()
        controls = [c[2] for c in cand[:3]]
        print(f"  control copying scores: "
              f"{[(c[2], round(c[1], 4)) for c in cand[:3]]}", flush=True)

    print(f"  copiers : {copiers}")
    print(f"  controls: {controls}", flush=True)

    base = measure(model, ids)["second_copy_nll"]
    print(f"  baseline second-copy NLL {base:.4f}", flush=True)

    A_src, B_src = ov_factors(model, SRC_LAYER, SRC_HEAD)
    Z = (np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))

    def nll_with(ablate):
        saved = {}
        for (L, H) in ablate:
            saved[(L, H)] = ov_factors(model, L, H)
            write_ov(model, L, H, *Z)
        v = measure(model, ids)["second_copy_nll"]
        for (L, H), (a, b) in saved.items():
            write_ov(model, L, H, a, b)
        return v

    d_src = nll_with([(SRC_LAYER, SRC_HEAD)]) - base
    print(f"  dNLL(ablate L7H8 alone) {d_src:+.4f}", flush=True)

    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "step": args.step, "n_seqs": args.seqs, "baseline_nll": base,
           "delta_src_alone": d_src, "pairs": {}}

    # ---- Test 2 prerequisites: source OV, once ------------------------------
    Wv_src, Wo_src = wv_wo(model, SRC_LAYER, SRC_HEAD)
    fro_src = ov_fro(Wv_src, Wo_src)

    for name in copiers + controls:
        L = int(name.split("H")[0][1:]); H = int(name.split("H")[1])
        d_c = nll_with([(L, H)]) - base
        d_both = nll_with([(SRC_LAYER, SRC_HEAD), (L, H)]) - base
        inter = d_both - d_src - d_c

        # composition, with a population control over every head below layer L
        W_Q, W_K, W_V = qkv_reads(model, L, H)
        obs = {p: comp(W, Wv_src, Wo_src, fro_src)
               for p, W in (("Q", W_Q), ("K", W_K), ("V", W_V))}
        pop = {"Q": [], "K": [], "V": []}
        for l2 in range(L):
            for h2 in range(N_HEADS):
                v2, o2 = wv_wo(model, l2, h2)
                f2 = ov_fro(v2, o2)
                for p, W in (("Q", W_Q), ("K", W_K), ("V", W_V)):
                    pop[p].append(comp(W, v2, o2, f2))
        entry = {"layer": L, "head": H,
                 "is_control": name in controls,
                 "copying_score": None,
                 "delta_c_alone": d_c, "delta_both": d_both,
                 "interaction": inter,
                 "interaction_over_min": inter / max(min(abs(d_src), abs(d_c)), 1e-9),
                 "composition": {}}
        for p in ("Q", "K", "V"):
            a = np.asarray(pop[p])
            entry["composition"][p] = {
                "value": obs[p], "pop_median": float(np.median(a)),
                "pop_n": int(a.size), "rank": int((a > obs[p]).sum()),
                "z": float((obs[p] - a.mean()) / max(a.std(), 1e-300))}
        res["pairs"][name] = entry
        print(f"    {name:>7} {'(ctrl)' if name in controls else '      '}: "
              f"dNLL(C) {d_c:+.4f}  both {d_both:+.4f}  I {inter:+.4f}  "
              f"| V rank {entry['composition']['V']['rank']:>3} "
              f"z {entry['composition']['V']['z']:+.2f}", flush=True)

    chk = abs(measure(model, ids)["second_copy_nll"] - base)
    res["restore_abs_diff"] = float(chk)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}   (restore {chk:.1e})")

    print(f"\n=== TEST 1: mediation, L7H8 x C  (step {args.step}) ===")
    print(f"  dNLL(L7H8 alone) = {d_src:+.4f}")
    print(f"{'head':>8} {'kind':>7} {'dNLL(C)':>9} {'dNLL(both)':>11} "
          f"{'I':>9} {'I/min':>8}  reading")
    for name, e in res["pairs"].items():
        r = ("SUB-additive" if e["interaction"] < -0.02 else
             "SUPER-additive" if e["interaction"] > 0.02 else "independent")
        print(f"{name:>8} {'ctrl' if e['is_control'] else 'copier':>7} "
              f"{e['delta_c_alone']:>+9.4f} {e['delta_both']:>+11.4f} "
              f"{e['interaction']:>+9.4f} {e['interaction_over_min']:>+8.2f}  {r}")

    print(f"\n=== TEST 2: composition L7H8 -> C, against heads in layers < L ===")
    print(f"{'head':>8} {'kind':>7} | " + " | ".join(
        f"{p} {'val':>6} {'rank':>5} {'z':>6}" for p in ("Q", "K", "V")))
    for name, e in res["pairs"].items():
        cells = " | ".join(
            f"{p} {e['composition'][p]['value']:>6.4f} "
            f"{e['composition'][p]['rank']:>5} {e['composition'][p]['z']:>+6.2f}"
            for p in ("Q", "K", "V"))
        print(f"{name:>8} {'ctrl' if e['is_control'] else 'copier':>7} | {cells}")


if __name__ == "__main__":
    main()
