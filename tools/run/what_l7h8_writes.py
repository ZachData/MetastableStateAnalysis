"""What does `L7H8`'s OV actually write? Five cheap probes, in priority order.

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-P left a sharp puzzle: `L7H8`'s OV write is causally enormous
(+1.107 at step 16000, ~2x the baseline NLL), is **not** token-identity copying
(§3.12-O), and does **not** route through the heads that do token-identity
copying (P1/P2). Before reaching for the MLPs, four other places are cheaper and
one of them may reframe the question.

PROBE 1 -- THE OFFSET, and it comes first because a core convention is at stake.
`core/battery_structure.induction_candidates` documents two conditions:

    strict=False : ids[key-1] == ids[query-1]      (the repo's condition)
    strict=True  : ids[key-1] == ids[query]        (the Anthropic condition)

The repo uses the first. On a repeated sequence `s+s` that pairs query `N_REP+j`
with key `j` -- the **same-token** position. Copying from there returns the
CURRENT token, not the successor, and would be wrong for next-token prediction.
Standard induction pairs query `N_REP+j` with key `j+1`, whose token IS the
successor. **So a head that genuinely does induction should put its mass at
`j+1`, and the repo's measure reads `j`.** This probe measures the attention
profile over offsets around `j` and settles which the head actually does. If the
mass is at `j+1`, then "induction attention 0.92" has been reading an adjacent
position and a near-zero token-identity copying score is exactly what one would
expect from the wrong probe.

PROBE 2 -- PROMOTION OR SUPPRESSION. `dNLL = -(d logit_correct - d logsumexp)`,
so the ablation effect splits exactly into how far the correct token's logit
fell and how far the partition function rose. A head that promotes the answer
and one that suppresses its competitors are different mechanisms and this
separates them for free from logprobs already computed.

PROBE 3 -- QK AGAINST OV. §3.11 ablated the static QK and read ATTENTION
(0.92 -> 0.02); it never read NLL. Ablating each half and reading the same
`second_copy_nll` says how much of the head's contribution is attending
correctly versus what it writes once it has.

PROBE 4 -- THE COMPOSED CIRCUIT. §3.12-O's copying score asks "does attending to
token X promote X", using the raw embedding as the OV's input. But the residual
at the attended position has already been written to by `L5H2`, the prev-token
head. The composed path `W_U OV_{L7H8} OV_{L5H2} W_E^T` asks the question that
matches the circuit: does attending to a position whose PREVIOUS-TOKEN signal is
X promote X? Weights only, same 64x64 reduction.

PROBE 5 -- THE MLPs, which no composition score in §3.12 has touched. Every one
is head->head, and the MLPs are the majority of the parameters. Composition of
`L7H8`'s OV into each downstream MLP's input projection, against a population of
every head below that layer.

COST: two forward passes plus weights. NO p-value; nothing registered.
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
    ov_factors, write_ov,
)
from tools.run.induction_qk_sweep import qk_static_factors, write_qk_static

OUT = DATA / "analysis" / "what_l7h8_writes.json"
SRC = (7, 8)
PREV = (5, 2)
ROT = 16


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
    return q3[head, 2 * D_HEAD:, :], W_O


def ov_fro(W_V, W_O):
    return float(np.sqrt(max(np.sum((W_O.T @ W_O) * (W_V @ W_V.T).T), 0.0)))


def comp(W_read, W_V, W_O, fro):
    return float(np.linalg.norm((W_read @ W_O) @ W_V)) / max(
        np.linalg.norm(W_read) * fro, 1e-300)


@torch.no_grad()
def logit_split(model, ids, chunk=16):
    """(mean logit of the correct token, mean logsumexp) on second-copy
    positions. dNLL = -(d logit_correct - d logsumexp) exactly."""
    lc, lz = [], []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk])
        lg = out.logits[:, :-1, :].float()
        tgt = ids[i:i + chunk, 1:]
        z = torch.logsumexp(lg, dim=-1)
        c = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        lc.append(c[:, N_REP - 1:].numpy().astype(np.float64))
        lz.append(z[:, N_REP - 1:].numpy().astype(np.float64))
        del out, lg, z, c
    return float(np.concatenate(lc).mean()), float(np.concatenate(lz).mean())


@torch.no_grad()
def offset_profile(model, ids, layer, head, offsets, chunk=8):
    """Mean attention from query N_REP+j to key j+offset, per offset."""
    acc = {o: [] for o in offsets}
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_attentions=True)
        att = out.attentions[layer][:, head, :, :].float().numpy().astype(np.float64)
        for o in offsets:
            js = np.arange(max(0, -o), N_REP - max(0, o))
            q = N_REP + js
            k = js + o
            acc[o].append(att[:, q, k].mean(axis=1))
        del out, att
    return {o: float(np.concatenate(acc[o]).mean()) for o in offsets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=16000)
    ap.add_argument("--seqs", type=int, default=16)
    args = ap.parse_args()

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    model, _ = load_causal_lm(f"pythia-410m-step{args.step}")
    model.eval()
    ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)
    res = {"_what_this_is": __doc__, "git_sha": git_sha, "step": args.step,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}

    # ---- PROBE 1: the offset -------------------------------------------------
    offs = [-2, -1, 0, 1, 2, 3]
    res["probe1_offsets"] = {
        "L7H8": offset_profile(model, ids, *SRC, offs),
        "L5H2": offset_profile(model, ids, *PREV, offs),
    }
    print("PROBE 1 -- attention from query N_REP+j to key j+offset")
    print(f"{'head':>6} " + " ".join(f"{('j'+f'{o:+d}') if o else 'j':>8}" for o in offs))
    for h, d in res["probe1_offsets"].items():
        print(f"{h:>6} " + " ".join(f"{d[o]:>8.4f}" for o in offs))

    # ---- PROBE 2 + 3: promotion/suppression, QK vs OV -------------------------
    c0, z0 = logit_split(model, ids)
    base_nll = -(c0 - z0)
    A0, B0 = ov_factors(model, *SRC)
    Aq, Bq = qk_static_factors(model, *SRC)

    write_ov(model, *SRC, np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
    c_ov, z_ov = logit_split(model, ids)
    write_ov(model, *SRC, A0, B0)

    write_qk_static(model, *SRC, np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
    c_qk, z_qk = logit_split(model, ids)
    write_qk_static(model, *SRC, Aq, Bq)

    c_r, z_r = logit_split(model, ids)
    res["probe23"] = {
        "baseline_nll": base_nll, "restore_abs_diff": abs(-(c_r - z_r) - base_nll),
        "ov_ablation": {"dnll": -(c_ov - z_ov) - base_nll,
                        "d_logit_correct": c_ov - c0, "d_logsumexp": z_ov - z0},
        "qk_ablation": {"dnll": -(c_qk - z_qk) - base_nll,
                        "d_logit_correct": c_qk - c0, "d_logsumexp": z_qk - z0},
    }
    print(f"\nPROBE 2/3 -- baseline NLL {base_nll:.4f} "
          f"(restore {res['probe23']['restore_abs_diff']:.1e})")
    print(f"{'ablation':>12} {'dNLL':>9} {'d logit(correct)':>18} "
          f"{'d logsumexp':>13}  reading")
    for k in ("ov_ablation", "qk_ablation"):
        e = res["probe23"][k]
        share = abs(e["d_logsumexp"]) / max(
            abs(e["d_logit_correct"]) + abs(e["d_logsumexp"]), 1e-12)
        print(f"{k:>12} {e['dnll']:>+9.4f} {e['d_logit_correct']:>+18.4f} "
              f"{e['d_logsumexp']:>+13.4f}  "
              f"{'competitor-rise' if share > 0.5 else 'correct-token-fall'} "
              f"{share:.0%}")

    # ---- PROBE 4: the composed circuit ---------------------------------------
    W_E = model.gpt_neox.embed_in.weight.detach().numpy().astype(np.float64)
    W_U = model.embed_out.weight.detach().numpy().astype(np.float64)
    G = W_E.T @ W_U
    Vs, Os = wv_wo(model, *SRC)
    Vp, Op = wv_wo(model, *PREV)

    def score(K):
        ev = np.linalg.eigvals(K)
        return float(np.real(np.sum(ev)) / max(np.sum(np.abs(ev)), 1e-300))

    direct = score(Vs @ G @ Os)                       # W_U OV_src W_E^T
    # composed: C = W_U (O_s V_s)(O_p V_p) W_E^T = (W_U O_s)(V_s O_p)(V_p W_E^T),
    # so the nonzero spectrum is that of (V_s O_p)(V_p G O_s) -- all 64x64.
    composed = score((Vs @ Op) @ (Vp @ G @ Os))
    res["probe4"] = {"direct_copying_score": direct,
                     "composed_L5H2_then_L7H8": composed}
    print(f"\nPROBE 4 -- copying score, direct vs composed through L5H2")
    print(f"    direct   W_U OV(L7H8) W_E^T            : {direct:+.4f}")
    print(f"    composed W_U OV(L7H8) OV(L5H2) W_E^T   : {composed:+.4f}")

    # ---- PROBE 5: the MLPs ---------------------------------------------------
    fro_s = ov_fro(Vs, Os)
    n_layers = len(model.gpt_neox.layers)
    rows = []
    for L in range(SRC[0], n_layers):
        W_in = model.gpt_neox.layers[L].mlp.dense_h_to_4h.weight
        W_in = W_in.detach().numpy().astype(np.float64)
        obs = comp(W_in, Vs, Os, fro_s)
        pop = []
        for l2 in range(L):
            for h2 in range(N_HEADS):
                if (l2, h2) == SRC:
                    continue
                v2, o2 = wv_wo(model, l2, h2)
                pop.append(comp(W_in, v2, o2, ov_fro(v2, o2)))
        a = np.asarray(pop)
        rows.append({"layer": L, "value": obs, "pop_median": float(np.median(a)),
                     "pop_n": int(a.size), "rank": int((a > obs).sum()),
                     "z": float((obs - a.mean()) / max(a.std(), 1e-300))})
    res["probe5_mlp"] = rows
    print(f"\nPROBE 5 -- L7H8 OV -> MLP input, vs every head below that layer")
    print(f"{'layer':>6} {'value':>8} {'pop med':>9} {'rank':>6} {'of':>5} {'z':>7}")
    for r in rows:
        print(f"{r['layer']:>6} {r['value']:>8.4f} {r['pop_median']:>9.4f} "
              f"{r['rank']:>6} {r['pop_n']:>5} {r['z']:>+7.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
