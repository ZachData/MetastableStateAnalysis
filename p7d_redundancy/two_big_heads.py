"""`L5H2` and `L7H8`: are they a circuit, or two heads doing the same job?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.12-Q3 found that ablating `L5H2`'s OV costs **+2.227** against
`L7H8`'s +1.107 -- the prev-token head has TWICE the effect of the matcher the
whole of §3.11-§3.12 was spent on -- and that both remove 11-16 % of the final
residual norm where an ordinary head removes under 0.4 %. §3.12-R then ruled out
the trivial reading: operator norm explains 0.1 % of the variance and the
relationship runs backwards.

Three things follow that nobody has measured, and the first is an omission.

PROBE A -- THE 2x2 THAT WAS NEVER RUN. §3.12-P ran the ablation interaction for
`L7H8` x each downstream COPIER and found redundancy. It never ran it for
`L7H8` x `L5H2` -- **the pair that is supposed to BE the circuit.** A serial
two-stage circuit predicts SUB-additivity: remove the prev-token head and the
matcher has less to match on, so ablating it too should cost less than it does
alone. Redundancy predicts super-additivity. This is the direct test of whether
the induction circuit behaves like a circuit, and it is four forward passes.

PROBE B -- DO THEY WRITE THE SAME THING? Two ways, and they answer different
questions. WEIGHTS: principal angles between the two heads' write subspaces
(`col W_O`) and read subspaces (`row W_V`) -- do the operators overlap at all?
ACTIVATIONS: the cosine between the mean residual-DELTA vectors that each
ablation produces. Weight overlap says the machinery could collide; delta
alignment says it actually does. A pair that is redundant in Probe A but
orthogonal here would be redundant through the downstream computation rather
than by writing the same direction.

PROBE C -- WHERE DOES THE NORM GO? §3.12-Q measured the residual norm only at
the FINAL layer. The per-layer profile says whether the missing 11-16 % is
removed at the head's own layer and carried forward, or whether it compounds --
a write that changes LayerNorm's scaling for everything after it would grow
down the stack rather than staying flat, and that is the difference between "a
big write" and "a change of regime".

NO p-value; nothing registered. `L5H2`'s under-characterisation is the largest
gap §3.12 left and this is the cheapest part of closing it.
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

OUT = DATA / "analysis" / "two_big_heads.json"
A_HEAD = (5, 2)      # prev-token, dNLL +2.227
B_HEAD = (7, 8)      # matcher,    dNLL +1.107


def batch(rng, n):
    return torch.tensor(np.stack([
        np.concatenate([s, s]) for s in
        (rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP) for _ in range(n))
    ]), dtype=torch.long)


def wv_wo(model, layer, head):
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight
    q3 = qkv.detach().numpy().astype(np.float64).reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    dense = model.gpt_neox.layers[layer].attention.dense.weight
    W_O = dense.detach().numpy().astype(np.float64)[:, head * D_HEAD:(head + 1) * D_HEAD]
    return q3[head, 2 * D_HEAD:, :], W_O


@torch.no_grad()
def probe(model, ids, chunk=8):
    """(nll, per-layer resid norms, mean final-layer resid vector)."""
    lc, lz, norms, vecs = [], [], None, []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_hidden_states=True)
        lg = out.logits[:, :-1, :].float()
        t = ids[i:i + chunk, 1:]
        z = torch.logsumexp(lg, dim=-1)
        c = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        lc.append(c[:, N_REP - 1:].numpy().astype(np.float64))
        lz.append(z[:, N_REP - 1:].numpy().astype(np.float64))
        hs = [h[:, N_REP - 1:, :].float() for h in out.hidden_states]
        n = np.stack([float(h.norm(dim=-1).mean()) for h in hs])
        norms = n if norms is None else norms + n
        vecs.append(hs[-1].reshape(-1, D_MODEL).mean(0).numpy().astype(np.float64))
        del out, lg, z, c, hs
    nb = int(np.ceil(len(ids) / chunk))
    return (float(-(np.concatenate(lc).mean() - np.concatenate(lz).mean())),
            norms / nb, np.mean(vecs, axis=0))


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
    Z = (np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))

    def run(ablate):
        saved = {k: ov_factors(model, *k) for k in ablate}
        for k in ablate:
            write_ov(model, *k, *Z)
        r = probe(model, ids)
        for k, (a, b) in saved.items():
            write_ov(model, *k, a, b)
        return r

    nll0, norm0, vec0 = probe(model, ids)
    nll_a, norm_a, vec_a = run([A_HEAD])
    nll_b, norm_b, vec_b = run([B_HEAD])
    nll_ab, norm_ab, _ = run([A_HEAD, B_HEAD])
    nll_chk, _, _ = probe(model, ids)

    dA, dB, dAB = nll_a - nll0, nll_b - nll0, nll_ab - nll0
    inter = dAB - dA - dB

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "step": args.step,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "restore_abs_diff": abs(nll_chk - nll0),
           "probeA": {"baseline_nll": nll0, "d_L5H2": dA, "d_L7H8": dB,
                      "d_both": dAB, "interaction": inter}}

    print(f"PROBE A -- the 2x2 that was never run  (step {args.step}, "
          f"restore {abs(nll_chk-nll0):.1e})")
    print(f"    baseline NLL          {nll0:>8.4f}")
    print(f"    dNLL(L5H2 alone)      {dA:>+8.4f}")
    print(f"    dNLL(L7H8 alone)      {dB:>+8.4f}")
    print(f"    dNLL(both)            {dAB:>+8.4f}   sum would be {dA+dB:+.4f}")
    print(f"    INTERACTION           {inter:>+8.4f}   "
          f"{'SUB-additive (serial circuit)' if inter < -0.02 else 'SUPER-additive (redundant)' if inter > 0.02 else 'independent'}")

    # ---- PROBE B ------------------------------------------------------------
    Va, Oa = wv_wo(model, *A_HEAD)
    Vb, Ob = wv_wo(model, *B_HEAD)

    def angles(X, Y):
        Qx, _ = np.linalg.qr(X)
        Qy, _ = np.linalg.qr(Y)
        s = np.clip(np.linalg.svd(Qx.T @ Qy, compute_uv=False), -1, 1)
        return float(s.mean()), float(s.max())

    w_mean, w_max = angles(Oa, Ob)             # write subspaces (col W_O)
    r_mean, r_max = angles(Va.T, Vb.T)         # read subspaces  (row W_V)
    da, db = vec_a - vec0, vec_b - vec0
    cos_delta = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db)))
    res["probeB"] = {"write_subspace_mean_cos": w_mean, "write_subspace_max_cos": w_max,
                     "read_subspace_mean_cos": r_mean, "read_subspace_max_cos": r_max,
                     "residual_delta_cosine": cos_delta,
                     "delta_norm_L5H2": float(np.linalg.norm(da)),
                     "delta_norm_L7H8": float(np.linalg.norm(db))}
    print(f"\nPROBE B -- do they write the same thing?")
    print(f"    write subspaces (col W_O)  mean cos {w_mean:.3f}  max {w_max:.3f}")
    print(f"    read  subspaces (row W_V)  mean cos {r_mean:.3f}  max {r_max:.3f}")
    print(f"    residual-DELTA cosine      {cos_delta:+.3f}   "
          f"(|dA| {np.linalg.norm(da):.2f}, |dB| {np.linalg.norm(db):.2f})")
    print(f"    random 64-dim subspaces of R^1024 would give mean cos ~"
          f"{np.sqrt(64/1024):.3f}")

    # ---- PROBE C ------------------------------------------------------------
    res["probeC"] = {"baseline": norm0.tolist(), "ablate_L5H2": norm_a.tolist(),
                     "ablate_L7H8": norm_b.tolist(), "ablate_both": norm_ab.tolist()}
    print(f"\nPROBE C -- per-layer residual norm (hidden_states index = block input)")
    print(f"{'idx':>4} {'baseline':>10} {'-L5H2':>10} {'-L7H8':>10} {'-both':>10}")
    for i in range(0, len(norm0), 2):
        print(f"{i:>4} {norm0[i]:>10.2f} {norm_a[i]:>10.2f} "
              f"{norm_b[i]:>10.2f} {norm_ab[i]:>10.2f}")
    i = len(norm0) - 1
    print(f"{i:>4} {norm0[i]:>10.2f} {norm_a[i]:>10.2f} "
          f"{norm_b[i]:>10.2f} {norm_ab[i]:>10.2f}   <- final")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
