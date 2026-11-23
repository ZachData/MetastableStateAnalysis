r"""
trackC_writeup/c4peg_close.py — C4′ (e)+(g) close-out, post hoc: do the two schedules' means
predict the same tokens? (U1's Parked 1; Blocked 171 E (a): at the close-out, marked post hoc.)

For a pair of arms (X, Y) at the same step and state l, z_X = M_X m̄_X^l with M = W_U diag γ_f at
states 0–5 and W_U at state 6 (E1's readouts, each arm's own checkpoint), over the vocabulary with
count ≥ 1. Read:

- **r_xy** = Spearman(z_X, z_Y): do the two means rank the tokens alike?
- **r_xy | f** = the partial rank correlation with log f's ranks removed from both: the agreement
  beyond what both sharing the unigram prior forces (E1 already found each r(z, log f) ≈ 0.4–0.9).

Pairs: (A0, F) and (A0′, F′), the question; (A0, A0′) and (F, F′), the order-only scale. No bar:
post hoc, description only, read after U1 and U2.

    python -m trackC_writeup.c4peg_close logits --u1 <U1 records dir> --out <dir>    # CPU
    python -m trackC_writeup.c4peg_close g3 --u2 <U2 records dir> --out <dir>        # GPU, U2's Parked 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_train as ct
from trackC_writeup import c4peg_read as e

PAIRS = {"A0|F": ("A0", "F"), "A0p|Fp": ("A0p", "Fp"), "A0|A0p": ("A0", "A0p"), "F|Fp": ("F", "Fp")}
STEPS = (1, 4, 16, 24, 48, 128, 256, 384, 640, 1000, 2000, 4000)
STATES = (3, 6)
REFS = ("seed1", "seed2")                    # other inits at 4000: the read's own scale (`/challenge-pr` on #370)
PREV_SETS = {"A0": (("L2H1",),), "A0p": (("L2H1",),),
             "F": (("L1H7",), ("L1H3",), ("L1H3", "L1H7")),
             "Fp": (("L1H6",), ("L1H3",), ("L1H3", "L1H6"))}   # each arm's top previous-token heads, alone


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4peg_close.py", "trackC_writeup/c4peg_read.py", "trackC_writeup/c4peg_g.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def partial_rank(z1: np.ndarray, z2: np.ndarray, ell: np.ndarray) -> float:
    """Spearman of z1, z2 with the ranks of ell regressed out of each (standardised ranks)."""
    r1, r2, rl = (e._std_ranks(e.rank_avg(v)) for v in (z1, z2, ell))
    a = r1 - (r1 @ rl) * rl
    b = r2 - (r2 @ rl) * rl
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _wu_gamma(arm: str, step: int):
    if arm in REFS:
        from trackC_writeup import c4peg_g as g
        m = g.load(arm, step, "cpu")
        sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    else:
        from safetensors.numpy import load_file
        sd = load_file(str(ct.ckpt_dir(e.CKPT_ROOT[arm], arm, step) / "model.safetensors"))
    return sd["embed_out.weight"].astype(np.float64), sd["gpt_neox.final_layer_norm.weight"].astype(np.float64)


def readout(arm: str, step: int, l: int, mbar: np.ndarray, sup: np.ndarray) -> np.ndarray:
    WU, gam = _wu_gamma(arm, step)
    M = WU if l == e.N_STATES - 1 else WU * gam[None, :]
    return M[sup] @ np.asarray(mbar, np.float64)


REF_REPOS = {"seed1": "EleutherAI/pythia-70m-seed1", "seed2": "EleutherAI/pythia-70m-seed2"}


def _register_refs() -> None:
    from trackC_writeup import c4peg_g as g
    for k, v in REF_REPOS.items():
        g.REPOS.setdefault(k, v)


def ref_mbar(kind: str, step: int = 4000) -> np.ndarray:
    """m̄ (7 states × 512) of a released model, as U1 computes it for the arms (v1's 7 prompts)."""
    import os
    from trackC_writeup import c4peg_g as g
    from trackA_force.a1a import V1
    from trackA_force.a5_seeds import TOKENIZER
    from transformers import AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    m = g.load(kind, step, "cpu")
    ms = []
    for p in V1:
        hs, _, _ = e.forward(m, tok, p, "cpu")
        ms.append([e.unit_mean(h) for h in hs])
    return np.mean(np.asarray(ms), 0)


def logits(a) -> int:
    _register_refs()
    u1 = Path(a.u1)
    cnt, c_sha = e.load_counts(u1)
    sup = np.flatnonzero(cnt > 0)
    ell = np.log(cnt[sup] / cnt.sum())
    mb = {arm: {s: np.asarray(json.loads((u1 / arm / f"step{s}.json").read_text())["mbar"]) for s in STEPS}
          for arm in e.ARMS}
    res = {"code": code_sha(), "counts_sha256": c_sha, "post_hoc": True, "rows": []}
    chk = float(np.max(np.abs(ref_mbar("A0") - mb["A0"][4000])))     # known answer: this path is U1's
    if chk > 1e-5:
        raise RuntimeError(f"ref_mbar(A0) differs from U1's record by {chk}")
    res["known_answer_mbar_A0"] = chk
    for k in REFS:
        mb[k] = {4000: ref_mbar(k)}
    for s in STEPS:
        for l in STATES:
            arms = e.ARMS + (REFS if s == 4000 else ())
            z = {arm: readout(arm, s, l, mb[arm][s][l], sup) for arm in arms}
            row = {"step": s, "state": l, "r_f": {arm: e.spearman(z[arm], ell) for arm in arms}}
            pairs = dict(PAIRS)
            if s == 4000:
                pairs.update({f"{x}|{r}": (x, r) for r in REFS for x in e.ARMS})
                pairs["seed1|seed2"] = REFS
            for k, (x, y) in pairs.items():
                row[k] = {"r": e.spearman(z[x], z[y]), "r_given_f": partial_rank(z[x], z[y], ell)}
            res["rows"].append(row)
            print(json.dumps({"step": s, "state": l, **{k: [round(row[k]["r"], 3), round(row[k]["r_given_f"], 3)]
                                                         for k in PAIRS}}), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ct._jdump(out / "logits_post_hoc.json", res)
    return 0


def copy_losses(model, seq: np.ndarray, device) -> Tuple[float, float]:
    """Mean loss on the first copy's targets 1 … L−1 and the second copy's L+1 … 2L−1."""
    import torch
    import torch.nn.functional as F
    L = seq.shape[1] // 2
    s1 = s2 = 0.0
    n1 = n2 = 0
    with torch.no_grad():
        xs = torch.from_numpy(seq).to(device)
        for k in range(0, len(xs), 16):
            x = xs[k:k + 16]
            lg = model(input_ids=x).logits.float()
            for (lo, hi), acc in (((0, L - 1), 1), ((L, 2 * L - 1), 2)):
                p, t = lg[:, lo:hi], x[:, lo + 1:hi + 1]
                v = float(F.cross_entropy(p.reshape(-1, p.shape[-1]), t.reshape(-1), reduction="sum"))
                if acc == 1:
                    s1, n1 = s1 + v, n1 + t.numel()
                else:
                    s2, n2 = s2 + v, n2 + t.numel()
    return s1 / n1, s2 / n2


def share_removed(g0: float, g: float) -> float:
    """The share of the copy gain (first-copy loss minus second-copy loss) an ablation removes."""
    return (g0 - g) / g0


def prev_token_scores(model, seq: np.ndarray, device, n: int = 16) -> np.ndarray:
    """Per head, the mean attention from first-copy position i (1 ≤ i < L) to i − 1."""
    import torch
    L = seq.shape[1] // 2
    with torch.no_grad():
        att = model(input_ids=torch.from_numpy(seq[:n]).to(device), output_attentions=True).attentions
    A = torch.stack(att, 1).float().cpu().numpy()                  # (n, layers, heads, T, T)
    i = np.arange(1, L)
    return A[:, :, :, i, i - 1].mean((0, -1))


def g3(a) -> int:
    """U2's Parked 1, post hoc: each G2 ablation re-scored as the share of the copy gain it
    removes, and a previous-token score per head at 4000."""
    import os
    from trackC_writeup import c4peg_g as g
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = g._torch()
    u2 = json.loads((Path(a.u2) / "g2.json").read_text())
    x = g.held_rows()
    seq = g.induction_batch()
    res = {"code": code_sha(), "post_hoc": True, "arms": {}}
    for arm in e.ARMS:
        m = g.load(arm, 4000, device)
        means = g.dense_means(m, x, device)
        l1, l2 = copy_losses(m, seq, device)
        if abs(l2 - u2["arms"][arm]["base"]["ind"]) > 1e-5 * l2:
            raise RuntimeError(f"{arm}: second-copy loss {l2} ≠ U2's {u2['arms'][arm]['base']['ind']}")
        g0 = l1 - l2
        rows = {}
        named = dict(u2["arms"][arm]["sets"])
        named.update({"prev:" + "+".join(s): {"heads": list(s)} for s in PREV_SETS[arm]})
        for name, d in named.items():
            hk = g.ablate(m, d["heads"], means)
            try:
                b1, b2 = copy_losses(m, seq, device)
            finally:
                for h in hk:
                    h.remove()
            rows[name] = {"heads": d["heads"], "l1": b1, "l2": b2, "share": share_removed(g0, b1 - b2)}
        pt = prev_token_scores(m, seq, device)
        res["arms"][arm] = {"l1": l1, "l2": l2, "gain": g0, "sets": rows, "prev_token": pt.tolist()}
        rnd = sorted((v["share"] for k, v in rows.items() if k.startswith("random")), reverse=True)
        top = np.unravel_index(np.argsort(-pt, axis=None)[:3], pt.shape)
        print(json.dumps({"arm": arm, "gain": round(g0, 3), "own": round(rows["own"]["share"], 3),
                          "other": round(rows["other"]["share"], 3), "random_top3": [round(v, 3) for v in rnd[:3]],
                          "prev": {k: round(v["share"], 3) for k, v in rows.items() if k.startswith("prev:")},
                          "prev_token_top3": [f"L{l}H{h} {pt[l, h]:.2f}" for l, h in zip(*top)]}), flush=True)
        del m
        torch.cuda.empty_cache()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ct._jdump(out / "g3_post_hoc.json", res)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(required=True)
    q = sub.add_parser("logits")
    q.add_argument("--u1", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=logits)
    q = sub.add_parser("g3")
    q.add_argument("--u2", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=g3)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
