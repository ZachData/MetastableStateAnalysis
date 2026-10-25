"""B7 unit 2's diagnostics after `/challenge-pr` on #276, outside the frozen instrument: they re-use
`b7_activity.py`'s counts, draws and seeds and change one thing each, so none is a second reading of
the grid. Tier 1: exploratory, unregistered.

    python -m trackB_circuit.b7_diag tail <read dir>      # findings 1: the interval with h* fixed
    python -m trackB_circuit.b7_diag prompts <read dir>   # finding 2: rows by prompt, the modal key

`tail`: per read cell, the description the frozen rule gives when every draw keeps the sample's
per-fold h* (P, N0 and each distant layer's P alike), against the recorded one; and, beside, the one
given on draws with ≥ 5 distinct prompts (the PR's first sizing, which also cuts sampling noise, so
it can only narrow an interval). Counts by recorded description only; no cell is listed.
`prompts`: per read cell, each prompt's share of the entering heads' jointly condensed rows (sample
h*), and per prompt the share of its condensed rows at layer l that sit on the key most heads of
layers 7–20 point at, with that key's position and decoded token.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from trackB_circuit import b7_activity as b

READ_STEPS = (512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
MIDDLE = range(7, 21)


def _records(d: str):
    return {s: json.loads((Path(d) / f"step{s}.json").read_text()) for s in READ_STEPS}


def _stages(step: int):
    return [b.prompt_stage(b.load_attn(b.run_dir(step, p) / "attentions.npz", "cuda")) for p in b.V1]


def fixed(agree, joint, M, hs0, enter, extras=None):
    """Per-draw cell values with each fold's h* held at the sample's (hs0: (Q, H))."""
    torch = b._torch()
    idx = hs0[..., None]

    def take(x):
        return M @ x.gather(-1, idx).squeeze(-1)                              # (D, H)

    Jh = take(joint)
    e = torch.as_tensor(enter, device=agree.device)
    den = Jh[:, e].sum(1)
    out = {"P": (take(agree)[:, e].sum(1) / den).cpu().numpy()}
    for k, v in (extras or {}).items():
        out[k] = (take(v)[:, e].sum(1) / den).cpu().numpy()
    return out


def tail(d: str) -> None:
    torch = b._torch()
    recs = _records(d)
    moved, kept5 = defaultdict(Counter), defaultdict(Counter)
    for step, rec in recs.items():
        cnt = b.pair_counts(_stages(step), lags=(1,))
        M = b.draws(cnt["Q"], b.N_BOOT, b.seed_for("B7", "read", step), cnt["agree"].device)
        dist = (M[1:] > 0).sum(1).cpu().numpy()
        ag, jt, n0 = cnt["agree"], cnt["joint"], cnt["n0"]
        for c in rec["cells"]:
            if c["description"] in (b.DESCRIPTIONS[0], b.DESCRIPTIONS[1]):
                continue
            l = c["l"]
            ex = {"N0s": n0[(l, l + 1)][0], "N0r": n0[(l, l + 1)][1]}
            m = b.matched(ag[l, l + 1], jt[l, l + 1], M, ex)
            f = fixed(ag[l, l + 1], jt[l, l + 1], M, torch.as_tensor(m["hstar"], device=ag.device), m["enter"], ex)
            far_m, far_f = [], []
            for l2 in range(cnt["L"]):
                if abs(l2 - l) >= 2:
                    mm = b.matched(ag[l, l2], jt[l, l2], M)
                    if mm["n_enter"] >= b.MIN_HEADS:
                        far_m.append(mm["P"])
                        far_f.append(fixed(ag[l, l2], jt[l, l2], M, torch.as_tensor(mm["hstar"], device=ag.device),
                                           mm["enter"])["P"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                N1m, N1f = np.nanmean(np.stack(far_m), 0), np.nanmean(np.stack(far_f), 0)
            N0m, N0f = m["N0s"] / m["N0r"], f["N0s"] / f["N0r"]
            assert abs(m["P"][0] - c["P"]) < 1e-12 and abs(N1m[0] - c["N1"]) < 1e-9, (step, l)
            assert abs(f["P"][0] - c["P"]) < 1e-12 and abs(N1f[0] - c["N1"]) < 1e-9, (step, l)
            desc_f = b.describe(c["n_enter"], float(N0f[0]), b.interval(f["P"] - N0f), b.interval(f["P"] - N1f))[0]
            sel = np.r_[True, dist >= 5]
            x0, x1 = (m["P"] - N0m)[sel], (m["P"] - N1m)[sel]
            desc_5 = b.describe(c["n_enter"], float(N0m[0]), b.interval(x0), b.interval(x1))[0]
            moved[c["description"]][desc_f != c["description"]] += 1
            kept5[c["description"]][desc_5 != c["description"]] += 1
        print(f"step {step} done", file=sys.stderr, flush=True)
    for k in moved:
        n = sum(moved[k].values())
        print(f"{k}: {n} cells; another description with h* fixed {moved[k][True]}; "
              f"on draws with >= 5 distinct prompts {kept5[k][True]}")


def prompts(d: str) -> None:
    from transformers import AutoTokenizer
    from core.prompts import PROMPTS
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    recs = _records(d)
    names = list(b.V1)
    print("| step | prompt | modal key (heads of 7–20) | token | middle cells: row share median (min–max) "
          "| middle cells: share of its condensed rows on the key, median |")
    print("|---|---|---|---|---|---|")
    for step in (2000, 8000, 143000):
        st = _stages(step)
        cnt = b.pair_counts(st, lags=(1,))
        M = b.draws(cnt["Q"], 0, 0, cnt["agree"].device)
        torch = b._torch()
        share, onkey = defaultdict(list), defaultdict(list)
        keys = {}
        for q, s in enumerate(st):
            part = s["part"].cpu().numpy()
            c = Counter()
            for l in MIDDLE:
                for h in range(b.N_HEADS):
                    v = part[l, h][part[l, h] >= 0]
                    if v.size >= 10:
                        c[int(np.bincount(v).argmax())] += 1
            keys[q] = c.most_common(1)[0] if c else (None, 0)
        for cell in recs[step]["cells"]:
            l = cell["l"]
            if l not in MIDDLE or cell["description"] == b.DESCRIPTIONS[0]:
                continue
            m = b.matched(cnt["agree"][l, l + 1], cnt["joint"][l, l + 1], M)
            hs = torch.as_tensor(m["hstar"], device=cnt["agree"].device)
            e = torch.as_tensor(m["enter"], device=cnt["agree"].device)
            J = cnt["joint"][l, l + 1].gather(-1, hs[..., None]).squeeze(-1)[:, e].sum(1).cpu().numpy()
            for q, s in enumerate(st):
                share[q].append(J[q] / J.sum())
                part = s["part"][l].cpu().numpy()[m["enter"]]
                v = part[part >= 0]
                if v.size and keys[q][0] is not None:
                    onkey[q].append(float((v == keys[q][0]).mean()))
        for q, p in enumerate(names):
            k, n = keys[q]
            ids = tok(PROMPTS[p])["input_ids"]
            t = repr(tok.decode(ids[k])) if k is not None and k < len(ids) else "—"
            sh = np.array(share[q])
            print(f"| {step} | {p} | {k} ({n} heads) | {t} | {np.median(sh):.2f} ({sh.min():.2f}–{sh.max():.2f}) | "
                  f"{np.median(onkey[q]):.2f} |")


if __name__ == "__main__":
    {"tail": tail, "prompts": prompts}[sys.argv[1]](sys.argv[2])
