"""From the saved regate C1 records: (i) G2's false pass for C1 on the 9-step T (frozen); (ii) the
frozen T and the signed-then-abs alternative S = mean over members of |mean over steps of (p - 1/2)|
on G3m's planted sets, all and late; (iii) S's false pass. Non-member fake sets only. Beside unit 1b
(`trackB_circuit/status-B.md` "B2 unit 1b"), after `/challenge-pr` on #220; not a gate arm.

    METS_DATA=<data> python -m trackB_circuit.b2_regate_alt [<regate dir>]
"""
import sys

import numpy as np

from trackB_circuit import b2
from trackB_circuit.b2a import seed_for

D = sys.argv[1] if len(sys.argv) > 1 else str(b2.DATA / "b2/2026-10-18-regate")
Z = {s: np.load(f"{D}/g3m_step{s}.npz") for s in b2.STAT_STEPS}
heads = Z[512]["heads"]; pos = {int(h): k for k, h in enumerate(heads)}
F = b2.fake_sets([b2.parse_head(m)[0] for m in b2.MEMBERS], b2.N_DRAW, seed_for("b2-g3"))
def stats(P):                       # P: (steps, 384) percentiles
    return np.abs(P - 0.5).mean(0), np.abs((P - 0.5).mean(0))
def rank(Dh, f, seed):
    return b2.member_rank(Dh, f, seed)["rank"]
def run(q, on):
    out = {"T": [], "S": []}
    for k, f in enumerate(F):
        P = []
        for s in b2.STAT_STEPS:
            x = Z[s]["C1"].copy()
            if q is not None and s in on:
                x[f] = Z[s][f"planted_q{q:.2f}"][[pos[int(i)] for i in f]]
            P.append(b2.percentiles(x))
        T, S = stats(np.array(P))
        out["T"].append(rank(T, f, seed_for("x-T", q, k))); out["S"].append(rank(S, f, seed_for("x-S", q, k)))
    return {k: float(np.mean(np.array(v) >= 0.99)) for k, v in out.items()}
print("false pass (unplanted, 9 steps):", run(None, ()))
late = [s for s in b2.STAT_STEPS if s >= 8000]
for q in (0.70, 0.75, 0.80, 0.90):
    print(q, "all", run(q, b2.STAT_STEPS), "late", run(q, late))
