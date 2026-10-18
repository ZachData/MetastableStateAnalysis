"""B2 unit 2's tables as markdown, from `table.json` (`b2.py table`). `main` only formats; `pairs`
computes C6 by pair from the saved `step<S>.npz` and records it in `c6_pairs.json`.
Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b2_show <dir>/table.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter

REACH = ("cannot exclude members held between their step's 25th and 75th, nor a late onset "
         "(from 8000) below about the 90th")


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main(path: str) -> None:
    t = json.load(open(path))
    steps = t["steps"]
    print(f"code `{t['code']}`, checks: {t['checks']}\n")
    print("| column | T (or C6) | rank | null 50 / 95 / 99 % | reads |\n|---|---|---|---|---|")
    for c in ("C1", "C2", "C3", "C4", "C6"):
        r = t["ranks"][c]
        q = r["null_q"]
        print(f"| {c} | {r.get('T', r.get('C6')):.4f} | {r['rank']:.4f} | {q[1]:.4f} / {q[2]:.4f} / {q[3]:.4f} | "
              f"{r['reads']}{'' if r['reads'] == 'members differ' else ' (' + REACH + ')'} |")
    print("\n| beside (not in the family) | T | rank |\n|---|---|---|")
    for c, r in t["beside_ranks"].items():
        print(f"| {c} | {r.get('T', r.get('C6')):.4f} | {r['rank']:.4f} |")
    for c in ("C1", "C2", "C3", "C4"):
        lo, hi = t["bands_step0"][c]
        print(f"\n**{c}** (step-0 band {lo:.4f}–{hi:.4f}; cell = value / percentile, * = moved)\n")
        print("| step | " + " | ".join(t["members"]) + " | pop. moved |")
        print("|---|" + "---|" * (len(t["members"]) + 1))
        for k, s in enumerate(steps):
            cells = [t["cells"][c][m][k] for m in t["members"]]
            print(f"| {s} | " + " | ".join(f"{x['value']:.3f} / {x['p']:.2f}{'*' if x['moved'] else ''}" for x in cells)
                  + f" | {t['moved_share_pop'][c][str(s)]:.2f} |")
    print("\n**C6 per step** (circuit mean overlap; chance 1/64 = 0.0156)\n\n| step | r16 | r8 | r32 | write–write r16 |\n|---|---|---|---|---|")
    for s in steps:
        print(f"| {s} | {t['ranks']['C6']['per_step'][str(s)]:.4f} | {t['beside_ranks']['C6_r8']['per_step'][str(s)]:.4f} | "
              f"{t['beside_ranks']['C6_r32']['per_step'][str(s)]:.4f} | {t['beside_ranks']['WW16']['per_step'][str(s)]:.4f} |")
    r = t["ranks"]["C6"]
    print(f"\nC6 with L10H7 (6 heads, mean over steps ≥ 512): {r['with_L10H7']:.4f}, rank {r['with_L10H7_rank']:.4f}")
    print("\n**C5** (members' rows, LN1 frame; population share of row 2 (both) and unclassified)\n")
    ab = {"row2_three_hyperplanes": "r2", "row2_eigen_only_qk_fails": "r2e", "unclassified": "u",
          "row1_V_identity": "r1", "row4_V_negative_identity": "r4"}
    print("| step | " + " | ".join(t["members"]) + " | pop r2 / r2e / u | raw pop r2 / r2e / u |")
    print("|---|" + "---|" * (len(t["members"]) + 2))
    for s in steps:
        c5 = t["C5"][str(s)]
        sh, shr = c5["pop_shares"], c5["pop_shares_raw"]
        print(f"| {s} | " + " | ".join(ab[c5["members"][m]] + ("" if c5["members"][m] == c5["members_raw"][m] else f" ({ab[c5['members_raw'][m]]})")
                                     for m in t["members"])
              + " | " + " / ".join(f"{sh.get(k, 0):.2f}" for k in ab if k in ("row2_three_hyperplanes", "row2_eigen_only_qk_fails", "unclassified"))
              + " | " + " / ".join(f"{shr.get(k, 0):.2f}" for k in ab if k in ("row2_three_hyperplanes", "row2_eigen_only_qk_fails", "unclassified")) + " |")
    print("\n**Negative-pair flag and 2d regime** (members flagged; population rate, LN1 / raw)\n")
    print("| step | flagged members | pop rate | raw | members' regime (non-'outside') | pop regimes |\n|---|---|---|---|---|---|")
    for s in steps:
        pr, rg = t["pair"][str(s)], t["regime"][str(s)]
        fl = [m for m, v in pr["members"].items() if v] or ["—"]
        nr = [f"{m} {v}" for m, v in rg["members"].items() if v != "outside"] or ["—"]
        print(f"| {s} | {', '.join(fl)} | {pr['pop_rate']:.3f} | {pr['pop_rate_raw']:.3f} | {', '.join(nr)} | {rg['pop']} |")
    print("\n**C7** r90 (smallest D with TV(D) ≤ 0.1 TV(0); — none)\n")
    heads = list(t["C7"])
    print("| head | " + " | ".join(str(s) for s in steps) + " |\n|---|" + "---|" * len(steps))
    for h in heads:
        print(f"| {h} | " + " | ".join(fmt(t["C7"][h][str(s)]) for s in steps) + " |")
    inV = Counter(tuple(x) for s in steps for x in t["inertia"][str(s)]["V"])
    inM = Counter(tuple(x) for s in steps for x in t["inertia"][str(s)]["M"])
    print(f"\nInertia (distinct (n₊, n₋) over steps): V {dict(inV)}, M_0 {dict(inM)}")


def pairs(path: str) -> None:
    """C6 by ordered pair (descriptive, beside; read after the rank, `/challenge-pr` on #221): each
    pair's overlap at r = 16, mean over steps ≥ 512, its percentile among the 16 × 16 head pairs
    between the same two layers, and, per member layer, the heads whose keys read `L5H2`'s write
    space most. Computed here, not in `b2.py`, so it writes `c6_pairs.json` beside `table.json`
    with this file's hash."""
    import hashlib
    import numpy as np
    from pathlib import Path
    from trackB_circuit import b2
    d = Path(path).parent
    O = np.mean([np.load(d / f"step{s}.npz")["O16"] for s in b2.STAT_STEPS], 0)
    circ = sorted((b2.head_index(m) for m in b2.CIRCUIT + ("L10H7",)), key=lambda i: i // 16)
    rec = {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
           "table_code": json.load(open(path))["code"], "r": b2.R_SUB, "steps": b2.STAT_STEPS,
           "pairs": [], "l5h2_readers": {}}
    print("\n**C6 by pair** (A writes, B's keys read; r = 16, mean over steps ≥ 512; percentile among "
          "the 256 pairs between A's and B's layers)\n\n| A → B | overlap | × chance | percentile |\n|---|---|---|---|")
    for a in circ:
        for b in circ:
            if a // 16 < b // 16:
                la, lb = a // 16, b // 16
                blk = O[la * 16:(la + 1) * 16, lb * 16:(lb + 1) * 16].ravel()
                p = float(((blk < O[a, b]).sum() + 0.5 * (blk == O[a, b]).sum()) / blk.size)
                rec["pairs"].append({"A": b2.head_name(a), "B": b2.head_name(b), "overlap": float(O[a, b]), "pct": p})
                print(f"| {b2.head_name(a)} → {b2.head_name(b)} | {O[a, b]:.4f} | {O[a, b] * 64:.1f} | {p:.2f} |")
    a = b2.head_index("L5H2")
    print("\n**Who reads `L5H2`'s write space** (top 4 key spaces per member layer)\n\n| layer | heads (overlap) | member's place of 16 |\n|---|---|---|")
    for m in b2.MEMBERS[1:]:
        lb = b2.head_index(m) // 16
        col = O[a, lb * 16:(lb + 1) * 16]
        o = np.argsort(-col)
        place = int(np.where(o == b2.head_index(m) % 16)[0][0]) + 1
        rec["l5h2_readers"][m] = {"top": [[f"L{lb}H{h}", float(col[h])] for h in o[:4]], "member_place": place}
        print(f"| {lb} | " + ", ".join(f"L{lb}H{h} ({col[h]:.3f})" for h in o[:4]) + f" | {m} {place} |")
    (d / "c6_pairs.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
    pairs(sys.argv[1])
