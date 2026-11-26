"""C9 U2b, after `/challenge-pr` on #383 (finding 4): the snapped arm's known answer, run post hoc.
On Pythia's grid the snapped arm is A3's own effect arm (snapping is the identity; the plant is
A3's), so on A3's units at A3's floors it must give A3's recorded "placed right" rates up to Monte
Carlo error (fresh draws, so not draw for draw). Its own module, so `c9_u2b.py`'s hash (the
reading's record) is unchanged.

    METS_DATA=<main>/data python -m trackC_writeup.c9_u2b_snapcheck
"""
from __future__ import annotations

import json

import numpy as np

from trackA_force import a3
from trackC_writeup import c9_u2 as u
from trackC_writeup import c9_u2b as b

ROWS = ("E2", "E5", "E6u")
MC_TOL = 0.03          # placed: ~3 binomial SEs at p = 0.9, n = 1000 (two independent sets of draws)


def main() -> int:
    A = u.a3_units()
    rec = A["record"]
    out = {"code_c9_u2b": b.code_sha(), "rows": {}}
    ok = True
    for k, rid in enumerate(ROWS):
        _, d, s, _ = a3.ROWS[rid]
        g = rec["gate"][rid]
        sn = b.snapped_arm("410m", u.PYTHIA, d, s, a3.kind_of(rid), A["units"][rid],
                           np.random.default_rng([b.SEED_U2B, 100, k]), g["floor"])
        want = {e["gap"]: e["placed_right"] for e in g["effect"]}
        got = {e["gap"]: e["placed_right_snapped"] for e in sn["effect"]}
        diff = {gap: got[gap] - want[gap] for gap in want}
        m = max(abs(v) for v in diff.values())
        out["rows"][rid] = {"a3": want, "snapped": got, "max_abs_diff": m}
        ok &= m <= MC_TOL
        print(f"{rid}: max |snapped − A3| {m:.3f}", flush=True)
    out["pass"] = bool(ok)
    (b.OUT / "snapped_known.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
