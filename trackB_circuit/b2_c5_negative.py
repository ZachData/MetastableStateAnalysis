"""B2 unit 2, beside (`/challenge-pr` on #221, finding 4): per step, the heads whose V (LN1 and raw
frame) has no eigenvalue with positive real part. There `b2.c5_row` reads "unclassified" (its top is
one of the d − 64 zeros), while the dense `classify_ov_row` decides on rounding noise among them.
Weights only, float64 CPU. Tier 1: exploratory, unregistered.
    METS_DATA=<data> python -m trackB_circuit.b2_c5_negative --out <dir>/c5_negative.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trackB_circuit import b2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = Path(a.out)
    if p.exists():
        raise SystemExit(f"refusing: {p} exists")
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    rec = {"code": code, "b2_code": b2.code_sha(), "steps": {}}
    for s in b2.STEPS:
        Ws, _ = b2.load_weights(s)
        row = {}
        for f in ("ln", "raw"):
            neg = [i for i, W in enumerate(Ws)
                   if np.linalg.eigvals(b2.v_factors(W, f)[1].T @ b2.v_factors(W, f)[0]).real.max() <= 0]
            row[f] = len(neg)
            row[f"{f}_members"] = [m for m in b2.MEMBERS if b2.head_index(m) in neg]
        rec["steps"][s] = row
        print(s, row, flush=True)
    p.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
