"""C2 unit 5 (after `/challenge-pr` on #260, finding 2): the tokens' mean term ‖m‖² by layer,
on the ten seeds, from the centred-count records already on disk. No forward pass.

Units 3 and 3s recorded, per prompt and state, the mean off-diagonal cosine of the unit token
cloud, cos = (‖Σ uᵢ‖² − n) / (n (n − 1)), and read its layer profile after the dates (post hoc,
`status-C.md` "C2 unit 3s"). That reading was quoted in prose but not saved. This recomputes it as
a record so the draft can quote it as keys. Per prompt ‖m‖² = (1 + (n − 1) cos) / n exactly
(E₁'s first-order term is ½‖m‖², `tools/math_checks/c2_energy_moments.py`); the profile is its
mean over the 7 prompts at each of the 25 states. A transition *falls* when the profile is lower
at the next state. Counted over all 24 transitions and over the 22 that do not change frame (the
first, embeddings into block 0's output, and the last, into the final layer norm, left out).

Description only, read after every date it is set beside; no rule, no threshold.

Run: METS_DATA=<main>/data python -m trackC_writeup.c2_mean_profile
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackC_writeup.c2_seeds import C2U3, load_seed

RUN = DATA / "trackC" / "c2u3s" / "2026-10-22-run"
OUT = DATA / "trackC" / "c2u5" / "2026-10-23-mean"
SEEDS = range(0, 10)


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_mean_profile.py", "trackC_writeup/c2_seeds.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def m2_profile(recs: Dict, step: int) -> List[float]:
    """The prompt-mean of ‖m‖² at each state."""
    per = [[(1 + (r["n"] - 1) * c) / r["n"] for c in r["cos"]]
           for r in (recs[(step, p)] for p in V1)]
    return np.mean(np.array(per), axis=0).tolist()


def falls(profile: List[float], inner: bool) -> int:
    steps = range(1, len(profile) - 2) if inner else range(len(profile) - 1)
    return sum(profile[i + 1] < profile[i] for i in steps)


def main() -> int:
    out: Dict[str, dict] = {}
    for s in SEEDS:
        recs = load_seed(DATA / C2U3 / "records", None) if s == 0 else \
            load_seed(RUN / f"seed{s}" / "records", s)
        prof = {st: m2_profile(recs, st) for st in STEPS}
        out[str(s)] = {"profile": {str(st): p for st, p in prof.items()},
                       "falls_all": {str(st): falls(p, False) for st, p in prof.items()},
                       "falls_inner": {str(st): falls(p, True) for st, p in prof.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"code": code_sha(), "steps": list(STEPS), "seeds": list(SEEDS),
           "transitions_all": 24, "transitions_inner": 22, "seeds_data": out}
    (OUT / "profile.json").write_text(json.dumps(rec, indent=1) + "\n")
    for s in SEEDS:
        print(s, {st: out[str(s)]["falls_all"][str(st)] for st in (128, 256, 512, 1000)},
              {st: out[str(s)]["falls_inner"][str(st)] for st in (128, 256, 512, 1000)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
