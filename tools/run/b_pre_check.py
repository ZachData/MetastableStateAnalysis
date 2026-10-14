"""B-pre: can `P-ST1`, `P-AB1` and `P-I3` be scored as built (`docs/ROADMAP.md` §7)?

Reads each row's pre-computed requirement (`claims/EVALUABILITY.md` "What the pilot must
produce") against what is on disk, and computes no statistic any of the three gates scores:

- `P-ST1`: `dim U_pos` (the Schur-attractive projector's trace) against the centred population's
  raw effective rank, the ratio the gate's own calibration tabulates (`claims/registry.json`
  P-ST1, "PRECONDITION ON THE PILOT"), and whether `span(U_pos + U_neg)` holds both arms; no
  vector is injected.
- `P-I3`: with 7d's causal induction set (`p7d_redundancy/status-7d.md` "Q1"), a classification
  that is not a cutoff on the behavioural score, how many induction heads have two control heads
  below and two above them in behavioural score within their own layer (the registered key
  `score_and_layer`). Reads behavioural scores only, never motif rates.
- `P-AB1` needs ablation runs that no producer writes; nothing to compute here.

Tier 1: a feasibility check, not a reading.
    python tools/run/b_pre_check.py --data <main>/data --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))
from core.metrics import effective_rank  # noqa: E402

V1_SCORED = ("camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
             "paper_excerpt", "sullivan_ballou", "wiki_paragraph")
STEPS = (0, 1000, 143000)
# Phase 1's v1 sweep, the runs `data/analysis/behavioural_series.json` names for these steps.
RUNS = {0: "2026-09-01_13-30-37", 1000: "2026-08-31_17-05-50", 143000: "2026-09-01_18-25-12"}
# 7d's single-head OV ablation readout at step 16000 (`status-7d.md` "Q1"), by cut.
CAUSAL_SETS = {
    "7d_above_0.2": ["5,2", "7,8", "12,5", "8,6"],
    "7d_above_0.05": ["5,2", "7,8", "12,5", "8,6", "11,14", "8,9", "15,14", "7,1", "9,13", "10,9"],
}
P_I3_STEPS = (16000, 143000)
HALF = 2  # N_CONTROLS_PER_INDUCTION_HEAD / 2


def _one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise SystemExit(f"refuse: {len(hits)} files match {pattern}; need exactly one")
    return Path(hits[0])


def p_st1(data: Path, used: dict) -> dict:
    out = {}
    for st in STEPS:
        pf = _one(str(data / f"phase12/*/ov_projectors_pythia-410m-step{st}.npz"))
        used[str(st)] = str(pf.relative_to(data.parent))
        proj = np.load(pf)
        cells = []
        for pr in V1_SCORED:
            act = np.load(data / f"phase12/{RUNS[st]}/pythia-410m-step{st}_{pr}/activations.npz")
            A = act["activations"].astype(np.float64)
            for L in range(1, A.shape[0]):
                if f"schur_attract_layer_{L - 1}" not in proj:
                    raise SystemExit(f"refuse: {pf} has no layer {L - 1}")
                # activations[0] is the embedding; block L-1's input is activations[L-1]
                X = A[L - 1]
                er = effective_rank(X - X.mean(0), "raw")
                P, N = proj[f"schur_attract_layer_{L - 1}"], proj[f"schur_repulse_layer_{L - 1}"]
                k = int(round(np.trace(P)))
                cells.append({"prompt": pr, "n_tokens": int(X.shape[0]), "layer": L - 1,
                              "dim_u_pos": k, "dim_u_neg": int(round(np.trace(N))),
                              "union_rank": int(np.linalg.matrix_rank(P + N, tol=1e-6)),
                              "er_raw_centred": round(float(er), 2), "ratio": round(k / er, 3)})
        r = np.array([c["ratio"] for c in cells])
        out[str(st)] = {"cells": cells, "n_cells": len(cells),
                        "ratio_min": float(r.min()), "ratio_median": float(np.median(r)),
                        "n_ratio_le_1.5": int((r <= 1.5).sum()), "n_ratio_le_2": int((r <= 2).sum()),
                        "n_ratio_le_3": int((r <= 3).sum()),
                        "union_full_everywhere": all(c["union_rank"] == c["dim_u_pos"] + c["dim_u_neg"]
                                                     for c in cells)}
    return out


def p_i3(data: Path) -> dict:
    b = json.loads((data / "analysis/behavioural_series.json").read_text())
    steps, S = b["steps"], b["series_excl_repeated"]
    out = {}
    for st in P_I3_STEPS:
        i = steps.index(st)
        sc = {k: v[i] for k, v in S.items()}
        for name, ind in CAUSAL_SETS.items():
            rows = []
            for h in ind:
                L = h.split(",")[0]
                same = [k for k in sc if k.split(",")[0] == L and k not in ind]
                below = sum(sc[k] < sc[h] for k in same)
                above = sum(sc[k] > sc[h] for k in same)
                rows.append({"head": f"L{L}H{h.split(',')[1]}", "below_in_layer": below,
                             "above_in_layer": above, "straddles": below >= HALF and above >= HALF})
            out[f"{st}|{name}"] = {"heads": rows,
                                   "n_straddling": sum(r["straddles"] for r in rows)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path, help="the main tree's data/")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    used: dict = {}
    st1 = p_st1(a.data, used)
    rec = {"what": "B-pre feasibility (docs/ROADMAP.md §7); no gate statistic computed",
           "input": {"projectors": used,
                     "activations": {str(k): f"data/phase12/{v}/" for k, v in RUNS.items()},
                     "prompts": list(V1_SCORED),
                     "behavioural": "data/analysis/behavioural_series.json (series_excl_repeated)"},
           "P-ST1": st1, "P-I3": p_i3(a.data)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    for st, v in rec["P-ST1"].items():
        print(f"P-ST1 step {st}: {v['n_cells']} cells, ratio min {v['ratio_min']:.2f} median "
              f"{v['ratio_median']:.2f}, <=1.5 {v['n_ratio_le_1.5']}, <=2 {v['n_ratio_le_2']}, "
              f"<=3 {v['n_ratio_le_3']}, union full {v['union_full_everywhere']}")
    for k, v in rec["P-I3"].items():
        print(f"P-I3 {k}: {v['n_straddling']} of {len(v['heads'])} straddle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
