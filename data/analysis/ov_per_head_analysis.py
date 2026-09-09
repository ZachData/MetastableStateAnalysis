#!/usr/bin/env python3
"""Read `ov_per_head_series.json` and answer the four questions it was run for.

  Q1  Does the summed-OV statistic describe any head?  (the "fiction" gap)
  Q2  Count vs energy-weighted: bulk or outlier?       (MATH_SPECTRAL_OT §3)
  Q3  Do the FORMING heads' OV operators turn repulsive during formation, and
      does the sign go the particle way or the copying way?
  Q4  Level contrast at matched behavioural score, controls in the SAME layer
      (p7_motifs/cross_head_gate.py's registered `score_and_layer` design).

Nothing here adjudicates: no p-value is written to claims/.
"""
import json
import os
import sys

import numpy as np

os.chdir("/run/media/system/WDS_500/Mets")
sys.path.insert(0, ".")

from core.changepoint_colocation import REGISTERED_P_I1_SWEEP, change_profile
from p7_motifs import cross_head_gate as chg

STEPS = list(REGISTERED_P_I1_SWEEP)
OV = json.load(open("data/analysis/ov_per_head_series.json"))
TB = json.load(open("data/analysis/dissipation_sublayer_series.json"))
BEH = json.load(open("data/analysis/behavioural_series.json"))["series_excl_repeated"]

steps = OV["steps"]
assert steps == STEPS, "runner used a different grid"
si = {s: i for i, s in enumerate(steps)}
NB, NH, D_HEAD = OV["n_blocks"], OV["n_heads"], OV["d_head"]

forming = sorted((tuple(int(x) for x in k.split(",")) for k in TB["per_head"]),
                 key=lambda t: (t[0], t[1]))
forming_set = set(forming)
all_heads = [(l, h) for l in range(NB) for h in range(NH)]

REP = "repulsive_energy_fraction_core"
DIM = "repulsive_dim_fraction_core"


def series(l, h, field=REP):
    return np.array(OV["per_head"][f"{l},{h}"][field], dtype=float)


def win(idx_steps):
    return [si[s] for s in idx_steps]


FORMATION = win([512, 1000, 2000, 4000, 8000])
LATE = win([54000, 143000])
EARLY = win([0, 1, 2, 4, 8])

print("=" * 78)
print("Q1  Does the summed-OV statistic describe any head?")
print("=" * 78)
gaps, cover = [], []
for l in range(NB):
    for s in (512, 2000, 8000, 143000):
        summed = OV["summed_ov_by_step_layer"][f"{s}|{l}"]["repulsive_energy_fraction"]
        per = np.array([series(l, h)[si[s]] for h in range(NH)])
        gaps.append(abs(summed - per.mean()))
        cover.append(float(np.mean(np.abs(per - summed) < 0.05)))
print(f"  |summed - mean(per-head)| : median {np.median(gaps):.4f}  max {np.max(gaps):.4f}")
print(f"  frac of heads within 0.05 of the summed value: mean {np.mean(cover):.3f}")
print("  (head_circuits.py's `head_agreement`: low means the summed number")
print("   describes no head in the layer.)")

print()
print("=" * 78)
print("Q2  Count vs energy-weighted (bulk or outlier?)")
print("=" * 78)
for s in (0, 512, 2000, 8000, 54000, 143000):
    e = np.array([series(l, h)[si[s]] for l, h in all_heads])
    d = np.array([series(l, h, DIM)[si[s]] for l, h in all_heads])
    print(f"  step {s:>6}   energy-weighted {e.mean():.4f}   count {d.mean():.4f}"
          f"   divergence {e.mean() - d.mean():+.4f}")
print("  Divergence != 0 means the sign structure is carried by eigenvalues")
print("  whose |lambda| differs from the bulk -- reading (2) of §3.")

print()
print("=" * 78)
print("Q3  Do FORMING heads' OV operators turn repulsive during formation?")
print("=" * 78)
F = np.array([series(l, h) for l, h in forming])                       # (116,19)
NFh = [x for x in all_heads if x not in forming_set]
N = np.array([series(l, h) for l, h in NFh])                           # (268,19)
print(f"  forming n={len(F)}   non-forming n={len(N)}")
print(f"  {'step':>7}  {'forming':>9}  {'non-forming':>11}  {'diff':>8}")
for s in steps:
    i = si[s]
    print(f"  {s:>7}  {F[:, i].mean():9.4f}  {N[:, i].mean():11.4f}  "
          f"{F[:, i].mean() - N[:, i].mean():+8.4f}")
print()
f_early, f_form, f_late = F[:, EARLY].mean(), F[:, FORMATION].mean(), F[:, LATE].mean()
print(f"  forming heads: early {f_early:.4f} -> formation {f_form:.4f} -> late {f_late:.4f}")
print(f"  0.5 is CHANCE (spectrum symmetric about the imaginary axis at init).")
print(f"  >0.5 = repulsive-dominant (particle reading)")
print(f"  <0.5 = attractive-dominant (copying reading)")
verdict = ("REPULSIVE (particle)" if f_form > 0.5 else "ATTRACTIVE (copying)")
print(f"  --> formation-window forming-head mean is {f_form:.4f}: {verdict}")

print()
print("-" * 78)
print("  Is 0.5 actually the chance value for this statistic? Two answers.")
print("-" * 78)
print(f"  (a) EMPIRICAL, step 0 (untrained): all heads {np.array([series(l,h)[si[0]] for l,h in all_heads]).mean():.4f}")
print("      CLAIM-A's logic: step 0 is what 'not learned' looks like on this")
print("      machine, with the real shapes and the real storage precision.")
rng = np.random.default_rng(11)
null_vals = []
for _ in range(200):
    # Matched-shape random head: same (d, d_head), Gaussian factors. The
    # eigenvalue cloud of a product of independent Gaussians is rotationally
    # symmetric about the origin, so the energy-weighted split should sit at
    # 0.5 -- measured here rather than argued.
    W_O = rng.normal(size=(256, D_HEAD)) / np.sqrt(256)
    W_V = rng.normal(size=(D_HEAD, 256)) / np.sqrt(256)
    ev = np.linalg.eigvals(W_V @ W_O)
    e = np.abs(ev) ** 2
    null_vals.append(float(e[ev.real < 0].sum() / e.sum()))
null_vals = np.array(null_vals)
print(f"  (b) SYNTHETIC, 200 matched-shape random heads: "
      f"mean {null_vals.mean():.4f}  sd {null_vals.std():.4f}  "
      f"[{np.percentile(null_vals, 2.5):.4f}, {np.percentile(null_vals, 97.5):.4f}]")

print()
print("=" * 78)
print("Q4  Level contrast, controls matched on behavioural score IN-LAYER")
print("=" * 78)
# Value per head: repulsive energy fraction averaged over the formation window.
# Behavioural score: peak over the sweep, the quantity P-I3 matches on.
val, score, layer = {}, {}, {}
for (l, h) in all_heads:
    key = f"{l},{h}"
    val[(l, h)] = float(series(l, h)[FORMATION].mean())
    score[(l, h)] = float(np.max(BEH[key])) if key in BEH else 0.0
    layer[(l, h)] = l
n_ind = len(forming)
print(f"  induction arm = the {n_ind} forming heads; controls = the other {len(NFh)}")
try:
    keys = all_heads
    sc = np.array([score[k] for k in keys])
    lay = np.array([layer[k] for k in keys])
    vv = np.array([val[k] for k in keys])
    is_ind = np.array([k in forming_set for k in keys])
    for mkey in ("score", "score_and_layer"):
        ms = chg.matched_sets(sc, is_ind, lay,
                              n_controls=chg.N_CONTROLS_PER_INDUCTION_HEAD,
                              key=mkey)
        tag = " (REGISTERED)" if mkey == chg.REGISTERED_CONTROL_MATCHING_KEY else ""
        print(f"  --- key={mkey!r}{tag}  n_sets={ms['n_sets']}  "
              f"n_dropped={ms['n_dropped']}  "
              f"median_score_gap={ms['median_within_set_score_gap']}")
        try:
            arm = chg.exact_rank_arm(vv, ms["sets"])
            for k, v in arm.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.6g}")
                elif not isinstance(v, (list, dict)):
                    print(f"      {k}: {v}")
        except Exception as e2:
            print(f"      arm refused: {type(e2).__name__}: {e2}")
except Exception as exc:
    print(f"  cross_head_gate refused / signature mismatch: {type(exc).__name__}: {exc}")
    print("  (falling back to a plain within-layer rank contrast)")
    wins = tot = 0
    for (l, h) in forming:
        peers = [(l, hh) for hh in range(NH) if (l, hh) not in forming_set]
        if not peers:
            continue
        v = val[(l, h)]
        wins += sum(1 for p in peers if v > val[p]); tot += len(peers)
    print(f"  forming head beats an in-layer non-forming head "
          f"{wins}/{tot} = {wins / max(tot, 1):.4f} of the time (0.5 = no effect)")

print()
print("=" * 78)
print("Q3b  Per-head co-location of the OV sign split with behaviour")
print("=" * 78)
from scipy.stats import spearmanr
c_ov, c_b = [], []
for (l, h) in forming:
    try:
        c_ov.append(change_profile(steps, list(series(l, h)), "rise")["centroid_log_step"])
    except Exception:
        c_ov.append(np.nan)
    try:
        c_b.append(change_profile(steps, list(BEH[f"{l},{h}"]), "rise")["centroid_log_step"])
    except Exception:
        c_b.append(np.nan)
c_ov, c_b = np.array(c_ov), np.array(c_b)
m = np.isfinite(c_ov) & np.isfinite(c_b)
print(f"  located {m.sum()}/{len(forming)}   distinct OV centroids: "
      f"{len(set(np.round(c_ov[m], 8)))}")
if m.sum() > 2:
    r, p = spearmanr(c_ov[m], c_b[m])
    print(f"  spearman(cen[OV rep-share], cen[behavioural]) = {r:.4f}  p={p:.4g}")
    lr = np.array([l for l, h in forming], float)
    rxz, ryz = spearmanr(c_ov[m], lr[m])[0], spearmanr(c_b[m], lr[m])[0]
    part = (r - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    print(f"  partial | layer = {part:.4f}   (shared_unit_factor: "
          f"rho(A,layer)={rxz:.3f}, rho(B,layer)={ryz:.3f})")
