#!/usr/bin/env python3
"""Tier B: (1) the attn/FFN dissipation panel on the P-I1 axis, (2) the per-head
co-location test using the attention-dissipation series as the B-anchor P-I1 §3.7
wants (defined at every checkpoint, not centroid-tied like the relay excess)."""
import json, os, collections, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/run/media/system/WDS_500/Mets")
import sys; sys.path.insert(0, ".")
from core.changepoint_colocation import REGISTERED_P_I1_SWEEP, change_profile
from p7_motifs.formation_gate import p_value_p_i1

STEPS = list(REGISTERED_P_I1_SWEEP)
tb = json.load(open("data/analysis/dissipation_sublayer_series.json"))
ta = json.load(open("data/analysis/dissipation_series.json"))
beh = json.load(open("data/analysis/behavioural_series.json"))["series_excl_repeated"]
co = None
try:
    import csv as _c
    co = list(_c.DictReader(open("data/analysis/colocation_panel.csv")))
except Exception:
    pass

NB = tb["n_blocks"]
pool = tb["pooled_by_step_layer"]
psl = tb["per_step_layer"]
prompts = tb["scored_prompts"]
per_head = tb["per_head"]
forming = sorted((tuple(int(x) for x in k.split(",")) for k in per_head), key=lambda t: (t[0], t[1]))
FL = sorted({l for l, h in forming})           # forming-head layers

def xs(s):
    return [max(v, 0.5) for v in s]

# ---------------------------------------------------------------- panel ----
def pooled_series(fn):
    return [fn(s) for s in STEPS]

def attn_rep_share(step, layers):
    ar = sum(pool[f"{step}|{l}"]["d_attn_rep"] for l in layers)
    aa = sum(pool[f"{step}|{l}"]["d_attn_att"] for l in layers)
    m = abs(ar) + abs(aa)
    return (abs(ar) / m) if m > 0 else np.nan

def ffn_rep_share(step, layers):
    fr = sum(pool[f"{step}|{l}"]["d_ffn_rep"] for l in layers)
    fa = sum(pool[f"{step}|{l}"]["d_ffn_att"] for l in layers)
    m = abs(fr) + abs(fa)
    return (abs(fr) / m) if m > 0 else np.nan

def chan(step, key, layers=range(NB)):
    return sum(pool[f"{step}|{l}"][key] for l in layers)

def gfa_attn_mean(step, layers):
    v = [psl[f"{step}|{p}|{l}"]["gfa_attn"]["mean"]
         for l in layers for p in prompts
         if psl[f"{step}|{p}|{l}"]["gfa_attn"]["status"] == "ok"]
    return float(np.mean(v)) if v else np.nan

# Tier A total-dx repulsive share (for the 3-way item-5 comparison)
def tierA_rep_share(step):
    da = sum(ta["pooled_by_step_layer"][f"1.0|{step}|{l}"]["d_attractive"] for l in range(NB))
    dr = sum(ta["pooled_by_step_layer"][f"1.0|{step}|{l}"]["d_repulsive"] for l in range(NB))
    m = abs(da) + abs(dr)
    return (abs(dr) / m) if m > 0 else np.nan

p2_frac_rep = {}
if co:
    for r in co:
        try:
            p2_frac_rep[int(r["step"])] = float(r["p2_frac_repulsive"])
        except Exception:
            pass

fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
for a in ax:
    a.axvspan(512, 8000, color="gold", alpha=0.12, lw=0)
    a.axvspan(512, 1000, color="crimson", alpha=0.12, lw=0)
    a.set_xscale("log"); a.grid(alpha=0.25, which="both")

a = ax[0]
a.plot(xs(STEPS), pooled_series(lambda s: chan(s, "d_attn")), "o-", color="C0", label="Σ d_attn (all 24 layers, 7 prompts)")
a.plot(xs(STEPS), pooled_series(lambda s: chan(s, "d_ffn")), "s-", color="C1", label="Σ d_ffn")
a.plot(xs(STEPS), pooled_series(lambda s: chan(s, "first_order")), ":", color="0.4", label="Σ first-order (=attn+ffn)")
a.axhline(0, color="0.5", lw=0.8)
a.set_ylabel("first-order ΔE_β\nby channel")
a.legend(fontsize=8, loc="upper left")
a.set_title("Tier B — exact attention / FFN split of the first-order energy change  (max channel sum_check 4e-15)\n"
            "gold = induction-formation window;  red = 512→1000", fontsize=10)

a = ax[1]
a.plot(xs(STEPS), [attn_rep_share(s, range(NB)) for s in STEPS], "o-", color="C0", label="ATTN channel repulsive share (all layers)")
a.plot(xs(STEPS), [attn_rep_share(s, FL) for s in STEPS], "o--", color="C0", alpha=.55, label=f"ATTN, forming layers L{FL[0]}–{FL[-1]}")
a.plot(xs(STEPS), [ffn_rep_share(s, range(NB)) for s in STEPS], "s-", color="C1", label="FFN channel repulsive share (all layers)")
a.plot(xs(STEPS), [tierA_rep_share(s) for s in STEPS], "^:", color="0.4", label="Tier A: TOTAL-dx repulsive share")
if p2_frac_rep:
    a.plot(xs(STEPS), [p2_frac_rep.get(s, np.nan) for s in STEPS], "D-", color="C3", label="Phase 2 β1.0 frac_repulsive (violations)")
a.axhline(0.5, color="0.5", lw=0.8)
a.set_ylabel("repulsive-subspace\nshare")
a.set_ylim(0, 1.05)
a.legend(fontsize=7.5, loc="lower left", ncol=2)
a.set_title("status-2 item 5 — is the frac_repulsive decay in the attention channel?", fontsize=9)

a = ax[2]
a.plot(xs(STEPS), [gfa_attn_mean(s, range(NB)) for s in STEPS], "o-", color="C5", label="ATTN gradient-flow alignment  mean cos(−G,v)")
a.plot(xs(STEPS), [gfa_attn_mean(s, FL) for s in STEPS], "o--", color="C5", alpha=.55, label=f"ATTN, forming layers")
a.axhline(0, color="0.5", lw=0.8)
a.set_ylabel("attn alignment cos")
a.legend(fontsize=8, loc="lower left")

a = ax[3]
# per-head d_attn_total: how many forming heads have a positive attn-dissipation contribution
arr = np.array([per_head[f"{l},{h}"]["d_attn_total"] for l, h in forming])
a.plot(xs(STEPS), (arr > 0).sum(axis=0), "o-", color="C2", label="# forming heads with d_attn_total > 0")
a.plot(xs(STEPS), np.nansum(np.clip(arr, 0, None), axis=0), "s--", color="C8", label="Σ positive d_attn_total over forming heads")
a.set_ylabel("forming-head\nattn dissipation")
a.legend(fontsize=8, loc="upper left")
ax[-1].set_xlabel("training step (log; step 0 at 0.5)")
for s in STEPS:
    ax[-1].annotate(str(s), (max(s, 0.5), 0), xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", fontsize=6, rotation=90, color="0.4")
fig.tight_layout()
fig.savefig("data/analysis/dissipation_tierB_panel.png", dpi=130, bbox_inches="tight")
print("wrote data/analysis/dissipation_tierB_panel.png")

# --------------------------------------------------- co-location test -----
def centroid_classes(series_by_head):
    cs = {}
    for k, v in series_by_head.items():
        try:
            cs[k] = round(change_profile(STEPS, v, "rise")["centroid_log_step"], 8)
        except Exception:
            cs[k] = None
    located = [x for x in cs.values() if x is not None]
    cnt = collections.Counter(located)
    return len(located), sorted(cnt.values(), reverse=True)

print("\n==== per-head co-location: attention-dissipation B-anchors vs behavioural rise ====")
Bside = [beh[f"{l},{h}"] for l, h in forming]

for name, Aside in (
    ("d_attn_total", [per_head[f"{l},{h}"]["d_attn_total"] for l, h in forming]),
    ("d_attn_repulsive (as |.|)", [list(np.abs(per_head[f"{l},{h}"]["d_attn_repulsive"])) for l, h in forming]),
    ("-gfa_cos (anti-alignment)", [list(-np.array(per_head[f"{l},{h}"]["gfa_cos"])) for l, h in forming]),
):
    nloc, sizes = centroid_classes({i: a for i, a in enumerate(Aside)})
    try:
        res = p_value_p_i1(STEPS, Aside, Bside, skip_no_rise=True)
        arm = (res.get("arms") or [{}])[0]
        print(f"\n  A = {name}")
        print(f"    located heads {nloc}/116 | centroid class sizes {sizes[:6]}{'...' if len(sizes)>6 else ''}")
        print(f"    p_value={res['p_value']}  p_reciprocal={res['p_reciprocal']}  verdict={res['verdict']}")
        print(f"    n_units={arm.get('n_units')}  skipped_no_rise={arm.get('n_skipped_no_rise')}  "
              f"mean_distance_log_step={arm.get('mean_distance_log_step')}")
    except Exception as e:
        print(f"\n  A = {name}: gate refused -> {e}")
