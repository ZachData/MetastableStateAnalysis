#!/usr/bin/env python3
"""From Tier A dissipation_series.json:
 (top)  early layers L0-L5 — each layer's dissipation character across the 19 checkpoints
 (bot)  512 -> 1000 focus — the per-layer depth profile at both checkpoints, side by side
Author's asks 2026-09-06: keep all layers, look closely at the early ones, focus on 512->1000.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/run/media/system/WDS_500/Mets")
d = json.load(open("data/analysis/dissipation_series.json"))
STEPS = d["steps"]; NB = d["n_blocks"]; B = 1.0
prompts = d["scored_prompts"]
psl = d["per_step_layer"]

def cell(step, layer, field):
    """7-prompt aggregate of `field` at (step, layer): mean for residual/gfa, sum for d_*."""
    vals = []
    for p in prompts:
        e = psl[f"{B}|{step}|{p}|{layer}"]
        if field == "relative_residual":
            v = e["relative_residual"]
        elif field == "gfa_mean":
            v = e["gfa"]["mean"] if e["gfa"]["status"] == "ok" else None
        elif field == "rep_share":
            da, dr = e["d_attractive"], e["d_repulsive"]
            m = abs(da) + abs(dr); v = (abs(dr) / m) if m > 0 else None
        else:
            v = e[field]
        if v is not None:
            vals.append(abs(v) if field == "relative_residual" else v)
    if not vals:
        return np.nan
    return float(np.mean(vals))

def xs(s):
    return [max(v, 0.5) for v in s]

fig = plt.figure(figsize=(13, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1], hspace=0.33, wspace=0.22)

# ---- row 1-2: early layers L0..L5 across checkpoints -----------------------
fields = [("first_order", "first-order dissipation Σᵢ⟨Gᵢ,vᵢ⟩ (7-prompt sum)"),
          ("relative_residual", "|linearisation residual| (rel., 7-prompt mean)"),
          ("rep_share", "repulsive-subspace share of |dissipation|"),
          ("gfa_mean", "gradient-flow alignment  mean cos(−G,v)")]
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
cmap = plt.get_cmap("viridis")
for ax, (fld, title) in zip(axes, fields):
    for L in range(6):
        y = [cell(s, L, fld) for s in STEPS]
        ax.plot(xs(STEPS), y, "o-", ms=3, color=cmap(L / 5.0), label=f"L{L}")
    ax.set_xscale("log")
    if fld == "relative_residual":
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.6", lw=0.8)
    if fld in ("rep_share",):
        ax.axhline(0.5, color="0.6", lw=0.8)
    if fld in ("gfa_mean", "first_order"):
        ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvspan(512, 1000, color="crimson", alpha=0.10, lw=0)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.set_xlabel("step", fontsize=8)
axes[0].legend(fontsize=7, ncol=3, loc="upper left")
axes[0].text(0.02, -0.30, "rows 1–2: early layers L0–L5, each across the 19 checkpoints  ·  "
             "red band = 512→1000", transform=axes[0].transAxes, fontsize=8, color="0.35")

# ---- row 3: 512 -> 1000 depth profile, all 24 layers ---------------------
axL = fig.add_subplot(gs[2, 0])
axR = fig.add_subplot(gs[2, 1])
Ls = list(range(NB))
for ax, fld, ttl in ((axL, "relative_residual", "|linearisation residual| by depth"),
                     (axR, "rep_share", "repulsive share by depth")):
    for s, c, mk in ((512, "C0", "o"), (1000, "C3", "s")):
        ax.plot(Ls, [cell(s, L, fld) for L in Ls], mk + "-", color=c, label=f"step {s}")
    ax.set_xlabel("layer (block index)", fontsize=8)
    ax.set_title(ttl, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if fld == "relative_residual":
        ax.set_yscale("log"); ax.axhline(1.0, color="0.6", lw=0.8)
    else:
        ax.axhline(0.5, color="0.6", lw=0.8)
axL.text(0.02, -0.22, "row 3: the single 512→1000 interval — where in depth the change sits",
         transform=axL.transAxes, fontsize=8, color="0.35")

fig.suptitle("Early-layer character across training, and the 512→1000 interval  (Tier A data)",
             fontsize=11, y=0.995)
png = "data/analysis/dissipation_earlylayer_512to1k.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", png)

# ---- text summary -------------------------------------------------------
print("\n=== early layers L0-L5: |rel. residual| across steps ===")
print("step   " + "  ".join(f"L{L}" for L in range(6)))
for s in STEPS:
    print(f"{s:<7d}" + "  ".join(f"{cell(s,L,'relative_residual'):.2f}" for L in range(6)))
print("\n=== 512 -> 1000, per-layer rep_share and |rel residual| ===")
print("L    repshare@512  repshare@1000   relres@512  relres@1000")
for L in Ls:
    print(f"{L:<3d}  {cell(512,L,'rep_share'):.3f}         {cell(1000,L,'rep_share'):.3f}"
          f"          {cell(512,L,'relative_residual'):.2f}        {cell(1000,L,'relative_residual'):.2f}")
