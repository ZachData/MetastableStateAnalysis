#!/usr/bin/env python3
"""Dissipation-identity panel (Tier A) on the P-I1 19-step checkpoint axis.
Reads data/analysis/dissipation_series.json; sits beside colocation_panel.png."""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/run/media/system/WDS_500/Mets"
os.chdir(REPO)
d = json.load(open("data/analysis/dissipation_series.json"))
STEPS = d["steps"]
NB = d["n_blocks"]
B = 1.0
psl = d["per_step_layer"]
pool = d["pooled_by_step_layer"]
prompts = d["scored_prompts"]

def X(s):
    return [max(v, 0.5) for v in s]
xs = X(STEPS)

# ---- per-step aggregates -------------------------------------------------
rows = []
for s in STEPS:
    fo = np.array([pool[f"{B}|{s}|{l}"]["first_order"] for l in range(NB)])
    ae = np.array([pool[f"{B}|{s}|{l}"]["actual_delta_E"] for l in range(NB)])
    da = np.array([pool[f"{B}|{s}|{l}"]["d_attractive"] for l in range(NB)])
    dr = np.array([pool[f"{B}|{s}|{l}"]["d_repulsive"] for l in range(NB)])

    # gfa + relative_residual over the 24 layers x 7 prompts
    gfa_mean, gfa_fdesc, relres = [], [], []
    for l in range(NB):
        for p in prompts:
            e = psl[f"{B}|{s}|{p}|{l}"]
            if e["gfa"]["status"] == "ok":
                gfa_mean.append(e["gfa"]["mean"])
                gfa_fdesc.append(e["gfa"]["frac_descending"])
            if e["relative_residual"] is not None:
                relres.append(abs(e["relative_residual"]))
    relres = np.array(relres)
    # layer-0 relative residual, pooled prompts
    rr_l0 = np.median([abs(psl[f"{B}|{s}|{p}|0"]["relative_residual"])
                       for p in prompts
                       if psl[f"{B}|{s}|{p}|0"]["relative_residual"] is not None])
    rr_deep = np.median([abs(psl[f"{B}|{s}|{p}|{l}"]["relative_residual"])
                         for l in range(4, NB) for p in prompts
                         if psl[f"{B}|{s}|{p}|{l}"]["relative_residual"] is not None])

    tot_fo = fo.sum()
    tot_ae = ae.sum()
    mag = np.abs(da).sum() + np.abs(dr).sum()
    rep_share = (np.abs(dr).sum() / mag) if mag > 0 else np.nan
    rows.append(dict(
        step=s,
        sum_first_order=tot_fo, sum_actual_dE=tot_ae,
        n_layers_uphill=int((fo > 0).sum()),
        repulsive_share=rep_share,
        gfa_mean=float(np.mean(gfa_mean)),
        gfa_frac_descending=float(np.mean(gfa_fdesc)),
        relres_median=float(np.median(relres)),
        relres_p90=float(np.quantile(relres, 0.9)),
        relres_layer0_median=float(rr_l0),
        relres_deep_median=float(rr_deep),
    ))

# ---- CSV --------------------------------------------------------------------
csv_path = "data/analysis/dissipation_panel.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
print("wrote", csv_path)

g = lambda k: [r[k] for r in rows]

# ---- PLOT ----------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
WIN = (512, 8000)
for a in ax:
    a.axvspan(*WIN, color="gold", alpha=0.13, lw=0)
    for m in (512, 1000, 3000, 4000):
        a.axvline(m, color="0.6", ls=":", lw=0.9)
    a.set_xscale("log")
    a.grid(alpha=0.25, which="both")

a = ax[0]
a.plot(xs, g("sum_first_order"), "o-", color="C3", label="Σ first-order dissipation  (β=1, 7 prompts, 24 layers)")
a.plot(xs, g("sum_actual_dE"), "s--", color="0.4", label="Σ actual ΔE_β  (same sum)")
a.axhline(0, color="0.5", lw=0.8)
a.set_ylabel("energy change")
a.legend(fontsize=8, loc="upper left")
a.set_title("Dissipation identity (Tier A) on the P-I1 axis — data/analysis/dissipation_series.json\n"
            "gold = induction-formation window;  first-order = Σᵢ⟨Gᵢ,vᵢ⟩ per token, exact per-particle attribution",
            fontsize=10)
at = a.twinx()
at.plot(xs, g("n_layers_uphill"), "d:", color="C9", label="# layer boundaries locally uphill on E_β")
at.set_ylabel("# layers uphill", color="C9")
at.legend(fontsize=8, loc="lower right")

a = ax[1]
a.plot(xs, g("repulsive_share"), "o-", color="C0",
       label="repulsive-subspace share of |dissipation|  (Σ|d_rep| / (Σ|d_att|+Σ|d_rep|))")
a.axhline(0.5, color="0.5", lw=0.8)
a.set_ylabel("repulsive share")
a.set_ylim(0, 1)
a.legend(fontsize=8, loc="lower left")
a.text(0.5, 0.04, "dissipation-side counterpart of Phase 2's frac_repulsive — see subspace caveat in the JSON",
       transform=a.transAxes, fontsize=7, color="0.4")

a = ax[2]
a.plot(xs, g("gfa_mean"), "o-", color="C5", label="mean cos(−G, v)  over tokens×layers×prompts")
a.axhline(0, color="0.5", lw=0.8)
a.set_ylabel("gradient-flow\nalignment  cos", color="C5")
a.legend(fontsize=8, loc="upper left")
at = a.twinx()
at.plot(xs, g("gfa_frac_descending"), "s--", color="C6", label="frac of tokens descending on E_β  (cos>0)")
at.set_ylabel("frac descending", color="C6")
at.legend(fontsize=8, loc="lower right")

a = ax[3]
a.plot(xs, g("relres_median"), "o-", color="C1", label="median |ΔE − Σ⟨G,v⟩| / max(|ΔE|,|Σ⟨G,v⟩|)  (all layers)")
a.plot(xs, g("relres_layer0_median"), "^--", color="C3", label="layer 0 only (emb → block 0)")
a.plot(xs, g("relres_deep_median"), "v:", color="C2", label="layers 4–23 only")
a.axhline(1.0, color="0.5", lw=0.8)
a.set_ylabel("linearisation\nresidual (rel.)")
a.set_yscale("log")
a.legend(fontsize=8, loc="upper left")
a.text(0.5, 0.9, "large ⇒ the forward-Euler/ODE framing does not hold at that layer (MATH_SPECTRAL_OT §5.3d)",
       transform=a.transAxes, fontsize=7, color="0.4", ha="center")

ax[-1].set_xlabel("training step (log; step 0 at 0.5)")
for s in STEPS:
    ax[-1].annotate(str(s), (max(s, 0.5), 0), xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", fontsize=6, rotation=90, color="0.4")
fig.tight_layout()
png = "data/analysis/dissipation_panel.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", png)
print("\n=== dissipation_panel.csv ===")
print(open(csv_path).read())
