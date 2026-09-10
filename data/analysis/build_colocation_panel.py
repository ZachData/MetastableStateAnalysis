#!/usr/bin/env python3
"""
Co-location panel: lay the Phase 1 / Phase 2 global observables on the exact
19-step P-I1 checkpoint axis, beside the P-I1 behavioural and relay series.

All inputs are already on disk:
  - Phase 1 : data/phase12/<ts>/llm_cross_run_report.txt   (one per checkpoint)
  - Phase 2 : data/phase12/p2_eigenspectra_<ts>/p2_eigenspectra_cross_run.json
  - P-I1    : data/analysis/{behavioural_series,relay_null_series_k50,formation_series}.json

Nothing here recomputes a metric; it parses committed run artifacts.
"""
import json, re, glob, os, csv
import numpy as np

REPO = "/run/media/system/WDS_500/Mets"
os.chdir(REPO)

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000]
META_PROMPTS = ["camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
                "paper_excerpt", "sullivan_ballou", "wiki_paragraph"]  # 7, excl repeated_tokens

# --------------------------------------------------------------------------
# Phase 1 : parse one llm_cross_run_report.txt per checkpoint
# --------------------------------------------------------------------------
def p1_dirs():
    out = {}
    for d in glob.glob("data/phase12/2026-*/"):
        r = os.path.join(d, "llm_cross_run_report.txt")
        if not os.path.isfile(r):
            continue
        txt = open(r).read()
        m = re.search(r"pythia-410m-step(\d+)", txt)
        if not m:
            continue
        out[int(m.group(1))] = r
    return out

def parse_p1(path):
    txt = open(path).read()
    lines = txt.splitlines()
    res = dict(er_normed=[], er_raw=[], normpr=[], nviol_summary=0,
               maxsev=[], nviol_energy_b1=0, raw_lambda2=[], dev_ratio=[],
               onset_sd=None, onset_regime=None)

    # ---- SUMMARY TABLE ----
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith("Model") and "MinRank" in l)
    except StopIteration:
        i = None
    if i is not None:
        for l in lines[i + 2:]:
            if not l.strip() or set(l.strip()) <= set("-"):
                break
            f = l.split()
            if len(f) < 14 or not f[0].startswith("pythia"):
                break
            prompt = f[1]
            if prompt not in META_PROMPTS:
                continue
            try:
                res["er_normed"].append(float(f[6]))
                res["er_raw"].append(float(f[7]))
                res["normpr"].append(float(f[8]))
                res["nviol_summary"] += int(f[12])
                res["maxsev"].append(float(f[13]))
            except ValueError:
                pass

    # ---- ENERGY MONOTONICITY  (beta=1.0 relative rule) ----
    for l in lines:
        m = re.search(r"\|\s*(\w+)\s*\|\s*beta=1\.0:\s*nViolRel=(\d+)", l)
        if m and m.group(1) in META_PROMPTS:
            res["nviol_energy_b1"] += int(m.group(2))

    # ---- FIEDLER : per-head RawLambda2 / DevRatio, averaged over head rows ----
    try:
        j = next(k for k, l in enumerate(lines) if "RawLambda2" in l and "DevRatio" in l)
        for l in lines[j + 1:]:
            f = l.split()
            if len(f) >= 3 and f[0].isdigit():
                try:
                    res["raw_lambda2"].append(float(f[1]))
                    res["dev_ratio"].append(float(f[2]))
                except ValueError:
                    pass
            elif l.strip().startswith("[") or ("Model:" in l):
                break
    except StopIteration:
        pass

    # ---- PROMPT SENSITIVITY : SD of plateau onset across prompts ----
    m = re.search(r"SD of onset across prompts:\s*([\d.]+)\s*→?\s*(\S+)", txt)
    if m:
        res["onset_sd"] = float(m.group(1))
        res["onset_regime"] = m.group(2).strip()

    return res

# --------------------------------------------------------------------------
# Phase 2 : parse p2_eigenspectra_cross_run.json per checkpoint
# --------------------------------------------------------------------------
def parse_p2():
    by_step = {}
    for d in glob.glob("data/phase12/p2_eigenspectra_*/"):
        j = os.path.join(d, "p2_eigenspectra_cross_run.json")
        if not os.path.isfile(j):
            continue
        rows = json.load(open(j))
        m = re.search(r"step(\d+)", rows[0]["model"])
        step = int(m.group(1))
        rows = [r for r in rows if r["prompt"] in META_PROMPTS]
        nviol = sum(int(r.get("beta1.0_n_violations") or 0) for r in rows)
        fr = [r["beta1.0_frac_repulsive"] for r in rows
              if (r.get("beta1.0_n_violations") or 0) > 0 and r.get("beta1.0_frac_repulsive") is not None]
        ovfr = [r["ov_frac_repulsive_mean"] for r in rows if r.get("ov_frac_repulsive_mean") is not None]
        by_step[step] = dict(
            nviol=nviol,
            frac_repulsive=float(np.mean(fr)) if fr else np.nan,
            ov_frac_repulsive=float(np.mean(ovfr)) if ovfr else np.nan,
        )
    return by_step

# --------------------------------------------------------------------------
# P-I1 : behavioural + relay
# --------------------------------------------------------------------------
def parse_pi1():
    beh = json.load(open("data/analysis/behavioural_series.json"))
    rn = json.load(open("data/analysis/relay_null_series_k50.json"))
    fs = json.load(open("data/analysis/formation_series.json"))

    S = beh["series_excl_repeated"]                 # "L,H" -> [19]
    arr = np.array([S[k] for k in S])               # (nheads, 19)
    base = np.median(arr[:, 0])
    thr = max(0.010, 5 * base)
    beh_max = arr.max(axis=0).tolist()
    beh_elev = (arr > thr).sum(axis=0).tolist()
    beh_L7H8 = S.get("7,8")
    beh_L6H0 = S.get("6,0")

    exc = rn["above_null_excess"]                   # "L,H" -> [19]
    exc_arr = np.array([exc[k] for k in exc])       # (116, 19)
    excess_total = exc_arr.sum(axis=0).tolist()

    matcher = fs["series"]["matcher"]               # "L,H" -> [19]
    mraw = np.array([matcher[k] for k in matcher])  # (nheads, 19)
    relay_raw_total = mraw.sum(axis=0).tolist()
    relay_heads = (mraw > 0).sum(axis=0).tolist()

    return dict(beh_max=beh_max, beh_elev=beh_elev, beh_thr=thr,
                beh_L7H8=beh_L7H8, beh_L6H0=beh_L6H0,
                excess_total=excess_total, relay_raw_total=relay_raw_total,
                relay_heads=relay_heads)

# --------------------------------------------------------------------------
def main():
    p1d = p1_dirs()
    p1 = {}
    for s in STEPS:
        if s in p1d:
            r = parse_p1(p1d[s])
            p1[s] = dict(
                er_normed=float(np.mean(r["er_normed"])) if r["er_normed"] else np.nan,
                er_normed_min=float(np.min(r["er_normed"])) if r["er_normed"] else np.nan,
                er_raw=float(np.mean(r["er_raw"])) if r["er_raw"] else np.nan,
                nviol=r["nviol_summary"],
                nviol_energy_b1=r["nviol_energy_b1"],
                maxsev=float(np.max(r["maxsev"])) if r["maxsev"] else np.nan,
                raw_lambda2=float(np.mean(r["raw_lambda2"])) if r["raw_lambda2"] else np.nan,
                dev_ratio=float(np.mean(r["dev_ratio"])) if r["dev_ratio"] else np.nan,
                onset_sd=r["onset_sd"], onset_regime=r["onset_regime"],
            )
        else:
            p1[s] = None

    p2 = parse_p2()
    pi1 = parse_pi1()

    # -------------------- CSV --------------------
    csv_path = "data/analysis/colocation_panel.csv"
    cols = ["step",
            "p1_energy_nviol_b1", "p1_summary_nviol", "p1_maxsev",
            "p2_beta1_nviol", "p2_frac_repulsive", "p2_ov_frac_repulsive",
            "p1_eff_rank_normed_mean", "p1_eff_rank_normed_min", "p1_eff_rank_raw_mean",
            "p1_raw_lambda2_mean", "p1_fiedler_dev_ratio_mean",
            "p1_plateau_onset_sd", "p1_plateau_onset_regime",
            "pi1_beh_max_attn", "pi1_beh_L7H8", "pi1_beh_L6H0", "pi1_beh_elevated_heads",
            "pi1_relay_raw_total", "pi1_relay_above_null_excess", "pi1_relay_forming_heads"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, s in enumerate(STEPS):
            a = p1[s] or {}
            b = p2.get(s, {})
            w.writerow([
                s,
                a.get("nviol_energy_b1", ""), a.get("nviol", ""),
                f"{a.get('maxsev', float('nan')):.4g}",
                b.get("nviol", ""), f"{b.get('frac_repulsive', float('nan')):.4g}",
                f"{b.get('ov_frac_repulsive', float('nan')):.4g}",
                f"{a.get('er_normed', float('nan')):.4g}", f"{a.get('er_normed_min', float('nan')):.4g}",
                f"{a.get('er_raw', float('nan')):.4g}",
                f"{a.get('raw_lambda2', float('nan')):.4g}", f"{a.get('dev_ratio', float('nan')):.4g}",
                a.get("onset_sd", ""), a.get("onset_regime", ""),
                f"{pi1['beh_max'][i]:.4g}", f"{pi1['beh_L7H8'][i]:.4g}", f"{pi1['beh_L6H0'][i]:.4g}",
                pi1["beh_elev"][i],
                f"{pi1['relay_raw_total'][i]:.6g}", f"{pi1['excess_total'][i]:.6g}",
                pi1["relay_heads"][i],
            ])
    print("wrote", csv_path)

    # -------------------- PLOT --------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def X(steps):                 # step 0 -> 0.5 so it sits on a log axis
        return [max(s, 0.5) for s in steps]
    xs = X(STEPS)

    g = lambda key: [((p1[s] or {}).get(key, np.nan)) for s in STEPS]
    g2 = lambda key: [p2.get(s, {}).get(key, np.nan) for s in STEPS]

    fig, ax = plt.subplots(6, 1, figsize=(11, 16), sharex=True)

    WIN = (512, 8000)             # induction-formation window
    MARKS = {256: "-", 512: "energy break /\nplateau flip", 1000: "-",
             3000: "Fiedler 0-x", 4000: "rank peak", 143000: None}
    for a in ax:
        a.axvspan(WIN[0], WIN[1], color="gold", alpha=0.13, lw=0)
        for mstep in (512, 1000, 3000, 4000):
            a.axvline(mstep, color="0.6", ls=":", lw=0.9)
        a.set_xscale("log")
        a.grid(alpha=0.25, which="both")

    # 1 — energy-monotonicity violations
    a = ax[0]
    a.plot(xs, g("nviol_energy_b1"), "o-", color="C3", label="Phase 1  Σ nViolRel  (β=1, 7 prompts)")
    a.plot(xs, g2("nviol"), "s--", color="C1", label="Phase 2  Σ β1.0 violations  (7 prompts)")
    a.set_ylabel("energy-monotonicity\nviolations")
    a.legend(fontsize=8, loc="upper left")
    a.set_title("Global (Phase 1 / Phase 2) observables on the P-I1 19-step checkpoint axis\n"
                "gold band = induction-formation window (≈512–8000);  dotted = 512 / 1000 / 3000 / 4000",
                fontsize=10)

    # 2 — frac_repulsive
    a = ax[1]
    a.plot(xs, g2("frac_repulsive"), "o-", color="C0",
           label="Phase 2  β1.0 frac_repulsive  (violation-mass share in V's repulsive subspace)")
    a.plot(xs, g2("ov_frac_repulsive"), "^--", color="C4", alpha=.8,
           label="ov_frac_repulsive_mean  (weights-only, per checkpoint)")
    a.axhline(0.5, color="0.5", lw=0.8)
    a.set_ylabel("frac_repulsive")
    a.legend(fontsize=8, loc="lower left")

    # 3 — effective rank
    a = ax[2]
    a.plot(xs, g("er_normed"), "o-", color="C2", label="eff. rank (normed, sphere)  — mean over 7 prompts")
    a.plot(xs, g("er_normed_min"), "o:", color="C2", alpha=.5, label="eff. rank (normed) — min over prompts")
    a.set_ylabel("effective rank\n(normed)")
    a.legend(fontsize=8, loc="upper left")
    at = a.twinx()
    at.plot(xs, g("er_raw"), "x--", color="0.55", label="eff. rank (RAW) — sink-count proxy")
    at.set_ylabel("eff. rank (raw)", color="0.55")
    at.legend(fontsize=8, loc="lower right")

    # 4 — Fiedler
    a = ax[3]
    a.plot(xs, g("raw_lambda2"), "o-", color="C5", label="raw λ₂  (mean over head rows)")
    a.set_ylabel("raw λ₂", color="C5")
    a.legend(fontsize=8, loc="upper left")
    at = a.twinx()
    at.plot(xs, g("dev_ratio"), "s--", color="C6", label="Fiedler deviation / baseline  (mean over head rows)")
    at.axhline(0.0, color="0.5", lw=0.8)
    at.set_ylabel("dev. ratio", color="C6")
    at.legend(fontsize=8, loc="lower left")

    # 5 — P-I1 behavioural
    a = ax[4]
    a.plot(xs, pi1["beh_max"], "o-", color="k", label="max pooled induction attention over 384 heads")
    a.plot(xs, pi1["beh_L7H8"], "-", color="C0", alpha=.8, label="L7H8")
    a.plot(xs, pi1["beh_L6H0"], "-", color="C1", alpha=.8, label="L6H0")
    a.set_ylabel("pooled induction\nattention")
    a.legend(fontsize=8, loc="upper left")
    at = a.twinx()
    at.plot(xs, pi1["beh_elev"], "d:", color="C9",
            label=f"# heads > {pi1['beh_thr']:.3f}")
    at.set_ylabel("# elevated heads", color="C9")
    at.legend(fontsize=8, loc="lower right")

    # 6 — P-I1 relay
    a = ax[5]
    a.plot(xs, pi1["relay_raw_total"], "s--", color="0.4", label="raw relay total (matcher)")
    a.plot(xs, pi1["excess_total"], "o-", color="C3", label="Σ above-null excess over 116 forming heads (K=50)")
    a.set_yscale("symlog")
    a.set_ylabel("relay count")
    a.legend(fontsize=8, loc="upper left")
    at = a.twinx()
    at.plot(xs, pi1["relay_heads"], "d:", color="C8", label="# forming heads with ≥1 relay")
    at.set_ylabel("# relay heads", color="C8")
    at.legend(fontsize=8, loc="lower right")

    ax[-1].set_xlabel("training step  (log; step 0 drawn at 0.5)")
    for s in STEPS:
        ax[-1].annotate(str(s), (max(s, 0.5), 0), xytext=(0, -22), textcoords="offset points",
                        ha="center", va="top", fontsize=6, rotation=90, color="0.4")
    fig.tight_layout()
    png = "data/analysis/colocation_panel.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print("wrote", png)

    # echo the table
    print("\n=== colocation_panel.csv ===")
    print(open(csv_path).read())


if __name__ == "__main__":
    main()
