"""
p1e_energy_field/u1_report.py — U1's labels by the rule fixed before output (`design-1e.md`
"U1: the rule", row *labels*), from `u1_field`'s records.

Per (step, band), each passage's value the mean over the band's layers, and U2's sign rule over
passages (`u2_report.label`): **(1) lumpy** ``Xe`` (β 1.6 / 3.5 / 5.6), **(2) wells** ``Xw``
(β 3.5), **(3) density and position** Spearman(``e``, log offset) (all three β), **(4) what the
wells are** ``ami_pos − ami_cls`` over passages with ≥ 2 wells of ≥ 2 targets in the band (fewer
than 7 → not read). Chance counts and isolated leans as U2. Writes ``labels.json`` and prints
the tables.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .extract_long8 import STEPS
from .u1_field import BANDS, MERGE_BESIDE, SWEEP, default_opts
from .u2_report import chance

TRAINED = ("L1-8", "L9-16", "L17-23")
NAMES = {"Xe": ("lumpier", "smoother"), "Xw": ("more wells", "fewer wells"),
         "rho_pos": ("denser later", "denser early"), "ami_diff": ("position", "content")}
SHORT = {"lumpier": "LUM", "smoother": "SMO", "more wells": "MORE", "fewer wells": "FEW",
         "denser later": "LATE", "denser early": "EARLY", "position": "POS", "content": "CON"}
MIN_READ = 7
#: The record keys of the beside merge tolerances (`u1_field.mean_shift`).
TOL_KEYS = tuple(f"k_{t:g}" for t in MERGE_BESIDE)


def sign_label(xs, names) -> str:
    xs = np.asarray(xs, dtype=float)
    n, pos, neg = xs.size, int(np.sum(xs > 0)), int(np.sum(xs < 0))
    up, down = names
    if pos == n:
        return up
    if neg == n:
        return down
    if pos == n - 1:
        return f"leans {up}"
    if neg == n - 1:
        return f"leans {down}"
    return "mixed"


def load(out: Path, kind: str) -> List[Dict]:
    recs = []
    for p in sorted((out / "records" / kind).glob("*.json")):
        r = json.loads(p.read_text())
        for c in r["cells"]:
            if c.get("ami_pos") is not None:
                c["ami_diff"] = c["ami_pos"] - c["ami_cls"]
            recs.append({"step": int(r["step"]), "passage": r["passage"], **c})
    return recs


def band_values(recs: List[Dict], stat: str, targets: str, beta: float) -> Dict:
    """``{(step, band): {passage: mean over the band's layers}}`` (layers lacking the stat skipped)."""
    acc = defaultdict(lambda: defaultdict(list))
    for c in recs:
        if c.get("sweep") or c["targets"] != targets or c["beta"] != beta or c.get(stat) is None:
            continue
        for band, layers in BANDS.items():
            if c["layer"] in layers:
                acc[(c["step"], band)][c["passage"]].append(c[stat])
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in acc.items()}


def labels(vals: Dict, stat: str, n_passages: int) -> Dict:
    out = {}
    for k, d in vals.items():
        if stat == "ami_diff":
            out[k] = sign_label(list(d.values()), NAMES[stat]) if len(d) >= MIN_READ else "not read"
        elif len(d) != n_passages:
            raise SystemExit(f"refusing: {stat} {k} has {len(d)} passages, expected {n_passages}")
        else:
            out[k] = sign_label(list(d.values()), NAMES[stat])
    return out


def _dir(lab: str, names) -> int:
    return 1 if names[0] in lab else (-1 if names[1] in lab else 0)


def isolated(labs: Dict, k: tuple, names) -> bool:
    step, band = k
    i = STEPS.index(step)
    near = [STEPS[j] for j in (i - 1, i + 1) if 0 <= j < len(STEPS)]
    return not any(_dir(labs.get((s, band), "mixed"), names) == _dir(labs[k], names) for s in near)


def run_opts(out: Path) -> Dict:
    """The run's frame and β set (`plan.json`; U1's own run predates them and is the default)."""
    plan = out / "plan.json"
    opts = json.loads(plan.read_text()).get("opts") if plan.exists() else None
    return opts or default_opts()


def summarise(out: Path, kind: str) -> Dict:
    opts = run_opts(out)
    BETAS, PRIMARY_BETA = tuple(opts["betas"]), opts["primary_beta"]
    recs = load(out, kind)
    n_pass = len({c["passage"] for c in recs})
    tsets = sorted({c["targets"] for c in recs})
    res = {"opts": opts, "n_passages": n_pass, "chance_per_54": chance(n_pass, 54), "tables": {}}
    for t in tsets:
        betas = BETAS if t in ("t12", "r0") else (PRIMARY_BETA,)
        for stat in ("Xe", "Xw", "rho_pos", "ami_diff"):
            for beta in (betas if stat in ("Xe", "rho_pos") else (PRIMARY_BETA,)):
                vals = band_values(recs, stat, t, beta)
                if not vals:
                    continue
                labs = labels(vals, stat, n_pass)
                tab = {}
                for (step, band), lab in sorted(labs.items()):
                    tab[f"{step}|{band}"] = {
                        "label": lab, "isolated": "leans" in lab and isolated(labs, (step, band), NAMES[stat]),
                        "median": float(np.median(list(vals[(step, band)].values()))),
                        "n": len(vals[(step, band)])}
                res["tables"][f"{t}|{stat}|{beta:g}"] = tab
    # beside: medians per (step, band) of the descriptive stats, primary target set and β
    prim = "t12" if kind == "long" else "r0"
    beside = {}
    for stat in ("k", "k2", "k_eff", "largest", "sd_e", "r2_pos", "r2_mean", "rho_pos_causal",
                 "ami_pos", "ami_cls", "open_share", "open_well_share", "purity", "purity_null",
                 "unconverged", *TOL_KEYS):
        for beta in BETAS:
            vals = band_values(recs, stat, prim, beta)
            for k, d in vals.items():
                beside.setdefault(f"{k[0]}|{k[1]}|{beta:g}", {})[stat] = float(np.median(list(d.values())))
    res["beside"] = beside
    # the merge tolerance (placed): cells whose well count changes at 1e-4 or 1e-2 (the record's keys)
    cells = [c for c in recs if not c.get("sweep") and c["targets"] == prim]
    if any(t not in c for c in cells for t in TOL_KEYS):
        raise SystemExit(f"refusing: a record lacks one of {TOL_KEYS}")
    res["merge_tolerance"] = {"cells": len(cells),
                              "k_changes": sum(any(c[t] != c["k"] for t in TOL_KEYS) for c in cells)}
    sweep = defaultdict(list)
    for c in recs:
        if c.get("sweep") or (c["targets"] == prim and c["layer"] in (4, 12, 20) and c["beta"] in BETAS):
            if c["targets"] == prim:
                sweep[(c["step"], c["layer"], c["beta"])].append(c["k_eff"])
    res["sweep_k_eff"] = {f"{s}|L{L}|{b:g}": float(np.median(v)) for (s, L, b), v in sorted(sweep.items())}
    return res


def print_table(res: Dict, key: str, title: str) -> None:
    tab = res["tables"].get(key)
    if not tab:
        return
    print(f"\n{title}  [{key}]")
    print(f"{'step':>7} " + " ".join(f"{b:>16}" for b in TRAINED) + f" {'L0':>16}")
    count = defaultdict(int)
    for s in STEPS:
        row = []
        for b in TRAINED + ("L0",):
            c = tab.get(f"{s}|{b}")
            if c is None:
                row.append(f"{'':>16}")
                continue
            lab = c["label"]
            short = lab if lab in ("mixed", "not read") else (
                SHORT[lab.replace("leans ", "")].lower() if lab.startswith("leans") else SHORT[lab])
            if b in TRAINED:
                count[lab] += 1
            row.append(f"{short + ('*' if c['isolated'] else ''):>8} {c['median']:+.3f}"[:16].rjust(16))
        print(f"{s:>7} " + " ".join(row))
    print("  trained cells:", dict(count), "| chance per 54:", res["chance_per_54"])


def report(out: Path) -> int:
    from .u1_audit import audit_passes
    audit_passes(out)                        # read only behind a passing audit (design-1e.md)
    full = {}
    opts = run_opts(out)
    BETAS, pb = tuple(opts["betas"]), opts["primary_beta"]
    print(f"frame {opts['frame']}, β {BETAS} (primary {pb:g}), sweep {opts['sweep']}")
    for kind in ("long", "v1"):
        if not (out / "records" / kind).exists():
            continue
        res = full[kind] = summarise(out, kind)
        print(f"\n===== {kind} ({res['n_passages']} passages) =====")
        prim = "t12" if kind == "long" else "r0"
        for beta in BETAS:
            print_table(res, f"{prim}|Xe|{beta:g}", f"(1) lumpy, β {beta:g}")
        print_table(res, f"{prim}|Xw|{pb:g}", f"(2) wells against the Gaussian, β {pb:g}")
        for beta in BETAS:
            print_table(res, f"{prim}|rho_pos|{beta:g}", f"(3) density and position, β {beta:g}")
        print_table(res, f"{prim}|ami_diff|{pb:g}", f"(4) what the wells are, β {pb:g}")
        if kind == "long":
            for k in ("Xe", "Xw", "rho_pos", "ami_diff"):
                print_table(res, f"t123|{k}|{pb:g}", f"T1–T3 beside: {k}")
    (out / "labels.json").write_text(json.dumps(full, indent=1) + "\n")
    for kind, r in full.items():
        print(f"merge tolerance ({kind}): well count changes at 1e-4 or 1e-2 in {r['merge_tolerance']['k_changes']} of {r['merge_tolerance']['cells']} cells")
    print(f"\nwrote {out / 'labels.json'}" + (f" (sweep β {SWEEP} at L4/12/20 in 'sweep_k_eff')"
                                               if opts["sweep"] else ""))
    return 0
