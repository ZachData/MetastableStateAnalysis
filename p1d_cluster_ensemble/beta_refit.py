"""
beta_refit.py — β per head with per-offset fixed effects and an R² floor
(Phase 1d, `status-1d.md` "β refit").

#108 measured ``beta_raw`` on unit LN1 rows (`attention_graph.fit_betas`:
row fixed effects plus a *linear* offset term) and got a median of 3.88
[1.92, 5.94] over 7 prompts × L1–23 × 16 heads of 410m step143000. Its
review asked whether recency heads inflate that: a recency head's
log-attention is not linear in offset, and the residual's similarity also
falls with offset, so a linear control can leave curvature on the slope.

This refits the same heads on the same inputs (real attention only, no
draws) under:

- ``linear``: `fit_betas`'s estimator, which must reproduce #108's stored
  per-head βs before anything else is read (refuses otherwise);
- ``fe_w<W>``: one dummy per offset below W, a pooled bin with a linear slope
  beyond (`core.beta_eff.estimate_beta_offset_fe`);
- ``fe_full``: one dummy per offset.

The R² floor is on ``fe_full``'s partial R² (the share of what row and
offset effects leave that similarity explains), so every variant is
summarised over the same heads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out

from .attention_graph import fit_betas, unit_ln_rows
from .attention_null import _step_prompt, blocks_for, run_checkpoint
from .gaussian_null import MIN_TOKENS, first_occurrences, run_tokens

WINDOWS_FE = (4, 16, 64, None)
VARIANTS = ("linear",) + tuple(f"fe_w{w}" if w else "fe_full" for w in WINDOWS_FE)
#: Floors on fe_full's partial R². 0 keeps every head with a finite fit.
FLOORS = (0.0, 0.01, 0.05, 0.10)
#: Prompts the headline tables leave out (as `attention_null_report.EXCLUDE`).
EXCLUDE = ("repeated_tokens",)
#: Largest |linear β − stored β| accepted as a reproduction (stored rounded to 5 dp).
REPRO_TOL = 1e-4


def _band(layer: int) -> str:
    """Attention blocks 0-23, in the bands the 1d reports use."""
    return "L0" if layer == 0 else ("L1-8" if layer <= 8 else ("L9-16" if layer <= 16 else "L17-23"))


def _job(args) -> List[Dict]:
    from core.beta_eff import estimate_beta_offset_fe
    run_dir, wroot, dedupe = args
    run_dir = Path(run_dir)
    step, prompt = _step_prompt(run_dir)
    z = np.load(run_dir / "activations.npz")
    acts, norms = z["activations"], z["norms"]
    n = acts.shape[1]
    keep = np.arange(1, n)
    if dedupe:
        keep = first_occurrences(run_tokens(run_dir))
        keep = keep[keep != 0]
        if keep.size < MIN_TOKENS:
            return []
    seq = np.concatenate([[0], keep])
    A_all = np.load(run_dir / "attentions.npz")["attentions"]
    _, rev = run_checkpoint(run_dir)
    n_blocks = acts.shape[0] - 1
    blocks = blocks_for(Path(wroot), rev, range(n_blocks))
    out = []
    for L in range(n_blocks):
        X = acts[L][seq].astype(np.float64) * norms[L][seq][:, None]
        A = A_all[L][:, seq[:, None], seq[None, :]].astype(np.float64)
        B = blocks[L]
        U = unit_ln_rows(X, B.ln1_w, B.ln1_b, B.eps)
        G = U @ U.T
        idx = np.arange(1, seq.size)
        lin = fit_betas(A, U, idx, seq)
        for h in range(A.shape[0]):
            rec = {"step": step, "prompt": prompt, "layer": L, "head": h, "dedupe": bool(dedupe),
                   "linear": lin[h]["beta"], "linear_r2": lin[h]["r2"]}
            for w in WINDOWS_FE:
                r = estimate_beta_offset_fe(A[h], G, idx, positions=seq, offset_window=w)
                k = f"fe_w{w}" if w else "fe_full"
                rec[k], rec[k + "_r2"], rec[k + "_pr2"] = r["beta_raw"], r["r2"], r["partial_r2"]
                rec[k + "_svr"] = r["design_sv_ratio"]
            out.append(rec)
    return out


def check_reproduction(recs: List[Dict], stored: Path) -> Dict:
    """
    Max |linear − #108's stored β| over the heads both have. A head on one
    side only, fitted twice, or finite on one side only counts as mismatched.
    """
    d = json.loads(stored.read_text())
    ref = {}
    for r in d["records"]:
        if "skipped" in r:
            continue
        for h, b in enumerate(r["betas"][0]):
            ref[(r["step"], r["prompt"], r["layer"], h)] = b["beta"]
    keys = [(x["step"], x["prompt"], x["layer"], x["head"]) for x in recs]
    mine = dict(zip(keys, (x["linear"] for x in recs)))
    # Every fitted head must be one #108 recorded, once: an unmatched head
    # would reach the summary unchecked.
    mismatched = (len(keys) - len(mine)) + sum(k not in ref for k in mine)
    diffs = []
    for k, b in ref.items():
        stored_ok = b is not None and np.isfinite(b)
        new = mine.get(k)
        new_ok = new is not None and np.isfinite(new)
        if stored_ok and new_ok:
            diffs.append(abs(new - b))
        elif stored_ok or (k in mine and new_ok):
            mismatched += 1          # finite on one side only, or a stored head not refitted
    return {"n_compared": len(diffs), "n_mismatched": mismatched,
            "max_abs_diff": float(max(diffs)) if diffs else None}


def summarise(recs: List[Dict]) -> List[Dict]:
    rows = []
    groups: Dict = {}
    for r in recs:
        if r["prompt"] in EXCLUDE:
            continue
        bands = [_band(r["layer"])] + (["L1-23"] if r["layer"] >= 1 else [])
        for b in bands:
            groups.setdefault((r["dedupe"], r["step"], b), []).append(r)
    for (dedupe, step, band), rs in sorted(groups.items()):
        pr2 = np.array([r["fe_full_pr2"] for r in rs], dtype=float)
        for floor in FLOORS:
            sel = [r for r, p in zip(rs, pr2) if np.isfinite(p) and p >= floor]
            row = {"dedupe": dedupe, "step": step, "band": band, "floor": floor,
                   "n": len(sel), "n_all": len(rs)}
            for v in VARIANTS:
                b = np.array([r[v] for r in sel], dtype=float)
                b = b[np.isfinite(b)]
                # A head a variant refused (NaN) drops out of that variant only;
                # its own count says so.
                row[v] = ([float(np.median(b)), float(np.percentile(b, 25)),
                           float(np.percentile(b, 75)), int(b.size)] if b.size else None)
            row["linear_r2_median"] = float(np.nanmedian([r["linear_r2"] for r in sel])) if sel else None
            row["fe_full_r2_median"] = float(np.nanmedian([r["fe_full_r2"] for r in sel])) if sel else None
            row["fe_full_pr2_median"] = float(np.nanmedian([r["fe_full_pr2"] for r in sel])) if sel else None
            rows.append(row)
    return rows


def text(rows: List[Dict], repro: Dict) -> str:
    lines = [f"reproduction of #108's linear β: {repro}", "",
             "dedupe step band floor n/n_all | " + " | ".join(VARIANTS)
             + " | R² lin, R² full, partial R² full"]
    for r in rows:
        cells = " | ".join("—" if r[v] is None else f"{r[v][0]:.2f} [{r[v][1]:.2f}, {r[v][2]:.2f}]"
                           + ("" if r[v][3] == r["n"] else f" (n={r[v][3]})") for v in VARIANTS)
        lines.append(f"{'dd' if r['dedupe'] else 'all'} {r['step']} {r['band']} "
                     f"{r['floor']:.2f} {r['n']}/{r['n_all']} | {cells} | "
                     + ", ".join("—" if r[k] is None else f"{r[k]:.3f}" for k in
                                 ("linear_r2_median", "fe_full_r2_median", "fe_full_pr2_median")))
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--stored", type=Path, default=None,
                    help="#108's null.json (all tokens), whose βs the linear fit must reproduce; "
                         "omit only for inputs #108 never fitted (the long prompts), and the "
                         "output records the check as not run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--resummarise", action="store_true",
                    help="rebuild the summary from --out's stored heads; no fits")
    add_holdout_args(ap)
    args = ap.parse_args(argv)
    if args.resummarise:
        out = json.loads(args.out.read_text())
        out["summary"] = summarise(out["heads"])
        args.out.write_text(json.dumps(out))
        args.out.with_suffix(".txt").write_text(text(out["summary"], out["reproduction"]))
        return 0
    runs, _ = refuse_held_out(list(args.runs), allow=args.allow_holdout,
                              drop=args.v1_only, context="beta_refit")
    if not runs:
        print("refusing: no runs", file=sys.stderr)
        return 1
    t0 = time.time()
    jobs = [(str(r), str(args.weights), dd) for dd in (False, True) for r in runs]
    # One part file per (run, dedupe) job, written as it finishes, so a
    # stopped run resumes (as `attention_null` and `gaussian_null`).
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)

    def part_of(job) -> Path:
        return parts / f"{Path(job[0]).parent.name}__{Path(job[0]).name}__dd{int(job[2])}.json"

    recs: List[Dict] = []
    todo = []
    for j in jobs:
        if part_of(j).exists():
            recs.extend(json.loads(part_of(j).read_text()))
        else:
            todo.append(j)
    print(f"  {len(jobs) - len(todo)} of {len(jobs)} jobs already done", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, j): j for j in todo}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            part = f.result()
            tmp = part_of(futs[f]).with_suffix(".tmp")
            tmp.write_text(json.dumps(part))
            tmp.replace(part_of(futs[f]))
            recs.extend(part)
            print(f"  done {futs[f][0]} dedupe={futs[f][2]} ({time.time() - t0:.0f} s)", flush=True)
    recs.sort(key=lambda r: (r["dedupe"], r["step"], r["prompt"], r["layer"], r["head"]))
    if args.stored is None:
        repro = {"not_run": "no --stored: #108 has no βs for these inputs; the estimator "
                            "reproduced #108 on the v1 runs (status-1d.md \"β refit\")"}
    else:
        repro = check_reproduction([r for r in recs if not r["dedupe"]], args.stored)
    if args.stored is not None and (not repro["n_compared"] or repro["n_mismatched"]
                                    or repro["max_abs_diff"] > REPRO_TOL):
        print(f"refusing: linear fit does not reproduce #108's βs: {repro}", file=sys.stderr)
        return 1
    rows = summarise(recs)
    out = {"runs": [str(r) for r in runs], "stored": args.stored and str(args.stored),
           "reproduction": repro,
           "variants": list(VARIANTS), "floors": list(FLOORS), "exclude": list(EXCLUDE),
           "seconds": round(time.time() - t0, 1), "summary": rows, "heads": recs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    args.out.with_suffix(".txt").write_text(text(rows, repro))
    print(text(rows, repro))
    return 0


if __name__ == "__main__":
    sys.exit(main())
