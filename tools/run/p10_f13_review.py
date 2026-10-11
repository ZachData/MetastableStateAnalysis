#!/usr/bin/env python3
"""
F13's post-review checks (`/challenge-pr` on #187, findings 3–5), beside the rule, after its output
was read: **post hoc, not labels the rule fixed**. Unit LN1 rows, A9's layers (1–23), stored order only.

Per (step, passage, layer), on the kept tokens (position 0 and the T-tokens never on either side):
  rho_all   rank correlation of log ρ with each token's nearest-predecessor distance m (every
            predecessor), both first residualised on a cubic in log position: threshold-free (finding 3)
  rho_kept  the same with m over kept predecessors only, so position 0 and the T-tokens cannot make
            a token a non-centre (finding 4)
  sink_nn   the share of c = 2 non-centres whose nearest predecessor is position 0 or a T-token (4)
  S_c_kept  A9's S_c with centres from the kept-only m (4)
  S_c_16, S_c_drop1   S_c with 16 position bins, and with the first of 8 bins dropped (5)
  S_c_q10   S_c with the top 10 % of kept tokens by m as centres, a fixed centre share (3)

    python tools/run/p10_f13_review.py --out <F13 output dir> [--workers 14]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_f13_centres as f

SHARE = 0.10                                                # placed: S_c_q10's centre share
STATS = ("rho_all", "rho_kept", "S_c_kept", "S_c_16", "S_c_drop1", "S_c_q10")


def rank(v: np.ndarray) -> np.ndarray:
    r = np.empty(v.size)
    r[np.argsort(v, kind="stable")] = np.arange(v.size)
    return r


def partial_rank_corr(x: np.ndarray, y: np.ndarray, pos: np.ndarray) -> Optional[float]:
    """Pearson of rank(x), rank(y) after each is residualised on a cubic in log(pos + 1)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < f.MIN_B:
        return None
    lp = np.log(pos[ok] + 1.0)
    X = np.vstack([np.ones_like(lp), lp, lp ** 2, lp ** 3]).T
    rx, ry = rank(x[ok]), rank(y[ok])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > 0 else None


def nearest_pred_from(D: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Each token's distance to its nearest earlier token in ``allowed`` (inf if none)."""
    n = D.shape[0]
    M = np.where(np.tril(np.ones((n, n), dtype=bool), -1) & allowed[None, :], D, np.inf)
    return M.min(axis=1)


def read_one(args) -> Dict:
    step, passage, run_dir, kept, ln1_dir = args
    from p1e_energy_field.u2_block import export_ln1, load_ln1, unit_rows
    rd = Path(run_dir)
    acts = np.load(rd / "activations.npz")["activations"]
    attn = np.load(rd / "attentions.npz")["attentions"]
    tokens = json.loads((rd / "geometry.json").read_text())["tokens"]
    if passage == "homer_iliad":
        acts, attn, tokens = acts[:, :f.HOMER_MAX], attn[:, :, :f.HOMER_MAX, :f.HOMER_MAX], tokens[:f.HOMER_MAX]
    n = acts.shape[1]
    kept = np.asarray(sorted(int(k) for k in kept if k != 0))
    kmask = np.zeros(n, dtype=bool)
    kmask[kept] = True
    sinks = np.zeros(n, dtype=bool)
    sinks[0] = True
    sinks[f.t2_positions(tokens, kept)] = True
    ln1 = load_ln1(export_ln1(f"step{step}", Path(ln1_dir)))
    dl = f.DELTAS[f.CS.index(f.PRIMARY_C)]
    b8, b16 = f.bins_of(kept, 8), f.bins_of(kept, 16)
    out: Dict = {"step": step, "passage": passage, "layers": {}}
    for L in f.A9_LAYERS:
        D = f.geodesic(unit_rows(acts[L].astype(np.float64), ln1["w"][L], ln1["b"][L], ln1["eps"]))
        lr = f.log_rho(attn[L])
        m_all = nearest_pred_from(D, np.ones(n, dtype=bool))
        m_kept = nearest_pred_from(D, kmask)
        lower = np.where(np.tril(np.ones((n, n), dtype=bool), -1), D, np.inf)
        nn = lower.argmin(axis=1)
        cen = kept[m_all[kept] > dl]
        non = np.setdiff1d(kept, cen)
        cen_k = kept[m_kept[kept] > dl]
        q = max(int(round(SHARE * kept.size)), f.MIN_A)
        cen_q = kept[np.argsort(-m_all[kept], kind="stable")[:q]]
        drop1 = [b for b in b8[1:]]
        mk = m_kept[kept]
        out["layers"][str(L)] = {
            "rho_all": partial_rank_corr(m_all[kept], lr[kept], kept.astype(float)),
            "rho_kept": partial_rank_corr(np.where(np.isfinite(mk), mk, np.nan), lr[kept], kept.astype(float)),
            "sink_nn": float(sinks[nn[non]].mean()) if non.size else None,
            "S_c_kept": f.binned_diff(lr, cen_k, np.setdiff1d(kept, cen_k), b8),
            "S_c_16": f.binned_diff(lr, cen, non, b16),
            "S_c_drop1": f.binned_diff(lr, cen, non, drop1),
            "S_c_q10": f.binned_diff(lr, cen_q, np.setdiff1d(kept, cen_q), b8),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.run import p10_label_source as ls
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=DATA / "p10" / "reread_r9_2026-10-08" / "labels")
    ap.add_argument("--ln1", type=Path, default=DATA / "p1e" / "u2_block_2026-10-07" / "ln1")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(argv)
    jobs = []
    for s in f.STEPS:
        d = ls.load_step(a.labels, f"step{s}")
        for p in f.PASSAGES:
            pr = d["prompts"][p]
            jobs.append((s, p, pr["stage0_run"], pr["kept"], str(a.ln1)))
    with Pool(a.workers) as pool:
        recs = {(r["step"], r["passage"]): r for r in pool.imap_unordered(read_one, jobs)}
    (a.out / "review.json").write_text(json.dumps([recs[k] for k in sorted(recs)]) + "\n")
    lines = ["F13 post-review checks (post hoc, beside; /challenge-pr on #187 findings 3-5); unit LN1, c = 2",
             "rho_*: + = farther from every predecessor -> attended more (position partialled out)"]
    for stat in STATS:
        names = ("attended more", "attended less") if stat.startswith("S_c") else ("positive", "negative")
        grid = {b: {s: [f.band_mean({int(L): v[stat] for L, v in recs[(s, p)]["layers"].items()}, Ls)
                         for p in f.PASSAGES] for s in f.STEPS} for b, Ls in f.BANDS.items()}
        labs = f.labelled(grid, *names)
        lines += f.table(stat, labs, f.STEPS)
        med = {b: {s: np.median([x for x in grid[b][s] if x is not None]) if any(x is not None for x in grid[b][s]) else np.nan
                   for s in f.STEPS} for b in f.BANDS}
        lines.append("  band medians: " + "; ".join(
            f"{s}: " + " ".join(f"{med[b][s]:+.2f}" for b in f.BANDS) for s in (64, 1000, 4000, 16000, 32000, 143000)))
    lines.append("\nsink_nn: share of c = 2 non-centres whose nearest predecessor is position 0 or a T-token (median, L9-16)")
    for s in f.STEPS:
        v = [f.band_mean({int(L): x["sink_nn"] for L, x in recs[(s, p)]["layers"].items()}, f.BANDS["L9-16"]) for p in f.PASSAGES]
        v = [x for x in v if x is not None]
        lines.append(f"  {s:<7} {np.median(v):.2f}" if v else f"  {s:<7} —")
    (a.out / "review.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
