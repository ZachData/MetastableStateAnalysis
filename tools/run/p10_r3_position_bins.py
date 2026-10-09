"""R3, after `/challenge-pr` on #148 (finding 2): is A0's corrected gap position, not membership?
(`p10_cluster_function/status-10.md` §1.17.) The mask correction divides out uniform causal
attention only; a trained model's own position profile (recency) is left in, and the definition's
members sit earlier than the rest. Per readable unit, the corrected gap (rest − members, in the
domain's mean) is read twice:

  pooled   as A0 reads it
  binned   inside ``N_BINS`` equal-count position bins of the domain, each bin's rest − members
           weighted by n_members·n_rest / n_bin, bins without both left out

so ``binned`` compares members with rest at the same positions. Sizes only, no permutations.
Columns: c0 (published reader, all positions) and the T4 columns named (``--columns``; R9 adds
c3x and its learned split on the c3x source). Run:
    python tools/run/p10_r3_position_bins.py --labels <R0 labels> --out <file> [--jobs 14] [--columns ...]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from core.parking import mask_corrected_received
from tools.run.p10_attention_baseline import OLD_READER, _load_attentions, t1_t2_positions, t4_vectors
from tools.run.p10_token_composition import OUTSIDE

N_BINS = 8  # placed
COLUMNS = ("c0", "c1", "c2", "c3", "c3_learned", "c3_unlearned")


def gaps(values, labels, n_bins: int = N_BINS):
    """(pooled, binned) rest − members over the domain, in units of the domain's mean."""
    lab = np.asarray(labels)
    dom = np.flatnonzero(lab != OUTSIDE)
    v, lab = np.asarray(values, dtype=np.float64)[dom], lab[dom]
    mem, scale = lab >= 0, v.mean()
    pooled = (v[~mem].mean() - v[mem].mean()) / scale
    num = den = 0.0
    for b in np.array_split(np.arange(dom.size), n_bins):  # dom is sorted: bins by position
        m, r = mem[b], ~mem[b]
        if m.any() and r.any():
            w = m.sum() * r.sum() / b.size
            num += w * (v[b][r].mean() - v[b][m].mean())
            den += w
    return float(pooled), (float(num / den / scale) if den else None)


def run(run_dir, labels_by_col, dropped, key):
    attn = _load_attentions(Path(run_dir))
    out = []
    for layer in sorted({L for by in labels_by_col.values() for L in by}):
        if layer >= attn.shape[0]:
            continue
        A, t4 = attn[layer], None
        for col, by in labels_by_col.items():
            if layer not in by:
                continue
            if col in OLD_READER:
                corr = mask_corrected_received(A)
            else:
                t4 = t4 if t4 is not None else t4_vectors(A, dropped)[1]
                corr = t4
            p, b = gaps(corr, by[layer])
            out.append({"key": key, "column": col, "layer": int(layer), "pooled": round(p, 4),
                        "binned": None if b is None else round(b, 4)})
    return out


def main():
    from tools.run.p10_label_source import reader_input
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--columns", nargs="+", default=list(COLUMNS), help=f"default {' '.join(COLUMNS)}")
    a = ap.parse_args()
    cols = tuple(a.columns)
    dropped, ts = t1_t2_positions(a.labels)
    srcs = {c: reader_input(a.labels, c) for c in cols}
    runs = {k: p for s in srcs.values() for k, p in s["runs"].items()}
    keys = sorted(k for k in runs if any(srcs[c]["labels"].get(k) for c in cols))
    jobs = [(runs[k], {c: srcs[c]["labels"][k] for c in cols if srcs[c]["labels"].get(k)},
             dropped[k[1]], f"{k[0]}|{k[1]}") for k in keys]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(run, *zip(*jobs)) for r in rs]
    summ = {}
    for c in cols:
        by = {}
        for r in rows:
            if r["column"] == c and r["binned"] is not None:
                by.setdefault(int(r["key"].split("|")[0]), []).append(r)
        summ[c] = {s: {"n": len(rs), "pooled": round(float(np.mean([r["pooled"] for r in rs])), 4),
                       "binned": round(float(np.mean([r["binned"] for r in rs])), 4)}
                   for s, rs in sorted(by.items())}
    a.out.write_text(json.dumps({"labels": str(a.labels), "token_sets": ts, "n_bins": N_BINS,
                                 "summary": summ, "rows": rows}, indent=1))
    for c, by in summ.items():
        print(c, " ".join(f"{s}:{v['pooled']:+.3f}/{v['binned']:+.3f}" for s, v in by.items()))


if __name__ == "__main__":
    main()
