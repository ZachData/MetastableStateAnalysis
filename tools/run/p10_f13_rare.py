#!/usr/bin/env python3
"""
F13r (`design-10.md` "F13r"): F13's strong centres at c = 2 joined to token frequency, class and
place in the sentence. Post hoc on F13's read output (`status-10.md` §1.41); no forward pass, no
centre recomputed.

Per (step, passage, layer 1–23, frame), on the kept tokens, centres = F13's stored
`centres_strong` (c = 2):
  C_f, C_id, C_sp, C_ws, C_ct   centres − others, A9's 8 position bins: lf = log10 Pile frequency,
                                log10 merge id (beside), tokens since the sentence began, the
                                word_start and continuation indicators
  R_f, R_sp                     on non-centres only: rank correlation of log ρ with lf (sp), both
                                residualised on a cubic in log position
  S_c4                          log ρ, centres − others, 4 position bins (the reference)
  S_c|f, S_c|sp, S_c|cls        the same inside strata = 4 position bins × lf tercile / sp tercile /
                                class (word_start, continuation, rest)
  S_c~f, S_c~sp, S_c~cls        the centre coefficient of least squares on centre, the covariate
                                and a cubic in log position (amended: terciles leave a tail)
  S_c|all                       the centre coefficient of least squares on centre, lf, the class
                                indicators, sp and a cubic in log position (beside)

    python tools/run/p10_f13_rare.py run --out <dir> [--workers 14]
    python tools/run/p10_f13_rare.py report --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_f13_centres as f
from tools.run.p10_f13_review import partial_rank_corr

F13_OUT = DATA / "p10" / "f13_centres_2026-10-10"
F13_CODE = "f0ce454c465c"
COUNTS = DATA / "p10" / "pile_unigram_2026-10-10"
N_POS = 4                                                   # position bins inside a stratum (the rule's)
N_Q = 3                                                     # lf and sp terciles
FIRST = (8000, "wiki_paragraph")
CLASSES = ("word_start", "continuation")                    # the rest is the third class
STATS = ("C_f", "C_id", "C_sp", "C_ws", "C_ct", "R_f", "R_sp", "S_c4", "S_c|f", "S_c|sp", "S_c|cls",
         "S_c~f", "S_c~sp", "S_c~cls", "S_c|all")
NAMES = {"C_f": ("centres commoner", "centres rarer"), "C_id": ("higher merge id", "lower merge id"),
         "C_sp": ("later in the sentence", "earlier in the sentence"),
         "C_ws": ("more word starts", "fewer word starts"), "C_ct": ("more continuations", "fewer continuations"),
         "R_f": ("commoner attended more", "commoner attended less"),
         "R_sp": ("later attended more", "later attended less")}
for _s in ("S_c4", "S_c|f", "S_c|sp", "S_c|cls", "S_c~f", "S_c~sp", "S_c~cls", "S_c|all"):
    NAMES[_s] = ("attended more", "attended less")
COVARIATES = {"f": ("S_c|f", "S_c~f", "C_f", "R_f"), "sp": ("S_c|sp", "S_c~sp", "C_sp", "R_sp"),
              "cls": ("S_c|cls", "S_c~cls", None, None)}


class RareError(RuntimeError):
    pass


# ---------------------------------------------------------------- features (pure)
def is_boundary(text: Optional[str]) -> bool:
    if text is None:
        return False
    return "\n" in text or text.strip().endswith((".", "!", "?"))


def sentence_pos(texts: Sequence[Optional[str]]) -> np.ndarray:
    """Tokens since the last boundary; position 0 and the token after a boundary have 0."""
    sp = np.zeros(len(texts))
    for i in range(1, len(texts)):
        sp[i] = 0 if is_boundary(texts[i - 1]) else sp[i - 1] + 1
    return sp


def tercile(v: np.ndarray, kept: np.ndarray, n_q: int = N_Q) -> np.ndarray:
    """Level 0..n_q−1 of every position by the kept tokens' quantiles of v."""
    edges = np.quantile(v[kept], np.arange(1, n_q) / n_q)
    return np.searchsorted(edges, v, side="right")


def strata(kept: np.ndarray, level: Optional[np.ndarray], n_pos: int = N_POS) -> List[np.ndarray]:
    """n_pos equal-count position bins of the kept tokens, each split by ``level``."""
    out = []
    for b in f.bins_of(kept, n_pos):
        if level is None:
            out.append(b)
        else:
            out += [b[level[b] == q] for q in np.unique(level[b])]
    return [b for b in out if b.size]


def matched_diff(v: np.ndarray, A: np.ndarray, B: np.ndarray, bins: List[np.ndarray],
                 min_a: int = f.MIN_A, min_b: int = f.MIN_B) -> Optional[float]:
    """Bin-wise mean(v[A]) − mean(v[B]) weighted by A's count, over bins holding both; None unless
    those bins hold ≥ min_a of A and ≥ min_b of B."""
    sa, sb = set(A.tolist()), set(B.tolist())
    num = na = nb = 0.0
    for b in bins:
        a = [x for x in b if x in sa]
        c = [x for x in b if x in sb]
        if a and c:
            num += len(a) * (v[a].mean() - v[c].mean())
            na += len(a)
            nb += len(c)
    if na < min_a or nb < min_b:
        return None
    return num / na


def ols_centre(y: np.ndarray, centre: np.ndarray, covs: Sequence[np.ndarray], pos: np.ndarray) -> Optional[float]:
    """The centre coefficient of least squares of y on centre, covs and a cubic in log(pos + 1)."""
    if centre.sum() < f.MIN_A or (~centre).sum() < f.MIN_B:
        return None
    lp = np.log(pos + 1.0)
    X = np.column_stack([np.ones_like(lp), centre.astype(float), *covs, lp, lp ** 2, lp ** 3])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if np.linalg.matrix_rank(X[ok]) < X.shape[1]:
        return None
    return float(np.linalg.lstsq(X[ok], y[ok], rcond=None)[0][1])


def layer_stats(lr: np.ndarray, kept: np.ndarray, cen: np.ndarray, feat: Dict[str, np.ndarray]) -> Dict:
    non = np.setdiff1d(kept, cen)
    b8, b4 = f.bins_of(kept, f.N_BINS), strata(kept, None)
    out = {k: f.binned_diff(feat[v], cen, non, b8)
           for k, v in (("C_f", "lf"), ("C_id", "lid"), ("C_sp", "sp"), ("C_ws", "ws"), ("C_ct", "ct"))}
    out["R_f"] = partial_rank_corr(feat["lf"][non], lr[non], non.astype(float))
    out["R_sp"] = partial_rank_corr(feat["sp"][non], lr[non], non.astype(float))
    out["S_c4"] = matched_diff(lr, cen, non, b4)
    out["S_c|f"] = matched_diff(lr, cen, non, strata(kept, feat["lf_q"]))
    out["S_c|sp"] = matched_diff(lr, cen, non, strata(kept, feat["sp_q"]))
    out["S_c|cls"] = matched_diff(lr, cen, non, strata(kept, feat["cls3"]))
    cmask = np.isin(kept, cen)
    for x, cols in (("f", ("lf",)), ("sp", ("sp",)), ("cls", ("ws", "ct"))):
        out[f"S_c~{x}"] = ols_centre(lr[kept], cmask, [feat[k][kept] for k in cols], kept.astype(float))
    out["S_c|all"] = ols_centre(lr[kept], cmask, [feat[k][kept] for k in ("lf", "ws", "ct", "sp")], kept.astype(float))
    out["n_centres"] = int(cen.size)
    return out


def features(tokens: Sequence[str], kept: np.ndarray, vocab: dict, added: set, counts: np.ndarray) -> Dict:
    from tools.run import p10_token_composition as tc
    tf = tc.token_features(np.asarray(tokens), vocab, added)
    ids = np.array([t["id"] for t in tf])
    texts = [t if i in added else tc.decode(t) for t, i in zip(tokens, ids)]
    total = counts.sum()
    cls = np.array([t["cls"] for t in tf])
    feat = {"lf": np.log10((counts[ids] + 0.5) / total),
            "lid": np.where(np.isin(ids, list(added)), np.nan, np.log10(np.maximum(ids, 1))),
            "sp": sentence_pos(texts),
            "ws": (cls == "word_start").astype(float), "ct": (cls == "continuation").astype(float)}
    feat["cls3"] = np.where(cls == "word_start", 0, np.where(cls == "continuation", 1, 2))
    feat["lf_q"] = tercile(feat["lf"], kept)
    feat["sp_q"] = tercile(feat["sp"], kept)
    feat["cls"] = cls
    feat["zero_count"] = int((counts[ids[kept]] == 0).sum())
    return feat


# ---------------------------------------------------------------- one (step, passage)
def check_first(feat: Dict, rec: Dict, kept: np.ndarray) -> None:
    if not np.isfinite(feat["lf"][kept]).all():
        raise RareError("first record: lf not finite at every kept token")
    lay = rec["frames"][f.PRIMARY_FRAME]["layers"]
    for L in f.BANDS["L1-8"]:
        cen = np.intersect1d(lay[str(L)]["centres_strong"], kept)
        if cen.size and np.unique(feat["cls3"][cen]).size < 2:
            raise RareError(f"first record L{L}: the centres are one class; amend before reading")


def _job(args) -> Dict:
    step, passage, run_dir, kept, f13_out, counts_dir = args
    from tools.run import p10_token_composition as tc
    t0 = time.monotonic()
    rec = json.loads((Path(f13_out) / "records" / f"step{step}_{passage}.json").read_text())
    if rec["code"] != F13_CODE:
        raise RareError(f"{step} {passage}: record code {rec['code']}, not {F13_CODE}")
    rd = Path(run_dir)
    attn = np.load(rd / "attentions.npz")["attentions"]
    tokens = json.loads((rd / "geometry.json").read_text())["tokens"]
    if passage == "homer_iliad":
        attn, tokens = attn[:, :, :f.HOMER_MAX, :f.HOMER_MAX], tokens[:f.HOMER_MAX]
    kept = np.asarray(sorted(int(k) for k in kept if k != 0))
    if kept.size != rec["kept"]:
        raise RareError(f"{step} {passage}: {kept.size} kept against the record's {rec['kept']}")
    counts = np.load(Path(counts_dir) / "counts.npy")
    vocab, added = tc.load_vocab(tc.find_tokenizer(Path(os.environ.get("HF_HOME", str(DATA / "hf")))))
    feat = features(tokens, kept, vocab, added, counts)
    if (step, passage) == FIRST:
        check_first(feat, rec, kept)
    out: Dict = {"step": step, "passage": passage, "kept": int(kept.size), "zero_count": feat["zero_count"],
                 "frames": {fr: {} for fr in f.FRAMES}}
    for L in f.A9_LAYERS:
        lr = f.log_rho(attn[L])
        if abs(lr[0] - rec["log_rho"]["pos0"][str(L)]) > 1e-9:
            raise RareError(f"{step} {passage} L{L}: log ρ at position 0 differs from the record's")
        for fr in f.FRAMES:
            cen = np.intersect1d(rec["frames"][fr]["layers"][str(L)]["centres_strong"], kept)
            out["frames"][fr][str(L)] = layer_stats(lr, kept, cen, feat)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def run(a) -> int:
    from tools.run import p10_label_source as ls
    meta = json.loads((Path(a.counts) / "meta.json").read_text())
    jobs = []
    for s in f.STEPS:
        d = ls.load_step(a.labels, f"step{s}")
        for p in f.PASSAGES:
            pr = d["prompts"][p]
            jobs.append((s, p, pr["stage0_run"], pr["kept"], str(a.f13), str(a.counts)))
    first = [j for j in jobs if (j[0], j[1]) == FIRST]
    rest = [j for j in jobs if (j[0], j[1]) != FIRST]
    a.out.mkdir(parents=True, exist_ok=True)
    recs = [_job(first[0])]                                  # the first record before the rest (the rule)
    print(f"first record {FIRST}: {recs[0]['seconds']}s, populated", flush=True)
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, rest):
            recs.append(r)
            print(f"done step {r['step']} {r['passage']}: {r['seconds']}s", flush=True)
    recs.sort(key=lambda r: (r["step"], r["passage"]))
    (a.out / "records.json").write_text(json.dumps(recs) + "\n")
    (a.out / "meta.json").write_text(json.dumps({
        "rule": 'design-10.md "F13r"', "git": f.git_head(), "code": code_sha(), "f13": str(a.f13),
        "f13_code": F13_CODE, "counts": str(a.counts), "counts_sha256": meta["counts_sha256"],
        "pile_total": meta["total"], "labels": str(a.labels)}, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------- report
def lab_sign(label: str, stat: str) -> int:
    pos, neg = NAMES[stat]
    return 1 if pos in label else -1 if neg in label else 0


def grids(recs: List[Dict], frame: str) -> Tuple[Dict, Dict]:
    """{stat: {band: {step: label}}}, {stat: {band: {step: median}}}."""
    by = {(r["step"], r["passage"]): r for r in recs}
    labs, meds = {}, {}
    for stat in STATS:
        g = {b: {s: [f.band_mean({int(L): v[stat] for L, v in by[(s, p)]["frames"][frame].items()}, Ls)
                     for p in f.PASSAGES if (s, p) in by] for s in f.STEPS} for b, Ls in f.BANDS.items()}
        labs[stat] = f.labelled(g, *NAMES[stat])
        meds[stat] = {b: {s: (float(np.median([x for x in v if x is not None]))
                              if any(x is not None for x in v) else None) for s, v in row.items()}
                      for b, row in g.items()}
    return labs, meds


def reading(labs: Dict, meds: Dict, band: str, step: int, x: str) -> Optional[str]:
    """The rule's reading for covariate x in a cell where S_c4 carries a label; None otherwise."""
    s4 = lab_sign(labs["S_c4"][band][step], "S_c4")
    if s4 == 0:
        return None
    sx, sl, cx, rx = COVARIATES[x]
    if all(lab_sign(labs[k][band][step], k) == s4 for k in (sx, sl)):
        return "survives"
    m4 = meds["S_c4"][band][step]
    shrunk = any(lab_sign(labs[k][band][step], k) != s4 and labs[k][band][step] != "too few"
                 and meds[k][band][step] is not None and m4 is not None
                 and abs(meds[k][band][step]) <= 0.5 * abs(m4) for k in (sx, sl))
    if x == "cls":
        moved = any(lab_sign(labs[c][band][step], c) != 0 for c in ("C_ws", "C_ct"))
    else:
        moved = lab_sign(labs[cx][band][step], cx) * lab_sign(labs[rx][band][step], rx) == s4
    return "carries" if shrunk and moved else "unclear"


def report(a) -> int:
    recs = json.loads((a.out / "records.json").read_text())
    meta = json.loads((a.out / "meta.json").read_text())
    lines = [f"F13r (design-10.md \"F13r\"); code {meta['code']}; F13 records {meta['f13_code']}; "
             f"Pile counts {meta['counts_sha256'][:12]} over {meta['pile_total']:,} tokens",
             f"kept tokens with Pile count 0: {sum(r['zero_count'] for r in recs)} of {sum(r['kept'] for r in recs)}"]
    out_json = {}
    for frame in f.FRAMES:
        labs, meds = grids(recs, frame)
        lines.append(f"\n==== frame {frame} (c = 2, stored strong centres) ====")
        for stat in STATS:
            lines += f.table(stat, labs[stat], f.STEPS)
        lines.append("\nmedians S_c4 / S_c|f / S_c~f / S_c|sp / S_c~sp / S_c|cls / S_c|all (passage median of band values)")
        for s in f.STEPS:
            cells = []
            for b in f.BANDS:
                v = [meds[k][b][s] for k in ("S_c4", "S_c|f", "S_c~f", "S_c|sp", "S_c~sp", "S_c|cls", "S_c|all")]
                cells.append(" ".join("   —  " if x is None else f"{x:+.2f}" for x in v))
            lines.append(f"{s:<7} " + " | ".join(f"{c:<48}" for c in cells))
        lines.append("\nreadings (cells where S_c4 is labelled; x = f primary, sp and cls beside)")
        rd = {}
        for b in f.BANDS:
            for s in f.STEPS:
                r = {x: reading(labs, meds, b, s, x) for x in COVARIATES}
                if r["f"] is None:
                    continue
                rd[f"{s} {b}"] = {"S_c4": labs["S_c4"][b][s], **r}
                lines.append(f"  {s:<7} {b:<7} S_c4 {labs['S_c4'][b][s]:<34} f: {r['f']:<9} sp: {r['sp']:<9} cls: {r['cls']}")
        out_json[frame] = {"labels": labs, "medians": meds, "readings": rd}
    (a.out / "summary.json").write_text(json.dumps(out_json) + "\n")
    (a.out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--f13", type=Path, default=F13_OUT)
    r.add_argument("--counts", type=Path, default=COUNTS)
    r.add_argument("--labels", type=Path, default=DATA / "p10" / "reread_r9_2026-10-08" / "labels")
    r.add_argument("--workers", type=int, default=14)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
