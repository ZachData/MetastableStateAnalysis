"""R6f: R6w's within split into members against frame (`p10_cluster_function/design-10.md` "R6f").

R6w measured each step's groups in that step's own layer 0, so its "within" mixes the groups'
members changing with the embedding moving under them. Here every step's groups are scored again
in **one fixed layer 0** (step 512's, then step 143000's; same prompt, same tokens), with R6w's
members, pools, weights and kinds unchanged. Every R6w term is linear in the per-token values, so
each splits exactly: **members** = the term in the fixed frame, **frame** = the own-frame term
minus it. Read on R6w's three embedding statistics, span 512 → 143000, c3 (c2a, c0 beside), the
fixed record set (R1's own records beside). Five first checks gate the output: R1's per-record
values reproduce (own frame), R1's frozen-frame ``emb_pct`` reproduces (frame 143000), each
frame's own step is unchanged, identical stable links do not move in a fixed frame, and the
own-frame span terms are `r6w.json`'s.
Tier 1: exploratory, unregistered, descriptive. Run (METS_DATA set, default BLAS threads as R1):
    python tools/run/p10_r6f_frame.py --labels <R0 labels> --r1 <R1 dir> --r6 <r6.json> \
        --r6w <r6w.json> --out <file>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_r6w_drift as w

FRAMES = (512, 143000)                        # the span's two ends (design-10 "R6f")
SPAN = {"512-143000": (512, 143000)}          # R6w's primary span; its 64 → 512 is not readable on c3
EMB = ("emb_pct_own", "emb_given_class", "cge40_minus_knn40")   # the statistics with a frame
TERM_KEYS = ("total", "within", *w.TERMS)
MEMBERS_HI, MEMBERS_LO = 2 / 3, 1 / 3        # placed, as R6w's
R6W_TOL = 1e-9


class FrameError(RuntimeError):
    pass


# ---------------------------------------------------------------- the split (pure)

def split(own: dict, fixed: dict) -> dict:
    """{term: {"own", "members", "frame"}}: members = the fixed-frame term, frame = own − members."""
    out = {}
    for k in TERM_KEYS:
        a, b = own.get(k), fixed.get(k)
        out[k] = ({"own": a, "members": None, "frame": None} if a is None or b is None
                  else {"own": a, "members": b, "frame": a - b})
    return out


def share_label(within, within_fixed) -> tuple:
    """(members share, label) for one frame: members ≥ 2/3, frame ≤ 1/3, both between."""
    if within is None or within_fixed is None:
        return None, None
    if abs(within) < w.FLOOR:
        return None, "no drift"
    share = within_fixed / within
    return share, "members" if share >= MEMBERS_HI else "frame" if share <= MEMBERS_LO else "both"


def row_label(labels) -> str:
    """The frames' common label, else 'frame-dependent'."""
    labels = list(labels)
    return labels[0] if len(set(labels)) == 1 else "frame-dependent"


# ---------------------------------------------------------------- per-token values in each frame

def _job(args):
    """One (step, prompt): every column's readable layers in the own frame and each fixed frame."""
    from tools.run.p10_ext_sem_threshold import layer0_gram, read_tokens
    step, prompt, run_dir, by_col, vocab, added, frame_dirs = args
    run_dir, tokens = Path(run_dir), read_tokens(Path(run_dir))
    out = {"own": {c: w.focal_rows(run_dir, lab, vocab, added, layer0_gram(run_dir)) for c, lab in by_col.items()}}
    for F, d in frame_dirs.items():
        ft = read_tokens(Path(d))
        if len(ft) != len(tokens) or (ft != tokens).any():
            raise FrameError(f"{prompt}: step {step}'s tokens differ from frame {F}'s run")
        gram = layer0_gram(Path(d))
        out[F] = {c: w.focal_rows(run_dir, lab, vocab, added, gram) for c, lab in by_col.items()}
    return (step, prompt), out


def _record_mean(rows, stat) -> float:
    return float(np.mean([r[2][stat] for r in rows]))


# ---------------------------------------------------------------- reading

def read_split(res_own: dict, res_fixed: dict, prompts_floor=w.FLOOR) -> dict:
    """Per statistic at the layer mean: the span's and each boundary's split per frame, labels."""
    out = {}
    for stat in EMB:
        own = res_own["mean"][stat]
        e = {"span": {}, "boundaries": {}, "share": {}, "label": {}, "prompts": {}}
        for F, res in res_fixed.items():
            fx = res["mean"][stat]
            e["span"][F] = split(own["span"], fx["span"])
            e["boundaries"][F] = [{"from": a["from"], "to": a["to"], **split(a, b)}
                                  for a, b in zip(own["boundaries"], fx["boundaries"])]
            e["share"][F], e["label"][F] = share_label(own["span"]["within"], fx["span"]["within"])
            # per prompt: the label among prompts whose own within passes the floor
            pl = {p: share_label(v["within"], fx["prompts"][p]["within"])[1]
                  for p, v in own["prompts"].items()
                  if v["within"] is not None and abs(v["within"]) >= prompts_floor}
            e["prompts"][F] = {"labels": pl, "counts": {k: list(pl.values()).count(k)
                                                       for k in ("members", "both", "frame")}}
        shares = [e["share"][F] for F in res_fixed]
        e["row_label"] = row_label(e["label"][F] for F in res_fixed)
        e["shapley_share"] = None if None in shares else float(np.mean(shares))
        e["interaction"] = None if None in shares else shares[0] - shares[1]
        e["own_within"], e["own_total"], e["own_label"] = own["span"]["within"], own["span"]["total"], own["label"]
        out[stat] = e
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def main(argv=None) -> int:
    from tools.run.p10_token_composition import find_tokenizer, load_vocab
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source")
    ap.add_argument("--r1", type=Path, required=True, help="R1's output dir (cm_<c>.json, lc_<c>.json)")
    ap.add_argument("--r6", type=Path, required=True, help="R6's r6.json")
    ap.add_argument("--r6w", type=Path, required=True, help="R6w's r6w.json (check (e))")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    summary = args.labels / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no {summary}: the input would be unnamed")
    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    cols = w.COLUMNS
    r1 = {c: {r: json.loads((args.r1 / f"{r}_{c}.json").read_text()) for r in ("cm", "lc")} for c in cols}
    r6 = json.loads(args.r6.read_text())
    r6w = json.loads(args.r6w.read_text())

    sel, readable, runs, link_kinds, stable_links, check_d = w.setup(args.labels, r6, SPAN)
    bad_d = [d for d in check_d if d["ours"] != d["r6"]]
    if bad_d:
        print(f"check (d) fails (R6's stable counts): {bad_d[0]}; refusing", file=sys.stderr)
        return 2
    (name, steps), = sel.items()
    missing = [(F, p) for F in FRAMES for _, p in runs if (F, p) not in runs]
    if missing:
        raise SystemExit(f"no run for frame {missing[0]}")
    jobs = [(s, p, str(runs[(s, p)]), {c: readable[c].get((s, p), {}) for c in cols}, vocab, added,
             {F: str(runs[(F, p)]) for F in FRAMES}) for s, p in sorted(runs)]
    with ProcessPoolExecutor(args.workers) as ex:
        got = dict(ex.map(_job, jobs))
    values = {fr: {c: {k: v[fr][c] for k, v in got.items()} for c in cols} for fr in ("own", *FRAMES)}

    # check (a): own frame reproduces R1's records; (a′): frame 143000 reproduces R1's frozen emb_pct;
    # (b): each frame's own step is the own frame exactly
    worst_a, worst_a2, bad_b = 0.0, 0.0, []
    for c in cols:
        for (s, p), layers in values["own"][c].items():
            for L, rows in layers.items():
                if not rows:
                    continue
                want = w.r1_record(r1, c, s, p, L)
                worst_a = max(worst_a, *(abs(_record_mean(rows, st) - want[st]) for st in w.STATS))
                frozen = r1[c]["cm"]["runs"][f"{s}|{p}"][str(L)]["emb_pct"][2]
                worst_a2 = max(worst_a2, abs(_record_mean(values[143000][c][(s, p)][L], "emb_pct_own") - frozen))
                if s in FRAMES and values[s][c][(s, p)][L] != rows:
                    bad_b.append((c, s, p, L))
    if worst_a > w.R1_TOL or worst_a2 > w.R1_TOL or bad_b:
        print(f"check (a) {worst_a:.1e}, (a′) {worst_a2:.1e}, (b) {bad_b[:3]}; refusing", file=sys.stderr)
        return 2

    prompts = sorted({p for _, p in runs})
    out = {"checks": {"a_r1_max_abs": worst_a, "a2_frozen_emb_pct_max_abs": worst_a2, "a_tol": w.R1_TOL,
                      "b_frame_step_unchanged": True, "d_r6_stable": check_d, "e_r6w_tol": R6W_TOL},
           "columns": {}}
    for c in cols:
        recs = w.span_records(readable[c], values["own"][c], steps)
        own_sets = {s: sorted((p, L) for (st, p), layers in values["own"][c].items() if st == s
                              for L, rows in layers.items() if rows) for s in steps}
        entry = {"steps": steps, "n_records": len(recs)}
        for set_name, rs, r6w_key in (("fixed_set", {s: recs for s in steps}, "levels"),
                                      ("r1_set", own_sets, "levels_r1_set")):
            res = {fr: w.read_span(c, steps, rs, values[fr][c], link_kinds[c], prompts) for fr in ("own", *FRAMES)}
            # check (e): the own-frame span terms are r6w.json's
            ref = r6w["columns"][c][name][r6w_key]["mean"]
            gap = max(abs(res["own"]["mean"][st]["span"][k] - ref[st]["span"][k])
                      for st in EMB for k in TERM_KEYS if res["own"]["mean"][st]["span"][k] is not None)
            if gap > R6W_TOL:
                print(f"check (e) fails: {c} {set_name}: own-frame terms differ from r6w.json by {gap:.1e}; "
                      "refusing", file=sys.stderr)
                return 2
            entry[set_name] = read_split(res["own"], {F: res[F] for F in FRAMES})
        # check (c): identical stable links do not move in a fixed frame (fixed set)
        entry["paired_links"] = {}
        for F in FRAMES:
            pl = w.paired_links(steps, recs, values[F][c], stable_links[c])
            moved = [st for st in EMB if pl["identical"][st]["max_abs"] not in (None, 0.0)]
            if moved:
                print(f"check (c) fails: {c} frame {F}: identical links move {moved}; refusing", file=sys.stderr)
                return 2
            entry["paired_links"][F] = pl
        out["columns"][c] = entry
    out["meta"] = {"labels": str(args.labels), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                   "r1": {f"{r}_{c}.json": _md5(args.r1 / f"{r}_{c}.json") for c in cols for r in ("cm", "lc")},
                   "r6": {args.r6.name: _md5(args.r6)}, "r6w": {args.r6w.name: _md5(args.r6w)},
                   "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
                   "columns": list(cols), "span": SPAN, "frames": list(FRAMES), "stats": list(EMB),
                   "floor": w.FLOOR, "members_bounds": [MEMBERS_LO, MEMBERS_HI], "prompts": prompts,
                   "python": sys.version.split()[0], "numpy": np.__version__,
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
                   "runner_dirty": bool(subprocess.run(
                       ["git", "-C", str(REPO), "status", "--porcelain", "--",
                        "tools/run/p10_r6f_frame.py", "tools/run/p10_r6w_drift.py"],
                       capture_output=True, text=True).stdout.strip())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}; check (a) {worst_a:.1e}, (a′) {worst_a2:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
