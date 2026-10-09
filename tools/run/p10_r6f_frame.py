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
        --r6w <r6w.json> --out <file> [--lead c3x --reproduce <R6f r6f.json> --reproduce-labels <R0 labels>]

``--lead c3x`` (R9 / R6f, `design-10.md` "R9 / R6f"): c3x joins the columns (births and deaths by
c2a's fate, as c3); ``c3x_vs_c3`` lists the cells whose label differs from c3's (the 6 row labels
are the headline's), and ``reference`` holds the random-drop reference: ``--draws`` columns that
drop, in every (step, passage, layer) of the span, as many c3 groups as c3x drops there, at random,
read through the same split and compared with c3 the same way. ``--reproduce``: refuse unless c3,
c2a and c0 equal that R6f run's (`p10_r9_lead`).
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
from tools.run import p10_r9_lead as lead_args

FRAMES = (512, 143000)                        # the span's two ends (design-10 "R6f")
SPAN = {"512-143000": (512, 143000)}          # R6w's primary span; its 64 → 512 is not readable on c3
EMB = ("emb_pct_own", "emb_given_class", "cge40_minus_knn40")   # the statistics with a frame
TERM_KEYS = ("total", "within", *w.TERMS)
MEMBERS_HI, MEMBERS_LO = 2 / 3, 1 / 3        # placed, as R6w's
R6W_TOL = 1e-9
RECORD_SETS = (("fixed_set", "levels"), ("r1_set", "levels_r1_set"))   # (R6f's name, R6w's key)
N_DRAWS, QUANTILE = 100, 0.95                 # the random-drop reference (design-10 "R9 / R6f"), placed


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
        # after /challenge-pr on #153: each prompt's values, its label on the two-frame mean (each fixed
        # frame favours its own step's groups), and whether members and frame oppose (both past the floor)
        pv = {}
        for p, v in own["prompts"].items():
            mem = {F: res["mean"][stat]["prompts"][p]["within"] for F, res in res_fixed.items()}
            if v["within"] is None or None in mem.values():
                continue
            m = float(np.mean(list(mem.values())))
            pv[p] = {"within": v["within"], "members": mem, "members_mean": m,
                     "label_mean": share_label(v["within"], m)[1],
                     "opposing": abs(m) >= w.FLOOR and abs(v["within"] - m) >= w.FLOOR
                                 and np.sign(m) != np.sign(v["within"] - m)}
        e["prompt_values"] = pv
        e["prompt_counts_mean"] = {k: sum(x["label_mean"] == k for x in pv.values())
                                   for k in ("members", "both", "frame", "no drift")}
        e["prompts_opposing"] = sorted(p for p, x in pv.items() if x["opposing"])
        shares = [e["share"][F] for F in res_fixed]
        e["row_label"] = row_label(e["label"][F] for F in res_fixed)
        e["shapley_share"] = None if None in shares else float(np.mean(shares))
        e["interaction"] = None if None in shares else shares[0] - shares[1]
        e["own_within"], e["own_total"], e["own_label"] = own["span"]["within"], own["span"]["total"], own["label"]
        out[stat] = e
    return out


# ---------------------------------------------------------------- one column, read (shared by the draws)

def read_column(c: str, steps: list, readable_c: dict, values_c: dict, link_kinds_c: dict,
                prompts: list) -> tuple:
    """
    ({"steps", "n_records", "fixed_set", "r1_set"}, {set name: the own-frame span result}) for one
    column: ``values_c`` {frame: {(step, prompt): {layer: rows}}}, frames "own" and FRAMES.
    """
    recs = w.span_records(readable_c, values_c["own"], steps)
    own_sets = {s: sorted((p, L) for (st, p), layers in values_c["own"].items() if st == s
                          for L, rows in layers.items() if rows) for s in steps}
    entry, own = {"steps": steps, "n_records": len(recs)}, {}
    for set_name, rs in (("fixed_set", {s: recs for s in steps}), ("r1_set", own_sets)):
        res = {fr: w.read_span(c, steps, rs, values_c[fr], link_kinds_c, prompts) for fr in ("own", *FRAMES)}
        own[set_name] = res["own"]
        entry[set_name] = read_split(res["own"], {F: res[F] for F in FRAMES})
    return entry, own


# ---------------------------------------------------------------- R9 / R6f: c3x beside c3 (pure)

def cells(entry: dict) -> dict:
    """{"set|stat|cell": label} of one column's (jsonable) entry: the row label, each frame's, and
    each prompt's on the two frames' mean (`design-10.md` "R9 / R6f"). ``None``: not held."""
    out = {}
    for set_name, _ in RECORD_SETS:
        for st in EMB:
            e = entry[set_name][st]
            out[f"{set_name}|{st}|row"] = e["row_label"]
            for F in FRAMES:
                out[f"{set_name}|{st}|frame {F}"] = e["label"][str(F)]
            for p, x in e["prompt_values"].items():
                out[f"{set_name}|{st}|prompt {p}"] = x["label_mean"]
    return out


def is_headline(key: str) -> bool:
    """§1.22's 6 cells: the row label on either record set."""
    return key.endswith("|row")


def compare_cells(ref: dict, col: dict) -> dict:
    """A cell is compared when either side holds a label, and changes when the two differ."""
    keys = sorted(set(ref) | set(col))
    compared = [k for k in keys if ref.get(k) is not None or col.get(k) is not None]
    changed = [k for k in compared if ref.get(k) != col.get(k)]
    return {"n_compared": len(compared), "n_changed": len(changed), "changed": changed,
            "headline_changed": [k for k in changed if is_headline(k)],
            "not_compared": [k for k in keys if k not in compared]}


def c3x_drops(dom: dict) -> dict:
    """{key: the c3 group ids c3x drops there}."""
    return {k: set(map(int, np.unique(d["c3"][d["c3"] >= 0]))) - set(map(int, np.unique(d["c3x"][d["c3x"] >= 0])))
            for k, d in dom.items()}


def drop_labels(dom: dict, rng=None, fixed: dict = None) -> tuple:
    """
    One draw over ``dom`` {(step, prompt, layer): {"c3", "c3x"}} (domain labels): in each record,
    as many c3 groups as c3x drops there, uniformly without replacement (``rng`` None drops none;
    ``fixed`` {key: ids} drops exactly those, check (r4)); dropped members → −1, as c3x's.
    ({key: labels}, {key: dropped ids}); keys in sorted order, so a seed fixes the draw.
    """
    labs, dropped = {}, {}
    for key in sorted(dom):
        c3, c3x = dom[key]["c3"], dom[key]["c3x"]
        ids = np.unique(c3[c3 >= 0])
        k = len(ids) - len(np.unique(c3x[c3x >= 0]))
        drop = (np.array(sorted(fixed[key]), dtype=int) if fixed is not None
                else rng.choice(ids, k, replace=False) if rng is not None and k else np.array([], dtype=int))
        labs[key] = np.where(np.isin(c3, drop), -1, c3)
        dropped[key] = set(map(int, drop))
    return labs, dropped


def dropped_sizes(dom: dict, dropped: dict) -> list:
    return [int((dom[key]["c3"] == g).sum()) for key in sorted(dropped) for g in sorted(dropped[key])]


def reference_reading(obs: int, counts: list) -> dict:
    """Within random drops if ``obs`` ≤ the draws' 95th percentile, else beyond; the rank p beside."""
    q = float(np.quantile(counts, QUANTILE, method="higher"))
    return {"obs": obs, "q95": q, "median": float(np.median(counts)),
            "reading": "within random drops" if obs <= q else "beyond random drops",
            "rank_p": (1 + sum(x >= obs for x in counts)) / (len(counts) + 1)}


# ---------------------------------------------------------------- the draws

_G: dict = {}   # set before the draws' pool forks: dom, c3 rows, c2a links, steps, prompts, c3's cells


def c3_domains(src: Path, steps: list) -> dict:
    """{(step, prompt, layer): {"c3", "c3x", "c2a" domain labels, "full" c3 over every position}}."""
    from tools.run.p10_label_source import LAYERS, OUTSIDE, load_column, load_step
    out = {}
    for s in steps:
        cache = {f"step{s}": load_step(src, f"step{s}")}
        for prompt, p in sorted(cache[f"step{s}"]["prompts"].items()):
            for L in LAYERS:
                cols = {c: load_column(src, f"step{s}", prompt, L, c, cache) for c in ("c3", "c3x", "c2a")}
                pos = cols["c3"][0]
                if any(not np.array_equal(cols[c][0], pos) for c in cols):
                    raise FrameError(f"step {s} {prompt} L{L}: c3, c3x and c2a domains differ")
                full = np.full(p["n_positions"], OUTSIDE, dtype=int)
                full[pos] = cols["c3"][1]
                out[(s, prompt, L)] = {**{c: v[1] for c, v in cols.items()}, "pos": pos, "full": full}
    return out


def draw_entry(seed, fixed: dict = None) -> tuple:
    """One draw (``seed`` None: drop nothing; ``fixed``: those ids) read as a column: (jsonable
    entry, dropped sizes)."""
    from tools.run.p10_label_source import readable as is_readable
    from tools.run.p10_r6_matcher import link
    dom, rows, c2a_links, steps = _G["dom"], _G["rows"], _G["c2a_links"], _G["steps"]
    labs, dropped = drop_labels(dom, None if seed is None else np.random.default_rng(seed), fixed)
    readable_d, values_d = {}, {fr: {} for fr in ("own", *FRAMES)}
    for (s, p, L), lab in labs.items():
        if not is_readable(lab):
            continue
        full = dom[(s, p, L)]["full"].copy()
        full[dom[(s, p, L)]["pos"]] = lab
        readable_d.setdefault((s, p), {})[L] = full
        for fr in values_d:
            values_d[fr].setdefault((s, p), {})[L] = [r for r in rows[fr][(s, p)][L]
                                                       if r[1] not in dropped[(s, p, L)]]
    for fr in values_d:      # the reader's shape: every (step, prompt) present
        for s, p, _ in labs:
            values_d[fr].setdefault((s, p), {})
    kinds_d = {}
    for (pl, s, t), ref in c2a_links.items():
        kinds_d[(pl, s, t)] = w.kinds(link(labs[(s, *pl)], labs[(t, *pl)]), ref)
    entry, _ = read_column("draw", steps, readable_d, values_d, kinds_d, _G["prompts"])
    return w.jsonable(entry), dropped_sizes(dom, dropped)


def _draw(seed):
    entry, sizes = draw_entry(seed)
    cmp = compare_cells(_G["c3_cells"], cells(entry))
    return {"seed": seed, "n_records": entry["n_records"], "n_compared": cmp["n_compared"],
            "n_changed": cmp["n_changed"], "n_headline_changed": len(cmp["headline_changed"]),
            "changed": cmp["changed"], "dropped_mean_size": float(np.mean(sizes)) if sizes else None}


def run_reference(n_draws: int, workers: int, c3x_cmp: dict, c3x_sizes: list, forced: dict) -> dict:
    """The random-drop reference (`design-10.md` "R9 / R6f"): (r3) the first draw is read alone,
    then the rest; each count against c3x's."""
    import multiprocessing
    first = _draw(0)
    if not first["n_records"] or not first["n_compared"]:
        raise FrameError(f"check (r3): the first draw has {first['n_records']} records, "
                         f"{first['n_compared']} compared cells; refusing")
    print(f"draw 0: {first['n_records']} records, {first['n_changed']} of {first['n_compared']} cells change")
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as ex:
        draws = [first, *ex.map(_draw, range(1, n_draws))]
    counts = [d["n_changed"] for d in draws]
    head = [d["n_headline_changed"] for d in draws]
    every = sorted({k for d in draws for k in d["changed"]} | set(c3x_cmp["changed"]))
    sizes = [d["dropped_mean_size"] for d in draws if d["dropped_mean_size"] is not None]
    # beside, after /challenge-pr on #174 (finding 1): where c3x empties a record every draw drops
    # the same groups, so the cells every draw changes are c3x's own; the count without them
    always = sorted(k for k in every if all(k in d["changed"] for d in draws))
    without = reference_reading(sum(k not in always for k in c3x_cmp["changed"]),
                                [sum(k not in always for k in d["changed"]) for d in draws])
    return {"forced": {**forced, "cells_every_draw_changes": always, "all_without_them": without},
            "n_draws": n_draws, "seeds": [0, n_draws - 1], "quantile": QUANTILE, "quantile_method": "higher",
            "all": reference_reading(c3x_cmp["n_changed"], counts),
            "headline": reference_reading(len(c3x_cmp["headline_changed"]), head),
            "cell_share": {k: sum(k in d["changed"] for d in draws) / n_draws for k in every},
            "dropped_mean_size": {"c3x": float(np.mean(c3x_sizes)) if c3x_sizes else None,
                                  "draws_median": float(np.median(sizes)) if sizes else None,
                                  "draws_5_95": [float(np.quantile(sizes, 0.05)), float(np.quantile(sizes, 0.95))]
                                  if sizes else None, "n_dropped": len(c3x_sizes)},
            "draws": [{k: d[k] for k in ("seed", "n_records", "n_compared", "n_changed", "n_headline_changed",
                                         "dropped_mean_size")} for d in draws]}


def reproduce(data: dict, other: dict) -> list:
    """Every column ``other`` holds, compared with ``data``'s entry key by key. The differing keys."""
    bad = []
    for c in other["meta"]["columns"]:
        mine = data["columns"].get(c)
        if mine is None:
            bad.append(f"{c}: missing")
            continue
        bad += [f"{c}/{k}" for k in other["columns"][c] if mine.get(k) != other["columns"][c][k]]
    return bad


def headline_rows(cols: dict) -> list:
    """c3x's within, members share per frame and Shapley share beside c3's, on the 6 row labels."""
    out = []
    for set_name, _ in RECORD_SETS:
        for st in EMB:
            out.append({"set": set_name, "stat": st, **{
                f"{c}_{k}": v for c in ("c3", "c3x") for k, v in (
                    ("label", cols[c][set_name][st]["row_label"]), ("within", cols[c][set_name][st]["own_within"]),
                    ("share", cols[c][set_name][st]["share"]), ("shapley", cols[c][set_name][st]["shapley_share"]))}})
    return out


# ---------------------------------------------------------------- main

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def main(argv=None) -> int:
    from tools.run.p10_r6_matcher import link
    from tools.run.p10_token_composition import find_tokenizer, load_vocab
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source")
    ap.add_argument("--r1", type=Path, required=True, help="R1's output dir (cm_<c>.json, lc_<c>.json)")
    ap.add_argument("--r6", type=Path, required=True, help="R6's r6.json")
    ap.add_argument("--r6w", type=Path, required=True, help="R6w's r6w.json (check (e))")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="random-drop draws (with --lead c3x)")
    lead_args.add_args(ap, "R6f")
    args = ap.parse_args(argv)
    lead_args.reproduce_inputs(args)  # refuse a half-given --reproduce before the run
    summary = args.labels / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no {summary}: the input would be unnamed")
    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    cols = w.columns_of(args.lead)
    reference = args.lead == "c3x" and args.draws > 0
    r1 = {c: {r: json.loads((args.r1 / f"{r}_{c}.json").read_text()) for r in ("cm", "lc")} for c in cols}
    r6 = json.loads(args.r6.read_text())
    r6w = json.loads(args.r6w.read_text())

    sel, readable, runs, link_kinds, stable_links, check_d = w.setup(args.labels, r6, SPAN, cols)
    bad_d = [d for d in check_d if d["ours"] != d["r6"]]
    if bad_d:
        print(f"check (d) fails (R6's stable counts): {bad_d[0]}; refusing", file=sys.stderr)
        return 2
    (name, steps), = sel.items()
    missing = [(F, p) for F in FRAMES for _, p in runs if (F, p) not in runs]
    if missing:
        raise SystemExit(f"no run for frame {missing[0]}")
    dom = c3_domains(args.labels, steps) if reference else {}
    jobs = []
    for s, p in sorted(runs):
        by_col = {c: readable[c].get((s, p), {}) for c in cols}
        if reference:   # every c3 record at the step, readable or not: a draw can make one readable
            by_col["c3all"] = {L: d["full"] for (st, q, L), d in dom.items() if (st, q) == (s, p)}
        jobs.append((s, p, str(runs[(s, p)]), by_col, vocab, added, {F: str(runs[(F, p)]) for F in FRAMES}))
    with ProcessPoolExecutor(args.workers) as ex:
        got = dict(ex.map(_job, jobs))
    values = {fr: {c: {k: v[fr][c] for k, v in got.items()} for c in got[next(iter(got))][fr]}
              for fr in ("own", *FRAMES)}
    if args.lead == "c3x":  # the lead opened and checked populated before any split is read
        n_first = len(w.span_records(readable["c3x"], values["own"]["c3x"], steps))
        if not n_first:
            raise SystemExit(f"refusing: c3x has no record with a focal token at every step of {name}")
        print(f"c3x over {name}: {n_first} records with a focal token at every step")

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
    c3x_sizes = []
    if reference:   # check (r1): c3's rows less c3x's dropped groups are c3x's own rows, every frame
        x_dropped = c3x_drops(dom)
        c3x_sizes = dropped_sizes(dom, x_dropped)
        forced = {"records_emptied": sum(1 for k, d in dom.items() if x_dropped[k] and not (d["c3x"] >= 0).any()),
                  "drops_forced": sum(len(x_dropped[k]) for k, d in dom.items() if not (d["c3x"] >= 0).any()),
                  "drops": len(c3x_sizes)}
        bad_r1 = [(fr, s, p, L) for fr in ("own", *FRAMES) for (s, p), layers in values[fr]["c3x"].items()
                  for L, rows in layers.items()
                  if rows != [r for r in values[fr]["c3all"][(s, p)][L] if r[1] not in x_dropped[(s, p, L)]]]
        if bad_r1:
            print(f"check (r1) fails: c3 less c3x's drops is not c3x at {bad_r1[:3]}; refusing", file=sys.stderr)
            return 2

    prompts = sorted({p for _, p in runs})
    out = {"checks": {"a_r1_max_abs": worst_a, "a2_frozen_emb_pct_max_abs": worst_a2, "a_tol": w.R1_TOL,
                      "b_frame_step_unchanged": True, "d_r6_stable": check_d, "e_r6w_tol": R6W_TOL},
           "columns": {}}
    for c in cols:
        entry, own = read_column(c, steps, readable[c], {fr: values[fr][c] for fr in values},
                                 link_kinds[c], prompts)
        for set_name, r6w_key in RECORD_SETS:   # check (e): the own-frame span terms are r6w.json's
            ref = r6w["columns"][c][name][r6w_key]["mean"]
            gap = max(abs(own[set_name]["mean"][st]["span"][k] - ref[st]["span"][k])
                      for st in EMB for k in TERM_KEYS if own[set_name]["mean"][st]["span"][k] is not None)
            if gap > R6W_TOL:
                print(f"check (e) fails: {c} {set_name}: own-frame terms differ from r6w.json by {gap:.1e}; "
                      "refusing", file=sys.stderr)
                return 2
        # check (c): identical stable links do not move in a fixed frame (fixed set)
        recs = w.span_records(readable[c], values["own"][c], steps)
        entry["paired_links"] = {}
        for F in FRAMES:
            pl = w.paired_links(steps, recs, values[F][c], stable_links[c])
            moved = [st for st in EMB if pl["identical"][st]["max_abs"] not in (None, 0.0)]
            if moved:
                print(f"check (c) fails: {c} frame {F}: identical links move {moved}; refusing", file=sys.stderr)
                return 2
            entry["paired_links"][F] = pl
        out["columns"][c] = entry
    if reference:
        out["checks"]["r1_c3_less_drops_is_c3x"] = True
    out = w.jsonable(out)
    reproduces = lead_args.check_reproduces(args, out, w.load_record, reproduce, "every column entry")
    if args.lead == "c3x":
        cmp = compare_cells(cells(out["columns"]["c3"]), cells(out["columns"]["c3x"]))
        out["c3x_vs_c3"] = {**cmp, "headline_cells": headline_rows(out["columns"])}
    if reference:
        c2a_links = {((p, L), s, t): link(dom[(s, p, L)]["c2a"], dom[(t, p, L)]["c2a"])
                     for (s, p, L) in dom for t in steps[steps.index(s) + 1:steps.index(s) + 2]}
        _G.update(dom=dom, rows={fr: values[fr]["c3all"] for fr in values}, c2a_links=c2a_links,
                  steps=steps, prompts=prompts, c3_cells=cells(out["columns"]["c3"]))
        zero, _ = draw_entry(None)    # check (r2): a draw that drops nothing is c3
        if {k: zero[k] for k in ("n_records", "fixed_set", "r1_set")} != \
                {k: out["columns"]["c3"][k] for k in ("n_records", "fixed_set", "r1_set")}:
            print("check (r2) fails: a draw dropping nothing is not c3; refusing", file=sys.stderr)
            return 2
        out["checks"]["r2_no_drop_is_c3"] = True
        own, _ = draw_entry(None, x_dropped)    # check (r4), after /challenge-pr on #174: c3x's drops are c3x
        if {k: own[k] for k in ("n_records", "fixed_set", "r1_set")} != \
                {k: out["columns"]["c3x"][k] for k in ("n_records", "fixed_set", "r1_set")}:
            print("check (r4) fails: c3x's own drops through the draw are not c3x; refusing", file=sys.stderr)
            return 2
        out["checks"]["r4_c3x_drops_are_c3x"] = True
        out["reference"] = run_reference(args.draws, args.workers, out["c3x_vs_c3"], c3x_sizes, forced)
    out["meta"] = {"labels": str(args.labels), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                   "lead": args.lead, "reproduces": reproduces,
                   "floor_step0": lead_args.floor(json.loads(summary.read_text()), args.lead),
                   "r1": {f"{r}_{c}.json": _md5(args.r1 / f"{r}_{c}.json") for c in cols for r in ("cm", "lc")},
                   "r6": {args.r6.name: _md5(args.r6)}, "r6w": {args.r6w.name: _md5(args.r6w)},
                   "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
                   "columns": list(cols), "span": SPAN, "frames": list(FRAMES), "stats": list(EMB),
                   "floor": w.FLOOR, "members_bounds": [MEMBERS_LO, MEMBERS_HI], "prompts": prompts,
                   "draws": args.draws if reference else 0,
                   "python": sys.version.split()[0], "numpy": np.__version__,
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
                   "runner_dirty": bool(subprocess.run(
                       ["git", "-C", str(REPO), "status", "--porcelain", "--",
                        "tools/run/p10_r6f_frame.py", "tools/run/p10_r6w_drift.py", "tools/run/p10_r9_lead.py"],
                       capture_output=True, text=True).stdout.strip())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    msg = f"wrote {args.out}; check (a) {worst_a:.1e}, (a′) {worst_a2:.1e}"
    if args.lead == "c3x":
        v = out["c3x_vs_c3"]
        msg += f"; c3x vs c3: {v['n_changed']} of {v['n_compared']} cells change, headline {len(v['headline_changed'])} of 6"
    if reference:
        a, h = out["reference"]["all"], out["reference"]["headline"]
        msg += (f"; random drops: all {a['reading']} (q95 {a['q95']:g}, p {a['rank_p']:.2f}), "
                f"headline {h['reading']} (q95 {h['q95']:g}, p {h['rank_p']:.2f})")
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
