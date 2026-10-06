"""R6w: §1.9–§1.10's drift within lineages or by replacement (`p10_cluster_function/design-10.md` "R6w").

Per column (c3, c2a, c0 of R0's label source) and span (512 → 143000, 64 → 512), every focal token
of R1's readers (a member unique token at position > 0) carries its per-token lifts (§1.9's five,
§1.10's `emb_given_class` and CGE40 − kNN40), its R1 aggregation weight, and its group's kind
from R6's matcher at each boundary it borders (stable / restructured / death or birth; on c3 a
birth or death by its c2a fate: flip kept, flip restructured, new / gone). Each boundary's change
in the weighted mean splits exactly into the persisting groups' change (within) and one term per
non-stable kind against them (Melitz–Polanec's form); the span sums the boundaries. Four first
checks gate the output: R1's per-record values reproduce, the terms sum to the total, identical
stable links leave the composition lifts unchanged, and the matcher's stable counts are R6's.
Tier 1: exploratory, unregistered, descriptive. Run (METS_DATA set, default BLAS threads as R1):
    python tools/run/p10_r6w_drift.py --labels <R0 labels> --r1 <R1 dir> --r6 <r6.json> --out <file>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

COLUMNS = ("c3", "c2a", "c0")                 # c3 primary; c2a (no filter), c0 (published) beside
SPANS = {"512-143000": (512, 143000), "64-512": (64, 512)}
PRIMARY_SPAN = "512-143000"
COMPOSITION = ("same_class", "copy_share", "no_copy", "adjacent")
STATS = (*COMPOSITION, "emb_pct_own", "emb_given_class", "cge40_minus_knn40")
LEVELS = (12, 24, "mean")
FLOOR, WITHIN_HI, WITHIN_LO = 0.05, 2 / 3, 1 / 3   # placed (design-10 "R6w")
R1_TOL, SUM_TOL = 1e-6, 1e-9
# a term pairs the later step's entrants with the earlier step's exits of the same kind
TERMS = {"restructured": ("restructured", "restructured"),
         "flip_kept": ("birth_flip_kept", "death_flip_kept"),
         "flip_restructured": ("birth_flip_restructured", "death_flip_restructured"),
         "new_gone": ("birth_new", "death_gone"),
         # R1's own record set (after /challenge-pr on #152): a record readable at one end only
         "records": ("record_entry", "record_exit")}
REPLACEMENT = tuple(t for t in TERMS if t != "records")   # the model's kinds only


class DriftError(RuntimeError):
    pass


# ---------------------------------------------------------------- the decomposition (pure)

def side_summary(rows) -> dict:
    """{kind: (weight, weighted mean)} over (weight, value, kind) rows; '_all' the whole side."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for w, x, k in rows:
        for key in (k, "_all"):
            acc[key][0] += w
            acc[key][1] += w * x
    return {k: (W, s / W) for k, (W, s) in acc.items() if W > 0}


def decompose(earlier, later) -> dict:
    """
    M_t − M_s = (S_t − S_s) + Σ_k w_k,t (X_k,t − S_t) − Σ_k w_k,s (X_k,s − S_s): within (the
    stable tokens' mean) and one term per pair of TERMS. ``within`` is None when either side has no
    stable token (no reference); the total is always given.
    """
    a, b = side_summary(earlier), side_summary(later)
    if "_all" not in a or "_all" not in b:
        raise DriftError("a side with no focal token: the record set must have one at every step")
    out = {"total": b["_all"][1] - a["_all"][1]}
    if "stable" not in a or "stable" not in b:
        return out | {"within": None, **{t: None for t in TERMS}}
    S_s, S_t, W_s, W_t = a["stable"][1], b["stable"][1], a["_all"][0], b["_all"][0]
    out["within"] = S_t - S_s
    for term, (kb, ka) in TERMS.items():
        v = 0.0
        if kb in b:
            v += b[kb][0] / W_t * (b[kb][1] - S_t)
        if ka in a:
            v -= a[ka][0] / W_s * (a[ka][1] - S_s)
        out[term] = v
    rest = sum(k not in ("stable", "_all", *(x for pair in TERMS.values() for x in pair)) for k in (*a, *b))
    if rest:
        raise DriftError(f"kinds outside the decomposition: {sorted(set(a) | set(b))}")
    return out


def label(total: float, within) -> str:
    if abs(total) < FLOOR:
        return "no drift"
    if within is None:
        return "unlabelled (a boundary without stable tokens)"
    share = within / total
    return "within" if share >= WITHIN_HI else "replacement" if share <= WITHIN_LO else "both"


# ---------------------------------------------------------------- kinds from R6's matcher

def kinds(col_link: dict, c2a_link: dict = None) -> tuple:
    """({earlier group: kind}, {later group: kind}); births and deaths split by c2a's fate if given."""
    def fate(g, k, side, ref):
        if k == "stable":
            return "stable"
        if k != side:
            return "restructured"
        if ref is None:
            return "birth_new" if side == "birth" else "death_gone"
        if g not in ref:
            raise DriftError(f"group {g} is not in c2a; c3 ids must be c2a's")
        r = ref[g]
        none = "birth_new" if side == "birth" else "death_gone"
        return none if r == side else f"{side}_flip_kept" if r == "stable" else f"{side}_flip_restructured"
    ka = {g: fate(g, k, "death", c2a_link and c2a_link["kind_a"]) for g, k in col_link["kind_a"].items()}
    kb = {g: fate(g, k, "birth", c2a_link and c2a_link["kind_b"]) for g, k in col_link["kind_b"].items()}
    return ka, kb


# ---------------------------------------------------------------- per-token values (R1's readers)

def focal_rows(run_dir: Path, labels: dict, vocab: dict, added: set, own_gram=None) -> dict:
    """{layer: [(position, group id, {stat: lift})]} for a column's readable layers at one run."""
    from tools.run.p10_comembership import focal_stats
    from tools.run.p10_ext_sem_threshold import layer0_gram, read_tokens
    from tools.run.p10_lexical_carry import control_stats, split_stats
    from tools.run.p10_token_composition import OUTSIDE, token_features
    tokens = read_tokens(run_dir)
    if own_gram is None:
        own_gram = layer0_gram(run_dir)
    feats = token_features(tokens, vocab, added)
    is_copy = np.array([f["copies"] != "unique" for f in feats])
    cls = np.array([f["cls"] for f in feats], dtype=object)
    unique_pos = [p for p in range(1, len(tokens)) if not is_copy[p]]
    out = {}
    for layer, lab in sorted(labels.items()):
        if len(lab) != len(tokens):
            raise DriftError(f"{run_dir.name} layer {layer}: {len(lab)} labels, {len(tokens)} tokens")
        domain = np.flatnonzero(lab != OUTSIDE)
        rows = []
        for f in (f for f in unique_pos if lab[f] >= 0):
            members = np.flatnonzero(lab == lab[f])
            co, pool = members[members != f], domain[domain != f]
            cm = focal_stats(f, members, pool, is_copy, cls, {"emb_pct_own": own_gram[f]})
            sp = split_stats(f, co, pool, own_gram[f], cls)
            ct = control_stats(f, len(co), pool, own_gram[f], cls)
            val = {p: cm[p][0] - cm[p][1] for p in (*COMPOSITION, "emb_pct_own")}
            val["emb_given_class"] = sp["emb_given_class"]
            val["cge40_minus_knn40"] = sp["class_given_emb_40"] - ct["class_given_emb_40_knn"]
            rows.append((int(f), int(lab[f]), val))
        out[layer] = rows
    return out


def _job(args):
    """One (step, prompt): every column's readable layers, one Gram load."""
    from tools.run.p10_ext_sem_threshold import layer0_gram
    step, prompt, run_dir, by_col, vocab, added = args
    gram = layer0_gram(Path(run_dir))
    return (step, prompt), {c: focal_rows(Path(run_dir), lab, vocab, added, gram) for c, lab in by_col.items()}


# ---------------------------------------------------------------- checks against R1 and R6

def r1_record(r1: dict, c: str, step: int, prompt: str, layer: int) -> dict:
    """R1's stored per-record values for the seven statistics."""
    cm = r1[c]["cm"]["runs"][f"{step}|{prompt}"][str(layer)]
    lc = r1[c]["lc"]["runs"][f"{step}|{prompt}"][str(layer)]
    out = {p: cm[p][2] for p in (*COMPOSITION, "emb_pct_own")}
    out["emb_given_class"] = lc["emb_given_class"]
    out["cge40_minus_knn40"] = lc["class_given_emb_40"] - lc["class_given_emb_40_knn"]
    return out


def r1_level(r1: dict, c: str, step: int, level) -> dict:
    """R1's pooled value at a level (all readable records at that step), per statistic."""
    cm = r1[c]["cm"]["summary"]["all"][str(step)][str(level)]
    lc = r1[c]["lc"]["summary"][str(step)][str(level)]
    out = {p: cm[p]["lift"] for p in (*COMPOSITION, "emb_pct_own")}
    out["emb_given_class"] = lc["emb_given_class"]
    out["cge40_minus_knn40"] = (None if lc["class_given_emb_40"] is None or lc["class_given_emb_40_knn"] is None
                                else lc["class_given_emb_40"] - lc["class_given_emb_40_knn"])
    return out


# ---------------------------------------------------------------- reading one span

def span_records(readable: dict, values: dict, steps: list) -> list:
    """(prompt, layer) records readable with ≥ 1 focal token at every step of the span."""
    keys = None
    for s in steps:
        here = {(p, L) for (st, p), layers in values.items() if st == s
                for L, rows in layers.items() if rows and L in readable.get((s, p), {})}
        keys = here if keys is None else keys & here
    return sorted(keys)


def weighted(records: list, values: dict, step: int, level, prompt=None) -> list:
    """[(weight, record key, row)] at one step with R1's aggregation weights for `level`."""
    recs = [(p, L) for p, L in records if (prompt is None or p == prompt) and (level == "mean" or L == level)]
    by_layer = defaultdict(list)
    for p, L in recs:
        by_layer[L].append(p)
    n_layers = len(by_layer)
    out = []
    for L, prompts in by_layer.items():
        for p in prompts:
            rows = values[(step, p)][L]
            w = 1.0 / (n_layers * len(prompts) * len(rows))
            out += [(w, (p, L), r) for r in rows]
    return out


def read_span(c: str, steps: list, recs: dict, values: dict, link_kinds: dict, prompts: list) -> dict:
    """
    Per level and statistic: per-boundary terms, the span's sums, the label; per prompt beside.
    ``recs`` {step: records}: one fixed set at every step (the rule), or R1's own set per step
    (after `/challenge-pr` on #152), where a record readable at one end only is its own kind.
    """
    def tagged(step, other, side, level, stat, prompt):
        there = set(recs[other])
        s, t = (step, other) if side == 0 else (other, step)
        return [(w, r[2][stat], link_kinds[(key, s, t)][side][r[1]] if key in there
                 else ("record_exit" if side == 0 else "record_entry"))
                for w, key, r in weighted(recs[step], values, step, level, prompt)]

    def boundary(level, stat, j, prompt=None):
        s, t = steps[j], steps[j + 1]
        d = decompose(tagged(s, t, 0, level, stat, prompt), tagged(t, s, 1, level, stat, prompt))
        parts = [d["within"], *(d[k] for k in TERMS)]
        if d["within"] is not None and abs(sum(parts) - d["total"]) > SUM_TOL:
            raise DriftError(f"{c} {level} {stat} {s}→{t}: terms sum to {sum(parts)}, total {d['total']}")
        return d

    def summed(ds):
        out = {"total": sum(d["total"] for d in ds)}
        for k in ("within", *TERMS):
            out[k] = None if any(d[k] is None for d in ds) else sum(d[k] for d in ds)
        return out

    res = {}
    for level in LEVELS:
        if level != "mean" and not all(any(L == level for _, L in recs[s]) for s in steps):
            res[str(level)] = None          # a step with no record at this layer: counted, not read
            continue
        res[str(level)] = {}
        for stat in STATS:
            per = [boundary(level, stat, j) for j in range(len(steps) - 1)]
            tot = summed(per)
            lab = label(tot["total"], tot["within"])
            repl = {k: tot[k] for k in REPLACEMENT if tot[k] is not None}
            rest = None if tot["within"] is None else tot["total"] - tot["within"]
            entry = {"boundaries": [{"from": steps[j], "to": steps[j + 1], **d} for j, d in enumerate(per)],
                     "span": tot, "label": lab,
                     "within_share": (tot["within"] / tot["total"]
                                      if tot["within"] is not None and tot["total"] else None),
                     "largest_replacement": max(repl, key=lambda k: abs(repl[k])) if repl else None,
                     # within and the rest both past the floor with opposite signs (/challenge-pr on #152)
                     "opposing": (rest is not None and abs(tot["within"]) >= FLOOR and abs(rest) >= FLOOR
                                  and np.sign(tot["within"]) != np.sign(rest))}
            if level == "mean":
                pp = {}
                for p in prompts:
                    if not all(any(q == p for q, _ in recs[s]) for s in steps):
                        continue
                    pt = summed([boundary(level, stat, j, p) for j in range(len(steps) - 1)])
                    pp[p] = {"total": pt["total"], "within": pt["within"]}
                big = {p: v for p, v in pp.items() if abs(v["total"]) >= FLOOR and v["within"] is not None}
                entry["prompts"] = pp
                entry["prompts_within_same_sign"] = [int(sum(np.sign(v["within"]) == np.sign(v["total"])
                                                             for v in big.values())), len(big)]
            res[str(level)][stat] = entry
    return res


def paired_links(steps: list, records: list, values: dict, stable_links: dict) -> dict:
    """
    Each stable link's paired change in its group's focal mean, identical (J = 1) vs changed, pooled
    over the span and, after `/challenge-pr` on #152, per boundary (the drift is not spread evenly).
    """
    def summary(d):
        return {"n_links": len(d[STATS[0]]),
                **{st: {"mean": float(np.mean(x)) if x else None,
                        "max_abs": float(np.max(np.abs(x))) if x else None} for st, x in d.items()}}

    acc = {kind: {s: [] for s in STATS} for kind in ("identical", "changed")}
    per = []
    for j in range(len(steps) - 1):
        s, t = steps[j], steps[j + 1]
        here = {kind: {st: [] for st in STATS} for kind in acc}
        for p, L in records:
            by_s, by_t = defaultdict(list), defaultdict(list)
            for _, g, v in values[(s, p)][L]:
                by_s[g].append(v)
            for _, g, v in values[(t, p)][L]:
                by_t[g].append(v)
            for gb, (ga, jac) in stable_links[((p, L), s, t)].items():
                if ga in by_s and gb in by_t:
                    kind = "identical" if jac == 1.0 else "changed"
                    for st in STATS:
                        x = np.mean([v[st] for v in by_t[gb]]) - np.mean([v[st] for v in by_s[ga]])
                        acc[kind][st].append(x)
                        here[kind][st].append(x)
        per.append({"from": s, "to": t, **{kind: summary(d) for kind, d in here.items()}})
    return {**{kind: summary(d) for kind, d in acc.items()}, "boundaries": per}


# ---------------------------------------------------------------- main

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


def setup(labels: Path, r6: dict, spans: dict) -> tuple:
    """
    Steps per span, each column's readable records and runs at those steps, and every record's
    link kinds and stable links per adjacent boundary; ``check_d`` sets the stable counts beside
    R6's (the caller refuses on a mismatch).
    """
    from tools.run.p10_label_source import reader_input
    from tools.run.p10_r6_matcher import link, load
    full, axis = load(labels, columns=COLUMNS)           # every record, domain labels (R6's input)
    sel = {name: [s for s in axis if a <= s <= b] for name, (a, b) in spans.items()}
    steps_needed = sorted({s for v in sel.values() for s in v})
    readable, runs = {}, {}
    for c in COLUMNS:
        src = reader_input(labels, c)
        readable[c] = {k: v for k, v in src["labels"].items() if k[0] in steps_needed}
        runs.update({k: v for k, v in src["runs"].items() if k[0] in steps_needed})

    link_kinds, stable_links, check_d = {c: {} for c in COLUMNS}, {c: {} for c in COLUMNS}, []
    bidx = {(b["from"], b["to"]): b for b in r6["boundaries"]}
    for s, t in zip(steps_needed, steps_needed[1:]):
        if axis.index(t) != axis.index(s) + 1:
            continue
        for c in COLUMNS:
            n_stable = 0
            for key, by_col in full.items():
                L = link(by_col[c][s], by_col[c][t])
                ref = link(by_col["c2a"][s], by_col["c2a"][t]) if c == "c3" else None
                link_kinds[c][(key, s, t)] = kinds(L, ref)
                stable_links[c][(key, s, t)] = L["stable"]
                n_stable += L["counts"]["stable"]
            check_d.append({"column": c, "from": s, "to": t, "ours": n_stable,
                            "r6": bidx[(s, t)][c]["null"]["observed"]})
    return sel, readable, runs, link_kinds, stable_links, check_d


def main(argv=None) -> int:
    from tools.run.p10_token_composition import find_tokenizer, load_vocab
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source")
    ap.add_argument("--r1", type=Path, required=True, help="R1's output dir (cm_<c>.json, lc_<c>.json)")
    ap.add_argument("--r6", type=Path, required=True, help="R6's r6.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(DATA / "hf")))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    summary = args.labels / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no {summary}: the input would be unnamed")
    tok_path = find_tokenizer(Path(args.hf_home))
    vocab, added = load_vocab(tok_path)
    r1 = {c: {r: json.loads((args.r1 / f"{r}_{c}.json").read_text()) for r in ("cm", "lc")} for c in COLUMNS}
    r6 = json.loads(args.r6.read_text())
    span_steps = sorted({s for a, b in SPANS.values() for s in (a, b)})

    # links per (record, boundary) for every column; check (d) against R6
    sel, readable, runs, link_kinds, stable_links, check_d = setup(args.labels, r6, SPANS)
    bad_d = [d for d in check_d if d["ours"] != d["r6"]]
    if bad_d:
        d = bad_d[0]
        print(f"check (d) fails: {d['column']} {d['from']}→{d['to']} stable {d['ours']}, R6 {d['r6']}; "
              "refusing", file=sys.stderr)
        return 2

    jobs = [(s, p, str(runs[(s, p)]), {c: readable[c].get((s, p), {}) for c in COLUMNS}, vocab, added)
            for s, p in sorted(runs)]
    with ProcessPoolExecutor(args.workers) as ex:
        got = dict(ex.map(_job, jobs))
    values = {c: {k: v[c] for k, v in got.items()} for c in COLUMNS}

    # check (a): every per-record mean reproduces R1's stored record
    worst = 0.0
    for c in COLUMNS:
        for (s, p), layers in values[c].items():
            for L, rows in layers.items():
                if not rows:
                    continue
                want = r1_record(r1, c, s, p, L)
                for st in STATS:
                    worst = max(worst, abs(float(np.mean([r[2][st] for r in rows])) - want[st]))
    if worst > R1_TOL:
        print(f"check (a) fails: per-record values differ from R1 by up to {worst:.2e}; refusing", file=sys.stderr)
        return 2

    prompts = sorted({p for _, p in runs})
    out = {"checks": {"a_r1_max_abs": worst, "a_tol": R1_TOL, "b_sum_tol": SUM_TOL, "d_r6_stable": check_d,
                      "e_r1_set_equals_r1_change_tol": SUM_TOL},
           "columns": {}}
    for c in COLUMNS:
        out["columns"][c] = {}
        for name, steps in sel.items():
            recs = span_records(readable[c], values[c], steps)
            paired = paired_links(steps, recs, values[c], stable_links[c])
            bad = [st for st in COMPOSITION if paired["identical"][st]["max_abs"] not in (None, 0.0)]
            if bad:
                print(f"check (c) fails: {c} {name}: identical links move {bad}; refusing", file=sys.stderr)
                return 2
            res = read_span(c, steps, {s: recs for s in steps}, values[c], link_kinds[c], prompts)
            # beside, after /challenge-pr on #152: R1's own records at each step (readable, ≥ 1 focal
            # token); records at one end only are the "records" term, so the span sums to R1's change
            own = {s: sorted((p, L) for (st, p), layers in values[c].items() if st == s
                             for L, rows in layers.items() if rows) for s in steps}
            res_r1 = read_span(c, steps, own, values[c], link_kinds[c], prompts)
            a, b = steps[0], steps[-1]
            for level in LEVELS:
                va, vb = r1_level(r1, c, a, level), r1_level(r1, c, b, level)
                for st in STATS:
                    ch = None if va[st] is None or vb[st] is None else vb[st] - va[st]
                    if res[str(level)] is not None:
                        e = res[str(level)][st]
                        e["r1_change"] = ch
                        e["r1_label_differs"] = (ch is not None and
                                                 (abs(ch) >= FLOOR) != (abs(e["span"]["total"]) >= FLOOR))
                    if res_r1[str(level)] is not None and ch is not None:
                        gap = abs(res_r1[str(level)][st]["span"]["total"] - ch)
                        if gap > SUM_TOL:
                            print(f"check (e) fails: {c} {name} L{level} {st}: R1-set total differs from R1's "
                                  f"change by {gap:.1e}; refusing", file=sys.stderr)
                            return 2
                        if res[str(level)] is not None:
                            res_r1[str(level)][st]["label_differs_from_fixed"] = (
                                res_r1[str(level)][st]["label"] != res[str(level)][st]["label"])
            out["columns"][c][name] = {"steps": steps, "n_records": len(recs),
                                       "records_per_level": {str(L): sum(1 for _, x in recs if x == L) for L in (12, 24)},
                                       "r1_set_records": {str(s): len(v) for s, v in own.items()},
                                       "paired_links": paired, "levels": res, "levels_r1_set": res_r1}
    out["meta"] = {"labels": str(args.labels), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                   "r1": {f"{r}_{c}.json": _md5(args.r1 / f"{r}_{c}.json") for c in COLUMNS for r in ("cm", "lc")},
                   "r6": {args.r6.name: _md5(args.r6)},
                   "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()[:12],
                   "columns": list(COLUMNS), "spans": SPANS, "stats": list(STATS), "levels": list(map(str, LEVELS)),
                   "floor": FLOOR, "within_bounds": [WITHIN_LO, WITHIN_HI], "prompts": prompts,
                   "python": sys.version.split()[0], "numpy": np.__version__,
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
                   # uncommitted changes to this runner at run time (/challenge-pr on #152, finding 4)
                   "runner_dirty": bool(subprocess.run(
                       ["git", "-C", str(REPO), "status", "--porcelain", "--", "tools/run/p10_r6w_drift.py"],
                       capture_output=True, text=True).stdout.strip())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}; check (a) max |Δ| vs R1 {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
