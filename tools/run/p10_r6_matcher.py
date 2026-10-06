"""R6: the cross-checkpoint matcher (`p10_cluster_function/design-10.md` "R6"; `status-10.md` §1.20).

Per (prompt, layer L1–24) and column (c0, c2a, c2, c3 of R0's label source), each step's groups
are linked to the next step's by `merge_tree.link_layer_pair` at containment ≥ 0.5 (MONIC's
taxonomy: stable / split / merge / tangle / birth / death). Beside: each stable link's Jaccard;
for c2 and c3, whether a birth or death is a filter flip (the same group id is linked in c2a's
matching at that boundary) or new / gone; a permutation null on the stable count
(`merge_tree.link_chain_null`); and each 143000 group's lineage origin (the earliest step an
unbroken chain of stable links reaches). The first check (c2a at 0 → 2, stable share ≥ 0.9)
gates the rest. Tier 1: exploratory, unregistered. Run:
    python tools/run/p10_r6_matcher.py --labels <R0 labels dir> --out <file.json>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from p1d_cluster_ensemble.merge_tree import _KINDS, link_chain_null, link_layer_pair

COLUMNS = ("c0", "c1", "c1c", "c2a", "c2", "c3")  # c1, c1c: /challenge-pr on #151, finding 2
FLIP_COLUMNS = ("c2", "c3")       # their group ids are c2a's (`p10_label_source` docstring)
LAYERS = tuple(range(1, 25))
MEASURE, MIN_OVERLAP = "containment", 0.5
SAME_JACCARD = 0.5                # unit 1's fixed bar, `lit-10.md` §16 row 1
N_DRAWS, SEED = 200, 0
FIRST_CHECK = ("c2a", 0, 0.9)     # (column, boundary index 0 → 2, min stable share): placed
A_KINDS = ("stable", "split", "merge", "tangle", "death")


def link(lab_a, lab_b) -> dict:
    """One boundary: each earlier group's kind, each later group's kind, the stable links' Jaccard."""
    out = link_layer_pair(np.asarray(lab_a), np.asarray(lab_b), min_overlap=MIN_OVERLAP, measure=MEASURE)
    kind_a, kind_b, stable = {}, {}, {}
    edge = {(a, b): jac for a, b, jac, _ in out["edges"]}
    for comp in out["components"]:
        for a in comp["prev"]:
            kind_a[a] = comp["kind"]
        for b in comp["curr"]:
            kind_b[b] = comp["kind"]
        if comp["kind"] == "stable":
            a, b = comp["prev"][0], comp["curr"][0]
            stable[b] = (a, edge[(a, b)])
    return {"kind_a": kind_a, "kind_b": kind_b, "stable": stable, "counts": out["counts"]}


def flips(col: dict, c2a: dict) -> dict:
    """
    A column's births / deaths by the same id's fate in c2a's matching at that boundary: ``new`` /
    ``gone`` (no c2a link), ``flip_kept`` (c2a's component is 1–1: the group persisted and passed
    or failed a filter) or ``flip_restructured`` (c2a split, merged or tangled it). The last two
    split after `/challenge-pr` on #151, finding 1. Refuses an id c2a does not hold.
    """
    out = Counter()
    for side, kinds, ref, none in (("birth", col["kind_b"], c2a["kind_b"], "birth_new"),
                                   ("death", col["kind_a"], c2a["kind_a"], "death_gone")):
        for g, k in kinds.items():
            if k != side:
                continue
            if g not in ref:
                raise ValueError(f"group {g} is not in c2a; c2/c3 ids must be c2a's")
            out[none if ref[g] == side else f"{side}_flip_kept" if ref[g] == "stable"
                else f"{side}_flip_restructured"] += 1
    return dict(out)


def lineage_origin(links: list, steps: list, gid: int) -> int:
    """The earliest step an unbroken chain of stable links reaches back from ``gid`` at the last step."""
    i = len(steps) - 1
    while i > 0 and gid in links[i - 1]["stable"]:
        gid = links[i - 1]["stable"][gid][0]
        i -= 1
    return steps[i]


def chain(job) -> dict:
    """Every column at one (prompt, layer): links per boundary, flips, lineages, the null's stable counts."""
    key, steps, by_col, seed = job
    links = {c: [link(by_col[c][a], by_col[c][b]) for a, b in zip(steps, steps[1:])] for c in by_col}
    out = {"key": key, "boundaries": [], "lineage": {}, "null_stable": {}}
    for j in range(len(steps) - 1):
        row = {}
        for c in by_col:
            L = links[c][j]
            row[c] = {"kind_a": Counter(L["kind_a"].values()), "births": L["counts"]["birth"],
                      "n_a": len(L["kind_a"]), "n_b": len(L["kind_b"]),
                      "stable_components": L["counts"]["stable"],
                      "jaccard": [jac for _, jac in L["stable"].values()]}
            if c in FLIP_COLUMNS:
                row[c]["flips"] = flips(L, links["c2a"][j])
        out["boundaries"].append(row)
    for c in by_col:
        last = sorted({int(x) for x in by_col[c][steps[-1]] if x >= 0})
        out["lineage"][c] = [lineage_origin(links[c], steps, g) for g in last]
        if c in FLIP_COLUMNS:
            out["lineage"][c + "_by_c2a"] = [lineage_origin(links["c2a"], steps, g) for g in last]
        null = link_chain_null({i: by_col[c][s] for i, s in enumerate(steps)}, n_draws=N_DRAWS,
                               seed=seed, min_overlap=MIN_OVERLAP, measure=MEASURE)
        out["null_stable"][c] = null["counts"][:, :, _KINDS.index("stable")].tolist()
    return out


def pool(chains: list, steps: list, columns) -> dict:
    """Per boundary and column, pooled over chains; prompts' stable shares beside."""
    rows = []
    for j, (a, b) in enumerate(zip(steps, steps[1:])):
        row = {"from": a, "to": b}
        for c in columns:
            ka, fl, jac, per_prompt = Counter(), Counter(), [], {}
            n_a = n_b = births = obs = 0
            null = np.zeros(N_DRAWS)
            for ch in chains:
                r = ch["boundaries"][j][c]
                ka.update(r["kind_a"])
                fl.update(r.get("flips", {}))
                jac += r["jaccard"]
                n_a, n_b, births, obs = n_a + r["n_a"], n_b + r["n_b"], births + r["births"], obs + r["stable_components"]
                null += np.asarray(ch["null_stable"][c])[:, j]
                pp = per_prompt.setdefault(ch["key"][0], [0, 0])
                pp[0] += r["kind_a"].get("stable", 0)
                pp[1] += r["n_a"]
            jac = np.asarray(jac)
            row[c] = {
                "n_a": n_a, "n_b": n_b, "births": births,
                "share_a": {k: (ka.get(k, 0) / n_a if n_a else None) for k in A_KINDS},
                "stable_jaccard": {"n": int(jac.size),
                                   "median": float(np.median(jac)) if jac.size else None,
                                   "identical": float(np.mean(jac == 1.0)) if jac.size else None,
                                   "same": float(np.mean(jac >= SAME_JACCARD)) if jac.size else None},
                "null": {"observed": obs, "mean": float(null.mean()), "max": float(null.max()),
                         "p": float((1 + np.sum(null >= obs)) / (1 + N_DRAWS))},
                "prompt_stable_share": {p: (s / n if n else None) for p, (s, n) in sorted(per_prompt.items())},
            }
            if c in FLIP_COLUMNS:
                row[c]["flips"] = dict(fl)
        rows.append(row)
    lineage, independent = {}, {}
    for name in chains[0]["lineage"]:
        hist = Counter(o for ch in chains for o in ch["lineage"][name])
        lineage[name] = {str(s): hist.get(s, 0) for s in steps}
    for c in columns:
        independent[c] = independent_lineage(rows, steps, c)
    return {"boundaries": rows, "lineage_origin_at_last": lineage,
            "lineage_independent_cumulative": independent}


def independent_lineage(rows: list, steps: list, column: str) -> dict:
    """
    The share of the last step's groups whose chain would reach back to step s or earlier if every
    boundary broke chains independently at its own pooled rate: the product, over the boundaries
    from s to the last step, of (stable components / later step's groups). A baseline for the
    lineage table (`/challenge-pr` on #151, finding 3), not a null: it keeps each boundary's rate.
    """
    out, prod = {str(steps[-1]): 1.0}, 1.0
    for j in range(len(rows) - 1, -1, -1):
        r = rows[j][column]
        prod *= r["null"]["observed"] / r["n_b"] if r["n_b"] else 0.0
        out[str(steps[j])] = prod
    return {s: out[str(s)] for s in map(str, steps)}


def first_check(pooled: dict) -> dict:
    col, j, bar = FIRST_CHECK
    share = pooled["boundaries"][j][col]["share_a"]["stable"]
    return {"column": col, "boundary": [pooled["boundaries"][j]["from"], pooled["boundaries"][j]["to"]],
            "stable_share": share, "bar": bar, "passes": share is not None and share >= bar}


def load(src: Path, columns=COLUMNS):
    """{(prompt, layer): {column: {step: labels over the column's domain}}} and the steps, in order."""
    from tools.run.p10_label_source import MODELS, load_column, load_step  # needs hdbscan; runtime only
    steps = sorted((int(s.removeprefix("step")) for s in MODELS if (src / f"{s}.json").exists()))
    if not steps:
        raise SystemExit(f"no label source step files in {src}")
    out, domain = {}, {}
    for s in steps:
        cache = {f"step{s}": load_step(src, f"step{s}")}
        if cache[f"step{s}"]["refused"]:
            raise SystemExit(f"step{s}: refused prompts {sorted(cache[f'step{s}']['refused'])}")
        for prompt in sorted(cache[f"step{s}"]["prompts"]):
            for L in LAYERS:
                for c in columns:
                    pos, lab = load_column(src, f"step{s}", prompt, L, c, cache)
                    d = domain.setdefault((prompt, c), pos.tolist())
                    if pos.tolist() != d:
                        raise SystemExit(f"{prompt}/{c}: domain differs at step{s}; tokens would not align")
                    out.setdefault((prompt, L), {}).setdefault(c, {})[s] = lab
    return out, steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="R0's label source (`p10_label_source build --out`)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    data, steps = load(args.labels)
    prompts = sorted({p for p, _ in data})
    jobs = [((p, L), steps, data[(p, L)], [SEED, prompts.index(p), L]) for p, L in sorted(data)]

    # The first check runs alone on c2a's first boundary before anything else is computed.
    probe = [link(data[k]["c2a"][steps[0]], data[k]["c2a"][steps[1]]) for k in sorted(data)]
    n_a = sum(len(x["kind_a"]) for x in probe)
    share = sum(v == "stable" for x in probe for v in x["kind_a"].values()) / n_a if n_a else None
    if share is None or share < FIRST_CHECK[2]:
        print(f"first check fails: c2a stable share at {steps[0]} → {steps[1]} is {share}; refusing", file=sys.stderr)
        return 2

    with ProcessPoolExecutor(args.workers) as ex:
        chains = list(ex.map(chain, jobs))
    pooled = pool(chains, steps, COLUMNS)
    summary = args.labels / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no {summary}: the input would be unnamed")
    meta = {"labels": str(args.labels), "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "python": sys.version.split()[0], "numpy": np.__version__, "columns": list(COLUMNS), "steps": steps, "prompts": prompts,
            "layers": list(LAYERS), "measure": MEASURE, "min_overlap": MIN_OVERLAP,
            "same_jaccard": SAME_JACCARD, "n_draws": N_DRAWS, "seed": SEED,
            "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()}
    out = {"meta": meta, "first_check": first_check(pooled), **pooled}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}; first check {out['first_check']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
