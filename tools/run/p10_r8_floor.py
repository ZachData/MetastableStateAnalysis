"""R8: the own-floor check on "moves" (`p10_cluster_function/design-10.md` "R8"; `STATE.md` Blocked 26 (f)).

Does unit 1's "moves" (best Jaccard >= the group's own floor `J0` at every P > 0 under every
preamble, EOD join) pass c2 groups that a random set of the same size would pass as well?
Per (step, passage, layer), centred, size 2: a chance level `Jc` per group size from N_DRAWS
random same-size subsets of the kept offsets, each scored by best-match Jaccard against the
P = 0 partition (unit 1's stored groups). *Proxy:* unit 1 stored each P > 0 condition's group
count, not its labels, so P = 0's partition stands in for every condition's. **c3c** = c2 groups
that still move when every EOD condition's J must reach max(`J0`, `Jc`).

First checks (refuse rather than degrade): the stored P = 0 groups are R0's c2a labels at every
layer; the recomputed c2 and c3 are R0's. No forward pass. Tier 1: exploratory, unregistered.
    python tools/run/p10_r8_floor.py --labels <R0 labels> --unit1 <R0 unit1> --out <r8.json>
"""
import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_label_source as ls

N_DRAWS, ALPHA = 2000, 0.05             # design-10 "R8": chance level, discrete 95 %
FRAME, MCS = "centred", "2"
KEEP_MIN = 0.90                         # the decision's bar, placed
PRIMARY_FROM = 64                       # c3 readable from 64 (status-10.md §1.14)
BANDS = {"L1-8": range(1, 9), "L9-16": range(9, 17), "L17-24": range(17, 25)}


class FloorError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure

def best_jaccards(draws: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Best-match Jaccard of each row of ``draws`` (index sets) against the groups of ``lab`` (−1 = none)."""
    sizes = np.bincount(lab[lab >= 0]) if np.any(lab >= 0) else np.zeros(0, int)
    out = np.zeros(len(draws))
    s = draws.shape[1]
    for r, d in enumerate(draws):
        lm = lab[d]
        lm = lm[lm >= 0]
        if lm.size:
            ids, inter = np.unique(lm, return_counts=True)
            out[r] = np.max(inter / (s + sizes[ids] - inter))
    return out


def chance_bar(js: np.ndarray, alpha: float = ALPHA) -> float:
    """The smallest drawn value v with share(J >= v) <= alpha (inf if none: every value is common)."""
    for v in np.unique(js):
        if np.mean(js >= v) <= alpha:
            return float(v)
    return float("inf")


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def chance_by_size(lab: np.ndarray, sizes, key, n: int = N_DRAWS) -> dict:
    """Per size: (Jc, the sorted draws) against partition ``lab``."""
    out = {}
    for s in sorted(set(sizes)):
        rng = np.random.default_rng(seed_of(*key, s))
        draws = np.stack([rng.choice(lab.size, size=s, replace=False) for _ in range(n)])
        js = np.sort(best_jaccards(draws, lab))
        out[s] = (chance_bar(js), js)
    return out


def share_at_least(js_sorted: np.ndarray, v: float) -> float:
    return float(1.0 - np.searchsorted(js_sorted, v, side="left") / js_sorted.size)


def classes_at(rec: dict, floors) -> list:
    """`move_text.group_classes` with a per-group bar (``floors[i]``; None = unstable / floor 0 as there)."""
    out = []
    for i, g in enumerate(rec["groups"]):
        if not g["stable"]:
            out.append("unstable")
            continue
        if g["J0"] <= 0:
            out.append(mt.FLOOR_ZERO)
            continue
        mv = {}
        for cid, pc in rec["conditions"].items():
            src, _, j = cid.split("|")
            if j == mt.PRIMARY_JOIN:
                mv[src] = mv.get(src, True) and pc["best_jaccard"][i] >= floors[i]
        out.append(mt.classify(mv, g["opening"], rec["c_holds"]) if mv else "unclassified")
    return out


def band_of(L: int) -> str:
    return next(b for b, r in BANDS.items() if L in r)


# ---------------------------------------------------------------- one (step, passage)

def read_prompt(step: str, prompt: str, u1: dict, lab_p: dict) -> dict:
    nk = len(u1["kept_offsets"])
    kept = np.asarray(u1["kept_offsets"], dtype=int)
    cells = {(c["layer"], c["frame"]): c for c in u1["layers"]}
    rows, k_ratio = [], []
    for L in ls.LAYERS:
        if str(L) not in lab_p["layers"]:
            continue
        rec = cells[(L, FRAME)]["mcs"][MCS]
        pos = {int(o): i for i, o in enumerate(kept)}
        groups = [[pos[int(o)] for o in g["offsets"]] for g in rec["groups"]]
        lab = np.asarray(ls.labels_of(nk, groups), dtype=int)       # raises on overlap: check (a)
        cols = lab_p["layers"][str(L)]
        if lab.tolist() != cols["c2a"]:
            raise FloorError(f"{step}/{prompt}/L{L}: stored P = 0 groups are not R0's c2a")
        sizes = [len(g) for g in groups]
        own = mt.group_classes(rec)
        keeps = ls.ladder_keeps(sizes, own, nk)
        for c in ("c2", "c3"):
            if ls.labels_of(nk, groups, keeps[c]) != cols[c]:
                raise FloorError(f"{step}/{prompt}/L{L}: recomputed {c} is not R0's")
        ch = chance_by_size(lab, [s for s, k in zip(sizes, keeps["c2"]) if k], (step, prompt, L))
        jc = [ch[s][0] if k else None for s, k in zip(sizes, keeps["c2"])]
        floors = [max(g["J0"], j) if j is not None else g["J0"] for g, j in zip(rec["groups"], jc)]
        chance_cls = classes_at(rec, floors)
        for cid, pc in rec["conditions"].items():
            if cid.endswith("|" + mt.PRIMARY_JOIN) and len(groups):
                k_ratio.append(pc["k"] / len(groups))
        for i, g in enumerate(rec["groups"]):
            if not keeps["c2"][i]:
                continue
            mins = [pc["best_jaccard"][i] for cid, pc in rec["conditions"].items()
                    if cid.endswith("|" + mt.PRIMARY_JOIN)]
            rows.append({"layer": L, "id": i, "band": band_of(L), "size": sizes[i], "J0": g["J0"],
                         "Jc": jc[i], "q_J0": share_at_least(ch[sizes[i]][1], g["J0"]),
                         "min_J": float(min(mins)) if mins else None,
                         "c3": bool(keeps["c3"][i]), "c3c": chance_cls[i] == "moves"})
    return {"rows": rows, "k_ratio": k_ratio}


# ---------------------------------------------------------------- pooling

def pool(rows: list) -> dict:
    c3 = [r for r in rows if r["c3"]]
    n3, n3c = len(c3), sum(r["c3c"] for r in rows)
    lenient = [r for r in c3 if r["J0"] < r["Jc"]]
    return {"c2": len(rows), "c3": n3, "c3c": n3c,
            "c3c_not_c3": sum(r["c3c"] and not r["c3"] for r in rows),
            "keep": (n3c / n3) if n3 else None,
            "J0_below_Jc": len(lenient), "J0_below_Jc_share": (len(lenient) / n3) if n3 else None,
            "dropped_among_lenient": sum(not r["c3c"] for r in lenient),
            "median_q_J0": float(np.median([r["q_J0"] for r in c3])) if c3 else None,
            "median_Jc": float(np.median([r["Jc"] for r in c3])) if c3 else None}


def step_num(s: str) -> int:
    return int(s.replace("step", ""))


def verdict(by_step: dict) -> dict:
    prim = {s: v["all"]["keep"] for s, v in by_step.items()
            if step_num(s) >= PRIMARY_FROM and v["all"]["keep"] is not None}
    low = {s: k for s, k in prim.items() if k < KEEP_MIN}
    return {"bar": KEEP_MIN, "primary_from": PRIMARY_FROM, "min_keep": min(prim.values()) if prim else None,
            "steps_below": sorted(low, key=step_num), "c3_stands": not low}


def r7_flags(r7_dir: Path, step: str, prompt: str) -> dict:
    """R7's 20-draw `J0_below_chance` per c3 group (median over its shuffled conditions), keyed (layer, c2a id)."""
    f = r7_dir / step / f"{prompt}.json"
    if not f.exists():
        return {}
    flags = {}
    for lay in json.loads(f.read_text())["layers"]:
        for g in lay["groups"]:
            if g.get("c3") and g["chance95"]:
                flags[(lay["layer"], g["id"])] = bool(np.nanmedian(list(g["chance95"].values())) > g["J0"])
    return flags


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--unit1", type=Path, required=True)
    ap.add_argument("--r7", type=Path, default=None, help="R7 records dir, for the agreement beside")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", nargs="*", default=None, help="a subset, for the first look")
    a = ap.parse_args(argv)
    steps = a.steps or sorted((p.stem for p in a.labels.glob("step*.json")), key=step_num)
    by_step, k_ratio, agree = {}, [], [0, 0]
    for step in steps:
        lab = ls.load_step(a.labels, step)
        rows = []
        for prompt, lp in sorted(lab["prompts"].items()):
            u1 = json.loads((a.unit1 / step / f"{prompt}.json").read_text())
            r = read_prompt(step, prompt, u1, lp)
            for x in r["rows"]:
                x["prompt"] = prompt
            rows += r["rows"]
            k_ratio += r["k_ratio"]
            if a.r7 is not None:
                fl = r7_flags(a.r7, step, prompt)
                for x in r["rows"]:
                    f7 = fl.get((x["layer"], x["id"])) if x["c3"] else None
                    if f7 is not None:
                        agree[0] += f7 == (x["J0"] < x["Jc"])
                        agree[1] += 1
        by_step[step] = {"all": pool(rows),
                         "band": {b: pool([r for r in rows if r["band"] == b]) for b in BANDS},
                         "size": {s: pool([r for r in rows if (r["size"] == 2) == (s == "2")])
                                  for s in ("2", ">2")},
                         "prompt": {p: pool([r for r in rows if r["prompt"] == p])
                                    for p in sorted(lab["prompts"])}}
        print(step, json.dumps(by_step[step]["all"]))
    out = {"by_step": by_step, "verdict": verdict(by_step),
           "k_ratio": {"median": float(np.median(k_ratio)), "p10": float(np.percentile(k_ratio, 10)),
                       "p90": float(np.percentile(k_ratio, 90)), "n": len(k_ratio)},
           "r7_flag_agreement": {"agree": agree[0], "n": agree[1]},
           "meta": {"git": ls._git_head(), "labels": str(a.labels), "unit1": str(a.unit1),
                    "labels_sha": hashlib.sha256(b"".join((a.labels / f"{s}.json").read_bytes()
                                                          for s in steps)).hexdigest()[:16],
                    "n_draws": N_DRAWS, "alpha": ALPHA, "frame": FRAME, "mcs": int(MCS)}}
    a.out.write_text(json.dumps(out, indent=1))
    print("verdict", json.dumps(out["verdict"]), "k_ratio", json.dumps(out["k_ratio"]))
    return 0


if __name__ == "__main__":
    sys.exit(run())
