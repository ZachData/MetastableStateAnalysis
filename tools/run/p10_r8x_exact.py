"""R8x: the exact check (`p10_cluster_function/design-10.md` "R8x"; `STATE.md` Blocked 27 (c)).

R8 (`p10_r8_floor.py`) scored random same-size sets against the **P = 0** partition, standing in
for each P > 0 condition's, which unit 1 never stored. Here unit 1's EOD-join P > 0 passes are
re-run in R0's environment, each condition's partition of R0's kept offsets (centred, size 2) is
recomputed and written, and the chance level `Jc` is drawn against **that condition's** partition.
**c3x** = c2 groups whose unit 1 class is still `moves` when each EOD condition's J must reach
max(`J0`, that condition's `Jc`). R8's proxy (`c3c`, its `Jc`) is recomputed beside.

First checks (refuse rather than degrade): R8's three (the stored P = 0 groups are R0's c2a, the
recomputed c2 and c3 are R0's); each condition's ids and start are unit 1's; per (condition, layer)
the recomputed partition reproduces unit 1's stored group count and every stored best Jaccard
exactly. Run in R0's environment (CPU, float32, 14 torch threads, OMP_NUM_THREADS=1).
Tier 1: exploratory, unregistered.
    python tools/run/p10_r8x_exact.py run --labels <R0 labels> --unit1 <R0 unit1> --out <dir> [--steps ...]
    python tools/run/p10_r8x_exact.py summary --out <dir>
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_label_source as ls
from tools.run import p10_r8_floor as r8

N_BOOT = 5000                           # design-10 "R8x": prompt bootstrap, as R8's reading
DECIDE = (64, 1000)                     # Blocked 27: "(a) if the drop at 64-1000 holds"


class ExactError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure

def eod_conditions(conds: list) -> list:
    return [c for c in conds if c["P"] > 0 and c["join"] == mt.PRIMARY_JOIN]


def classes_by_condition(rec: dict, floors: list) -> list:
    """`move_text.group_classes` with a bar per (group, condition): ``floors[i][cid]`` (EOD conditions)."""
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
                mv[src] = mv.get(src, True) and pc["best_jaccard"][i] >= floors[i][cid]
        out.append(mt.classify(mv, g["opening"], rec["c_holds"]) if mv else "unclassified")
    return out


def check_condition(stored: dict, offs: list, gp_off: list, where: str) -> None:
    """The recomputed partition must be the one unit 1 classified on: its count and every best Jaccard."""
    if len(gp_off) != stored["k"]:
        raise ExactError(f"{where}: {len(gp_off)} groups, unit 1 stored k = {stored['k']}")
    got = [mt.best_jaccard(o, gp_off) for o in offs]
    if got != stored["best_jaccard"]:
        bad = sum(a != b for a, b in zip(got, stored["best_jaccard"]))
        raise ExactError(f"{where}: {bad} of {len(got)} best Jaccards differ from unit 1's")


def boot_keep(per_prompt: dict, num: str, den: str = "c3", n: int = N_BOOT, seed: int = 0) -> list:
    """95 % interval of sum(num) / sum(den) over prompts resampled with replacement."""
    ps = sorted(per_prompt)
    a = np.array([per_prompt[p][num] for p in ps], float)
    b = np.array([per_prompt[p][den] for p in ps], float)
    idx = np.random.default_rng(seed).integers(0, len(ps), size=(n, len(ps)))
    sb = b[idx].sum(1)
    ok = sb > 0
    k = a[idx].sum(1)[ok] / sb[ok]
    return [float(np.percentile(k, 2.5)), float(np.percentile(k, 97.5))]


def pool(rows: list) -> dict:
    c3 = [r for r in rows if r["c3"]]
    n3 = len(c3)
    errs = [e for r in c3 for e in r["jc_err"].values()]
    return {"c2": len(rows), "c3": n3, "c3c": sum(r["c3c"] for r in rows),
            "c3x": sum(r["c3x"] for r in rows),
            "keep_c3c": (sum(r["c3c"] for r in rows) / n3) if n3 else None,
            "keep_c3x": (sum(r["c3x"] for r in rows) / n3) if n3 else None,
            "c3x_not_c3": sum(r["c3x"] and not r["c3"] for r in rows),
            "c3c_only": sum(r["c3c"] and not r["c3x"] for r in rows),
            "c3x_only": sum(r["c3x"] and not r["c3c"] for r in rows),
            "J0_below_any_Jcx": sum(any(r["J0"] < v for v in r["Jcx"].values()) for r in c3),
            "jc_err_median": float(np.median(errs)) if errs else None,
            "jc_err_p10": float(np.percentile(errs, 10)) if errs else None,
            "jc_err_p90": float(np.percentile(errs, 90)) if errs else None,
            "jc_err_abs_gt_0.05": (float(np.mean(np.abs(errs) > 0.05)) if errs else None)}


def verdict(by_step: dict) -> dict:
    keep = {s: v["all"]["keep_c3x"] for s, v in by_step.items()
            if r8.step_num(s) >= r8.PRIMARY_FROM and v["all"]["keep_c3x"] is not None}
    low = sorted((s for s, k in keep.items() if k < r8.KEEP_MIN), key=r8.step_num)
    drop = [s for s in low if DECIDE[0] <= r8.step_num(s) <= DECIDE[1]]
    return {"bar": r8.KEEP_MIN, "primary_from": r8.PRIMARY_FROM, "decide_window": list(DECIDE),
            "min_keep": min(keep.values()) if keep else None, "steps_below": low,
            "drop_at_64_1000_holds": bool(drop), "c3_stands": not low,
            "recommend": "(a) c3x as the definition's column" if drop else
                         ("c3 stands, c3x beside" if not low else "below only outside 64-1000: the user's")}


# ---------------------------------------------------------------- one (step, passage)

_G: dict = {}


def _layer_job(L: int) -> dict:
    g = _G
    kept, nk, rec = g["kept"], g["kept"].size, g["recs"][L]
    offs = [np.asarray(x["offsets"]) for x in rec["groups"]]
    sizes = sorted(set(g["sizes"][L]))
    out = {"labels": {}, "k": {}, "jc": {}}
    for c in g["conds"]:
        cid = c["id"]
        _, gp = mt.groups_of(g["hidden"][cid][L - 1], r8.FRAME, int(r8.MCS))
        gp_off = [kept[x] for x in gp]
        check_condition(rec["conditions"][cid], offs, gp_off, f"{g['where']}/L{L}/{cid}")
        lab = np.asarray(ls.labels_of(nk, [x.tolist() for x in gp]), dtype=int)
        ch = r8.chance_by_size(lab, sizes, (g["step"], g["prompt"], L, cid))
        out["labels"][cid] = lab.tolist()
        out["k"][cid] = len(gp)
        out["jc"][cid] = {s: ch[s][0] for s in sizes}
    return out


def read_prompt(model, tok, step: str, prompt: str, u1: dict, lab_p: dict, pin: dict,
                conts: dict, joins: dict, workers: int) -> dict:
    t0 = time.monotonic()
    proxy = {(x["layer"], x["id"]): x for x in r8.read_prompt(step, prompt, u1, lab_p)["rows"]}  # R8's checks
    kept = np.asarray(u1["kept_offsets"], dtype=int)
    stored = {c["id"]: c for c in u1["conditions"]}
    conds = eod_conditions(mt.conditions(prompt, pin["ids"], conts, joins))
    for c in conds:
        s = stored.get(c["id"])
        if s is None or s["start"] != c["start"] or s["n"] != len(c["ids"]):
            raise ExactError(f"{step}/{prompt}/{c['id']}: not unit 1's condition ({s})")
    if len(conds) != sum(1 for c in u1["conditions"] if c["P"] > 0 and c["join"] == mt.PRIMARY_JOIN):
        raise ExactError(f"{step}/{prompt}: EOD condition set differs from unit 1's")
    hidden = {}
    for c in conds:
        H, _ = mt.forward(model, c["ids"])
        hidden[c["id"]] = H[1:, c["start"] + kept]                  # (24, n_kept, d), L1-24
    t_fwd = time.monotonic() - t0
    cells = {(c["layer"], c["frame"]): c for c in u1["layers"]}
    layers = [L for L in ls.LAYERS if str(L) in lab_p["layers"]]
    recs = {L: cells[(L, r8.FRAME)]["mcs"][r8.MCS] for L in layers}
    sizes = {L: [x["size"] for (LL, _), x in proxy.items() if LL == L] for L in layers}
    _G.clear()
    _G.update(kept=kept, recs=recs, sizes=sizes, conds=conds, hidden=hidden, step=step, prompt=prompt,
              where=f"{step}/{prompt}")
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as p:
            per = dict(zip(layers, p.map(_layer_job, layers, chunksize=1)))
    else:
        per = {L: _layer_job(L) for L in layers}
    _G.clear()
    rows = []
    for L in layers:
        rec, jc = recs[L], per[L]["jc"]
        floors, keep_i = [], []
        for i, g in enumerate(rec["groups"]):
            x = proxy.get((L, i))
            keep_i.append(x is not None)
            floors.append({cid: max(g["J0"], jc[cid][x["size"]]) if x else g["J0"] for cid in jc})
        cls = classes_by_condition(rec, floors)
        n0 = len(rec["groups"])
        for i, ok in enumerate(keep_i):
            if not ok:
                continue
            x = proxy[(L, i)]
            jcx = {cid: jc[cid][x["size"]] for cid in jc}
            rows.append({**x, "prompt": prompt, "c3x": cls[i] == "moves", "Jcx": jcx,
                         "jc_err": {cid: v - x["Jc"] for cid, v in jcx.items()},
                         "k_ratio": {cid: per[L]["k"][cid] / n0 for cid in jc} if n0 else {}})
    labels = {"kept_offsets": kept.tolist(), "conditions": [c["id"] for c in conds],
              "layers": {str(L): per[L]["labels"] for L in layers}}
    return {"rows": rows, "labels": labels,
            "seconds": {"forward": round(t_fwd, 1), "total": round(time.monotonic() - t0, 1)}}


# ---------------------------------------------------------------- run / summary

def _env_meta() -> dict:
    import torch
    return {"git": ls._git_head(), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "omp": os.environ.get("OMP_NUM_THREADS"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p10_r8x_exact run")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--unit1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--torch-threads", type=int, default=14)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1d_cluster_ensemble.long_prompts import long_prompts_hash
    torch.set_num_threads(a.torch_threads)
    if torch.cuda.is_available():
        print("refusing: R0 ran on CPU; set CUDA_VISIBLE_DEVICES=''", file=sys.stderr)
        return 1
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    pins = mt.passage_inputs("v1", tok)
    conts, joins = mt.continuations(tok), mt.join_ids(tok)
    steps = a.steps or sorted((p.stem for p in a.labels.glob("step*.json")), key=r8.step_num)
    meta = _env_meta()
    for step in steps:
        lab = ls.load_step(a.labels, step)
        model = None
        for prompt, lp in sorted(lab["prompts"].items()):
            f = a.out / "rows" / step / f"{prompt}.json"
            if f.exists():
                print(f"already {f}", flush=True)
                continue
            u1 = json.loads((a.unit1 / step / f"{prompt}.json").read_text())
            if u1["meta"]["long_prompts_hash"] != long_prompts_hash():
                raise ExactError(f"{step}/{prompt}: long prompts changed since unit 1")
            if model is None:
                model, _ = load_model(mt.MODELS[step])
            r = read_prompt(model, tok, step, prompt, u1, lp, pins[prompt], conts, joins, a.workers)
            if not r["rows"] or not any(r["labels"]["layers"].values()):
                raise ExactError(f"{step}/{prompt}: empty output")
            for sub, body in (("labels", r["labels"]), ("rows", {"rows": r["rows"], "meta": meta})):
                g = a.out / sub / step / f"{prompt}.json"
                g.parent.mkdir(parents=True, exist_ok=True)
                g.write_text(json.dumps(body) + "\n")
            print(f"done {step}/{prompt} rows {len(r['rows'])} {r['seconds']}", flush=True)
        del model
    return 0


def summary(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="p10_r8x_exact summary")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    by_step, k_ratio, hashes = {}, [], hashlib.sha256()
    steps = sorted((p.name for p in (a.out / "rows").iterdir()), key=r8.step_num)
    for step in steps:
        rows = []
        for f in sorted((a.out / "rows" / step).glob("*.json")):
            rows += json.loads(f.read_text())["rows"]
            hashes.update((a.out / "labels" / step / f.name).read_bytes())
        k_ratio += [v for r in rows for v in r["k_ratio"].values()]
        prompts = sorted({r["prompt"] for r in rows})
        per_prompt = {p: pool([r for r in rows if r["prompt"] == p]) for p in prompts}
        by_step[step] = {"all": pool(rows),
                         "boot_c3x": boot_keep(per_prompt, "c3x"), "boot_c3c": boot_keep(per_prompt, "c3c"),
                         "band": {b: pool([r for r in rows if r["band"] == b]) for b in r8.BANDS},
                         "prompt": per_prompt}
        al = by_step[step]["all"]
        print(f"{step:>10} c3 {al['c3']:4d} c3c {al['c3c']:4d} c3x {al['c3x']:4d} "
              f"keep c3c {al['keep_c3c'] or 0:.3f} c3x {al['keep_c3x'] or 0:.3f} "
              f"boot {by_step[step]['boot_c3x']} c3c-only {al['c3c_only']} c3x-only {al['c3x_only']} "
              f"err med {al['jc_err_median']}")
    out = {"by_step": by_step, "verdict": verdict(by_step),
           "k_ratio": {"median": float(np.median(k_ratio)), "p10": float(np.percentile(k_ratio, 10)),
                       "p90": float(np.percentile(k_ratio, 90)), "n": len(k_ratio)},
           "meta": {"labels_sha": hashes.hexdigest()[:16], "n_draws": r8.N_DRAWS, "alpha": r8.ALPHA,
                    "n_boot": N_BOOT, "frame": r8.FRAME, "mcs": int(r8.MCS)}}
    (a.out / "r8x.json").write_text(json.dumps(out, indent=1))
    print("verdict", json.dumps(out["verdict"]), "k_ratio", json.dumps(out["k_ratio"]))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"run": run, "summary": summary}.get(cmd, lambda _: (print(__doc__), 2)[1])(sys.argv[2:]))
