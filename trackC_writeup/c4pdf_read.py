"""
trackC_writeup/c4pdf_read.py — C4′ (d)+(f): the gates' drivers, the records and heads on the new
runs, and the reads (`design-C4pdf.md` "The gates", "The readings"). May grow without moving
`c4pdf_rules`' hash.

**Gates** (`gate`, U1, no forward pass, on the four trained arms' records and heads):
G-T (F, F′ a transient at every gated τ arm, A0, A0′ none, each cut at 1000), G-d1 (F stretched
by 1/r onto grid D: "dose" at ⅒, "refused" at ½ on F's coarse reference; F unstretched "step"),
G-I (the bar rule returns each arm's half-rise date; places b = ½ P̄), G-d2 (F's top-head series
stretched: "dose" at ½, "not step" at ⅒; unstretched "step"), G-f (F's 0–256 a transient, A0's
256–512 none). H-d and H-f are `c4pdf_train`'s.

**Records and heads** (`records`, `heads`): C4's instruments (`c4_read._records_of`,
`c4_read.load_local`, `c4_read.induction_scores`, the same sequences and seed as `c4_close.heads`),
unchanged, looped over an arm's own saved states (grid D, or 256 + offsets).

**Reads** (`read`, U3): d1 per gated τ arm and joint, d2, f; each with its beside rows.

    METS_DATA=<main>/data python -m trackC_writeup.c4pdf_read gate --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, V1
from trackC_writeup import c4_close as cc
from trackC_writeup import c4_read as cr
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pdf_rules as rl
from trackC_writeup import c4pdf_train as pt

C4_U2 = DATA / "trackC/c4/2026-11-12-u2"            # A0 and F records, gate.json
C4_CLOSE = DATA / "trackC/c4/2026-11-14-close"      # heads_{A0,F}.json
C4PC_U3 = DATA / "trackC/c4pc/2026-11-22-u3"        # A0p and Fp records, heads_{A0p,Fp}.json
TRAINED = {"A0": (C4_U2, C4_CLOSE), "F": (C4_U2, C4_CLOSE), "A0p": (C4PC_U3, C4PC_U3), "Fp": (C4PC_U3, C4PC_U3)}
CUT = 1000
GRID_CUT = tuple(s for s in ct.GRID if s <= CUT)


class C4pdfReadRefused(RuntimeError):
    pass


def gated_arms() -> Tuple[str, ...]:
    g = json.loads((C4_U2 / "gate.json").read_text())
    return tuple(k for k in cr.KEYS[1:] if g["arms_gated"][k])


def load_recs(out: Path, arm: str, steps: Sequence[int]) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in steps:
        for p in V1:
            f = cr.rec_path(out, arm, s, p)
            if not f.exists():
                raise C4pdfReadRefused(f"{f}: missing")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def mats_of(recs, steps, arms) -> Dict[str, np.ndarray]:
    return {k: cc.per_transition(recs, k, list(steps)) for k in arms}


def top_series(per_step: Dict[str, list], steps: Sequence[int]) -> list:
    return [float(np.max(per_step[str(s)])) for s in steps]


def trained(arm: str, arms: Sequence[str], steps: Sequence[int] = GRID_CUT) -> dict:
    rdir, hdir = TRAINED[arm]
    recs = load_recs(rdir, arm, steps)
    heads = json.loads((hdir / f"heads_{arm}.json").read_text())["per_step"]
    return {"mats": mats_of(recs, steps, arms), "top": top_series(heads, steps),
            "top_full": top_series(heads, ct.GRID)}


def bar_of(sides: Dict[str, dict]) -> dict:
    """b = ½ P̄, P̄ the median over the four arms of `c4_read.date_on`'s plateau (top head)."""
    dates, plats = {}, {}
    for arm, s in sides.items():
        d = cr.date_on(s["top_full"], ct.GRID)
        dates[arm], plats[arm] = d["step"], d["plateau"]
    pbar = float(np.median(list(plats.values())))
    return {"plateaus": plats, "half_rise": dates, "P_bar": pbar, "b": pbar / 2}


def gate(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = gated_arms()
    sides = {arm: trained(arm, arms) for arm in TRAINED}
    res: Dict[str, dict] = {"gated_arms": list(arms), "rules": rl.code_sha(), "code": pt.code_sha()}
    # G-T
    gt = {arm: {k: rl.transient(sides[arm]["mats"][k], GRID_CUT) for k in arms} for arm in TRAINED}
    ref = {k: gt["F"][k]["onset"] for k in arms}
    gt_pass = (all(gt[x][k]["present"] for x in ("F", "Fp") for k in arms)
               and not any(gt[x][k]["present"] for x in ("A0", "A0p") for k in arms)
               and all(gt["Fp"][k]["onset"] == ref[k] for k in arms))
    res["G-T"] = {"per_arm": gt, "F_onset": ref, "pass": bool(gt_pass)}
    # G-d1 (F's coarse reference)
    gd1, ok = {}, True
    for k in arms:
        t = ref[k]
        coarse = (rl.previous(GRID_CUT, t), t)
        row = {}
        for r in (0.5, 0.1):
            st = rl.stretch(sides["F"]["mats"][k], GRID_CUT, rl.GRID_D, r)
            row[f"stretched_{r}"] = rl.label_onset(rl.transient(st, rl.GRID_D), coarse, r, rl.HORIZON)
        un = rl.stretch(sides["F"]["mats"][k], GRID_CUT, rl.GRID_D, 1.0)
        row["unstretched_0.1"] = rl.label_onset(rl.transient(un, rl.GRID_D), coarse, 0.1, rl.HORIZON)
        ok &= (row["stretched_0.1"]["label"] == "dose" and row["stretched_0.5"]["label"] == "refused"
               and row["unstretched_0.1"]["label"] == "step")
        gd1[k] = {"coarse_ref": coarse, **row}
    res["G-d1"] = {"per_arm": gd1, "pass": bool(ok)}
    # G-I
    bar = bar_of(sides)
    gi = {arm: rl.bar_date(sides[arm]["top"], GRID_CUT, bar["b"]) for arm in TRAINED}
    res["G-I"] = {**bar, "bar_dates": gi, "pass": bool(all(gi[x] == bar["half_rise"][x] for x in TRAINED)
                                                      and [gi[x] for x in ("A0", "F", "A0p", "Fp")] == [768, 320, 768, 320])}
    # G-d2
    topF = np.asarray(sides["F"]["top"])[:, None]
    gd2 = {}
    for r in (0.5, 0.1):
        st = rl.stretch(topF, GRID_CUT, rl.GRID_D, r)[:, 0]
        gd2[f"stretched_{r}"] = rl.label_induction(rl.bar_date(st, rl.GRID_D, bar["b"]), r, rl.HORIZON)
        un = rl.stretch(topF, GRID_CUT, rl.GRID_D, 1.0)[:, 0]
        gd2[f"unstretched_{r}"] = rl.label_induction(rl.bar_date(un, rl.GRID_D, bar["b"]), r, rl.HORIZON)
    res["G-d2"] = {**gd2, "pass": bool(gd2["stretched_0.5"]["label"] == "dose" and gd2["stretched_0.1"]["label"] == "not step"
                                       and gd2["unstretched_0.5"]["label"] == "step" and gd2["unstretched_0.1"]["label"] == "step")}
    # G-f
    f_states = [s for s in ct.GRID if s <= rl.OFFSETS[-1]]
    a_states = [s for s in ct.GRID if rl.BRANCH_FROM <= s <= rl.BRANCH_FROM + rl.OFFSETS[-1]]
    recsF = load_recs(C4_U2, "F", f_states)
    recsA = load_recs(C4_U2, "A0", a_states)
    gf = {k: {"F_0_256": rl.transient(cc.per_transition(recsF, k, f_states), f_states),
              "A0_256_512": rl.transient(cc.per_transition(recsA, k, a_states), a_states)} for k in arms}
    res["G-f"] = {"per_arm": gf, "pass": bool(all(v["F_0_256"]["present"] and not v["A0_256_512"]["present"]
                                                  for v in gf.values()))}
    res["pass"] = all(res[g]["pass"] for g in ("G-T", "G-d1", "G-I", "G-d2", "G-f"))
    ct._jdump(out / "gate.json", _np(res))
    print(json.dumps({g: res[g]["pass"] for g in ("G-T", "G-d1", "G-I", "G-d2", "G-f")} | {"b": bar["b"], "F_onset": ref}))
    return 0


def _np(o):
    if isinstance(o, dict):
        return {str(k): _np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return _np(o.tolist())
    return o


def saved(root: Path, arm: str) -> list:
    return [s for s in pt.save_steps(arm) if ct.ckpt_dir(root, arm, s).exists()]


def records(a) -> int:
    """C4's per-checkpoint records (`c4_read._records_of`, unchanged) on an arm's saved states."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = cr.c._torch_setup()
    from transformers import AutoTokenizer
    out, root = Path(a.out), Path(a.root)
    tok = AutoTokenizer.from_pretrained(cr.TOKENIZER[0], revision=cr.TOKENIZER[1])
    for arm in a.arm:
        first = True
        for s in saved(root, arm):
            if all(cr.rec_path(out, arm, s, p).exists() for p in V1):
                continue
            d = ct.ckpt_dir(root, arm, s)
            model = cr.load_local(d, device)
            cr._records_of(model, tok, device, out, arm, s, str(d))
            if first:
                r = json.loads(cr.rec_path(out, arm, s, V1[0]).read_text())
                print(f"first record populated: {arm} step {s}, n {r['n']}, e1c[0] {r['e1c'][0]:.6f}", flush=True)
                first = False
            del model
            torch.cuda.empty_cache()
        print(json.dumps({"arm": arm, "records": saved(root, arm)}), flush=True)
    return 0


def heads(a) -> int:
    """C4's per-head induction scores (as `c4_close.heads`: same sequences, seed, scorer)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = cr.c._torch_setup()
    out, root = Path(a.out), Path(a.root)
    g = np.random.default_rng(cr.IND_SEED)
    half = g.integers(1000, 50000, (cr.IND_N, cr.IND_LEN))
    seqs = torch.from_numpy(np.concatenate([half, half], 1)).to(device)
    for arm in a.arm:
        f = out / f"heads_{arm}.json"
        res = json.loads(f.read_text())["per_step"] if f.exists() else {}
        for s in saved(root, arm):
            if str(s) in res:
                continue
            mdl = cr.load_local(ct.ckpt_dir(root, arm, s), device)
            with torch.no_grad():
                att = mdl(input_ids=seqs, output_attentions=True).attentions
            A = torch.stack(att, 1).float().cpu().numpy()
            sc = np.mean([cr.induction_scores(A[i], cr.IND_LEN) for i in range(cr.IND_N)], 0)
            if not np.isfinite(sc).all():
                raise C4pdfReadRefused(f"{arm} step {s}: a score is not finite")
            res[str(s)] = sc.tolist()
            del mdl
            torch.cuda.empty_cache()
        ct._jdump(f, {"code": cc.code_sha(), "len": cr.IND_LEN, "n": cr.IND_N, "seed": cr.IND_SEED,
                      "model": arm, "per_step": res})
        print(json.dumps({"arm": arm, "heads": len(res)}), flush=True)
    return 0


def _new(out: Path, root: Path, arm: str, arms: Sequence[str]) -> dict:
    st = saved(root, arm)
    if st != list(pt.save_steps(arm)):
        raise C4pdfReadRefused(f"{arm}: states {sorted(set(pt.save_steps(arm)) - set(st))} missing")
    recs = load_recs(out, arm, st)
    heads = json.loads((out / f"heads_{arm}.json").read_text())["per_step"]
    return {"states": st, "mats": mats_of(recs, st, arms), "top": top_series(heads, st), "heads": heads}


def read(a) -> int:
    """U3: d1, d2 and f, after the gate passed (and, for d1 and d2, the registration decided)."""
    out, root = Path(a.out), Path(a.root)
    g = json.loads((Path(a.gate) / "gate.json").read_text())
    if not g["pass"] or g["rules"] != rl.code_sha():
        raise C4pdfReadRefused(f"gate pass {g['pass']}, rules {g['rules']} against {rl.code_sha()}")
    arms = tuple(g["gated_arms"])
    res: Dict[str, dict] = {"rules": rl.code_sha(), "gate": g["rules"]}
    f1 = _new(out, root, "F1", arms)
    ref = {}
    for k in arms:
        tr = rl.transient(f1["mats"][k], f1["states"])
        if not tr["present"]:
            raise C4pdfReadRefused(f"F1 has no transient at {k}")
        ref[k] = (rl.previous(f1["states"], tr["onset"]), tr["onset"])
    res["reference"] = ref
    if not a.f_only:
        sides = {arm: _new(out, root, arm, arms) for arm in ("Fh", "Ft")}
        per = {k: {arm: {**rl.label_onset(rl.transient(sides[arm]["mats"][k], sides[arm]["states"]), ref[k],
                                          rl.RATES[arm], rl.HORIZON),
                         "transient": rl.transient(sides[arm]["mats"][k], sides[arm]["states"])}
                   for arm in ("Fh", "Ft")} for k in arms}
        res["d1"] = {**rl.d1_joint(per, arms), "cells": per}
        d2 = {}
        for arm in ("Fh", "Ft"):
            dt = rl.bar_date(sides[arm]["top"], sides[arm]["states"], g["G-I"]["b"])
            sc = np.asarray(sides[arm]["heads"][str(rl.HORIZON)])
            d2[arm] = {**rl.label_induction(dt, rl.RATES[arm], rl.HORIZON),
                       "beside_top_head_at_1000": list(map(int, np.unravel_index(int(np.argmax(sc)), sc.shape)))}
        res["d2"] = d2
    br = {arm: _new(out, root, arm, arms) for arm in ("BP", "BO")}
    res["f"] = {k: {"BP": rl.transient(br["BP"]["mats"][k], br["BP"]["states"]),
                    "BO": rl.transient(br["BO"]["mats"][k], br["BO"]["states"])} for k in arms}
    for k in arms:
        res["f"][k]["label"] = rl.f_label(res["f"][k]["BP"], res["f"][k]["BO"])
    labs = {res["f"][k]["label"] for k in arms}
    res["f_joint"] = labs.pop() if len(labs) == 1 else "mixed"
    ct._jdump(out / ("reading_f.json" if a.f_only else "reading.json"), _np(res))
    print(json.dumps({"f": res["f_joint"], "d1": res.get("d1", {}).get("joint")}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="c4pdf_read")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gate")
    q.add_argument("--out", required=True)
    for cmd in ("records", "heads"):
        q = sub.add_parser(cmd)
        q.add_argument("--out", required=True)
        q.add_argument("--root", default=str(pt.ROOT))
        q.add_argument("--arm", nargs="+", choices=pt.ARMS, required=True)
    q = sub.add_parser("read")
    q.add_argument("--out", required=True)
    q.add_argument("--root", default=str(pt.ROOT))
    q.add_argument("--gate", required=True)
    q.add_argument("--f-only", action="store_true")
    a = p.parse_args(argv)
    return {"gate": gate, "records": records, "heads": heads, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
