"""
trackC_writeup/c8_read.py — C8 U1, candidate (i): references, constants, gate and read
(`design-C8.md` "U1"; the rule is `c8_rules`). CPU only, logs and saved records only.

The order is enforced, so nothing is read before what must precede it:

1. ``reference`` — F's and F′'s onsets by rule T on their own records (C4's grid), H-ref, and the
   other conditions' onsets from the records that hold them. **No log is opened.**
2. ``constants`` — δ_L and κ on A0's updates 2001–4000 (a control's log).
3. ``gate`` — G-null, G-plant, G-edge on the control logs only (A0, A0′, B_K, BKO, B_O).
   **No with-transient log is opened**; ``_open`` refuses one before the gate has passed.
4. ``read`` — R_i at each gated τ, joint across τ, the clip label, and the beside rows.

Each output names its input: the sha256 of every log, record set and file it reads.

    METS_DATA=<main>/data python -m trackC_writeup.c8_read {reference,constants,gate,read} --out <dir>

**Deviation, fixed before any log was opened (2026-11-27):** B_K's log is 128 updates (257–384), so
at a τ where B_P's window is 24 long (3e-4) B_K's background has 2–3 blocks at every placement
(< 4: refused). At such a τ the G-null's B_P slot is B_O (the one control branch on B_P's span,
256 updates), with B_K and BKO as its clean check; elsewhere it is B_K as designed. The A0 and A0′
stand-ins have no batch twin (every other arm on their order has a transient), so their events
all count; the read's twins are as designed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from trackC_writeup import c8_rules as r8

HDD = Path("/run/media/system/HDD_1TB/mets_data")
LOGS = {"A0": "c4/A0", "F": "c4/F", "A0p": "c4pc/A0p", "Fp": "c4pc/Fp", "F1": "c4pdf/F1", "F1p": "c4pdf/rep2/F1",
        "Fh": "c4pdf/Fh", "Ft": "c4pdf/Ft", "BP": "c4pdf/BP", "BO": "c4pdf/BO", "BK": "c4pdf/BK", "BKO": "c4pdf/BKO"}
CONTROLS = ("A0", "A0p", "BK", "BKO", "BO")
BRANCHES = ("BP", "BO", "BK", "BKO")
TWIN = {"F": "A0", "Fh": "A0", "Ft": "A0", "F1": "A0", "F1p": "A0", "Fp": "A0p", "BP": "BO", "BK": "BO", "BKO": "BO"}
STAND_IN = {"F": "A0", "Fh": "A0", "Ft": "A0", "Fp": "A0p", "BP": "BK"}
BESIDE = ("F1", "F1p")
CURVE = 64


class C8ReadRefused(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dump(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _branch(arm: str) -> Optional[int]:
    return r8.BRANCH_FROM if arm in BRANCHES else None


class Logs:
    """Opens logs, recording each one's sha256; with-transient arms only once ``allow_all``."""

    def __init__(self, allow_all: bool):
        self.allow_all, self.opened, self._c = allow_all, {}, {}

    def __call__(self, arm: str) -> Tuple[Dict[int, float], Dict[int, float]]:
        if arm not in CONTROLS and not self.allow_all:
            raise C8ReadRefused(f"{arm}'s log is not opened before the gate passes")
        if arm not in self._c:
            p = HDD / LOGS[arm] / "train_log.jsonl"
            rows = r8.parse_log(p.read_text().splitlines())
            self.opened[arm] = {"path": str(p), "sha256": _sha(p), "updates": [min(rows), max(rows)]}
            self._c[arm] = ({s: float(v["loss"]) for s, v in rows.items()},
                            {s: float(v["grad_norm"]) for s, v in rows.items()})
        return self._c[arm]


# ---------------------------------------------------------------- 1. reference (no log opened)

def _records_sha(root: Path, arm: str, steps) -> str:
    from trackA_force.a1a import V1
    from trackC_writeup import c4_read as cr
    h = hashlib.sha256()
    for s in steps:
        for p in V1:
            f = cr.rec_path(root, arm, s, p)
            h.update(f"{s}/{p}".encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def reference(out: Path) -> dict:
    from trackA_force.a1a import DATA
    from trackC_writeup import c4_close as cc
    from trackC_writeup import c4_train as ct
    from trackC_writeup import c4pc_rules as pcr
    from trackC_writeup import c4pdf_read as pr
    from trackC_writeup import c4pdf_rules as rl
    arms = pr.gated_arms()
    cut = list(pr.GRID_CUT)
    res: Dict[str, dict] = {"rules": r8.code_sha(), "gated": list(arms), "per_tau": {k: {} for k in arms},
                            "H-ref": {}, "inputs": {}}
    ok = True
    for arm in ("F", "Fp"):
        rdir, _ = pr.TRAINED[arm]
        recs = pr.load_recs(rdir, arm, ct.GRID)
        res["inputs"][f"{arm}_records"] = {"dir": str(rdir / arm / "records"), "steps": "c4_train.GRID",
                                          "sha256": _records_sha(rdir, arm, ct.GRID)}
        for k in arms:
            full = cc.per_transition(recs, k, list(ct.GRID))
            tr = rl.transient(full[:len(cut)], cut)
            if not tr["present"]:
                raise C8ReadRefused(f"{arm} has no transient by rule T at {k}")
            _, l7 = pcr.end_and_transients(full, ct.GRID)
            run = sorted({t for i, s in enumerate(cut) if tr["onset"] <= s <= tr["end"]
                          for t in rl.outside(full[i])})
            hit = tuple(l7) == (3, 4, 5)
            ok &= hit
            res["H-ref"].setdefault(arm, {})[k] = {"L7_transient": list(l7), "pass": bool(hit),
                                                   "rule_T_run_outside (beside)": run}
            res["per_tau"][k][arm] = {"ref": [rl.previous(cut, tr["onset"]), tr["onset"]], "transient": tr,
                                      "source": f"rule T on {arm}'s records, C4's grid cut at 1000"}
    u3c = DATA / "trackC/c4pdf/2026-11-25-u3c/reading.json"
    rep2 = DATA / "trackC/c4pdf/2026-11-24-u3b/rep2/reference.json"
    d, d2 = json.loads(u3c.read_text()), json.loads(rep2.read_text())
    res["inputs"]["u3c_reading"] = {"path": str(u3c), "sha256": _sha(u3c)}
    res["inputs"]["rep2_reference"] = {"path": str(rep2), "sha256": _sha(rep2)}
    bgrid = [rl.BRANCH_FROM + o for o in rl.OFFSETS]
    for k in arms:
        pt = res["per_tau"][k]
        for arm in ("Fh", "Ft"):
            c = d["d1"]["cells"][k][arm]
            pt[arm] = {"ref": [c["prev"], c["onset"]], "source": "u3c reading.json d1"}
        on = d["f"][k]["BP"]["onset"]
        pt["BP"] = {"ref": [rl.previous(bgrid, on), on], "source": "u3c reading.json f (branch grid 256 + OFFSETS)"}
        pt["F1"] = {"ref": list(d["reference"][k]), "source": "u3c reading.json reference"}
        pt["F1p"] = {"ref": list(d2["reference"][k]["ref"]), "source": "u3b rep2/reference.json"}
        for arm, v in pt.items():
            v["window"] = list(r8.window(v["ref"][0], v["ref"][1], _branch(arm)))
    res["H-ref"]["pass"] = bool(ok)
    _dump(out / "reference.json", res)
    return res


# ---------------------------------------------------------------- 2. constants

def constants(out: Path) -> dict:
    lg = Logs(allow_all=False)
    loss, gn = lg("A0")
    c = r8.constants(loss, gn)
    res = {"rules": r8.code_sha(), "constants": c, "logs": lg.opened}
    _dump(out / "constants.json", res)
    return res


def _load(out: Path, name: str) -> Tuple[dict, str]:
    p = out / name
    if not p.exists():
        raise C8ReadRefused(f"{p} missing: run its step first")
    return json.loads(p.read_text()), _sha(p)


# ---------------------------------------------------------------- 3. gate (controls only)

def _cond(lg: Logs, arm: str, twin: Optional[str], lo: int, hi: int, c: dict) -> dict:
    loss, gn = lg(arm)
    tw = lg(twin) if twin else None
    return r8.condition(loss, gn, tw, lo, hi, c["delta_L"], c["kappa"], _branch(arm))


def _bp_stand_in(lg: Logs, win: Tuple[int, int]) -> str:
    lo, hi = win
    span = r8.span_of(lg("BK")[0].keys(), r8.BRANCH_FROM)
    return "BK" if len(r8.blocks(lo, hi, span)) >= r8.MIN_BLOCKS else "BO"


def _null_at(lg: Logs, wins: Dict[str, Tuple[int, int]], bp_arm: str, c: dict) -> dict:
    conds = {}
    for cnd in r8.CONDITIONS:
        arm = bp_arm if cnd == "BP" else STAND_IN[cnd]
        twin = "BO" if arm == "BK" else None
        conds[cnd] = _cond(lg, arm, twin, *wins[cnd], c)
    clean_arms = ("BKO",) if bp_arm == "BK" else ("BK", "BKO")
    clean = {a: _cond(lg, a, "BO", *wins["BP"], c)["has_event"] for a in clean_arms}
    lab = r8.label(conds, not any(clean.values()))
    return {"label": lab, "clean_check": clean,
            "b": {k: v["background"]["b"] for k, v in conds.items()},
            "events": {k: v["events"] for k, v in conds.items()}}


def _plant(lg: Logs, arm: str, twin: Optional[str], win: Tuple[int, int], c: dict, m: float) -> dict:
    loss, gn = lg(arm)
    tw = lg(twin) if twin else None
    lo, hi = win
    br = _branch(arm)
    det = {k: 0 for k in r8.KINDS}
    n, ratio, gap = 0, [], []
    for s in range(max(lo + 1, r8.first_update(br)), hi + 1):
        n += 1
        tr = range(s - r8.TRAIL, s)
        ratio.append(gn[s] / float(np.median([gn[j] for j in tr])))
        gap.append((loss[s] - min(loss[j] for j in tr)) / c["delta_L"])
        for kind in r8.KINDS:
            L, G = dict(loss), dict(gn)
            if kind == "loss":
                L[s] += m * c["delta_L"]
            else:
                G[s] *= c["kappa"] ** m
            ev = r8.events(L, G, c["delta_L"], c["kappa"], s - 1, s, br)
            te = r8.events(tw[0], tw[1], c["delta_L"], c["kappa"], s - 1, s, br) if tw else None
            det[kind] += s in r8.counted(ev, te)[kind]
    # the diagnostic (/challenge-pr on #389, finding 4): a grad plant κ^m is found iff g_s/median > κ^(1−m),
    # so with the plant tied to κ, ≥ 90 % needs κ > r_(j)^(−1/(m−1)), r_(j) the ratio with j = misses allowed
    allowed = n - int(np.ceil(0.9 * n))
    rs = sorted(ratio)
    need = float(rs[allowed] ** (-1.0 / (m - 1))) if m > 1 and rs[allowed] < 1 else None
    return {"positions": n, **{f"share_{k}": det[k] / n for k in r8.KINDS},
            "g_over_trailing_median": [round(x, 4) for x in ratio],
            "loss_minus_trailing_min_over_dL": [round(x, 4) for x in gap],
            "kappa_needed_for_0.9 (twin ignored)": need}


def gate(out: Path) -> dict:
    ref, ref_sha = _load(out, "reference.json")
    cst, cst_sha = _load(out, "constants.json")
    c = cst["constants"]
    lg = Logs(allow_all=False)
    res: Dict[str, object] = {"rules": r8.code_sha(), "inputs": {"reference.json": ref_sha, "constants.json": cst_sha},
                              "H-ref": ref["H-ref"]["pass"], "G-null": {}, "G-plant": {}, "G-edge": {}}
    ok = bool(ref["H-ref"]["pass"])
    for k in ref["gated"]:
        wins = {cnd: tuple(ref["per_tau"][k][cnd]["window"]) for cnd in r8.CONDITIONS}
        bp_arm = _bp_stand_in(lg, wins["BP"])
        spans = {cnd: r8.span_of(lg(bp_arm if cnd == "BP" else STAND_IN[cnd])[0].keys(),
                                 _branch(bp_arm if cnd == "BP" else STAND_IN[cnd])) for cnd in r8.CONDITIONS}
        ks = r8.shift_range(wins, spans)
        place = {}
        for sh in ks:
            w2 = {cnd: (lo + sh * (hi - lo), hi + sh * (hi - lo)) for cnd, (lo, hi) in wins.items()}
            place[str(sh)] = _null_at(lg, w2, bp_arm, c)
        labs = [v["label"]["label"] for v in place.values()]
        readable = [x for x in labs if x != "refused"]
        n_mark = sum(x in ("marked", "marked at ≥ half the peak") for x in readable)
        share = n_mark / len(readable) if readable else None
        passed = share is not None and share <= 0.05
        ok &= passed
        res["G-null"][k] = {"bp_stand_in": bp_arm, "shifts": ks, "n_placements": len(ks),
                            "n_refused": len(labs) - len(readable), "n_marked": n_mark, "share_marked": share,
                            "labels": dict(zip(place, labs)), "at_real_windows": place.get("0"),
                            "placements": place, "pass": bool(passed)}
        pl, ed, pok = {}, {}, True
        slots = [(cnd, STAND_IN[cnd]) for cnd in r8.CONDITIONS] + ([("BP", "BO")] if bp_arm == "BO" else [])
        for cnd, arm in slots:
            twin = "BO" if arm == "BK" else None
            key = f"{cnd}:{arm}"
            pl[key] = _plant(lg, arm, twin, wins[cnd], c, 1.5)
            ed[key] = _plant(lg, arm, twin, wins[cnd], c, 1.0)
            pok &= all(pl[key][f"share_{kind}"] >= 0.9 for kind in r8.KINDS)
        ok &= pok
        res["G-plant"][k] = {"per_window": pl, "pass": bool(pok)}
        res["G-edge"][k] = {"per_window": ed, "bar": "none (reported)"}
    res["pass"] = bool(ok)
    res["logs"] = lg.opened
    _dump(out / "gate.json", res)
    return res


# ---------------------------------------------------------------- 4. read

def _first_below_one(gn: Dict[int, float]) -> Optional[int]:
    return next((s for s in sorted(gn) if gn[s] < 1.0), None)


def _curve(lg: Logs, arm: str, ref: List[int]) -> dict:
    loss, gn = lg(arm)
    a, b = ref[0] - CURVE, ref[1] + CURVE
    return {str(s): [loss[s], gn[s]] for s in range(a + 1, b + 1) if s in loss}


def read(out: Path) -> dict:
    ref, ref_sha = _load(out, "reference.json")
    cst, cst_sha = _load(out, "constants.json")
    g, g_sha = _load(out, "gate.json")
    if not g["pass"]:
        raise C8ReadRefused("the gate did not pass: U1 reports a refusal (design 'If a gate fails')")
    if g["inputs"] != {"reference.json": ref_sha, "constants.json": cst_sha}:
        raise C8ReadRefused("gate.json was not run on these reference.json / constants.json")
    if g["rules"] != r8.code_sha():
        raise C8ReadRefused(f"rules moved since the gate: {g['rules']} → {r8.code_sha()}")
    c = cst["constants"]
    lg = Logs(allow_all=True)
    res: Dict[str, object] = {"rules": r8.code_sha(), "inputs": {"reference.json": ref_sha, "constants.json": cst_sha,
                                                               "gate.json": g_sha},
                              "constants": c, "per_tau": {}, "beside": {}}
    for k in ref["gated"]:
        pt = ref["per_tau"][k]
        conds = {cnd: _cond(lg, cnd, TWIN[cnd], *pt[cnd]["window"], c) for cnd in r8.CONDITIONS}
        clean = {a: _cond(lg, a, "BO", *pt["BP"]["window"], c) for a in ("BK", "BKO")}
        lab = r8.label(conds, not any(v["has_event"] for v in clean.values()))
        clip = {cnd: r8.clip_label(lg(cnd)[1], *pt[cnd]["window"]) for cnd in r8.CONDITIONS}
        res["per_tau"][k] = {"label": lab, "conditions": conds, "clip": clip,
                             "bk_bko": {a: {"events": v["events"], "has_event": v["has_event"]}
                                        for a, v in clean.items()}}
        bs = {}
        for cnd in BESIDE:
            bs[cnd] = _cond(lg, cnd, TWIN[cnd], *pt[cnd]["window"], c)
            bs[cnd]["clip"] = r8.clip_label(lg(cnd)[1], *pt[cnd]["window"])
        for cnd in ("F", "Fh", "Ft"):
            bs[f"A0@{cnd}"] = _cond(lg, "A0", None, *pt[cnd]["window"], c)
        bs["A0p@Fp"] = _cond(lg, "A0p", None, *pt["Fp"]["window"], c)
        bs["curves"] = {cnd: {"arm": _curve(lg, cnd, pt[cnd]["ref"]), "twin": _curve(lg, TWIN[cnd], pt[cnd]["ref"])}
                        for cnd in r8.CONDITIONS + BESIDE}
        res["beside"][k] = bs
    res["joint"] = r8.joint({k: v["label"] for k, v in res["per_tau"].items()})
    res["first_g_below_1"] = {a: _first_below_one(lg(a)[1]) for a in LOGS}
    res["logs"] = lg.opened
    _dump(out / "reading.json", res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("reference", "constants", "gate", "read"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = {"reference": reference, "constants": constants, "gate": gate, "read": read}[a.step](out)
    if a.step == "reference":
        print(json.dumps({"H-ref": res["H-ref"]["pass"],
                          "windows": {k: {c: v["window"] for c, v in pt.items()} for k, pt in res["per_tau"].items()}}))
    elif a.step == "constants":
        print(json.dumps(res["constants"]))
    elif a.step == "gate":
        print(json.dumps({"pass": res["pass"], "H-ref": res["H-ref"],
                          "G-null": {k: {x: v[x] for x in ("bp_stand_in", "n_placements", "n_refused", "n_marked", "pass")}
                                     for k, v in res["G-null"].items()},
                          "G-plant": {k: v["pass"] for k, v in res["G-plant"].items()}}, ensure_ascii=False))
    else:
        print(json.dumps({"joint": res["joint"], **{k: v["label"]["label"] for k, v in res["per_tau"].items()}},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
