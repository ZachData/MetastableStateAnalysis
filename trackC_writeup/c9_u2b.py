"""C9 unit 2b: the GPU rows' known answers, every row's gates (D7 (a)), and the reading: A3's
eleven pairs on GPT-2 small × 5 and the harness arms A0, A0′, each date snapped onto Pythia's
18-step grid (D4 (a); D8 (a), GPT-2 by tokens) (`design-C9.md` "U2b as it runs"; Blocked 179
D7 (a), D8 (a), the user's, 2026-11-26). Tier 1. The producer is `c9_u2b_pass.py`.

- **Snap (D8 (a)).** A run's step s stands for s / f Pythia steps, f = 4 on GPT-2 (524 288 against
  2 097 152 tokens a step) and 1 on 410m and the harness arms; it snaps to the first Pythia grid
  step at or above s / f.
- **The gate (D7 (a)).** A row is gated on a run only if both arms pass: U2a's (a plant a quarter
  of its own gap wide, scored at the own-grid step; `c9_u2.gate_row`) and the **snapped arm**:
  A3's fixed 0.25-octave plant (a dip for a turn row) centred in each own-grid gap at A3's 9 ×
  σ̄_ref, at the row's floor, placed right in ≥ 90 % when the located step **snaps** to the step
  the planted one snaps to. The null arm is U2a's (it does not depend on the plant).
- **The known answers** (each producer on its reference, before any run is gated): U2a's
  (``known.json`` passing, with its null rates equal: #382 finding 3 folded in); E5 against the
  Phase 1 reports' ``MinRank`` on 410m seed 0; E4 at ``L5H2`` against A3's ``prev``; E3 against
  C4's ``heads_A0.json`` and C4′ (c)'s ``heads_A0p.json`` (the mean over sequences); E6u, E6b,
  E6b′ against A3 units 4 and 5's records on 410m; every prompt's ids hash against the run's own
  energy records (GPT-2 and the arms).
- **The reading.** Per run: E1 (and E1c beside) from the records the paper reads (C2 3c, C4,
  C4′ (c)); E2, E3, E4, E5, E6u, E6b (E6b′), E7, E8 (E8 from its peak) located by A3's §4 on the
  run's own grid at the row's floor; E10 beside; then A3's eleven pairs on snapped indices, A3's
  rule (aligned if equal or one Pythia step apart). **The order replicates** on a run for a pair
  if its verdict equals 410m seed 0's in `a3_rows.json`. Beside, never read: the same pairs
  against 410m read with D2 (a)'s instrument (the bridge), each pair's own-grid distance, and a
  pair aligned after snapping but two or more own-grid steps apart.

    METS_DATA=<main>/data python -m trackC_writeup.c9_u2b known
    METS_DATA=<main>/data python -m trackC_writeup.c9_u2b gate --run alias
    METS_DATA=<main>/data python -m trackC_writeup.c9_u2b read
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackA_force.a3 import AMPS, GATE_EFFECT, LOW_POWER, SIGN, STATS
from trackC_writeup import c9_u2 as u
from trackC_writeup.c9_u2 import C9Refused, GPT2_SEEDS, Grid, grid_of

OUT = DATA / "trackC" / "c9" / "2026-11-26-u2b"
U2A = u.OUT
TC = DATA / "trackC"
SEED_U2B = 20261127
A3_WIDTH = 0.25                    # A3's fixed plant, octaves (D7 (a)'s arm)
TOL = {"E5": 0.005,                # the reports print MinRank to 2 decimals
       "E5_mean_rtol": 5e-4,       # a3.rows_e1_e5's own check against the panel
       "E4": 1e-3,                 # A3's prev from Phase 1's stored attentions (another pass)
       "E3": 1e-5,                 # C4's heads files, same instrument and batch
       "E6": 1e-4}                 # nats, A3's records (same code path)
READ_RUNS = GPT2_SEEDS + ("A0", "A0p")
GATE_RUNS = ("410m",) + READ_RUNS


def stat_maxmean(x):
    """k = heads: the max over heads of the mean over units (D2 (a))."""
    return x.mean(axis=-3).max(axis=-1)


STATS.setdefault("maxmean", stat_maxmean)     # a3's table, so c9_u2.noise_draws can name it

# id: (direction, statistic, kind)
ROWS_PASS = {"E3": ("up", "maxmean", "half"), "E4": ("up", "maxmean", "half"),
             "E5": ("down", "mean", "half"), "E6u": ("turn", "mean", "turn"),
             "E6b": ("turn", "mean", "turn"), "E6b_full": ("turn", "mean", "turn")}
ROWS_W = {k: (d, s, "half") for k, (d, s) in u.WROWS.items()}
ROW_ORDER = ("E2", "E7", "E8", "E3", "E4", "E5", "E6u", "E6b", "E6b_full")


class U2bRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u2b.py", "trackC_writeup/c9_u2.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- snapping (pure)

def factor(run: str) -> int:
    return 4 if run in GPT2_SEEDS else 1


def snap(run: str, step: int) -> int:
    t = step / factor(run)
    for s in STEPS:
        if s >= t:
            return s
    raise U2bRefused(f"{run} step {step} is past Pythia's grid")


def snap_index(run: str, step: int) -> int:
    return STEPS.index(snap(run, step))


# ---------------------------------------------------------------- the snapped arm (D7 (a))

def shape_fixed(g: Grid, i: int, width: float = A3_WIDTH) -> np.ndarray:
    X = g.X
    sh = 1 / (1 + np.exp(-(X - (X[i] + X[i + 1]) / 2) / width))
    return sh - sh[0]


def dip_fixed(g: Grid, i: int) -> np.ndarray:
    return -shape_fixed(g, i) + 0.5 * shape_fixed(g, i + 1)


def snapped_arm(run: str, g: Grid, direction: str, sname: str, kind: str, units, rng,
                floor: float) -> dict:
    """A3's 0.25-octave plant in every own-grid gap at 9 σ̄_ref, at the row's floor; placed right
    when the located step snaps where the planted step snaps."""
    xc, sc = u.noise_draws(g, units, direction, sname, rng, a3.N_GATE)
    sbar = sc.mean(axis=0)
    out = []
    for i in u.gaps(g, kind):
        sref = float(sbar[i + 1:].max())
        sh = dip_fixed(g, i) if kind == "turn" else SIGN[direction] * shape_fixed(g, i)
        # the planted step: where the noise-free plant itself is located. For a half row it is
        # i + 1; a fixed-width dip on a sub-octave grid has its minimum a step or more later, so
        # i + 1 was wrong for turn rows (found on the first gate run; LESSONS.md 3)
        free = u.place(g, LOW_POWER * sref * sh, direction, None, kind, floor=None)
        if kind != "turn" and free["step"] != g.steps[i + 1]:
            raise U2bRefused(f"{g.name} gap {i}: a noise-free half plant is located at {free['step']}")
        want = snap(run, free["step"])
        hits = []
        for x, s in zip(xc, sc):
            y = x + LOW_POWER * sref * sh
            r = u.place(g, y, direction, float(s[u.reference(g, y, direction, kind)]), kind,
                        floor=floor)
            hits.append(r["step"] is not None and snap(run, r["step"]) == want)
        ok = float(np.mean(hits))
        out.append({"gap": f"{g.steps[i]}-{g.steps[i + 1]}", "to_index": i + 1, "planted_step": free["step"], "snaps_to": want,
                    "sigma_ref": sref, "placed_right_snapped": ok, "pass": ok >= GATE_EFFECT})
    return {"width_octaves": A3_WIDTH, "effect": out, "pass": all(e["pass"] for e in out)}


# ---------------------------------------------------------------- units from the records

def pass_records(run: str, d: Path) -> Dict[int, dict]:
    from trackC_writeup.c9_u2b_pass import code_sha as pass_sha
    g = grid_of(run)
    recs = {}
    for s in g.steps:
        f = d / "pass" / run / f"step{s}.json"
        if not f.exists():
            raise U2bRefused(f"{run}: no pass record {f}")
        recs[s] = json.loads(f.read_text())
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise U2bRefused(f"{run}: pass records from {len(codes)} codes")
    if codes != {pass_sha()}:
        raise U2bRefused(f"{run}: pass records' code {codes} is not the producer's {pass_sha()}")
    return recs


def pass_units(run: str, d: Path) -> Dict[str, np.ndarray]:
    g = grid_of(run)
    R = pass_records(run, d)
    S = g.steps
    e3 = np.array([np.reshape(R[s]["e3"], (len(R[s]["e3"]), -1)) for s in S])   # (S, N, heads)
    e4 = np.array([[np.ravel(R[s]["e4"][p]) for p in V1] for s in S])         # (S, P, heads)
    out = {"E3": e3.transpose(1, 0, 2), "E4": e4.transpose(1, 0, 2),
           "E5": np.array([[R[s]["e5"][p] for s in S] for p in V1])[..., None]}
    if "e6" in R[S[0]]:
        for rid, key in (("E6u", "kl_u"), ("E6b", "kl_b"), ("E6b_full", "kl_b_full")):
            out[rid] = np.array([R[s]["e6"][key] for s in S]).T[..., None]
        out["_loss"] = np.array([R[s]["e6"]["loss"] for s in S]).mean(1)
        out["_E6u_full"] = np.array([R[s]["e6"]["kl_u_full"] for s in S]).mean(1)
    out["_ids_sha"] = {s: R[s]["ids_sha"] for s in S}
    out["_code"] = R[S[0]]["code"]
    return out


# ---------------------------------------------------------------- the known answers

def known(d: Path) -> dict:
    res, ok = {}, True
    ka = json.loads((U2A / "known.json").read_text())
    nulls = ka["i"].get("null_no_change_equal", {})
    res["u2a"] = {"pass": bool(ka.get("pass")), "null_no_change_equal": all(nulls.values()) and bool(nulls)}
    ok &= res["u2a"]["pass"] and res["u2a"]["null_no_change_equal"]
    # 410m seed 0: E5 against the reports, E4 at L5H2 against A3's prev, E6 against A3's records
    w = pass_units("410m", d)
    rep = {s: a3.report_rows(s) for s in STEPS}
    want5 = np.array([[rep[s][p]["min_rank"] for s in STEPS] for p in V1])
    gap5 = float(np.max(np.abs(w["E5"][..., 0] - want5)))
    pan = a3.panel()
    pm = np.array([float(pan[s]["p1_eff_rank_normed_mean"]) for s in STEPS])
    rel5 = float(np.max(np.abs(w["E5"][..., 0].mean(0) - pm) / pm))
    res["E5"] = {"max_abs_vs_report": gap5, "max_rel_mean_vs_panel": rel5,
                 "pass": gap5 <= TOL["E5"] and rel5 <= TOL["E5_mean_rtol"]}
    attn = json.loads((u.A3U5 / "attn_rows.json").read_text())
    l, h = a3.PREV
    n_heads = 16
    got4 = w["E4"][:, :, l * n_heads + h]                                     # (P, S)
    want4 = np.array([[attn[f"{s}|{p}"]["prev"] for s in STEPS] for p in V1])
    gap4 = float(np.max(np.abs(got4 - want4)))
    res["E4"] = {"head": f"L{l}H{h}", "max_abs_vs_a3": gap4, "pass": gap4 <= TOL["E4"]}
    from trackA_force.a3_ngram import load_steps
    e6 = {}
    for rid, key, src in (("E6u", "kl_u", u.DATA / "trackA" / "a3" / "2026-10-20-A3u4-ngram"),
                          ("E6b", "kl_b", u.DATA / "trackA" / "a3" / "2026-10-20-A3u4-ngram"),
                          ("E6b_full", "kl_b", u.DATA / "trackA" / "a3" / "2026-10-20-A3u5-ngram-full")):
        old = load_steps(src)
        ref = np.array([old[s][key] for s in STEPS]).T
        e6[rid] = float(np.max(np.abs(w[rid][..., 0] - ref)))
    res["E6"] = {"max_abs_nats": e6, "pass": max(e6.values()) <= TOL["E6"]}
    # E3: C4's heads files on the harness arms
    e3 = {}
    for run, f in (("A0", TC / "c4" / "2026-11-14-close" / "heads_A0.json"),
                   ("A0p", TC / "c4pc" / "2026-11-22-u3" / "heads_A0p.json")):
        hu = pass_units(run, d)["E3"]                                          # (N, S, heads)
        ref = json.loads(f.read_text())["per_step"]
        e3[run] = float(max(np.max(np.abs(hu[:, k].mean(0) - np.ravel(ref[str(s)])))
                            for k, s in enumerate(grid_of(run).steps)))
    res["E3"] = {"max_abs_vs_heads_files": e3, "pass": max(e3.values()) <= TOL["E3"]}
    # prompt ids against each read run's energy records
    ids = {}
    for run in READ_RUNS:
        pu = pass_units(run, d)["_ids_sha"]
        bad = 0
        for s in grid_of(run).steps:
            for p in V1:
                if run in GPT2_SEEDS:
                    f = TC / "c2u3c" / "2026-10-22-run" / run / "records" / f"step{s}_{p}.json"
                elif run == "A0":
                    f = TC / "c4" / "2026-11-12-u2" / "A0" / "records" / f"step{s}_{p}.json"
                else:
                    f = TC / "c4pc" / "2026-11-22-u3" / "A0p" / "records" / f"step{s}_{p}.json"
                bad += json.loads(f.read_text())["ids_sha"] != pu[s][p]
        ids[run] = bad
    res["ids"] = {"mismatches": ids, "pass": sum(ids.values()) == 0}
    for k in ("E5", "E4", "E6", "E3", "ids"):
        ok &= res[k]["pass"]
    res.update(code=code_sha(), tol=TOL, pass_=bool(ok))
    res["pass"] = res.pop("pass_")
    d.mkdir(parents=True, exist_ok=True)
    (d / "known.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v.get("pass") if isinstance(v, dict) else v for k, v in res.items()}), flush=True)
    return res


# ---------------------------------------------------------------- the gates

def all_units(run: str, d: Path) -> dict:
    U = pass_units(run, d)
    W = u.weight_units(run, U2A)
    return {**{k: W[k] for k in ("E2", "E7", "E8")}, **U, "_weights_code": W["_code"]}


def gate(run: str, d: Path) -> dict:
    kn = d / "known.json"
    if not kn.exists() or not json.loads(kn.read_text()).get("pass"):
        raise U2bRefused("U2b's known answers have not passed: no run is gated before them")
    g = grid_of(run)
    U = all_units(run, d)
    t0 = time.time()
    res = {"code": code_sha(), "run": run, "grid": list(g.steps), "pass_code": U["_code"],
           "weights_code": U["_weights_code"], "rows": {}}
    old = None if run == "410m" else json.loads((U2A / "gate" / f"{run}.json").read_text())
    for k, rid in enumerate(ROW_ORDER):
        if rid not in U or (run == "410m" and rid not in ("E3", "E4")):
            continue                                       # 410m: only the bridge rows
        direction, sname, kind = {**ROWS_W, **ROWS_PASS}[rid]
        rng = np.random.default_rng([SEED_U2B, GATE_RUNS.index(run), k])
        if rid in ROWS_W and old is not None:
            own = old["rows"][rid]                         # U2a's own-grid arm, as recorded
        elif rid in ROWS_W:
            continue                                       # 410m's weights rows are A3's
        else:
            own = u.gate_or_refuse(g, rid, direction, sname, kind, U[rid], rng)
        if own.get("refused"):
            res["rows"][rid] = {"own": own, "snapped": None, "pass": False}
            continue
        sn = snapped_arm(run, g, direction, sname, kind, U[rid], rng, own["floor"])
        res["rows"][rid] = {"own": own, "snapped": sn, "floor": own["floor"],
                            "n_units": int(U[rid].shape[0]),
                            "pass": bool(own["pass"] and sn["pass"])}
        print(f"{run} gate {rid}: own {'pass' if own['pass'] else 'FAIL'}, snapped min "
              f"{min(e['placed_right_snapped'] for e in sn['effect']):.3f} -> "
              f"{'pass' if res['rows'][rid]['pass'] else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    (d / "gate").mkdir(parents=True, exist_ok=True)
    (d / "gate" / f"{run}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ---------------------------------------------------------------- the reading (pure parts)

def from_peak(g: Grid, x, direction: str) -> dict:
    w = np.asarray(x, float)[list(g.win)]
    p = int(np.argmax(-SIGN[direction] * w))
    dd = SIGN[direction] * (w[p + 1:] - w[p])
    if len(dd) == 0 or dd.max() <= 0:
        return {"reading": "no change", "index": None, "step": None, "peak_step": g.steps[p]}
    j = int(np.argmax(dd)) + p + 1
    i = int(np.argmax(dd >= dd.max() / 2)) + p + 1
    return {"reading": "located", "index": i, "step": g.steps[i], "peak_step": g.steps[p],
            "censored": j == g.win[-1]}


def dates(rows: dict, rid: str):
    """Own-grid steps, one per date, or (None, why)."""
    out = []
    for k in (rid,) + ((a3.SECOND_DATE[rid],) if rid in a3.SECOND_DATE else ()):
        r = rows.get(k)
        if r is None:
            return None, f"{k} not read on this run"
        if r["reading"] != "located":
            return None, f"{k} {r['reading']}"
        if r.get("censored"):
            return None, f"{k} censored"
        out.append(r["step"])
    return out, None


def read_pair(run: str, rows: dict, a: str, b: str, own_steps) -> dict:
    da, wa = dates(rows, a)
    db, wb = dates(rows, b)
    if da is None or db is None:
        return {"pair": [a, b], "verdict": "unread", "why": wa or wb}
    si = {x: snap_index(run, x) for x in da + db}
    vs = {a3.verdict(si[x], si[y], a, b) for x in da for y in db}
    st = {a: da, b: db, "snapped": {a: [snap(run, x) for x in da], b: [snap(run, y) for y in db]}}
    if len(vs) > 1:
        return {"pair": [a, b], "steps": st, "verdict": "unread (two dates)",
                "why": " and ".join(sorted(" ".join(v) for v in vs))}
    v = vs.pop()
    out = {"pair": [a, b], "steps": st, "verdict": v[0]}
    if v[0] == "misaligned":
        out["first"] = v[1]
    own = [abs(own_steps.index(x) - own_steps.index(y)) for x in da for y in db]
    out["own_grid_distance"] = own
    out["finer_ordered"] = bool(v[0] == "aligned" and min(own) >= 2)
    return out


def same(x: dict, ref: dict) -> Optional[bool]:
    if ref["verdict"] not in ("aligned", "misaligned") or x["verdict"] not in ("aligned", "misaligned"):
        return None
    return x["verdict"] == ref["verdict"] and x.get("first") == ref.get("first")


# ---------------------------------------------------------------- the reading

E1_SOURCES = {"A0": ("c4", 384, 640), "A0p": ("c4pc", 320, 640)}   # checked against the records in read()


def e1_rows(run: str) -> dict:
    if run in GPT2_SEEDS:
        r = json.loads((TC / "c2u3c" / "2026-10-22-run" / "reading.json").read_text())["per_seed"][run]
        e1 = {"reading": "located", "step": r["uncentred"]["step"], "censored": False,
              "source": "c2u3c/2026-10-22-run/reading.json uncentred"}
        dd = r["centred"]["dates"]
        e1c = {"reading": "no date (arms disagree)", "step": None, "arms": dd,
               "source": "c2u3c reading.json centred"}
        return {"E1": e1, "E1c": e1c}
    if run == "A0":
        r = json.loads((TC / "c4" / "2026-11-12-u2" / "reading.json").read_text())
        k = [x for x in r["gated"] if x != "u" and r["gated"][x]]
        c = {r["dates"]["A0"][x]["step"] for x in k}
        e1 = r["dates"]["A0"]["u"]["step"]
        src = "c4/2026-11-12-u2/reading.json"
    else:
        r = json.loads((TC / "c4pc" / "2026-11-22-u3" / "measured.json").read_text())
        k = [x for x, v in r["gate3500"]["arms_gated"].items() if v]
        c = {r["M1"][x]["t_A0p"] for x in k}
        e1 = r["M1"]["u"]["t_A0p"]
        src = "c4pc/2026-11-22-u3/measured.json M1"
    if len(c) != 1 or (e1, c.copy().pop()) != E1_SOURCES[run][1:]:
        raise U2bRefused(f"{run}: E1 {e1}, centred {c} from {src}, not {E1_SOURCES[run][1:]}")
    return {"E1": {"reading": "located", "step": e1, "censored": False, "source": src},
            "E1c": {"reading": "located", "step": c.pop(), "censored": False, "source": src,
                    "arms": k}}


def e10_row(run: str) -> dict:
    """The confound row, beside: Pythia's schedule on the harness arms; on GPT-2 the logged LR,
    whose tags 10–80 are not logged (half the peak lies in (50, 100], which snaps to 32 either way)."""
    if run in GPT2_SEEDS:
        lr = json.loads((REPO / "trackC_writeup" / "gpt2_small_lr_log.json").read_text())["seeds"][run]["lr"]
        peak = max(v for s, v in lr if s <= 4000)
        first = next(s for s, v in lr if v >= peak / 2)
        prev_ = max(s for s, v in lr if s < first)
        return {"reading": "interval", "interval": [prev_, first], "step": None,
                "snapped": sorted({snap(run, s) for s in grid_of(run).steps if prev_ < s <= first}),
                "warmup_end": 4000}
    g = grid_of(run)
    x = np.array([a3.lr(s) for s in g.steps])
    r = u.place(g, x, "up", None, floor=None)
    return {**r, "snapped": [snap(run, r["step"])], "warmup_end": a3.WARMUP}


def read_run(run: str, d: Path) -> dict:
    g = grid_of(run)
    gt = json.loads((d / "gate" / f"{run}.json").read_text())
    U = all_units(run, d)
    rng = np.random.default_rng([SEED_U2B, GATE_RUNS.index(run), 99])
    rows = e1_rows(run)
    for rid in ROW_ORDER:
        if rid not in U:
            continue
        direction, sname, kind = {**ROWS_W, **ROWS_PASS}[rid]
        x = STATS[sname](U[rid])
        r = {"series": x.tolist(), "n_units": int(U[rid].shape[0])}
        if not gt["rows"][rid]["pass"]:
            r.update(reading="refused", step=None)
        else:
            r.update(u.locate_units(g, U[rid], STATS[sname], direction, rng, kind,
                                    floor=gt["rows"][rid]["floor"]))
            r["censored"] = u.censored(g, r)
        rows[rid] = r
    if "E8" in rows:
        rows["E8_peak"] = from_peak(g, np.asarray(rows["E8"]["series"]), "down")
    rows["E10"] = e10_row(run)
    pairs = [read_pair(run, rows, a, b, list(g.steps)) for a, b in a3.PAIRS]
    if "_loss" in U:
        rows["E6u"]["beside"] = {"model_loss": U["_loss"].tolist(), "E6u_full": U["_E6u_full"].tolist()}
    return {"rows": rows, "pairs": pairs}


def bridge(d: Path) -> dict:
    """410m seed 0 with D2 (a)'s E3, E4 (gated on Pythia's grid), A3's other rows as recorded."""
    rec = json.loads((u.A3U5 / "a3_rows.json").read_text())
    gt = json.loads((d / "gate" / "410m.json").read_text())
    U = pass_units("410m", d)
    rng = np.random.default_rng([SEED_U2B, 0, 99])
    rows = {k: dict(v) for k, v in rec["rows"].items()}
    out = {}
    for rid in ("E3", "E4"):
        direction, sname, kind = ROWS_PASS[rid]
        if not gt["rows"][rid]["pass"]:
            rows[rid] = {"reading": "refused", "step": None}
        else:
            r = u.locate_units(u.PYTHIA, U[rid], STATS[sname], direction, rng, kind,
                               floor=gt["rows"][rid]["floor"])
            r["censored"] = u.censored(u.PYTHIA, r)
            rows[rid] = r
        out[rid] = {k: rows[rid].get(k) for k in ("reading", "step", "censored", "ref_over_sigma")}
    pairs = [read_pair("410m", rows, a, b, list(STEPS)) for a, b in a3.PAIRS]
    return {"rows": out, "pairs": pairs}


def read(d: Path) -> dict:
    kn = json.loads((d / "known.json").read_text())
    if not kn.get("pass"):
        raise U2bRefused("the known answers have not passed")
    a3rec = json.loads((u.A3U5 / "a3_rows.json").read_text())
    ref = {tuple(p["pair"]): p for p in a3rec["reading"]["pairs"]}
    br = bridge(d)
    bref = {tuple(p["pair"]): p for p in br["pairs"]}
    res = {"code": code_sha(), "a3_record_sha256": hashlib.sha256((u.A3U5 / "a3_rows.json").read_bytes()).hexdigest(),
           "bridge_410m": br, "runs": {}, "replicates": {}}
    for run in READ_RUNS:
        rr = read_run(run, d)
        for p in rr["pairs"]:
            k = tuple(p["pair"])
            p["reference"] = {"verdict": ref[k]["verdict"], "first": ref[k].get("first")}
            p["replicates"] = same(p, ref[k])
            p["replicates_bridge"] = same(p, bref[k])
        res["runs"][run] = rr
    for a, b in a3.PAIRS:
        k = (a, b)
        res["replicates"][f"{a}-{b}"] = {
            "reference": ref[k]["verdict"] + (f" ({ref[k]['first']} first)" if ref[k].get("first") else ""),
            "gpt2": [next(p for p in res["runs"][s]["pairs"] if tuple(p["pair"]) == k)["replicates"] for s in GPT2_SEEDS],
            "A0": next(p for p in res["runs"]["A0"]["pairs"] if tuple(p["pair"]) == k)["replicates"],
            "A0p": next(p for p in res["runs"]["A0p"]["pairs"] if tuple(p["pair"]) == k)["replicates"]}
    (d / "reading.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def table(res: dict) -> str:
    lines = ["| pair | 410m s0 | GPT-2 × 5 (replicates) | A0 | A0′ |", "|---|---|---|---|---|"]
    for k, v in res["replicates"].items():
        g = v["gpt2"]
        gs = f"{sum(1 for x in g if x)} of {sum(1 for x in g if x is not None)} read"
        f = lambda x: "—" if x is None else ("yes" if x else "no")
        lines.append(f"| {k} | {v['reference']} | {gs} | {f(v['A0'])} | {f(v['A0p'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("known"); k.add_argument("--out", type=Path, default=OUT)
    gq = sub.add_parser("gate"); gq.add_argument("--run", required=True, choices=GATE_RUNS)
    gq.add_argument("--out", type=Path, default=OUT)
    r = sub.add_parser("read"); r.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "known":
        return 0 if known(a.out)["pass"] else 1
    if a.cmd == "gate":
        gate(a.run, a.out)
        return 0
    res = read(a.out)
    print(table(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
