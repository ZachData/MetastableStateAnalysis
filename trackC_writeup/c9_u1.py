"""C9 unit 1: the mean's term on every model (`design-C9.md` "U1"; Blocked 178 D1 (a)). No forward
pass: every model's records already hold `e1`, `cos` and `n` at each state.

By E₁ = ½ + ½‖m‖² + R (R ≥ 0; `tools/math_checks/c2_energy_moments.py` M4), ΔE₁ = ½Δ‖m‖² + ΔR.
Per model (a row of the design's table), pooled over its runs, prompts, transitions and
checkpoints:

- S_any, the paper's sentence (description, no label): #(E₁ falls with ‖m‖² falling) / #(E₁ falls);
- M, the reading: over the drops the count counts (C2's D2 (a) REL_TOL), the share at which
  s = ½Δ‖m‖² / ΔE₁ > ½. Labelled against M₄₁₀ − 0.1 (M₄₁₀ = 410m's pooled M from C2 5b's record):
  "carries" if M ≥ M₄₁₀ − 0.1, "partly" if ½ ≤ M < M₄₁₀ − 0.1, "not" if M < ½.

Order, each step refusing before the next: (1) every (step, prompt) record of every run present,
one state count per run and one `n` per prompt; (2) the known answer, U1's code against `c2_readings` on
410m's records and C2 5b's R6 numbers exactly; (3) per model, the two plants (R held at the
series' first state: M = 1, S_any = 1; ‖m‖² held there: M = 0, "not"); a model failing a plant,
or R ≥ 0, is refused; (4) M₄₁₀; (5) the other models. Beside, never read: the base rate, the
same-direction share, per-checkpoint and per-run values, S_break.

Run: METS_DATA=<main>/data python -m trackC_writeup.c9_u1
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackA_force.a5_seeds import REL_TOL

TC = DATA / "trackC"
C2U5B = TC / "c2u5b" / "2026-10-23" / "readings.json"
OUT = TC / "c9" / "2026-11-25-u1"
MARGIN = 0.1                                    # D1 (a): placed, not calibrated
GPT2 = ("alias", "battlestar", "caprica", "darkmatter", "expanse")


class Refused(RuntimeError):
    pass


def models() -> Dict[str, Dict[str, Path]]:
    """The design's table, one entry a model, each run's records folder (70m: C6's ten seeds;
    seed 0 is `pythia-70m`, so C2 3b's copy is not read twice)."""
    return {
        "pythia-410m": {f"seed{s}": (TC / "c2u3" / "2026-10-22-run" / "records" if s == 0 else
                                     TC / "c2u3s" / "2026-10-22-run" / f"seed{s}" / "records") for s in range(10)},
        "pythia-160m": {"seed0": TC / "c2u3b" / "2026-10-22-run" / "160m" / "records"},
        "pythia-70m": {f"seed{s}": TC / "c6" / "2026-10-31-u0-seeds" / f"seed{s}" / "records" for s in range(10)},
        "pythia-70m-warmup01": {"run": TC / "c6" / "2026-11-01-u1b" / "run" / "records"},
        "gpt2-small": {k: TC / "c2u3c" / "2026-10-22-run" / k / "records" for k in GPT2},
        "70m-harness": {"A0": TC / "c4" / "2026-11-12-u2" / "A0" / "records",
                        "A0p": TC / "c4pc" / "2026-11-22-u3" / "A0p" / "records"},
        "70m-flat": {"F": TC / "c4" / "2026-11-12-u2" / "F" / "records",
                     "Fp": TC / "c4pc" / "2026-11-22-u3" / "Fp" / "records"},
    }


def breaks() -> Dict[str, Dict[str, Optional[int]]]:
    """Each run's E₁ break as table 2 (§4.2) prints it, from the records it was read from; None where
    table 2 has no E₁ date (70m seeds 1–9: centred only; warmup01: refused). Beside only (S_break)."""
    from trackC_writeup.c2_readings import a5_break
    b3 = json.loads((TC / "c2u3b" / "2026-10-22-run" / "reading.json").read_text())["per_size"]
    b3c = json.loads((TC / "c2u3c" / "2026-10-22-run" / "reading.json").read_text())["per_seed"]
    c4 = json.loads((TC / "c4" / "2026-11-12-u2" / "reading.json").read_text())["dates"]
    c4pc = json.loads((TC / "c4pc" / "2026-11-22-u3" / "measured.json").read_text())["dates"]
    return {"pythia-410m": {f"seed{s}": a5_break(s) for s in range(10)},
            "pythia-160m": {"seed0": b3["160m"]["uncentred"]["step"]},
            "pythia-70m": {f"seed{s}": (b3["70m"]["uncentred"]["step"] if s == 0 else None) for s in range(10)},
            "pythia-70m-warmup01": {"run": None},
            "gpt2-small": {k: b3c[k]["uncentred"]["step"] for k in GPT2},
            "70m-harness": {"A0": c4["A0"]["u"]["step"], "A0p": c4pc["A0p"]["u"]["step"]},
            "70m-flat": {"F": c4["F"]["u"]["step"], "Fp": c4pc["Fp"]["u"]["step"]}}


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u1.py", "trackC_writeup/c2_readings.py", "trackA_force/a5_seeds.py",
              "trackA_force/a1a.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def inputs_sha(d: Path) -> str:
    """As `c2_readings.inputs_sha`: sha256 over the folder's record files, in name order."""
    h = hashlib.sha256()
    for f in sorted(d.glob("step*_*.json")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- (1) the records

def load_run(d: Path) -> Tuple[List[int], Dict[Tuple[int, str], dict]]:
    """Every (step, prompt) record in ``d``; refuses on a missing prompt or step, or a run whose
    records disagree in state count or n."""
    fs = sorted(d.glob("step*_*.json"))
    if not fs:
        raise Refused(f"{d}: no records")
    recs = {}
    for f in fs:
        st, p = f.name[4:-5].split("_", 1)
        recs[(int(st), p)] = json.loads(f.read_text())
    steps = sorted({s for s, _ in recs})
    missing = [(s, p) for s in steps for p in V1 if (s, p) not in recs]
    extra = sorted({p for _, p in recs} - set(V1))
    if missing or extra:
        raise Refused(f"{d}: missing {missing[:5]} ({len(missing)}), unknown prompts {extra}")
    shapes = {(len(r["e1"]), len(r["cos"])) for r in recs.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] != next(iter(shapes))[1]:
        raise Refused(f"{d}: (states e1, states cos) not one value: {shapes}")
    for p in V1:                                # n is the prompt's token count: one per prompt
        ns = {recs[(s, p)]["n"] for s in steps}
        if len(ns) != 1:
            raise Refused(f"{d}: {p} has n {sorted(ns)} across steps")
    return steps, recs


# ---------------------------------------------------------------- pure

def m2(r: dict) -> np.ndarray:
    """‖m‖² at each state from the record's mean off-diagonal cosine (math check M5)."""
    return (1 + (r["n"] - 1) * np.asarray(r["cos"], float)) / r["n"]


def series(e: np.ndarray, m: np.ndarray) -> dict:
    """One (step, prompt) series of E₁ and ‖m‖² over depth; the state count is the record's
    (`c2_readings.one_series` hard-codes 410m's 25)."""
    e, m = np.asarray(e, float), np.asarray(m, float)
    rest = e - 0.5 - 0.5 * m
    de, dm = np.diff(e), 0.5 * np.diff(m)
    rel = -de / np.maximum(np.abs(e[:-1]), 1e-12)
    d = rel > REL_TOL
    return {"d": d, "m_fall": dm < 0, "e_fall": de < 0, "same": np.sign(de) == np.sign(dm),
            "s": np.where(d, dm / np.where(d, de, 1.0), np.nan), "rest_min": float(rest.min()),
            "net": float((e[-1] - e[0]) / e[0]), "net_inner": float((e[-2] - e[1]) / e[1])}


def counts(x: dict) -> Dict[str, int]:
    d = x["d"]
    return {"drops": int(d.sum()), "s_gt_half": int((x["s"][d] > 0.5).sum()),
            "m_fall_at": int((d & x["m_fall"]).sum()), "e_fall": int(x["e_fall"].sum()),
            "m_fall_e_fall": int((x["e_fall"] & x["m_fall"]).sum()), "m_fall_all": int(x["m_fall"].sum()),
            "same": int(x["same"].sum()), "trans": int(d.size)}


def add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a.get(k, 0) + b[k] for k in b}


def share(num: int, den: int) -> Optional[float]:
    return None if den == 0 else float(num / den)


def stats(q: Dict[str, int]) -> dict:
    return {"M": share(q["s_gt_half"], q["drops"]), "S_any": share(q["m_fall_e_fall"], q["e_fall"]),
            "drops": q["drops"], "e_falls": q["e_fall"], "transitions": q["trans"],
            "base_rate": share(q["m_fall_all"], q["trans"]), "same_direction": share(q["same"], q["trans"]),
            "m_fall_at_drops": share(q["m_fall_at"], q["drops"])}


def label(M: Optional[float], m410: float) -> str:
    """D1 (a)."""
    if M is None:
        return "refused (no drops)"
    if M >= m410 - MARGIN:
        return "carries"
    return "partly" if M >= 0.5 else "not"


def plant(e: np.ndarray, m: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """The design's plants, held at the series' first state: "mean" holds R (E₁ rebuilt as
    ½ + ½‖m‖² + R₀), "R" holds ‖m‖² (E₁ rebuilt as ½ + ½‖m₀‖² + R, and ‖m‖² = ‖m₀‖² throughout)."""
    e, m = np.asarray(e, float), np.asarray(m, float)
    rest = e - 0.5 - 0.5 * m
    if kind == "mean":
        return 0.5 + 0.5 * m + rest[0], m
    if kind == "R":
        m0 = np.full_like(m, m[0])
        return 0.5 + 0.5 * m0 + rest, m0
    raise ValueError(kind)


# ---------------------------------------------------------------- one model

def read_model(runs: Dict[str, Tuple[List[int], dict]], kind: Optional[str] = None,
               brk: Optional[Dict[str, Optional[int]]] = None) -> dict:
    pooled: Dict[str, int] = {}
    by_step: Dict[int, Dict[str, int]] = {}
    per_run, at_break, rest_min = {}, {}, np.inf
    for name, (steps, recs) in runs.items():
        q_run: Dict[str, int] = {}
        for st in steps:
            for p in V1:
                r = recs[(st, p)]
                e, m = np.asarray(r["e1"], float), m2(r)
                if kind:
                    e, m = plant(e, m, kind)
                x = series(e, m)
                rest_min = min(rest_min, x["rest_min"])
                c = counts(x)
                q_run = add(q_run, c)
                by_step[st] = add(by_step.get(st, {}), c)
                if brk and brk.get(name) == st:
                    at_break[name] = add(at_break.get(name, {}), c)
        per_run[name] = stats(q_run)
        pooled = add(pooled, q_run)
    out = {"pooled": stats(pooled), "rest_min": float(rest_min),
           "per_run": per_run, "by_step": {str(s): stats(q) for s, q in sorted(by_step.items())}}
    if brk is not None:
        out["at_break"] = {k: ({"step": brk[k], **stats(at_break[k])} if brk.get(k) is not None else None)
                           for k in runs}
    return out


# ---------------------------------------------------------------- (2) the known answer

KNOWN_KEYS = ("drops", "m_fall_share_at_drops", "m_fall_share_all", "s_gt_half_share", "same_direction_share",
              "m_fall_share_at_any_e_fall")


def known_answer(runs410: Dict[str, Tuple[List[int], dict]]) -> dict:
    """U1's code on 410m's records against `c2_readings.one_series` record by record, and against
    C2 5b's R6 numbers by step, exactly; returns M₄₁₀ from C2 5b's record."""
    from trackC_writeup.c2_readings import one_series
    for name, (steps, recs) in runs410.items():
        if steps != list(STEPS):
            raise Refused(f"410m {name}: steps {steps} are not Pythia's 18")
        for key, r in recs.items():
            a, b = one_series(r), series(r["e1"], m2(r))
            for k in ("d", "m_fall", "e_fall", "same"):
                if not np.array_equal(a[k], b[k]):
                    raise Refused(f"known answer: {k} differs at 410m {name} {key}")
            if not np.array_equal(a["s"], b["s"], equal_nan=True) or a["net_inner"] != b["net_inner"]:
                raise Refused(f"known answer: s or net_inner differs at 410m {name} {key}")
    rec = json.loads(C2U5B.read_text())
    for s in range(10):
        if rec["inputs"]["sha256"][str(s)] != inputs_sha(runs410_dirs()[f"seed{s}"]):
            raise Refused(f"known answer: 410m seed {s}'s records are not the ones C2 5b read")
    r6 = rec["R6"]["by_step"]
    mine = read_model(runs410)["by_step"]
    num = den = 0
    for st in STEPS:
        q, g = mine[str(st)], r6[str(st)]
        ours = {"drops": q["drops"], "m_fall_share_at_drops": q["m_fall_at_drops"],
                "m_fall_share_all": q["base_rate"], "s_gt_half_share": q["M"],
                "same_direction_share": q["same_direction"], "m_fall_share_at_any_e_fall": q["S_any"]}
        for k in KNOWN_KEYS:
            if ours[k] != g[k]:
                raise Refused(f"known answer: step {st} {k}: U1 {ours[k]} against C2 5b {g[k]}")
        num += round(g["s_gt_half_share"] * g["drops"]) if g["drops"] else 0
        den += g["drops"]
    return {"M410": num / den, "drops": den, "s_gt_half": num, "steps_equal": len(STEPS),
            "record": str(C2U5B.relative_to(DATA)), "record_code": rec["code"]}


def runs410_dirs() -> Dict[str, Path]:
    return models()["pythia-410m"]


# ---------------------------------------------------------------- main

def main() -> int:
    table = models()
    loaded = {g: {k: load_run(d) for k, d in runs.items()} for g, runs in table.items()}   # (1)
    for g, runs in loaded.items():
        grids = {tuple(s) for s, _ in runs.values()}
        if len(grids) != 1:
            raise Refused(f"{g}: its runs are on different grids")
    ka = known_answer(loaded["pythia-410m"])                                               # (2)
    m410 = ka["M410"]
    if read_model(loaded["pythia-410m"])["pooled"]["M"] != m410:
        raise Refused("known answer: U1's pooled M on 410m is not C2 5b's")
    brk = breaks()
    rec = {"code": code_sha(), "design": 'trackC_writeup/design-C9.md "U1"', "decided": "Blocked 178 all (a)",
           "rel_tol": REL_TOL, "margin": MARGIN, "known_answer": ka, "bar": m410 - MARGIN,
           "inputs": {g: {k: {"dir": str(d.relative_to(DATA)), "sha256": inputs_sha(d),
                              "steps": len(loaded[g][k][0]), "states": len(loaded[g][k][1][(loaded[g][k][0][0], V1[0])]["e1"]),
                              "codes": sorted({r["code"] for r in loaded[g][k][1].values()})}
                          for k, d in runs.items()} for g, runs in table.items()},
           "models": {}}
    for g in table:                                     # 410m first: M₄₁₀ read before any other model
        runs = loaded[g]
        plants = {kind: read_model(runs, kind)["pooled"] for kind in ("mean", "R")}         # (3)
        why = []
        if plants["mean"]["M"] != 1.0 or plants["mean"]["S_any"] != 1.0:
            why.append(f"planted mean reads M {plants['mean']['M']}, S_any {plants['mean']['S_any']}")
        if plants["R"]["M"] != 0.0 or label(plants["R"]["M"], m410) != "not":
            why.append(f"planted R reads M {plants['R']['M']}")
        real = read_model(runs, None, brk[g])
        if real["rest_min"] < -1e-9:
            why.append(f"R < 0 ({real['rest_min']:.2e}): E₁'s identity broken on these records")
        entry = {"plants": plants, "runs": len(runs)}
        if why:
            entry.update({"label": "refused", "why": why})
        else:
            p = real["pooled"]
            entry.update({"M": p["M"], "S_any": p["S_any"], "label": label(p["M"], m410),
                          "S_any_against_0.96": "no label (design D1 (a))",
                          "beside": {"pooled": p, "rest_min": real["rest_min"], "per_run": real["per_run"],
                                     "by_step": real["by_step"], "at_break": real["at_break"],
                                     "S_any_min_step": min((v["S_any"] for v in real["by_step"].values()
                                                            if v["S_any"] is not None), default=None),
                                     "M_min_step": min((v["M"] for v in real["by_step"].values()
                                                        if v["M"] is not None), default=None)}})
        rec["models"][g] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "u1.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"code {rec['code']}; known answer: 18 steps equal, M410 {m410:.4f} on {ka['drops']} drops; bar {m410 - MARGIN:.4f}")
    for g, e in rec["models"].items():
        if e["label"] == "refused":
            print(f"{g:22s} refused: {e['why']}")
        else:
            b = e["beside"]["pooled"]
            print(f"{g:22s} {e['label']:8s} M {e['M']:.4f}  S_any {e['S_any']:.4f}  drops {b['drops']}  "
                  f"base {b['base_rate']:.3f}  min-step S_any {e['beside']['S_any_min_step']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
