"""
trackT_timaeus/t5i.py — T5 U4b: the reading of U4's arms (`design-T5.md` "U4, the causal arm, designed",
frozen 2026-11-03; the rule is `t5h`'s pure functions, fixed before any arm output).

Kept out of `t5h.py` so that writing the reading does not move `t5h.code_sha()`, which every arm's resume
state carries (as `t5g` left `t5b`'s hash alone). Reads the records `t5h read` and `t5h gate` wrote.

Commands:
    replicates   the seed-noise gate (design "Gate", last row), before A1–A3 start: L of A0, A0′, A0″
                 (mean ρ_mid at 3738 and 4369); τ = max(0.02, the largest of the three pairs);
                 bar = max(0.2, 2τ); bar_I = √2 · bar; **readable iff bar < R_A0 / 2**. Refuses unless
                 A0's late gate and A0′'s and A0″'s start gates passed. Beside, no label (Parked 1 on #310):
                 the three's ρ_mid at 1253, 1713, 2002 and their spread (max − min), each one's T, its step
                 and R, the v1 loss at 4369, the codes the records were read at. The replicates differ in
                 the init seed only (one shard, one order: `t5h`'s stream), so τ and the timing spread are
                 the noise between arms, which share that stream; against another run (Dodecahedron) they
                 understate it (/challenge-pr on #311, finding 2).
    arms         U4b's reading at 4369 (design "The statistic and the rule"): per arm A1, A2, A3, ΔL_k = L_A0 − L_k
                 and its label (lowers ≥ R_A0/2, partly ≥ bar, does not, raises ≤ −bar) with τ, bar from the
                 three replicates as ``replicates``; the interaction I = L_A1 + L_A2 − L_A0 − L_A3 at bar_I. H1
                 (A2) and H2 (A1) are final only where the arm does not lower L ("not supported"); an arm that
                 lowers, fully or partly, waits on the delay check (its continuation to 5970, the next stage),
                 so its H label reads "pending the delay check". Refuses unless the replicates' gates passed,
                 the seed-noise gate is readable, each arm's start gate passed on its current records, and A1
                 = A0, A3 = A2 at step 0 on the current records. Beside, no label: each ΔL_k's 7-prompt
                 bootstrap (U2's: one resample shared across arms and steps, 1,000, ``default_rng(20261030)``),
                 T_k, its step, R_k, the step-0 ρ_mid, ρ_mid at the rise's steps, the v1 loss at 4369.
    step0        the gate row "A1 is A0 at step 0, A3 is A2 at step 0: ρ_ℓ equal, ≤ 1e-6" (design "Gate"), on
                 the arm's and its base's step-0 records; writes ``<arm>/gate_step0_same.json``; exit 3 if not.

Tier 1, description (nothing here is registered).
    python -m trackT_timaeus.t5i replicates          # exit 0 readable, 3 not (back to the user)
    python -m trackT_timaeus.t5i step0 --arm A1      # exit 0 equal to its base at step 0, 3 not
    python -m trackT_timaeus.t5i arms                # writes arms.json; exit 3 if a gate refuses
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from typing import Dict, Optional, Sequence

from trackA_force.a1a import REPO
from trackT_timaeus import t5h

#: the rise's steps where A0 led Dodecahedron (`status-T.md` "T5 U4a: Blocked 133 decided; A0"); the
#: replicates' spread there is the noise scale for any timing read (Parked 1 on #310, finding 2)
TIMING_STEPS = (1253, 1713, 2002)


def own_sha() -> str:
    return hashlib.sha256((REPO / "trackT_timaeus/t5i.py").read_bytes()).hexdigest()[:12]


def replicates_read(series: Dict[str, Dict[int, float]], v1: Optional[Dict[str, Dict[int, float]]] = None) -> dict:
    """The seed-noise gate on the three replicates' ρ_mid series (pure). ``series[arm][step]``."""
    reps = t5h.REPLICATES
    miss = [a for a in reps if a not in series]
    if miss:
        raise t5h.T5hRefused(f"replicates missing: {miss}")
    arms = {}
    for a in reps:
        L = t5h.late_level(series[a])
        T, ts = t5h.trough(series[a])
        arms[a] = {"L": L, "T": T, "T_step": ts, "R": L - T}
    levels = [arms[a]["L"] for a in reps]
    tau, bar = t5h.tau_bar(*levels)
    r_a0 = arms["A0"]["R"]
    out = {
        "arms": arms,
        "pairs": {f"{x}-{y}": arms[x]["L"] - arms[y]["L"] for x, y in itertools.combinations(reps, 2)},
        "tau": tau, "bar": bar, "bar_I": math.sqrt(2) * bar, "R_A0": r_a0, "half_R_A0": r_a0 / 2,
        "readable": t5h.readable(r_a0, bar),
        "labels_in_use": (["lowers", "partly", "does not", "raises"] if t5h.readable(r_a0, bar)
                          else ["lowers", "does not", "raises"]),
    }
    beside = {}
    for s in TIMING_STEPS:
        vals = {a: series[a].get(s) for a in reps}
        got = [v for v in vals.values() if v is not None]
        beside[str(s)] = {"rho_mid": vals, "spread": (max(got) - min(got)) if len(got) == len(reps) else None}
    out["beside_timing"] = beside
    if v1 is not None:
        out["beside_v1_loss_4369"] = {a: v1.get(a, {}).get(t5h.READ_STEP) for a in reps}
    return out


def _gate_pass(root: Path, arm: str, stage: str, recs: Dict[int, dict]) -> bool:
    """A saved gate counts only if it passed and was read on these records: every ρ_mid in its stored
    ``series`` equals the record's now, so a ``t5h read --force`` after the gate refuses here instead of
    carrying the old pass (CodeRabbit on #311)."""
    p = root / arm / f"gate_{stage}.json"
    if not p.exists():
        return False
    g = json.loads(p.read_text())
    if not g["pass"]:
        return False
    series = g.get("series") or {}
    stale = [s for s, v in series.items() if int(s) not in recs or recs[int(s)]["rho_mid"] != v]
    if not series or stale:
        raise t5h.T5hRefused(f"{arm} gate_{stage}: not read on the current records "
                             f"({'no series' if not series else f'differs at steps {stale[:5]}'}); re-gate")
    return True


def replicates(a) -> int:
    root = Path(a.root)
    recs = {x: t5h.records(root, x) for x in t5h.REPLICATES}
    need = [("A0", "late")] + [(x, "start") for x in t5h.REPLICATES if x != "A0"]
    bad = [f"{x} {st}" for x, st in need if not _gate_pass(root, x, st, recs[x])]
    if bad:
        raise t5h.T5hRefused(f"gates not passed: {bad}")
    series = {x: {s: r["rho_mid"] for s, r in rs.items()} for x, rs in recs.items()}
    v1 = {x: {s: r["v1_loss_mean"] for s, r in rs.items()} for x, rs in recs.items()}
    res = replicates_read(series, v1)
    res.update({"design": "trackT_timaeus/design-T5.md, U4, Gate (last row)", "t5i": own_sha(),
                "t5h_now": t5h.code_sha(),
                "read_codes": {x: sorted({r["code"] for r in rs.values()}) for x, rs in recs.items()},
                "pass": res["readable"]})
    t5h._jdump(root / "replicates.json", res)
    print(json.dumps({k: res[k] for k in ("tau", "bar", "bar_I", "R_A0", "half_R_A0", "readable")}))
    print(json.dumps({x: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
                      for x, d in res["arms"].items()}))
    print(json.dumps(res["beside_timing"]))
    return 0 if res["readable"] else 3


#: an arm with biases and the arm it equals at step 0 (design "The arms": zero biases on the base's weights)
BASE = {"A1": "A0", "A3": "A2"}


def step0_same(rho: Sequence[float], rho_base: Sequence[float], tol: float = t5h.KNOWN_TOL) -> dict:
    """The step-0 equality row (pure): max |Δρ_ℓ| over the states, at ``tol``."""
    if len(rho) != len(rho_base):
        raise t5h.T5hRefused(f"step 0: {len(rho)} states against {len(rho_base)}")
    d = max(abs(x - y) for x, y in zip(rho, rho_base))
    return {"max_abs_diff": d, "tol": tol, "pass": d <= tol}


def step0(a) -> int:
    root = Path(a.root)
    base = BASE[a.arm]
    recs = {x: t5h.records(root, x).get(0) for x in (a.arm, base)}
    miss = [x for x, r in recs.items() if r is None]
    if miss:
        raise t5h.T5hRefused(f"step-0 record missing: {miss}")
    res = {"arm": a.arm, "base": base, **step0_same(recs[a.arm]["rho"], recs[base]["rho"]),
           "rho": recs[a.arm]["rho"], "rho_base": recs[base]["rho"], "t5i": own_sha(), "t5h_now": t5h.code_sha()}
    t5h._jdump(root / a.arm / "gate_step0_same.json", res)
    print(json.dumps({k: res[k] for k in ("arm", "base", "max_abs_diff", "tol", "pass")}))
    return 0 if res["pass"] else 3


CHANGE_ARMS = ("A1", "A2", "A3")
HYP = {"A1": "H2 (biases)", "A2": "H1 through training (Pythia's init)",
       "A3": "biases and Pythia's init together"}


def _mid(P: np.ndarray) -> np.ndarray:
    """ρ_mid from per-prompt ρ_ℓ (…, prompts, states): the prompts' mean, then the median over the middle
    states (``t5c.mid_series``, which the records' ``rho_mid`` equals exactly)."""
    return np.median(P.mean(-2)[..., 1:-1], -1)


def boot_dl(pp: Dict[str, Dict[int, np.ndarray]], arms: Sequence[str], n_boot: int = t5h.N_BOOT,
            seed: int = t5h.BOOT_SEED) -> Dict[str, dict]:
    """ΔL_k's 7-prompt bootstrap: one resample of the prompts shared across arms and steps (U2's)."""
    n = len(pp["A0"][t5h.L_STEPS[0]])
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    def L(a):
        return np.mean([_mid(np.asarray(pp[a][s], np.float64)[idx]) for s in t5h.L_STEPS], 0)
    l0 = L("A0")
    out = {}
    for a in arms:
        d = l0 - L(a)
        out[a] = {"p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)), "n_boot": n_boot}
    return out


def arms_read(series: Dict[str, Dict[int, float]], step0_rho: Dict[str, Sequence[float]],
              v1: Optional[Dict[str, Dict[int, float]]] = None,
              pp: Optional[Dict[str, Dict[int, np.ndarray]]] = None) -> dict:
    """The reading at 4369 on the six arms' ρ_mid series (pure). ``step0_rho[arm]`` is its step-0 ρ_ℓ."""
    miss = [a for a in t5h.REPLICATES + CHANGE_ARMS if a not in series]
    if miss:
        raise t5h.T5hRefused(f"arms missing: {miss}")
    rep = replicates_read({a: series[a] for a in t5h.REPLICATES})
    if not rep["readable"]:
        raise t5h.T5hRefused(f"the seed-noise gate is not readable (bar {rep['bar']:.3f} >= R_A0/2 "
                             f"{rep['half_R_A0']:.3f}); the rule reads only lowers / does not / raises: back to the user")
    same = {a: step0_same(step0_rho[a], step0_rho[b]) for a, b in BASE.items()}
    bad = [a for a, g in same.items() if not g["pass"]]
    if bad:
        raise t5h.T5hRefused(f"step-0 equality fails: {bad}")
    bar, r_a0 = rep["bar"], rep["R_A0"]
    l_a0 = rep["arms"]["A0"]["L"]
    L = {a: t5h.late_level(series[a]) for a in t5h.REPLICATES + CHANGE_ARMS}
    out_arms = {}
    for a in CHANGE_ARMS:
        T, ts = t5h.trough(series[a])
        dl = l_a0 - L[a]
        lab = t5h.arm_label(dl, r_a0, bar)
        lowers = lab in ("lowers", "partly")
        out_arms[a] = {"hypothesis": HYP[a], "L": L[a], "dL": dl, "label": lab, "T": T, "T_step": ts,
                       "R": L[a] - T, "continues_to_5970": lowers,
                       "hyp_label": ("pending the delay check" if lowers else
                                     ("not supported" if a != "A3" else lab))}
    i, ilab = t5h.interaction(L, bar)
    lowering = [a for a in CHANGE_ARMS if out_arms[a]["continues_to_5970"]]
    res = {
        "replicates": rep, "bar": bar, "bar_I": rep["bar_I"], "R_A0": r_a0, "half_R_A0": rep["half_R_A0"],
        "L_A0": l_a0, "arms": out_arms, "interaction": {"I": i, "label": ilab},
        "neither_alone_together": (out_arms["A1"]["label"] in ("does not", "raises")
                                   and out_arms["A2"]["label"] in ("does not", "raises")
                                   and out_arms["A3"]["continues_to_5970"]),
        "step0_same": same,
        "continuations": ["A0"] + lowering + ["A0pp"],      # design "Throughput": A0's first, A0″'s last
    }
    beside = {a: {"step0_rho_mid": series[a].get(0),
                  "timing": {str(s): series[a].get(s) for s in TIMING_STEPS}} for a in ("A0",) + CHANGE_ARMS}
    if v1 is not None:
        for a in beside:
            beside[a]["v1_loss_4369"] = v1.get(a, {}).get(t5h.READ_STEP)
    if pp is not None:
        for a, b in boot_dl(pp, CHANGE_ARMS).items():
            beside[a]["dL_boot"] = b
    res["beside"] = beside
    return res


def arms(a) -> int:
    root = Path(a.root)
    every = t5h.REPLICATES + CHANGE_ARMS
    recs = {x: t5h.records(root, x) for x in every}
    need = [("A0", "late")] + [(x, "start") for x in every if x != "A0"]
    bad = [f"{x} {st}" for x, st in need if not _gate_pass(root, x, st, recs[x])]
    if bad:
        raise t5h.T5hRefused(f"gates not passed: {bad}")
    for x in every:
        if t5h.READ_STEP not in recs[x] or any(s not in recs[x] for s in t5h.L_STEPS + (0,)):
            raise t5h.T5hRefused(f"{x}: the late steps or step 0 unread")
    series = {x: {s: r["rho_mid"] for s, r in rs.items()} for x, rs in recs.items()}
    v1 = {x: {s: r["v1_loss_mean"] for s, r in rs.items()} for x, rs in recs.items()}
    pp = {x: {s: np.asarray(recs[x][s]["rho_per_prompt"]) for s in t5h.L_STEPS} for x in ("A0",) + CHANGE_ARMS}
    res = arms_read(series, {x: recs[x][0]["rho"] for x in ("A0", "A2", "A1", "A3")}, v1, pp)
    res.update({"design": "trackT_timaeus/design-T5.md, U4, The statistic and the rule", "t5i": own_sha(),
                "t5h_now": t5h.code_sha(),
                "read_codes": {x: sorted({r["code"] for r in rs.values()}) for x, rs in recs.items()}})
    t5h._jdump(root / "arms.json", res)
    print(json.dumps({k: res[k] for k in ("L_A0", "bar", "bar_I", "R_A0", "half_R_A0", "interaction",
                                          "neither_alone_together", "continuations")}))
    print(json.dumps({x: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
                      for x, d in res["arms"].items()}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="t5i")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replicates")
    p.add_argument("--root", default=str(t5h.ROOT))
    p = sub.add_parser("step0")
    p.add_argument("--arm", required=True, choices=sorted(BASE))
    p.add_argument("--root", default=str(t5h.ROOT))
    p = sub.add_parser("arms")
    p.add_argument("--root", default=str(t5h.ROOT))
    a = ap.parse_args(argv)
    return {"replicates": replicates, "step0": step0, "arms": arms}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
