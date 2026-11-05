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
    step0        the gate row "A1 is A0 at step 0, A3 is A2 at step 0: ρ_ℓ equal, ≤ 1e-6" (design "Gate"), on
                 the arm's and its base's step-0 records; writes ``<arm>/gate_step0_same.json``; exit 3 if not.

Tier 1, description (nothing here is registered).
    python -m trackT_timaeus.t5i replicates          # exit 0 readable, 3 not (back to the user)
    python -m trackT_timaeus.t5i step0 --arm A1      # exit 0 equal to its base at step 0, 3 not
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="t5i")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replicates")
    p.add_argument("--root", default=str(t5h.ROOT))
    p = sub.add_parser("step0")
    p.add_argument("--arm", required=True, choices=sorted(BASE))
    p.add_argument("--root", default=str(t5h.ROOT))
    a = ap.parse_args(argv)
    return {"replicates": replicates, "step0": step0}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
