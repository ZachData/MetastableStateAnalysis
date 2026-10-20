"""
tools/design_preflight.py — can this design reject at all? Asked before it freezes.

    python -m tools.design_preflight values.json --out report.json

WHY THIS EXISTS

`LESSONS.md` 6 ("Statistical designs that could not have rejected") holds the
same failure several ways, each found after a gate or a run had been spent on it.
Five have a shape this module checks:

  * **a band nearly everything leaves** (B2's design, 2026-10-17): step 0's
    2.5–97.5 % band of a head's attractive share was 0.4963–0.5037 at d = 1024,
    and 99.7 % of trained heads sat outside it, so members and random sets
    both scored about 1 and tied;
  * **a control already at its ceiling** (A2 unit 5, 2026-10-17): the rule
    needed the planted arm to *raise* a share that was already 1.000 unablated;
  * **a dose below the null's mean** (B2's re-gate, 2026-10-18): with the
    two-sided |p − ½| the null's mean is exactly ¼, so a set held at the 70th
    percentile scores 0.20 and reads as *less* than chance at any power;
  * **a floor below the threshold** (`P-S1`, `docs/PHASE_SYNTHESIS.md`): 500
    null draws put the best rank p at 1/501, so E could never reach 1/α = 20;
  * **a largest plant the design cannot see** (the shape of A3 unit 2's effect
    arm; its cause there, calibration noise shaped unlike the data, is not
    checked here).

Each was visible from one read of the statistic on the right population before
the rule was written. This module is that read, made mechanical. It takes the
statistic's values on four kinds of population:

  * its nulls (step 0, matched random weights, ...);
  * the trained population at one trained step that is **not** among the
    design's scored cells;
  * a planted control, unplanted (dose 0);
  * the planted control at three or more doses.

It reports the attainable floor and the detection curve, and **refuses** (exit 1)
on any of the shapes above. A design cites the report's sha256 before it freezes
(`docs/ROADMAP.md` §5).

WHAT IT TAKES

Values, not models. The statistic is computed by the design's own code on its
own populations (`evaluate` is a convenience for a callable), so this module
never needs torch and runs in the pure tier. The JSON input, every key named
(an unknown key is refused, so a misspelling cannot switch a check off):

    {"statistic": "...", "direction": "greater" | "less",
     "nulls": {"step0": [...], "random": [...]},
     "moved_marks": ["step0"],          # nulls declared, before output, as
                                        # "moved" marks rather than nulls ([] if none)
     "trained": [...],                  # per-unit values (heads, sets, ...) at
     "trained_step": "8000",            # one trained step, at least 20 of them,
     "scored_steps": ["0", "512"],      # and that step not among the scored cells
     "planted": {"0": [...], "1": [...], "2": [...], "3": [...]},
     "bounds": [0, 1] | null,           # the statistic's range, or null: unbounded
     "n_null_draws": 2000,              # optional; or "p_floor": the test's best p
     "decision_p": 0.05,                # optional: the p the design decides at
     "alpha": 0.05, "kappa": 0.5}       # optional: default to the registry's

`direction` is the side a real effect moves the statistic to. A two-sided
question is put to it one-sided (e.g. |p − ½|, "greater"), which is exactly
the case where the dose check matters.

**The decision level.** A planted value counts as detected when its rank p
against a null is at most `decision_p`. By default that is the p a single
factor needs to reach E ≥ 1/α, (ακ)^(1/(1−κ)) = 6.25e-4 at α 0.05, κ 0.5, the
same bar the floor check uses; at α a +3σ dose looks seen 94 % of the time and
crosses the E bar about 41 % of the time (`/challenge-pr` on #242, finding 1).
A design that decides at another level (an exploratory read at α, a combined
statistic) says so with `decision_p`, and the report records it.

WHAT IT DOES NOT DO

It does not check that a null is *valid* (exchangeable, matched, the right
population); it checks only that, granting validity, the design can reject.
It does not check that "matched random weights" are matched, or that a planted
control's noise has the data's shape (A3 unit 2), or a planted case's
generality (B2a), or a comparator's bias (E1m, E1p). It does not choose doses:
the design names them, and a dose the report refuses stays visible, labelled,
rather than being dropped.

Dependencies: numpy, scipy (pure tier), `core.evalues.calibrate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import beta as _beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evalues import calibrate  # noqa: E402

#: "Nearly everything": the share of the trained population outside a null's
#: central band at which the band stops being a null (B2: 99.7 %).
NEARLY = 0.9
#: A control whose dose-0 median sits within this share of the statistic's
#: range from its bound, in the effect's direction, has no headroom (A2: 0).
MIN_HEADROOM = 0.05
#: The largest planted dose must be detected at least this often.
MIN_POWER = 0.8
MIN_DOSES = 3
#: The trained population must be a population: per-unit values, not one score.
MIN_TRAINED = 20
#: The size check's confidence: a correct control is refused about 0.1 % of the
#: time (`/challenge-pr` on #242, finding 5: at 0.95 it was 5-7 %).
SIZE_CONF = 0.999

#: Marks a required field left out (`None` is a value: an unbounded statistic).
UNSET = "<unset>"
REQUIRED = ("statistic", "direction", "nulls", "trained", "trained_step", "scored_steps",
            "planted", "bounds", "moved_marks")


@dataclass
class Inputs:
    statistic: str = UNSET
    direction: str = UNSET
    nulls: object = UNSET
    trained: object = UNSET
    trained_step: object = UNSET
    scored_steps: object = UNSET
    planted: object = UNSET
    bounds: object = UNSET
    moved_marks: object = UNSET
    n_null_draws: Optional[int] = None
    p_floor: Optional[float] = None
    decision_p: Optional[float] = None
    alpha: Optional[float] = None
    kappa: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Inputs":
        """Unknown keys are kept aside and refused by `preflight`, not dropped."""
        known = {f.name for f in fields(cls)}
        inp = cls(**{k: v for k, v in d.items() if k in known})
        inp._unknown = sorted(set(d) - known)
        return inp


def evaluate(statistic: Callable[[object], float],
             populations: Dict[str, Iterable[object]]) -> Dict[str, List[float]]:
    """The statistic on each population's draws, as the lists `Inputs` takes."""
    return {name: [float(statistic(x)) for x in draws] for name, draws in populations.items()}


def _registry_alpha_kappa():
    try:
        reg = json.loads((ROOT / "claims" / "registry.json").read_text(encoding="utf-8"))
        return float(reg.get("alpha", 0.05)), float(reg.get("kappa", 0.5))
    except Exception:                          # pragma: no cover - defensive
        return 0.05, 0.5


def e_bar_p(alpha: float, kappa: float) -> float:
    """The p one factor needs for E ≥ 1/α: (ακ)^(1/(1−κ))."""
    return (alpha * kappa) ** (1.0 / (1.0 - kappa))


def min_null_draws(alpha: float, kappa: float) -> int:
    """
    The fewest null draws whose best rank p, 1/(n + 1), calibrates to E ≥ 1/α.

    From kappa * (n + 1) ** (1 - kappa) ≥ 1/α; at α 0.05, κ 0.5 that is
    n ≥ 1599 (`tools/math_checks/design_preflight_floor.py`).
    """
    return max(0, math.ceil(1.0 / e_bar_p(alpha, kappa) - 1 - 1e-9))


def rank_p(x: np.ndarray, null: np.ndarray, s: float) -> np.ndarray:
    """One-sided rank p of each x against the null: (1 + #null at or beyond x) / (n + 1)."""
    y, z = np.sort(s * null), s * np.asarray(x, dtype=float)
    at_or_beyond = y.size - np.searchsorted(y, z, side="left")
    return (1.0 + at_or_beyond) / (y.size + 1.0)


def _cp_lower(k: int, n: int, conf: float) -> float:
    """Clopper–Pearson one-sided lower bound on a rate."""
    return 0.0 if k == 0 else float(_beta.ppf(1 - conf, k, n - k + 1))


def _tool_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def preflight(inp: Inputs) -> dict:
    """The report. `report["refusals"]` is empty exactly when the design passes."""
    reg_alpha, reg_kappa = _registry_alpha_kappa()
    alpha = float(inp.alpha if inp.alpha is not None else reg_alpha)
    kappa = float(inp.kappa if inp.kappa is not None else reg_kappa)
    refusals: List[dict] = []

    def refuse(code: str, msg: str) -> None:
        refusals.append({"code": code, "reason": msg})

    def done(**parts) -> dict:
        return _report(inp, alpha, kappa, refusals, **parts)

    # -- the input's shape: nothing silently defaulted ------------------------
    unknown = getattr(inp, "_unknown", [])
    if unknown:
        refuse("unknown_keys", f"unknown keys {unknown}: a misspelt key would switch a check off "
                               f"(known: {sorted(f.name for f in fields(Inputs))})")
    missing = [k for k in REQUIRED if getattr(inp, k) is UNSET]
    if missing:
        refuse("missing", f"required keys not given: {missing} (`bounds` may be null, for an "
                          f"unbounded statistic; `moved_marks` may be [])")
    if inp.direction not in ("greater", "less"):
        refuse("direction", f"direction must be 'greater' or 'less', got {inp.direction!r}")
    if refusals:
        return done()
    s = 1.0 if inp.direction == "greater" else -1.0

    arrays = {f"null:{k}": v for k, v in inp.nulls.items()}
    arrays.update({"trained": inp.trained}, **{f"dose:{k}": v for k, v in inp.planted.items()})
    clean: Dict[str, np.ndarray] = {}
    for name, v in arrays.items():
        a = np.asarray(v, dtype=float)
        if a.size == 0 or not np.all(np.isfinite(a)):
            refuse("empty", f"{name} is empty or holds non-finite values; nothing can be read off it")
        clean[name] = a
    if not inp.nulls:
        refuse("empty", "no null population given")
    if any(r["code"] == "empty" for r in refusals):
        return done()
    bad_marks = sorted(set(inp.moved_marks) - set(inp.nulls))
    if bad_marks:
        refuse("moved_marks", f"moved_marks names {bad_marks}, which are not among the nulls")
    zero = next((k for k in inp.planted if float(k) == 0), None)
    doses = sorted((k for k in inp.planted if float(k) != 0), key=float)
    if zero is None:
        refuse("doses", "no dose 0: the planted control is never read unplanted, so its headroom "
                        "and its size cannot be checked")
    if len(doses) < MIN_DOSES:
        refuse("doses", f"{len(doses)} non-zero doses; a detection curve needs at least {MIN_DOSES}")
    trained = clean["trained"]
    if trained.size < MIN_TRAINED:
        refuse("trained", f"the trained population has {trained.size} values (< {MIN_TRAINED}); it "
                          f"is the per-unit values at one step, not the design's own score")
    if str(inp.trained_step) in {str(x) for x in inp.scored_steps}:
        refuse("family", f"trained step {inp.trained_step} is among the scored steps "
                         f"{list(inp.scored_steps)}: reading it now reads a scored cell before the "
                         f"freeze (CLAUDE.md 'Name the family'); take a step off the scored grid")

    # -- the floor and the decision level --------------------------------------
    if inp.p_floor is not None:
        p_floor, floor_from = float(inp.p_floor), "p_floor as given"
    elif inp.n_null_draws is not None:
        p_floor = 1.0 / (int(inp.n_null_draws) + 1)
        floor_from = f"rank p against {int(inp.n_null_draws)} null draws, 1/(n + 1)"
    else:
        p_floor = 1.0 / (min(clean[f"null:{k}"].size for k in inp.nulls) + 1)
        floor_from = "rank p against the smallest null given, 1/(n + 1)"
    max_e = calibrate(p_floor, kappa)
    floor = {"p_floor": p_floor, "from": floor_from, "max_attainable_E": max_e,
             "threshold": 1.0 / alpha, "min_null_draws": min_null_draws(alpha, kappa)}
    if max_e < 1.0 / alpha:
        refuse("floor", f"the best p ({p_floor:.3g}, {floor_from}) calibrates to E {max_e:.3g} < "
                        f"1/α = {1 / alpha:g}: no outcome can cross; a rank test needs ≥ "
                        f"{floor['min_null_draws']} null draws")
    decision = ({"p": float(inp.decision_p), "from": "decision_p as given"}
                if inp.decision_p is not None else
                {"p": e_bar_p(alpha, kappa), "from": "one factor's E ≥ 1/α, (ακ)^(1/(1−κ))"})

    # -- each null, and where the trained population falls in it ---------------
    nulls: Dict[str, dict] = {}
    for name in inp.nulls:
        x = clean[f"null:{name}"]
        lo, hi = np.quantile(x, [alpha / 2, 1 - alpha / 2])
        outside = float(np.mean((trained < lo) | (trained > hi)))
        role = "moved_mark" if name in inp.moved_marks else "null"
        nulls[name] = {"role": role, "n": int(x.size), "mean": float(x.mean()),
                       "band": [float(lo), float(hi)], "trained_outside_band": outside}
        if role == "null" and outside >= NEARLY:
            refuse(f"band:{name}",
                   f"{outside:.1%} of the trained population at step {inp.trained_step} sits "
                   f"outside {name}'s central band [{lo:.4g}, {hi:.4g}]; a band nearly everything "
                   f"leaves is a 'moved' mark, not a null (declare it in moved_marks, or use "
                   f"another null)")
    real = [n for n, v in nulls.items() if v["role"] == "null"]
    if not real:
        refuse("band", "every null is declared a moved mark; nothing is left to test against")

    # -- the planted control: headroom, size, each dose's side, the curve -----
    control: dict = {}
    if zero is not None:
        z = clean[f"dose:{zero}"]
        med = float(np.median(z))
        control["dose0_median"] = med
        if inp.bounds is not None:
            lo_b, hi_b = map(float, inp.bounds)
            room = ((hi_b - med) if s > 0 else (med - lo_b)) / (hi_b - lo_b)
            control["headroom"] = room
            if room <= MIN_HEADROOM:
                refuse("ceiling", f"the planted control reads {med:.4g} unplanted, {room:.1%} of the "
                                  f"range from its bound in the effect's direction; a readout at its "
                                  f"ceiling cannot be raised by any plant")
        else:
            control["headroom"] = None
            control["headroom_note"] = "bounds null (declared unbounded): headroom not checked"
        control["dose0_rate_at_alpha"] = {}
        for n in real:
            k = int(np.sum(rank_p(z, clean[f"null:{n}"], s) <= alpha))
            control["dose0_rate_at_alpha"][n] = k / z.size
            if _cp_lower(k, int(z.size), SIZE_CONF) > alpha:
                refuse("size", f"the planted control unplanted reaches p ≤ α against {n} in "
                               f"{k / z.size:.1%} of draws ({SIZE_CONF:.1%} lower bound above "
                               f"α = {alpha:g}); it reads an effect with nothing planted")
    curve: Dict[str, dict] = {}
    for d in doses:
        x = clean[f"dose:{d}"]
        row = {"mean": float(x.mean()), "n": int(x.size),
               "detected": {n: float(np.mean(rank_p(x, clean[f"null:{n}"], s) <= decision["p"]))
                            for n in real}}
        below = [n for n in real if s * (row["mean"] - nulls[n]["mean"]) <= 0]
        row["below_null_mean"] = below
        if below:
            refuse(f"dose:{d}", f"dose {d}'s mean ({row['mean']:.4g}) is not beyond the null mean of "
                                f"{below} in the effect's direction; its detection rate measures the "
                                f"statistic's floor, not power")
        curve[d] = row
    if doses and real:
        worst = min(curve[doses[-1]]["detected"].values())
        if worst < MIN_POWER:
            refuse("power", f"the largest dose ({doses[-1]}) reaches p ≤ {decision['p']:.3g} "
                            f"({decision['from']}) in {worst:.1%} of draws against the strictest "
                            f"null (< {MIN_POWER:.0%}); the design cannot see its own largest plant")
    return done(floor=floor, decision=decision, nulls=nulls, control=control, curve=curve)


def _report(inp, alpha, kappa, refusals, floor=None, decision=None, nulls=None, control=None,
            curve=None) -> dict:
    return {"statistic": inp.statistic, "direction": inp.direction,
            "trained_step": inp.trained_step, "scored_steps": inp.scored_steps,
            "alpha": alpha, "kappa": kappa, "tool_sha256": _tool_sha(),
            "thresholds": {"nearly": NEARLY, "min_headroom": MIN_HEADROOM, "min_power": MIN_POWER,
                           "min_doses": MIN_DOSES, "min_trained": MIN_TRAINED,
                           "size_conf": SIZE_CONF},
            "floor": floor or {}, "decision": decision or {}, "nulls": nulls or {},
            "control": control or {}, "detection_curve": curve or {},
            "refusals": refusals, "verdict": "refused" if refusals else "pass"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Can this design reject? (LESSONS.md 6)")
    ap.add_argument("values", help="JSON of the statistic's values (see the module docstring)")
    ap.add_argument("--out", help="write the report here; a design cites its sha256")
    a = ap.parse_args(argv)
    raw = Path(a.values).read_bytes()
    report = preflight(Inputs.from_dict(json.loads(raw)))
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(report, indent=1, default=list) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]})")
    f, dec = report["floor"], report["decision"]
    if f:
        print(f"floor: p {f['p_floor']:.3g} -> max E {f['max_attainable_E']:.3g} "
              f"(threshold {f['threshold']:g}); deciding at p <= {dec['p']:.3g} ({dec['from']})")
    for d, row in report["detection_curve"].items():
        print(f"dose {d}: " + ", ".join(f"{n} {v:.2f}" for n, v in row["detected"].items()))
    for r in report["refusals"]:
        print(f"REFUSED [{r['code']}] {r['reason']}")
    print(report["verdict"])
    return 1 if report["refusals"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
