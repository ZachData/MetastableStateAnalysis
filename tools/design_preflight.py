"""
tools/design_preflight.py — can this design reject at all? Asked before it freezes.

    python -m tools.design_preflight values.json --out report.json

WHY THIS EXISTS

`LESSONS.md` 6 ("Statistical designs that could not have rejected") holds the
same failure five ways, each found after a gate or a run had been spent on it:

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
  * **an effect arm that cannot see its own largest plant** (A3 unit 2,
    2026-10-19).

Each was visible from one read of the statistic on the right population before
the rule was written. This module is that read, made mechanical: given the
statistic's values on its nulls (step 0, matched random weights, ...), on the
trained population at one trained step, and on a planted control at dose 0 and
at least three doses, it reports the attainable floor and the detection curve,
and **refuses** (exit 1) on any of the shapes above. A design cites the
report's sha256 before it freezes (`docs/ROADMAP.md` §5).

WHAT IT TAKES

Values, not models. The statistic is computed by the design's own code on its
own populations (`evaluate` is a convenience for a callable), so this module
never needs torch and runs in the pure tier. The JSON input:

    {"statistic": "...", "direction": "greater" | "less",
     "nulls": {"step0": [...], "random": [...]},
     "moved_marks": ["step0"],          # optional: nulls declared, before output,
                                        # as "moved" marks rather than nulls
     "trained": [...], "trained_step": "512",
     "planted": {"0": [...], "1": [...], "2": [...], "3": [...]},
     "bounds": [0, 1],                  # optional: the statistic's range
     "n_null_draws": 2000,              # or "p_floor": the test's best p
     "alpha": 0.05, "kappa": 0.5}       # optional: default to the registry's

`direction` is the side a real effect moves the statistic to. A two-sided
question is put to it one-sided (e.g. |p − ½|, "greater"), which is exactly
the case where the dose check matters.

WHAT IT DOES NOT DO

It does not check that a null is *valid* (exchangeable, matched, the right
population); it checks only that, granting validity, the design can reject.
It does not choose doses: the design names them, and a dose the report
refuses stays visible, labelled, rather than being dropped.

Dependencies: numpy, scipy (pure tier), `core.evalues.calibrate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


@dataclass
class Inputs:
    statistic: str
    direction: str
    nulls: Dict[str, List[float]]
    trained: List[float]
    planted: Dict[str, List[float]]
    trained_step: str = ""
    moved_marks: Sequence[str] = ()
    bounds: Optional[Tuple[float, float]] = None
    n_null_draws: Optional[int] = None
    p_floor: Optional[float] = None
    alpha: Optional[float] = None
    kappa: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Inputs":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        b = d.get("bounds")
        return cls(**{k: v for k, v in d.items() if k in known and k != "bounds"},
                   bounds=tuple(b) if b is not None else None,
                   extra={k: v for k, v in d.items() if k not in known})


def evaluate(statistic: Callable[[object], float],
             populations: Dict[str, Iterable[object]]) -> Dict[str, List[float]]:
    """The statistic on each population's draws, as the lists `Inputs` takes."""
    return {name: [float(statistic(x)) for x in draws] for name, draws in populations.items()}


def _registry_alpha_kappa() -> Tuple[float, float]:
    try:
        reg = json.loads((ROOT / "claims" / "registry.json").read_text(encoding="utf-8"))
        return float(reg.get("alpha", 0.05)), float(reg.get("kappa", 0.5))
    except Exception:                          # pragma: no cover - defensive
        return 0.05, 0.5


def min_null_draws(alpha: float, kappa: float) -> int:
    """
    The fewest null draws whose best rank p, 1/(n + 1), calibrates to E ≥ 1/α.

    From kappa * (n + 1) ** (1 - kappa) ≥ 1/α; at α 0.05, κ 0.5 that is
    n ≥ 1599 (`tools/math_checks/design_preflight_floor.py`).
    """
    return max(0, math.ceil((1.0 / (alpha * kappa)) ** (1.0 / (1.0 - kappa)) - 1 - 1e-9))


def _sign(direction: str) -> float:
    if direction not in ("greater", "less"):
        raise ValueError(f"direction must be 'greater' or 'less', got {direction!r}")
    return 1.0 if direction == "greater" else -1.0


def _cut(null: np.ndarray, alpha: float, s: float) -> float:
    """The null's one-sided (1 − α) cut on the effect's side."""
    return float(np.quantile(null, 1 - alpha if s > 0 else alpha))


def _beyond(x: np.ndarray, cut: float, s: float) -> np.ndarray:
    return (x > cut) if s > 0 else (x < cut)


def _cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper–Pearson one-sided lower bound on a rate."""
    return 0.0 if k == 0 else float(_beta.ppf(1 - conf, k, n - k + 1))


def preflight(inp: Inputs) -> dict:
    """The report. `report["refusals"]` is empty exactly when the design passes."""
    reg_alpha, reg_kappa = _registry_alpha_kappa()
    alpha = float(inp.alpha if inp.alpha is not None else reg_alpha)
    kappa = float(inp.kappa if inp.kappa is not None else reg_kappa)
    s = _sign(inp.direction)
    refusals: List[dict] = []

    def refuse(code: str, msg: str) -> None:
        refusals.append({"code": code, "reason": msg})

    arrays = {f"null:{k}": v for k, v in inp.nulls.items()}
    arrays.update({"trained": inp.trained}, **{f"dose:{k}": v for k, v in inp.planted.items()})
    clean: Dict[str, np.ndarray] = {}
    for name, v in arrays.items():
        a = np.asarray(v, dtype=float)
        if a.size == 0 or not np.all(np.isfinite(a)):
            refuse("empty", f"{name} is empty or holds non-finite values; nothing can be read off it")
        clean[name] = a
    unknown = sorted(set(inp.moved_marks) - set(inp.nulls))
    if unknown:
        refuse("moved_marks", f"moved_marks names {unknown}, which are not among the nulls")
    doses = sorted((k for k in inp.planted if float(k) != 0), key=float)
    if "0" not in inp.planted and not any(float(k) == 0 for k in inp.planted):
        refuse("doses", "no dose 0: the planted control is never read unplanted, so its headroom "
                        "and its size cannot be checked")
    if len(doses) < MIN_DOSES:
        refuse("doses", f"{len(doses)} non-zero doses; a detection curve needs at least {MIN_DOSES}")
    if not inp.nulls:
        refuse("empty", "no null population given")
    if refusals and any(r["code"] == "empty" for r in refusals):
        return _report(inp, alpha, kappa, {}, {}, {}, {}, refusals)

    # -- the attainable floor -------------------------------------------------
    if inp.p_floor is not None:
        p_floor = float(inp.p_floor)
        floor_from = "p_floor as given"
    elif inp.n_null_draws is not None:
        p_floor = 1.0 / (int(inp.n_null_draws) + 1)
        floor_from = f"rank p against {int(inp.n_null_draws)} null draws, 1/(n + 1)"
    else:
        p_floor = 1.0 / (min(len(clean[f"null:{k}"]) for k in inp.nulls) + 1)
        floor_from = "rank p against the smallest null given, 1/(n + 1)"
    max_e = calibrate(p_floor, kappa)
    floor = {"p_floor": p_floor, "from": floor_from, "max_attainable_E": max_e,
             "threshold": 1.0 / alpha, "min_null_draws": min_null_draws(alpha, kappa)}
    if max_e < 1.0 / alpha:
        refuse("floor", f"the best p ({p_floor:.3g}, {floor_from}) calibrates to E {max_e:.3g} < "
                        f"1/α = {1 / alpha:g}: no outcome can cross; a rank test needs ≥ "
                        f"{floor['min_null_draws']} null draws")

    # -- each null, and where the trained population falls in it ---------------
    nulls: Dict[str, dict] = {}
    trained = clean["trained"]
    for name in inp.nulls:
        x = clean[f"null:{name}"]
        lo, hi = np.quantile(x, [alpha / 2, 1 - alpha / 2])
        cut = _cut(x, alpha, s)
        outside = float(np.mean((trained < lo) | (trained > hi)))
        role = "moved_mark" if name in inp.moved_marks else "null"
        nulls[name] = {"role": role, "n": int(x.size), "mean": float(x.mean()),
                       "band": [float(lo), float(hi)], "cut": cut,
                       "trained_outside_band": outside,
                       "trained_beyond_cut": float(np.mean(_beyond(trained, cut, s)))}
        if role == "null" and outside >= NEARLY:
            refuse(f"band:{name}",
                   f"{outside:.1%} of the trained population at step {inp.trained_step or '?'} sits "
                   f"outside {name}'s central band [{lo:.4g}, {hi:.4g}]; a band nearly everything "
                   f"leaves is a 'moved' mark, not a null (declare it in moved_marks, or use "
                   f"another null)")
    real = [n for n, v in nulls.items() if v["role"] == "null"]
    if not real:
        refuse("band", "every null is declared a moved mark; nothing is left to test against")

    # -- the planted control: headroom, size, each dose's side, the curve -----
    control: dict = {}
    zero = next((k for k in inp.planted if float(k) == 0), None)
    curve: Dict[str, dict] = {}
    strict = (max if s > 0 else min)((nulls[n]["cut"] for n in real), default=None)
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
            control["headroom_note"] = "no bounds given: headroom not checked"
        if strict is not None:
            k = int(np.sum(_beyond(z, strict, s)))
            control["dose0_detected"] = k / z.size
            if _cp_lower(k, int(z.size)) > alpha:
                refuse("size", f"the planted control unplanted is detected at {k / z.size:.1%} "
                               f"(95 % lower bound above α = {alpha:g}); it reads an effect with "
                               f"nothing planted")
    for d in doses:
        x = clean[f"dose:{d}"]
        row = {"mean": float(x.mean()), "n": int(x.size),
               "detected": {n: float(np.mean(_beyond(x, nulls[n]["cut"], s))) for n in real}}
        below = [n for n in real if s * (row["mean"] - nulls[n]["mean"]) <= 0]
        row["below_null_mean"] = below
        if below:
            refuse(f"dose:{d}", f"dose {d}'s mean ({row['mean']:.4g}) is not beyond the null mean of "
                                f"{below} in the effect's direction; its detection rate measures the "
                                f"statistic's floor, not power")
        curve[d] = row
    if doses and real:
        top = curve[doses[-1]]["detected"]
        worst = min(top.values())
        if worst < MIN_POWER:
            refuse("power", f"the largest dose ({doses[-1]}) is detected at {worst:.1%} against the "
                            f"strictest null (< {MIN_POWER:.0%}); the design cannot see its own "
                            f"largest plant")
    return _report(inp, alpha, kappa, floor, nulls, control, curve, refusals)


def _report(inp, alpha, kappa, floor, nulls, control, curve, refusals) -> dict:
    return {"statistic": inp.statistic, "direction": inp.direction,
            "trained_step": inp.trained_step, "alpha": alpha, "kappa": kappa,
            "thresholds": {"nearly": NEARLY, "min_headroom": MIN_HEADROOM,
                           "min_power": MIN_POWER, "min_doses": MIN_DOSES},
            "floor": floor, "nulls": nulls, "control": control, "detection_curve": curve,
            "refusals": refusals, "verdict": "refused" if refusals else "pass"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Can this design reject? (LESSONS.md 6)")
    ap.add_argument("values", help="JSON of the statistic's values (see the module docstring)")
    ap.add_argument("--out", help="write the report here; a design cites its sha256")
    a = ap.parse_args(argv)
    raw = Path(a.values).read_bytes()
    report = preflight(Inputs.from_dict(json.loads(raw)))
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(report, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out} (sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]})")
    f = report["floor"]
    if f:
        print(f"floor: p {f['p_floor']:.3g} -> max E {f['max_attainable_E']:.3g} "
              f"(threshold {f['threshold']:g})")
    for d, row in report["detection_curve"].items():
        print(f"dose {d}: " + ", ".join(f"{n} {v:.2f}" for n, v in row["detected"].items()))
    for r in report["refusals"]:
        print(f"REFUSED [{r['code']}] {r['reason']}")
    print(report["verdict"])
    return 1 if report["refusals"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
