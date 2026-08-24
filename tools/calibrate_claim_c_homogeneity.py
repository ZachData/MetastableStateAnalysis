"""
tools/calibrate_claim_c_homogeneity.py — CLAIM-C's homogeneity calibration curve.

WHAT THIS MEASURES, AND WHY IT IS NOT OPTIONAL

`p1_mstate_tracking/replication_gate.py` enumerates 2^n sign-flip patterns over
n prompts on the premise that the prompts carry n pieces of information. They do
not carry n INDEPENDENT pieces -- every prompt runs on the same model and shares
its weights -- and `POPPER_PLAN.md` §6f measured what that costs at the two
ends of the range:

    independent prompt sign-rows   rejection rate at alpha=0.05 ~ 0.015
    identical prompt sign-rows     rejection rate at alpha=0.05 ~ 0.34

The gate refuses at the exactly-degenerate end and reports `sign_homogeneity`
in between. Everything BETWEEN the two ends was uncontrolled, and a real run
lands in the middle. This module removes that gap by measuring the whole curve
offline, once, and storing it, so the gate can report what its p actually
corresponds to at the homogeneity it observed.

Nothing here needs a run artifact. The gate's statistic is a function of the
(prompt x metric) concordance table alone, so H0 can be simulated exactly.

THE H0 FAMILY: PER-METRIC SHARED SIGN PROPENSITY

`sign_homogeneity` is a COLUMN-WISE statistic -- the mean over metrics of that
metric's majority sign fraction across prompts -- so the family has to be one
that controls column-wise agreement. The one chosen is literally the threat
§6f names: *"a pythia-wide effect present in every prompt (a metric that moves
the same way with training for reasons unrelated to gpt2-large)"*.

    Metric j carries a pythia-wide propensity q_j = 0.5 +/- b_j toward
    concordance with the (fixed, arbitrary) reference sign. Prompt rows are
    conditionally independent given q. b_j = 0 for every metric is the
    independent-rows end; b_j = 0.5 for every metric is the degenerate end.

Both ends reproduce the two rates §6f already measured, which is the check that
the family is the right one rather than a convenient one.

A single scalar summary cannot determine a distribution, so ONE b-vector shape
would be measuring one route to a homogeneity rather than the homogeneity. The
sweep therefore covers three shapes -- every metric biased equally, k of six
biased, and a graded ramp -- and each homogeneity bin stores the WORST
(highest-rejecting) configuration that reached it. Conservative by construction.

CONDITIONING ON EMISSION, WHICH IS THE SUBTLE PART

The rate stored here is conditional on the gate actually EMITTING a p-value:

    R(h, p) = P( p_reported <= p | the gate emitted a p, homogeneity in bin h )

Not conditioning would let the gate look well calibrated by refusing. At high
homogeneity most draws hit the identical-rows refusal and contribute nothing;
counting those as non-rejections would push R down exactly where the inflation
is worst and report a correction of zero where the correction is most needed.
The ledger only ever receives runs that emitted, so the conditional rate is the
one that governs the ledger's Type-I behaviour. `emission_rate` is stored
beside it so a reader can see how much the conditioning is doing.

WHY THE SIMULATION IS EXACT RATHER THAN SAMPLED

For one metric subset the null contributes `conc_i` or `m - conc_i` from row i,
so the null distribution -- and therefore the p-value -- depends ONLY on the
multiset of per-row concordant counts. The number of such multisets is
C(n + m, m): 3003 for eight prompts and six metrics. So every attainable
p-value is tabulated once by exact integer convolution and each simulated draw
becomes a table lookup. No Monte-Carlo error enters the p-values themselves;
the only sampling error is in the rejection RATES, and `n_emitted` per bin
records how much of it there is.

This is a second implementation of the gate's own arithmetic, which is a real
risk -- so `tests/test_claim_c_homogeneity.py::TestFastPathMatchesTheGate`
pins it against `p_value_claim_c` cell by cell on random tables. If the two
ever disagree, the gate is right and this file is wrong.

USAGE

    python3 tools/calibrate_claim_c_homogeneity.py --write     # regenerate
    python3 tools/calibrate_claim_c_homogeneity.py --check     # in step?

The curve is committed (`claims/calibration/claim_c_homogeneity.json`) because
regenerating it per run would make the correction a per-run quantity, which is
the same objection that makes every other CLAIM-C choice a module constant.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVE_PATH = ROOT / "claims" / "calibration" / "claim_c_homogeneity.json"

#: Schema of the stored curve. Bumped when the stored shape changes, so a gate
#: reading an older file refuses instead of misreading it.
CURVE_SCHEMA_VERSION = 1

#: Prompt counts tabulated. The lower end is where the attainable-floor refusal
#: already bites (five prompts cannot express a p below 0.061 at all, so no
#: curve can help); the upper end is generous against the eight metastability
#: prompts, since "spend effort on prompts" is the standing remedy.
N_PROMPTS_TABULATED: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)

#: Homogeneity bins. 0.5 is the floor of the statistic (a metric column can be
#: no more balanced than half and half) and 1.0 is the degeneracy the gate
#: already refuses at.
HOMOGENEITY_BIN_EDGES: Tuple[float, ...] = tuple(
    round(0.5 + 0.025 * k, 6) for k in range(21))

#: Levels at which each bin's p-value distribution is stored. The curve is kept
#: as a QUANTILE FUNCTION rather than a CDF on a p-grid: the levels are the
#: rejection rates we care about resolving, and they are dense exactly where
#: alpha lives. Reading it back gives a conservative R by rounding UP to the
#: next level (see `rejection_rate_at`).
CORRECTION_LEVELS: Tuple[float, ...] = tuple(sorted(set(
    [round(float(v), 6) for v in np.geomspace(2e-4, 0.2, 44)]
    + [0.01, 0.02, 0.025, 0.05, 0.1, 0.15, 0.2]
    + [round(float(v), 6) for v in np.linspace(0.25, 1.0, 16)]
)))

#: Draws per configuration per prompt count.
N_TRIALS_PER_CONFIG = 40000

#: Emitted draws a single configuration needs in a bin before that
#: configuration is allowed to set the bin's rate. PLACED, not calibrated. Two
#: things set it: a quantile at level 0.05 read off fewer draws than this is
#: noise presented as a correction, and the bin takes the MIN over
#: configurations, so under-resolved configurations would win on sampling
#: error rather than on being genuinely worse. Bins no configuration reaches
#: are stored as unmeasured and filled from above, or refused.
MIN_EMITTED_PER_CONFIG_BIN = 2000

_SEED = 20260824


# ---------------------------------------------------------------------------
# Exact per-subset p-values, tabulated over row profiles
# ---------------------------------------------------------------------------

def _compositions(n: int, k: int) -> Iterable[Tuple[int, ...]]:
    """Every way to write `n` as an ordered sum of `k` non-negative integers."""
    for bars in itertools.combinations(range(n + k - 1), k - 1):
        prev, out = -1, []
        for b in bars:
            out.append(b - prev - 1)
            prev = b
        out.append(n + k - 2 - prev)
        yield tuple(out)


class SubsetPTable:
    """
    Exact (p_greater, p_less) for every attainable row profile of one metric
    subset, keyed by the profile itself.

    A row profile is the histogram over conc values 0..m of the n prompts. The
    sign-flip null lets row i contribute `conc_i` or `m - conc_i`, so both the
    observed statistic (sum of conc_i) and the whole null distribution are
    determined by that histogram -- which is what makes an exhaustive table
    possible instead of a resampled one.

    p is formed exactly as `core.nulls.p_from_null` forms it, with the same
    (n_extreme + 1) / (n_null + 1) floor over the 2^n enumerated patterns.
    """

    def __init__(self, n_prompts: int, m: int):
        self.n_prompts, self.m = int(n_prompts), int(m)
        n, span = self.n_prompts, self.n_prompts * self.m
        n_patterns = 2 ** n
        radix = np.array([(n + 1) ** v for v in range(m + 1)], dtype=np.int64)

        keys, pg, pl = [], [], []
        for counts in _compositions(n, m + 1):
            dp = np.zeros(span + 1, dtype=np.int64)
            dp[0] = 1
            observed = 0
            for v, c in enumerate(counts):
                if not c:
                    continue
                observed += v * c
                lo, hi = v, self.m - v
                for _ in range(c):
                    nxt = np.zeros_like(dp)
                    nxt[lo:] += dp[:span + 1 - lo]
                    nxt[hi:] += dp[:span + 1 - hi]
                    dp = nxt
            n_ge = int(dp[observed:].sum())
            n_le = int(dp[:observed + 1].sum())
            keys.append(int(np.dot(np.asarray(counts, dtype=np.int64), radix)))
            pg.append((n_ge + 1.0) / (n_patterns + 1.0))
            pl.append((n_le + 1.0) / (n_patterns + 1.0))

        order = np.argsort(np.asarray(keys, dtype=np.int64))
        self.radix = radix
        self.keys = np.asarray(keys, dtype=np.int64)[order]
        self.p_greater = np.asarray(pg, dtype=np.float64)[order]
        self.p_less = np.asarray(pl, dtype=np.float64)[order]

    def lookup(self, conc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(p_greater, p_less) for a (n_trials, n_prompts) array of counts."""
        hist = np.stack([(conc == v).sum(axis=1) for v in range(self.m + 1)],
                        axis=1).astype(np.int64)
        idx = np.searchsorted(self.keys, hist @ self.radix)
        return self.p_greater[idx], self.p_less[idx]


def best_attainable_p(n_prompts: int) -> float:
    """The smallest p an exhaustive sign-flip null over n prompts can express."""
    return 2.0 / (2 ** int(n_prompts) + 1.0)


# ---------------------------------------------------------------------------
# The H0 family
# ---------------------------------------------------------------------------

def _bias_vectors(shape: str, b: float, n_metrics: int, n_trials: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Per-trial per-metric bias magnitudes |q_j - 0.5| for one configuration.

    Three shapes, because one scalar summary cannot determine a distribution:
    the same homogeneity reached by biasing every metric a little and by
    biasing two metrics a lot are different designs, and the gate should be
    calibrated against whichever of them rejects more often.
    """
    if shape == "uniform":
        return np.full((n_trials, n_metrics), b)
    if shape.startswith("khot:"):
        k = int(shape.split(":", 1)[1])
        out = np.zeros((n_trials, n_metrics))
        # A fresh subset per trial, so the configuration is a family rather
        # than one arbitrary choice of which metrics misbehave.
        cols = np.argsort(rng.random((n_trials, n_metrics)), axis=1)[:, :k]
        np.put_along_axis(out, cols, b, axis=1)
        return out
    if shape == "ramp":
        ramp = (np.arange(1, n_metrics + 1) / n_metrics) * b
        return np.tile(ramp, (n_trials, 1))
    raise ValueError(f"unknown bias shape {shape!r}")


def _configs(n_metrics: int) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = [("uniform", round(0.05 * k, 4))
                                    for k in range(11)]
    for k in range(1, n_metrics):
        for b in (0.2, 0.3, 0.4, 0.5):
            out.append((f"khot:{k}", b))
    for b in (0.2, 0.3, 0.4, 0.5):
        out.append(("ramp", b))
    return out


# ---------------------------------------------------------------------------
# Simulating the gate
# ---------------------------------------------------------------------------

def simulate_config(n_prompts: int, n_metrics: int, subsets: Sequence[Tuple[str, Tuple[int, ...]]],
                    tables: Dict[int, SubsetPTable], shape: str, b: float,
                    n_trials: int, rng: np.random.Generator) -> dict:
    """
    One configuration's draws: homogeneity, whether the gate emitted, and the
    intersection-union p in each direction when it did.

    The refusals reproduced here are exactly the ones that depend on the
    concordance table: the full table's identical-rows degeneracy and the
    per-subset one. The refusals that depend only on the prompt count (the
    attainable floor) are handled by the caller, since they do not vary
    across draws.
    """
    bias = _bias_vectors(shape, b, n_metrics, n_trials, rng)
    sign = rng.choice(np.array([-1.0, 1.0]), size=(n_trials, n_metrics))
    q = 0.5 + sign * bias
    conc_cell = rng.random((n_trials, n_prompts, n_metrics)) < q[:, None, :]

    f = conc_cell.mean(axis=1)
    homogeneity = np.maximum(f, 1.0 - f).mean(axis=1)

    refused = np.zeros(n_trials, dtype=bool)
    p_greater = np.zeros(n_trials)
    p_less = np.zeros(n_trials)
    for _, cols in subsets:
        sub = conc_cell[:, :, list(cols)]
        # The same degeneracy check the gate makes per subset: dropping a
        # metric can leave the remaining sign rows identical even when the
        # full table's are not.
        refused |= (sub == sub[:, :1, :]).all(axis=(1, 2))
        pg, pl = tables[len(cols)].lookup(sub.sum(axis=2))
        p_greater = np.maximum(p_greater, pg)
        p_less = np.maximum(p_less, pl)

    return {"homogeneity": homogeneity, "emitted": ~refused,
            "p_greater": p_greater, "p_less": p_less}


# ---------------------------------------------------------------------------
# Building the curve
# ---------------------------------------------------------------------------

def _bin_index(homogeneity: np.ndarray) -> np.ndarray:
    edges = np.asarray(HOMOGENEITY_BIN_EDGES)
    return np.clip(np.searchsorted(edges, homogeneity, side="right") - 1,
                   0, len(edges) - 2)


#: Decimal places the stored quantiles are truncated to. TRUNCATED, never
#: rounded: the attainable p-values are k / (2^n + 1) and the derived refusal
#: turns on whether a stored quantile is at or below the FLOOR, 2 / (2^n + 1).
#: Rounding to 8 places rounds that value UP for n in {7, 9, 10, 11} and down
#: for n in {6, 8, 12}, which silently switched the refusal off for four of the
#: seven tabulated prompt counts while leaving the file looking fine. Truncation
#: can only make a stored quantile smaller, and a smaller quantile is a LARGER
#: rejection rate -- so the error it can introduce is in the conservative
#: direction by construction.
QUANTILE_DECIMALS = 10


def _truncate(v: float, places: int = QUANTILE_DECIMALS) -> float:
    return math.floor(float(v) * 10 ** places) / 10 ** places


def rejection_rate_at(quantiles: Sequence[float], p: float) -> float:
    """
    Read a stored quantile row as a rejection rate, at this module's levels.

    The reading rule lives in `replication_gate.rejection_rate_at` rather than
    here, and is imported rather than restated: the gate is the consumer, and
    two copies of "how a stored rate is read" could disagree without anything
    noticing -- which is the shape of defect UPDATE_PLAN.md standing rule 4 is
    about. This wrapper only supplies the levels.
    """
    from p1_mstate_tracking.replication_gate import rejection_rate_at as _read
    return _read(CORRECTION_LEVELS, quantiles, p)


def build_curve_for_n(n_prompts: int, metrics: Sequence[str],
                      subsets: Sequence[Tuple[str, Tuple[int, ...]]],
                      n_trials: int, seed: int) -> dict:
    """
    One prompt count's curve: per homogeneity bin, the quantile function of the
    reported p under H0, taken from the WORST configuration that reached that
    bin.

    Cross-bin monotonicity is deliberately NOT imposed. R does rise with
    homogeneity, but forcing it by propagating the top bins downward would let
    the near-degenerate bins -- where almost every draw hits the identical-rows
    refusal and the survivors are a selected minority -- dictate the correction
    at homogeneities that were measured directly. Holes are filled from the
    nearest measured bin ABOVE (the conservative direction) and a bin with
    nothing above it stays unmeasured, where the gate refuses.
    """
    n_metrics = len(metrics)
    tables = {len(cols): SubsetPTable(n_prompts, len(cols))
              for _, cols in subsets}
    n_bins, n_levels = len(HOMOGENEITY_BIN_EDGES) - 1, len(CORRECTION_LEVELS)
    levels = np.asarray(CORRECTION_LEVELS)

    best_g = np.full((n_bins, n_levels), np.inf)
    best_l = np.full((n_bins, n_levels), np.inf)
    n_emitted = np.zeros(n_bins, dtype=np.int64)
    n_drawn = np.zeros(n_bins, dtype=np.int64)
    n_qualifying = np.zeros(n_bins, dtype=np.int64)

    rng = np.random.default_rng(seed)
    for shape, b in _configs(n_metrics):
        res = simulate_config(n_prompts, n_metrics, subsets, tables,
                              shape, b, n_trials, rng)
        bins = _bin_index(res["homogeneity"])
        for bi in range(n_bins):
            sel = bins == bi
            n_drawn[bi] += int(sel.sum())
            emitted = sel & res["emitted"]
            k = int(emitted.sum())
            n_emitted[bi] += k
            if k < MIN_EMITTED_PER_CONFIG_BIN:
                continue
            n_qualifying[bi] += 1
            # A smaller quantile at a fixed level IS a higher rejection rate,
            # so the worst configuration is the elementwise minimum.
            # method="lower" returns an order statistic rather than an
            # interpolation between two, so every stored quantile is a p-value
            # the test can actually express -- and it is never above the
            # interpolated quantile, which is the conservative side.
            best_g[bi] = np.minimum(best_g[bi], np.quantile(
                res["p_greater"][emitted], levels, method="lower"))
            best_l[bi] = np.minimum(best_l[bi], np.quantile(
                res["p_less"][emitted], levels, method="lower"))

    measured = np.isfinite(best_g).all(axis=1)
    for arr in (best_g, best_l):
        arr[measured] = np.maximum.accumulate(arr[measured], axis=1)

    filled = np.zeros(n_bins, dtype=bool)
    for bi in range(n_bins - 1, -1, -1):
        if measured[bi]:
            continue
        above = [k for k in range(bi + 1, n_bins) if measured[k] or filled[k]]
        if not above:
            continue
        src = above[0]
        best_g[bi], best_l[bi] = best_g[src].copy(), best_l[src].copy()
        filled[bi] = True

    usable = measured | filled
    return {
        "n_prompts": int(n_prompts),
        "best_attainable_p": best_attainable_p(n_prompts),
        "bins": [
            {
                "lo": HOMOGENEITY_BIN_EDGES[bi],
                "hi": HOMOGENEITY_BIN_EDGES[bi + 1],
                "measured": bool(measured[bi]),
                "filled_from_above": bool(filled[bi]),
                "n_drawn": int(n_drawn[bi]),
                "n_emitted": int(n_emitted[bi]),
                "emission_rate": (float(n_emitted[bi] / n_drawn[bi])
                                  if n_drawn[bi] else None),
                "n_configs_qualifying": int(n_qualifying[bi]),
                "quantiles_greater": ([_truncate(v) for v in best_g[bi]]
                                      if usable[bi] else None),
                "quantiles_less": ([_truncate(v) for v in best_l[bi]]
                                   if usable[bi] else None),
                "rejection_at_alpha_0.05": (
                    rejection_rate_at(best_g[bi], 0.05) if usable[bi] else None),
                "p_for_a_true_5_percent_test": (
                    float(best_g[bi][int(np.searchsorted(levels, 0.05))])
                    if usable[bi] else None),
            }
            for bi in range(n_bins)
        ],
    }


def build_curve(n_trials: int = N_TRIALS_PER_CONFIG, seed: int = _SEED) -> dict:
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_METRICS, CLAIM_C_RECIPROCAL_ALTERNATIVE,
        CLAIM_C_TOOL_AXIS, CLAIM_C_TOOL_RULE, _metric_subsets,
    )

    subsets = _metric_subsets()
    curves = {}
    for i, n in enumerate(N_PROMPTS_TABULATED):
        curves[str(n)] = build_curve_for_n(
            n, CLAIM_C_METRICS, subsets, n_trials, seed + 1000 * i)

    return {
        "schema_version": CURVE_SCHEMA_VERSION,
        "_what": (
            "CLAIM-C's homogeneity calibration curve. R(h, p) is the measured "
            "probability, under H0 at prompt sign-row homogeneity h, that the "
            "replication gate reports a p at or below p -- CONDITIONAL on the "
            "gate emitting a p-value at all. Generated offline by "
            "tools/calibrate_claim_c_homogeneity.py and committed, so the "
            "correction is a fixed property of the construction rather than a "
            "per-run quantity. Regenerate with --write; verify with --check."),
        "_h0_family": (
            "per-metric shared sign propensity: metric j carries a "
            "candidate-wide bias q_j = 0.5 +/- b_j toward concordance with the "
            "reference sign and prompt rows are conditionally independent given "
            "q. b = 0 everywhere is the independent-rows end; b = 0.5 "
            "everywhere is the degenerate end the gate already refuses. Three "
            "bias shapes are swept (uniform, k-of-six, graded ramp) and each "
            "bin keeps the WORST configuration that reached it, because one "
            "scalar summary does not determine a distribution."),
        "_conditioning": (
            "rates are conditional on the gate EMITTING a p. Not conditioning "
            "would let the gate look calibrated by refusing: at high "
            "homogeneity most draws hit the identical-rows refusal, and "
            "counting those as non-rejections would report a correction of "
            "zero exactly where the inflation is worst."),
        "metrics": list(CLAIM_C_METRICS),
        "n_metrics": len(CLAIM_C_METRICS),
        "alternative": CLAIM_C_ALTERNATIVE,
        "reciprocal_alternative": CLAIM_C_RECIPROCAL_ALTERNATIVE,
        "tool_axis": CLAIM_C_TOOL_AXIS,
        "tool_rule": CLAIM_C_TOOL_RULE,
        "n_subsets": len(subsets),
        "subset_names": [name for name, _ in subsets],
        "assumes_complete_table": True,
        "_assumes_complete_table_note": (
            "every simulated draw has all n_prompts x n_metrics cells usable. "
            "A real run that drops cells (a non-finite or exactly-zero "
            "contrast) has a coarser statistic than anything tabulated here, "
            "so the correction is read off a table measured on a different "
            "design. The gate reports n_cells_dropped beside the correction "
            "rather than pretending otherwise."),
        "homogeneity_bin_edges": list(HOMOGENEITY_BIN_EDGES),
        "correction_levels": list(CORRECTION_LEVELS),
        "n_trials_per_config": int(n_trials),
        "min_emitted_per_config_bin": int(MIN_EMITTED_PER_CONFIG_BIN),
        "seed": int(seed),
        "n_prompts_tabulated": list(N_PROMPTS_TABULATED),
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def check_curve(path: Path = CURVE_PATH) -> List[str]:
    """
    Structural agreement between the stored curve and the gate it corrects.

    Does NOT re-run the simulation -- that is what `--write` is for. What it
    catches is the curve going stale: a metric added to CLAIM_C_METRICS, a tail
    swapped, a subset rule changed. Any of those makes the stored rates
    measurements of a different test, and reading them would be worse than
    having no correction, because it would look like one.
    """
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_METRICS, CLAIM_C_RECIPROCAL_ALTERNATIVE,
        CLAIM_C_TOOL_AXIS, CLAIM_C_TOOL_RULE, _metric_subsets,
    )

    problems: List[str] = []
    if not path.exists():
        return [f"{path} does not exist; run --write"]
    curve = json.loads(path.read_text())

    expected = {
        "schema_version": CURVE_SCHEMA_VERSION,
        "metrics": list(CLAIM_C_METRICS),
        "alternative": CLAIM_C_ALTERNATIVE,
        "reciprocal_alternative": CLAIM_C_RECIPROCAL_ALTERNATIVE,
        "tool_axis": CLAIM_C_TOOL_AXIS,
        "tool_rule": CLAIM_C_TOOL_RULE,
        "subset_names": [name for name, _ in _metric_subsets()],
        "homogeneity_bin_edges": list(HOMOGENEITY_BIN_EDGES),
        "correction_levels": list(CORRECTION_LEVELS),
        "n_prompts_tabulated": list(N_PROMPTS_TABULATED),
    }
    for key, want in expected.items():
        if curve.get(key) != want:
            problems.append(f"{key}: stored {curve.get(key)!r} != current {want!r}")

    for n in N_PROMPTS_TABULATED:
        c = curve.get("curves", {}).get(str(n))
        if c is None:
            problems.append(f"no curve tabulated for n_prompts={n}")
            continue
        if len(c.get("bins", [])) != len(HOMOGENEITY_BIN_EDGES) - 1:
            problems.append(f"n_prompts={n}: wrong bin count")
        if not math.isclose(c.get("best_attainable_p", -1),
                            best_attainable_p(n), rel_tol=1e-12):
            problems.append(f"n_prompts={n}: best_attainable_p is stale")
    return problems


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--write", action="store_true", help="regenerate the curve")
    ap.add_argument("--check", action="store_true",
                    help="stored curve still describes the current gate?")
    ap.add_argument("--summary", action="store_true",
                    help="print the curve at alpha for each prompt count")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_PER_CONFIG)
    ap.add_argument("--out", type=Path, default=CURVE_PATH)
    args = ap.parse_args(argv)

    if args.write:
        curve = build_curve(n_trials=args.n_trials)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(curve, indent=1, sort_keys=False) + "\n")
        print(f"wrote {args.out} "
              f"({args.out.stat().st_size / 1024:.0f} KiB, "
              f"{len(curve['curves'])} prompt counts)")

    if args.check:
        problems = check_curve(args.out)
        for p in problems:
            print(f"STALE: {p}")
        if problems:
            return 1
        print(f"calibrate_claim_c_homogeneity: {args.out.name} in step with the gate")

    if args.summary:
        curve = json.loads(args.out.read_text())
        for n, c in sorted(curve["curves"].items(), key=lambda kv: int(kv[0])):
            print(f"\nn_prompts={n}  floor p={c['best_attainable_p']:.4f}")
            print(f"  {'homogeneity':>14}  {'emit':>6}  {'R(h,0.05)':>10}  "
                  f"{'p for a true 5%':>16}")
            for b in c["bins"]:
                if b["quantiles_greater"] is None:
                    continue
                print(f"  {b['lo']:.3f}-{b['hi']:.3f}  "
                      f"{(b['emission_rate'] or 0):6.3f}  "
                      f"{b['rejection_at_alpha_0.05']:10.4f}  "
                      f"{b['p_for_a_true_5_percent_test']:16.5f}"
                      + ("  (filled)" if b["filled_from_above"] else ""))
    if not (args.write or args.check or args.summary):
        ap.error("nothing to do: pass --write, --check or --summary")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
