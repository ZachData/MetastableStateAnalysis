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

THE SECOND DIMENSION: DROPPED CELLS (added 2026-08-25)

Every draw in the first version of this curve had a COMPLETE (prompt x metric)
table. A real run does not: the gate drops a cell whose trained-minus-random
contrast is non-finite or exactly zero in either architecture, because the
criterion is ordinal and a cell without a sign has nothing to contribute. Such a
run read its correction -- and therefore its refusal -- off a table measured on a
design it does not have. POPPER_PLAN.md 6g named a second curve dimension as the
honest fix and did not build it; 6j made it the binding gap by showing the
correction is what drives the refusal boundary.

Dropping cells is not the same statistic made noisier. It changes three things
at once: the sum runs over fewer cells, the per-row null weights `valid_i` stop
being equal, and a row can lose its SWING altogether -- a row with no usable cell,
or with an even number of usable cells splitting exactly half and half,
contributes the same number to the observed sum and to all 2^n null patterns.

So the curve is now indexed by `(n_prompts, drop bin, homogeneity bin)`. Bin 0
of the drop dimension is `n_cells_dropped == 0` exactly and nothing else; the
rest run up to `DROP_BIN_UPPER_EDGES[-1]`, above which the gate refuses rather
than reading the nearest row. Three mechanisms reach each rate, for the same
reason the bias family has three shapes, and each cell keeps the worst
configuration over all of them. NOTHING IS FILLED ACROSS THE DROP DIMENSION:
coarsening pushes p-values up while selecting for tables that survived the
informative-row floor pushes the conditional rate down, and measured, the two do
not resolve -- at eight prompts 98 of 117 adjacent bin pairs are neither
non-decreasing nor non-increasing. A hole in that dimension is therefore a
refusal, not an interpolation.

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

Row i contributes `conc_i` or `valid_i - conc_i`, so with `e_i = 2 conc_i -
valid_i` the null of the centred statistic is exactly the distribution of
`sum ±g_i` where `g_i = |e_i|`: it depends ONLY on the multiset of per-row
SWINGS, and the observation is `sum e_i`. The number of such multisets is
C(n + m, m): 3003 for eight prompts and six metrics, and -- this is the part
that makes the drop dimension affordable at all -- that count does not change
when cells are dropped, where keying on the pair `(valid_i, conc_i)` would have
put it at 23 million. So every attainable null distribution is tabulated once by
exact integer convolution and each simulated draw becomes a table lookup. No Monte-Carlo error enters the p-values themselves;
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
CURVE_SCHEMA_VERSION = 2

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

#: Base draws per configuration. The count actually used is this SCALED BY
#: PROMPT COUNT -- see `trials_for`.
N_TRIALS_PER_CONFIG = 40000

#: Ceiling on that scale, so a prompt count where the gate refuses almost
#: everything cannot make the run unbounded. PLACED: it is a budget, and if it
#: ever binds the coverage report will show the holes rather than the run
#: hiding them.
MAX_TRIAL_SCALE = 3

#: Emitted draws a single configuration needs in a bin before that
#: configuration is allowed to set the bin's rate. PLACED, not calibrated. Two
#: things set it: a quantile at level 0.05 read off fewer draws than this is
#: noise presented as a correction, and the bin takes the MIN over
#: configurations, so under-resolved configurations would win on sampling
#: error rather than on being genuinely worse. Bins no configuration reaches
#: are stored as unmeasured and filled from above, or refused.
MIN_EMITTED_PER_CONFIG_BIN = 2000

_SEED = 20260824


def informative_row_emission_probability(n_prompts: int, n_metrics: int,
                                         a: float) -> float:
    """
    P(the informative-row floor does NOT refuse) under independent-row H0.

    A row's swing is `|valid - 2 conc|`, so on a complete table with an EVEN
    metric count a row is uninformative exactly when it splits half and half,
    which has probability C(m, m/2) / 2^m -- 20/64 at six metrics. Rows are
    independent under this H0, so the informative count is Binomial, and the
    gate emits when it is at least the smallest k whose floor clears alpha.

    Closed form, not simulated, because it is used to SIZE the simulation.
    """
    p_un = (math.comb(n_metrics, n_metrics // 2) / 2 ** n_metrics
            if n_metrics % 2 == 0 else 0.0)
    n = int(n_prompts)
    ks = [k for k in range(n + 1)
          if (2.0 ** (n - k) + 1.0) / (2 ** n + 1.0) <= a]
    if not ks:
        return 0.0
    kmin = min(ks)
    return float(sum(math.comb(n, k) * (1 - p_un) ** k * p_un ** (n - k)
                     for k in range(kmin, n + 1)))


def trials_for(n_prompts: int, n_metrics: int, a: float,
               base: int = N_TRIALS_PER_CONFIG) -> int:
    """
    Draws per configuration at this prompt count.

    WHY IT IS NOT ONE NUMBER ANY MORE. Every rate here is conditional on the
    gate EMITTING, so what a bin needs is a fixed number of EMITTED draws, not
    of drawn ones. The informative-row floor (POPPER_PLAN.md 6l) refuses a
    table whose prompts cannot move the statistic, and how often that happens
    depends sharply on the prompt count: measured, six prompts emit on 39% of
    independent-row H0 draws, eight on 78%, twelve on 99%. Drawing the same
    40000 everywhere therefore measures the small prompt counts to a coarser
    resolution than the large ones -- and it showed: the first generation of
    this curve left the six-prompt (0, 5%] drop slab with NO measured bin at
    all, so the gate refused there outright.

    The scale is DERIVED from that probability rather than tuned: `1 / P`,
    rounded and capped. It is not exact -- the drop slabs refuse more than the
    complete-table case this computes -- so `curve_coverage` reports what was
    actually measured and the docs record whatever hole is left.
    """
    p = informative_row_emission_probability(n_prompts, n_metrics, a)
    scale = 1 if p <= 0 else min(MAX_TRIAL_SCALE, max(1, round(1.0 / p)))
    return int(base * scale)


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
    subset, keyed by the profile itself. Handles INCOMPLETE tables.

    THE KEY IS THE SWING HISTOGRAM, and that is what makes dropped cells
    tabulable at all. Row i contributes `conc_i` unflipped and
    `valid_i - conc_i` flipped, so writing `e_i = 2 conc_i - valid_i` and
    `g_i = |e_i|`, the null of `2 x statistic - sum(valid)` is exactly the
    distribution of `sum ±g_i` over the 2^n sign patterns -- it depends only on
    the multiset of SWINGS. The observed value is `sum e_i`, which is not
    determined by that multiset, so the table stores the whole null distribution
    per key and the observation indexes into it.

    Why not the obvious key. On a complete table every row has `valid_i = m` and
    the profile is the histogram of `conc_i`, which is what this table used to
    store. Once cells drop, `valid_i` varies per row and the pair `(valid, conc)`
    has (m+1)(m+2)/2 = 28 values at six metrics: multisets of eight rows over 28
    types is 23 million keys, and the table stops being buildable. The swing
    histogram has only m+1 types, so it is C(n+m, m) keys -- 3003 at eight
    prompts and six metrics, exactly the size of the complete-table version.
    Dropping cells costs the tabulation nothing.

    It is a strict generalisation: on a complete table `g_i = |m - 2 conc_i|`
    and the two agree cell for cell, which
    `tests/test_claim_c_homogeneity.py::TestFastPathMatchesTheGate` pins against
    `p_value_claim_c` on both complete and holed tables. If the two ever
    disagree, the gate is right and this file is wrong.

    p is formed exactly as `core.nulls.p_from_null` forms it, with the same
    (n_extreme + 1) / (n_null + 1) floor over the 2^n enumerated patterns.
    """

    def __init__(self, n_prompts: int, m: int):
        self.n_prompts, self.m = int(n_prompts), int(m)
        n, span = self.n_prompts, self.n_prompts * self.m
        self.n_patterns = 2 ** n
        self.span = span
        radix = np.array([(n + 1) ** v for v in range(m + 1)], dtype=np.int64)

        # `ge[k, j]` is the number of sign patterns whose null index is at or
        # above j, where the null value is `2j - S` and `S = sum(g)`. Rows are
        # padded to `span` so every key has the same length; beyond S the
        # counts are 0 for `ge` and the full pattern count for `le`, which is
        # what a query outside the support should read.
        keys: List[int] = []
        rows_ge, rows_le, s_of_key = [], [], []
        for counts in _compositions(n, m + 1):
            dp = np.zeros(span + 1, dtype=np.int64)
            dp[0] = 1
            total = 0
            for g, c in enumerate(counts):
                if not c or g == 0:
                    # A swing of 0 doubles every pattern without moving the
                    # statistic -- exactly the rows the informative-row floor
                    # is about. Folded in as a multiplier below.
                    continue
                total += g * c
                for _ in range(c):
                    nxt = np.zeros_like(dp)
                    nxt += dp                       # this row takes -g
                    nxt[g:] += dp[:span + 1 - g]    # this row takes +g
                    dp = nxt
            n_zero = counts[0]
            if n_zero:
                dp = dp * (2 ** n_zero)
            suffix = np.cumsum(dp[::-1])[::-1]
            prefix = np.cumsum(dp)
            rows_ge.append(suffix)
            rows_le.append(prefix)
            s_of_key.append(total)
            keys.append(int(np.dot(np.asarray(counts, dtype=np.int64), radix)))

        order = np.argsort(np.asarray(keys, dtype=np.int64))
        self.radix = radix
        self.keys = np.asarray(keys, dtype=np.int64)[order]
        self.ge = np.asarray(rows_ge, dtype=np.int64)[order]
        self.le = np.asarray(rows_le, dtype=np.int64)[order]
        self.s_of_key = np.asarray(s_of_key, dtype=np.int64)[order]

    def lookup(self, valid: np.ndarray, conc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        (p_greater, p_less) for (n_trials, n_prompts) arrays of per-row usable
        and concordant counts.
        """
        valid = np.asarray(valid, dtype=np.int64)
        conc = np.asarray(conc, dtype=np.int64)
        e = 2 * conc - valid
        g = np.abs(e)
        hist = np.stack([(g == v).sum(axis=1) for v in range(self.m + 1)],
                        axis=1).astype(np.int64)
        idx = np.searchsorted(self.keys, hist @ self.radix)
        S = self.s_of_key[idx]
        # The observed null index. `sum(e)` has the same parity as S, so this is
        # exact integer arithmetic and not a rounded one.
        j = ((e.sum(axis=1) + S) // 2).astype(np.int64)
        j = np.clip(j, 0, self.span)
        n_ge = self.ge[idx, j]
        n_le = self.le[idx, j]
        denom = self.n_patterns + 1.0
        return (n_ge + 1.0) / denom, (n_le + 1.0) / denom


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
# The second dimension: dropped cells
# ---------------------------------------------------------------------------
#
# WHY THIS DIMENSION EXISTS. The gate drops a (prompt, metric) cell whose
# trained-minus-random contrast is non-finite or exactly zero in either
# architecture -- a degeneracy, since the criterion is ordinal and a cell
# without a sign has nothing to contribute. Every draw in the first version of
# this curve had a COMPLETE table, so a real run that drops cells read its
# correction, and therefore its refusal, off a table measured on a design it
# does not have. POPPER_PLAN.md 6g named that as the honest fix and did not
# build it; 6j made it pointed by showing the correction is what drives the
# refusal boundary.
#
# Dropping changes three things at once, which is why it could not be argued
# away: the statistic is summed over fewer cells, the per-row null weights
# `valid_i` stop being equal, and a row can lose its SWING entirely -- a row
# with no usable cell, or with an even number splitting half and half,
# contributes identically to the observed sum and to all 2^n null patterns.
#
# WHAT IS ASSUMED, stated because it is the boundary of what this covers.
# Drops are independent of concordance GIVEN THE POSITION: which cells go is
# modelled, whether a surviving cell agrees is not conditioned on it. A
# mechanism that preferentially removes discordant cells is not in this family
# and would not be corrected by this curve.

#: Where the drops land. One rate reached three ways, for the same reason the
#: bias family has three shapes: a scalar summary does not determine a
#: distribution, and 5% of cells gone at random is not the same design as 5%
#: gone from one prompt. Each (homogeneity, drop) cell keeps the WORST
#: configuration that reached it, over the bias shapes and these together.
#:
#:   mcar    every cell independently, which is the benign reading
#:   column  concentrated in one metric -- an instrument that failed
#:   row     concentrated in one prompt -- the severe one, because a rate above
#:           1/n_prompts empties that row and costs an informative unit outright
DROP_MECHANISMS: Tuple[str, ...] = ("mcar", "column", "row")

#: Target overall drop fractions swept. Chosen to put draws in every bin below
#: rather than to be round: the realised fraction is random, so the grid is
#: denser than the bins.
DROP_RATES: Tuple[float, ...] = (0.02, 0.05, 0.08, 0.12, 0.17, 0.22, 0.28)

#: Upper edges of the drop-fraction bins ABOVE the exact-zero one. Bin 0 is
#: `n_cells_dropped == 0` exactly and nothing else: a complete table is the
#: design the gate was built on and the common case, and lumping it with a table
#: that lost one cell would hide precisely the transition this dimension exists
#: to measure. Bin i covers (edges[i-2], edges[i-1]].
#:
#: There is deliberately no bin above the last edge. A run that has lost 30% of
#: its cells is not a noisier version of this design, and the gate refuses there
#: rather than reading the nearest row -- the same refusal it already makes at a
#: homogeneity the curve has no measurement for.
DROP_BIN_UPPER_EDGES: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

N_DROP_BINS = 1 + len(DROP_BIN_UPPER_EDGES)


def drop_bin_index_vec(n_dropped, n_cells_total):
    """
    Which drop bin each of many tables falls in; -1 means off the top of the
    tabulated range.

    A VECTORIZED SECOND IMPLEMENTATION of
    `replication_gate.drop_bin_index`, which is the authority. It exists only
    because the simulation addresses forty thousand tables at a time and the
    gate's scalar version would be the whole cost of the run. That makes it the
    same risk as this file's p-value fast path -- two copies of one rule that
    could drift apart without anything noticing -- so it is pinned against the
    gate's version across the whole range in
    `tests/test_claim_c_homogeneity.py::TestDropBinLookupMatchesTheGate`. If the
    two ever disagree, the gate is right and this file is wrong.
    """
    n_dropped = np.asarray(n_dropped)
    total = int(n_cells_total)
    frac = (n_dropped / float(total)) if total else np.zeros(n_dropped.shape, float)
    idx = 1 + np.searchsorted(np.asarray(DROP_BIN_UPPER_EDGES), frac, side="left")
    idx = np.where(n_dropped == 0, 0, idx)
    return np.where(idx > len(DROP_BIN_UPPER_EDGES), -1, idx)


def drop_bin_bounds(bi: int) -> Tuple[float, float]:
    """(lo, hi] of a drop bin; bin 0 is the single point 0.0."""
    if bi == 0:
        return 0.0, 0.0
    lo = 0.0 if bi == 1 else DROP_BIN_UPPER_EDGES[bi - 2]
    return lo, DROP_BIN_UPPER_EDGES[bi - 1]


def _drop_mask(mechanism: str, rate: float, n_prompts: int, n_metrics: int,
               n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """
    (n_trials, n_prompts, n_metrics) boolean, True where the cell is dropped.

    Each mechanism is scaled so its EXPECTED overall fraction is `rate`, which
    is what makes the three comparable at one point of the grid rather than
    three different rates wearing one label.
    """
    if mechanism == "none" or rate <= 0:
        return np.zeros((n_trials, n_prompts, n_metrics), dtype=bool)
    if mechanism == "mcar":
        return rng.random((n_trials, n_prompts, n_metrics)) < rate

    # The concentrated mechanisms spread over as FEW lines as the rate allows
    # and no fewer. One metric column is 1/n_metrics of the table and one
    # prompt row is 1/n_prompts of it, so a single line saturates at that
    # fraction: sweeping one line at every rate would leave the upper drop bins
    # reachable only by `mcar`, which is the benign mechanism, and the
    # worst-configuration rule would then be taking a worst over one candidate
    # exactly where the design is most stressed. `k` is the smallest number of
    # lines that can carry the rate, so concentration stays maximal all the way
    # up.
    if mechanism in ("column", "row"):
        n_lines = n_metrics if mechanism == "column" else n_prompts
        k = max(1, int(math.ceil(rate * n_lines - 1e-9)))
        k = min(k, n_lines)
        per = min(1.0, rate * n_lines / k)
        # A fresh choice of which lines misbehave per trial, so the
        # configuration is a family rather than one arbitrary choice.
        lines = np.argsort(rng.random((n_trials, n_lines)), axis=1)[:, :k]
        out = np.zeros((n_trials, n_prompts, n_metrics), dtype=bool)
        if mechanism == "column":
            hit = rng.random((n_trials, k, n_prompts)) < per
            t = np.arange(n_trials)[:, None, None]
            pr = np.arange(n_prompts)[None, None, :]
            out[t, pr, lines[:, :, None]] = hit
        else:
            hit = rng.random((n_trials, k, n_metrics)) < per
            t = np.arange(n_trials)[:, None, None]
            me = np.arange(n_metrics)[None, None, :]
            out[t, lines[:, :, None], me] = hit
        return out

    raise ValueError(f"unknown drop mechanism {mechanism!r}")


def _drop_configs() -> List[Tuple[str, float]]:
    """`("none", 0.0)` first: at rate 0 the three mechanisms coincide, so
    sweeping them separately there would be three copies of one design."""
    return [("none", 0.0)] + [(m, r) for m in DROP_MECHANISMS for r in DROP_RATES]


# ---------------------------------------------------------------------------
# Simulating the gate
# ---------------------------------------------------------------------------

def alpha() -> float:
    """The registry's alpha, read through the gate so the two cannot drift.

    The curve depends on it twice over: the derived refusal is `R(h, floor) >
    alpha`, and the informative-row refusal reproduced below is `floor > alpha`.
    A curve measured at one alpha and read at another is a measurement of a
    different test, so it is stored in the artifact and `check_curve` compares
    it.
    """
    from p1_mstate_tracking.replication_gate import _alpha
    return float(_alpha())


def simulate_config(n_prompts: int, n_metrics: int,
                    subsets: Sequence[Tuple[str, Tuple[int, ...]]],
                    tables: Dict[int, SubsetPTable], shape: str, b: float,
                    mechanism: str, rate: float,
                    n_trials: int, rng: np.random.Generator,
                    a: float) -> dict:
    """
    One configuration's draws: homogeneity, dropped-cell count, whether the gate
    emitted, and the intersection-union p in each direction when it did.

    A configuration is now a (bias shape, bias magnitude, drop mechanism, drop
    rate) -- the H0 family crossed with the drop family.

    THE REFUSALS REPRODUCED HERE are exactly the ones that depend on the
    concordance table, and dropping cells added three to the list:

      * the full table having no cell with a sign at all;
      * the full table's identical-rows degeneracy (unchanged);
      * a SUBSET having no cell with a sign -- reachable now without a dead
        metric, because drops can empty a five-column subset;
      * a subset's identical-rows degeneracy (unchanged); and
      * the INFORMATIVE-ROW FLOOR, which is the refusal drops make bite: a row
        with no usable cell, or with an even number splitting half and half,
        cannot move the statistic, and with fewer than five that can the null
        expresses no p below alpha.

    The refusals that depend only on the prompt count (the design's attainable
    floor) are handled by the caller, since they do not vary across draws. The
    CORRECTED attainable floor is deliberately not reproduced: it is defined in
    terms of R, and R is what this file is measuring. The stored rate is
    therefore conditional on a slightly larger emission set than the gate's,
    which can only make the correction blunter, and the artifact says so.
    """
    bias = _bias_vectors(shape, b, n_metrics, n_trials, rng)
    sign = rng.choice(np.array([-1.0, 1.0]), size=(n_trials, n_metrics))
    q = 0.5 + sign * bias
    conc_cell = rng.random((n_trials, n_prompts, n_metrics)) < q[:, None, :]

    dropped = _drop_mask(mechanism, rate, n_prompts, n_metrics, n_trials, rng)
    usable = ~dropped
    n_dropped = dropped.sum(axis=(1, 2)).astype(np.int64)

    # `sign_homogeneity` over USABLE cells only, and columns with no usable cell
    # contribute nothing -- which is what `_row_independence` does.
    cnt = usable.sum(axis=1)
    pos = (conc_cell & usable).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(cnt > 0, pos / np.maximum(cnt, 1), np.nan)
    frac = np.maximum(f, 1.0 - f)
    with np.errstate(invalid="ignore"):
        homogeneity = np.where(np.isnan(frac).all(axis=1), np.nan,
                               np.nanmean(np.where(np.isnan(frac), np.nan, frac),
                                          axis=1))

    refused = np.zeros(n_trials, dtype=bool)
    refused |= usable.sum(axis=(1, 2)) == 0
    refused |= np.isnan(homogeneity)

    # The full table's identical-rows degeneracy. Rows are compared only on the
    # metrics usable in EVERY row, as the gate does. Concordance stands in for
    # the candidate sign: the two differ by a per-column constant, and equality
    # ACROSS rows is invariant to that.
    common_all = usable.all(axis=1)
    if n_prompts > 1:
        eq = (conc_cell == conc_cell[:, :1, :]) | ~common_all[:, None, :]
        refused |= eq.all(axis=(1, 2)) & common_all.any(axis=1)

    n_patterns = 2 ** n_prompts
    p_greater = np.zeros(n_trials)
    p_less = np.zeros(n_trials)
    floor_iut = np.zeros(n_trials)
    for _, cols in subsets:
        idx = list(cols)
        u = usable[:, :, idx]
        c = conc_cell[:, :, idx] & u
        v_i = u.sum(axis=2).astype(np.int64)
        c_i = c.sum(axis=2).astype(np.int64)

        refused |= v_i.sum(axis=1) == 0
        com = u.all(axis=1)
        if n_prompts > 1:
            sub = conc_cell[:, :, idx]
            eqs = (sub == sub[:, :1, :]) | ~com[:, None, :]
            refused |= eqs.all(axis=(1, 2)) & com.any(axis=1)

        # The informative-row floor, per subset. `2^(n-k) + 1` over `2^n + 1`.
        k = (np.abs(v_i - 2 * c_i) > 0).sum(axis=1)
        floor_iut = np.maximum(
            floor_iut, (2.0 ** (n_prompts - k) + 1.0) / (n_patterns + 1.0))

        pg, pl = tables[len(cols)].lookup(v_i, c_i)
        p_greater = np.maximum(p_greater, pg)
        p_less = np.maximum(p_less, pl)

    refused |= floor_iut > a

    return {"homogeneity": homogeneity, "n_dropped": n_dropped,
            "emitted": ~refused, "p_greater": p_greater, "p_less": p_less}


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
                      n_trials: int, seed: int, a: float) -> dict:
    """
    One prompt count's curve: per (drop-fraction bin, homogeneity bin), the
    quantile function of the reported p under H0, taken from the WORST
    configuration that reached that cell.

    TWO FILL RULES, AND THEY ARE DELIBERATELY DIFFERENT.

    Within a drop slab, holes in homogeneity are filled from the nearest
    measured bin ABOVE, as they always were: R rises with homogeneity, so
    reading a higher bin is the conservative direction, and cross-bin
    monotonicity is still not imposed on measured bins -- forcing it would let
    the near-degenerate bins, where almost every draw refuses and the survivors
    are a selected minority, dictate the correction where it was measured
    directly.

    ACROSS drop bins, nothing is filled. There is no argument that R rises with
    the drop fraction: dropping cells coarsens the statistic, which pushes
    p-values UP, while the same drops select for tables that survived the
    informative-row floor, which pushes the conditional rate DOWN. The two
    effects point opposite ways and neither obviously wins, so no direction is
    assumed -- an unmeasured (h, d) cell has no correction and the gate refuses
    there. `drop_monotone_in_d` records what the measurement actually says, as
    a finding rather than as an input.
    """
    n_metrics = len(metrics)
    tables = {len(cols): SubsetPTable(n_prompts, len(cols))
              for _, cols in subsets}
    n_h = len(HOMOGENEITY_BIN_EDGES) - 1
    n_d = N_DROP_BINS
    n_levels = len(CORRECTION_LEVELS)
    levels = np.asarray(CORRECTION_LEVELS)
    n_cells = n_d * n_h

    best_g = np.full((n_cells, n_levels), np.inf)
    best_l = np.full((n_cells, n_levels), np.inf)
    n_emitted = np.zeros(n_cells, dtype=np.int64)
    n_drawn = np.zeros(n_cells, dtype=np.int64)
    n_qualifying = np.zeros(n_cells, dtype=np.int64)
    n_off_top = 0

    n_cells_total = n_prompts * n_metrics
    rng = np.random.default_rng(seed)
    for shape, b in _configs(n_metrics):
        for mech, rate in _drop_configs():
            res = simulate_config(n_prompts, n_metrics, subsets, tables,
                                  shape, b, mech, rate, n_trials, rng, a)
            di = drop_bin_index_vec(res["n_dropped"], n_cells_total)
            hi = _bin_index(np.nan_to_num(res["homogeneity"], nan=0.5))
            on = di >= 0
            n_off_top += int((~on).sum())
            cell = np.where(on, di * n_h + hi, 0)

            np.add.at(n_drawn, cell[on], 1)
            emitted = on & res["emitted"]
            np.add.at(n_emitted, cell[emitted], 1)

            counts = np.bincount(cell[emitted], minlength=n_cells)
            for ci in np.nonzero(counts >= MIN_EMITTED_PER_CONFIG_BIN)[0]:
                sel = emitted & (cell == ci)
                n_qualifying[ci] += 1
                # A smaller quantile at a fixed level IS a higher rejection
                # rate, so the worst configuration is the elementwise minimum.
                # method="lower" returns an order statistic rather than an
                # interpolation between two, so every stored quantile is a
                # p-value the test can actually express -- and it is never
                # above the interpolated quantile, which is the conservative
                # side.
                best_g[ci] = np.minimum(best_g[ci], np.quantile(
                    res["p_greater"][sel], levels, method="lower"))
                best_l[ci] = np.minimum(best_l[ci], np.quantile(
                    res["p_less"][sel], levels, method="lower"))

    measured = np.isfinite(best_g).all(axis=1)
    for arr in (best_g, best_l):
        arr[measured] = np.maximum.accumulate(arr[measured], axis=1)

    filled = np.zeros(n_cells, dtype=bool)
    for di in range(n_d):
        base = di * n_h
        for hi in range(n_h - 1, -1, -1):
            ci = base + hi
            if measured[ci]:
                continue
            above = [k for k in range(hi + 1, n_h)
                     if measured[base + k] or filled[base + k]]
            if not above:
                continue
            src = base + above[0]
            best_g[ci], best_l[ci] = best_g[src].copy(), best_l[src].copy()
            filled[ci] = True

    usable = measured | filled

    def _cell(di: int, hi: int) -> dict:
        ci = di * n_h + hi
        return {
            "lo": HOMOGENEITY_BIN_EDGES[hi],
            "hi": HOMOGENEITY_BIN_EDGES[hi + 1],
            "measured": bool(measured[ci]),
            "filled_from_above": bool(filled[ci]),
            "n_drawn": int(n_drawn[ci]),
            "n_emitted": int(n_emitted[ci]),
            "emission_rate": (float(n_emitted[ci] / n_drawn[ci])
                              if n_drawn[ci] else None),
            "n_configs_qualifying": int(n_qualifying[ci]),
            "quantiles_greater": ([_truncate(v) for v in best_g[ci]]
                                  if usable[ci] else None),
            "quantiles_less": ([_truncate(v) for v in best_l[ci]]
                               if usable[ci] else None),
            "rejection_at_alpha_0.05": (
                rejection_rate_at(best_g[ci], 0.05) if usable[ci] else None),
            "p_for_a_true_5_percent_test": (
                float(best_g[ci][int(np.searchsorted(levels, 0.05))])
                if usable[ci] else None),
        }

    drop_bins = []
    for di in range(n_d):
        lo, hi_ = drop_bin_bounds(di)
        drop_bins.append({
            "drop_bin_index": di,
            "drop_lo": lo,
            "drop_hi": hi_,
            "exact_zero": di == 0,
            "bins": [_cell(di, hi) for hi in range(n_h)],
        })

    return {
        "n_prompts": int(n_prompts),
        "best_attainable_p": best_attainable_p(n_prompts),
        "n_trials_per_config": int(n_trials),
        "n_drawn_off_top_of_drop_range": int(n_off_top),
        "coverage": {
            "n_measured_bins_per_drop_slab": [
                int(sum(1 for hi in range(n_h) if measured[di * n_h + hi]))
                for di in range(n_d)],
            "n_usable_bins_per_drop_slab": [
                int(sum(1 for hi in range(n_h) if usable[di * n_h + hi]))
                for di in range(n_d)],
            "drop_slabs_with_no_measurement": [
                int(di) for di in range(n_d)
                if not any(usable[di * n_h + hi] for hi in range(n_h))],
            "_what": ("what was actually measured, per drop slab, so a hole is "
                      "visible in the artifact rather than found by a run "
                      "getting refused. A slab in "
                      "drop_slabs_with_no_measurement is one the gate refuses "
                      "outright at this prompt count."),
        },
        "drop_monotone_in_d": _monotone_in_d(best_g, usable, n_d, n_h),
        "drop_bins": drop_bins,
    }


def _monotone_in_d(best_g: np.ndarray, usable: np.ndarray, n_d: int,
                   n_h: int) -> dict:
    """
    Does R rise, fall, or neither as cells are dropped?

    Reported, never used. The fill rule above assumes no direction precisely
    because this is a measurement and not a premise -- POPPER_PLAN.md 6j's
    monotonicity result about R in p was worth having exactly because it was
    checked rather than asserted, and the honest version of that here is to
    check and then say what came back.
    """
    up = down = neither = 0
    for hi in range(n_h):
        cols = [di for di in range(n_d) if usable[di * n_h + hi]]
        for x, y in zip(cols, cols[1:]):
            a_ = best_g[x * n_h + hi]
            b_ = best_g[y * n_h + hi]
            # A SMALLER quantile is a HIGHER rejection rate.
            if np.all(b_ <= a_ + 1e-15):
                up += 1
            elif np.all(b_ >= a_ - 1e-15):
                down += 1
            else:
                neither += 1
    total = up + down + neither
    return {
        "adjacent_pairs_compared": total,
        "rate_rises_with_drops": up,
        "rate_falls_with_drops": down,
        "neither": neither,
        "note": ("counted over adjacent measured drop bins at a fixed "
                 "homogeneity; a smaller stored quantile is a higher rejection "
                 "rate. REPORTED, NOT USED: nothing is filled across the drop "
                 "dimension, so no direction is assumed anywhere"),
    }


def build_curve(n_trials: int = N_TRIALS_PER_CONFIG, seed: int = _SEED) -> dict:
    from p1_mstate_tracking.replication_gate import (
        CLAIM_C_ALTERNATIVE, CLAIM_C_METRICS, CLAIM_C_RECIPROCAL_ALTERNATIVE,
        CLAIM_C_TOOL_AXIS, CLAIM_C_TOOL_RULE, _metric_subsets,
    )

    subsets = _metric_subsets()
    a = alpha()
    n_m = len(CLAIM_C_METRICS)
    trials = {n: trials_for(n, n_m, a, base=n_trials) for n in N_PROMPTS_TABULATED}
    curves = {}
    for i, n in enumerate(N_PROMPTS_TABULATED):
        curves[str(n)] = build_curve_for_n(
            n, CLAIM_C_METRICS, subsets, trials[n], seed + 1000 * i, a)

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
            "zero exactly where the inflation is worst. The refusals folded "
            "into 'emitted' are the ones that depend on the concordance table: "
            "an empty table, the full table's identical-rows degeneracy, an "
            "empty or identical-rows metric subset, and the INFORMATIVE-ROW "
            "FLOOR (a row with no usable cell, or with an even number of "
            "usable cells splitting half and half, cannot move the statistic; "
            "with fewer than five that can, the null expresses no p below "
            "alpha). The CORRECTED attainable floor is deliberately not "
            "reproduced -- it is defined in terms of R and R is what this file "
            "measures -- so the stored rate conditions on a slightly LARGER "
            "emission set than the gate's, which can only make the correction "
            "blunter."),
        "metrics": list(CLAIM_C_METRICS),
        "n_metrics": len(CLAIM_C_METRICS),
        "alternative": CLAIM_C_ALTERNATIVE,
        "reciprocal_alternative": CLAIM_C_RECIPROCAL_ALTERNATIVE,
        "tool_axis": CLAIM_C_TOOL_AXIS,
        "tool_rule": CLAIM_C_TOOL_RULE,
        "n_subsets": len(subsets),
        "subset_names": [name for name, _ in subsets],
        "assumes_complete_table": False,
        "alpha": float(a),
        "_alpha_note": (
            "the registry's alpha at generation time. The curve depends on it "
            "twice: the gate's derived refusal is R(h, floor) > alpha, and the "
            "informative-row refusal reproduced in the simulation is floor > "
            "alpha, which decides which draws emit. A curve measured at one "
            "alpha and read at another is a measurement of a different test, so "
            "check_curve compares it."),
        "drop_bin_upper_edges": list(DROP_BIN_UPPER_EDGES),
        "drop_mechanisms": list(DROP_MECHANISMS),
        "drop_rates": list(DROP_RATES),
        "n_drop_bins": int(N_DROP_BINS),
        "_drop_family": (
            "the second curve dimension, added 2026-08-25 (POPPER_PLAN.md 6l). "
            "The first version of this curve gave every simulated draw a "
            "COMPLETE n_prompts x n_metrics table, so a real run that dropped "
            "cells read its correction -- and therefore its refusal -- off a "
            "table measured on a design it does not have. Drops are now swept "
            "as a dimension: bin 0 is exactly zero dropped cells, and the "
            "remaining bins run to the last of drop_bin_upper_edges, above "
            "which the gate refuses rather than reading the nearest row. Three "
            "mechanisms reach each rate -- mcar (every cell independently), "
            "column (concentrated in as few metrics as the rate allows) and row "
            "(concentrated in as few prompts, which is the severe one, because "
            "a rate above 1/n_prompts empties a row and costs an informative "
            "unit outright) -- and each (homogeneity, drop) cell keeps the "
            "WORST configuration that reached it, over the bias shapes and the "
            "drop mechanisms together. WHAT IS ASSUMED: drops are independent "
            "of concordance GIVEN THE POSITION. A mechanism that "
            "preferentially removes discordant cells is not in this family. "
            "NOTHING IS FILLED ACROSS THIS DIMENSION: coarsening pushes "
            "p-values up while selection on the informative-row floor pushes "
            "the conditional rate down, the two point opposite ways, and "
            "drop_monotone_in_d per prompt count records what was measured "
            "rather than assuming a direction."),
        "homogeneity_bin_edges": list(HOMOGENEITY_BIN_EDGES),
        "correction_levels": list(CORRECTION_LEVELS),
        "n_trials_per_config": {str(n): int(v) for n, v in trials.items()},
        "n_trials_base": int(n_trials),
        "_n_trials_note": (
            "draws per configuration, PER PROMPT COUNT. Every rate here is "
            "conditional on the gate emitting, so a bin needs a fixed number "
            "of EMITTED draws and not of drawn ones -- and the informative-row "
            "floor refuses far more often at six prompts (39% of "
            "independent-row H0 draws emit) than at twelve (99%). The scale is "
            "derived from that probability, 1/P rounded and capped at "
            "MAX_TRIAL_SCALE, not tuned. Drawing 40000 everywhere left the "
            "six-prompt (0, 5%] drop slab with no measured bin at all."),
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
        "drop_bin_upper_edges": list(DROP_BIN_UPPER_EDGES),
        "drop_mechanisms": list(DROP_MECHANISMS),
        "drop_rates": list(DROP_RATES),
        "n_drop_bins": int(N_DROP_BINS),
        "assumes_complete_table": False,
    }
    for key, want in expected.items():
        if curve.get(key) != want:
            problems.append(f"{key}: stored {curve.get(key)!r} != current {want!r}")

    # alpha decides which draws emit (the informative-row floor) and where the
    # gate's derived refusal falls, so a curve measured at another alpha is a
    # measurement of another test.
    stored_alpha = curve.get("alpha")
    if stored_alpha is None or not math.isclose(float(stored_alpha), alpha(),
                                                rel_tol=1e-12):
        problems.append(f"alpha: stored {stored_alpha!r} != registry {alpha()!r}")

    for n in N_PROMPTS_TABULATED:
        c = curve.get("curves", {}).get(str(n))
        if c is None:
            problems.append(f"no curve tabulated for n_prompts={n}")
            continue
        slabs = c.get("drop_bins", [])
        if len(slabs) != N_DROP_BINS:
            problems.append(f"n_prompts={n}: {len(slabs)} drop slabs, want {N_DROP_BINS}")
        for di, slab in enumerate(slabs):
            lo, hi = drop_bin_bounds(di)
            if (slab.get("drop_lo"), slab.get("drop_hi")) != (lo, hi):
                problems.append(f"n_prompts={n} slab {di}: bounds are stale")
            if len(slab.get("bins", [])) != len(HOMOGENEITY_BIN_EDGES) - 1:
                problems.append(f"n_prompts={n} slab {di}: wrong homogeneity bin count")
        if not math.isclose(c.get("best_attainable_p", -1),
                            best_attainable_p(n), rel_tol=1e-12):
            problems.append(f"n_prompts={n}: best_attainable_p is stale")
        want_trials = trials_for(n, len(CLAIM_C_METRICS), alpha(),
                                 base=int(curve.get("n_trials_base", 0)))
        if c.get("n_trials_per_config") != want_trials:
            problems.append(
                f"n_prompts={n}: stored draw count "
                f"{c.get('n_trials_per_config')!r} != {want_trials!r}. The "
                f"scale is derived from the informative-row emission "
                f"probability, so a change in alpha or the metric count moves "
                f"it and the curve has to be regenerated")
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
            print(f"\nn_prompts={n}  floor p={c['best_attainable_p']:.4f}  "
                  f"monotone in drops: {c['drop_monotone_in_d']['rate_rises_with_drops']} up / "
                  f"{c['drop_monotone_in_d']['rate_falls_with_drops']} down / "
                  f"{c['drop_monotone_in_d']['neither']} neither")
            for slab in c["drop_bins"]:
                tag = ("no cells dropped" if slab["exact_zero"]
                       else f"drops in ({slab['drop_lo']:.2f}, {slab['drop_hi']:.2f}]")
                print(f"  -- {tag}")
                print(f"     {'homogeneity':>14}  {'emit':>6}  {'R(h,0.05)':>10}  "
                      f"{'p for a true 5%':>16}")
                for b in slab["bins"]:
                    if b["quantiles_greater"] is None:
                        continue
                    print(f"     {b['lo']:.3f}-{b['hi']:.3f}  "
                          f"{(b['emission_rate'] or 0):6.3f}  "
                          f"{b['rejection_at_alpha_0.05']:10.4f}  "
                          f"{b['p_for_a_true_5_percent_test']:16.5f}"
                          + ("  (filled)" if b["filled_from_above"] else ""))
    if not (args.write or args.check or args.summary):
        ap.error("nothing to do: pass --write, --check or --summary")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
