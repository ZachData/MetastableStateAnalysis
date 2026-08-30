#!/usr/bin/env python3
"""
tools/calibrate_cross_head_association.py — offline calibration of P-I3's
cross-head construction (`p7_motifs/cross_head_gate.py`).

Run once, offline; the result is committed to
`claims/calibration/cross_head_association.json` and pinned by
`tests/test_p7_cross_head_gate.py::TestCommittedCalibration`.

WHY THIS IS A TOOL AND NOT A TEST

The same division of labour as the five calibrations before it, for the same
reason: measuring a rejection rate to three digits takes hundreds of replicates
of an end-to-end gate call, plus a 20,000-draw permutation study, which is
minutes; the CI gating tier runs in tens of seconds. What the pure tier pins is
the MECHANISM -- deterministically and in milliseconds -- plus the committed
numbers, so the record cannot drift without a failure. The record stores its own
`elapsed_seconds` rather than a cost quoted in prose, which is the convention
since 6o.

WHAT IS MEASURED, AND WHY EACH SECTION EXISTS

`registered_null` -- what "permutation over the head classification" does. It is
here because a floor read off a group size rather than off the design is the
commonest defect these passes have found, and this group is the largest one any
row in this registry names: C(384, 8) = 1.09e16, nominal floor 9.2e-17. The
section measures the two things that make it not a floor of anything -- that the
observed induction group's score spread lies outside every one of 20,000 draws,
and that neither statistic computed against that null can tell a genuine effect
from its absence.

`degeneracy` -- the selection attenuation. One population, one relation, no
interaction of any kind, split at the top k by score. It is the answer to 6o's
"matched on what", and the reason the registered wording's within-group
correlation reaches no ledger.

`grid` -- what the matching contributes. Three sub-sections, and each of them
changed the construction: `straddle` (a one-sided match reads curvature as an
effect), `n_controls_frontier` (power stops rising at four controls while the
analysis keeps shrinking off the top of the induction ranking), and
`overlap_frontier` (how much of the design survives as the classification stops
being a cutoff on the score).

`validity` -- the H0 and power families, run through the real gate, under BOTH
matching keys on the SAME draws so the comparison is paired. `tautology` is the
family this construction exists to catch and it carries the counterfactual rate
with the refusal removed, rather than a claim that the refusal costs nothing.

`limitation` -- the confound the score key does not remove: induction heads
cluster in a band of layers and a shared elevation on that band is invisible to
a control matched on score alone. It is in the file because 6m records that a
calibration whose families cannot express the failure it is meant to rule out is
an audit arm incapable of failing.

`floor_arithmetic` -- the design floor as a table, with no draw count in it.

Usage
-----
    python tools/calibrate_cross_head_association.py            # measure, print
    python tools/calibrate_cross_head_association.py --write    # and commit it
    python tools/calibrate_cross_head_association.py --check    # re-read the file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p7_motifs.cross_head_gate import (            # noqa: E402
    CONTROL_MATCHING_KEYS,
    N_CONTROLS_PER_INDUCTION_HEAD,
    attainable_floor_report,
    exact_rank_arm,
    matched_sets,
    p_value_p_i3,
    registered_null_invariance_report,
)

SCHEMA_VERSION = 1
OUT_PATH = ROOT / "claims" / "calibration" / "cross_head_association.json"
CONSTRUCTION_PATH = ROOT / "p7_motifs" / "cross_head_gate.py"

#: The head grid every measurement here runs on: Pythia-410M's shape, which is
#: the model `core/pythia_registry.py` names for the pilot. PLACED as a family
#: definition rather than calibrated -- it defines what the rates are rates
#: under, and no distribution was consulted for it.
N_LAYERS, N_HEADS_PER_LAYER = 24, 16
N_HEADS = N_LAYERS * N_HEADS_PER_LAYER

#: How many heads the classification calls induction heads. PLACED, for the
#: same reason.
K_INDUCTION = 8

#: The layer band the induction heads are confined to in the `limitation`
#: family. PLACED, and chosen to match what the mechinterp literature reports
#: rather than derived from anything here.
INDUCTION_LAYER_BAND = (6, 11)

_SEED = 20260830


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alpha() -> float:
    from core.adjudication import load_registry
    from core.evalues import DEFAULT_ALPHA
    try:
        return float(load_registry().get("alpha", DEFAULT_ALPHA))
    except Exception:
        return float(DEFAULT_ALPHA)


# ---------------------------------------------------------------------------
# The synthetic head table
# ---------------------------------------------------------------------------

def draw_heads(seed: int, *, rho_c: float = 0.8, k: int = K_INDUCTION,
               main: float = 1.0, curve: float = 0.0, effect: float = 0.0,
               band_elevation: float = 0.0, confine_to_band: bool = False):
    """
    One checkpoint's head table.

    `rho_c` is how far the classification is from being a cutoff on the
    behavioural score itself: 1.0 IS that cutoff, and lower values are a
    classification with its own independent component, which is what a
    classification drawn from a different task or a different criterion has.
    `main` is the score-to-motif relation P-I3's tautology risk is about --
    present under H0 and under H1 alike, and the thing the matching removes.
    `effect` is the prediction's own quantity: motif rate carried by an
    induction head OVER a control head at the same behavioural score.
    """
    r = np.random.default_rng(seed)
    keys = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS_PER_LAYER)]
    layer = np.array([k_[0] for k_ in keys])
    b = r.standard_normal(N_HEADS)
    c = rho_c * b + np.sqrt(max(0.0, 1.0 - rho_c ** 2)) * r.standard_normal(N_HEADS)
    if confine_to_band:
        lo, hi = INDUCTION_LAYER_BAND
        c = np.where((layer >= lo) & (layer <= hi), c, c - 3.0)
    lab = np.zeros(N_HEADS, dtype=bool)
    lab[np.argsort(c, kind="mergesort")[-k:]] = True
    lo, hi = INDUCTION_LAYER_BAND
    elevation = np.where((layer >= lo) & (layer <= hi), band_elevation, 0.0)
    x = (main * (b + curve * (b ** 2 - 1.0)) + effect * lab + elevation
         + r.standard_normal(N_HEADS))
    return (keys, x, b, lab, layer)


def _as_dicts(keys, x, b, lab):
    return ({k: float(x[i]) for i, k in enumerate(keys)},
            {k: float(b[i]) for i, k in enumerate(keys)},
            {k: bool(lab[i]) for i, k in enumerate(keys)})


# ---------------------------------------------------------------------------
# 1. What the registered null does
# ---------------------------------------------------------------------------

def _rank(v):
    """
    Average ranks. The tie loop runs only when there ARE ties, rather than once
    per distinct value: this is called hundreds of thousands of times in the
    permutation study, and a loop over every distinct value made that section
    minutes rather than seconds.
    """
    o = np.argsort(v, kind="mergesort")
    r = np.empty(v.size, dtype=np.float64)
    r[o] = np.arange(v.size, dtype=np.float64)
    s = v[o]
    if s.size > 1 and (s[1:] == s[:-1]).any():
        for u in np.unique(s[:-1][s[1:] == s[:-1]]):
            t = v == u
            r[t] = r[t].mean()
    return r


def _corr(b, x, sel):
    if sel.sum() < 3:
        return float("nan")
    rx, rb = _rank(x[sel]), _rank(b[sel])
    if rx.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, rb)[0, 1])


def _slope(b, x, sel):
    if sel.sum() < 3 or b[sel].std() == 0:
        return float("nan")
    return float(np.cov(b[sel], x[sel], bias=True)[0, 1] / b[sel].var())


def _contrast(stat):
    return lambda b, x, m: stat(b, x, m) - stat(b, x, ~m)


def label_permutation_p(b, x, lab, contrast, draws: int, rng) -> float:
    """A p-value against the REGISTERED null: draw which k heads are labelled."""
    n, k = b.size, int(lab.sum())
    t_obs = contrast(b, x, lab)
    if not np.isfinite(t_obs):
        return float("nan")
    ge = 0
    for _ in range(draws):
        m = np.zeros(n, dtype=bool)
        m[rng.permutation(n)[:k]] = True
        t = contrast(b, x, m)
        if np.isfinite(t) and t >= t_obs:
            ge += 1
    return (1.0 + ge) / (draws + 1.0)


def registered_null_section(replicates: int, draws: int, spread_draws: int,
                            seed: int, alpha: float) -> dict:
    """Everything about the null the registry names, measured rather than argued."""
    rng = np.random.default_rng(seed)
    arithmetic = registered_null_invariance_report(N_HEADS, K_INDUCTION)

    # The induction group's score spread against the permuted family's, at
    # both readings of the classification. `rho_c = 1.0` is the case the
    # argument is about -- the classification IS a cutoff on the score -- and
    # 0.8 is the case this gate can actually adjudicate, so it is the harder
    # one for the claim and is reported beside it rather than instead of it.
    spread = {"permuted_draws": int(spread_draws), "rows": []}
    for rho_c in (1.0, 0.8):
        # Averaged over draws on BOTH sides. A single observed group's spread
        # is a draw from a distribution with real variability -- one of them
        # read 0.14 and another 0.51 on the same construction -- and a ratio
        # computed from one of each is a number that moves on regeneration.
        obs = np.array([float(b_[lab_].std()) for b_, lab_ in
                        (draw_heads(seed + 1 + t, rho_c=rho_c)[2:4]
                         for t in range(replicates))])
        obs_sd = float(obs.mean())
        sds = np.array([b_[rng.permutation(N_HEADS)[:K_INDUCTION]].std()
                        for b_ in (draw_heads(seed + 1 + (t % replicates),
                                              rho_c=rho_c)[2]
                                   for t in range(spread_draws))])
        spread["rows"].append({
            "classification_correlation_with_the_score": rho_c,
            "observed_induction_group_score_sd": obs_sd,
            "observed_draws": int(replicates),
            "permuted_mean_sd": float(sds.mean()),
            "permuted_min_sd": float(sds.min()),
            "permuted_mean_over_observed": float(sds.mean() / obs_sd),
            "draws_at_or_below_the_observed": int((sds <= obs_sd).sum()),
            "share_at_or_below_the_observed": float((sds <= obs_sd).mean()),
        })
    spread["_note"] = (
        "the observed group is a slice at the TOP of the score ranking; a "
        "permuted group is k heads from anywhere. The exact statement is the "
        "combinatorial one above -- one assignment of C(n,k) puts all k at the "
        "top -- and this is its consequence in the currency the statistics "
        "care about. A permuted draw CAN be tightly clustered somewhere else, "
        "so the share below the observed spread is small rather than zero, and "
        "what is checked is the ratio of the means")

    rates = []
    for name, stat in (("correlation_contrast", _contrast(_corr)),
                       ("slope_contrast", _contrast(_slope))):
        row = {"statistic": name}
        for tag, effect in (("h0", 0.0), ("genuine_effect", 1.5)):
            hits = 0
            for s in range(replicates):
                _, x, b, lab, _ = draw_heads(seed + 1000 + s, effect=effect)
                p = label_permutation_p(b, x, lab, stat, draws,
                                        np.random.default_rng(seed + 5000 + s))
                if np.isfinite(p) and p <= alpha:
                    hits += 1
            row[tag] = hits / replicates
        row["discrimination"] = row["genuine_effect"] - row["h0"]
        rates.append(row)

    return {
        "arithmetic": arithmetic,
        "spread_under_permutation": spread,
        "rejection_rates": rates,
        "replicates": replicates,
        "permutation_draws": draws,
        # DERIVED from the rows above rather than written beside them. The
        # first version of this string carried the digits of an exploratory
        # run -- +0.010 from 0.065 and -0.005 from 0.285 -- which the committed
        # measurement contradicted, and nothing was failing on it because
        # nothing compares a record's prose to its own fields. A finding
        # sentence that is built from the numbers cannot disagree with them.
        "_the_finding": (
            "the registered null's group has {size:.3g} members and exactly one "
            "of them is a possible classification. Measured, neither statistic "
            "computed against it discriminates a planted effect from its "
            "absence: the correlation contrast at {c:+.3f} from an H0 rate of "
            "{ch:.3f}, the slope contrast at {s:+.3f} from an H0 rate of "
            "{sh:.3f}. Both are the same mechanism read in opposite directions "
            "-- the permuted groups carry the full score spread and the "
            "observed one does not. These are proportions over {n} replicates "
            "and move on regeneration; what check_record enforces is that "
            "neither discrimination leaves +/-0.10 and that the slope reading "
            "stays anti-conservative"
        ).format(
            size=float(arithmetic["group_size"]),
            c=rates[0]["discrimination"], ch=rates[0]["h0"],
            s=rates[1]["discrimination"], sh=rates[1]["h0"],
            n=replicates),
    }


# ---------------------------------------------------------------------------
# 2. What the statistic degenerates on
# ---------------------------------------------------------------------------

def selection_attenuation(replicates: int, seed: int) -> List[dict]:
    """
    One population, ONE relation, no interaction. The two arms' correlations
    and slopes as a function of where the classification cuts.
    """
    out = []
    for k in (4, 8, 16, 32, 64):
        ri, rc, si, sc, sdi, sdc = [], [], [], [], [], []
        for s in range(replicates):
            _, x, b, lab, _ = draw_heads(seed + 17 * k + s, rho_c=1.0, k=k,
                                         main=1.0, effect=0.0)
            ri.append(_corr(b, x, lab)); rc.append(_corr(b, x, ~lab))
            si.append(_slope(b, x, lab)); sc.append(_slope(b, x, ~lab))
            sdi.append(float(b[lab].std())); sdc.append(float(b[~lab].std()))
        out.append({
            "n_induction_heads": k,
            "spearman_induction": float(np.nanmean(ri)),
            "spearman_control": float(np.nanmean(rc)),
            "spearman_contrast": float(np.nanmean(ri) - np.nanmean(rc)),
            "slope_induction": float(np.nanmean(si)),
            "slope_control": float(np.nanmean(sc)),
            "slope_induction_sd": float(np.nanstd(si)),
            "slope_induction_standard_error":
                float(np.nanstd(si) / np.sqrt(max(1, replicates))),
            "slope_control_sd": float(np.nanstd(sc)),
            "slope_sd_ratio": float(np.nanstd(si) / np.nanstd(sc)),
            "score_spread_ratio": float(np.mean(sdi) / np.mean(sdc)),
        })
    return out


# ---------------------------------------------------------------------------
# 3. What the measurement grid contributes
# ---------------------------------------------------------------------------

def _nearest_sets(b, lab, m):
    """The obvious matching this construction does NOT use: nearest by score."""
    ind = np.flatnonzero(lab)
    available = list(np.flatnonzero(~lab))
    sets = []
    for i in ind[np.argsort(b[ind], kind="mergesort")]:
        if len(available) < m:
            continue
        d = np.abs(b[np.array(available)] - b[i])
        pick = [available[t] for t in np.argsort(d, kind="mergesort")[:m]]
        for j in pick:
            available.remove(j)
        sets.append((int(i), [int(j) for j in pick]))
    return sets


def straddle_section(replicates: int, seed: int, alpha: float) -> dict:
    """
    The discarded matching, and the sweep that discarded it.

    A one-sided (nearest-by-score) match is what the phrase "matched control"
    first suggests. It is in the record because it is what corrected the
    construction -- 6o's discarded rank refusal and 6q's discarded bend
    contrast, same posture, third time.
    """
    rows = []
    for curve in (0.0, 0.4, 0.8):
        near = strad = 0
        n_near = n_strad = 0
        for s in range(replicates):
            _, x, b, lab, layer = draw_heads(seed + 71 + s, curve=curve,
                                             main=1.0, effect=0.0)
            ns = _nearest_sets(b, lab, N_CONTROLS_PER_INDUCTION_HEAD)
            if ns:
                n_near += 1
                if exact_rank_arm(x, ns)["p_greater"] <= alpha:
                    near += 1
            ss = matched_sets(b, lab, layer, key="score",
                              n_controls=N_CONTROLS_PER_INDUCTION_HEAD)["sets"]
            if ss:
                n_strad += 1
                if exact_rank_arm(x, ss)["p_greater"] <= alpha:
                    strad += 1
        rows.append({
            "curvature_in_the_score_to_motif_relation": curve,
            "nearest_by_score_rejection": near / max(1, n_near),
            "straddled_rejection": strad / max(1, n_strad),
        })
    return {
        "rows": rows,
        "_why": (
            "an induction head sits high in the score ranking, so its NEAREST "
            "control heads are almost all below it. The residual score gap is "
            "then one-signed, and any curvature in the score-to-motif relation "
            "is read as an effect. Requiring half the controls below and half "
            "above makes the gap cancel to first order -- and is also what "
            "turns a classification that is a cutoff on the score into zero "
            "informative sets rather than into a plausible-looking answer"),
    }


def tautology_leak(replicates: int, seed: int, alpha: float) -> List[dict]:
    """
    How much a threshold classification leaks under the DISCARDED nearest-by-
    score matching, as a function of how hard the motif tracks the score.

    This is PREDICTIONS.md's Phase 7 adjudication constraint 2 as a curve: the
    danger is not that the two variables are literally the same number -- that
    is the degenerate case P-I1's gate already refuses -- but that the motif
    tracks the score at all, which is the case the constraint describes and no
    gate before this one could see. At `main = 0` there is nothing to leak.
    """
    out = []
    for main in (0.0, 0.5, 1.0, 2.0, 4.0):
        hits = 0
        n = 0
        for s in range(replicates):
            _, x, b, lab, layer = draw_heads(seed + 1234 + s, rho_c=1.0,
                                             main=main, effect=0.0)
            ns = _nearest_sets(b, lab, N_CONTROLS_PER_INDUCTION_HEAD)
            if not ns:
                continue
            n += 1
            if exact_rank_arm(x, ns)["p_greater"] <= alpha:
                hits += 1
        straddled = matched_sets(b, lab, layer, key="score",
                                 n_controls=N_CONTROLS_PER_INDUCTION_HEAD)
        out.append({
            "score_to_motif_relation": main,
            "nearest_matching_rejection": hits / max(1, n),
            "straddled_matched_sets": straddled["n_sets"],
        })
    return out


def n_controls_frontier(replicates: int, seed: int, alpha: float) -> List[dict]:
    """
    Power, validity, and WHICH induction heads survive, as the control count
    grows. The row that fixes `N_CONTROLS_PER_INDUCTION_HEAD`.
    """
    out = []
    for m in (2, 4, 6, 8, 10):
        h0 = pw = 0
        n = 0
        kinf, retained, alls = [], [], []
        for s in range(replicates):
            keys, x, b, lab, layer = draw_heads(seed + 313 + s, effect=0.0)
            ms = matched_sets(b, lab, layer, key="score", n_controls=m)
            if not ms["sets"]:
                continue
            n += 1
            kinf.append(ms["n_sets"])
            retained.append(float(np.mean(ms["retained_score_ranks"])))
            order = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
            alls.append(float(np.mean(order[lab])))
            if exact_rank_arm(x, ms["sets"])["p_greater"] <= alpha:
                h0 += 1
            _, x2, b2, lab2, layer2 = draw_heads(seed + 313 + s, effect=0.8)
            ms2 = matched_sets(b2, lab2, layer2, key="score", n_controls=m)
            if ms2["sets"] and exact_rank_arm(x2, ms2["sets"])["p_greater"] <= alpha:
                pw += 1
        out.append({
            "n_controls": m,
            "mean_informative_sets": float(np.mean(kinf)) if kinf else 0.0,
            "h0_rejection": h0 / max(1, n),
            "power_at_effect_0.8": pw / max(1, n),
            "mean_score_rank_of_retained_induction_heads":
                float(np.mean(retained)) if retained else None,
            "mean_score_rank_of_all_induction_heads":
                float(np.mean(alls)) if alls else None,
        })
    return out


def overlap_frontier(replicates: int, seed: int, alpha: float) -> List[dict]:
    """
    How much of the design survives as the classification stops being a cutoff
    on the behavioural score. `rho_c = 1.0` IS that cutoff.
    """
    out = []
    for rho_c in (1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.0):
        sets, floors, sufficient = [], [], 0
        for s in range(replicates):
            _, x, b, lab, layer = draw_heads(seed + 909 + s, rho_c=rho_c)
            ms = matched_sets(b, lab, layer, key="score",
                              n_controls=N_CONTROLS_PER_INDUCTION_HEAD)
            arm = (exact_rank_arm(x, ms["sets"]) if ms["sets"] else None)
            n_inf = arm["n_informative_sets"] if arm else 0
            fl = attainable_floor_report(ms["n_sets"], n_inf,
                                         N_CONTROLS_PER_INDUCTION_HEAD, alpha)
            sets.append(ms["n_sets"]); floors.append(fl["attainable_floor"])
            sufficient += int(fl["sufficient"])
        out.append({
            "classification_correlation_with_the_score": rho_c,
            "mean_matched_sets": float(np.mean(sets)),
            "median_attainable_floor": float(np.median(floors)),
            "share_of_draws_whose_floor_clears_alpha": sufficient / replicates,
        })
    return out


# ---------------------------------------------------------------------------
# 4. Validity, power, and the limitation
# ---------------------------------------------------------------------------

FAMILIES = (
    # name, kwargs, what the correct verdict is
    ("independent",            dict(main=0.0, effect=0.0), "INSUFFICIENT"),
    ("pure-score-relation",    dict(main=1.0, effect=0.0), "INSUFFICIENT"),
    ("curved-score-relation",  dict(main=1.0, curve=0.6, effect=0.0), "INSUFFICIENT"),
    ("tautology",              dict(main=1.0, effect=0.0, rho_c=1.0), "REFUSED"),
    ("effect-0.8",             dict(main=1.0, effect=0.8), "TRACKS_CLASSIFICATION"),
    ("effect-1.5",             dict(main=1.0, effect=1.5), "TRACKS_CLASSIFICATION"),
    ("reciprocal-1.5",         dict(main=1.0, effect=-1.5), "ACTIVATION_PROPERTY"),
)


def validity_section(replicates: int, seed: int, alpha: float) -> List[dict]:
    """
    Every family through the REAL gate, under both matching keys, on the same
    draws so the comparison is paired.
    """
    rows = []
    for name, kw, expect in FAMILIES:
        for key in CONTROL_MATCHING_KEYS:
            emitted = refused = tracks = activation = 0
            cf_tracks = 0
            floors = []
            for s in range(replicates):
                keys, x, b, lab, layer = draw_heads(seed + 4242 + s, **kw)
                m, bs, ls = _as_dicts(keys, x, b, lab)
                res = p_value_p_i3(m, bs, ls, "two_stage", key=key, alpha=alpha)
                if res["p_value"] is None:
                    refused += 1
                else:
                    emitted += 1
                    floors.append(res["floor"]["attainable_floor"])
                    if res["verdict"] == "TRACKS_CLASSIFICATION":
                        tracks += 1
                    if res["verdict"] == "ACTIVATION_PROPERTY":
                        activation += 1
                # The counterfactual, re-scored on every draw rather than
                # asserted to be rare (6q's convention). It is the DISCARDED
                # nearest-by-score matching and not "the same arm with the
                # refusal switched off", because on the family the refusal
                # exists for the straddle yields no sets at all -- so the only
                # counterfactual that answers "what would the obvious
                # construction have returned here" is the obvious construction.
                ns = _nearest_sets(b, lab, N_CONTROLS_PER_INDUCTION_HEAD)
                if ns and exact_rank_arm(x, ns)["p_greater"] <= alpha:
                    cf_tracks += 1
            rows.append({
                "family": name,
                "matching_key": key,
                "correct_verdict": expect,
                "replicates": replicates,
                "emitted": emitted,
                "refused": refused,
                "refusal_rate": refused / replicates,
                "tracks_classification_rate": tracks / max(1, emitted),
                "activation_property_rate": activation / max(1, emitted),
                "counterfactual_rate_nearest_matching": cf_tracks / replicates,
                "median_attainable_floor":
                    (float(np.median(floors)) if floors else None),
            })
    return rows


def layer_band_section(replicates: int, seed: int, alpha: float) -> List[dict]:
    """
    The limitation. Induction heads confined to a band of layers, as the
    literature reports, with a shared elevation across that band that has
    nothing to do with the classification.
    """
    rows = []
    for elevation in (0.0, 0.3, 0.6, 1.0):
        for key in CONTROL_MATCHING_KEYS:
            emitted = tracks = 0
            sets = []
            for s in range(replicates):
                keys, x, b, lab, layer = draw_heads(
                    seed + 6161 + s, main=1.0, effect=0.0,
                    band_elevation=elevation, confine_to_band=True)
                m, bs, ls = _as_dicts(keys, x, b, lab)
                res = p_value_p_i3(m, bs, ls, "two_stage", key=key, alpha=alpha)
                if res["matched_sets"]:
                    sets.append(res["matched_sets"]["n_sets"])
                if res["p_value"] is None:
                    continue
                emitted += 1
                if res["verdict"] == "TRACKS_CLASSIFICATION":
                    tracks += 1
            rows.append({
                "shared_elevation_on_the_induction_band_sd": elevation,
                "matching_key": key,
                "emitted_rate": emitted / replicates,
                "tracks_classification_rate": tracks / max(1, emitted),
                "mean_matched_sets": float(np.mean(sets)) if sets else 0.0,
            })
    return rows


def layer_band_power(replicates: int, seed: int, alpha: float) -> List[dict]:
    """What the layer key costs, on the same confined-band geometry."""
    rows = []
    for effect in (0.8, 1.5):
        for key in CONTROL_MATCHING_KEYS:
            emitted = tracks = 0
            for s in range(replicates):
                keys, x, b, lab, layer = draw_heads(
                    seed + 7171 + s, main=1.0, effect=effect,
                    confine_to_band=True)
                m, bs, ls = _as_dicts(keys, x, b, lab)
                res = p_value_p_i3(m, bs, ls, "two_stage", key=key, alpha=alpha)
                if res["p_value"] is None:
                    continue
                emitted += 1
                if res["verdict"] == "TRACKS_CLASSIFICATION":
                    tracks += 1
            rows.append({
                "planted_effect": effect,
                "matching_key": key,
                "emitted_rate": emitted / replicates,
                "power": tracks / max(1, emitted),
            })
    return rows


# ---------------------------------------------------------------------------
# 5. The floor, as arithmetic with no draw count in it
# ---------------------------------------------------------------------------

def floor_arithmetic(alpha: float) -> dict:
    table = []
    for m in (2, 4, 6, 8):
        row = {"n_controls": m}
        for k in (1, 2, 3, 4, 6, 8):
            row[f"sets_{k}"] = (m + 1.0) ** (-k)
        row["min_informative_sets_for_alpha"] = \
            attainable_floor_report(0, 0, m, alpha)["min_informative_sets_for_alpha"]
        table.append(row)
    degenerate = attainable_floor_report(0, 0, N_CONTROLS_PER_INDUCTION_HEAD, alpha)
    return {
        "table": table,
        "alpha": alpha,
        "no_informative_sets": {
            "attainable_floor": degenerate["attainable_floor"],
            "sufficient": degenerate["sufficient"],
            "_note": ("a classification that is a cutoff on the behavioural "
                      "score leaves no induction head with a control above it, "
                      "so no set can be straddled, so the floor is 1.000 and "
                      "no input whatever could reject. PREDICTIONS.md's "
                      "Phase 7 adjudication constraint 2 as arithmetic"),
        },
        "sampling_floor_is_absent_because": (
            "the null enumerates: the induction label is uniform over each "
            "matched set's M+1 members and the sets are disjoint. 6p's rule -- "
            "the attainable floor is the MAX of the design floor and the "
            "sampling resolution -- applies with its second term absent, and "
            "the record says so rather than omitting it"),
    }


# ---------------------------------------------------------------------------
# What must still be true of the committed record
# ---------------------------------------------------------------------------

def check_record(doc: dict) -> List[str]:
    """
    Returns the failures. It fails if the FINDINGS stop being in the file, not
    only if a field goes missing: this record is the evidence for a section
    that says the registered null cannot discriminate, that a threshold
    classification floors the design at 1.000, and that the layer key trades
    power for a confound. A record that no longer shows those does not support
    it. Same posture as `patching_exponent.json`'s and
    `claim_b_grid_feasibility.json`'s.
    """
    bad: List[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        bad.append(f"schema_version {doc.get('schema_version')} != {SCHEMA_VERSION}")
    if doc.get("construction_sha256") != _sha256(CONSTRUCTION_PATH):
        bad.append("construction_sha256 does not match p7_motifs/cross_head_gate.py")

    rn = doc.get("registered_null", {})
    spread = rn.get("spread_under_permutation", {})
    srows = {r["classification_correlation_with_the_score"]: r
             for r in spread.get("rows", [])}
    cutoff = srows.get(1.0)
    if cutoff is None:
        bad.append("spread_under_permutation lost the rho_c=1.0 row, which is "
                   "the case the section's argument is about")
    else:
        if cutoff["permuted_mean_over_observed"] < 2.0:
            bad.append(
                f"with the classification a cutoff on the score, a permuted "
                f"group's spread is now only "
                f"{cutoff['permuted_mean_over_observed']:.2f}x the observed "
                f"group's; the permuted family stops being a family of "
                f"impossible configurations and the section should be re-read")
        if cutoff["share_at_or_below_the_observed"] > 0.05:
            bad.append(
                f"{cutoff['share_at_or_below_the_observed']} of label-permuted "
                f"draws are as tightly clustered as the observed group; this "
                f"study is the empirical reading of the exact combinatorics, "
                f"and if the two disagree the combinatorics are what to read")
    if rn.get("arithmetic", {}).get(
            "assignments_that_are_a_threshold_on_the_score") != 1:
        bad.append("the registered null's group no longer contains exactly one "
                   "possible classification")
    rates = {r["statistic"]: r for r in rn.get("rejection_rates", [])}
    for name in ("correlation_contrast", "slope_contrast"):
        row = rates.get(name)
        if row is None:
            bad.append(f"registered_null.rejection_rates lost {name}")
        elif abs(row["discrimination"]) > 0.10:
            bad.append(
                f"the registered null's {name} now discriminates a planted "
                f"effect from its absence at {row['discrimination']:+.3f}; "
                f"that is a better answer than this record gives and the "
                f"section should be rewritten rather than this check relaxed")
    if rates.get("slope_contrast", {}).get("h0", 0.0) < 0.10:
        bad.append("the registered null's slope reading is no longer "
                   "anti-conservative on plain H0, which is half of why it is "
                   "not a null")

    # The finding sentence is derived from the rows; this is what makes that
    # load-bearing rather than a convention. The first version of it carried an
    # exploratory run's digits that the committed measurement contradicted, and
    # nothing was failing on it because nothing compares a record's prose to its
    # own fields.
    finding = rn.get("_the_finding", "")
    for name, row in rates.items():
        for value in (row.get("discrimination"), row.get("h0")):
            if value is None:
                continue
            if f"{value:+.3f}" not in finding and f"{value:.3f}" not in finding:
                bad.append(
                    f"registered_null._the_finding does not carry {name}'s "
                    f"measured {value:.3f}; the sentence has drifted from the "
                    f"rows it is derived from")

    deg = {r["n_induction_heads"]: r for r in doc.get("degeneracy", {})
           .get("selection_attenuation", [])}
    row = deg.get(K_INDUCTION)
    if row is None:
        bad.append("degeneracy.selection_attenuation lost the k=8 row")
    else:
        if row["spearman_contrast"] > -0.20:
            bad.append("the selection attenuation stopped being large and "
                       "negative; it is the whole argument for not adjudicating "
                       "the within-group correlation")
        # Against the induction arm's OWN standard error, not a fixed number:
        # that arm's slope has a spread this section exists to report, so a
        # constant tolerance either fires on a healthy artifact at a low
        # replicate count or says nothing at a high one.
        tol = 3.0 * row.get("slope_induction_standard_error", 0.0)
        if abs(row["slope_induction"] - row["slope_control"]) > max(tol, 1e-9):
            bad.append(
                f"the within-group SLOPES stopped agreeing "
                f"({row['slope_induction']:.3f} against "
                f"{row['slope_control']:.3f}, 3 se = {tol:.3f}), so the claim "
                f"that selection biases the correlation and not the slope no "
                f"longer holds as stated")
        if row["slope_sd_ratio"] < 5.0:
            bad.append("the induction arm's slope is no longer far noisier "
                       "than the control arm's, which is why the unbiased "
                       "statistic is unusable")

    grid = doc.get("grid", {})
    st = {r["curvature_in_the_score_to_motif_relation"]: r
          for r in grid.get("straddle", {}).get("rows", [])}
    worst = max((r["nearest_by_score_rejection"] for r in st.values()),
                default=0.0)
    worst_str = max((r["straddled_rejection"] for r in st.values()), default=1.0)
    if worst <= 0.10:
        bad.append("the discarded nearest-by-score matching no longer leaks on "
                   "curvature; the straddle's justification is that it does")
    if worst_str > 0.10:
        bad.append(f"the straddled matching now rejects at {worst_str:.3f} on a "
                   f"curved score relation with no effect present")
    front = {r["n_controls"]: r for r in grid.get("n_controls_frontier", [])}
    if N_CONTROLS_PER_INDUCTION_HEAD not in front:
        bad.append("n_controls_frontier does not cover the registered count")
    else:
        chosen = front[N_CONTROLS_PER_INDUCTION_HEAD]["power_at_effect_0.8"]
        better = [m for m, r in front.items()
                  if m > N_CONTROLS_PER_INDUCTION_HEAD
                  and r["power_at_effect_0.8"] > chosen + 0.05]
        if better:
            bad.append(f"power still rises past {N_CONTROLS_PER_INDUCTION_HEAD} "
                       f"controls (at {better}); the count was fixed at the "
                       f"point where it stops")
        ranks = [r["mean_score_rank_of_retained_induction_heads"]
                 for m, r in sorted(front.items())]
        if not (ranks[0] > ranks[-1]):
            bad.append("adding controls no longer moves the retained induction "
                       "heads down the score ranking, which is the cost the "
                       "frontier exists to show")
    leak = {r["score_to_motif_relation"]: r
            for r in grid.get("tautology_leak", [])}
    if leak:
        if max(r["straddled_matched_sets"] for r in leak.values()) != 0:
            bad.append("a classification that is a cutoff on the score now "
                       "yields a straddled matched set")
        if leak.get(0.0, {}).get("nearest_matching_rejection", 1.0) > 0.15:
            bad.append("the nearest-by-score matching leaks even with NO "
                       "score-to-motif relation, so the leak is not the "
                       "tautology and this section names the wrong cause")
        strongest = max(leak, key=lambda m: m)
        if leak[strongest]["nearest_matching_rejection"] <= 0.5:
            bad.append(
                f"the tautology leak no longer grows with the score-to-motif "
                f"relation (at {strongest} it is "
                f"{leak[strongest]['nearest_matching_rejection']:.3f}); the "
                f"curve is the argument that the danger is the relation and "
                f"not a literal identity")
    ovl = {r["classification_correlation_with_the_score"]: r
           for r in grid.get("overlap_frontier", [])}
    if ovl.get(1.0, {}).get("mean_matched_sets", 1.0) != 0.0:
        bad.append("a classification that IS a cutoff on the score now yields "
                   "matched sets; the tautology floor of 1.000 depends on it "
                   "yielding none")

    val = {(r["family"], r["matching_key"]): r for r in doc.get("validity", [])}
    for (fam, key), r in val.items():
        if r["correct_verdict"] == "INSUFFICIENT" and \
                r["tracks_classification_rate"] > 0.15:
            bad.append(f"validity family {fam!r} under key {key!r} confirms at "
                       f"{r['tracks_classification_rate']:.3f} where the "
                       f"correct verdict is INSUFFICIENT")
    taut = val.get(("tautology", "score"))
    if taut is None:
        bad.append("validity lost the tautology family")
    else:
        if taut["refusal_rate"] < 1.0:
            bad.append(f"the tautology family is no longer refused on every "
                       f"draw ({taut['refusal_rate']:.3f}); the refusal is the "
                       f"design's answer to adjudication constraint 2")
        if taut["counterfactual_rate_nearest_matching"] <= 0.20:
            bad.append(
                f"the tautology family no longer confirms under the discarded "
                f"nearest-by-score matching "
                f"({taut['counterfactual_rate_nearest_matching']:.3f}); that "
                f"rate is the whole argument that the straddle is doing the "
                f"work rather than the family being harmless")
    for tag in ("effect-1.5", "reciprocal-1.5"):
        r = val.get((tag, "score"))
        if r is None:
            bad.append(f"validity lost the {tag} family")
        elif tag == "effect-1.5" and r["tracks_classification_rate"] < 0.5:
            bad.append("the confirming branch no longer fires against a planted "
                       "effect")
        elif tag == "reciprocal-1.5" and r["activation_property_rate"] < 0.5:
            bad.append("the falsification branch no longer fires against a "
                       "planted inversion; a branch nothing can trigger is not "
                       "a branch")

    lim = doc.get("limitation", {})
    band = {(r["shared_elevation_on_the_induction_band_sd"], r["matching_key"]): r
            for r in lim.get("layer_band", [])}
    hi_score = band.get((1.0, "score"), {}).get("tracks_classification_rate", 0.0)
    hi_layer = band.get((1.0, "score_and_layer"), {}).get(
        "tracks_classification_rate", 1.0)
    if hi_score <= 0.15:
        bad.append("a shared elevation across the induction heads' layer band "
                   "no longer confounds the score key; that is the limitation "
                   "this section exists to state")
    if hi_layer > 0.15:
        bad.append(f"the layer key no longer removes the band confound "
                   f"({hi_layer:.3f}); it is the whole reason the key is a "
                   f"registered decision")
    pw = {(r["planted_effect"], r["matching_key"]): r
          for r in lim.get("layer_band_power", [])}
    if pw.get((1.5, "score"), {}).get("power", 0.0) <= \
            pw.get((1.5, "score_and_layer"), {}).get("power", 1.0):
        bad.append("the layer key no longer costs power, so the decision it "
                   "presents is not a trade and the record should say so")

    fl = doc.get("floor_arithmetic", {})
    if fl.get("no_informative_sets", {}).get("attainable_floor") != 1.0:
        bad.append("the no-informative-sets floor is no longer 1.000")
    return bad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=300)
    ap.add_argument("--permutation-draws", type=int, default=399)
    ap.add_argument("--spread-draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=_SEED)
    ap.add_argument("--write", action="store_true",
                    help="write the artifact; without it, measure and print only")
    ap.add_argument("--check", action="store_true",
                    help="re-read the committed artifact and report on it")
    args = ap.parse_args(argv)

    if args.check:
        if not OUT_PATH.exists():
            print(f"{OUT_PATH.relative_to(ROOT)} does not exist")
            return 1
        doc = json.loads(OUT_PATH.read_text())
        bad = check_record(doc)
        for b in bad:
            print(f"FAIL {b}")
        print(f"{'FAILED' if bad else 'ok'}: {len(bad)} finding(s), "
              f"generated in {doc.get('elapsed_seconds')}s")
        return 1 if bad else 0

    alpha = _alpha()
    t0 = time.time()

    print("the registered null ...", flush=True)
    rn = registered_null_section(args.replicates, args.permutation_draws,
                                 args.spread_draws, args.seed + 11, alpha)
    for r in rn["rejection_rates"]:
        print(f"  {r['statistic']:22s} h0={r['h0']:.3f} "
              f"effect={r['genuine_effect']:.3f} "
              f"discrimination={r['discrimination']:+.3f}", flush=True)
    for r in rn["spread_under_permutation"]["rows"]:
        print(f"  rho={r['classification_correlation_with_the_score']:.2f} "
              f"observed group score sd {r['observed_induction_group_score_sd']:.4f} "
              f"vs permuted mean {r['permuted_mean_sd']:.4f} "
              f"({r['permuted_mean_over_observed']:.2f}x, "
              f"{r['share_at_or_below_the_observed']:.4f} of draws at or below)",
              flush=True)

    print("\nselection attenuation ...", flush=True)
    att = selection_attenuation(args.replicates, args.seed + 22)
    for r in att:
        print(f"  k={r['n_induction_heads']:>3} r_ind={r['spearman_induction']:.3f} "
              f"r_ctrl={r['spearman_control']:.3f} "
              f"slope {r['slope_induction']:.3f}/{r['slope_control']:.3f} "
              f"sd ratio {r['slope_sd_ratio']:.1f}", flush=True)

    print("\nthe discarded matching ...", flush=True)
    strad = straddle_section(args.replicates, args.seed + 33, alpha)
    for r in strad["rows"]:
        print(f"  curvature={r['curvature_in_the_score_to_motif_relation']:.1f} "
              f"nearest={r['nearest_by_score_rejection']:.3f} "
              f"straddled={r['straddled_rejection']:.3f}", flush=True)

    print("\nthe tautology leak ...", flush=True)
    leak = tautology_leak(args.replicates, args.seed + 39, alpha)
    for r in leak:
        print(f"  score-to-motif={r['score_to_motif_relation']:.1f} "
              f"nearest={r['nearest_matching_rejection']:.3f} "
              f"straddled sets={r['straddled_matched_sets']}", flush=True)

    print("\nthe control count ...", flush=True)
    front = n_controls_frontier(args.replicates, args.seed + 44, alpha)
    for r in front:
        print(f"  M={r['n_controls']:>3} sets={r['mean_informative_sets']:.2f} "
              f"h0={r['h0_rejection']:.3f} power={r['power_at_effect_0.8']:.3f} "
              f"retained rank {r['mean_score_rank_of_retained_induction_heads']:.1f} "
              f"of {r['mean_score_rank_of_all_induction_heads']:.1f}", flush=True)

    print("\nthe overlap frontier ...", flush=True)
    ovl = overlap_frontier(args.replicates, args.seed + 55, alpha)
    for r in ovl:
        print(f"  rho={r['classification_correlation_with_the_score']:.2f} "
              f"sets={r['mean_matched_sets']:.2f} "
              f"floor={r['median_attainable_floor']:.3g} "
              f"clears alpha on {r['share_of_draws_whose_floor_clears_alpha']:.3f}",
              flush=True)

    print("\nvalidity and power, through the gate ...", flush=True)
    val = validity_section(args.replicates, args.seed + 66, alpha)
    for r in val:
        print(f"  {r['family']:22s} {r['matching_key']:16s} "
              f"refused={r['refusal_rate']:.3f} "
              f"tracks={r['tracks_classification_rate']:.3f} "
              f"activation={r['activation_property_rate']:.3f} "
              f"cf_nearest={r['counterfactual_rate_nearest_matching']:.3f}", flush=True)

    print("\nthe layer band ...", flush=True)
    band = layer_band_section(args.replicates, args.seed + 77, alpha)
    for r in band:
        print(f"  elevation={r['shared_elevation_on_the_induction_band_sd']:.1f} "
              f"{r['matching_key']:16s} emitted={r['emitted_rate']:.3f} "
              f"tracks={r['tracks_classification_rate']:.3f} "
              f"sets={r['mean_matched_sets']:.2f}", flush=True)
    bandpw = layer_band_power(args.replicates, args.seed + 88, alpha)
    for r in bandpw:
        print(f"  effect={r['planted_effect']:.1f} {r['matching_key']:16s} "
              f"emitted={r['emitted_rate']:.3f} power={r['power']:.3f}", flush=True)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "tools/calibrate_cross_head_association.py",
        "construction_sha256": _sha256(CONSTRUCTION_PATH),
        "alpha": alpha,
        "design": {
            "n_layers": N_LAYERS,
            "n_heads_per_layer": N_HEADS_PER_LAYER,
            "n_heads": N_HEADS,
            "n_induction_heads": K_INDUCTION,
            "n_controls_per_induction_head": N_CONTROLS_PER_INDUCTION_HEAD,
            "induction_layer_band": list(INDUCTION_LAYER_BAND),
            "_placed": ("the head grid is Pythia-410M's shape and the rest are "
                        "PLACED, not calibrated: they define the family the "
                        "rates are rates under and no distribution was "
                        "consulted for any of them"),
        },
        "synthetic_family": {
            "behavioural_score": "b ~ N(0,1) over heads",
            "classification": ("top k by c = rho_c*b + sqrt(1-rho_c^2)*z; "
                               "rho_c = 1.0 IS a cutoff on b itself"),
            "motif_rate": ("main*(b + curve*(b^2-1)) + effect*induction "
                           "+ band_elevation + N(0,1)"),
            "_note": ("`main` is the score-to-motif relation the tautology "
                      "risk is about -- present under H0 and H1 alike. "
                      "`effect` is the prediction's own quantity: motif "
                      "carried by an induction head OVER a control at the same "
                      "behavioural score"),
        },
        "replicates": args.replicates,
        "seed": args.seed,
        "registered_null": rn,
        "degeneracy": {
            "selection_attenuation": att,
            "_note": ("one population, ONE relation, no interaction present. "
                      "The correlation contrast reads large and negative -- the "
                      "falsifier's direction -- on nothing but where the "
                      "classification cut. The slope is unbiased and its "
                      "spread is the same fact in the other currency"),
        },
        "grid": {
            "straddle": strad,
            "n_controls_frontier": front,
            "tautology_leak": leak,
            "overlap_frontier": ovl,
        },
        "validity": val,
        "limitation": {
            "layer_band": band,
            "layer_band_power": bandpw,
            "_note": ("induction heads cluster in a band of layers and a shared "
                      "elevation across that band is invisible to a control "
                      "matched on score alone. Unlike 6i's shared unit factor "
                      "and 6q's fixed offset this one CAN be removed, by "
                      "matching within the layer, and the price is in "
                      "layer_band_power beside it -- fewer informative sets, "
                      "fewer draws emitting a p-value at all, and less power"),
        },
        "floor_arithmetic": floor_arithmetic(alpha),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    findings = check_record(doc)
    doc["self_check"] = {"findings": findings, "ok": not findings}
    print("\nself-check:", "ok" if not findings else f"{len(findings)} finding(s)")
    for f in findings:
        print("  FAIL", f)

    if not args.write:
        print("\n(not written: pass --write)")
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} in {doc['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
