"""
p6_subspace/probe_subspace.py — Track B/D: Linear probes on real vs imaginary projections.

Tests prediction P6-R4: cluster membership is recoverable from the real
subspace alone (z_i^S = U_pos^T x_i) with near-full accuracy, while the
imaginary subspace projection (z_i^A) gives near-chance accuracy.

If true, all cluster structure is linearly encoded in a low-dimensional
real subspace — the imaginary subspace adds nothing for cluster identity.

The probe is a simple logistic regression on:
  (a) Full activation x_i                      → accuracy_full
  (b) Real projection z_i^S = U_pos^T x_i      → accuracy_real
  (c) Imaginary projection z_i^A = U_A^T x_i   → accuracy_imag
  (d) Random projection of same dimension as S  → accuracy_random (baseline)

Cross-validation is used (stratified k-fold) to handle small token counts.

Falsifiable prediction tested
------------------------------
P6-R4 : accuracy_real ≈ accuracy_full  AND  accuracy_imag ≈ chance (1/K).
         Criterion: accuracy_real ≥ 0.9 * accuracy_full
                AND accuracy_imag ≤ 1/K + 0.10

Bug fixes applied in this version
-----------------------------------
1. Float truthiness  (was: `if (acc_real and acc_full)`)
   Zero accuracy evaluates as False in Python, causing those branches to return
   None instead of evaluating the actual condition. Replaced with explicit
   `is not None` checks throughout.

2. Per-layer chance level for p6_r4_imag_near_chance
   The number of clusters K decreases as layers merge, so chance = 1/K rises
   layer-by-layer. The previous code took chance from only the first probed layer
   and used it globally, making the imag threshold inconsistent across layers.
   Now evaluates `acc_imag <= chance + 0.10` per layer using that layer's own K,
   then aggregates: p6_r4_imag_near_chance passes if the majority of layers pass.

3. multi_class="auto" removed from LogisticRegression
   Deprecated in sklearn ≥ 1.5; the solver auto-selects the correct strategy.

4. Equal-capacity imaginary probe (probe_all_channels / run_probe_subspace)
   U_A has 1818 dims vs U_S ~230 dims for ALBERT. The original imag probe
   trained a logistic regression on 1818 features vs ~230, giving U_A a free
   capacity advantage. Added imag_matched channel: U_A subsampled to dim(U_S)
   columns before projection. P6-R4 verdict now uses imag_matched.
   acc_imag (biased) is retained in vc and per_layer for diagnostic comparison.

Functions
---------
probe_accuracy       : fit and evaluate linear probe on given projections
probe_all_channels   : run (a)-(d) at one layer, controls for capacity
run_probe_subspace   : full pipeline → SubResult
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN
from core.population import resolve_population_mask


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------

import warnings  # add to existing imports if not present

def probe_accuracy(
    Z:          np.ndarray,
    labels:     np.ndarray,
    n_splits:   int = 5,
    max_iter:   int = 1000,
    population: str = "clustered",
) -> dict:
    """
    Fit and evaluate a linear probe of `labels` from `Z` via stratified
    k-fold CV.

    Parameters
    ----------
    population : which tokens enter the probe (transition plan v2, core
        analysis primitives -- population selector). Default "clustered"
        reproduces this function's exact pre-existing behavior (`labels
        >= 0`, i.e. noise silently dropped) -- that default is intentional,
        not just backward-compatibility inertia: a classification probe
        needs at least the option to exclude the population with no
        stable class identity. Pass population="all" to keep noise
        tokens too, in which case -1 becomes its own class the probe must
        also separate from every real cluster -- a direct test of whether
        "unclustered" is itself a linearly-readable population rather
        than an artifact of dropping it before the probe ever sees it.
        See core.population for the full spec.
    """
    valid = resolve_population_mask(labels, population)
    Z_v   = Z[valid].astype(np.float32)
    L_v   = labels[valid]

    n_classes = len(np.unique(L_v))
    chance    = 1.0 / max(n_classes, 1)

    if n_classes < 2 or len(Z_v) < 2 * n_classes:
        return {
            "mean_accuracy": chance,
            "std_accuracy":  0.0,
            "n_samples":     int(valid.sum()),
            "n_classes":     n_classes,
            "chance_level":  chance,
        }

    le = LabelEncoder()
    y  = le.fit_transform(L_v)

    # Bound n_splits by minimum class size to avoid StratifiedKFold warning
    # when a cluster has fewer members than requested folds.
    min_class_count = int(np.bincount(y).min())
    n_splits_actual = min(n_splits, n_classes, len(Z_v), min_class_count)
    n_splits_actual = max(n_splits_actual, 2)  # need at least 2 folds

    # If even 2-fold CV isn't possible (min class has 1 member), fall back to
    # returning chance level rather than crashing or warning.
    if min_class_count < 2:
        return {
            "mean_accuracy": chance,
            "std_accuracy":  0.0,
            "n_samples":     int(valid.sum()),
            "n_classes":     n_classes,
            "chance_level":  chance,
        }

    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=0)

    accs = []
    # Suppress the type_of_target "looks like regression" warning — it fires
    # when n_classes > n_samples/2 (many small clusters), which is expected here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning,
        )
        for train_idx, test_idx in cv.split(Z_v, y):
            clf = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                C=1.0,
            )
            clf.fit(Z_v[train_idx], y[train_idx])
            accs.append(float(clf.score(Z_v[test_idx], y[test_idx])))

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "n_samples":     int(valid.sum()),
        "n_classes":     n_classes,
        "chance_level":  chance,
    }


# ---------------------------------------------------------------------------
# All-channel probe at one layer
# ---------------------------------------------------------------------------


def probe_all_channels(
    X:          np.ndarray,
    labels:     np.ndarray,
    U_pos:      np.ndarray,     # (d, r_pos) — attractive real subspace
    U_A:        np.ndarray,     # (d, r_A)   — imaginary subspace
    U_S:        np.ndarray | None = None,  # (d, r_S) — full real subspace (U_pos ∪ U_neg)
    seed:       int = 42,
    population: str = "clustered",
    ) -> dict:
    """
    Run probes on full activations and four projection channels:
      "full"        — raw activations                    (d dims)
      "real"        — U_pos projection                   (r_pos dims)
      "imag"        — U_A projection                     (r_A dims)  [biased if r_A >> r_pos]
      "imag_matched"— U_A subsampled to r_S directions   (r_S dims)  [equal capacity to real_full]
      "random"      — random projection of r_pos dims    (r_pos dims) [baseline]

    "imag_matched" is the fair comparison: same number of input features as
    U_S (or U_pos if U_S not supplied).  If acc_imag_matched > acc_real_full,
    the imaginary subspace genuinely encodes more cluster information per
    direction.

    Parameters
    ----------
    U_S : full real subspace (U_pos ∪ U_neg).  When supplied, "real_full" channel
          uses U_S and "imag_matched" is subsampled to U_S.shape[1] columns.
          When None, falls back to U_pos for both.
    population : forwarded to every probe_accuracy call below (see that
          function's docstring). Default "clustered" — unchanged behavior.
    """
    d    = X.shape[1]
    rng  = np.random.default_rng(seed=seed)

    # Decide reference basis for capacity matching
    U_real = U_S if (U_S is not None and U_S.shape[1] > 0) else U_pos
    r_ref  = U_real.shape[1]   # capacity reference dimension

    # --- full ---
    res_full = probe_accuracy(X, labels, population=population)

    # --- real (U_pos, as per original P6-R4 definition) ---
    res_real = (
        probe_accuracy(X @ U_pos, labels, population=population)
        if U_pos.shape[1] > 0
        else _empty_probe_result()
    )

    # --- real_full (U_S, includes U_neg — fairer for total real capacity) ---
    res_real_full = (
        probe_accuracy(X @ U_real, labels, population=population)
        if U_real.shape[1] > 0
        else _empty_probe_result()
    )

    # --- imag (full U_A, biased by dimension) ---
    res_imag = (
        probe_accuracy(X @ U_A, labels, population=population)
        if U_A.shape[1] > 0
        else _empty_probe_result()
    )

    # --- imag_matched (U_A subsampled to r_ref columns, equal capacity) ---
    if U_A.shape[1] > 0 and r_ref > 0:
        if U_A.shape[1] <= r_ref:
            # U_A is already smaller — use as-is (rare but possible)
            U_A_sub = U_A
        else:
            # Random column subset, reproducible via seed
            idx     = rng.choice(U_A.shape[1], size=r_ref, replace=False)
            U_A_sub = U_A[:, idx]
        res_imag_matched = probe_accuracy(X @ U_A_sub, labels, population=population)
    else:
        res_imag_matched = _empty_probe_result()

    # --- random (same dimension as U_pos, original baseline) ---
    r_rand = max(U_pos.shape[1], 1)
    Q, _   = np.linalg.qr(rng.standard_normal((d, r_rand)))
    res_rand = probe_accuracy(X @ Q[:, :r_rand], labels, population=population)

    return {
        "full":         res_full,
        "real":         res_real,         # U_pos only (original P6-R4)
        "real_full":    res_real_full,    # U_S = U_pos ∪ U_neg
        "imag":         res_imag,         # full U_A (biased)
        "imag_matched": res_imag_matched, # U_A subsampled to r_ref (fair)
        "random":       res_rand,
        "dim_r_ref":    r_ref,            # capacity reference stored for reporting
        "dim_r_A":      U_A.shape[1],
    }


def _empty_probe_result() -> dict:
    return {
        "mean_accuracy": None, "std_accuracy": None,
        "n_samples": 0, "n_classes": 0, "chance_level": None,
    }

# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_probe_subspace(ctx: dict) -> SubResult:
    """
    Track B/D sub-experiment: linear probes on real vs imaginary projections.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n, d)
    labels_per_layer      : list of (n,) HDBSCAN labels
    layer_type_labels     : list of str
    layer_names           : list of str
    projectors            : output of subspace_build

    Optional ctx keys
    -----------------
    probe_layers : list of str — which layer names to probe (default: plateau + merge)
    population   : population selector forwarded to every probe (default
                   "clustered" — unchanged behavior; see core.population
                   and probe_accuracy's docstring for the full spec, e.g.
                   "all" to make -1 a probed class rather than dropping it).
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]
    projectors  = ctx["projectors"]

    probe_layers_override = ctx.get("probe_layers", None)
    population            = ctx.get("population", "clustered")

    proj_entries = projectors["per_layer"]
    if len(proj_entries) == 1 and len(acts) > 1:
        proj_entries = proj_entries * len(acts)

    per_layer_results = []

    for L, (X, lab, ltype, lname, pe) in enumerate(zip(
        acts, labels, layer_types, layer_names, proj_entries
    )):
        if probe_layers_override is not None:
            if lname not in probe_layers_override:
                continue
        else:
            if ltype not in ("plateau", "merge"):
                continue

        if int(resolve_population_mask(lab, population).sum()) < 10:
            continue

        res = probe_all_channels(
            X, lab, pe["U_pos"], pe["U_A"], U_S=pe.get("U_S"), population=population
        )
        res["layer_name"] = lname
        res["layer_type"] = ltype
        per_layer_results.append(res)

    if not per_layer_results:
        return SubResult(
            name="probe_subspace",
            applicable=False,
            payload={},
            summary_lines=["probe_subspace: no applicable layers found"],
            verdict_contribution={},
        )

    # -----------------------------------------------------------------------
    # Aggregate accuracy means
    # -----------------------------------------------------------------------
    def _mean_acc(results, channel):
        vals = [
            r[channel]["mean_accuracy"] for r in results
            if r[channel]["mean_accuracy"] is not None
        ]
        return float(np.mean(vals)) if vals else None

    acc_full         = _mean_acc(per_layer_results, "full")
    acc_real         = _mean_acc(per_layer_results, "real")
    acc_real_full    = _mean_acc(per_layer_results, "real_full")
    acc_imag         = _mean_acc(per_layer_results, "imag")
    acc_imag_matched = _mean_acc(per_layer_results, "imag_matched")
    acc_random       = _mean_acc(per_layer_results, "random")

    # -----------------------------------------------------------------------
    # P6-R4 verdict — Fix 1: explicit None checks (not float truthiness)
    #                 Fix 2: per-layer chance for imag threshold
    # -----------------------------------------------------------------------

    # (a) real sufficient: acc_real >= 0.9 * acc_full
    if acc_real is not None and acc_full is not None:
        p6_r4_real_sufficient = bool(acc_real >= 0.9 * acc_full)
    else:
        p6_r4_real_sufficient = None

    # (b) imag near chance: evaluated per layer using that layer's own chance level
    imag_chance_per_layer: list[bool] = []
    for r in per_layer_results:
        acc_i    = r["imag_matched"]["mean_accuracy"]   # WAS: r["imag"]["mean_accuracy"]
        chance_i = r["full"]["chance_level"]
        if acc_i is not None and chance_i is not None:
            imag_chance_per_layer.append(bool(acc_i <= chance_i + 0.10))

    if imag_chance_per_layer:
        n_pass_imag            = sum(imag_chance_per_layer)
        p6_r4_imag_near_chance = bool(n_pass_imag > len(imag_chance_per_layer) // 2)
    else:
        n_pass_imag            = 0
        p6_r4_imag_near_chance = None

    p6_r4_satisfied = bool(p6_r4_real_sufficient and p6_r4_imag_near_chance)

    # -----------------------------------------------------------------------
    # Payload
    # -----------------------------------------------------------------------
    payload = {
        "n_layers_probed":            len(per_layer_results),
        "mean_accuracy_full":         acc_full,
        "mean_accuracy_real":         acc_real,
        "mean_accuracy_imag":         acc_imag,
        "mean_accuracy_random":       acc_random,
        "p6_r4_real_sufficient":      p6_r4_real_sufficient,
        "p6_r4_imag_near_chance":     p6_r4_imag_near_chance,
        "p6_r4_imag_layers_passing":  n_pass_imag,
        "p6_r4_imag_layers_total":    len(imag_chance_per_layer),
        "p6_r4_satisfied":            p6_r4_satisfied,
        "per_layer": [
            {
                "layer_name":       r["layer_name"],
                "layer_type":       r["layer_type"],
                "acc_full":         r["full"]["mean_accuracy"],
                "acc_real":         r["real"]["mean_accuracy"],
                "acc_real_full":    r["real_full"]["mean_accuracy"],
                "acc_imag":         r["imag"]["mean_accuracy"],
                "acc_imag_matched": r["imag_matched"]["mean_accuracy"],
                "acc_random":       r["random"]["mean_accuracy"],
                "dim_r_ref":        r.get("dim_r_ref"),
                "dim_r_A":          r.get("dim_r_A"),
                "n_classes":        r["full"]["n_classes"],
                "n_samples":        r["full"]["n_samples"],
                "chance_level":     r["full"]["chance_level"],
                        }
            for r in per_layer_results
        ],
    }

    # -----------------------------------------------------------------------
    # Summary lines
    # -----------------------------------------------------------------------
    lines = [
        SEP_THICK,
        "LINEAR PROBE: REAL vs IMAGINARY SUBSPACE  [Track B/D]",
        SEP_THICK,
        f"Layers probed:       {len(per_layer_results)}",
        "  (chance level is per-layer; K decreases as clusters merge)",
        "",
        "Probe accuracy averaged across probed layers:",
        _bullet("Full activation x_i",           acc_full),
        _bullet("Real projection z_i^S (U_pos)",  acc_real),
        _bullet("Random projection (baseline)",   acc_random),
        # _bullet("Imaginary projection z_i^A",     acc_imag),
        _bullet("Imaginary projection (full U_A, biased)",    acc_imag),
        _bullet("Imaginary projection (matched dim, fair)",   acc_imag_matched),
        "",
        "  Note: 'matched dim' subsamples U_A to dim(U_S) columns.",
        f"  dim(U_S)={per_layer_results[0].get('dim_r_ref','?')}  "
        f"dim(U_A)={per_layer_results[0].get('dim_r_A','?')}",
        "",
        "P6-R4: z_i^S preserves cluster membership; z_i^A near chance.",
        "  Criteria:  acc_real >= 0.9 * acc_full  (global means)",
        "             acc_imag <= chance + 0.10   (per-layer, majority vote)",
        _bullet("real sufficient (>= 0.9 * full)", p6_r4_real_sufficient),
        _bullet(
            f"imag near chance (majority: {n_pass_imag}/{len(imag_chance_per_layer)} layers)",
            p6_r4_imag_near_chance,
        ),
        _verdict_line(
            "P6-R4",
            p6_r4_satisfied,
            f"real={_fmt(acc_real)} full={_fmt(acc_full)} imag={_fmt(acc_imag)}",
        ),
        "",
        "Per-layer probe results:",
        f"  {'layer':<18s} {'type':<8s} {'K':>3}"
        f"  {'acc_full':>9} {'acc_real':>9} {'acc_imag':>9} {'acc_rand':>9}"
        f"  {'chance':>7} {'imag≤c+0.10':>11}",
    ]

    for r, imag_ok in zip(per_layer_results, imag_chance_per_layer):
        acc_i    = r["imag"]["mean_accuracy"]
        chance_i = r["full"]["chance_level"]
        lines.append(
            f"  {r['layer_name']:<18s} {r['layer_type']:<8s} {r['full']['n_classes']:>3d}"
            f"  {_fmt(r['full']['mean_accuracy']):>9}"
            f"  {_fmt(r['real']['mean_accuracy']):>9}"
            f"  {_fmt(acc_i):>9}"
            f"  {_fmt(r['random']['mean_accuracy']):>9}"
            f"  {_fmt(chance_i):>7}"
            f"  {'pass' if imag_ok else 'fail':>11}"
        )

    vc = {
        "probe_acc_full":             acc_full,
        "probe_acc_real":             acc_real,
        "probe_acc_real_full":        acc_real_full,        # ADD
        "probe_acc_imag":             acc_imag,             # keep (biased, for comparison)
        "probe_acc_imag_matched":     acc_imag_matched,     # ADD (fair, used for verdict)
        "probe_acc_random":           acc_random,
        "probe_p6_r4_satisfied":      p6_r4_satisfied,
    }

    return SubResult(
        name="probe_subspace",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )