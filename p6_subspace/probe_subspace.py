"""
probe_subspace.py — Track B/D: Linear probes on real vs imaginary projections.

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

Functions
---------
probe_accuracy       : fit and evaluate linear probe on given projections
probe_all_channels   : run (a)-(d) at one layer
run_probe_subspace   : full pipeline → SubResult
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------

def probe_accuracy(
    Z:        np.ndarray,
    labels:   np.ndarray,
    n_splits: int = 5,
    max_iter: int = 1000,
) -> dict:
    """
    Fit a logistic regression probe and evaluate with cross-validation.

    Noise tokens (label == -1) are excluded before fitting.

    Parameters
    ----------
    Z      : (n, r) — projected activations
    labels : (n,)   — cluster labels
    n_splits: k in k-fold CV

    Returns
    -------
    dict with:
      mean_accuracy  : float
      std_accuracy   : float
      n_samples      : int
      n_classes      : int
      chance_level   : float — 1/n_classes
    """
    valid = labels >= 0
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

    n_splits_actual = min(n_splits, n_classes, len(Z_v))
    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=0)

    accs = []
    for train_idx, test_idx in cv.split(Z_v, y):
        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            # Fix 3: removed multi_class="auto" — deprecated in sklearn ≥1.5;
            # the solver selects multinomial vs OvR automatically.
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
    X:      np.ndarray,
    labels: np.ndarray,
    U_pos:  np.ndarray,
    U_A:    np.ndarray,
    seed:   int = 42,
) -> dict:
    """
    Run probes on full activations, real projection, imaginary projection,
    and a random same-dimension projection.

    Parameters
    ----------
    X      : (n, d)
    labels : (n,)
    U_pos  : (d, r_S) — attractive real subspace basis (tests P6-R4 specifically
             with U_pos, not the full U_S, as per the prediction's definition of z_i^S)
    U_A    : (d, r_A) — imaginary subspace basis

    Returns
    -------
    dict with probe results for each channel: "full", "real", "imag", "random"
    """
    d = X.shape[1]

    res_full = probe_accuracy(X, labels)

    if U_pos.shape[1] > 0:
        res_real = probe_accuracy(X @ U_pos, labels)
    else:
        res_real = {
            "mean_accuracy": None, "std_accuracy": None,
            "n_samples": 0, "n_classes": 0, "chance_level": None,
        }

    if U_A.shape[1] > 0:
        res_imag = probe_accuracy(X @ U_A, labels)
    else:
        res_imag = {
            "mean_accuracy": None, "std_accuracy": None,
            "n_samples": 0, "n_classes": 0, "chance_level": None,
        }

    r_rand = max(U_pos.shape[1], 1)
    rng    = np.random.default_rng(seed=seed)
    Q, _   = np.linalg.qr(rng.standard_normal((d, r_rand)))
    res_rand = probe_accuracy(X @ Q[:, :r_rand], labels)

    return {
        "full":   res_full,
        "real":   res_real,
        "imag":   res_imag,
        "random": res_rand,
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
    """
    acts        = ctx["activations_per_layer"]
    labels      = ctx["labels_per_layer"]
    layer_types = ctx["layer_type_labels"]
    layer_names = ctx["layer_names"]
    projectors  = ctx["projectors"]

    probe_layers_override = ctx.get("probe_layers", None)

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

        if int((lab >= 0).sum()) < 10:
            continue

        res = probe_all_channels(X, lab, pe["U_pos"], pe["U_A"])
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

    acc_full   = _mean_acc(per_layer_results, "full")
    acc_real   = _mean_acc(per_layer_results, "real")
    acc_imag   = _mean_acc(per_layer_results, "imag")
    acc_random = _mean_acc(per_layer_results, "random")

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
        acc_i    = r["imag"]["mean_accuracy"]
        chance_i = r["full"]["chance_level"]   # 1/K at this layer
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
                "layer_name":  r["layer_name"],
                "layer_type":  r["layer_type"],
                "acc_full":    r["full"]["mean_accuracy"],
                "acc_real":    r["real"]["mean_accuracy"],
                "acc_imag":    r["imag"]["mean_accuracy"],
                "acc_random":  r["random"]["mean_accuracy"],
                "n_classes":   r["full"]["n_classes"],
                "n_samples":   r["full"]["n_samples"],
                "chance_level": r["full"]["chance_level"],
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
        _bullet("Imaginary projection z_i^A",     acc_imag),
        _bullet("Random projection (baseline)",   acc_random),
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
        "probe_acc_full":           acc_full,
        "probe_acc_real":           acc_real,
        "probe_acc_imag":           acc_imag,
        "probe_acc_random":         acc_random,
        "probe_p6_r4_satisfied":    p6_r4_satisfied,
    }

    return SubResult(
        name="probe_subspace",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )