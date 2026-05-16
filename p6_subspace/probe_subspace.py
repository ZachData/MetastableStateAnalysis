"""
probe_subspace.py — Track B/D: Linear probes on real vs imaginary projections.

Tests prediction P6-R4: cluster membership is recoverable from the real
subspace alone (z_i^S = U_+^T x_i) with near-full accuracy, while the
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

Functions
---------
probe_accuracy       : fit and evaluate linear probe on given projections
run_probe_subspace   : full pipeline → SubResult

Fixes:
  1 : Float truthiness — explicit None checks (not `if (acc_real and acc_full)`)
  2 : Per-layer chance level for imag threshold (K changes as clusters merge)
  3 : Remove deprecated multi_class="auto" from LogisticRegression
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# -----------
# Core probe
# -----------

def probe_accuracy(
    Z:      np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 1000,
) -> dict:
    """
    Fit a logistic regression probe and evaluate with cross-validation.

    Noise tokens (label == -1) are excluded before fitting.

    FIX 3: Remove multi_class="auto" (deprecated in sklearn ≥1.5).
    Solver auto-selects multinomial vs OvR.

    Parameters
    ----------
    Z       : (n, r) — projected activations
    labels  : (n,)   — cluster labels
    n_splits: k in k-fold CV
    max_iter: max iterations for LogisticRegression

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

    # Encode labels to contiguous integers
    le = LabelEncoder()
    y  = le.fit_transform(L_v)

    n_splits_actual = min(n_splits, n_classes, len(Z_v))
    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=0)

    accs = []
    for train_idx, test_idx in cv.split(Z_v, y):
        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            # FIX 3: multi_class removed (auto-select)
            C=1.0,
        )
        clf.fit(Z_v[train_idx], y[train_idx])
        acc = float(clf.score(Z_v[test_idx], y[test_idx]))
        accs.append(acc)

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "n_samples":     int(valid.sum()),
        "n_classes":     n_classes,
        "chance_level":  chance,
    }


# -----------
# Full pipeline → SubResult
# -----------

def run_probe_subspace(ctx: dict):
    """
    Track B/D sub-experiment: linear probes on real vs imaginary projections.

    FIX 1: Explicit None checks (not float truthiness).
    FIX 2: Per-layer chance threshold for imag accuracy.

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

    # Broadcast single projector entry for ALBERT
    proj_entries = projectors["per_layer"]
    if len(proj_entries) == 1 and len(acts) > 1:
        proj_entries = proj_entries * len(acts)

    per_layer_results = []

    for L, (X, lab, ltype, lname, pe) in enumerate(zip(
        acts, labels, layer_types, layer_names, proj_entries
    )):
        # Only probe plateau and merge layers unless overridden
        if probe_layers_override is not None:
            if lname not in probe_layers_override:
                continue
        else:
            if ltype not in ("plateau", "merge"):
                continue

        n_valid = int((lab >= 0).sum())
        if n_valid < 10:
            continue

        U_pos = pe["U_pos"]
        U_A   = pe["U_A"]

        res_full = probe_accuracy(X, lab)
        res_real = probe_accuracy(X @ U_pos, lab) if U_pos.shape[1] > 0 else None
        res_imag = probe_accuracy(X @ U_A, lab) if U_A.shape[1] > 0 else None

        r_rand = max(U_pos.shape[1], 1)
        rng = np.random.default_rng(seed=L)
        Q, _ = np.linalg.qr(rng.standard_normal((X.shape[1], r_rand)))
        res_rand = probe_accuracy(X @ Q[:, :r_rand], lab)

        per_layer_results.append({
            "layer_name": lname,
            "layer_type": ltype,
            "full": res_full,
            "real": res_real,
            "imag": res_imag,
            "random": res_rand,
        })

    if not per_layer_results:
        return {
            "name": "probe_subspace",
            "applicable": False,
            "payload": {},
            "verdict_contribution": {},
        }

    # FIX 1: Explicit None checks
    def _mean_acc(results, channel):
        vals = [
            r[channel]["mean_accuracy"]
            for r in results
            if r[channel] is not None and r[channel].get("mean_accuracy") is not None
        ]
        return float(np.mean(vals)) if vals else None

    acc_full = _mean_acc(per_layer_results, "full")
    acc_real = _mean_acc(per_layer_results, "real")
    acc_imag = _mean_acc(per_layer_results, "imag")
    acc_random = _mean_acc(per_layer_results, "random")

    # FIX 2: Per-layer chance for imag threshold (majority vote)
    imag_chance_per_layer: list[bool] = []
    for r in per_layer_results:
        acc_i = r["imag"]["mean_accuracy"] if r["imag"] is not None else None
        chance_i = r["full"]["chance_level"]
        if acc_i is not None and chance_i is not None:
            imag_chance_per_layer.append(bool(acc_i <= chance_i + 0.10))

    if imag_chance_per_layer:
        n_pass_imag = sum(imag_chance_per_layer)
        p6_r4_imag_near_chance = bool(n_pass_imag > len(imag_chance_per_layer) // 2)
    else:
        n_pass_imag = 0
        p6_r4_imag_near_chance = None

    # FIX 1: Explicit None checks
    if acc_real is not None and acc_full is not None:
        p6_r4_real_sufficient = bool(acc_real >= 0.9 * acc_full)
    else:
        p6_r4_real_sufficient = None

    p6_r4_satisfied = bool(p6_r4_real_sufficient and p6_r4_imag_near_chance)

    return {
        "name": "probe_subspace",
        "applicable": True,
        "payload": {
            "n_layers_probed": len(per_layer_results),
            "mean_accuracy_full": acc_full,
            "mean_accuracy_real": acc_real,
            "mean_accuracy_imag": acc_imag,
            "mean_accuracy_random": acc_random,
            "p6_r4_real_sufficient": p6_r4_real_sufficient,
            "p6_r4_imag_near_chance": p6_r4_imag_near_chance,
            "p6_r4_satisfied": p6_r4_satisfied,
            "per_layer": [
                {
                    "layer_name": r["layer_name"],
                    "layer_type": r["layer_type"],
                    "acc_full": r["full"]["mean_accuracy"],
                    "acc_real": r["real"]["mean_accuracy"] if r["real"] else None,
                    "acc_imag": r["imag"]["mean_accuracy"] if r["imag"] else None,
                    "acc_random": r["random"]["mean_accuracy"],
                    "n_classes": r["full"]["n_classes"],
                    "n_samples": r["full"]["n_samples"],
                }
                for r in per_layer_results
            ],
        },
        "verdict_contribution": {
            "probe_p6_r4_satisfied": p6_r4_satisfied,
        },
    }