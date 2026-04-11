"""
threshold_analysis.py — Detection threshold for V-repulsive violations.

The README notes an empirical threshold rep_frac × β ≈ 2.8 below which
neither the displacement nor rescaling tests detect V-repulsive dynamics.
This module derives the threshold from data rather than reading it off a
table, quantifies its uncertainty, and tests whether β adds independent
predictive value over rep_frac alone.

Functions
---------
collect_threshold_features   : aggregate per-run features from verdict dicts
fit_detection_threshold      : logistic regression of n_violations > 0
                               on rep_frac × qk_norm
perlayer_partial_correlation : per-layer rep_frac vs violation indicator,
                               controlling for layer depth
print_threshold_summary      : terminal output
"""

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def collect_threshold_features(verdict_list: list) -> dict:
    """
    Extract per-run features needed for threshold analysis.

    Parameters
    ----------
    verdict_list : list of verdict dicts from build_verdict_v2

    Returns
    -------
    dict with parallel arrays (one entry per run):
      model_names, prompt_keys,
      rep_frac_mean       : mean OV repulsive fraction across layers
      qk_norm_mean        : mean QK spectral norm (β proxy)
      coupling_product    : rep_frac_mean × qk_norm_mean
      n_violations        : n_violations at β=1.0
      has_violations      : bool array (n_violations > 0)
      v_score             : continuous V-score if present
    """
    model_names      = []
    prompt_keys      = []
    rep_frac_means   = []
    qk_norm_means    = []
    coupling_prods   = []
    n_violations_arr = []
    has_violations   = []
    v_scores         = []

    for v in verdict_list:
        model_names.append(v.get("model", ""))
        prompt_keys.append(v.get("prompt", ""))

        # Rep frac: prefer per-layer mean, fall back to shared
        rep = v.get("ov_frac_repulsive_mean", v.get("ov_frac_repulsive", float("nan")))
        rep_frac_means.append(rep)

        # QK norm: stored in ov_data summary; use what was propagated to verdict
        # (may not be present in older verdicts)
        qk = v.get("qk_norm_mean", float("nan"))
        qk_norm_means.append(qk)
        coupling_prods.append(rep * qk if np.isfinite(rep) and np.isfinite(qk) else float("nan"))

        nv = v.get("beta1.0_n_violations", 0)
        n_violations_arr.append(nv)
        has_violations.append(bool(nv > 0))
        v_scores.append(v.get("v_score", float("nan")))

    return {
        "model_names":    model_names,
        "prompt_keys":    prompt_keys,
        "rep_frac_mean":  np.array(rep_frac_means),
        "qk_norm_mean":   np.array(qk_norm_means),
        "coupling_product": np.array(coupling_prods),
        "n_violations":   np.array(n_violations_arr),
        "has_violations": np.array(has_violations),
        "v_scores":       np.array(v_scores),
    }


# ---------------------------------------------------------------------------
# Logistic regression threshold fit
# ---------------------------------------------------------------------------

def fit_detection_threshold(features: dict) -> dict:
    """
    Fit a logistic regression of has_violations ~ coupling_product.

    The decision boundary (where P(violation) = 0.5) gives the empirical
    threshold for the coupling product rep_frac × β.

    Also fits:
      - rep_frac alone (to test whether β adds value)
      - qk_norm alone  (to test whether spectral norm is a predictor)

    Parameters
    ----------
    features : dict from collect_threshold_features

    Returns
    -------
    dict with per-predictor logistic fits and decision boundaries
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils import check_array
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    y = features["has_violations"].astype(int)

    if y.sum() == 0 or (~y.astype(bool)).sum() == 0:
        return {"applicable": False,
                "reason": "all runs have violations or none do"}

    results = {"n_runs": len(y), "n_with_violations": int(y.sum())}

    def _fit_logistic(X_raw, name):
        """Fit one logistic model and return decision boundary + AUC."""
        mask = np.isfinite(X_raw)
        if mask.sum() < 5:
            return {"predictor": name, "applicable": False, "reason": "too few finite values"}

        Xf = X_raw[mask].reshape(-1, 1)
        yf = y[mask]

        if not has_sklearn:
            # Fallback: use scipy to minimise log-loss manually
            from scipy.optimize import minimize
            from scipy.special import expit

            def neg_log_likelihood(params):
                a, b = params
                p = expit(a * Xf.ravel() + b)
                p = np.clip(p, 1e-9, 1 - 1e-9)
                return -float(np.sum(yf * np.log(p) + (1 - yf) * np.log(1 - p)))

            res = minimize(neg_log_likelihood, [1.0, -2.0], method="Nelder-Mead")
            a, b = res.x
            boundary = -b / (a + 1e-12)
            return {
                "predictor":          name,
                "applicable":         True,
                "coef":               float(a),
                "intercept":          float(b),
                "decision_boundary":  float(boundary),
                "method":             "scipy_nelder_mead",
            }

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Xf, yf)
        coef       = float(clf.coef_[0][0])
        intercept  = float(clf.intercept_[0])
        boundary   = -intercept / (coef + 1e-12)

        # AUC (manual — avoids sklearn.metrics import for minimal dependencies)
        proba = clf.predict_proba(Xf)[:, 1]
        pairs = [(proba[i] > proba[j]) + 0.5 * (proba[i] == proba[j])
                 for i in range(len(yf)) for j in range(len(yf))
                 if yf[i] == 1 and yf[j] == 0]
        auc = float(np.mean(pairs)) if pairs else float("nan")

        return {
            "predictor":          name,
            "applicable":         True,
            "coef":               coef,
            "intercept":          intercept,
            "decision_boundary":  boundary,
            "auc":                auc,
            "n_samples":          int(mask.sum()),
        }

    results["coupling_product"] = _fit_logistic(features["coupling_product"],
                                                 "rep_frac × qk_norm")
    results["rep_frac_only"]    = _fit_logistic(features["rep_frac_mean"], "rep_frac")
    results["qk_norm_only"]     = _fit_logistic(features["qk_norm_mean"],  "qk_norm")

    # Compare AUCs: does coupling_product beat rep_frac alone?
    auc_coup = results["coupling_product"].get("auc", float("nan"))
    auc_rep  = results["rep_frac_only"].get("auc", float("nan"))
    if np.isfinite(auc_coup) and np.isfinite(auc_rep):
        results["beta_adds_value"] = auc_coup > auc_rep + 0.02
        results["auc_improvement"] = float(auc_coup - auc_rep)
    else:
        results["beta_adds_value"] = None

    return results


# ---------------------------------------------------------------------------
# Per-layer partial correlation
# ---------------------------------------------------------------------------

def perlayer_partial_correlation(
    ov_data_list: list,
    phase1_events_list: list,
    beta: float = 1.0,
) -> dict:
    """
    For each model, compute Spearman correlation of per-layer rep_frac vs
    violation indicator, and partial correlation controlling for layer depth.

    If rep_frac predicts violations independently of depth (which also
    correlates with rep_frac in GPT-2 due to the early-repulsive gradient),
    the partial correlation after removing depth should remain significant.

    Parameters
    ----------
    ov_data_list       : list of ov_data dicts (one per model)
    phase1_events_list : list of phase1_events dicts

    Returns
    -------
    dict with per-model results:
      raw_rho, raw_pval       : Spearman rep_frac vs violation indicator
      partial_rho, partial_pval : partial correlation controlling for depth
      depth_rho               : rep_frac vs depth (confound strength)
    """
    def _rank_residual(x, z):
        """Residuals of x ~ z in rank space (linear regression on ranks)."""
        from scipy.stats import rankdata
        rx = rankdata(x).astype(float)
        rz = rankdata(z).astype(float)
        cov = np.cov(rx, rz)
        slope = cov[0, 1] / (np.var(rz) + 1e-12)
        return rx - slope * rz

    results = []

    for ov_data, events in zip(ov_data_list, phase1_events_list):
        if not ov_data.get("is_per_layer"):
            continue

        decomps    = ov_data["decomps"]
        rep_frac   = np.array([d["frac_repulsive"] for d in decomps])
        violations = set(events["energy_violations"].get(beta, []))
        n          = len(rep_frac)

        if n < 6:
            continue

        violation_ind = np.array([1.0 if i in violations else 0.0 for i in range(n)])
        depth         = np.arange(n, dtype=float)

        # Raw correlation
        raw_rho, raw_pval = spearmanr(rep_frac, violation_ind)

        # Depth confound: rep_frac vs depth
        depth_rho, _ = spearmanr(rep_frac, depth)

        # Partial correlation: rep_frac vs violation_ind, partialling out depth
        resid_rep  = _rank_residual(rep_frac,    depth)
        resid_viol = _rank_residual(violation_ind, depth)
        part_rho, part_pval = spearmanr(resid_rep, resid_viol)

        results.append({
            "model":         events.get("model", "unknown"),
            "prompt":        events.get("prompt", ""),
            "n_layers":      n,
            "n_violations":  int(violation_ind.sum()),
            "raw_rho":       float(raw_rho),
            "raw_pval":      float(raw_pval),
            "partial_rho":   float(part_rho),
            "partial_pval":  float(part_pval),
            "depth_rho":     float(depth_rho),
            "depth_confound_strong": abs(depth_rho) > 0.4,
        })

    if not results:
        return {"applicable": False, "reason": "No per-layer models found"}

    # Aggregate
    raw_rhos     = [r["raw_rho"]     for r in results if np.isfinite(r["raw_rho"])]
    partial_rhos = [r["partial_rho"] for r in results if np.isfinite(r["partial_rho"])]
    n_raw_sig    = sum(1 for r in results if r["raw_pval"]     < 0.05)
    n_part_sig   = sum(1 for r in results if r["partial_pval"] < 0.05)

    return {
        "applicable":         True,
        "per_model":          results,
        "mean_raw_rho":       float(np.mean(raw_rhos))     if raw_rhos     else float("nan"),
        "mean_partial_rho":   float(np.mean(partial_rhos)) if partial_rhos else float("nan"),
        "n_raw_significant":  n_raw_sig,
        "n_partial_significant": n_part_sig,
        "n_runs":             len(results),
        "interpretation": (
            "If partial_rho remains significant after controlling for depth, "
            "rep_frac predicts violations independently — not just because "
            "both cluster in early layers."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_threshold_summary(fit: dict, partial: dict) -> None:
    """Print concise threshold analysis output."""
    print("\n  Detection threshold analysis (fix 13):")

    if not fit.get("applicable", True):
        print(f"    Threshold fit: {fit.get('reason', 'not applicable')}")
    else:
        print(f"    Logistic regression on {fit.get('n_runs', 0)} runs "
              f"({fit.get('n_with_violations', 0)} with violations):")
        for key, label in [("coupling_product", "rep_frac × qk_norm"),
                           ("rep_frac_only",    "rep_frac only"),
                           ("qk_norm_only",     "qk_norm only")]:
            r = fit.get(key, {})
            if r.get("applicable"):
                print(f"      {label:25s}  "
                      f"boundary={r['decision_boundary']:.3f}  "
                      f"AUC={r.get('auc', float('nan')):.3f}")
        if fit.get("beta_adds_value") is not None:
            adds = "YES" if fit["beta_adds_value"] else "NO"
            print(f"    β (qk_norm) adds value over rep_frac alone: {adds}  "
                  f"(ΔAUC={fit.get('auc_improvement', 0):+.3f})")

    if partial.get("applicable"):
        print(f"\n    Per-layer partial correlation ({partial['n_runs']} model×prompt):")
        print(f"      Mean raw ρ(rep_frac, violation):     {partial['mean_raw_rho']:+.3f}  "
              f"sig: {partial['n_raw_significant']}/{partial['n_runs']}")
        print(f"      Mean partial ρ (controlling depth):  {partial['mean_partial_rho']:+.3f}  "
              f"sig: {partial['n_partial_significant']}/{partial['n_runs']}")
        print(f"      {partial['interpretation']}")
