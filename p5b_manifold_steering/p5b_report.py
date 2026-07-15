"""
p5b_manifold/report.py — Report writing for Phase 5b.

Formats Phase 5b manifold fitting, isometry, and teleportation results
into a human-readable summary report.
"""

from __future__ import annotations

from pathlib import Path


def write_report(
    out_dir: Path,
    results: dict,
    model: str,
    prompt: str,
) -> Path:
    """
    Write Phase 5b results to a formatted text report.

    Parameters
    ----------
    out_dir : Path or str
        Directory to write report into
    results : dict
        Results dict with keys:
          - fit_summary: dict with PCA/spline metrics
          - isometry: dict with correlation and p-value data
          - teleportation: dict with merge event analysis
          - subspace: dict with subspace isometry scores (optional)
    model : str
        Model name (e.g., "gpt2-large")
    prompt : str
        Prompt key (e.g., "wiki_paragraph")

    Returns
    -------
    Path
        Path to written report file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "p5b_report.txt"

    lines = []
    W = lines.append

    # ========================================================================
    # Header
    # ========================================================================
    W("=" * 72)
    W("PHASE 5b — METASTABLE STATES AS ACTIVATION MANIFOLD CONTROL POINTS")
    W("=" * 72)
    W("")
    W("CONTEXT")
    W("-" * 40)
    W("Wurgaft et al. (2026) show that concept centroids lie on an activation")
    W("manifold Mh that is approximately isometric to a behavior manifold My")
    W("(fit to output probability distributions). This phase tests whether our")
    W("unsupervised HDBSCAN cluster centroids (from Phase 1) are the same")
    W("objects as Wurgaft's concept centroids.")
    W("")
    W(f"  Model  : {model}")
    W(f"  Prompt : {prompt}")
    W("")

    # ========================================================================
    # Sub-experiment A: Manifold Fitting
    # ========================================================================
    W("=" * 72)
    W("SUB-EXP A — MANIFOLD FITTING")
    W("=" * 72)
    fs = results.get("fit_summary", {})
    W(f"  PCA explained variance (32d)   : {fs.get('pca_explained_var', float('nan')):.4f}")
    W(f"  PCA dims needed for 80%        : {fs.get('pca_n_dims_for_80pct', '?')}")
    W(f"  Mh spline residual RMS         : {fs.get('mh_spline_residual_rms', float('nan')):.4f}")
    W(f"  My spline residual RMS         : {fs.get('my_spline_residual_rms', float('nan')):.4f}")
    W(f"  N control points               : {fs.get('n_control_points', '?')}")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-A1", fs.get("p5b_a1_pass"),
        "32d PCA retains >= 80% variance")
    _pf(W, "P5b-A2", fs.get("p5b_a2_pass"),
        "Spline residuals < 10% of inter-centroid distances")
    W("")

    # ========================================================================
    # Sub-experiment B: Isometry Test
    # ========================================================================
    W("=" * 72)
    W("SUB-EXP B — ISOMETRY TEST")
    W("=" * 72)
    iso = results.get("isometry", {})
    W(f"  Pearson r (manifold vs behavior): {iso.get('r_manifold', float('nan')):.4f}")
    W(f"  Pearson r (linear vs behavior)  : {iso.get('r_linear',   float('nan')):.4f}")
    W(f"  p-value (manifold)              : {iso.get('p_manifold', float('nan')):.2e}")
    W(f"  N pairs                         : {iso.get('n_pairs', '?')}")
    W("")
    W("  Wurgaft reference values (concept-labeled tasks):")
    W("    weekdays r=0.99, months r=0.89, letters r=0.999, ages r=0.999")
    W("")
    W("  Predictions:")
    _pf(W, "P5b-B1", iso.get("p5b_b1_pass"),
        "r_manifold > r_linear")
    _pf(W, "P5b-B2", iso.get("p5b_b2_pass"),
        "r_manifold > 0.7")
    _pf(W, "P5b-B3", iso.get("p5b_b3_pass"),
        "r_manifold - r_linear > 0.1")
    W("")

    # ========================================================================
    # Sub-experiment C: Merge-Event Teleportation
    # ========================================================================
    W("=" * 72)
    W("SUB-EXP C — MERGE-EVENT TELEPORTATION")
    W("=" * 72)
    tel = results.get("teleportation", {})
    W(f"  N merge events analyzed         : {tel.get('n_merge_events', '?')}")
    W(f"  Mean arc-length change (Mh)     : {tel.get('mh_mean_arc_change', float('nan')):.6f}")
    W(f"  Std arc-length change (Mh)      : {tel.get('mh_std_arc_change', float('nan')):.6f}")
    W(f"  Mean logit shift (My)           : {tel.get('my_mean_logit_shift', float('nan')):.6f}")
    W(f"  Std logit shift (My)            : {tel.get('my_std_logit_shift', float('nan')):.6f}")
    W(f"  Teleportation distance (KL)     : {tel.get('teleportation_distance', float('nan')):.6f}")
    W("")
    W("  Prediction:")
    _pf(W, "P5b-C1", tel.get("p5b_c1_pass"),
        "Arc-length change < mean inter-centroid distance")
    W("")

    # ========================================================================
    # Sub-experiment D: Subspace Analysis (if present)
    # ========================================================================
    sub = results.get("subspace", {})
    if sub:
        W("=" * 72)
        W("SUB-EXP D — SUBSPACE ISOMETRY ANALYSIS")
        W("=" * 72)
        W(f"  OV subspace dimension          : {sub.get('ov_dim', '?')}")
        W(f"  Isometry score (OV vs Mh)      : {sub.get('isometry_ov', float('nan')):.4f}")
        W(f"  Isometry score (OV vs My)      : {sub.get('isometry_ov_my', float('nan')):.4f}")
        W(f"  Concentration in OV subspace   : {sub.get('concentration', float('nan')):.4f}")
        W("")
        W("  Prediction:")
        _pf(W, "P5b-D1", sub.get("p5b_d1_pass"),
            "Centroids concentrate in OV subspace (>50%)")
        W("")

    # ========================================================================
    # Summary
    # ========================================================================
    W("=" * 72)
    W("SUMMARY")
    W("=" * 72)
    all_pass = []
    all_fail = []

    for exp in ("A", "B", "C", "D"):
        for pred_num in range(1, 4):
            key = f"p5b_{exp.lower()}{pred_num}_pass"
            # Check in appropriate result dict
            if exp == "A":
                val = fs.get(key)
            elif exp == "B":
                val = iso.get(key)
            elif exp == "C":
                val = tel.get(key)
            elif exp == "D":
                val = sub.get(key) if sub else None
            else:
                val = None

            if val is True:
                all_pass.append(f"P5b-{exp}{pred_num}")
            elif val is False:
                all_fail.append(f"P5b-{exp}{pred_num}")

    W(f"  Passed: {len(all_pass)} predictions")
    if all_pass:
        W(f"    {', '.join(all_pass)}")
    W(f"  Failed: {len(all_fail)} predictions")
    if all_fail:
        W(f"    {', '.join(all_fail)}")
    W("")
    W("End of Phase 5b report.")
    W("=" * 72)

    # Write to file
    text = "\n".join(lines)
    path.write_text(text)
    return path


def _pf(W, pred_id: str, passed: bool | None, description: str):
    """
    Format and write a single prediction result.

    Parameters
    ----------
    W : callable
        Line-appending function (e.g., lines.append)
    pred_id : str
        Prediction ID (e.g., "P5b-A1")
    passed : bool or None
        True if passed, False if failed, None if not evaluated
    description : str
        Human-readable description of the prediction
    """
    if passed is True:
        status = "✓ PASS"
    elif passed is False:
        status = "✗ FAIL"
    else:
        status = "  —"

    W(f"    [{status}] {pred_id}: {description}")
