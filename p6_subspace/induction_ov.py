"""
induction_ov.py — Track A: Induction score and OV write direction alignment.

Two questions answered here:

1. Which heads are induction heads?
   Score each head by mean post-softmax attention weight on canonical
   induction pairs (query > key, offset >= min_offset, preceding-token
   identity match).  Heads above threshold are flagged as induction
   candidates.

2. Do induction heads write into the imaginary channel?
   For each head's OV circuit (W_V @ W_O), compute the fraction of its
   dominant write directions that land in the imaginary (A) subspace.
   Compare induction heads to semantic (high-CC) heads.

Falsifiable predictions tested
-------------------------------
P6-I1 : Induction head OV write directions project more strongly onto the
         imaginary subspace than do semantic heads.
         Test: mean f_rot(induction heads) > mean f_rot(semantic heads)
         with MWU p < 0.05.

Functions
---------
induction_score              : per-head induction strength from attention
classify_induction_heads     : threshold-based binary classification
ov_write_alignment           : fraction of OV singular vectors in A subspace
compare_induction_vs_semantic: P6-I1 test
run_induction_ov             : full pipeline → SubResult

Fixes applied
-------------
Bug 5  (Doc 2): induction_score — canonical query > key orientation enforced;
    bidirectional iteration removed; background-mean subtraction removed;
    returns mean attn on induction pairs only (0.0 when none found).
Bug 10 (Doc 1): ov_write_alignment — docstring updated to clarify that
    head_write_matrix is the composed OV circuit (W_V @ W_O), not W_O alone,
    and documents when column-space equivalence holds.  Function body
    unchanged.
"""

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Induction score  [Bug 5 fix applied]
# ---------------------------------------------------------------------------

def induction_score(
    attn_weights:      np.ndarray,
    token_ids:         np.ndarray,
    token_activations: np.ndarray | None = None,
    sim_threshold:     float = 0.85,
    min_offset:        int = 2,
) -> float:
    """
    Mean post-softmax attention weight on canonical induction pairs.

    Canonical orientation: query > key, query - key >= min_offset, key >= 1,
    and token_ids[query - 1] == token_ids[key - 1] (exact) OR
    cos_sim(token_activations[query - 1], token_activations[key - 1]) >
    sim_threshold (soft).

    `token_activations` should be L2-normalized before being passed in if the
    soft-match branch is used.

    Returns 0.0 if no induction pairs are found in this prompt.

    Bug 5 fix: enforces query > key (canonical direction only); removes the
    bidirectional loop that allowed key > query pairs.  Also removes the
    background-mean subtraction — score is now purely the mean attention on
    induction pairs, making it directly comparable to the softmax baseline
    (1/n) without sign ambiguity.
    """
    n = attn_weights.shape[0]
    induction_mask = np.zeros((n, n), dtype=bool)
    for query in range(2, n):
        for key in range(1, query):
            if query - key < min_offset:
                continue
            exact = (token_ids[query - 1] == token_ids[key - 1])
            soft  = False
            if (not exact) and (token_activations is not None):
                cos = float(
                    token_activations[query - 1] @ token_activations[key - 1]
                )
                soft = cos > sim_threshold
            if exact or soft:
                induction_mask[query, key] = True

    if not induction_mask.any():
        return 0.0

    return float(attn_weights[induction_mask].mean())


def classify_induction_heads(
    scores:    list[float],
    threshold: float = 0.05,
) -> list[int]:
    """Return indices of heads with induction score above threshold."""
    return [i for i, s in enumerate(scores) if s > threshold]


# ---------------------------------------------------------------------------
# OV write direction alignment  [Bug 10 fix applied — docstring only]
# ---------------------------------------------------------------------------

def ov_write_alignment(
    head_write_matrix: np.ndarray,
    P_A:               np.ndarray,
    P_S:               np.ndarray,
    rank:              int = 8,
) -> dict:
    """
    Alignment of a head's residual-stream write subspace with the global
    A and S projectors.

    Parameters
    ----------
    head_write_matrix : (d_model, d_model) — the head's OV circuit
        (W_V @ W_O).  This is what is stored in Phase 2's "ov_head*" keys
        and what `ctx["head_write_matrices"]` carries through.  The column
        space of OV equals the column space of W_O for full-rank W_V (which
        is the standard case), so the top-r left singular vectors of this
        matrix are exactly the head's dominant write directions in the
        residual stream — the quantity the P6-I1 / P6-C1 prediction is
        about.  See Bug 10 in the Track A audit for the naming history.
    P_A, P_S : (d_model, d_model) projectors onto the global rotation /
        real subspaces (or the _excl variants from the Bug 4 fix when a
        clean partition is wanted).
    rank     : number of top left singular vectors to evaluate.

    Returns
    -------
    dict with:
      align_rot       : float — (1/r) Σ ‖P_A e_k‖² over top-r left singular vectors
      align_real      : float — (1/r) Σ ‖P_S e_k‖² over top-r left singular vectors
      rank            : int   — r actually used (min of `rank` and available vectors)
      singular_values : list[float] — top-r singular values (for diagnostic plots)

    Bug 10 fix: docstring updated to reflect that the expected input is the
    composed OV matrix (W_V @ W_O), not W_O alone.  Function body unchanged.
    """
    U, s, _ = np.linalg.svd(head_write_matrix, full_matrices=False)   # U: (d_model, min(d_model, d_model))
    r = min(rank, U.shape[1])
    U_top = U[:, :r]   # (d_model, r)

    # |P_A e_k|^2 = e_k^T P_A^T P_A e_k = e_k^T P_A e_k  (projectors are idempotent)
    rot_scores  = np.array([float(U_top[:, k] @ P_A @ U_top[:, k]) for k in range(r)])
    real_scores = np.array([float(U_top[:, k] @ P_S @ U_top[:, k]) for k in range(r)])

    return {
        "align_rot":       float(rot_scores.mean()),
        "align_real":      float(real_scores.mean()),
        "rank":            r,
        "singular_values": s[:r].tolist(),
    }


# ---------------------------------------------------------------------------
# P6-I1 test
# ---------------------------------------------------------------------------

def compare_induction_vs_semantic(
    head_records:      list[dict],
    induction_indices: list[int],
    semantic_indices:  list[int],
) -> dict:
    """
    P6-I1: Induction heads have higher rotational write alignment than semantic heads.

    Parameters
    ----------
    head_records       : list of dicts, each with 'head_idx' and 'align_rot'
    induction_indices  : head indices classified as induction
    semantic_indices   : head indices classified as semantic (high CC, low PC)

    Returns
    -------
    dict with test results and P6-I1 verdict
    """
    ind_set = set(induction_indices)
    sem_set = set(semantic_indices)

    ind_vals = [r["align_rot"] for r in head_records if r["head_idx"] in ind_set]
    sem_vals = [r["align_rot"] for r in head_records if r["head_idx"] in sem_set]

    if len(ind_vals) < 2 or len(sem_vals) < 2:
        return {
            "mean_align_rot_induction": (float(np.mean(ind_vals)) if ind_vals else None),
            "mean_align_rot_semantic":  (float(np.mean(sem_vals)) if sem_vals else None),
            "mwu_pvalue":              None,
            "n_induction":             len(ind_vals),
            "n_semantic":              len(sem_vals),
            "p6_i1_satisfied":         False,
        }

    mu_ind = float(np.mean(ind_vals))
    mu_sem = float(np.mean(sem_vals))
    stat, pval = mannwhitneyu(ind_vals, sem_vals, alternative="greater")

    return {
        "mean_align_rot_induction": mu_ind,
        "mean_align_rot_semantic":  mu_sem,
        "delta_align_rot":          mu_ind - mu_sem,
        "mwu_statistic":            float(stat),
        "mwu_pvalue":               float(pval),
        "n_induction":              len(ind_vals),
        "n_semantic":               len(sem_vals),
        "p6_i1_satisfied":          (mu_ind > mu_sem and float(pval) < 0.05),
    }


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_induction_ov(ctx: dict) -> SubResult:
    """
    Track A sub-experiment: induction detection + OV write direction alignment.

    Required ctx keys
    -----------------
    attn_matrices       : list of (n, n) softmax attention per head
    head_write_matrices : list of (d_model, d_model) OV circuits (W_V @ W_O)
                          per head  [renamed from wo_matrices per Bug 10]
    token_ids           : (n,) int
    token_activations   : (n, d_model) L2-normed
    projectors          : output of subspace_build.build_global_projectors,
                          used as projectors["per_layer"][layer_idx]
    layer_idx           : int (default 0 for ALBERT)

    Optional ctx keys
    -----------------
    head_classify_result: output of head_classify.py — used to identify
                          semantic heads for P6-I1 comparison
    induction_threshold : float (default 0.05)
    layer_name          : str
    """
    attn_matrices       = ctx["attn_matrices"]
    head_write_matrices = ctx["head_write_matrices"]   # Bug 10: was wo_matrices
    token_ids           = np.asarray(ctx["token_ids"])
    X                   = ctx["token_activations"]
    projectors          = ctx["projectors"]
    layer_idx           = ctx.get("layer_idx", 0)
    layer_name          = ctx.get("layer_name", "shared")
    ind_threshold       = ctx.get("induction_threshold", 0.05)

    proj_entry = projectors["per_layer"][layer_idx]
    P_A = proj_entry["P_A"]
    P_S = proj_entry["P_S"]

    n_heads = len(attn_matrices)

    # 1. Induction scores  [Bug 5: canonical orientation, no background subtraction]
    scores = [
        induction_score(attn_matrices[h], token_ids, token_activations=X)
        for h in range(n_heads)
    ]
    induction_idx = classify_induction_heads(scores, ind_threshold)

    # 2. OV write alignment per head  [Bug 10: head_write_matrices carries W_V @ W_O]
    alignments = [
        ov_write_alignment(head_write_matrices[h], P_A, P_S)
        for h in range(n_heads)
    ]

    # 3. Semantic head indices from head_classify (if available)
    hc_result = ctx.get("head_classify_result")
    if hc_result and "head_records" in hc_result:
        semantic_idx = [
            r["head_idx"] for r in hc_result["head_records"]
            if r.get("quadrant") == "semantic"
        ]
    else:
        # Fallback: heads with low induction score are proxies for semantic
        semantic_idx = [
            h for h, s in enumerate(scores)
            if s < 0.01 and h not in induction_idx
        ]

    # Merge into per-head records
    head_records = []
    for h in range(n_heads):
        head_records.append({
            "head_idx":        h,
            "induction_score": float(scores[h]),
            "is_induction":    h in induction_idx,
            "align_rot":       alignments[h]["align_rot"],
            "align_real":      alignments[h]["align_real"],
        })

    # 4. P6-I1 test
    p6i1 = compare_induction_vs_semantic(head_records, induction_idx, semantic_idx)

    # 5. P6-I1 adjudication  [POPPER_PLAN.md item B6-first]
    #
    # P6-I1 is the first prediction in this project threaded end to end into the
    # falsification ledger, chosen because it needs no new statistics: the
    # Mann-Whitney U above is already a valid one-sided test under the null
    # "induction and semantic heads have the same align_rot distribution", which
    # is exactly h0 as claims/registry.json records it.
    #
    # OPT-IN, and that is deliberate rather than cautious. This function is
    # exercised by the test suite against synthetic fixtures; adjudicating by
    # default would write those fixture p-values into claims/adjudications/ as
    # real evidence for H-OPERATOR. Since adjudicate() refuses to overwrite an
    # existing record -- correctly, since silent replacement is how evidence
    # disappears without trace -- a single accidental fixture run would
    # permanently occupy P6-I1's slot in the ledger. So a run adjudicates only
    # when it says so, and passes the artifact hashes that make the record
    # checkable.
    adjudication = None
    if ctx.get("adjudicate"):
        from core.adjudication import adjudicate_if_registered
        adjudication = adjudicate_if_registered(
            "P6-I1",
            p6i1.get("mwu_pvalue"),
            artifact_hashes=ctx.get("artifact_hashes", ()),
            run_manifest=ctx.get("run_manifest"),
            test_name=("scipy.stats.mannwhitneyu(align_rot[induction], "
                       "align_rot[semantic], alternative='greater')"),
            notes=(f"layer={layer_name} n_induction={p6i1.get('n_induction')} "
                   f"n_semantic={p6i1.get('n_semantic')}"),
            adjudications_dir=ctx.get("adjudications_dir"),
        )

    # 5. Aggregate
    n_induction = len(induction_idx)
    mean_align_rot_all   = float(np.mean([a["align_rot"]  for a in alignments]))
    mean_align_real_all  = float(np.mean([a["align_real"] for a in alignments]))
    mean_align_rot_ind   = p6i1.get("mean_align_rot_induction")
    mean_align_rot_sem   = p6i1.get("mean_align_rot_semantic")

    payload = {
        "layer_name":          layer_name,
        "n_heads":             n_heads,
        "n_induction_heads":   n_induction,
        "induction_indices":   induction_idx,
        "semantic_indices":    semantic_idx,
        "mean_align_rot_all":  mean_align_rot_all,
        "mean_align_real_all": mean_align_real_all,
        "p6_i1":               p6i1,
        "p6_i1_adjudication":  adjudication,
        "head_records":        head_records,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "INDUCTION DETECTION + OV WRITE ALIGNMENT  [Track A]",
        SEP_THICK,
        f"Layer:                {layer_name}",
        f"Heads analysed:       {n_heads}",
        f"Induction threshold:  {ind_threshold}",
        "",
        "Induction scores (mean attn weight on canonical induction pairs):",
    ]
    for h, s in enumerate(scores):
        flag = " ← INDUCTION" if h in induction_idx else ""
        lines.append(f"  head {h:02d}:  score={_fmt(s)}{flag}")

    lines += [
        "",
        f"Induction heads detected: {n_induction} of {n_heads}",
        "",
        "OV write direction alignment with S/A channels (top-8 singular vectors):",
        _bullet("mean align_rot (all heads)", mean_align_rot_all),
        _bullet("mean align_real (all heads)", mean_align_real_all),
        _bullet("mean align_rot (induction heads)", mean_align_rot_ind),
        _bullet("mean align_rot (semantic heads)", mean_align_rot_sem),
        "",
        "P6-I1: induction heads write into imaginary channel more than semantic heads?",
        _bullet("delta align_rot (induction - semantic)", p6i1.get("delta_align_rot")),
        _bullet("MWU p-value", p6i1.get("mwu_pvalue")),
        _verdict_line(
            "P6-I1",
            p6i1["p6_i1_satisfied"],
            f"mu_ind={_fmt(mean_align_rot_ind)} vs mu_sem={_fmt(mean_align_rot_sem)}"
            f" p={_fmt(p6i1.get('mwu_pvalue'))}",
        ),
        "",
    ]

    if adjudication is not None:
        lines += [
            "Adjudication (claims/adjudications/P6-I1.json):",
            _bullet("e-value", adjudication["e_value"]),
            _bullet(f"cumulative E for {adjudication['claim']}",
                    adjudication["claim_E_after"]),
            f"  decision at alpha={adjudication['alpha']}: "
            f"{adjudication['claim_decision_after']}",
            f"  next experiment on this claim must return p < "
            f"{adjudication['next_p_needed']:.4g} to cross the threshold",
            "",
        ]
    elif ctx.get("adjudicate"):
        lines += [
            "Adjudication: NOT recorded. Either the MWU could not run (too few "
            "heads in one arm) or core.adjudication refused -- see stderr. A "
            "refusal is not a failed prediction and must not be read as one.",
            "",
        ]

    lines += [
        "Note on ALBERT: shared weights mean the same heads implement both channels.",
        "If P6-I1 passes, channel separation arises from which residual-stream",
        "subspace the incoming activation occupies, not from separate weight matrices.",
    ]

    vc = {
        "ind_n_induction_heads":        n_induction,
        "ind_mean_align_rot_all":       mean_align_rot_all,
        "ind_mean_align_rot_induction": mean_align_rot_ind,
        "ind_mean_align_rot_semantic":  mean_align_rot_sem,
        "ind_p6_i1_satisfied":          p6i1["p6_i1_satisfied"],
    }

    return SubResult(
        name="induction_ov",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )