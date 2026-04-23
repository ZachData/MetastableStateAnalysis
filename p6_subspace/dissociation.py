"""
dissociation.py — Track C: Double dissociation via forward-pass interventions.

The strongest causal test in Phase 6.  Two surgical interventions on the
residual stream during inference:

  Intervention 1 — zero the imaginary channel:
    Before adding each head's attention output to the residual stream:
      h_attn ← h_attn − Π_A h_attn
    Prediction: induction score drops; cluster structure preserved;
                E_β violations unchanged.

  Intervention 2 — zero the real channel:
    h_attn ← h_attn − Π_S h_attn
    Prediction: induction score preserved; cluster structure disrupted;
                E_β violations eliminated.

  Intervention 3 — zero a random subspace of matching dimension (control):
    Both induction and clusters should degrade, replicating neither
    dissociation pattern.

The double dissociation (both arms confirming their respective predictions)
is falsified if:
  - Arm 1: clusters are disrupted (P6-DD1 fail)
  - Arm 1: induction is preserved (P6-DD1 fail)
  - Arm 2: induction is disrupted (P6-DD2 fail)
  - Arm 2: clusters are preserved (P6-DD2 fail)

NOTE: this is the ONLY Phase 6 module that requires a live model forward pass.

Functions
---------
make_projection_hook   : create a forward hook that zeros one channel
run_intervened_forward : run model with a given hook, return activations + attentions
measure_induction_score: scalar induction score from attention matrices
measure_cluster_overlap: HDBSCAN label agreement with baseline (Adjusted Rand Index)
run_dissociation       : full pipeline → SubResult
"""

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

import hdbscan

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Forward-pass intervention hooks
# ---------------------------------------------------------------------------

def make_projection_hook(
    P:      torch.Tensor,
    device: str,
) -> callable:
    """
    Create a forward hook that zeros the subspace spanned by projector P.

    The hook computes:
        h_out = h - P h     (subtract the projected component)

    Pass P_A to zero the imaginary channel; pass P_S to zero the real channel;
    pass any other projector for control experiments.  The distinction between
    interventions is entirely determined by which projector is supplied — the
    hook implementation is the same in all cases.

    FIX (Bug 5): the original signature included a `mode` parameter
    ("zero_imag" | "zero_real") that had no effect on the hook body — both
    modes executed identical code.  The parameter has been removed.  Callers
    select the channel by passing the appropriate projector matrix.

    Parameters
    ----------
    P      : (d, d) projector tensor (e.g. P_A, P_S, or P_rand)
    device : torch device string

    Returns
    -------
    hook : callable for use with register_forward_hook
    """
    P = P.to(device)

    def hook(module, input, output):
        # output may be a tuple (attn_output, attn_weights, ...)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # h : (batch, seq_len, d_model)
        h_flat    = h.reshape(-1, h.shape[-1]).float()   # (b*s, d)
        projected = h_flat @ P.T                          # Π h
        h_modified = h_flat - projected
        h_out = h_modified.reshape(h.shape).to(h.dtype)

        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out

    return hook


# ---------------------------------------------------------------------------
# Intervened forward pass
# ---------------------------------------------------------------------------

def run_intervened_forward(
    model,
    tokenizer,
    text:          str,
    hook_fn:       callable | None,
    hook_targets:  list,
    device:        str,
    max_length:    int = 512,
) -> dict:
    """
    Run a forward pass with an optional hook registered on each target module.

    Parameters
    ----------
    model        : HuggingFace model with output_attentions support
    tokenizer    : corresponding tokenizer
    text         : input text
    hook_fn      : hook callable, or None for baseline (no intervention)
    hook_targets : list of nn.Module objects to attach the hook to
    device       : "cuda" or "cpu"
    max_length   : tokenization max length

    Returns
    -------
    dict with:
      activations  : list of (n_tokens, d_model) per layer/iteration — float32 np
      attentions   : list of (n_heads, n_tokens, n_tokens) per layer — float32 np
      tokens       : list of str
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    handles = []
    if hook_fn is not None:
        for target in hook_targets:
            handles.append(target.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()

    # Extract hidden states: list of (1, n_tokens, d_model)
    hidden = out.hidden_states   # tuple
    activations = [
        h[0].float().cpu().numpy()   # (n_tokens, d_model)
        for h in hidden[1:]          # skip embedding layer
    ]

    # Extract attentions: tuple of (1, n_heads, n_tokens, n_tokens)
    attentions = [
        a[0].float().cpu().numpy()   # (n_heads, n_tokens, n_tokens)
        for a in out.attentions
    ]

    return {"activations": activations, "attentions": attentions, "tokens": tokens}


# ---------------------------------------------------------------------------
# Induction score from attention matrices
# ---------------------------------------------------------------------------

def measure_induction_score(
    attentions: list[np.ndarray],
    token_ids:  np.ndarray,
) -> float:
    """
    Compute mean induction score across all heads and layers.

    Induction score per head: mean attn[i,j] for induction pairs minus background.
    Aggregated as the mean over heads with any positive score.
    """
    from p6_subspace.induction_ov import induction_score

    scores = []
    for attn_layer in attentions:
        n_heads = attn_layer.shape[0]
        for h in range(n_heads):
            s = induction_score(attn_layer[h], token_ids)
            if s > 0:
                scores.append(s)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Cluster structure measurement
# ---------------------------------------------------------------------------

def measure_cluster_structure(
    activations:     list[np.ndarray],
    baseline_labels: list[np.ndarray],
    min_cluster_size: int = 3,
) -> dict:
    """
    Measure how well cluster structure is preserved after intervention,
    using Adjusted Rand Index between baseline HDBSCAN labels and
    post-intervention HDBSCAN labels.

    Returns
    -------
    dict with:
      mean_ari  : float — mean ARI across layers
      n_layers  : int
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    aris = []

    for acts, base_lab in zip(activations, baseline_labels):
        norms  = np.linalg.norm(acts, axis=1, keepdims=True)
        acts_n = acts / np.maximum(norms, 1e-8)
        try:
            new_labels = clusterer.fit_predict(acts_n)
        except Exception:
            new_labels = np.zeros(acts_n.shape[0], dtype=int)

        valid = base_lab >= 0
        if valid.sum() < 4:
            continue
        ari = adjusted_rand_score(base_lab[valid], new_labels[valid])
        aris.append(float(ari))

    return {
        "mean_ari": float(np.mean(aris)) if aris else 0.0,
        "n_layers": len(aris),
    }


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_dissociation(ctx: dict) -> SubResult:
    """
    Track C sub-experiment: double dissociation via forward-pass interventions.

    Required ctx keys
    -----------------
    model             : loaded HuggingFace model
    tokenizer         : corresponding tokenizer
    text              : input text to run inference on
    token_ids         : (n,) int — token IDs for induction scoring
    projectors        : output of subspace_build
    hook_targets      : list of nn.Module — attention output modules to hook
    device            : str

    Optional ctx keys
    -----------------
    baseline_labels   : list of (n,) HDBSCAN labels from baseline run
                        If absent, baseline run is performed first.
    layer_idx         : int (default 0 for ALBERT projector lookup)
    random_seed       : int (default 42) for random-subspace control
    """
    model        = ctx["model"]
    tokenizer    = ctx["tokenizer"]
    text         = ctx["text"]
    token_ids    = np.asarray(ctx["token_ids"])
    projectors   = ctx["projectors"]
    hook_targets = ctx["hook_targets"]
    device       = ctx.get("device", "cpu")
    layer_idx    = ctx.get("layer_idx", 0)
    rand_seed    = ctx.get("random_seed", 42)

    pe   = projectors["per_layer"][layer_idx]
    d    = projectors["d_model"]

    P_A = torch.tensor(pe["P_A"], dtype=torch.float32)
    P_S = torch.tensor(pe["P_S"], dtype=torch.float32)

    # Random subspace projector (control) — same rank as imaginary subspace
    rng    = np.random.default_rng(seed=rand_seed)
    dim_A  = pe["dim_A"]
    Q, _   = np.linalg.qr(rng.standard_normal((d, max(dim_A, 1))))
    P_rand = torch.tensor(Q[:, :dim_A] @ Q[:, :dim_A].T, dtype=torch.float32)

    # --- Baseline (no intervention) ---
    baseline = run_intervened_forward(
        model, tokenizer, text, None, hook_targets, device
    )
    baseline_ind   = measure_induction_score(baseline["attentions"], token_ids)
    baseline_labels_per_layer = ctx.get("baseline_labels")
    if baseline_labels_per_layer is None:
        # Cluster baseline activations
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
        baseline_labels_per_layer = []
        for acts in baseline["activations"]:
            norms = np.linalg.norm(acts, axis=1, keepdims=True)
            acts_n = acts / np.maximum(norms, 1e-8)
            try:
                lab = clusterer.fit_predict(acts_n)
            except Exception:
                lab = np.zeros(acts_n.shape[0], dtype=int)
            baseline_labels_per_layer.append(lab)

    # --- Intervention 1: zero imaginary channel (pass P_A) ---
    # FIX (Bug 5): mode parameter removed from make_projection_hook.
    # Channel selection is entirely determined by which projector is passed.
    hook_imag = make_projection_hook(P_A, device)
    interv1   = run_intervened_forward(
        model, tokenizer, text, hook_imag, hook_targets, device
    )
    ind_after_zero_imag = measure_induction_score(interv1["attentions"], token_ids)
    clust_after_zero_imag = measure_cluster_structure(
        interv1["activations"], baseline_labels_per_layer
    )

    # --- Intervention 2: zero real channel (pass P_S) ---
    hook_real = make_projection_hook(P_S, device)
    interv2   = run_intervened_forward(
        model, tokenizer, text, hook_real, hook_targets, device
    )
    ind_after_zero_real = measure_induction_score(interv2["attentions"], token_ids)
    clust_after_zero_real = measure_cluster_structure(
        interv2["activations"], baseline_labels_per_layer
    )

    # --- Intervention 3: control (random subspace, pass P_rand) ---
    hook_rand = make_projection_hook(P_rand, device)
    interv3   = run_intervened_forward(
        model, tokenizer, text, hook_rand, hook_targets, device
    )
    ind_after_rand  = measure_induction_score(interv3["attentions"], token_ids)
    clust_after_rand = measure_cluster_structure(
        interv3["activations"], baseline_labels_per_layer
    )

    # --- P6-DD1/DD2 verdicts ---
    # DD1: zeroing imaginary channel → induction drops, clusters preserved
    ind_drop_after_zero_imag   = baseline_ind - ind_after_zero_imag
    clust_ari_after_zero_imag  = clust_after_zero_imag["mean_ari"]

    p6_dd1_ind_drops     = ind_drop_after_zero_imag > 0.02
    p6_dd1_clust_ok      = clust_ari_after_zero_imag > 0.5
    p6_dd1_satisfied     = p6_dd1_ind_drops and p6_dd1_clust_ok

    # DD2: zeroing real channel → clusters disrupted, induction preserved
    clust_ari_after_zero_real  = clust_after_zero_real["mean_ari"]
    ind_preserved_after_zero_real = abs(ind_after_zero_real - baseline_ind) < 0.05

    p6_dd2_clust_disrupted = clust_ari_after_zero_real < 0.3
    p6_dd2_ind_ok          = ind_preserved_after_zero_real
    p6_dd2_satisfied       = p6_dd2_clust_disrupted and p6_dd2_ind_ok

    payload = {
        "baseline_induction_score":     float(baseline_ind),
        "ind_after_zero_imag":          float(ind_after_zero_imag),
        "ind_after_zero_real":          float(ind_after_zero_real),
        "ind_after_zero_rand":          float(ind_after_rand),
        "clust_ari_after_zero_imag":    float(clust_ari_after_zero_imag),
        "clust_ari_after_zero_real":    float(clust_ari_after_zero_real),
        "clust_ari_after_zero_rand":    float(clust_after_rand["mean_ari"]),
        "ind_drop_after_zero_imag":     float(ind_drop_after_zero_imag),
        "p6_dd1_ind_drops":             p6_dd1_ind_drops,
        "p6_dd1_clust_ok":              p6_dd1_clust_ok,
        "p6_dd1_satisfied":             p6_dd1_satisfied,
        "p6_dd2_clust_disrupted":       p6_dd2_clust_disrupted,
        "p6_dd2_ind_ok":                p6_dd2_ind_ok,
        "p6_dd2_satisfied":             p6_dd2_satisfied,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "DOUBLE DISSOCIATION — FORWARD-PASS INTERVENTIONS  [Track C]",
        SEP_THICK,
        "Each intervention zeros one channel from every head's attention output.",
        "",
        "Baseline (no intervention):",
        _bullet("induction score", baseline_ind),
        "",
        "Intervention 1: zero imaginary channel (Π_A h ← 0)",
        _bullet("induction score post-intervention", ind_after_zero_imag),
        _bullet("induction drop (baseline - post)",  ind_drop_after_zero_imag),
        _bullet("cluster ARI vs baseline",            clust_ari_after_zero_imag),
        "",
        "P6-DD1 prediction: induction drops (>0.02), clusters preserved (ARI>0.5).",
        _bullet("induction dropped?", p6_dd1_ind_drops),
        _bullet("clusters preserved?", p6_dd1_clust_ok),
        _verdict_line(
            "P6-DD1",
            p6_dd1_satisfied,
            f"ind_drop={_fmt(ind_drop_after_zero_imag)} ARI={_fmt(clust_ari_after_zero_imag)}",
        ),
        "",
        "Intervention 2: zero real channel (Π_S h ← 0)",
        _bullet("induction score post-intervention",  ind_after_zero_real),
        _bullet("cluster ARI vs baseline",             clust_ari_after_zero_real),
        "",
        "P6-DD2 prediction: clusters disrupted (ARI<0.3), induction preserved (|Δ|<0.05).",
        _bullet("clusters disrupted?",   p6_dd2_clust_disrupted),
        _bullet("induction preserved?",  p6_dd2_ind_ok),
        _verdict_line(
            "P6-DD2",
            p6_dd2_satisfied,
            f"ARI={_fmt(clust_ari_after_zero_real)} ind_post={_fmt(ind_after_zero_real)}",
        ),
        "",
        "Intervention 3: zero random subspace (control, dim matched to A channel):",
        _bullet("induction score post-rand",    ind_after_rand),
        _bullet("cluster ARI post-rand",        clust_after_rand["mean_ari"]),
        f"  Control should NOT replicate either dissociation pattern cleanly.",
        "",
        "Double dissociation summary:",
        f"  Both arms confirmed: {p6_dd1_satisfied and p6_dd2_satisfied}",
        f"  P6-DD1 only:         {p6_dd1_satisfied and not p6_dd2_satisfied}",
        f"  P6-DD2 only:         {not p6_dd1_satisfied and p6_dd2_satisfied}",
        f"  Neither:             {not p6_dd1_satisfied and not p6_dd2_satisfied}",
    ]

    vc = {
        "dd_baseline_induction":       float(baseline_ind),
        "dd_ind_after_zero_imag":      float(ind_after_zero_imag),
        "dd_ind_after_zero_real":      float(ind_after_zero_real),
        "dd_clust_ari_zero_imag":      float(clust_ari_after_zero_imag),
        "dd_clust_ari_zero_real":      float(clust_ari_after_zero_real),
        "dd_p6_dd1_satisfied":         p6_dd1_satisfied,
        "dd_p6_dd2_satisfied":         p6_dd2_satisfied,
        "dd_both_arms_confirmed":      p6_dd1_satisfied and p6_dd2_satisfied,
    }

    return SubResult(
        name="dissociation",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
