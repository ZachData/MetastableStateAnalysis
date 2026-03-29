"""
induction.py — Induction head detection and cross-referencing with Phase 1 geometry.

Measures three things:
  1. Per-head induction score (Olsson et al.) from attention patterns + token sequence
  2. Correlation between induction score and Fiedler value (Phase 1 sinkhorn data)
  3. Per-pair attention attribution: which heads attend between mutual-NN pairs,
     and are those heads the induction heads?

The connection to the energy functional: induction heads create local attractive
forces between specific token pairs (subword completions, repeated contexts).
These pairs should show up as category-2 mutual NNs (NN but not same HDBSCAN
cluster) because the attraction is routing-mediated, not density-mediated.
At plateau layers, the energy contribution of these pairs should be increasing
(attraction), even when global energy drops due to V-repulsive forces through FFN.
"""

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Induction score per head
# ---------------------------------------------------------------------------

def induction_score_per_head(
    attn: np.ndarray,
    tokens: list,
) -> np.ndarray:
    """
    Compute the Olsson et al. induction score for each attention head.

    The induction pattern: at destination position i, attend to source
    position j+1 where token[j] == token[i-1].  This is the signature
    of the "previous-token-head → induction-head" composition.

    For causal models (GPT-2), j+1 <= i (can only attend backward).
    For bidirectional models (BERT, ALBERT), all positions are valid.

    Parameters
    ----------
    attn   : (n_heads, n_tokens, n_tokens) attention weights for one layer
    tokens : list of str, decoded token strings

    Returns
    -------
    (n_heads,) float array — induction score per head.
      Higher = stronger induction behavior.
      Typical induction heads score 0.1–0.5; non-induction heads < 0.02.
    """
    n_heads, n_tok, _ = attn.shape
    scores = np.zeros(n_heads, dtype=np.float64)

    if n_tok < 3:
        return scores

    # Build lookup: token_string -> list of positions where it appears.
    # This avoids O(n^2) scanning for each destination position.
    token_positions = {}
    for pos, tok in enumerate(tokens):
        token_positions.setdefault(tok, []).append(pos)

    # For each destination position i (i >= 2 so that i-1 >= 1 and
    # there exists a prior context to match against):
    total_pairs = 0
    score_accum = np.zeros(n_heads, dtype=np.float64)

    for i in range(2, n_tok):
        prev_tok = tokens[i - 1]
        # Find all positions j where token[j] == token[i-1]
        # and j+1 is a valid source (j+1 != i, j+1 < n_tok)
        matching_positions = token_positions.get(prev_tok, [])
        for j in matching_positions:
            src = j + 1
            if src >= n_tok or src == i:
                continue
            # For causal models: src must be <= i.
            # We check this at the attention level — if attn[h, i, src] is
            # zero (masked), it contributes zero and doesn't inflate the score.
            # For bidirectional models, all positions contribute.
            # Also skip if j == i-1 (trivial self-match: the "previous token"
            # is the same position we're using as context).
            if j == i - 1:
                continue
            score_accum += attn[:, i, src]  # (n_heads,)
            total_pairs += 1

    if total_pairs > 0:
        scores = score_accum / total_pairs

    return scores


def induction_scores_all_layers(
    attentions: np.ndarray,
    tokens: list,
) -> np.ndarray:
    """
    Compute induction score per head per layer.

    Parameters
    ----------
    attentions : (n_layers, n_heads, n_tokens, n_tokens)
    tokens     : list of str

    Returns
    -------
    (n_layers, n_heads) float array
    """
    n_layers = attentions.shape[0]
    n_heads = attentions.shape[1]
    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for layer_idx in range(n_layers):
        scores[layer_idx] = induction_score_per_head(
            attentions[layer_idx], tokens,
        )
    return scores


# ---------------------------------------------------------------------------
# 2. Induction score × Fiedler correlation
# ---------------------------------------------------------------------------

def induction_fiedler_correlation(
    results: dict,
    induction_scores: np.ndarray,
) -> dict:
    """
    Spearman correlation between per-head induction score and per-head
    Fiedler value at each layer.

    Prediction: negative correlation.  Induction heads route narrowly
    (low Fiedler = near-disconnected components).  Phase 2 found that
    V-repulsive heads have high Fiedler (broad mixing).  If induction
    heads are a distinct population, they should cluster in the low-Fiedler,
    high-induction-score quadrant.

    Parameters
    ----------
    results          : Phase 1 results dict (must have sinkhorn data per layer)
    induction_scores : (n_layers, n_heads) from induction_scores_all_layers

    Returns
    -------
    dict with keys:
      per_layer : list of dicts per layer, each with:
                    rho, pvalue, n_heads,
                    induction_scores, fiedler_values (for plotting)
      mean_rho  : float — mean Spearman rho across layers with valid data
      summary   : str — human-readable interpretation
    """
    layers = results["layers"]
    n_layers = min(len(layers), induction_scores.shape[0])
    per_layer = []

    valid_rhos = []

    for li in range(n_layers):
        lr = layers[li]
        sinkhorn = lr.get("sinkhorn", {})
        fiedler_per_head = sinkhorn.get("fiedler_per_head", [])

        if not fiedler_per_head:
            per_layer.append({
                "layer": li,
                "rho": float("nan"),
                "pvalue": float("nan"),
                "n_heads": 0,
            })
            continue

        fiedler = np.array(fiedler_per_head)
        ind_scores = induction_scores[li]

        # Ensure matching head count
        n = min(len(fiedler), len(ind_scores))
        if n < 4:
            per_layer.append({
                "layer": li,
                "rho": float("nan"),
                "pvalue": float("nan"),
                "n_heads": n,
            })
            continue

        fiedler = fiedler[:n]
        ind_scores = ind_scores[:n]

        rho, pval = spearmanr(ind_scores, fiedler)
        per_layer.append({
            "layer": li,
            "rho": float(rho),
            "pvalue": float(pval),
            "n_heads": n,
            "induction_scores": ind_scores.tolist(),
            "fiedler_values": fiedler.tolist(),
        })
        if not np.isnan(rho):
            valid_rhos.append(rho)

    mean_rho = float(np.mean(valid_rhos)) if valid_rhos else float("nan")

    if np.isnan(mean_rho):
        summary = "Insufficient Fiedler data for correlation."
    elif mean_rho < -0.3:
        summary = (
            f"Mean ρ = {mean_rho:.3f}: induction heads cluster in low-Fiedler "
            f"regime (narrow routing), consistent with local-attraction mechanism."
        )
    elif mean_rho > 0.3:
        summary = (
            f"Mean ρ = {mean_rho:.3f}: induction heads have HIGH Fiedler "
            f"(broad attention). Unexpected — may indicate induction heads "
            f"are not the narrow-routing population."
        )
    else:
        summary = (
            f"Mean ρ = {mean_rho:.3f}: weak or no correlation between "
            f"induction score and Fiedler value."
        )

    return {
        "per_layer": per_layer,
        "mean_rho": mean_rho,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 3. Per-pair attention attribution
# ---------------------------------------------------------------------------

def pair_attention_attribution(
    attn: np.ndarray,
    mutual_pairs: list,
    induction_scores: np.ndarray,
) -> list:
    """
    For each mutual-NN pair (i, j), identify which heads attend most
    strongly between the pair members.

    Parameters
    ----------
    attn              : (n_heads, n_tokens, n_tokens) attention at one layer
    mutual_pairs      : list of dicts from pair_hdbscan_agreement, each with
                        'i', 'j', 'tok_i', 'tok_j', 'tag'
    induction_scores  : (n_heads,) induction scores for this layer

    Returns
    -------
    list of dicts, one per pair, each with:
      i, j, tok_i, tok_j, tag  : from input
      total_attn_per_head      : (n_heads,) A[h,i,j] + A[h,j,i]
      top_head                 : head index with highest bidirectional attention
      top_head_induction_score : induction score of the top head
      mean_induction_of_top3   : mean induction score of top 3 attending heads
    """
    n_heads = attn.shape[0]
    attributed = []

    for pair in mutual_pairs:
        i = pair["i"]
        j = pair["j"]

        # Bidirectional attention between the pair
        bidir = attn[:, i, j] + attn[:, j, i]  # (n_heads,)

        top3_heads = np.argsort(bidir)[-3:][::-1]
        top_head = int(top3_heads[0])

        attributed.append({
            "i": i,
            "j": j,
            "tok_i": pair.get("tok_i", "?"),
            "tok_j": pair.get("tok_j", "?"),
            "tag": pair.get("tag", "?"),
            "total_attn_per_head": bidir.tolist(),
            "top_head": top_head,
            "top_head_induction_score": float(induction_scores[top_head]),
            "top3_heads": top3_heads.tolist(),
            "mean_induction_of_top3": float(
                np.mean([induction_scores[h] for h in top3_heads])
            ),
        })

    return attributed


def aggregate_pair_attribution(
    attributed_pairs: list,
) -> dict:
    """
    Aggregate pair attribution results to test the core prediction:
    category-2 (artifact) pairs should be preferentially attended by
    high-induction heads; category-1 (semantic) pairs should not.

    Returns
    -------
    dict with:
      semantic_mean_induction   : mean induction score of top heads for semantic pairs
      artifact_mean_induction   : mean induction score of top heads for artifact pairs
      noise_mean_induction      : mean induction score of top heads for noise pairs
      separation                : artifact_mean - semantic_mean (positive = prediction holds)
      n_semantic, n_artifact, n_noise : counts
    """
    by_tag = {"semantic": [], "artifact": [], "noise": []}

    for p in attributed_pairs:
        tag = p["tag"]
        if tag in by_tag:
            by_tag[tag].append(p["top_head_induction_score"])

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    sem = _mean(by_tag["semantic"])
    art = _mean(by_tag["artifact"])
    noi = _mean(by_tag["noise"])

    return {
        "semantic_mean_induction": sem,
        "artifact_mean_induction": art,
        "noise_mean_induction": noi,
        "separation": art - sem if not (np.isnan(art) or np.isnan(sem)) else float("nan"),
        "n_semantic": len(by_tag["semantic"]),
        "n_artifact": len(by_tag["artifact"]),
        "n_noise": len(by_tag["noise"]),
    }


# ---------------------------------------------------------------------------
# 4. Energy contribution of induction-mediated pairs
# ---------------------------------------------------------------------------

def induction_pair_energy_trajectory(
    activations: np.ndarray,
    mutual_pairs_by_layer: dict,
    beta: float = 1.0,
) -> dict:
    """
    Track the per-pair energy contribution of category-2 (artifact) pairs
    across layers.

    Prediction: at layers where induction heads activate, artifact pairs
    should have INCREASING per-pair energy (attraction), even when global
    energy drops (V-repulsive).  This is the geometric signature of induction
    creating local attraction against a global repulsive field.

    Parameters
    ----------
    activations           : (n_layers, n_tokens, d) L2-normed activations
    mutual_pairs_by_layer : dict mapping layer_idx -> list of pair dicts
                            (from pair_agreement at that layer)
    beta                  : interaction energy beta

    Returns
    -------
    dict mapping pair_key -> list of (layer, energy_contribution) tuples,
    plus aggregate statistics.
    """
    n_layers = activations.shape[0]
    n_tok = activations.shape[1]
    norm = 2.0 * beta * n_tok * n_tok

    # Collect all artifact pairs that appear at any layer
    # Use (min(i,j), max(i,j)) as stable key
    artifact_pairs = set()
    for li, pairs in mutual_pairs_by_layer.items():
        for p in pairs:
            if p.get("tag") == "artifact":
                key = (min(p["i"], p["j"]), max(p["i"], p["j"]))
                artifact_pairs.add(key)

    semantic_pairs = set()
    for li, pairs in mutual_pairs_by_layer.items():
        for p in pairs:
            if p.get("tag") == "semantic":
                key = (min(p["i"], p["j"]), max(p["i"], p["j"]))
                semantic_pairs.add(key)

    if not artifact_pairs and not semantic_pairs:
        return {"artifact_trajectories": {}, "semantic_trajectories": {},
                "summary": "No mutual-NN pairs found."}

    def _track_pairs(pair_set):
        trajectories = {}
        for (i, j) in pair_set:
            traj = []
            for li in range(n_layers):
                ip = float(activations[li, i] @ activations[li, j])
                e_pair = float(np.exp(beta * ip) / norm)
                traj.append((li, e_pair, ip))
            trajectories[(i, j)] = traj
        return trajectories

    art_trajs = _track_pairs(artifact_pairs)
    sem_trajs = _track_pairs(semantic_pairs)

    # Compute mean per-layer energy delta for each category
    def _mean_delta(trajs):
        if not trajs:
            return []
        n = len(list(trajs.values())[0])
        deltas = []
        for li in range(1, n):
            d = []
            for traj in trajs.values():
                d.append(traj[li][1] - traj[li - 1][1])
            deltas.append((li, float(np.mean(d))))
        return deltas

    return {
        "artifact_trajectories": {
            f"{i}_{j}": traj for (i, j), traj in art_trajs.items()
        },
        "semantic_trajectories": {
            f"{i}_{j}": traj for (i, j), traj in sem_trajs.items()
        },
        "artifact_mean_delta": _mean_delta(art_trajs),
        "semantic_mean_delta": _mean_delta(sem_trajs),
        "n_artifact_pairs": len(artifact_pairs),
        "n_semantic_pairs": len(semantic_pairs),
    }


# ---------------------------------------------------------------------------
# 5. Identify induction heads (threshold-based)
# ---------------------------------------------------------------------------

def identify_induction_heads(
    scores: np.ndarray,
    threshold: float = 0.04,
) -> list:
    """
    Identify (layer, head) pairs that exceed the induction score threshold.

    Parameters
    ----------
    scores    : (n_layers, n_heads) induction scores
    threshold : minimum score to qualify.  0.04 is conservative;
                Olsson et al. use ~0.02 for detection.

    Returns
    -------
    list of (layer, head, score) tuples, sorted by score descending
    """
    n_layers, n_heads = scores.shape
    heads = []
    for li in range(n_layers):
        for hi in range(n_heads):
            if scores[li, hi] >= threshold:
                heads.append((li, hi, float(scores[li, hi])))
    heads.sort(key=lambda x: -x[2])
    return heads
