"""
geometric.py — Track 2: direct geometric methods (no learned dictionary).

These bypass the crosscoder and work on raw activation geometry. They
serve as ground truth for Track 1 — if crosscoder features track cluster
structure, they should agree with these results.

Methods:
  1. LDA / contrastive directions per plateau layer
  2. PCA on layer-to-layer deltas
  3. Supervised linear probes for cluster label prediction

All methods take raw activations + HDBSCAN labels, no crosscoder needed.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GeometricResult:
    """Results from one geometric method at one layer."""
    method: str
    model_layer: int
    accuracy: Optional[float] = None
    direction: Optional[np.ndarray] = None  # (d,) or (k, d)
    explained_variance: Optional[float] = None
    extra: Optional[dict] = None


# ---------------------------------------------------------------------------
# 1. LDA / contrastive directions
# ---------------------------------------------------------------------------

def lda_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    reg: float = 1e-4,
) -> dict:
    """
    Compute Fisher's LDA directions that maximally separate HDBSCAN
    clusters at a single layer.

    Parameters
    ----------
    activations : (T, d) residual stream at one layer
    labels : (T,) integer cluster labels (-1 = noise, excluded)
    reg : Tikhonov regularization for within-class scatter

    Returns
    -------
    dict with:
      directions: (n_classes-1, d) LDA projection axes
      eigenvalues: (n_classes-1,) discriminant ratios
      accuracy: leave-one-out nearest-centroid accuracy in LDA space
    """
    valid = labels >= 0
    X = activations[valid].astype(np.float64)
    y = labels[valid]
    n, d = X.shape

    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        return {"error": "fewer than 2 clusters", "n_classes": int(n_classes)}

    # Class means and overall mean
    overall_mean = X.mean(axis=0)
    class_means = {}
    class_counts = {}
    for c in classes:
        mask = y == c
        class_means[c] = X[mask].mean(axis=0)
        class_counts[c] = int(mask.sum())

    # Between-class scatter: S_B
    S_B = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        diff = (class_means[c] - overall_mean).reshape(-1, 1)
        S_B += class_counts[c] * (diff @ diff.T)

    # Within-class scatter: S_W
    S_W = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        Xc = X[y == c] - class_means[c]
        S_W += Xc.T @ Xc

    # Regularize
    S_W += reg * np.eye(d)

    # Solve generalized eigenvalue problem via S_W^{-1} S_B
    # Use Cholesky for numerical stability
    try:
        L_chol = np.linalg.cholesky(S_W)
        L_inv = np.linalg.inv(L_chol)
        M = L_inv @ S_B @ L_inv.T
        eigvals, eigvecs = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        # Fall back to direct solve
        try:
            S_W_inv = np.linalg.inv(S_W)
        except np.linalg.LinAlgError:
            S_W_inv = np.linalg.pinv(S_W)
        M = S_W_inv @ S_B
        eigvals, eigvecs = np.linalg.eigh(M)

    # Take top (n_classes - 1) directions
    k = min(n_classes - 1, d)
    idx = np.argsort(eigvals)[::-1][:k]
    directions = eigvecs[:, idx].T  # (k, d)
    top_eigvals = eigvals[idx]

    # If we used Cholesky, transform back
    try:
        directions = directions @ L_inv
    except NameError:
        pass

    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    directions = directions / norms

    # Nearest-centroid accuracy in LDA space
    X_proj = X @ directions.T  # (n, k)
    centroids_proj = {
        c: X_proj[y == c].mean(axis=0) for c in classes
    }
    correct = 0
    for i in range(n):
        dists = {c: np.sum((X_proj[i] - cent) ** 2)
                 for c, cent in centroids_proj.items()}
        pred = min(dists, key=dists.get)
        if pred == y[i]:
            correct += 1
    accuracy = correct / n

    return {
        "directions": directions,
        "eigenvalues": top_eigvals.tolist(),
        "accuracy": float(accuracy),
        "n_classes": int(n_classes),
        "n_tokens": int(n),
    }


def lda_stability_across_layers(
    activations_per_layer: dict[int, np.ndarray],
    labels_per_layer: dict[int, np.ndarray],
    reg: float = 1e-4,
) -> dict:
    """
    Compute LDA direction at each layer and track cosine similarity
    between consecutive layers. Stable direction = metastable window.
    Rotating direction = merge event.

    Parameters
    ----------
    activations_per_layer : {model_layer: (T, d)}
    labels_per_layer : {model_layer: (T,)} — can differ per layer

    Returns
    -------
    dict with per-layer LDA results and inter-layer cosine similarities
    """
    sorted_layers = sorted(activations_per_layer.keys())
    per_layer = {}
    cosines = []

    prev_dir = None
    prev_layer = None

    for layer in sorted_layers:
        acts = activations_per_layer[layer]
        labs = labels_per_layer.get(layer)
        if labs is None:
            continue

        result = lda_directions(acts, labs, reg)
        per_layer[layer] = {
            "accuracy": result.get("accuracy"),
            "n_classes": result.get("n_classes"),
            "eigenvalues": result.get("eigenvalues"),
        }

        if "error" in result:
            prev_dir = None
            prev_layer = layer
            continue

        curr_dir = result["directions"][0]  # top LDA direction

        if prev_dir is not None and prev_dir.shape == curr_dir.shape:
            cos = float(np.abs(np.dot(prev_dir, curr_dir)))
            cosines.append({
                "layer_from": prev_layer,
                "layer_to": layer,
                "cosine": cos,
            })

        prev_dir = curr_dir
        prev_layer = layer

        # Store direction for Phase 5 export
        per_layer[layer]["direction"] = curr_dir

    return {
        "per_layer": per_layer,
        "cosine_trajectory": cosines,
        "mean_cosine": float(np.mean([c["cosine"] for c in cosines]))
        if cosines else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. PCA on layer-to-layer deltas
# ---------------------------------------------------------------------------

def pca_on_deltas(
    activations_per_layer: dict[int, np.ndarray],
    n_components: int = 10,
    v_projectors: Optional[dict] = None,
) -> dict:
    """
    Compute Δx = x^{L+1} - x^{L} at each layer transition. PCA on
    these deltas reveals which directions carry the most update variance.

    At plateau layers, update variance should be low. At violation/merge
    layers, the top PC should point into V's repulsive subspace.

    Parameters
    ----------
    activations_per_layer : {model_layer: (T, d)}
    n_components : number of PCA components to retain
    v_projectors : optional dict with 'repulsive' and 'attractive'
                   projector matrices (d, d) for V-alignment test

    Returns
    -------
    dict with per-transition PCA results
    """
    sorted_layers = sorted(activations_per_layer.keys())
    results = []

    for i in range(len(sorted_layers) - 1):
        l_from = sorted_layers[i]
        l_to = sorted_layers[i + 1]

        x_from = activations_per_layer[l_from]
        x_to = activations_per_layer[l_to]
        T = min(x_from.shape[0], x_to.shape[0])

        delta = (x_to[:T] - x_from[:T]).astype(np.float64)  # (T, d)

        # Total update variance
        total_var = float(np.var(delta))

        # PCA
        delta_centered = delta - delta.mean(axis=0)
        if T < delta.shape[1]:
            # More dimensions than samples: use covariance trick
            C = delta_centered @ delta_centered.T / max(T - 1, 1)
            eigvals, eigvecs_small = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1][:n_components]
            eigvals = eigvals[idx]
            # Project back to full space
            components = (delta_centered.T @ eigvecs_small[:, idx]).T
            norms = np.linalg.norm(components, axis=1, keepdims=True)
            components = components / np.maximum(norms, 1e-10)
        else:
            C = delta_centered.T @ delta_centered / max(T - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1][:n_components]
            eigvals = eigvals[idx]
            components = eigvecs[:, idx].T  # (k, d)

        explained = eigvals / max(eigvals.sum(), 1e-10)

        entry = {
            "layer_from": l_from,
            "layer_to": l_to,
            "total_variance": total_var,
            "explained_ratio": explained.tolist(),
            "top1_explained": float(explained[0]) if len(explained) > 0 else 0.0,
        }

        # V-alignment test for top PCs
        if v_projectors is not None and len(components) > 0:
            P_rep = v_projectors.get("repulsive")
            P_att = v_projectors.get("attractive")
            if P_rep is not None and P_att is not None:
                v_align = []
                for j in range(min(3, len(components))):
                    pc = components[j]
                    rep_proj = float(pc @ P_rep @ pc)
                    att_proj = float(pc @ P_att @ pc)
                    v_align.append({
                        "pc": j,
                        "repulsive_projection": rep_proj,
                        "attractive_projection": att_proj,
                        "dominance": "repulsive" if rep_proj > att_proj
                        else "attractive",
                    })
                entry["v_alignment"] = v_align

        results.append(entry)

    return {
        "per_transition": results,
        "summary": {
            "n_transitions": len(results),
            "mean_total_variance": float(
                np.mean([r["total_variance"] for r in results])
            ) if results else 0.0,
            "mean_top1_explained": float(
                np.mean([r["top1_explained"] for r in results])
            ) if results else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 3. Supervised linear probes
# ---------------------------------------------------------------------------

def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Train a linear classifier (softmax regression via closed-form
    one-vs-rest) to predict cluster label from activations.

    Uses ridge regression for each class, then predicts via argmax.
    Fast, no iterative optimization needed.

    Parameters
    ----------
    X : (n, d) activations
    y : (n,) integer labels (non-negative)
    reg : L2 regularization

    Returns
    -------
    W : (n_classes, d) weight matrix
    b : (n_classes,) bias
    accuracy : training accuracy (used as proxy; small datasets)
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n, d = X.shape

    # One-hot encode
    Y = np.zeros((n, n_classes), dtype=np.float64)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, yi in enumerate(y):
        Y[i, class_to_idx[yi]] = 1.0

    # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
    XtX = X.T @ X + reg * np.eye(d)
    try:
        W = np.linalg.solve(XtX, X.T @ Y)  # (d, n_classes)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(XtX, X.T @ Y, rcond=None)[0]

    # Bias: not critical for accuracy, set to 0
    b = np.zeros(n_classes)

    # Training accuracy
    preds = X @ W  # (n, n_classes)
    pred_labels = np.array([classes[j] for j in preds.argmax(axis=1)])
    accuracy = float((pred_labels == y).mean())

    # Convert W to (n_classes, d) for consistency
    W = W.T

    return W, b, accuracy


def probe_accuracy_trajectory(
    activations_per_layer: dict[int, np.ndarray],
    labels_per_layer: dict[int, np.ndarray],
    reg: float = 1e-3,
) -> dict:
    """
    Train a linear probe at each layer and report accuracy trajectory.

    Accuracy vs. layer should mirror NN-stability from Phase 1:
    high during plateaus, dropping at merge events.

    Parameters
    ----------
    activations_per_layer : {model_layer: (T, d)}
    labels_per_layer : {model_layer: (T,)}

    Returns
    -------
    dict with per-layer accuracy and probe directions
    """
    sorted_layers = sorted(activations_per_layer.keys())
    per_layer = {}
    probe_directions = {}

    for layer in sorted_layers:
        acts = activations_per_layer[layer]
        labs = labels_per_layer.get(layer)
        if labs is None:
            continue

        valid = labs >= 0
        if valid.sum() < 10:
            per_layer[layer] = {"accuracy": 0.0, "error": "too few tokens"}
            continue

        X = acts[valid].astype(np.float64)
        y = labs[valid]

        n_classes = len(np.unique(y))
        if n_classes < 2:
            per_layer[layer] = {"accuracy": 1.0, "n_classes": 1}
            continue

        W, b, acc = train_linear_probe(X, y, reg)

        per_layer[layer] = {
            "accuracy": float(acc),
            "n_classes": int(n_classes),
            "n_tokens": int(valid.sum()),
        }

        # Store weight matrix for Phase 5 export and V-alignment
        probe_directions[layer] = W  # (n_classes, d)

    accuracies = [v["accuracy"] for v in per_layer.values()
                  if "accuracy" in v]

    return {
        "per_layer": per_layer,
        "probe_directions": probe_directions,
        "summary": {
            "mean_accuracy": float(np.mean(accuracies))
            if accuracies else 0.0,
            "max_accuracy": float(np.max(accuracies))
            if accuracies else 0.0,
            "min_accuracy": float(np.min(accuracies))
            if accuracies else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 4. Probe direction alignment with V eigensubspaces
# ---------------------------------------------------------------------------

def probe_v_alignment(
    probe_directions: dict[int, np.ndarray],
    v_projectors: dict,
) -> dict:
    """
    Test whether linear probe weight vectors (cluster identity
    directions) align with V's eigensubspaces.

    Parameters
    ----------
    probe_directions : {model_layer: (n_classes, d)} from probes
    v_projectors : dict with 'repulsive'/'attractive' projectors,
                   or per-layer keyed by layer index

    Returns
    -------
    dict with per-layer alignment scores
    """
    results = {}
    for layer, W in probe_directions.items():
        # Get projector for this layer
        if "repulsive" in v_projectors:
            P_rep = v_projectors["repulsive"]
            P_att = v_projectors["attractive"]
        elif str(layer) in v_projectors:
            P_rep = v_projectors[str(layer)].get("repulsive")
            P_att = v_projectors[str(layer)].get("attractive")
        else:
            continue

        if P_rep is None or P_att is None:
            continue

        # For each class direction, compute alignment
        alignments = []
        for i in range(W.shape[0]):
            w = W[i]
            w_norm = w / max(np.linalg.norm(w), 1e-10)
            rep = float(w_norm @ P_rep @ w_norm)
            att = float(w_norm @ P_att @ w_norm)
            alignments.append({
                "class": i,
                "repulsive": rep,
                "attractive": att,
                "dominance": "repulsive" if rep > att else "attractive",
            })

        results[layer] = {
            "per_class": alignments,
            "mean_repulsive": float(
                np.mean([a["repulsive"] for a in alignments])
            ),
            "mean_attractive": float(
                np.mean([a["attractive"] for a in alignments])
            ),
        }

    return results


# ---------------------------------------------------------------------------
# 5. Convenience: build activations_per_layer from PromptActivationStore
# ---------------------------------------------------------------------------

def extract_per_layer_activations(
    prompt_store,
    prompt_key: str,
    layer_indices: list,
) -> dict[int, np.ndarray]:
    """
    Extract per-layer activations from the prompt store for a single
    prompt. Returns {model_layer: (T, d)}.
    """
    x = prompt_store.get_stacked_tensor(prompt_key).numpy()  # (T, L, d)
    result = {}
    for i, li in enumerate(layer_indices):
        result[li] = x[:, i, :]
    return result


def build_labels_per_layer(
    hdbscan_labels: dict,
    prompt_key: str,
) -> dict[int, np.ndarray]:
    """
    Extract HDBSCAN labels for a prompt, keyed by model layer.

    Handles the nested structure from Phase 1:
    hdbscan_labels[prompt_key][layer_key] -> list of labels
    """
    prompt_labels = hdbscan_labels.get(prompt_key, {})
    result = {}
    for layer_key, labs in prompt_labels.items():
        try:
            layer = int(layer_key.replace("layer_", ""))
        except (ValueError, AttributeError):
            continue
        result[layer] = np.array(labs)
    return result
