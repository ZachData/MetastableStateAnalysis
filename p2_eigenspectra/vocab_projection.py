"""
p2_eigenspectra/vocab_projection.py — Vocabulary-space labels for OV subspace directions.

Motivated by Gurnee et al. 2026 ("Verbalizable Representations Form a
Global Workspace in Language Models", Transformer Circuits), §A.24.3
("Interpreting Model Components using the J-lens") and Dar et al. 2023
("Analyzing Transformers in Embedding Space"): any direction in
residual-stream space — here, an eigenvector of the OV symmetric part —
can be projected through the model's unembedding to get a ranked token
readout. Applied to Phase 2's decomposition, this attaches semantic
labels to the attractive/repulsive directions that carry the causal
weight in Phase 2b's `rotation_neutral` result (the signed residue
S = (OV + OVᵀ)/2).

Method choice, stated up front
------------------------------
This is the *logit-lens-on-weights* variant, NOT the J-lens. The paper's
J-lens composes W_U with a corpus-averaged Jacobian J_ℓ; here J_ℓ = I.
The paper's own comparison (§2.4, A.5) finds the two agree closely in
late layers and that the logit lens "captures much of the workspace-like
structure ... though with somewhat lower reliability (particularly in
earlier layers)." For weight directions (as opposed to activations) the
identity map is the standard practice (Dar et al.; paper Figs. 91–94 use
W_U · J · W, and the pure-W_U version is their acknowledged cheap form).
No Jacobian training is required, which is the point: one matmul per
direction.

LayerNorm handling
------------------
The true residual→logits map is W_U(γ ⊙ norm(h) + β). Full LN is
data-dependent and has no principled evaluation at a *weight direction*
(a direction is not an activation; feeding it through norm() injects the
statistics of a fake activation — the same class of concern
layernorm_jacobian.py exists to handle carefully for activations). The
linear, data-independent part is the diagonal gain γ, so the default
here folds γ only:  scores(v) = W_U @ (γ ⊙ v). `apply_ln_gain=False`
gives the raw Dar-et-al. projection. Whichever was used is recorded in
the output. (For contrast: tuned_lens_cluster.frozen_head_decode applies
full LN to *activations* on the Pythia path — different object,
different convention, both documented at their sites.)

Sign convention
---------------
eigh returns eigenvectors up to sign, so "promoted" vs. "suppressed"
tokens swap under v → −v. The meaningful object is the *pair* of poles
of the axis: for a repulsive direction, the two token sets the value
pathway pushes apart in logit space. Output therefore reports
`pole_pos` (top-k of +v) and `pole_neg` (top-k of −v) and makes no
claim about which pole is "the" direction.

Scope
-----
Causal LMs only (GPT-2 lm_head, GPT-NeoX/Pythia embed_out), matching
core/lm_loading.py's scope rule — masked-LM heads are refused, not
approximated. torch is deferred inside extract_unembedding (project
convention: core/metrics.py, core/lm_loading.py) so everything else is
importable and testable torch-free.

Functions
---------
extract_unembedding       : (torch) pull W_U + ln γ/β off a ForCausalLM
save_unembedding / load_unembedding : npz round-trip so downstream
                            consumers (lens_band.py) stay torch-free
project_directions        : (numpy) directions (n, d) → scores (n, vocab)
decode_scores             : scores → pole_pos / pole_neg token lists
label_ov_directions       : run over load_weight_decomposition output
vocab_projection_to_json / vocab_projection_summary_lines
main                      : CLI — saved decomposition + registry model →
                            vocab_projection_{stem}.json + .summary.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Unembedding extraction (torch, deferred)
# ---------------------------------------------------------------------------

def extract_unembedding(model) -> dict:
    """
    Extract the unembedding matrix and final-LN parameters from a causal
    LM loaded via core.lm_loading.load_causal_lm.

    Head resolution mirrors tuned_lens_cluster.frozen_head_decode's
    attribute walk, restricted to the causal heads this module serves:
      lm_head   : GPT-2 (tied) — weight (vocab, d)
      embed_out : GPT-NeoX/Pythia (untied) — weight (vocab, d)
    Masked-LM heads (predictions / cls) are refused for the same reason
    core/lm_loading.py refuses the models that carry them.

    Final-LN parameters:
      GPT-NeoX : gpt_neox.final_layer_norm (γ, β)
      GPT-2    : transformer.ln_f (γ, β)

    Returns
    -------
    dict: W_U (vocab, d) float32, ln_gamma (d,) or None, ln_beta (d,) or
    None, head_attr, vocab_size, d_model.
    """
    for refused in ("predictions", "cls"):
        if hasattr(model, refused):
            raise ValueError(
                f"extract_unembedding: model exposes masked-LM head "
                f"'{refused}' — out of scope (see core/lm_loading.py's "
                f"causal-only rule)."
            )

    head = None
    head_attr = None
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            head_attr = attr
            break
    if head is None:
        raise RuntimeError(
            "extract_unembedding: no causal LM head found (checked "
            "lm_head, embed_out). Bare registry models have no head — "
            "load via core.lm_loading.load_causal_lm."
        )

    W_U = head.weight.detach().to("cpu").float().numpy()  # (vocab, d)

    ln_gamma = None
    ln_beta = None
    ln = None
    inner_neox = getattr(model, "gpt_neox", None)
    if inner_neox is not None and hasattr(inner_neox, "final_layer_norm"):
        ln = inner_neox.final_layer_norm
    else:
        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "ln_f"):
            ln = transformer.ln_f
    if ln is not None:
        ln_gamma = ln.weight.detach().to("cpu").float().numpy()
        if getattr(ln, "bias", None) is not None:
            ln_beta = ln.bias.detach().to("cpu").float().numpy()

    return {
        "W_U":        W_U,
        "ln_gamma":   ln_gamma,
        "ln_beta":    ln_beta,
        "head_attr":  head_attr,
        "vocab_size": int(W_U.shape[0]),
        "d_model":    int(W_U.shape[1]),
    }


def save_unembedding(unemb: dict, path: Path) -> None:
    """npz round-trip so torch-free consumers (lens_band.py) can reuse it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"W_U": unemb["W_U"]}
    if unemb.get("ln_gamma") is not None:
        arrays["ln_gamma"] = unemb["ln_gamma"]
    if unemb.get("ln_beta") is not None:
        arrays["ln_beta"] = unemb["ln_beta"]
    arrays["head_attr"] = np.array(unemb.get("head_attr", ""))
    np.savez_compressed(path, **arrays)


def load_unembedding(path: Path) -> dict:
    data = np.load(Path(path), allow_pickle=False)
    return {
        "W_U":        data["W_U"],
        "ln_gamma":   data["ln_gamma"] if "ln_gamma" in data else None,
        "ln_beta":    data["ln_beta"] if "ln_beta" in data else None,
        "head_attr":  str(data["head_attr"]) if "head_attr" in data else None,
        "vocab_size": int(data["W_U"].shape[0]),
        "d_model":    int(data["W_U"].shape[1]),
    }


# ---------------------------------------------------------------------------
# Projection (pure numpy)
# ---------------------------------------------------------------------------

def project_directions(
    directions: np.ndarray,     # (n, d) or (d,)
    unemb: dict,
    apply_ln_gain: bool = True,
) -> np.ndarray:
    """
    Vocabulary scores for residual-stream directions.

      scores = (γ ⊙ v) @ W_U.T      (apply_ln_gain=True, γ available)
      scores =        v @ W_U.T     (otherwise)

    Directions are unit-normalised first so scores are comparable across
    directions and layers (eigh output is already unit-norm; renormalising
    is a no-op there and a guard everywhere else).

    Returns (n, vocab) float32; (vocab,) for a single (d,) input.
    """
    v = np.asarray(directions, dtype=np.float32)
    single = v.ndim == 1
    if single:
        v = v[None, :]
    W_U = unemb["W_U"]
    if v.shape[1] != W_U.shape[1]:
        raise ValueError(
            f"project_directions: direction dim {v.shape[1]} != "
            f"unembedding d_model {W_U.shape[1]}"
        )
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(norms, 1e-12)
    if apply_ln_gain and unemb.get("ln_gamma") is not None:
        v = v * unemb["ln_gamma"][None, :].astype(np.float32)
    scores = v @ W_U.T
    return scores[0] if single else scores


def decode_scores(
    scores: np.ndarray,          # (vocab,)
    tokenizer=None,
    top_k: int = 12,
) -> dict:
    """
    Top-k promoted (+v pole) and suppressed (−v pole) tokens for one
    direction's score vector. Without a tokenizer, token strings are
    None and only ids/scores are reported (keeps this path torch- and
    transformers-free for tests).
    """
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    pos_idx = order[:top_k]
    neg_idx = order[::-1][:top_k]

    def _entry(i):
        tok = None
        if tokenizer is not None:
            try:
                tok = tokenizer.convert_ids_to_tokens(int(i))
            except Exception:
                tok = None
        return {"token": tok, "id": int(i), "score": round(float(scores[i]), 6)}

    return {
        "pole_pos": [_entry(i) for i in pos_idx],
        "pole_neg": [_entry(i) for i in neg_idx],
    }


# ---------------------------------------------------------------------------
# OV labelling over a saved decomposition
# ---------------------------------------------------------------------------

def _select_sym_directions(decomp: dict, n_directions: int) -> list:
    """
    Most-repulsive and most-attractive eigenvectors of S from one
    eigendecompose() dict (or its load_weight_decomposition subset).
    eigh sorts ascending, so repulsive = leading columns, attractive =
    trailing columns.
    """
    vals = np.asarray(decomp["sym_eigenvalues"], dtype=np.float64)
    vecs = np.asarray(decomp["sym_eigenvectors"], dtype=np.float32)  # columns
    d = vals.shape[0]
    n = min(n_directions, d)
    picked = []
    for j in range(n):                      # most negative first
        if vals[j] < 0:
            picked.append(("repulsive", j, float(vals[j]), vecs[:, j]))
    for j in range(d - 1, d - 1 - n, -1):   # most positive first
        if vals[j] > 0:
            picked.append(("attractive", j, float(vals[j]), vecs[:, j]))
    return picked


def label_ov_directions(
    loaded: dict,
    unemb: dict,
    tokenizer=None,
    n_directions: int = 8,
    top_k: int = 12,
    apply_ln_gain: bool = True,
) -> dict:
    """
    Vocabulary labels for the top attractive/repulsive S-eigenvectors of
    each layer's OV, from weights.load_weight_decomposition output.

    Handles both the per-layer (list decomp) and shared (single dict)
    layouts. Only the symmetric-part basis is labelled here — it is the
    component Phase 2b assigns 100% of causal weight, and its
    eigenvectors are orthonormal with a clean sign/eigenvalue reading.
    (Schur vectors are available in ov_decomp_{stem}.npz for a follow-up
    pass; they span invariant subspaces of the full non-normal OV and
    individual columns are not eigenvector-interpretable, so labelling
    them one-by-one would overclaim.)

    Returns
    -------
    dict: model metadata + layers: {layer_name: [ {kind, eig_index,
    eigenvalue, pole_pos, pole_neg}, ... ]}
    """
    summary = loaded.get("summary", {})
    is_per_layer = bool(summary.get("is_per_layer", isinstance(loaded["decomp"], list)))
    layer_names = list(summary.get("layers", {}).keys())

    if is_per_layer:
        decomps = loaded["decomp"]
        if not layer_names:
            layer_names = [f"layer{i}" for i in range(len(decomps))]
    else:
        decomps = [loaded["decomp"]]
        layer_names = layer_names or ["shared"]

    d_model_dec = int(np.asarray(decomps[0]["sym_eigenvectors"]).shape[0])
    if d_model_dec != unemb["d_model"]:
        raise ValueError(
            f"label_ov_directions: decomposition d_model {d_model_dec} != "
            f"unembedding d_model {unemb['d_model']} — decomposition and "
            f"LM head are from different models."
        )

    layers_out = {}
    for name, dec in zip(layer_names, decomps):
        picked = _select_sym_directions(dec, n_directions)
        if not picked:
            layers_out[name] = []
            continue
        dirs = np.stack([p[3] for p in picked], axis=0)
        scores = project_directions(dirs, unemb, apply_ln_gain=apply_ln_gain)
        entries = []
        for (kind, j, val, _), sc in zip(picked, scores):
            entry = {"kind": kind, "eig_index": int(j), "eigenvalue": val}
            entry.update(decode_scores(sc, tokenizer=tokenizer, top_k=top_k))
            entries.append(entry)
        layers_out[name] = entries

    return {
        "basis":          "sym",
        "apply_ln_gain":  bool(apply_ln_gain and unemb.get("ln_gamma") is not None),
        "head_attr":      unemb.get("head_attr"),
        "n_directions":   n_directions,
        "top_k":          top_k,
        "d_model":        unemb["d_model"],
        "vocab_size":     unemb["vocab_size"],
        "layers":         layers_out,
    }


# ---------------------------------------------------------------------------
# Reporting (project conventions: *_to_json + *_summary_lines)
# ---------------------------------------------------------------------------

def vocab_projection_to_json(result: dict) -> dict:
    """Already JSON-ready; exists for symmetry with sibling modules."""
    return result


def vocab_projection_summary_lines(result: dict, max_dirs_per_layer: int = 4) -> list:
    """
    LLM-ready plain-text lines (subresult.py contract: self-contained
    prose, no ANSI, < ~100 chars/line).
    """
    lines = [
        "--- Vocab projection: OV symmetric-part eigenvector labels ---",
        f"  Basis: {result['basis']}  |  LN gain folded: {result['apply_ln_gain']}"
        f"  |  head: {result.get('head_attr')}",
        "  Sign caveat: pole_pos/pole_neg are the two poles of each axis;",
        "  which pole is which is arbitrary (eigh sign convention).",
    ]

    def _fmt_pole(entries):
        toks = [(e["token"] if e["token"] is not None else f"#{e['id']}")
                for e in entries[:6]]
        return " ".join(repr(t) for t in toks)

    for name, entries in result["layers"].items():
        lines.append(f"  [{name}]")
        shown = 0
        for e in entries:
            if shown >= max_dirs_per_layer:
                lines.append(f"    ... {len(entries) - shown} more directions in json")
                break
            lines.append(
                f"    {e['kind']:<10} lambda={e['eigenvalue']:+.4f}"
            )
            lines.append(f"      pole_pos: {_fmt_pole(e['pole_pos'])}")
            lines.append(f"      pole_neg: {_fmt_pole(e['pole_neg'])}")
            shown += 1
        if not entries:
            lines.append("    (no signed directions found)")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    python -m p2_eigenspectra.vocab_projection \\
        --model pythia-1.4b-step143000 \\
        --weights-dir results/p2_eigenspectra_<ts> \\
        [--out DIR] [--n-directions 8] [--top-k 12] [--no-ln-gain] \\
        [--save-unembedding]

    Loads the saved Phase 2 decomposition, loads the ForCausalLM at the
    same pinned revision (core.lm_loading), extracts W_U once, frees the
    model, and runs the pure-numpy labelling. --save-unembedding writes
    unembedding_{stem}.npz next to the outputs so lens_band.py can run
    torch-free afterwards.
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--weights-dir", required=True,
                   help="Directory holding ov_decomp_{stem}.npz etc. "
                        "(save_weight_decomposition output).")
    p.add_argument("--out", default=None,
                   help="Output dir; default: --weights-dir.")
    p.add_argument("--n-directions", type=int, default=8)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--no-ln-gain", action="store_true",
                   help="Raw W_U projection (Dar et al. form), no gamma fold.")
    p.add_argument("--save-unembedding", action="store_true")
    args = p.parse_args(argv)

    from p2_eigenspectra.weights import load_weight_decomposition
    from core.lm_loading import load_causal_lm

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out) if args.out else weights_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.model.replace("/", "_")

    loaded = load_weight_decomposition(weights_dir, args.model)

    print(f"[vocab-projection] loading causal LM for {args.model} ...")
    model, tokenizer = load_causal_lm(args.model)
    unemb = extract_unembedding(model)
    del model  # W_U extracted; the rest is numpy
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if args.save_unembedding:
        upath = out_dir / f"unembedding_{stem}.npz"
        save_unembedding(unemb, upath)
        print(f"[vocab-projection] saved {upath}")

    result = label_ov_directions(
        loaded, unemb, tokenizer=tokenizer,
        n_directions=args.n_directions, top_k=args.top_k,
        apply_ln_gain=not args.no_ln_gain,
    )
    result["model"] = args.model

    jpath = out_dir / f"vocab_projection_{stem}.json"
    with open(jpath, "w") as f:
        json.dump(vocab_projection_to_json(result), f, indent=2)
    spath = out_dir / f"vocab_projection_{stem}.summary.txt"
    with open(spath, "w") as f:
        f.write("\n".join(vocab_projection_summary_lines(result)) + "\n")
    print(f"[vocab-projection] wrote {jpath} and {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
