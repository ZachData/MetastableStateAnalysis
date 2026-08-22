"""
train_tuned_lens.py — Fit per-layer affine translators for Group E.

Closes Phase 5 known blocker 4 (status-5.md): Group E's tuned lens was
never trained, so decode_cluster_trajectory silently fell back to the
frozen head everywhere (used_tuned_lens=false) and stored probabilities
rounded to 0.000. This module produces the artifact run_5.py's
`--tuned-lens` flag and tuned_lens_cluster.load_tuned_lens expect:

    tuned_lens_{model_stem}.npz with keys A_L{i}, b_L{i}

Method — affine least squares, not KL descent
---------------------------------------------
For each layer L, fit (A_L, b_L) minimizing

    || H_L @ A_L.T + b_L  -  H_final ||_F^2  + ridge * ||A_L||_F^2

over token positions pooled across the prompt battery, where H_L is the
layer-L hidden state and H_final the final hidden state (the vector the
frozen head was trained to read). apply_tuned_lens then computes
A_L @ v + b_L and hands it to frozen_head_decode — so regressing onto
the final hidden state is exactly the target that pipeline decodes.

This is the affine-translator simplification of Belrose et al.'s tuned
lens (theirs minimizes KL against the model's final distribution by
gradient descent). Least squares is chosen deliberately: deterministic,
seconds on CPU, no training loop to babysit, and closes the
used_tuned_lens=false gap. If a KL-trained lens is ever wanted, this
file is where it goes — same output contract, different objective. Note
the objective difference in any writeup that leans on Group E numbers.

Identity guard: at L = n_layers-1 the regression target equals the
input, so A ≈ I, b ≈ 0 up to the ridge penalty — a free sanity check,
asserted (loosely) after fitting and reported per layer as
`identity_deviation` in the sidecar JSON.

Data: the versioned prompt battery (core.prompts) by default, so the
lens provably saw the same distribution every phase runs on; the hash
goes in the sidecar JSON.

CLI
---
    python -m p5_single_mstate_analysis.train_tuned_lens \\
        --model pythia-1.4b-step143000 \\
        --out results/tuned_lens \\
        [--prompts wiki_paragraph short_heterogeneous ...] \\
        [--ridge 1e-3] [--min-tokens 64]

Then:  run_5.py ... --tuned-lens results/tuned_lens/tuned_lens_{stem}.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure fitting logic (numpy only — tested in tests/test_train_tuned_lens.py)
# ---------------------------------------------------------------------------

def fit_affine_translator(
    H_layer: np.ndarray,     # (n_tokens, d) hidden states at layer L
    H_final: np.ndarray,     # (n_tokens, d) final hidden states (same tokens)
    ridge: float = 1e-3,
) -> tuple:
    """
    Least-squares fit of v' = A v + b mapping layer-L vectors onto final
    vectors. Returns (A, b) with A (d, d), b (d,).

    Solved in one augmented system: append a ones column to H_layer and
    solve ridge-regularized normal equations. The bias column is not
    penalized (standard practice — penalizing b pulls predictions toward
    the origin for no modeling reason).
    """
    H_layer = np.asarray(H_layer, dtype=np.float64)
    H_final = np.asarray(H_final, dtype=np.float64)
    if H_layer.ndim != 2 or H_final.shape != H_layer.shape:
        raise ValueError(
            f"H_layer/H_final must be matching (n_tokens, d) arrays; got "
            f"{H_layer.shape} and {H_final.shape}"
        )
    n, d = H_layer.shape

    X = np.concatenate([H_layer, np.ones((n, 1))], axis=1)   # (n, d+1)
    # Normal equations with ridge on the weight block only
    XtX = X.T @ X                                            # (d+1, d+1)
    reg = np.eye(d + 1) * ridge
    reg[d, d] = 0.0                                          # don't penalize bias
    XtY = X.T @ H_final                                      # (d+1, d)
    W = np.linalg.solve(XtX + reg, XtY)                      # (d+1, d)

    A = W[:d, :].T.astype(np.float32)                        # (d, d): v' = A v + b
    b = W[d, :].astype(np.float32)                           # (d,)
    return A, b


def identity_deviation(A: np.ndarray, b: np.ndarray) -> float:
    """||A - I||_F / sqrt(d) + ||b||_2 / sqrt(d) — scale-free distance from
    the identity translator. Near 0 at the final layer by construction."""
    d = A.shape[0]
    return float(
        np.linalg.norm(A - np.eye(d)) / np.sqrt(d)
        + np.linalg.norm(b) / np.sqrt(d)
    )


def fit_lens_from_activation_stack(
    activations: np.ndarray,   # (n_layers, n_tokens, d) — Phase 1 convention,
                               # embedding at index 0, final layer last
    ridge: float = 1e-3,
) -> dict:
    """
    Fit one (A, b) per layer against the final layer of the same stack.
    Returns {layer_index: {"A": A, "b": b, "identity_deviation": float}}.

    The final layer gets a translator too (≈ identity) so lookups never
    KeyError on it; apply_tuned_lens's own missing-layer fallback also
    covers it either way.
    """
    acts = np.asarray(activations)
    if acts.ndim != 3:
        raise ValueError(f"expected (n_layers, n_tokens, d), got {acts.shape}")
    n_layers = acts.shape[0]
    H_final  = acts[-1]

    lens = {}
    for L in range(n_layers):
        A, b = fit_affine_translator(acts[L], H_final, ridge=ridge)
        lens[L] = {"A": A, "b": b, "identity_deviation": identity_deviation(A, b)}
    return lens


def save_lens(lens: dict, out_path: Path, meta: dict | None = None) -> Path:
    """Write A_L{i}/b_L{i} npz (the exact format load_tuned_lens reads)
    plus a sidecar JSON with fit diagnostics and provenance."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for L, entry in lens.items():
        arrays[f"A_L{L}"] = entry["A"]
        arrays[f"b_L{L}"] = entry["b"]
    np.savez_compressed(out_path, **arrays)

    sidecar = {
        "layers": sorted(int(L) for L in lens),
        "identity_deviation_per_layer": {
            str(L): round(entry["identity_deviation"], 6)
            for L, entry in lens.items()
        },
        "objective": "affine least squares onto final hidden state "
                     "(not KL-trained — see module docstring)",
    }
    if meta:
        sidecar.update(meta)
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# Model-facing collection (torch — deferred, smoke-tier verified only)
# ---------------------------------------------------------------------------

def collect_activation_stack(model_name: str, prompt_keys: list) -> np.ndarray:
    """
    Pool per-layer hidden states across prompts via the same
    core.models.extract_activations every phase uses, concatenated along
    the token axis: returns (n_layers, total_tokens, d).
    """
    from core.models import load_model, extract_activations
    from core.config import PROMPTS

    model, tokenizer = load_model(model_name)

    stacks = []
    for key in prompt_keys:
        text = PROMPTS[key] if key in PROMPTS else key
        hidden_states, _attns, _tokens = extract_activations(
            model, tokenizer, text, model_name
        )
        # list of (n_tokens, d) tensors, embedding at index 0 → one stack
        acts = np.stack(
            [np.asarray(h.cpu().numpy() if hasattr(h, "cpu") else h,
                        dtype=np.float32)
             for h in hidden_states],
            axis=0,
        )
        stacks.append(acts)

    n_layers = stacks[0].shape[0]
    if any(s.shape[0] != n_layers for s in stacks):
        raise RuntimeError("prompt stacks disagree on n_layers — mixed models?")
    return np.concatenate(stacks, axis=1)   # pool tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="results/tuned_lens")
    p.add_argument("--prompts", nargs="+", default=None,
                   help="Prompt-battery keys (default: every key in "
                        "core.prompts.PROMPTS). Raw text also accepted.")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--min-tokens", type=int, default=64,
                   help="Refuse to fit a (d,d) map on fewer pooled tokens "
                        "than this — an underdetermined lens decodes noise.")
    args = p.parse_args(argv)

    from core.config import PROMPTS
    from core.prompts import compute_prompt_battery_hash
    prompt_keys = args.prompts or list(PROMPTS.keys())

    acts = collect_activation_stack(args.model, prompt_keys)
    n_layers, n_tokens, d = acts.shape
    print(f"[tuned-lens] {args.model}: {n_layers} layers, "
          f"{n_tokens} pooled tokens, d={d}")
    if n_tokens < args.min_tokens:
        print(f"[tuned-lens] ERROR: only {n_tokens} tokens pooled "
              f"(< --min-tokens {args.min_tokens}); add prompts.",
              file=sys.stderr)
        return 1
    if n_tokens < d:
        print(f"[tuned-lens] WARNING: n_tokens ({n_tokens}) < d ({d}); "
              f"fit is ridge-dominated. More prompts recommended.")

    lens = fit_lens_from_activation_stack(acts, ridge=args.ridge)

    final_dev = lens[n_layers - 1]["identity_deviation"]
    if final_dev > 0.1:
        print(f"[tuned-lens] WARNING: final-layer translator deviates from "
              f"identity ({final_dev:.4f}) — check activation ordering "
              f"(embedding must be index 0, final layer last).")

    stem = args.model.replace("/", "_")
    out_path = Path(args.out) / f"tuned_lens_{stem}.npz"
    try:
        bhash = compute_prompt_battery_hash()
    except Exception:
        bhash = None
    save_lens(lens, out_path, meta={
        "model": args.model,
        "prompt_keys": prompt_keys,
        "prompt_battery_hash": bhash,
        "ridge": args.ridge,
        "n_pooled_tokens": int(n_tokens),
    })
    print(f"[tuned-lens] saved {out_path} (+ .json sidecar)")
    print(f"[tuned-lens] use with: run_5.py ... --tuned-lens {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
