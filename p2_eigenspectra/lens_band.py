"""
p2_eigenspectra/lens_band.py — Layer-band scalars from logit-lens readouts of Phase 1
activations.

Motivated by Gurnee et al. 2026 ("Verbalizable Representations Form a
Global Workspace in Language Models", Transformer Circuits), §4.1
(Fig. 28): a transformer's depth splits into three regimes — an early
"sensory" band whose vocabulary readouts are noise, a middle band
carrying persistent abstract content, and a late "motor" band aligned
with the imminent output. The band boundaries are located by cheap
per-layer statistics of vocabulary readouts:

  (a) top-k agreement with the model's own final prediction  (Fig. 28a)
  (b) excess kurtosis of the readout logit distribution      (Fig. 28b)
  (c) top-1 readout autocorrelation across token positions   (Fig. 28c)

Each reduces to one number per checkpoint (onset layers, band width,
peak kurtosis, mid-band autocorrelation), which is exactly the Class-2
shape checkpoint_scalars.py plots vs. log(step). Whether these band
transitions co-locate with the energy-monotonicity break / Fiedler drop
/ effective-rank collapse is an open empirical question — the paper
itself flags workspace emergence over pretraining as untested (§9.1),
and the anchor-checkpoint runs are positioned to answer it as a
ride-along.

Deviation from the paper, stated up front
-----------------------------------------
The paper computes these statistics on J-lens readouts (W_U · J_ℓ · h);
this module uses the *logit lens* (J_ℓ = I), because no averaged
Jacobian has been trained for these checkpoints and training one is
deliberately out of scope (see CHANGES_jlens_adjacent.md). Consequences,
per the paper's own comparison (§2.4, A.5): the two lenses agree closely
in late layers and diverge earlier, with the logit lens noisier in early
layers. The kurtosis/agreement signatures are qualitatively preserved,
but the detected band ONSET is an upper bound — the true onset (as a
J-lens would see it) may be earlier. Treat onset *trends across
checkpoints* as the signal, not absolute layer indices.

Layer conventions (matches project extraction contracts)
--------------------------------------------------------
- activations: (n_layers, n_tokens, d), embedding at index 0
  (core/io.py contract).
- Final entry is post-ln_f on the standard path (status-5.md item 1 /
  core/models.py convention), so with `final_is_post_ln=True` (default)
  LN is applied to layers 0..L-2 only and the final layer is decoded
  raw; the final layer's readout then IS the model's next-token
  prediction, which is what the agreement metric references.
- Index 0 is the embedding; band-onset search starts at index 1.

Scope: causal models only (needs a real unembedding — see
vocab_projection.extract_unembedding's refusal rule). Note cka_prev
already exists in checkpoints.CHECKPOINT_METRICS; layer-similarity block
structure is deliberately not duplicated here.

torch is never imported: the unembedding arrives as the npz written by
vocab_projection --save-unembedding (or an in-memory dict of the same
shape). Everything here is numpy — same rule as the visualization
package.

Functions
---------
numpy_layernorm            : LN with saved gamma/beta
compute_lens_band          : activations + unembedding → per-layer series
detect_band                : series → onset/motor/width scalars
lens_band_to_json / lens_band_summary_lines
Scalar extractors          : run_dir -> float, reading
                             lens_band_scalars.json (merge into
                             checkpoint_scalars.SCALAR_EXTRACTORS)
main                       : CLI — phase-1 run dir + unembedding npz →
                             lens_band_scalars.json in the run dir
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Output filename inside a Phase 1 run directory. Scalar extractors and
# the CLI both use this constant (artifact-contract discipline,
# core/artifacts.py rationale — one name, two sides).
LENS_BAND_FILENAME = "lens_band_scalars.json"

DEFAULT_TOPK = (1, 8)
DEFAULT_DELTAS = (1, 2, 4, 8)
DEFAULT_KURT_THRESHOLD = 1.0
DEFAULT_AGREE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Primitives (pure numpy)
# ---------------------------------------------------------------------------

def numpy_layernorm(
    x: np.ndarray,                  # (..., d)
    gamma: Optional[np.ndarray],
    beta: Optional[np.ndarray],
    eps: float = 1e-5,
) -> np.ndarray:
    """LN over the last axis with saved affine params (None → skip affine)."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        xh = xh * gamma.astype(np.float32)
    if beta is not None:
        xh = xh + beta.astype(np.float32)
    return xh


def excess_kurtosis_rows(X: np.ndarray) -> np.ndarray:
    """
    Excess kurtosis of each row of X (n, m) over its m entries:
    E[(z-mu)^4]/sigma^4 - 3. The paper's Fig. 28b statistic, computed per
    (position, layer) over the vocabulary axis. ~0 for Gaussian rows,
    large for readouts sharply peaked on a few tokens.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    z = X - mu
    var = (z ** 2).mean(axis=1)
    m4 = (z ** 4).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(var > 1e-24, m4 / np.maximum(var ** 2, 1e-48) - 3.0, 0.0)
    return k


def top1_autocorr(top1_ids: np.ndarray, delta: int) -> Tuple[float, float]:
    """
    (match_rate, shuffled_null) for one layer's top-1 id sequence at
    offset delta. Simplification of the paper's Fig. 28c (delta-log-prob
    vs. positional-shuffle null): match rate P(top1_t == top1_{t+delta}),
    null = sum_id p_id^2 from the layer's empirical top-1 frequencies —
    the expected match rate if positions were independent draws. Report
    the excess (match - null); ~0 for position-local content, high when
    the same concept persists across the token stream.
    """
    ids = np.asarray(top1_ids)
    n = ids.shape[0]
    if n <= delta:
        return float("nan"), float("nan")
    match = float((ids[:-delta] == ids[delta:]).mean())
    _, counts = np.unique(ids, return_counts=True)
    p = counts / n
    null = float((p ** 2).sum())
    return match, null


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_lens_band(
    activations: np.ndarray,          # (n_layers, n_tokens, d)
    unemb: dict,                      # vocab_projection unembedding dict
    final_is_post_ln: bool = True,
    apply_ln: bool = True,
    topk: Tuple[int, ...] = DEFAULT_TOPK,
    deltas: Tuple[int, ...] = DEFAULT_DELTAS,
) -> dict:
    """
    Per-layer logit-lens band statistics.

    For each layer: decode activations through (LN +) W_U, then reduce to
      kurtosis_median / kurtosis_p90 : excess kurtosis over vocab,
                                       median/p90 over tokens
      agree_top{k}                   : fraction of positions whose final-
                                       layer top-1 id is in this layer's
                                       top-k readout
      autocorr_d{delta}              : top-1 match rate minus shuffled
                                       null at each offset

    Memory: one (n_tokens, vocab) float32 logits block at a time.
    """
    acts = np.asarray(activations)
    if acts.ndim != 3:
        raise ValueError(f"compute_lens_band: activations must be "
                         f"(n_layers, n_tokens, d), got {acts.shape}")
    n_layers, n_tokens, d = acts.shape
    W_U = np.asarray(unemb["W_U"], dtype=np.float32)
    if W_U.shape[1] != d:
        raise ValueError(f"compute_lens_band: activation d={d} != "
                         f"unembedding d_model={W_U.shape[1]}")
    gamma = unemb.get("ln_gamma")
    beta = unemb.get("ln_beta")

    def _logits(layer_idx: int) -> np.ndarray:
        h = acts[layer_idx].astype(np.float32)
        is_final = layer_idx == n_layers - 1
        if apply_ln and not (final_is_post_ln and is_final):
            h = numpy_layernorm(h, gamma, beta)
        return h @ W_U.T                              # (n_tokens, vocab)

    # Final layer first: its top-1 is the model's own prediction and the
    # reference for every agreement number below.
    final_logits = _logits(n_layers - 1)
    final_top1 = np.argmax(final_logits, axis=1)      # (n_tokens,)

    per_layer: List[dict] = []
    max_k = max(topk)
    for L in range(n_layers):
        logits = final_logits if L == n_layers - 1 else _logits(L)

        kurt = excess_kurtosis_rows(logits)
        top1 = np.argmax(logits, axis=1)

        # top-k membership of the final prediction, via argpartition
        part = np.argpartition(-logits, max_k - 1, axis=1)[:, :max_k]
        # order the partition so top-k prefixes are valid for every k
        row = np.arange(n_tokens)[:, None]
        part = part[row, np.argsort(-logits[row, part], axis=1)]
        agree = {}
        for k in topk:
            agree[k] = float((part[:, :k] == final_top1[:, None]).any(axis=1).mean())

        auto = {}
        for dlt in deltas:
            match, null = top1_autocorr(top1, dlt)
            auto[dlt] = (match - null) if math.isfinite(match) else float("nan")

        per_layer.append({
            "layer":           L,
            "kurtosis_median": float(np.median(kurt)),
            "kurtosis_p90":    float(np.percentile(kurt, 90)),
            **{f"agree_top{k}": agree[k] for k in topk},
            **{f"autocorr_d{dlt}": auto[dlt] for dlt in deltas},
        })

    return {
        "n_layers":         n_layers,
        "n_tokens":         n_tokens,
        "d_model":          d,
        "vocab_size":       int(W_U.shape[0]),
        "final_is_post_ln": bool(final_is_post_ln),
        "apply_ln":         bool(apply_ln),
        "lens":             "logit",      # honesty marker: not a J-lens
        "topk":             list(topk),
        "deltas":           list(deltas),
        "per_layer":        per_layer,
    }


# ---------------------------------------------------------------------------
# Band detection
# ---------------------------------------------------------------------------

def detect_band(
    result: dict,
    kurt_threshold: float = DEFAULT_KURT_THRESHOLD,
    agree_threshold: float = DEFAULT_AGREE_THRESHOLD,
    persistence: int = 2,
) -> dict:
    """
    Reduce the per-layer series to checkpoint scalars.

      band_onset_layer  : first layer >= 1 where kurtosis_median stays
                          above kurt_threshold for `persistence`
                          consecutive layers (Fig. 28b's rise; index 0 is
                          the embedding and excluded from the search)
      motor_onset_layer : first layer where agree_top1 stays above
                          agree_threshold for `persistence` layers
                          (Fig. 28a's late jump)
      band_width        : motor_onset - band_onset (layers)
      *_frac            : the same, normalised by (n_layers - 1) to [0,1]
                          so 410M and 1.4B share an axis (paper's 0-100
                          reindexing)
      peak_kurtosis_median, midband_autocorr_d4 (mean over the detected
      band, or over the middle third when no band was detected)

    NaN when a crossing never happens — the intended reading at early
    checkpoints, where no band exists yet; the scalar *becoming finite*
    over training is itself the emergence signal.
    """
    pl = result["per_layer"]
    n_layers = result["n_layers"]
    kurt = np.array([r["kurtosis_median"] for r in pl])
    agree1 = np.array([r.get("agree_top1", float("nan")) for r in pl])

    def _first_persistent(series, threshold, start):
        n = len(series)
        for i in range(start, n):
            end = min(i + persistence, n)
            window = series[i:end]
            if np.all(np.isfinite(window)) and np.all(window > threshold):
                return i
        return None

    onset = _first_persistent(kurt, kurt_threshold, start=1)
    motor = _first_persistent(agree1, agree_threshold, start=1)

    denom = max(n_layers - 1, 1)
    width = (motor - onset) if (onset is not None and motor is not None
                                and motor >= onset) else None

    if onset is not None and motor is not None and motor > onset:
        band_slice = slice(onset, motor)
    else:
        band_slice = slice(n_layers // 3, max(2 * n_layers // 3, n_layers // 3 + 1))
    d4 = np.array([r.get("autocorr_d4", float("nan")) for r in pl])[band_slice]
    d4 = d4[np.isfinite(d4)]

    def _f(x):
        return float(x) if x is not None else float("nan")

    return {
        "band_onset_layer":      _f(onset),
        "motor_onset_layer":     _f(motor),
        "band_width":            _f(width),
        "band_onset_frac":       _f(onset / denom if onset is not None else None),
        "motor_onset_frac":      _f(motor / denom if motor is not None else None),
        "band_width_frac":       _f(width / denom if width is not None else None),
        "peak_kurtosis_median":  float(np.nanmax(kurt)) if np.isfinite(kurt).any() else float("nan"),
        "midband_autocorr_d4":   float(d4.mean()) if d4.size else float("nan"),
        "kurt_threshold":        float(kurt_threshold),
        "agree_threshold":       float(agree_threshold),
        "persistence":           int(persistence),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def lens_band_to_json(result: dict, band: dict) -> dict:
    return {"series": result, "band": band}


def lens_band_summary_lines(result: dict, band: dict) -> list:
    n_layers = result["n_layers"]
    lines = [
        "--- Lens band scalars (logit-lens variant of workspace-band metrics) ---",
        f"  Layers: {n_layers} (embedding at 0, final "
        f"{'post-ln_f, decoded raw' if result['final_is_post_ln'] else 'pre-ln_f'})",
        f"  Band onset (kurtosis crossing):  layer "
        f"{band['band_onset_layer']:.0f}" if math.isfinite(band["band_onset_layer"])
        else "  Band onset (kurtosis crossing):  none (no band detected)",
        f"  Motor onset (agree_top1 > {band['agree_threshold']}): layer "
        f"{band['motor_onset_layer']:.0f}" if math.isfinite(band["motor_onset_layer"])
        else f"  Motor onset (agree_top1 > {band['agree_threshold']}): none",
        f"  Band width: {band['band_width']:.0f} layers "
        f"({band['band_width_frac']:.2f} of depth)"
        if math.isfinite(band["band_width"]) else "  Band width: n/a",
        f"  Peak median kurtosis: {band['peak_kurtosis_median']:.2f}"
        f"  |  mid-band autocorr(d=4): {band['midband_autocorr_d4']:.3f}",
        "  Caveat: logit lens degrades in early layers (paper SS2.4/A.5), so",
        "  onset is an upper bound; compare trends across checkpoints, not",
        "  absolute layer indices.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Checkpoint scalar extractors (run_dir -> float; numpy/json only)
# ---------------------------------------------------------------------------

def _load_band(run_dir: Path) -> Optional[dict]:
    p = Path(run_dir) / LENS_BAND_FILENAME
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get("band")
    except Exception:
        return None


def _band_scalar(run_dir: Path, key: str) -> float:
    band = _load_band(run_dir)
    if band is None:
        return float("nan")
    v = band.get(key)
    return float(v) if v is not None else float("nan")


def _s_band_onset_frac(run_dir: Path) -> float:
    return _band_scalar(run_dir, "band_onset_frac")


def _s_motor_onset_frac(run_dir: Path) -> float:
    return _band_scalar(run_dir, "motor_onset_frac")


def _s_band_width_frac(run_dir: Path) -> float:
    return _band_scalar(run_dir, "band_width_frac")


def _s_peak_kurtosis(run_dir: Path) -> float:
    return _band_scalar(run_dir, "peak_kurtosis_median")


def _s_midband_autocorr(run_dir: Path) -> float:
    return _band_scalar(run_dir, "midband_autocorr_d4")


# Same (fn, label) tuple shape as checkpoint_scalars.SCALAR_EXTRACTORS —
# merge with:  SCALAR_EXTRACTORS.update(LENS_BAND_SCALAR_EXTRACTORS)
LENS_BAND_SCALAR_EXTRACTORS: Dict[str, Tuple] = {
    "lens_band_onset_frac":   (_s_band_onset_frac,  "Lens band onset (frac of depth)"),
    "lens_motor_onset_frac":  (_s_motor_onset_frac, "Lens motor onset (frac of depth)"),
    "lens_band_width_frac":   (_s_band_width_frac,  "Lens band width (frac of depth)"),
    "lens_peak_kurtosis":     (_s_peak_kurtosis,    "Peak lens kurtosis (median)"),
    "lens_midband_autocorr":  (_s_midband_autocorr, "Mid-band top-1 autocorr (d=4)"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """
    python -m p2_eigenspectra.lens_band \\
        --run-dir results/<ts>/pythia-1.4b-step1000_wiki_paragraph \\
        --unembedding results/p2_.../unembedding_pythia-1.4b-step1000.npz \\
        [--out DIR] [--pre-ln-final] [--no-ln]

    Reads activations.npz from the Phase 1 run dir, decodes through the
    saved unembedding (vocab_projection --save-unembedding), and writes
    lens_band_scalars.json (+ .summary.txt) into the run dir so the
    checkpoint scalar extractors find it. Never imports torch.
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Phase 1 per-prompt run dir containing activations.npz")
    p.add_argument("--unembedding", required=True,
                   help="unembedding_{stem}.npz from vocab_projection "
                        "--save-unembedding (must be the same checkpoint "
                        "as the activations).")
    p.add_argument("--out", default=None, help="Output dir; default: --run-dir")
    p.add_argument("--pre-ln-final", action="store_true",
                   help="Set if the final activation entry is PRE-ln_f "
                        "(legacy manual GPT-2 loop). Default assumes the "
                        "standard-path post-ln_f convention.")
    p.add_argument("--no-ln", action="store_true",
                   help="Decode all layers raw (no LN before W_U).")
    args = p.parse_args(argv)

    from p2_eigenspectra.vocab_projection import load_unembedding

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    acts_path = run_dir / "activations.npz"
    if not acts_path.exists():
        print(f"[lens-band] ERROR: {acts_path} not found", file=sys.stderr)
        return 1
    data = np.load(acts_path)
    key = "activations" if "activations" in data else list(data.keys())[0]
    acts = data[key]

    unemb = load_unembedding(args.unembedding)
    print(f"[lens-band] activations {acts.shape}, vocab {unemb['vocab_size']}")

    result = compute_lens_band(
        acts, unemb,
        final_is_post_ln=not args.pre_ln_final,
        apply_ln=not args.no_ln,
    )
    band = detect_band(result)

    jpath = out_dir / LENS_BAND_FILENAME
    with open(jpath, "w") as f:
        json.dump(lens_band_to_json(result, band), f, indent=2)
    spath = out_dir / "lens_band_scalars.summary.txt"
    with open(spath, "w") as f:
        f.write("\n".join(lens_band_summary_lines(result, band)) + "\n")
    print(f"[lens-band] wrote {jpath} and {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
