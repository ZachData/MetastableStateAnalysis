"""
run.py — Run the induction head tangent experiment from saved Phase 1 data.

Usage:
    python -m tangents.induction.run results/2026-03-15_18-55-33/gpt2_wiki_paragraph
    python -m tangents.induction.run results/2026-03-15_18-55-33/gpt2_wiki_paragraph --threshold 0.03
    python -m tangents.induction.run --scan results/2026-03-15_18-55-33   # all GPT-2 runs in dir

Loads Phase 1 results (metrics.json, attentions.npz, activations.npz) and
runs the three-measurement induction analysis without re-running inference.

Outputs are written to tangent_results/induction/<run_stem>/.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from phase1.io_utils import load_run, load_attentions, load_activations
from tangents.induction.induction import (
    induction_scores_all_layers,
    induction_fiedler_correlation,
    pair_attention_attribution,
    aggregate_pair_attribution,
    induction_pair_energy_trajectory,
    identify_induction_heads,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

TANGENT_RESULTS_DIR = _project_root / "tangent_results" / "induction"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_induction_analysis(
    run_dir: Path,
    out_dir: Path = None,
    threshold: float = 0.04,
    beta: float = 1.0,
) -> dict:
    """
    Full induction head analysis on one Phase 1 run.

    Parameters
    ----------
    run_dir   : path to a Phase 1 run directory (e.g. .../gpt2_wiki_paragraph)
    out_dir   : output directory (default: tangent_results/induction/<stem>)
    threshold : induction score threshold for head identification
    beta      : interaction energy beta for pair energy tracking

    Returns
    -------
    dict with all results
    """
    run_dir = Path(run_dir)
    stem = run_dir.name

    if out_dir is None:
        out_dir = TANGENT_RESULTS_DIR / stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Phase 1 data from: {run_dir}")
    results = load_run(run_dir)
    tokens = results["tokens"]
    model = results["model"]
    prompt = results["prompt"]

    print(f"  Model: {model}  Prompt: {prompt}")
    print(f"  {results['n_layers']} layers, {results['n_tokens']} tokens")

    # Load attention weights (all layers)
    print("  Loading attentions...")
    try:
        attentions = load_attentions(run_dir)  # (n_layers, n_heads, n_tok, n_tok)
    except FileNotFoundError:
        print("  ERROR: attentions.npz not found. Cannot run induction analysis.")
        return {}

    print("  Loading activations...")
    try:
        activations = load_activations(run_dir)  # (n_layers, n_tok, d)
    except FileNotFoundError:
        print("  WARNING: activations.npz not found. Skipping energy trajectory.")
        activations = None

    n_layers, n_heads = attentions.shape[0], attentions.shape[1]
    print(f"  Attentions: {n_layers} layers × {n_heads} heads")

    # ------------------------------------------------------------------
    # Measurement 1: Induction score per head per layer
    # ------------------------------------------------------------------
    print("\n--- Measurement 1: Induction scores ---")
    ind_scores = induction_scores_all_layers(attentions, tokens)  # (n_layers, n_heads)

    # Identify induction heads
    induction_heads = identify_induction_heads(ind_scores, threshold=threshold)
    print(f"  Induction heads (score >= {threshold}):")
    if induction_heads:
        for li, hi, score in induction_heads[:20]:
            print(f"    L{li} H{hi}: {score:.4f}")
        if len(induction_heads) > 20:
            print(f"    ... and {len(induction_heads) - 20} more")
    else:
        print("    None detected. Consider lowering --threshold.")

    # Per-layer summary
    print(f"\n  Per-layer max induction score:")
    for li in range(n_layers):
        layer_max = ind_scores[li].max()
        max_head = int(ind_scores[li].argmax())
        marker = " <<<" if layer_max >= threshold else ""
        print(f"    L{li:2d}: max={layer_max:.4f} (H{max_head}){marker}")

    # ------------------------------------------------------------------
    # Measurement 2: Induction × Fiedler correlation
    # ------------------------------------------------------------------
    print("\n--- Measurement 2: Induction × Fiedler correlation ---")
    corr_results = induction_fiedler_correlation(results, ind_scores)
    print(f"  {corr_results['summary']}")
    print(f"  Per-layer ρ:")
    for lr in corr_results["per_layer"]:
        if not np.isnan(lr["rho"]):
            sig = " *" if lr["pvalue"] < 0.05 else ""
            print(f"    L{lr['layer']:2d}: ρ = {lr['rho']:+.3f}  (p = {lr['pvalue']:.3f}){sig}")

    # ------------------------------------------------------------------
    # Measurement 3: Per-pair attention attribution at plateau layers
    # ------------------------------------------------------------------
    print("\n--- Measurement 3: Per-pair attention attribution ---")
    plateau_layers = results.get("plateau_layers", [])
    if not plateau_layers:
        # Fall back: use layers where nn_stability > 0.8
        plateau_layers = [
            lr["layer"] for lr in results["layers"]
            if lr.get("nn_stability") is not None and lr["nn_stability"] > 0.8
        ]
    print(f"  Plateau layers: {plateau_layers}")

    all_attributed = {}
    all_aggregated = {}
    mutual_pairs_by_layer = {}

    for li in plateau_layers:
        if li >= n_layers:
            continue
        lr = results["layers"][li]
        pa = lr.get("pair_agreement", {})
        mutual_pairs = pa.get("mutual_pairs", [])
        if not mutual_pairs:
            continue

        mutual_pairs_by_layer[li] = mutual_pairs

        attributed = pair_attention_attribution(
            attentions[li], mutual_pairs, ind_scores[li],
        )
        agg = aggregate_pair_attribution(attributed)
        all_attributed[li] = attributed
        all_aggregated[li] = agg

        n_s = agg["n_semantic"]
        n_a = agg["n_artifact"]
        n_n = agg["n_noise"]
        sep = agg["separation"]
        print(f"\n  Layer {li}: {n_s} semantic, {n_a} artifact, {n_n} noise pairs")
        print(f"    Semantic pairs — mean top-head induction: {agg['semantic_mean_induction']:.4f}")
        print(f"    Artifact pairs — mean top-head induction: {agg['artifact_mean_induction']:.4f}")
        if not np.isnan(sep):
            direction = "CONFIRMS" if sep > 0 else "CONTRADICTS"
            print(f"    Separation: {sep:+.4f}  ({direction} prediction)")

        # Print specific pairs
        if attributed:
            print(f"    Top attributed pairs:")
            for p in sorted(attributed, key=lambda x: -x["top_head_induction_score"])[:8]:
                tag_str = f"[{p['tag']:>8s}]"
                print(f"      {tag_str}  {p['tok_i']:>15s} ↔ {p['tok_j']:<15s}  "
                      f"top_head=L{li}H{p['top_head']}  "
                      f"ind={p['top_head_induction_score']:.4f}")

    # ------------------------------------------------------------------
    # Measurement 4: Energy trajectory of artifact vs semantic pairs
    # ------------------------------------------------------------------
    if activations is not None and mutual_pairs_by_layer:
        print("\n--- Measurement 4: Pair energy trajectories ---")
        energy_traj = induction_pair_energy_trajectory(
            activations, mutual_pairs_by_layer, beta=beta,
        )
        print(f"  Tracking {energy_traj['n_artifact_pairs']} artifact pairs, "
              f"{energy_traj['n_semantic_pairs']} semantic pairs across all layers")

        # Summarize: at which layers do artifact pairs gain energy (attraction)?
        art_deltas = energy_traj.get("artifact_mean_delta", [])
        sem_deltas = energy_traj.get("semantic_mean_delta", [])
        if art_deltas:
            art_gaining = [li for li, d in art_deltas if d > 0]
            art_losing = [li for li, d in art_deltas if d < 0]
            print(f"  Artifact pairs gain energy at layers: {art_gaining[:15]}")
            print(f"  Artifact pairs lose energy at layers: {art_losing[:15]}")
        if sem_deltas:
            sem_gaining = [li for li, d in sem_deltas if d > 0]
            sem_losing = [li for li, d in sem_deltas if d < 0]
            print(f"  Semantic pairs gain energy at layers: {sem_gaining[:15]}")
            print(f"  Semantic pairs lose energy at layers: {sem_losing[:15]}")
    else:
        energy_traj = {}

    # ------------------------------------------------------------------
    # Assemble and save
    # ------------------------------------------------------------------
    output = {
        "model": model,
        "prompt": prompt,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": results["n_tokens"],
        "threshold": threshold,
        "beta": beta,
        "induction_scores": ind_scores.tolist(),
        "induction_heads": [
            {"layer": li, "head": hi, "score": s}
            for li, hi, s in induction_heads
        ],
        "fiedler_correlation": corr_results,
        "pair_attribution": {
            str(li): agg for li, agg in all_aggregated.items()
        },
        "plateau_layers": plateau_layers,
        "energy_trajectories": {
            "n_artifact": energy_traj.get("n_artifact_pairs", 0),
            "n_semantic": energy_traj.get("n_semantic_pairs", 0),
            "artifact_mean_delta": energy_traj.get("artifact_mean_delta", []),
            "semantic_mean_delta": energy_traj.get("semantic_mean_delta", []),
        },
    }

    # Save JSON
    with open(out_dir / "induction_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save induction score matrix as npz for downstream use
    np.savez_compressed(
        out_dir / "induction_scores.npz",
        scores=ind_scores,
    )

    # Save full pair attributions (large, separate file)
    if all_attributed:
        with open(out_dir / "pair_attributions.json", "w") as f:
            json.dump(
                {str(li): pairs for li, pairs in all_attributed.items()},
                f, indent=2,
            )

    # Generate text report
    _write_report(output, out_dir)

    print(f"\nResults written to: {out_dir}")
    return output


def _write_report(output: dict, out_dir: Path) -> None:
    """Write a human-readable report."""
    lines = [
        f"INDUCTION HEAD TANGENT EXPERIMENT",
        f"{'=' * 50}",
        f"Model: {output['model']}",
        f"Prompt: {output['prompt']}",
        f"Layers: {output['n_layers']}, Heads: {output['n_heads']}, "
        f"Tokens: {output['n_tokens']}",
        f"Threshold: {output['threshold']}, Beta: {output['beta']}",
        f"",
        f"INDUCTION HEADS DETECTED",
        f"{'-' * 50}",
    ]
    for h in output["induction_heads"][:30]:
        lines.append(f"  L{h['layer']:2d} H{h['head']:2d}: {h['score']:.4f}")
    if len(output["induction_heads"]) > 30:
        lines.append(f"  ... {len(output['induction_heads']) - 30} more")
    if not output["induction_heads"]:
        lines.append("  None detected.")

    lines += [
        f"",
        f"FIEDLER CORRELATION",
        f"{'-' * 50}",
        f"  {output['fiedler_correlation']['summary']}",
        f"  Mean ρ = {output['fiedler_correlation']['mean_rho']:.3f}",
    ]

    lines += [f"", f"PAIR ATTRIBUTION AT PLATEAU LAYERS", f"{'-' * 50}"]
    for li_str, agg in output["pair_attribution"].items():
        sep = agg["separation"]
        lines.append(
            f"  Layer {li_str}: semantic={agg['n_semantic']} artifact={agg['n_artifact']} "
            f"noise={agg['n_noise']}"
        )
        lines.append(
            f"    Semantic mean induction: {agg['semantic_mean_induction']:.4f}"
        )
        lines.append(
            f"    Artifact mean induction: {agg['artifact_mean_induction']:.4f}"
        )
        if not np.isnan(sep):
            direction = "CONFIRMS" if sep > 0 else "CONTRADICTS"
            lines.append(f"    Separation: {sep:+.4f} ({direction} prediction)")

    lines += [f"", f"ENERGY TRAJECTORIES", f"{'-' * 50}"]
    et = output["energy_trajectories"]
    lines.append(f"  Artifact pairs tracked: {et['n_artifact']}")
    lines.append(f"  Semantic pairs tracked: {et['n_semantic']}")

    lines += [f"", f"PREDICTION SUMMARY", f"{'-' * 50}"]
    # Evaluate the three predictions
    mean_rho = output["fiedler_correlation"]["mean_rho"]
    if not np.isnan(mean_rho):
        if mean_rho < -0.2:
            lines.append("  [1] Induction × Fiedler: NEGATIVE correlation — PASS")
        elif mean_rho > 0.2:
            lines.append("  [1] Induction × Fiedler: POSITIVE correlation — FAIL")
        else:
            lines.append("  [1] Induction × Fiedler: WEAK correlation — INCONCLUSIVE")
    else:
        lines.append("  [1] Induction × Fiedler: NO DATA")

    seps = [
        agg["separation"]
        for agg in output["pair_attribution"].values()
        if not np.isnan(agg["separation"])
    ]
    if seps:
        mean_sep = np.mean(seps)
        if mean_sep > 0.005:
            lines.append(f"  [2] Artifact pairs have higher-induction top heads — PASS (Δ={mean_sep:+.4f})")
        elif mean_sep < -0.005:
            lines.append(f"  [2] Semantic pairs have higher-induction top heads — FAIL (Δ={mean_sep:+.4f})")
        else:
            lines.append(f"  [2] No separation between pair types — INCONCLUSIVE (Δ={mean_sep:+.4f})")
    else:
        lines.append("  [2] Pair attribution: NO DATA")

    with open(out_dir / "report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Scan mode: run on all GPT-2 runs in an experiment directory
# ---------------------------------------------------------------------------

def scan_experiment_dir(exp_dir: Path, threshold: float = 0.04, beta: float = 1.0):
    """
    Find all GPT-2 (and optionally BERT) run directories under exp_dir
    and run induction analysis on each.
    """
    exp_dir = Path(exp_dir)
    run_dirs = sorted(exp_dir.iterdir())

    # Filter to runs that have attentions.npz and are GPT-2 or BERT
    targets = []
    for d in run_dirs:
        if not d.is_dir():
            continue
        if not (d / "attentions.npz").exists():
            continue
        name = d.name.lower()
        # Include GPT-2 and BERT runs; skip ALBERT (shared weights, different regime)
        if "gpt2" in name or "bert" in name:
            targets.append(d)

    if not targets:
        print(f"No GPT-2 or BERT runs with attentions.npz found in {exp_dir}")
        return

    print(f"Found {len(targets)} runs to analyze:")
    for t in targets:
        print(f"  {t.name}")

    all_outputs = []
    for run_dir in targets:
        print(f"\n{'=' * 60}")
        try:
            output = run_induction_analysis(
                run_dir, threshold=threshold, beta=beta,
            )
            all_outputs.append(output)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Cross-run summary
    if all_outputs:
        _write_cross_run_summary(all_outputs, TANGENT_RESULTS_DIR)


def _write_cross_run_summary(outputs: list, out_dir: Path) -> None:
    """Summary across all runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"INDUCTION HEAD TANGENT: CROSS-RUN SUMMARY",
        f"{'=' * 50}",
        f"Runs analyzed: {len(outputs)}",
        f"",
    ]

    for o in outputs:
        model = o.get("model", "?")
        prompt = o.get("prompt", "?")
        n_ind = len(o.get("induction_heads", []))
        rho = o.get("fiedler_correlation", {}).get("mean_rho", float("nan"))

        seps = [
            agg["separation"]
            for agg in o.get("pair_attribution", {}).values()
            if not np.isnan(agg.get("separation", float("nan")))
        ]
        mean_sep = float(np.mean(seps)) if seps else float("nan")

        lines.append(f"{model:30s} | {prompt:25s} | "
                     f"induction_heads={n_ind:3d} | "
                     f"ρ={rho:+.3f} | sep={mean_sep:+.4f}")

    with open(out_dir / "cross_run_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nCross-run summary: {out_dir / 'cross_run_summary.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Induction head tangent: detect induction heads in Phase 1 data "
                    "and cross-reference with geometric structure"
    )
    parser.add_argument(
        "run_dir", nargs="?", type=str, default=None,
        help="Path to a Phase 1 run directory (e.g. results/.../gpt2_wiki_paragraph)"
    )
    parser.add_argument(
        "--scan", type=str, default=None, metavar="EXP_DIR",
        help="Scan all GPT-2/BERT runs in an experiment directory"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.04,
        help="Induction score threshold for head identification (default: 0.04)"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Interaction energy beta for pair tracking (default: 1.0)"
    )
    args = parser.parse_args()

    if args.scan:
        scan_experiment_dir(Path(args.scan), threshold=args.threshold, beta=args.beta)
    elif args.run_dir:
        run_induction_analysis(
            Path(args.run_dir), threshold=args.threshold, beta=args.beta,
        )
    else:
        parser.print_help()
