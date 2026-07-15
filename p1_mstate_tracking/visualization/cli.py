"""
visualization/cli.py

The single command-line entry point:

    python -m p1_mstate_tracking.visualization \\
        --results_dir results/phase1 \\
        --random_seed_dirs results/p1_random \\
        --out         blog_figures \\
        [--phase2_dir results/phase2] [--no_energy_decomposition] \\
        [--all_prompts] [--list_runs] \\
        [--random_run results/p1_random/20250101_120000]

One run of main() prints what was discovered (results dir, run count,
random-seed aggregation, Phase 2 dir) and then drives every figure module
in the package — overview, trained/random pairs, ablation, energy
decomposition, the prompt-aggregate energy attribution, and every
per-model figure. Nothing needs to be run by hand first: random_aggregate
is called in-process (build_aggregate is a normal top-level import now —
moving the per-layer series extractors into series.py removed the
circular-import problem that used to force a lazy import here).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from .style import DEFAULT_PROMPT
from .naming import filter_iteration_depths
from .loaders import discover_runs
from .pipeline import generate_all, _discover_random_dir
from .random_aggregate import build_aggregate
from .energy_attribution_aggregate import generate_energy_attribution_prompt_aggregate_figures


def main():
    parser = argparse.ArgumentParser(
        description="Generate every Phase 1 visualization, one folder per model variant.",
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results/phase1"),
        help="Directory containing per-run subdirs (default: results/phase1)",
    )
    parser.add_argument(
        "--random_seed_dirs", "--random_dirs", dest="random_dirs",
        nargs="*", type=Path, default=[],
        help=(
            "Directories containing the multi-seed untrained/random control "
            "runs, e.g. results/p1_random. Accepts either a single "
            "timestamped seed dir or a parent dir containing several — both "
            "are handled automatically. Every seed found is folded into a "
            "mean ± std aggregate used for the random side of every "
            "overview/pair figure; `runs` itself still keeps just one "
            "representative seed per (model, prompt) for the per-model "
            "figures (PCA scatters etc.), which can only show a single run. "
            "Multiple values are merged; overlapping keys are won by "
            "whichever is discovered last. --random_dirs is accepted as an "
            "alias for this flag."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("blog_figures"),
        help="Output directory for figures (default: blog_figures)",
    )
    parser.add_argument(
        "--phase2_dir", type=Path, default=None,
        help=(
            "Directory containing Phase 2 decompose runs (e.g. "
            "results/phase2), used to additionally generate the "
            "energy-decomposition figures (attention structure vs. "
            "attn/FFN energy attribution) for every (model, prompt) that "
            "has a matching run there. If omitted, defaults to a "
            "'phase2' sibling of --results_dir (e.g. results/phase1 -> "
            "results/phase2) if one exists; models with no Phase 2 run "
            "are skipped silently either way."
        ),
    )
    parser.add_argument(
        "--no_energy_decomposition", action="store_true",
        help="Skip the energy-decomposition figures even if a Phase 2 dir is found.",
    )
    parser.add_argument(
        "--all_prompts", action="store_true",
        help=(
            f"Generate for every prompt found in the data instead of just "
            f"the default ({DEFAULT_PROMPT!r}). To run one different prompt "
            f"by itself, edit DEFAULT_PROMPT in style.py instead."
        ),
    )
    parser.add_argument(
        "--list_runs", action="store_true",
        help="Print discovered runs and exit without generating figures.",
    )
    parser.add_argument(
        "--no_random_aggregate", action="store_true",
        help=(
            "Skip building the multi-seed random aggregate — falls back to "
            "whichever single seed --random_dirs discovers, as before."
        ),
    )
    parser.add_argument(
        "--random_run", type=Path, default=None,
        help=(
            "Path to one specific p1_random seed directory (hand-picked, e.g. "
            "results/p1_random/20250101_120000) used for the multi-prompt "
            "energy-attribution comparison. Unlike --random_seed_dirs, this "
            "is NOT auto-aggregated across every available seed — exactly "
            "one run is used, picked by hand, so a known-degenerate seed can "
            "be avoided. Requires a matching Phase 2 decompose run under "
            "--phase2_dir (or --random_phase2_dir) for the random model's "
            "prompts."
        ),
    )
    parser.add_argument(
        "--energy_aggregate_model", type=str, default="gpt2-large",
        help="Model variant to run the multi-prompt energy-attribution "
             "aggregate for (default: gpt2-large).",
    )
    parser.add_argument(
        "--random_model", type=str, default=None,
        help="Model name to look up inside --random_run "
             "(default: '<energy_aggregate_model>-random').",
    )
    parser.add_argument(
        "--random_phase2_dir", type=Path, default=None,
        help="Phase 2 decompose dir for the random model, if different from "
             "--phase2_dir (default: same as --phase2_dir).",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results_dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovering runs in: {args.results_dir}")
    runs = discover_runs(args.results_dir)
    print(f"  found {len(runs)} runs")

    # random_agg collects the multi-seed (mean, std) summary per (model,
    # prompt) from every --random_seed_dirs entry, in addition to (not
    # instead of) the single representative run `runs` keeps for per-model
    # figures like the PCA scatters, which can only ever show one seed at
    # a time.
    random_agg: Dict[Tuple[str, str], dict] = {}

    for rd in args.random_dirs:
        if not rd.exists():
            print(f"WARNING: random_dir not found, skipping: {rd}", file=sys.stderr)
            continue
        extra = _discover_random_dir(rd)
        if not extra:
            print(f"WARNING: no runs found under random_dir: {rd}", file=sys.stderr)
            continue
        overlap = set(extra.keys()) & set(runs.keys())
        if overlap:
            print(
                f"  Note: {len(overlap)} run key(s) from {rd} overlap existing "
                f"runs — random_dir version will be used",
            )
        runs.update(extra)
        print(f"  + {len(extra)} runs from {rd}")

        if not args.no_random_aggregate:
            agg = build_aggregate(rd)
            if agg:
                multi = {k: v for k, v in agg.items() if v.get("n_runs", 1) > 1}
                print(f"  + aggregate over {len(agg)} (model, prompt) group(s) from {rd}"
                      f" ({len(multi)} with >1 seed)")
                random_agg.update(agg)

    if not runs:
        print("No runs found. Check that geometry.json exists in subdirs.", file=sys.stderr)
        sys.exit(1)

    runs = filter_iteration_depths(runs)

    models = sorted({m for (m, p) in runs.keys()})
    print(f"\nTotal: {len(runs)} runs across {len(models)} model variant(s):")
    for (model, prompt) in sorted(runs.keys()):
        n_seeds = random_agg.get((model, prompt), {}).get("n_runs", 1)
        seed_note = f"  [aggregate: n={n_seeds} seeds]" if n_seeds > 1 else ""
        print(f"  {model:<40} {prompt}{seed_note}")
    print()

    if args.list_runs:
        return

    prompts = sorted({p for (m, p) in runs.keys()}) if args.all_prompts else [DEFAULT_PROMPT]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out}")
    print(f"Prompt(s): {prompts}\n")

    phase2_dir = args.phase2_dir
    if phase2_dir is None and not args.no_energy_decomposition:
        guess = args.results_dir.parent / "phase2"
        phase2_dir = guess if guess.exists() else None
    if args.no_energy_decomposition:
        phase2_dir = None
    if phase2_dir is not None:
        print(f"Phase 2 dir: {phase2_dir}")
    elif not args.no_energy_decomposition:
        print(
            "Phase 2 dir: none found (looked for "
            f"{args.results_dir.parent / 'phase2'}) — "
            "energy-decomposition figures skipped"
        )

    generate_all(
        runs, args.out, prompts=prompts, random_agg=random_agg or None,
        phase2_dir=phase2_dir,
    )

    if phase2_dir is not None:
        print(f"\nPrompt-aggregate energy attribution — {args.energy_aggregate_model}")
        generate_energy_attribution_prompt_aggregate_figures(
            runs, phase2_dir, args.out,
            model=args.energy_aggregate_model,
            random_run_dir=args.random_run,
            random_model=args.random_model,
            random_phase2_dir=args.random_phase2_dir,
        )

    print(f"\nDone. Figures written to: {args.out}")


if __name__ == "__main__":
    main()
