"""
extract_eigenspectrum_stats.py
Usage: python extract_eigenspectrum_stats.py path/to/v_eigenspectrum_*.json [...]

Reads one or more v_eigenspectrum JSON files produced by plots.analyze_value_eigenspectrum
and writes a compact text summary to stdout and to <model_name>_eigen_stats.txt beside
each input file.

Statistics extracted per layer:
  eig_spectral_radius   max |eigenvalue|
  sv_mean / sv_std      singular value distribution summary
  sv_max                leading singular value
  spectral_gap          σ₁ / σ₂
  eig_frac_complex      fraction of eigenvalues with non-trivial imaginary part
  eig_frac_pos_real     fraction Re > 0
  eig_frac_neg_real     fraction Re < 0
  eig_real_mean         mean of Re(eigenvalues)

Cross-layer summary:
  spectral radius  min / mean / max / argmax
  sv_mean          min / mean / max
  eig_frac_complex min / mean / max  (should be ~0.97 everywhere)
  pos/neg real     min / mean / max  (should be ~0.50 everywhere)

Output is designed for direct copy-paste into the cross-run report and README.
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev


def fmt(x, decimals=4):
    if x is None or x != x:  # nan check
        return "nan"
    return f"{x:.{decimals}f}"


def summarize_file(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)

    model = data.get("model", path.stem)
    layers = data.get("layers", {})

    if not layers:
        return f"[{model}] No layer data found.\n"

    # Sort layers: "shared" first, then layer_N by N
    def sort_key(k):
        if k == "shared":
            return -1
        try:
            return int(k.split("_")[1])
        except (IndexError, ValueError):
            return 9999

    sorted_keys = sorted(layers.keys(), key=sort_key)

    lines = []
    lines.append("=" * 72)
    lines.append(f"Model: {model}   ({len(sorted_keys)} layer(s))")
    lines.append("=" * 72)

    # Per-layer table
    header = f"{'Layer':<12} {'Spec.Rad':>10} {'sv_mean':>8} {'sv_max':>8} {'spec_gap':>9} {'frac_cpx':>9} {'frac_pos':>9} {'frac_neg':>9} {'eig_Re_mu':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    radii, sv_means, sv_maxs, frac_cpxs, frac_poss, frac_negs = [], [], [], [], [], []

    for k in sorted_keys:
        d = layers[k]
        r  = d.get("eig_spectral_radius")
        sm = d.get("sv_mean")
        sx = d.get("sv_max")
        sg = d.get("spectral_gap")
        fc = d.get("eig_frac_complex")
        fp = d.get("eig_frac_pos_real")
        fn = d.get("eig_frac_neg_real")
        rm = d.get("eig_real_mean")

        lines.append(
            f"{k:<12} {fmt(r):>10} {fmt(sm):>8} {fmt(sx):>8} {fmt(sg):>9} "
            f"{fmt(fc):>9} {fmt(fp):>9} {fmt(fn):>9} {fmt(rm):>10}"
        )

        for lst, val in [(radii, r), (sv_means, sm), (sv_maxs, sx),
                         (frac_cpxs, fc), (frac_poss, fp), (frac_negs, fn)]:
            if val is not None and val == val:
                lst.append(val)

    lines.append("-" * len(header))

    # Cross-layer summary
    def row(label, vals):
        if not vals:
            return f"  {label:<22} no data"
        mn, mx, mu = min(vals), max(vals), mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        # argmax layer for spectral radius
        return f"  {label:<22} min={fmt(mn)}  mean={fmt(mu)}  max={fmt(mx)}  std={fmt(sd)}"

    lines.append("")
    lines.append("Cross-layer summary:")
    lines.append(row("spectral_radius", radii))

    # argmax layer for spectral radius
    if radii:
        argmax_key = sorted_keys[radii.index(max(radii))]
        lines.append(f"  {'  argmax layer':<22} {argmax_key}")

    lines.append(row("sv_mean", sv_means))
    lines.append(row("sv_max (leading σ)", sv_maxs))
    lines.append(row("eig_frac_complex", frac_cpxs))
    lines.append(row("eig_frac_pos_real", frac_poss))
    lines.append(row("eig_frac_neg_real", frac_negs))

    # Early / middle / late layer spectral radii (for depth-conditioning table)
    if len(sorted_keys) > 1:
        lines.append("")
        lines.append("Spectral radius by position (for depth-conditioning table):")
        n = len(sorted_keys)
        thirds = [
            ("early (first third)",  sorted_keys[:max(1, n//3)]),
            ("middle (middle third)", sorted_keys[n//3: 2*n//3]),
            ("late (final third)",   sorted_keys[2*n//3:]),
        ]
        for label, keys in thirds:
            vals = [layers[k]["eig_spectral_radius"] for k in keys
                    if layers[k].get("eig_spectral_radius") is not None]
            if vals:
                lines.append(f"  {label:<28}  {fmt(min(vals))} – {fmt(max(vals))}  (mean {fmt(mean(vals))})")

    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_eigenspectrum_stats.py path/to/v_eigenspectrum_*.json [...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"[SKIP] File not found: {path}")
            continue

        summary = summarize_file(path)
        print(summary)

        # Write beside input file
        out_path = path.parent / (path.stem + "_stats.txt")
        out_path.write_text(summary)
        print(f"[Saved] {out_path}\n")


if __name__ == "__main__":
    main()
