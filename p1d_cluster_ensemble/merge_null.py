"""
p1d_cluster_ensemble/merge_null.py — do stage F's link counts beat a
structureless sequence of partitions with the same cluster sizes?

Reads stage F's saved outputs (one `run_1d --subexp F` directory per
prompt), re-links each prompt's saved partitions, and checks the counts
equal the ones the run wrote (a reader that disagrees with its producer
refuses). Then it draws `merge_tree.link_chain_null` per prompt, pools
the prompts draw by draw (draw i of every prompt summed; prompts are
independent, so this is a draw from the pooled null), and bands by the
boundary's `layer_from` as `status-1d.md` "Merge tree over scales..."
does.

Two statistics carry the questions (`status-1d.md` Parked 2):

- `merge_minus_split`: the L0–L7 merge excess, and `merge_fraction`,
  merges over merges plus splits. Both are needed: the null turns most
  components into tangles, so it has fewer merges *and* fewer splits, and
  a raw difference can beat it while the direction does not.
- `tangle_share`: tangles over all non-stable events (merge, split,
  tangle, birth, death), the "most common change" from L8 on.

Each gets the null's mean, 2.5/97.5 % quantiles and one-sided p in both
tails, `(1 + #{null >= obs}) / (1 + n_draws)` and the same with `<=`.

It also restates the structural counts (blob layers, two-cluster picks,
token 0 out) from the run's JSON, so a `--drop-tokens 0` run can be read
against the full one on the same table.

    python -m p1d_cluster_ensemble.merge_null --v1-only \\
        --in <main>/data/p1d/merge_tree_2026-09-26 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.holdout import add_holdout_args, refuse_held_out

from .merge_tree import _KINDS, link_chain_null, link_counts

#: The bands of `status-1d.md`'s depth table, by the boundary's layer_from.
BANDS = (("L0-L7", 0, 7), ("L8-L15", 8, 15), ("L16-L23", 16, 23))

NON_STABLE = ("merge", "split", "tangle", "birth", "death")


def load_prompt(prompt_dir: Path) -> Dict:
    """One prompt's saved partitions, link settings and structural readout."""
    res = json.loads((prompt_dir / "p1d_results.json").read_text())
    z = np.load(prompt_dir / "p1d_ensemble.npz")
    labels = {int(k.rsplit("_L", 1)[1]): np.asarray(z[k])
              for k in z.files if k.startswith("merge_tree_labels_L")}
    links = res.get("layer_links") or {}
    if "boundaries" not in links:
        raise ValueError(f"{prompt_dir}: no stage F layer_links in p1d_results.json")
    return {"name": prompt_dir.name, "results": res, "labels": labels,
            "layers": [int(l) for l in res["layers"]],
            "measure": links["measure"], "min_overlap": float(links["min_overlap"])}


def observed_counts(prompt: Dict) -> Dict:
    """
    Re-link the saved partitions and refuse unless every boundary matches
    what the run wrote. Returns boundaries and `(n_boundaries, 6)` counts.
    """
    written = [b for b in prompt["results"]["layer_links"]["boundaries"]
               if "skipped" not in b]
    rows, pairs = [], []
    for b in written:
        h, t = int(b["layer_from"]), int(b["layer_to"])
        got = link_counts(prompt["labels"][h], prompt["labels"][t],
                          min_overlap=prompt["min_overlap"], measure=prompt["measure"])
        want = [int(b["counts"][k]) for k in _KINDS]
        if got.tolist() != want:
            raise ValueError(f"{prompt['name']} L{h}->L{t}: re-linked counts "
                             f"{got.tolist()} != written {want}")
        rows.append(got)
        pairs.append((h, t))
    return {"boundaries": pairs, "counts": np.array(rows, dtype=np.int64)}


def band_totals(boundaries: Sequence, counts: np.ndarray) -> Dict[str, np.ndarray]:
    """Sum `counts[..., boundary, kind]` over each band's boundaries."""
    starts = np.array([b[0] for b in boundaries])
    out = {}
    for name, lo, hi in BANDS:
        sel = (starts >= lo) & (starts <= hi)
        out[name] = counts[..., sel, :].sum(axis=-2)
    return out


def statistics(totals: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-kind counts plus the two headline statistics, over the last axis."""
    k = {kind: totals[..., i] for i, kind in enumerate(_KINDS)}
    non_stable = sum(k[x] for x in NON_STABLE)
    ms = k["merge"] + k["split"]
    share = np.where(non_stable > 0, k["tangle"] / np.maximum(non_stable, 1), np.nan)
    frac = np.where(ms > 0, k["merge"] / np.maximum(ms, 1), np.nan)
    return {**k, "non_stable": non_stable,
            "merge_minus_split": k["merge"] - k["split"], "merge_fraction": frac,
            "tangle_share": share}


def compare(obs: float, null: np.ndarray) -> Dict:
    null = np.asarray(null, dtype=float)
    null = null[~np.isnan(null)]
    n = null.size
    if n == 0 or np.isnan(obs):
        return {"observed": float(obs), "n_draws": int(n), "undefined": True}
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_upper": float((1 + (null >= obs).sum()) / (1 + n)),
            "p_lower": float((1 + (null <= obs).sum()) / (1 + n)),
            "n_draws": int(n)}


def structure(prompt: Dict) -> Dict:
    """Blob layers (L1-24), two-cluster picks (L3-24), token 0 out (L1-24)."""
    res = prompt["results"]
    kept = res.get("kept_tokens")
    pos0 = (0 if kept is None else (kept.index(0) if 0 in kept else None))
    blob = two = tok0 = n13 = n324 = 0
    for l_str, rec in res["per_layer"].items():
        l, tree = int(l_str), rec.get("merge_tree") or {}
        if l >= 1:
            n13 += 1
            longest = tree.get("longest_any")
            blob += bool(longest and longest["k_substantial"] == 1)
            if pos0 is not None and l in prompt["labels"]:
                tok0 += bool(prompt["labels"][l][pos0] < 0)
        if l >= 3:
            n324 += 1
            top = tree.get("top_plateau")
            two += bool(top and top["k_substantial"] == 2)
    return {"n_tokens": int(len(next(iter(prompt["labels"].values())))),
            "blob_layers": blob, "of_L1_24": n13,
            "two_cluster_picks": two, "of_L3_24": n324,
            "token0_out": tok0 if pos0 is not None else None}


def run(prompt_dirs: Sequence[Path], n_draws: int, seed: int) -> Dict:
    prompts = [load_prompt(d) for d in sorted(prompt_dirs)]
    obs_band = {name: 0 for name, _, _ in BANDS}
    null_band = {name: 0 for name, _, _ in BANDS}
    per_prompt = {}
    for i, p in enumerate(prompts):
        obs = observed_counts(p)
        null = link_chain_null(p["labels"], layers=p["layers"], n_draws=n_draws,
                               seed=[seed, i], min_overlap=p["min_overlap"],
                               measure=p["measure"])
        if [tuple(b) for b in null["boundaries"]] != obs["boundaries"]:
            raise ValueError(f"{p['name']}: null and observed boundaries differ")
        ob = band_totals(obs["boundaries"], obs["counts"])
        nb = band_totals(obs["boundaries"], null["counts"])
        for name in obs_band:
            obs_band[name] = obs_band[name] + ob[name]
            null_band[name] = null_band[name] + nb[name]
        per_prompt[p["name"]] = {
            "structure": structure(p),
            "bands": {name: {k: int(v) for k, v in zip(_KINDS, ob[name])}
                      for name in ob}}

    bands = {}
    for name in obs_band:
        so, sn = statistics(obs_band[name]), statistics(null_band[name])
        bands[name] = {stat: compare(so[stat], sn[stat]) for stat in so}
    return {"n_prompts": len(prompts), "prompts": [p["name"] for p in prompts],
            "n_draws": int(n_draws), "seed": int(seed),
            "measure": prompts[0]["measure"] if prompts else None,
            "min_overlap": prompts[0]["min_overlap"] if prompts else None,
            "drop_tokens": [p["results"]["settings"].get("drop_tokens") or []
                            for p in prompts][0] if prompts else None,
            "bands": bands, "per_prompt": per_prompt}


def summary_text(out: Dict) -> str:
    lines = [f"merge-tree link null: {out['n_prompts']} prompts, {out['n_draws']} "
             f"draws, seed {out['seed']}, {out['measure']} >= {out['min_overlap']}, "
             f"dropped tokens {out['drop_tokens']}"]
    for name, stats in out["bands"].items():
        lines.append(f"  {name}")
        for stat in ("merge", "split", "tangle", "birth", "death", "stable",
                     "merge_minus_split", "merge_fraction", "tangle_share"):
            c = stats[stat]
            if c.get("undefined"):
                lines.append(f"    {stat:<18} undefined (no events)")
                continue
            lines.append(f"    {stat:<18} obs {c['observed']:8.3f}  null mean "
                         f"{c['null_mean']:8.3f} [{c['null_q025']:.3f}, "
                         f"{c['null_q975']:.3f}]  p_up {c['p_upper']:.4f}  "
                         f"p_lo {c['p_lower']:.4f}")
    lines.append("  structure (blob L1-24, two-cluster pick L3-24, token 0 out L1-24)")
    for name, rec in out["per_prompt"].items():
        s = rec["structure"]
        lines.append(f"    {name:<45} n={s['n_tokens']:<4} blob {s['blob_layers']}/"
                     f"{s['of_L1_24']}  two {s['two_cluster_picks']}/{s['of_L3_24']}"
                     f"  tok0 {s['token0_out']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True,
                    help="stage F output directories (one subdirectory per prompt)")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write")
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    dirs: List[Path] = []
    for d in args.inputs:
        dirs += sorted(x for x in d.iterdir() if (x / "p1d_results.json").exists())
    dirs, record = refuse_held_out(dirs, allow=args.allow_holdout, drop=args.v1_only,
                                   context="merge_null")
    if not dirs:
        print("no stage F outputs found", file=sys.stderr)
        return 1
    out = run(dirs, args.n_draws, args.seed)
    out["holdout"] = record
    out["inputs"] = [str(d) for d in dirs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(summary_text(out))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
