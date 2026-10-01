"""
p1d_cluster_ensemble/position_check.py — are admission's groups stretches of text?

Build step 2 of `design-1d.md`, its position row ("Checks reported beside
each admitted group"), plus the two diagnostics Blocked 11 asked for: what
step 0's groups are, and whether step 0's failure is only the prompt's
opening (`status-1d.md` "Position").

Per group of an `admit.py` file (every group, admitted or not, both arms):

- ``span``: last minus first member's absolute position;
- ``near_share``: share of member pairs within ``NEAR`` positions (the
  per-group form of `gaussian_null`'s "nearest neighbour within 3");
- ``contiguous``: the members are an unbroken run of kept tokens;
- ``has_first_kept``: the group holds the first kept token (position 0, or
  the first kept position at or after ``--min-position``);
- ``p_near`` / ``p_span``: rank p against ``N_DRAWS`` random groups of the
  same size drawn from the *same record's kept positions* (kept first
  occurrences sit early in the prompt, so all positions would be the wrong
  baseline). ``p_near`` is the upper tail, ``p_span`` the lower.

A group is *positional* at ``p_near <= ALPHA``: a significance flag, not an
effect size, so a large group with a slight tilt toward nearby members is
flagged too (`/challenge-pr` on #123). ``mostly_near`` (``near_share >=
MOSTLY_NEAR``) is the effect-size column beside it. Both are descriptions, not
a gate (`design-1d.md`: a failed check is reported, never a veto). About 5 %
of groups would be flagged by chance; `summarise` puts the not-admitted
groups beside the admitted ones for that reason.

`attention_uniformity` and `cos_to_first` read a run directory: how far
each layer's attention is from uniform over the prefix, and the centred
cosine of each kept token to the first one by position. Tier 1:
exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .admit import BANDS, CONTROL_STEP, load
from .gaussian_null import band_of, first_occurrences, frame_vectors, run_tokens

#: PLACED: "nearby" in positions, as `gaussian_null`'s nearest-neighbour share.
NEAR = 3
#: PLACED: per-group level of the position test; descriptive, not a gate.
ALPHA = 0.05
N_DRAWS = 2000
#: PLACED: a group "mostly of nearby tokens" has at least this share of its
#: member pairs within ``NEAR`` positions.
MOSTLY_NEAR = 0.5
#: Position bins for `cos_to_first` (absolute positions, half-open).
POSITION_BINS = ((1, 4), (4, 16), (16, 64), (64, 200), (200, 2048))


# ---------------------------------------------------------------------------
# One group
# ---------------------------------------------------------------------------

def _near_share(pos: np.ndarray) -> np.ndarray:
    """Share of member pairs within ``NEAR`` positions; ``pos`` is (..., k)."""
    k = pos.shape[-1]
    d = np.abs(pos[..., :, None] - pos[..., None, :])
    iu = np.triu_indices(k, 1)
    return (d[..., iu[0], iu[1]] <= NEAR).mean(axis=-1)


def random_groups(kept: np.ndarray, size: int, n_draws: int, rng: np.random.Generator
                  ) -> Dict[str, np.ndarray]:
    """``near_share`` and ``span`` of ``n_draws`` random size-``size`` subsets of ``kept``."""
    kept = np.asarray(kept)
    idx = np.argpartition(rng.random((n_draws, kept.size)), size - 1, axis=1)[:, :size]
    pos = kept[idx]
    return {"near_share": _near_share(pos), "span": pos.max(1) - pos.min(1)}


def group_position(members: Sequence[int], kept: np.ndarray, null: Dict[str, np.ndarray]
                   ) -> Dict:
    """Position statistics of one group (``members`` index ``kept``) against ``null``."""
    kept = np.asarray(kept)
    m = np.sort(np.asarray(members, dtype=int))
    pos = kept[m]
    near = float(_near_share(pos))
    span = int(pos.max() - pos.min())
    B = null["span"].size
    return {"span": span, "near_share": near,
            "contiguous": bool(np.all(np.diff(m) == 1)),
            "has_first_kept": bool(m[0] == 0),
            "first_quarter_share": float(np.mean(m < kept.size / 4)),
            "p_near": float((1 + np.sum(null["near_share"] >= near)) / (B + 1)),
            "p_span": float((1 + np.sum(null["span"] <= span)) / (B + 1))}


# ---------------------------------------------------------------------------
# A whole admission file
# ---------------------------------------------------------------------------

def check_file(d: Dict, n_draws: int = N_DRAWS, seed: int = 0) -> List[Dict]:
    """One row per group per record per arm, with `group_position`'s fields."""
    rng = np.random.default_rng(seed)
    nulls: Dict[Tuple, Dict] = {}
    rows = []
    for r in d["records"]:
        kept = np.asarray(r["keep"])
        for arm, a in r["arms"].items():
            for g in a["groups"]:
                key = (r["run_dir"], tuple(r["keep"][:1]), kept.size, g["size"])
                if key not in nulls:
                    nulls[key] = random_groups(kept, g["size"], n_draws, rng)
                rows.append({"step": r["step"], "prompt": r["prompt"], "layer": r["layer"],
                             "band": band_of(r["layer"]), "frame": r["info"]["frame"],
                             "arm": int(arm), "size": g["size"],
                             "admitted_excess": g["admitted_excess"],
                             "admitted_log_life": g["admitted_log_life"],
                             **group_position(g["members"], kept, nulls[key])})
    return rows


def summarise(rows: List[Dict], stat: str = "excess") -> List[Dict]:
    """Per (arm, step, frame, band, admitted): counts and position shares."""
    out = []
    cells = sorted({(r["arm"], r["step"], r["frame"], r["band"]) for r in rows},
                   key=lambda c: (c[0], c[1] != CONTROL_STEP, c[2], BANDS.index(c[3])))
    for arm, step, frame, band in cells:
        for adm in (True, False):
            sel = [r for r in rows if (r["arm"], r["step"], r["frame"], r["band"]) == (arm, step, frame, band)
                   and r[f"admitted_{stat}"] is adm]
            if not sel:
                out.append({"arm": arm, "step": step, "frame": frame, "band": band,
                            "admitted": adm, "n": 0})
                continue
            out.append({"arm": arm, "step": step, "frame": frame, "band": band, "admitted": adm,
                        "n": len(sel),
                        "positional": float(np.mean([r["p_near"] <= ALPHA for r in sel])),
                        "compact_span": float(np.mean([r["p_span"] <= ALPHA for r in sel])),
                        "contiguous": float(np.mean([r["contiguous"] for r in sel])),
                        "mostly_near": float(np.mean([r["near_share"] >= MOSTLY_NEAR for r in sel])),
                        "has_first_kept": float(np.mean([r["has_first_kept"] for r in sel])),
                        "median_size": float(np.median([r["size"] for r in sel]))})
    return out


def summary_text(summ: List[Dict], title: str) -> str:
    lines = [title, "",
             "arm step       frame   band   adm    n  positional  mostly_near  compact  contig  first_kept  med size"]
    for s in summ:
        if not s["n"]:
            lines.append(f"{s['arm']:>3} {s['step']:<10} {s['frame']:<7} {s['band']:<6} "
                         f"{'yes' if s['admitted'] else 'no':<4} {0:>4}")
            continue
        lines.append(f"{s['arm']:>3} {s['step']:<10} {s['frame']:<7} {s['band']:<6} "
                     f"{'yes' if s['admitted'] else 'no':<4} {s['n']:>4}  {s['positional']:>10.2f}  {s['mostly_near']:>11.2f}  "
                     f"{s['compact_span']:>7.2f}  {s['contiguous']:>6.2f}  {s['has_first_kept']:>10.2f}  "
                     f"{s['median_size']:>8.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# What step 0's groups are: attention and the first token
# ---------------------------------------------------------------------------

def attention_uniformity(run_dir: Path) -> List[float]:
    """Per layer, the median over heads and positions t >= 1 of the total
    variation distance between the attention row and uniform over ``0..t``."""
    A = np.load(Path(run_dir) / "attentions.npz")["attentions"]
    n = A.shape[-1]
    U = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]
    return [float(np.median(0.5 * np.abs(A[L].astype(np.float64) - U).sum(-1)[:, 1:]))
            for L in range(A.shape[0])]


def cos_to_first(run_dir: Path, layers: Sequence[int], frame: str = "centred") -> Dict[int, List[float]]:
    """Per layer, the mean cosine (in ``frame``, deduped) of kept tokens to
    the first kept token, per `POSITION_BINS` bin of absolute position."""
    acts = np.load(Path(run_dir) / "activations.npz")["activations"]
    keep = first_occurrences(run_tokens(Path(run_dir)))
    out = {}
    for L in layers:
        Z, _ = frame_vectors(acts[L][keep].astype(np.float64), frame)
        c = Z @ Z[0]
        out[int(L)] = [float(c[(keep >= a) & (keep < b)].mean()) if np.any((keep >= a) & (keep < b))
                       else float("nan") for a, b in POSITION_BINS]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="position_check",
                                 description="Position check on admission's groups.")
    ap.add_argument("--real", type=Path, required=True, help="an admit.py real file")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (.txt beside it)")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stat", default="excess", choices=("excess", "log_life"))
    args = ap.parse_args(argv)
    d = load(args.real)
    if d.get("calibrate"):
        print("refusing: a --calibrate file has no token positions to check", file=sys.stderr)
        return 1
    rows = check_file(d, args.n_draws, args.seed)
    summ = summarise(rows, args.stat)
    runs = sorted({r["run_dir"] for r in d["records"]})
    probe = {}
    for rd in runs:
        probe[rd] = {"attention_tv_to_uniform": attention_uniformity(Path(rd)),
                     "cos_to_first_centred": cos_to_first(Path(rd), (4, 12, 20))}
    out = {"real": str(args.real), "n_draws": args.n_draws, "seed": args.seed, "near": NEAR,
           "alpha": ALPHA, "stat": args.stat, "min_position": d.get("min_position", 0),
           "position_bins": POSITION_BINS, "summary": summ, "runs": probe, "groups": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    txt = summary_text(summ, f"position check on {args.real} ({args.stat}, p_near <= {ALPHA}, "
                             f"{args.n_draws} random groups per (run, size))")
    args.out.with_suffix(".txt").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
