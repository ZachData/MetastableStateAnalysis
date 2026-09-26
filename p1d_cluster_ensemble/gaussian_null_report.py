"""
p1d_cluster_ensemble/gaussian_null_report.py — the tables `status-1d.md`
"Matched-covariance Gaussian null" quotes, from `gaussian_null.py`'s output.

Two parts:

- ``table``: per (step, frame, band, statistic), the real records against
  the calibration records *with the same step, frame, band and dedupe
  setting*. Refuses a calibration file whose dedupe flag differs from the
  real one's (`/challenge-pr` on #106 found the first write-up reading
  deduped results against an all-token calibration pooled over layers).
- ``neighbours``: what a token's nearest neighbour is, per step and band
  (same token string; the adjacent position; within 3 positions), in the raw
  and centred frames, on all tokens or deduped.

Both exclude ``repeated_tokens`` by default (3 distinct strings; it is
skipped when deduped, so leaving it in would change the prompt set between
columns).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .gaussian_null import (FRAMES, STATISTICS, band_of, first_occurrences,
                            frame_vectors, run_tokens)

#: The tail read as "lumpier". hdb_k and mt_k have no direction in
#: `gaussian_null.LUMPIER`; here their upper tail (more clusters than the
#: Gaussian) is the one counted.
TAIL = {"ci2": "p_lower", "nn1": "p_lower", "hdb_noise": "p_lower",
        "mt_life": "p_upper", "hdb_k": "p_upper", "mt_k": "p_upper"}
BANDS = ("L1-8", "L9-16", "L17-24", "L1-24")
EXCLUDE = ("repeated_tokens",)


def load(path: Path) -> Dict:
    d = json.loads(Path(path).read_text())
    d["records"] = [r for r in d["records"] if "skipped" not in r]
    return d


def _z(s: Dict) -> Optional[float]:
    return (s["obs"] - s["null_mean"]) / s["null_sd"] if s["null_sd"] > 0 else None


def _cell(recs: List[Dict], stat: str, alpha: float) -> Dict:
    zs = [z for z in (_z(r["stats"][stat]) for r in recs) if z is not None]
    return {"n": len(recs),
            "obs": float(np.median([r["stats"][stat]["obs"] for r in recs])) if recs else None,
            "null": float(np.median([r["stats"][stat]["null_mean"] for r in recs])) if recs else None,
            "z": float(np.median(zs)) if zs else None,
            "tail": int(sum(r["stats"][stat][TAIL[stat]] <= alpha for r in recs))}


def _select(records: List[Dict], step: str, frame: str, band: str,
            exclude: Sequence[str]) -> List[Dict]:
    return [r for r in records if r["step"] == step and r["info"]["frame"] == frame
            and r["layer"] > 0 and r["prompt"] not in exclude
            and (band == "L1-24" or band_of(r["layer"]) == band)]


def table(real: Dict, cal: Dict, alpha: float = 0.025,
          exclude: Sequence[str] = EXCLUDE) -> List[Dict]:
    if not cal.get("calibrate"):
        raise ValueError("the calibration file was not written with --calibrate")
    if bool(real.get("dedupe_strings")) != bool(cal.get("dedupe_strings")):
        raise ValueError("real and calibration differ in --dedupe-strings; "
                         "a calibration is only read against its own inputs")
    rows = []
    for stat in STATISTICS:
        for step in sorted({r["step"] for r in real["records"]}):
            for frame in FRAMES:
                for band in BANDS:
                    rr = _select(real["records"], step, frame, band, exclude)
                    cc = _select(cal["records"], step, frame, band, exclude)
                    if not rr:
                        continue
                    c = _cell(cc, stat, alpha)
                    rows.append({"stat": stat, "step": step, "frame": frame, "band": band,
                                 "real": _cell(rr, stat, alpha),
                                 "cal": {"n": c["n"], "z": c["z"], "tail": c["tail"]}})
    return rows


def neighbours(run_dirs: Sequence[Path], dedupe: bool,
               exclude: Sequence[str] = EXCLUDE) -> List[Dict]:
    """Nearest-neighbour shares per (step, frame, band): same string, |i-j| = 1, |i-j| <= 3."""
    acc: Dict[tuple, Dict[str, List[float]]] = {}
    for rd in run_dirs:
        rd = Path(rd)
        model, _, prompt = rd.name.partition("_")
        if prompt in exclude:
            continue
        step = model.rsplit("-", 1)[-1]
        acts = np.load(rd / "activations.npz")["activations"]
        tokens = np.array(run_tokens(rd))
        if tokens.size != acts.shape[1]:
            raise ValueError(f"{rd}: {tokens.size} token strings for {acts.shape[1]} rows")
        keep = first_occurrences(tokens) if dedupe else np.arange(tokens.size)
        for layer in range(1, acts.shape[0]):
            for frame in ("raw", "centred"):
                Z, _ = frame_vectors(acts[layer][keep], frame)
                G = Z @ Z.T
                np.fill_diagonal(G, -np.inf)
                nn = keep[G.argmax(axis=1)]
                gap = np.abs(nn - keep)
                vals = {"same_string": float(np.mean(tokens[nn] == tokens[keep])),
                        "adjacent": float(np.mean(gap == 1)),
                        "within_3": float(np.mean(gap <= 3))}
                for band in (band_of(layer), "L1-24"):
                    d = acc.setdefault((step, frame, band), {k: [] for k in vals})
                    for k, v in vals.items():
                        d[k].append(v)
    return [{"step": s, "frame": f, "band": b, "n": len(v["adjacent"]),
             **{k: [float(np.min(x)), float(np.median(x)), float(np.max(x))] for k, x in v.items()}}
            for (s, f, b), v in sorted(acc.items())]


def text(rows: List[Dict], nbrs: List[Dict], dedupe: bool) -> str:
    out = [f"Gaussian null report ({'deduped' if dedupe else 'all tokens'}; "
           f"without {', '.join(EXCLUDE)}; L1-24). real: median obs, median null mean, "
           "median z, records in the lumpier 2.5 % tail. cal: the same step/frame/band "
           "on one Gaussian draw per record."]
    stat = None
    for r in rows:
        if r["stat"] != stat:
            stat = r["stat"]
            out.append(f"\n[{stat}] ({TAIL[stat]})  step       frame            band     "
                       "real: n   obs      null     z      tail   | cal: n  z      tail")
        a, c = r["real"], r["cal"]
        cz = f"{c['z']:+6.2f}" if c["z"] is not None else "   n/a"
        az = f"{a['z']:+6.2f}" if a["z"] is not None else "   n/a"
        out.append(f"  {r['step']:<10} {r['frame']:<16} {r['band']:<7}  {a['n']:>4} "
                   f"{a['obs']:8.4f} {a['null']:8.4f} {az} {a['tail']:>4}   | "
                   f"{c['n']:>4} {cz} {c['tail']:>4}")
    if nbrs:
        out.append("\n[nearest neighbour] share of tokens whose nearest neighbour is ... "
                   "(min / median / max over prompt-layers)")
        for r in nbrs:
            out.append(f"  {r['step']:<10} {r['frame']:<8} {r['band']:<7} n={r['n']:>3}  "
                       + "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}"
                                   for k, v in r.items() if k in ("same_string", "adjacent", "within_3")))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--null", type=Path, required=True, help="gaussian_null.py output")
    ap.add_argument("--calibrate", type=Path, required=True, help="its --calibrate output")
    ap.add_argument("--out", type=Path, required=True, help="JSON to write (text beside it)")
    args = ap.parse_args(argv)
    real, cal = load(args.null), load(args.calibrate)
    try:
        rows = table(real, cal)
    except ValueError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 1
    dedupe = bool(real.get("dedupe_strings"))
    nbrs = neighbours([Path(p) for p in real["inputs"]], dedupe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"null": str(args.null), "calibrate": str(args.calibrate),
                                    "dedupe_strings": dedupe, "table": rows,
                                    "neighbours": nbrs}, indent=1))
    t = text(rows, nbrs, dedupe)
    args.out.with_suffix(".txt").write_text(t + "\n")
    print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
