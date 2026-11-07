"""
prompt_library/sizing_u0.py — the library's size, from the spread T5 U2 already recorded.

Reads T5 U2's per-checkpoint records (`data/trackT/t5u2/2026-10-30/<trajectory>/step*.json`,
written by `trackT_timaeus/t5c.py`, code `3d425b32db90`, the 7 v1 prompts, <= 512 tokens) and
prints, per trajectory, the across-prompt SD of a per-prompt ρ_mid (the median of the prompt's own
ρ_ℓ over the middle states 1 … L−1), its median and maximum over steps, the step of the maximum,
and the standard error of a k-prompt mean at that maximum for k = 7, 256, 1024.

What it does not give: the SD on real held-out documents. The 7 v1 prompts are hand-picked
genres, so a random Pile draw may spread more or less; U1 re-measures on the library itself.
The per-prompt median is not the recorded ρ_mid (the median over states of the prompts' mean);
the two agree to 0.018 or better on these records (printed as `max |gap|`).

    METS_DATA=/path/to/data python prompt_library/sizing_u0.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
RECORDS = DATA / "trackT" / "t5u2" / "2026-10-30"
TRAJECTORIES = ("pythia-70m", "pythia-160m", "pythia-410m",
                "dodecahedron-32m", "dodecahedron-67m", "gpt-2-small")
KS = (7, 256, 1024)


def per_prompt_mid(record: dict) -> np.ndarray:
    """Per-prompt ρ_mid: each prompt's median ρ_ℓ over states 1 … L−1."""
    P = np.asarray(record["rho_per_prompt"], dtype=float)   # (prompts, states)
    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError(f"rho_per_prompt has shape {P.shape}")
    return np.median(P[:, 1:P.shape[1] - 1], axis=1)


def trajectory_rows(traj: str) -> list[tuple[int, float, float]]:
    """(step, across-prompt SD, |recorded ρ_mid − mean of per-prompt medians|) per checkpoint."""
    files = sorted(glob.glob(str(RECORDS / traj / "step*.json")))
    if not files:
        raise SystemExit(f"refused: no records under {RECORDS / traj}")
    rows = []
    for f in files:
        d = json.load(open(f))
        pm = per_prompt_mid(d)
        if len(pm) != 7 or not np.all(np.isfinite(pm)):
            raise SystemExit(f"refused: {f} has {len(pm)} prompts or a non-finite value")
        rows.append((int(d["step"]), float(pm.std(ddof=1)), abs(float(d["rho_mid"]) - float(pm.mean()))))
    return sorted(rows)


def main() -> None:
    print(f"records: {RECORDS}")
    print("trajectory          steps  SD median  SD max  at step  max|gap|  " +
          "  ".join(f"SE(k={k}) at max" for k in KS))
    for traj in TRAJECTORIES:
        rows = trajectory_rows(traj)
        sd = np.array([r[1] for r in rows])
        i = int(np.argmax(sd))
        se = "  ".join(f"{sd[i] / np.sqrt(k):15.4f}" for k in KS)
        print(f"{traj:18s} {len(rows):6d} {np.median(sd):10.3f} {sd[i]:7.3f} {rows[i][0]:8d} "
              f"{max(r[2] for r in rows):9.3f}  {se}")


if __name__ == "__main__":
    main()
