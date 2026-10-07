"""
p1e_energy_field/long8_targets.py — two facts about the 8 long-passage runs that U2's rule needs
before any field is read (`/challenge-pr` on #157, findings 1 and 4).

1. **Targets.** Per passage: positions kept by T1 + T2 (offset 0 and massive tokens out; massive
   = `move_text.massive_positions` at any of the 18 steps, the union, so the set is one across
   steps) and by T1–T3 (repeats also out, `move_text.kept_offsets`, R0's rule), with the share of
   each quarter of the passage kept.
2. **GPU against CPU by position.** The stored CPU long runs (1d: 4 passages × steps 0 and 143000)
   against this batch: max |Δ| of unit rows at L0–13 and L14–24, per quarter of the positions, so
   a gap that grows with position (length) can be told from one that does not.

No forward pass. Tier 1: exploratory, unregistered.
    python -m p1e_energy_field.long8_targets --runs <p1e_long8 dir> --cpu <p1d_long dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p1d_cluster_ensemble.move_text import kept_offsets, massive_positions

from .extract_long8 import STEPS


def quarters(pos: np.ndarray, n: int) -> list:
    """Share of each quarter of ``range(n)`` that ``pos`` keeps."""
    edges = np.linspace(0, n, 5).astype(int)
    return [float(np.mean(np.isin(np.arange(a, b), pos))) for a, b in zip(edges[:-1], edges[1:])]


def target_positions(runs: Path, key: str) -> tuple:
    """``(tokens, massive {position: [steps]}, T1 + T2, T1–T3)`` of one long passage."""
    massive, tokens = {}, None
    for s in STEPS:
        d = runs / f"pythia-410m-step{s}_{key}"
        for p, (r, L) in massive_positions(np.load(d / "activations.npz")["norms"]).items():
            massive.setdefault(p, []).append(s)
        if tokens is None:
            tokens = json.loads((d / "geometry.json").read_text())["tokens"]
    t12 = np.asarray([p for p in range(len(tokens)) if p != 0 and p not in massive])
    return tokens, massive, t12, kept_offsets(tokens, list(massive))


def targets(runs: Path, key: str) -> dict:
    tokens, massive, t12, t123 = target_positions(runs, key)
    n = len(tokens)
    return {"n": n, "massive": {str(p): v for p, v in sorted(massive.items())},
            "t12": int(t12.size), "t12_quarters": quarters(t12, n),
            "t123": int(t123.size), "t123_quarters": quarters(t123, n)}


def gpu_cpu(gpu: Path, cpu: Path) -> dict:
    g, c = np.load(gpu / "activations.npz")["activations"], np.load(cpu / "activations.npz")["activations"]
    n = g.shape[1]
    edges = np.linspace(0, n, 5).astype(int)
    out = {}
    for name, Ls in (("L0-13", slice(0, 14)), ("L14-24", slice(14, 25))):
        d = np.abs(g[Ls] - c[Ls]).max(axis=(0, 2))
        out[name] = [float(d[a:b].max()) for a, b in zip(edges[:-1], edges[1:])]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--cpu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    keys = sorted({d.name.split("_", 1)[1] for d in a.runs.glob("pythia-410m-step0_*")})
    rec = {"targets": {k: targets(a.runs, k) for k in keys}, "gpu_cpu": {}}
    for k, t in rec["targets"].items():
        print(f"{k:22s} n={t['n']:4d} massive={len(t['massive'])} T1+T2={t['t12']:4d} "
              f"({t['t12'] / t['n']:.0%}) q={[round(x, 2) for x in t['t12_quarters']]} "
              f"T1-T3={t['t123']:4d} ({t['t123'] / t['n']:.0%}) q={[round(x, 2) for x in t['t123_quarters']]}")
    for c in sorted(a.cpu.glob("pythia-410m-step*_long")):
        g = a.runs / c.name
        if g.exists():
            rec["gpu_cpu"][c.name] = r = gpu_cpu(g, c)
            print(c.name, {k: [f"{x:.1e}" for x in v] for k, v in r.items()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
