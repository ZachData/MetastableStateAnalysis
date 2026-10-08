"""
p1e_energy_field/u1_calib.py — U1's (1′), the calibrated lumpiness, and the multi-well GPU check
(`design-1e.md` "U1: the rule", rows *(1′)*; `/challenge-pr` on #164, findings 2 and 4).

The matched Gaussian is put on the sphere after drawing, which shrinks its density spread once
the covariance is concentrated, so `Xe > 0` even for a cloud with no structure beyond its first
two moments. Per (run, layer, β): 4 null clouds ``Y_k`` (matched Gaussians of the targets), each
scored as if it were the data against 4 matched Gaussians of its own;
``bias = mean_k [sd(e_{Y_k}) − mean_j sd(e_{G_j(Y_k)})]`` and ``Xe′ = Xe − bias``.

Grams in float32 on the GPU (``--device cuda``), densities in float64; the first cell's bias is
checked against a CPU float64 run (refuse beyond 1e-4). Tier 1: exploratory, unregistered.
    python -m p1e_energy_field.u1_calib run --out <u1 dir> [--device cuda|cpu]
    python -m p1e_energy_field.u1_calib wells --out <u1 dir>
    python -m p1e_energy_field.u1_calib report --out <u1 dir>
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .extract_long8 import STEPS
from .u1_field import (AGREE_MIN, BANDS, BETAS, FIRST, LAYERS, N_DRAW, SEED, agreement, density,
                       load_ln1, mean_shift, unit_rows)
from .u1_report import TRAINED, sign_label

N_NULL = 4
BIAS_TOL = 1e-4
NAMES = ("lumpier", "smoother")
#: Finding 4: cells with several wells (passage, step, layer, β).
WELL_CHECKS = (("wiki_paragraph_long", "128", 4, 5.6), ("wiki_paragraph_long", "128", 4, 10.0),
               ("wiki_paragraph_long", "143000", 12, 10.0))


def _draw(U, rng, xp):
    n = U.shape[0]
    mu = U.mean(axis=0)
    G = xp.asarray(rng.standard_normal((n, n)), dtype=U.dtype)
    Y = mu + G @ (U - mu) / np.sqrt(n - 1)
    return Y / xp.linalg.norm(Y, axis=1, keepdims=True)


def _sd_e_torch(S, beta: float) -> float:
    """``density(S, β).std()`` on the device, float64 (`u1_field.density`: LOO, centred)."""
    import torch
    Z = beta * S
    Z.fill_diagonal_(-float("inf"))
    return float(torch.logsumexp(Z, dim=1).std(unbiased=False))


def bias_cell(U: np.ndarray, rng_key, device: str = "cpu") -> Dict[float, float]:
    """``{β: bias}`` for one target cloud (unit rows, float64)."""
    rng = np.random.default_rng(rng_key)
    if device == "cuda":
        import torch

        class _X:                                    # the two numpy calls ``_draw`` needs
            asarray = staticmethod(lambda a, dtype: torch.as_tensor(a, device="cuda", dtype=dtype))
            linalg = torch.linalg
        xp, U0 = _X, torch.as_tensor(U, device="cuda", dtype=torch.float32)
        gram = lambda Y: (Y @ Y.T).double()                        # noqa: E731
        sd_e = _sd_e_torch
    else:
        xp, U0, gram = np, U, (lambda Y: Y @ Y.T)
        sd_e = lambda S, b: float(density(S, b).std())             # noqa: E731
    sds = {b: [] for b in BETAS}
    for _ in range(N_NULL):
        Y = _draw(U0, rng, xp)
        Sy = gram(Y)
        Sg = [gram(_draw(Y, rng, xp)) for _ in range(N_DRAW)]
        for b in BETAS:
            sds[b].append(sd_e(Sy, b) - np.mean([sd_e(S, b) for S in Sg]))
    return {b: float(np.mean(v)) for b, v in sds.items()}


def _clouds(out: Path, kind: str, step: str, key: str):
    """``(layer, target set, U)`` for every primary cell of one stored record."""
    rec = json.loads((out / "records" / kind / f"step{step}_{key}.json").read_text())
    z = np.load(out / "records" / kind / f"step{step}_{key}.npz")
    rd = Path(rec["run"])
    rev = json.loads((rd / "manifest.json").read_text())["hf_revision"]
    ln1 = load_ln1(out / "ln1" / f"{rev}.npz")
    a = np.load(rd / "activations.npz")
    A, N = a["activations"], a["norms"]
    t = "t12" if kind == "long" else "r0"
    pos = z[f"{t}/positions"]
    for L in LAYERS:
        yield L, t, unit_rows(A[L] * N[L][:, None], ln1["w"][L], ln1["b"][L], ln1["eps"])[pos], rec


def run_one(out: Path, kind: str, step: str, key: str, device: str) -> Path:
    path = out / "calib" / kind / f"step{step}_{key}.json"
    if path.exists():
        return path
    cells = []
    for L, t, U, rec in _clouds(out, kind, step, key):
        bias = bias_cell(U, [SEED + 1, zlib.crc32(f"{key}|{step}|{L}|{t}".encode())], device)
        xe = {c["beta"]: c["Xe"] for c in rec["cells"]
              if c["layer"] == L and c["targets"] == t and not c.get("sweep")}
        cells += [{"layer": L, "targets": t, "beta": b, "Xe": xe[b], "bias": bias[b],
                   "Xe_cal": xe[b] - bias[b]} for b in BETAS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "step": step, "passage": key, "device": device,
                                "cells": cells}) + "\n")
    return path


def run(a) -> int:
    out = a.out
    # first check: the first cell's bias on the device against CPU float64, three layers
    gen = _clouds(out, "long", *FIRST)
    worst = 0.0
    for L, t, U, _ in gen:
        if L not in (4, 12, 20):
            continue
        key = [SEED + 1, zlib.crc32(f"{FIRST[1]}|{FIRST[0]}|{L}|{t}".encode())]
        g, c = bias_cell(U, key, a.device), bias_cell(U, key, "cpu")
        worst = max(worst, max(abs(g[b] - c[b]) for b in BETAS))
    print(f"first check: {a.device} bias against CPU float64, max |Δ| {worst:.1e}", flush=True)
    if worst > BIAS_TOL:
        raise SystemExit(f"refusing: device bias differs from CPU by {worst:.1e} > {BIAS_TOL}")
    jobs = sorted((p.parent.name, p.stem.split("_", 1)[0][4:], p.stem.split("_", 1)[1])
                  for p in (out / "records").glob("*/*.json"))
    for kind, step, key in jobs:
        print(run_one(out, kind, step, key, a.device).name, flush=True)
    (out / "calib" / "first_check.json").write_text(json.dumps({"device": a.device, "max_abs": worst,
                                                                "tol": BIAS_TOL}) + "\n")
    return 0


def wells(a) -> int:
    """Finding 4: GPU float32 wells against CPU float64 on cells with several wells."""
    res = []
    for key, step, L, b in WELL_CHECKS:
        for LL, t, U, _ in _clouds(a.out, "long", step, key):
            if LL != L:
                continue
            g = mean_shift(U, b, "cuda")["wells"]
            c = mean_shift(U, b, exact=True)["wells"]
            res.append({"passage": key, "step": step, "layer": L, "beta": b,
                        "k_gpu": int(g.max() + 1), "k_cpu": int(c.max() + 1),
                        "agreement": agreement(g, c)})
            print(res[-1], flush=True)
    ok = all(r["agreement"] >= AGREE_MIN for r in res)
    (a.out / "calib" / "wells_check.json").write_text(json.dumps({"pass": ok, "cells": res}, indent=1) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def report(a) -> int:
    full = {}
    for kind in ("long", "v1"):
        recs = []
        for p in sorted((a.out / "calib" / kind).glob("*.json")):
            r = json.loads(p.read_text())
            recs += [{"step": int(r["step"]), "passage": r["passage"], **c} for c in r["cells"]]
        if not recs:
            continue
        n_pass = len({c["passage"] for c in recs})
        res = full[kind] = {}
        for b in BETAS:
            acc = defaultdict(lambda: defaultdict(list))
            for c in recs:
                if c["beta"] != b:
                    continue
                for band, layers in BANDS.items():
                    if c["layer"] in layers:
                        acc[(c["step"], band)][c["passage"]].append((c["Xe_cal"], c["bias"], c["Xe"]))
            print(f"\n{kind}: (1′) calibrated lumpiness, β {b:g}  (label, median Xe′, median bias)")
            print(f"{'step':>7} " + " ".join(f"{x:>24}" for x in TRAINED + ("L0",)))
            count = defaultdict(int)
            tab = {}
            for s in STEPS:
                row = []
                for band in TRAINED + ("L0",):
                    d = acc[(s, band)]
                    if len(d) != n_pass:
                        raise SystemExit(f"refusing: {kind} {s} {band}: {len(d)} passages")
                    v = {p: np.mean(np.asarray(x), axis=0) for p, x in d.items()}
                    lab = sign_label([x[0] for x in v.values()], NAMES)
                    med = np.median(np.stack(list(v.values())), axis=0)
                    tab[f"{s}|{band}"] = {"label": lab, "Xe_cal": float(med[0]), "bias": float(med[1]),
                                          "Xe": float(med[2])}
                    if band in TRAINED:
                        count[lab] += 1
                    row.append(f"{lab[:14]:>14} {med[0]:+.3f} {med[1]:+.3f}".rjust(24))
                print(f"{s:>7} " + " ".join(row))
            print("  trained cells:", dict(count))
            res[f"{b:g}"] = {"table": tab, "counts": dict(count)}
    (a.out / "calib" / "labels.json").write_text(json.dumps(full, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "wells", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name == "run":
            p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = ap.parse_args(argv)
    return {"run": run, "wells": wells, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
