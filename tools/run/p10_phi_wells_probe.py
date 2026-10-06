"""A probe, not a result: the wells of the theory's own density `φ_β` on frozen states
(`p1d_cluster_ensemble/lit-1d.md` §3; `p10_cluster_function/handoff-10.md` Parked, 2026-10-06).

`φ_β(x) = Σ_j exp(β⟨x, x_j⟩)` on the unit sphere; mean shift from every kept token (the states
held fixed: the landscape of one layer, not the dynamics) until no point moves by more than
1e-10 in cosine distance; converged points within 1e-3 are one well. Per (step, passage, layer):
wells, and wells holding >= 2 tokens, in the raw and centred frames, at each β. β = 3.5 [1.6, 5.6]
is the measured all-head value (`STATE.md` Blocked 9, decided 2026-10-04). Both tolerances are
placed. Inputs: R0's labels (kept offsets, Stage 0 run dir). No forward pass. Tier 1, unregistered.
    python tools/run/p10_phi_wells_probe.py --labels <R0 labels> [--steps ...] [--passages ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BETAS = (1.6, 3.5, 5.6, 20.0, 100.0)
MOVE_TOL, MERGE_TOL, MAX_ITER = 1e-10, 1e-3, 300


def wells(X: np.ndarray, beta: float):
    """(wells, wells with >= 2 tokens) of mean shift on φ_β over unit rows X."""
    Y = X.copy()
    for _ in range(MAX_ITER):
        Yn = np.exp(beta * (Y @ X.T - 1)) @ X
        Yn /= np.linalg.norm(Yn, axis=1, keepdims=True)
        done = np.max(1 - np.sum(Yn * Y, 1)) < MOVE_TOL
        Y = Yn
        if done:
            break
    modes, assign = [], np.empty(len(Y), int)
    for i, y in enumerate(Y):
        for k, m in enumerate(modes):
            if 1 - y @ m < MERGE_TOL:
                assign[i] = k
                break
        else:
            modes.append(y)
            assign[i] = len(modes) - 1
    return len(modes), int((np.bincount(assign) >= 2).sum())


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--steps", nargs="*", default=["step143000", "step0"])
    ap.add_argument("--passages", nargs="*", default=["homer_iliad", "wiki_paragraph"])
    ap.add_argument("--layers", nargs="*", type=int, default=[4, 12, 20])
    a = ap.parse_args(argv)
    for step in a.steps:
        lab = json.loads((a.labels / f"{step}.json").read_text())
        for prompt in a.passages:
            p = lab["prompts"][prompt]
            act = np.load(Path(p["stage0_run"]) / "activations.npz")["activations"]
            kept = np.asarray(p["kept"], dtype=int)
            for L in a.layers:
                X = act[L][kept].astype(np.float64)
                frames = {"raw": unit(X), "cen": unit(X - X.mean(0))}
                cells = [f"{f} b{b}:{'/'.join(map(str, wells(Z, b)))}"
                         for f, Z in frames.items() for b in BETAS]
                print(f"{step} {prompt} L{L} n={kept.size} | " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
