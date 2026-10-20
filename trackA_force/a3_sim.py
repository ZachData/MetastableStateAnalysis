"""trackA_force/a3_sim.py — A3 unit 3, row E11's producer (`design-A3.md` §2 E11; "E11's
constants, placed before output").

1e's one-well question as a row: the causal identity-weight dynamics (`sa_sim.field`, causal,
forward Euler with retraction, float64) run from the measured cloud ``f_0(x_0)`` (A1a's start:
the embedding in block 0's LN1 frame, position 0 out) at the prompt's β (A1a's bracket p50, the
median over blocks 0–22 of the layer β), and the cloud's cluster count is read at fixed times.
A cluster is a connected component of the graph "cos ≥ ``CLUSTER_COS``". The located row is the
count at ``T_FINAL``; the other times are beside.

The start is rebuilt from the stored ``activations.npz`` (unit rows × ``norms``), and its mean
cosine must reproduce A1a's record ``g0`` (or the run refuses). Each cell's record is written as
it finishes, so a stopped run resumes.

Tier 1: exploratory, unregistered; v1 prompts only. GPU (float64), about 1 min per cell.
    python -m trackA_force.a3_sim run --out <dir> [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .a1a import DATA, STEPS, V1, frame, mean_cos, run_dir

A1A = DATA / "trackA" / "a1a_batch_2026-10-15" / "records"
LN_FILE = DATA / "trackA" / "ln1_18steps.npz"
DT = 0.01                     # sa_sim's default step; checked by halving on one cell (--quick)
TIMES = (1, 2, 5, 10, 20, 50, 100, 200)
T_FINAL = 200                 # placed: A1a's T_MAX, the simulator's horizon there
CLUSTER_COS = 0.99            # placed
BETA_END = "p50"              # placed: the bracket's median, the prompt's one β


class SimRefused(RuntimeError):
    pass


def n_clusters(U: np.ndarray, thr: float = CLUSTER_COS) -> int:
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    G = csr_matrix((U @ U.T) >= thr)
    return int(connected_components(G, directed=False)[0])


def trajectory(U0: np.ndarray, beta: float, dt: float = DT, times=TIMES) -> dict:
    """Cluster count (and mean cosine) at each time in ``times``, causal dynamics."""
    import torch
    from .sa_sim import field
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = torch.as_tensor(U0, dtype=torch.float64, device=dev)
    U = U / U.norm(dim=1, keepdim=True)
    out, t, k = {}, 0, 0
    marks = [int(round(T / dt)) for T in times]
    for T, m in zip(times, marks):
        while k < m:
            U = U + dt * field(U, beta, True)
            U = U / U.norm(dim=1, keepdim=True)
            k += 1
        Un = U.cpu().numpy()
        out[str(T)] = {"clusters": n_clusters(Un), "mean_cos": mean_cos(Un)}
    return out


def start(step: int, prompt: str, ln) -> np.ndarray:
    a = np.load(run_dir(step, prompt) / "activations.npz")
    x = a["activations"][0].astype(np.float64) * a["norms"][0].astype(np.float64)[:, None]
    i = list(ln["steps"]).index(step)
    return frame(x, ln["w"][i, 0], ln["b"][i, 0], float(ln["eps"]))[1:]


def cell(step: int, prompt: str, ln, dt: float = DT) -> dict:
    a1 = json.loads((A1A / f"step{step}_{prompt}.json").read_text())
    U0 = start(step, prompt, ln)
    g0 = mean_cos(U0)
    if U0.shape[0] != a1["tstar"]["n"] or abs(g0 - a1["tstar"]["g0"]) > 1e-6:
        raise SimRefused(f"step {step} {prompt}: start n {U0.shape[0]} g0 {g0:.9f}, A1a has "
                         f"{a1['tstar']['n']} {a1['tstar']['g0']:.9f}")
    beta = float(a1["beta"]["bracket"][BETA_END])
    return {"step": step, "prompt": prompt, "n": int(U0.shape[0]), "g0": g0, "beta": beta,
            "dt": dt, "clusters_at_start": n_clusters(U0), "traj": trajectory(U0, beta, dt)}


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("a3_sim.py", "sa_sim.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def run(out: Path, quick: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ln = np.load(LN_FILE)
    code = code_sha()
    if quick:      # the dt check: one cell at DT and DT / 2; the counts at T_FINAL must agree
        t0 = time.time()
        a = cell(143000, "homer_iliad", ln)
        b = cell(143000, "homer_iliad", ln, dt=DT / 2)
        rec = {"code": code, "dt": a, "dt_half": b, "elapsed_s": round(time.time() - t0),
               "agree_at_T_FINAL": a["traj"][str(T_FINAL)]["clusters"]
               == b["traj"][str(T_FINAL)]["clusters"]}
        (out / "a3_sim_quick.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: rec[k] for k in ("elapsed_s", "agree_at_T_FINAL")}),
              {T: (a["traj"][str(T)]["clusters"], b["traj"][str(T)]["clusters"]) for T in TIMES})
        return
    cells = out / "cells"
    cells.mkdir(exist_ok=True)
    for s in STEPS:
        for p in V1:
            f = cells / f"step{s}_{p}.json"
            if f.exists():
                if json.loads(f.read_text()).get("code") != code:
                    raise SimRefused(f"refusing to resume: {f} was written by other code")
                continue
            t0 = time.time()
            r = cell(s, p, ln)
            r["code"] = code
            f.write_text(json.dumps(r) + "\n")
            print(f"step {s} {p}: n {r['n']} beta {r['beta']:.2f} clusters "
                  f"{r['clusters_at_start']} -> {r['traj'][str(T_FINAL)]['clusters']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print("done", flush=True)


def load_cells(out: Path) -> dict:
    """{(step, prompt): record} for all 126 cells, or refuse."""
    recs = {}
    for s in STEPS:
        for p in V1:
            f = out / "cells" / f"step{s}_{p}.json"
            if not f.exists():
                raise SimRefused(f"E11: no record {f}")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    run(a.out, a.quick)


if __name__ == "__main__":
    main()
