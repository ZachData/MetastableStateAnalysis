"""
U1 audit: how far the float32 merge defect reached into a finished U1 output.

Before the fix, the GPU phase merged rows within ``MERGE_RUN`` using a float32 dot, which is good
only to ~1e-6, so a row still moving near a ridge could join a neighbour's basin (design-1e.md
"U1 beside", 2026-10-08). Per record and target set read at every β (``t12``, ``r0``), at
layers 4 / 12 / 20 and the primary β, this compares the stored wells against a CPU float64 run's
(``exact``) and the fixed producer's. For each of the 4 Gaussian draws it compares the stored
``k`` and ``k_eff`` against the fixed producer's, and it refits ``Xw``. Exact runs on the draws
are made on the first cell only, as a check that the fix equals the reference there too.

    python -m p1e_energy_field.u1_audit --out <U1 dir> [--kinds long v1] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .u1_field import (CHECK_LAYERS, FIRST, N_DRAW, SEED, agreement, code_sha, default_opts,
                       frame_rows, gaussian_draw, load_ln1, mean_shift, plan, well_stats)


def audit_record(job, out: Path, opts, device: str) -> list:
    kind, step, key, rd, rev, tg, _ = job
    path = out / "records" / kind / f"step{step}_{key}.json"
    rec = json.loads(path.read_text())
    z = np.load(path.with_suffix(".npz"))
    pb = opts["primary_beta"]
    ln1 = load_ln1(out / "ln1" / f"{rev}.npz")
    acts = np.load(rd / "activations.npz")
    A, N = acts["activations"], acts["norms"]
    first = kind == "long" and (step, key) == FIRST
    rows = []
    for name in (n for n in ("t12", "r0") if f"{n}/wells_{pb:g}" in z.files):
        for L in CHECK_LAYERS:
            cell = next(c for c in rec["cells"] if c["layer"] == L and c["targets"] == name
                        and c["beta"] == pb and not c.get("sweep"))
            U = frame_rows(A[L] * N[L][:, None], ln1, L, tg[name], opts["frame"])[tg[name]]
            stored = z[f"{name}/wells_{pb:g}"][L]
            ref = mean_shift(U, pb, exact=True)["wells"]
            fixed = mean_shift(U, pb, device)["wells"]
            rng = np.random.default_rng([SEED, zlib.crc32(f"{key}|{step}|{L}|{name}".encode())])
            gauss = [gaussian_draw(U, rng) for _ in range(N_DRAW)]
            gw = [mean_shift(Y, pb, device)["wells"] for Y in gauss]
            g = [well_stats(w) for w in gw]
            st, sr = well_stats(stored), well_stats(ref)
            xw = float(np.log(sr["k_eff"]) - np.mean(np.log([x["k_eff"] for x in g])))
            row = {"kind": kind, "step": step, "passage": key, "targets": name, "layer": L,
                   "n": int(len(U)), "agree_stored": agreement(stored, ref),
                   "agree_fixed": agreement(fixed, ref), "k_stored": st["k"], "k_exact": sr["k"],
                   "k_eff_stored": st["k_eff"], "k_eff_exact": sr["k_eff"],
                   "k_G_stored": cell.get("k_G"), "k_G_fixed": [x["k"] for x in g],
                   "Xw_stored": cell.get("Xw"), "Xw_fixed": xw}
            if first:
                row["agree_fixed_G"] = [agreement(w, mean_shift(Y, pb, exact=True)["wells"])
                                        for w, Y in zip(gw, gauss)]
            rows.append(row)
    return rows


def summary(rows: list) -> dict:
    a = np.asarray([r["agree_stored"] for r in rows])
    n = np.asarray([r["n"] for r in rows])
    dx = [abs(r["Xw_fixed"] - r["Xw_stored"]) for r in rows if r["Xw_stored"] is not None]
    return {"cells": len(rows), "cells_stored_differ": int((a < 1).sum()),
            "cells_below_gate": int((a < 0.999).sum()), "min_agree_stored": float(a.min()),
            "rows_differ_upper": int(np.round((1 - a) * n).sum()), "rows": int(n.sum()),
            "cells_k_changed": sum(r["k_stored"] != r["k_exact"] for r in rows),
            "cells_k_G_changed": sum(r["k_G_stored"] != r["k_G_fixed"] for r in rows),
            "max_abs_dXw": float(max(dx)) if dx else None,
            "min_agree_fixed": float(min(r["agree_fixed"] for r in rows)),
            "first_cell_draws_fixed": [x for r in rows for x in r.get("agree_fixed_G", [])]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, required=True, help="a finished U1 output dir")
    ap.add_argument("--kinds", nargs="+", default=["long", "v1"])
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = ap.parse_args(argv)
    meta = json.loads((a.out / "plan.json").read_text())
    opts = meta.get("opts") or default_opts()
    inp = meta["inputs"]
    jobs, _, _ = plan(Path(inp["runs"]), Path(inp["r0"]) if inp.get("r0") else None,
                      Path(inp["r8x"]) if inp.get("r8x") else None)
    code = code_sha()
    dest = a.out / "audit.json"
    done = json.loads(dest.read_text()) if dest.exists() else {"code": code, "rows": []}
    if done["code"] != code:
        raise SystemExit(f"refusing to resume: {dest} was written by {done['code']}, this is {code}")
    seen = {(r["kind"], r["step"], r["passage"]) for r in done["rows"]}
    for j in jobs:
        if j[0] not in a.kinds or (j[0], j[1], j[2]) in seen:
            continue
        done["rows"] += audit_record(j, a.out, opts, a.device)
        done["summary"] = summary(done["rows"])
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(done) + "\n")
        tmp.rename(dest)
        print(f"audited {j[0]} step{j[1]}_{j[2]}", flush=True)
    print(json.dumps(done.get("summary"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
