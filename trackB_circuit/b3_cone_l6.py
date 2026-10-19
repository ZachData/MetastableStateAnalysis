"""B3 unit 2b, Blocked 69 (c) (the user's, 2026-10-19): layer 6's 16 heads ablated one at a time
in L1's pass (`trackB_circuit/b3_cone.py`). Descriptive, outside the family of four.

**Why.** At 143000 ℓ is 0.11 at ``resid_pre`` 6 and 2.03 at ``resid_pre`` 7 (`status-B.md`
"B3 unit 2"; `by_layer[l]` is block l's input), so most of the cone is in place after block 6,
before any of L1's members runs. This names which of block 6's heads, if any, carries it.

**What runs.** L1's inputs, c, truncation (tokens < j + 68) and means (``means.npz``, repeated) at
143000, cached from block 6. Configurations: ``none``; ``L6H0`` … ``L6H15``, each mean-ablated
alone; ``attn6`` (all 16 together, beside: attention against block 6's MLP). Per configuration,
ℓ at the final residual (floored as L1) and at ``resid_pre`` 7 (unfloored, as the sweep's by-layer
row). Δℓ = ℓ_none − ℓ_config at both reads.

**Read** (fixed before any output). Per head: Δℓ final, its rank against L1's 200-set null
(``null/``; a 3-head null for a 1-head ablation, so a rank ≥ 0.99 says "larger than three random
later heads remove", not a family reading), Δℓ at ``resid_pre`` 7 and its share of ℓ_none there.
ℓ_none here against the null pass's (cached at block 7): their difference is recorded, refused
above 1e-6. Nothing here changes L1's reading or B3's family.

NUMERICS: as `b3_cone.py` (GPU float32, deterministic, TF32 off; norms float64; statistics CPU
float64).

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b3_cone_l6 run  --out <unit 2 dir>
    python -m trackB_circuit.b3_cone_l6 read --out <unit 2 dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from trackB_circuit import b3_cone as bc
from core import nudge as nd

LAYER = 6
READ_MID = LAYER + 1                        # resid_pre 7: block 6's output
SUB = "l6"
TOL_NONE = 1e-6


def code_sha() -> str:
    h = hashlib.sha256((bc.REPO / "trackB_circuit/b3_cone_l6.py").read_bytes())
    h.update(bc.code_sha().encode())
    return h.hexdigest()[:12]


def configs() -> List[Dict]:
    heads = [LAYER * bc.N_HEADS + h for h in range(bc.N_HEADS)]
    return ([{"name": "none"}] + [{"name": bc.head_name(h), "ablate": [h]} for h in heads]
            + [{"name": "attn6", "ablate": heads}])


def seq_pass(st, hooks, cfgs: Sequence[Dict], s: int, means: np.ndarray, out_file: Path) -> None:
    """One sequence: blocks SITE..5 once per j, then each configuration from block 6; σ at the
    final residual (floored) and at resid_pre 7 (unfloored)."""
    import torch
    J, R = bc.positions(s), bc.directions(s)
    x, r = bc.site_of(st, bc.tokens(s, False))
    eps = bc.C_STAT * r
    mean_t = torch.as_tensor(means, device=st.device, dtype=st.dtype)
    T = int(J.max()) + bc.CUT_L1
    sig = np.full((len(cfgs), bc.N_J, T), np.nan)
    mid = np.full((len(cfgs), bc.N_J, T), np.nan)
    below = np.zeros((len(cfgs), bc.N_J, T), bool)
    floor = np.zeros(bc.N_J)
    with torch.no_grad():
        for a, j in enumerate(J):
            j = int(j)
            n = j + bc.CUT_L1
            hooks.set()
            xc = st.run(bc.batch_input(st, x, R, eps, j, n), bc.SITE, stop=LAYER,
                        keep={LAYER})["resid"][LAYER]
            for k, cf in enumerate(cfgs):
                hooks.set({h: mean_t[h] for h in cf.get("ablate", ())})
                o = st.run(xc, LAYER, keep={READ_MID, bc.N_LAYERS})["resid"]
                if k == 0:
                    alone = st.run(x[:, :n], bc.SITE, keep={bc.N_LAYERS})["resid"][bc.N_LAYERS][0]
                    floor[a] = bc.floor_of(o[bc.N_LAYERS][0], alone)
                sig[k, a, :n], below[k, a, :n] = bc.floored(bc.med_norms(o[bc.N_LAYERS]), floor[a], eps)
                mid[k, a, :n] = bc.med_norms(o[READ_MID]) / eps
            hooks.set()
    np.savez_compressed(out_file, sig=sig, mid=mid, below=below, floor=floor, J=J,
                        names=np.array([c["name"] for c in cfgs]))


def cmd_run(a) -> int:
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    if not g["gate"]["pass"] or g["code"] != bc.code_sha():
        raise SystemExit("refusing: unit 2's gate has not passed on b3_cone.py as it stands")
    bc._gpu()
    import torch
    from trackB_circuit.b0_gate import load
    means = np.load(out / "means.npz")["repeated"]
    model, _ = load(bc.LAST, "cuda", torch.float32)
    st = nd.Stack(model)
    hooks = bc.Hooks(st)
    d = out / SUB
    d.mkdir(exist_ok=True)
    cfgs = configs()
    (d / "meta.json").write_text(json.dumps({"code": code_sha(), "b3_cone": bc.code_sha(),
                                             "numerics": nd.numerics_record(st), "cache_at": LAYER,
                                             "configs": cfgs}) + "\n")
    for s in range(bc.N_SEQ):
        f = d / f"seq{s:02d}.npz"
        if f.exists():
            continue
        t = time.monotonic()
        tmp = d / f"seq{s:02d}.tmp.npz"
        seq_pass(st, hooks, cfgs, s, means, tmp)
        tmp.rename(f)
        z = np.load(f)
        part = z["sig"][:, np.arange(bc.N_J), z["J"] + bc.PARTNER]
        if s == 0 and not (np.isfinite(part).all() and (part > 0).all()):
            raise bc.B3ConeError("seq 0: σ at the partner is non-finite or zero")
        print(f"{SUB} seq {s}: {time.monotonic() - t:.0f}s", flush=True)
    hooks.remove()
    return 0


def read(out: Path) -> Dict:
    d = out / SUB
    files = sorted(d.glob("seq[0-9][0-9].npz"))
    if len(files) != bc.N_SEQ:
        raise bc.B3ConeError(f"{d}: {len(files)} of {bc.N_SEQ} sequences")
    zs = [np.load(f) for f in files]
    T = max(z["sig"].shape[-1] for z in zs)
    pad = lambda a: np.concatenate([a, np.full(a.shape[:-1] + (T - a.shape[-1],), np.nan)], -1)
    sig = np.stack([pad(z["sig"]) for z in zs], 1)
    mid = np.stack([pad(z["mid"]) for z in zs], 1)
    J = np.stack([z["J"] for z in zs])
    names = [str(x) for x in zs[0]["names"]]
    fin = {nm: bc.ell(bc.log_ratio(sig[k], J, bc.PARTNER)) for k, nm in enumerate(names)}
    with np.errstate(divide="ignore", invalid="ignore"):
        mv = {nm: bc.log_ratio(mid[k], J, bc.PARTNER) for k, nm in enumerate(names)}
    md = {nm: (bc.ell(v) if np.all(np.isfinite(v)) else None) for nm, v in mv.items()}
    nul = bc.read_null(out)
    diff = fin["none"] - nul["ell_none"]
    if abs(diff) > TOL_NONE:
        raise bc.B3ConeError(f"ℓ_none here {fin['none']} vs the null pass's {nul['ell_none']}")
    rows = []
    for nm in names[1:]:
        dl = fin["none"] - fin[nm]
        dm = None if md[nm] is None or md["none"] is None else md["none"] - md[nm]
        rows.append({"config": nm, "ell": fin[nm], "delta": dl, "share": dl / fin["none"],
                     "rank_vs_L1_null": bc.rank_of(dl, nul["null"]),
                     "ell_mid": md[nm], "delta_mid": dm,
                     "share_mid": None if dm is None else dm / md["none"]})
    return {"code": code_sha(), "b3_cone": bc.code_sha(), "ell_none": fin["none"],
            "ell_none_null_pass_diff": diff, "ell_none_mid": md["none"], "rows": rows,
            "null_q99": float(np.quantile(nul["null"], 0.99)),
            "L1_members_delta": json.loads((out / "result.json").read_text())["L1"]["delta"]}


def cmd_read(a) -> int:
    out = Path(a.out)
    res = read(out)
    (out / "l6.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"ℓ_none final {res['ell_none']:.3f} (vs null pass {res['ell_none_null_pass_diff']:+.1e}), "
          f"at resid_pre 7 {res['ell_none_mid']:.3f}; L1's null 99th {res['null_q99']:.3f}")
    print("| ablated | Δℓ final | share | rank vs L1 null | Δℓ resid_pre 7 | share there |")
    print("|---|---|---|---|---|---|")
    for r in sorted(res["rows"], key=lambda r: -r["delta"]):
        dm = "—" if r["delta_mid"] is None else f"{r['delta_mid']:.3f}"
        sm = "—" if r["share_mid"] is None else f"{r['share_mid']:.2f}"
        print(f"| {r['config']} | {r['delta']:.3f} | {r['share']:.2f} | {r['rank_vs_L1_null']:.3f} | {dm} | {sm} |")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "read"):
        sub.add_parser(name).add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
