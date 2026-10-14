"""B1, `L5H2` as positive control (`trackB_circuit/design-B.md` "B1"): does the nudge harness see
`L5H2`'s previous-token signature switch on inside (512, 4000]? Tier 1, 410m (an exploration
rung), no registration.

    python trackB_circuit/b1_prev_token.py run    --out <dir> --gate <B0 records> [--steps ...] [--first-only]
    python trackB_circuit/b1_prev_token.py report --out <dir> --gate <B0 records>

**Refuses unless B0's gate is the one Blocked 44 (a) waived** (the user's, 2026-10-14): a
complete `gate.json` on code `1e78e2be9d94` whose sole failure is G4b at step 128, from one arm,
`camus_letranger` j = 82 (records `2026-10-14c`). Any other failure, or a gate on other code,
and B1 does not run (`design-B.md` "If any arm fails").

**Measured** (the design's, made exact where it left a choice; fixed before any output): B0's
design points (`b0_gate.site`: the 7 scored v1 prompts, block 5's input, B1's 32 seeded positions
J, K = 32 seeded random directions R). For each j in J and each c, one batch of 1 + 2K rows (base,
then ±ε·R_k at token j only) through **block 5 alone**, the sequence cut at j + 6 tokens (causal:
exact for every kept position). Each head h's output at token i, after its `W_O` slice (the
input of `attention.dense`, sliced per head, times that head's columns of its weight; the bias is
shared and cancels in A), is read at i = j + k, k = 0…5:

    σ_k(h) = median over the K directions of ‖A‖ / ε,      A = (y₊ − y₋)/2
    s(h)   = log(σ₁ / median(σ₂ … σ₅)),  median over the 32 j,  at c = 0.03 (the other c beside)

**Verdict** (design-B.md, verbatim in substance): a prompt is *on* at a step when `L5H2`'s s
exceeds that of ≥ 14 of layer 5's other 15 heads; the step is **on** at ≥ 6 of 7 prompts,
**leans** at 5, else **off**. **Pass:** off at step 0 and on at every step of 1000, 2000, 4000,
8000. Steps 2–512 place the onset (recorded, not judged; recorded before any reading: the onset
falls in (256, 1000]).

**Rounding floor** (fixed here, `status-B.md` Fences "B1's floor"; B0's same-shape floor is 0
under deterministic algorithms): per (prompt, j), the larger of (i) the max ‖Δ‖ over heads and
the six tokens between the base row of the 65-row batch and the same block run on the base
alone (another batch shape), and (ii) float32's eps times the largest ‖y‖ there (one ulp of the
readout). A σ_k(h) whose median ‖A‖ is ≤ 10× the floor is *below floor*; in s, a σ under the
floor itself is replaced by it (a detection limit: a sharp head's softmax weight underflows to
exactly 0 a few tokens back in float32, so σ_k = 0 occurs; *set after the first run refused on
it, step 1000, before any verdict was read*). A prompt cannot count
as on where `L5H2`'s σ₁ is below floor at more than half its 32 j (the verdict would rest on
rounding); below-floor counts for every head and k are reported.

**Beside, not judged:** `cos(A, W_O W_V · LN′(x_j)[d])` for `L5H2` at k = 1, median over the K
directions (is the response the OV-mapped nudge; LN′ is block 5's input LayerNorm's exact JVP at
token j); the yardstick in residual units, median ‖A‖ of `L5H2`'s output at k = 1 over the RMS
token norm of block 5's output, at every c.

NUMERICS: GPU float32, deterministic algorithms, TF32 off (`core.nudge.set_numerics`); every pass
of one comparison is one batch on one device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

import numpy as np

from core import nudge as nd
from trackB_circuit import b0_gate as g0

STEPS = g0.STEPS
FIRST = 1000                         # run first: inside the pass window, the populated check
LAYER, HEAD, N_HEADS, HEAD_DIM = 5, 2, 16, 64
KS = (0, 1, 2, 3, 4, 5)
C_STAT = 0.03
ON_HEADS, ON_PROMPTS, LEAN_PROMPTS = 14, 6, 5
PASS_STEPS = (1000, 2000, 4000, 8000)
U32 = float(np.finfo(np.float32).eps)
WAIVED = {"code": "1e78e2be9d94", "records": "2026-10-14c", "step": 128,
          "prompt": "camus_letranger", "j": 82}          # Blocked 44 (a), the user's, 2026-10-14


class B1Error(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    for p in ("core/nudge.py", "trackB_circuit/b0_gate.py"):
        h.update((REPO / p).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- B0's gate, as waived

def check_gate(gate_dir: Path) -> Dict:
    """Refuse unless B0's gate failed only at Blocked 44 (a)'s named arm."""
    gate_dir = Path(gate_dir)
    if gate_dir.name != WAIVED["records"]:
        raise B1Error(f"B0 records {gate_dir.name}, the waiver names {WAIVED['records']}")
    g = json.loads((gate_dir / "gate.json").read_text())
    if g["code"] != WAIVED["code"] or not g["complete"]:
        raise B1Error(f"gate code {g['code']} complete={g['complete']}: not the waived gate")
    if g["prompts_hash"] != g0.prompts_hash():
        raise B1Error("the gate ran on another prompt set")
    bad = [s for s, v in g["steps"].items() if not v["pass"]]
    if bad != [str(WAIVED["step"])]:
        raise B1Error(f"gate fails at steps {bad}; the waiver covers step {WAIVED['step']} only")
    v = g["steps"][str(WAIVED["step"])]
    if not all(v[k] is True for k in ("G0", "G1", "G2", "G3", "G4a")) or v["G4b_lin"]["rank1_mismatch"] != 1:
        raise B1Error("step 128 fails more than G4b's one arm")
    rec = json.loads((gate_dir / f"gpu_step{WAIVED['step']}.json").read_text())
    arms = [(key, a["j"]) for key, p in rec["prompts"].items() for a in p["G4b"]["arms"]
            if a["c"][str(g0.G4B_C)]["rank"] is not None
            and (a["c"][str(g0.G4B_C)]["rank"] == 1) != (a["lin_rank"] == 1)]
    if arms != [(WAIVED["prompt"], WAIVED["j"])]:
        raise B1Error(f"step 128's mismatched arms are {arms}, not the waived one")
    return dict(WAIVED, gate_pass=False, waived=True)


# ---------------------------------------------------------------- the statistic (pure)

def s_stat(sigma: np.ndarray, lo: Optional[np.ndarray] = None) -> np.ndarray:
    """``sigma`` (J, 6, H) → s per head: median over j of log(σ₁ / median(σ₂…σ₅)). ``lo`` (J,),
    the floor in σ units per j: a σ under it is replaced by it (a detection limit)."""
    sig = np.asarray(sigma, np.float64)
    if lo is not None:
        sig = np.maximum(sig, np.asarray(lo, np.float64)[:, None, None])
    return np.median(np.log(sig[:, 1] / np.median(sig[:, 2:6], axis=1)), axis=0)


def heads_beaten(s: np.ndarray, head: int = HEAD) -> int:
    return int(np.sum(np.delete(s, head) < s[head]))


def step_reading(n_on: int) -> str:
    return "on" if n_on >= ON_PROMPTS else "leans" if n_on >= LEAN_PROMPTS else "off"


def verdict(readings: Dict[int, str]) -> Dict:
    """Pass: off at step 0, on at every step of PASS_STEPS (design-B.md "Pass (exit)")."""
    missing = [s for s in (0,) + PASS_STEPS if s not in readings]
    if missing:
        return {"verdict": None, "missing": missing}
    ok = readings[0] == "off" and all(readings[s] == "on" for s in PASS_STEPS)
    return {"verdict": "sees" if ok else "does not see", "step0": readings[0],
            "window": {s: readings[s] for s in PASS_STEPS}}


# ---------------------------------------------------------------- the GPU pass

class HeadOut:
    """Captures block 5's ``attention.dense`` input; ``y`` gives every head's output after its
    ``W_O`` slice, (B, T, H, D)."""

    def __init__(self, st: "nd.Stack"):
        att = st.layers[LAYER].attention
        self.W = att.dense.weight.detach().view(-1, N_HEADS, HEAD_DIM)       # (D, H, e)
        self.z = None
        self.h = att.dense.register_forward_pre_hook(self._grab)

    def _grab(self, mod, args):
        self.z = args[0]

    def y(self, t0: int, t1: int):
        import torch
        z = self.z[:, t0:t1].reshape(self.z.shape[0], t1 - t0, N_HEADS, HEAD_DIM)
        return torch.einsum("bthe,dhe->bthd", z, self.W)


def ov_of(st: "nd.Stack", xj, R: np.ndarray):
    """``W_O W_V`` of `L5H2` applied to the input LayerNorm's JVP at token j along each R_k,
    (K, D). NeoX's ``query_key_value`` rows are per head [q | k | v]."""
    import torch
    blk = st.layers[LAYER]
    d = torch.as_tensor(R, device=st.device, dtype=st.dtype)
    _, t = torch.func.jvp(blk.input_layernorm, (xj.expand(len(R), -1).contiguous(),), (d,))
    Wqkv = blk.attention.query_key_value.weight.view(N_HEADS, 3 * HEAD_DIM, -1)
    Wv = Wqkv[HEAD, 2 * HEAD_DIM:]                                          # (e, D)
    Wo = blk.attention.dense.weight.view(-1, N_HEADS, HEAD_DIM)[:, HEAD]     # (D, e)
    return (t @ Wv.T) @ Wo.T


def run_prompt(st, ho: HeadOut, ids, key: str) -> Dict:
    import torch
    x, r, J, R = g0.site(st, ids, key)
    with torch.no_grad():
        out5 = st.run(x, LAYER, stop=LAYER + 1, keep={LAYER + 1})["resid"][LAYER + 1]
    r_out = nd.rms_norm(out5[0].double().cpu().numpy())
    rec = {"n": len(ids), "r": r, "r_out": r_out, "J": [int(j) for j in J],
           "floor": [], "c": {str(c): {"sigma": [], "medA": [], "cos1": [], "yard1": []} for c in nd.C_GRID}}
    for j in J:
        j = int(j)
        n = j + 6
        xs = x[:, :n]
        with torch.no_grad():
            st.block(LAYER, xs)
            y1 = ho.y(j, n)[0].double()                                     # (6, H, D)
            ov = ov_of(st, x[0, j], R).double()
        floor_j = None
        for c in nd.C_GRID:
            eps = c * r
            with torch.no_grad():
                st.block(LAYER, xs.expand(2 * len(R) + 1, -1, -1) + st.deltas(R, eps, [j], n))
                y = ho.y(j, n).double()                                     # (65, 6, H, D)
            b, p, m = nd.split_rows(y)
            if floor_j is None:
                shape = float((b - y1).norm(dim=-1).max())
                floor_j = max(shape, U32 * float(y1.norm(dim=-1).max()))
                rec["floor"].append({"shape": shape, "floor": floor_j})
            A, _ = nd.anti_sym(p, m, b)                                      # (K, 6, H, D)
            nA = A.norm(dim=-1).cpu().numpy()                                # (K, 6, H)
            medA = np.median(nA, axis=0)                                     # (6, H)
            cos = torch.nn.functional.cosine_similarity(A[:, 1, HEAD], ov, dim=-1).cpu().numpy()
            C = rec["c"][str(c)]
            C["sigma"].append((medA / eps).tolist())
            C["medA"].append(medA.tolist())
            C["cos1"].append(float(np.median(cos)))
            C["yard1"].append(float(medA[1, HEAD]) / r_out)
    return rec


def run_step(step: int) -> Dict:
    import torch
    t0 = time.monotonic()
    model, tok = g0.load(step, "cuda", torch.float32)
    st = nd.Stack(model)
    ho = HeadOut(st)
    rec = {"step": step, "code": code_sha(), "prompts_hash": g0.prompts_hash(),
           "numerics": nd.numerics_record(st), "head": f"L{LAYER}H{HEAD}", "prompts": {}}
    for key in g0.V1_SCORED:
        rec["prompts"][key] = run_prompt(st, ho, g0.token_ids(tok, key), key)
    ho.h.remove()
    rec["seconds"] = time.monotonic() - t0
    del model
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- reading a record

def read_step(rec: Dict) -> Dict:
    out = {"c": {}}
    for c in nd.C_GRID:
        per = {}
        for key, p in rec["prompts"].items():
            C = p["c"][str(c)]
            sig, medA = np.array(C["sigma"]), np.array(C["medA"])           # (J, 6, H)
            fl = np.array([f["floor"] for f in p["floor"]])[:, None, None]
            below = medA <= nd.FLOOR_MULT * fl
            s = s_stat(sig, fl[:, 0, 0] / (c * p["r"]))
            sig1_below = int(below[:, 1, HEAD].sum())
            beaten = heads_beaten(s)
            per[key] = {"s": s.tolist(), "s_L5H2": float(s[HEAD]), "beaten": beaten,
                        "sigma1_below_floor": sig1_below, "below_floor_any": int(below.sum()),
                        "on": beaten >= ON_HEADS and sig1_below <= len(sig) // 2,
                        "cos1_median": float(np.median(C["cos1"])),
                        "yard1_median": float(np.median(C["yard1"]))}
        n_on = sum(v["on"] for v in per.values())
        out["c"][str(c)] = {"prompts": per, "n_on": n_on, "reading": step_reading(n_on)}
    out["reading"] = out["c"][str(C_STAT)]["reading"]
    return out


# ---------------------------------------------------------------- CLI

def _steps(a) -> List[int]:
    return sorted(a.steps or STEPS, key=lambda s: s != FIRST)


def cmd_run(a) -> int:
    waiver = check_gate(a.gate)
    nd.set_numerics()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (B1 is GPU float32)")
    a.out.mkdir(parents=True, exist_ok=True)
    for step in _steps(a):
        path = a.out / f"b1_step{step}.json"
        if path.exists():
            continue
        rec = run_step(step)
        rec["gate"] = waiver
        sig = np.array([p["c"][str(c)]["sigma"] for p in rec["prompts"].values() for c in nd.C_GRID])
        if not np.all(np.isfinite(sig)) or not np.all(sig[:, :, 0] > 0):
            raise B1Error(f"step {step}: a non-finite σ, or a head silent at its own token; not written")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        tmp.rename(path)
        r = read_step(rec)
        cc = r["c"][str(C_STAT)]
        print(f"done step {step} in {rec['seconds']:.0f}s: {r['reading']} ({cc['n_on']}/7 on); L5H2 s "
              + " ".join(f"{v['s_L5H2']:.2f}[{v['beaten']}]" for v in cc["prompts"].values()), flush=True)
        if a.first_only:
            return 0
    return 0


def cmd_report(a) -> int:
    check_gate(a.gate)
    rows = {}
    for step in STEPS:
        path = a.out / f"b1_step{step}.json"
        if not path.exists():
            print(f"step {step}: missing record")
            continue
        rec = json.loads(path.read_text())
        if rec["code"] != code_sha() or rec["prompts_hash"] != g0.prompts_hash():
            raise B1Error(f"step {step}: record on code {rec['code']} (now {code_sha()}) or another prompt set")
        rows[step] = read_step(rec)
    v = verdict({s: r["reading"] for s, r in rows.items()})
    summary = {"code": code_sha(), "prompts_hash": g0.prompts_hash(), "gate": WAIVED,
               "complete": len(rows) == len(STEPS), "verdict": v,
               "steps": {s: {"reading": r["reading"],
                             "c": {c: {"n_on": x["n_on"], "reading": x["reading"],
                                       "s_L5H2_median": float(np.median([p["s_L5H2"] for p in x["prompts"].values()])),
                                       "beaten": [p["beaten"] for p in x["prompts"].values()],
                                       "sigma1_below_floor": sum(p["sigma1_below_floor"] for p in x["prompts"].values()),
                                       "below_floor_any": sum(p["below_floor_any"] for p in x["prompts"].values()),
                                       "cos1_median": float(np.median([p["cos1_median"] for p in x["prompts"].values()])),
                                       "yard1_median": float(np.median([p["yard1_median"] for p in x["prompts"].values()]))}
                                   for c, x in r["c"].items()}} for s, r in rows.items()}}
    (a.out / "b1.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("| step | reading (c = 0.03) | prompts on | L5H2 s, median | heads beaten (of 15), per prompt "
          "| σ₁ below floor | cos(A, OV d) at k = 1 | ‖A‖/r_out at k = 1 | reading at c = " + ", ".join(map(str, nd.C_GRID)) + " |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s, row in summary["steps"].items():
        x = row["c"][str(C_STAT)]
        print(f"| {s} | {row['reading']} | {x['n_on']}/7 | {x['s_L5H2_median']:.2f} | {x['beaten']} | "
              f"{x['sigma1_below_floor']} | {x['cos1_median']:.3f} | {x['yard1_median']:.2e} | "
              + ", ".join(row["c"][str(c)]["reading"] for c in nd.C_GRID) + " |")
    print(f"B1: {v}")
    return 0 if v.get("verdict") == "sees" else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--gate", type=Path, required=True)
        if name == "run":
            p.add_argument("--steps", type=int, nargs="*")
            p.add_argument("--first-only", action="store_true")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
