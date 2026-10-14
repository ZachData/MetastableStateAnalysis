"""B0's calibration gate (`trackB_circuit/design-B.md` "The calibration gate"), on B1's design
points: `pythia-410m`, the 18 steps, the 7 scored v1 prompts, `resid_pre` at layer 5, B1's 32
seeded nudge positions per prompt, K = 32 random directions, the four c. Tier 1.

    python trackB_circuit/b0_gate.py run    --out <dir> [--steps ...]   # GPU float32: G0, G1, G3, G4a, G4b
    python trackB_circuit/b0_gate.py g2     --out <dir> [--steps ...]   # CPU float64: G2
    python trackB_circuit/b0_gate.py report --out <dir>

**Rules, fixed before any output** (the design's, made exact where it left a choice):

| arm | read | passes when (at every step) |
|---|---|---|
| G0 (added) | the harness's loop against the HF forward, all logits | max abs Δ ≤ 1e-5, every prompt |
| G1 | block 5's input, all 32 positions × 32 directions × 4 c | at j, ‖A − εd‖∞ ≤ 8·u·(‖x_j‖∞ + ε) and ‖S‖∞ ≤ the same (u = float32 eps); off j, A = S = 0 exactly |
| G2 | block 5's output, CPU float64; positions J[0:4], direction R[0], tokens i ≥ j | at c = 0.01, ‖A/ε − Jd‖ ≤ 5 % of ‖Jd‖ at ≥ 90 % of the pooled tokens of the 7 prompts |
| G3 | loss at i = J[0]; d = ∇ₓL_i at token i, normalised | per prompt: A/ε within 10 % of ‖∇ₓL_i‖ at c = 0.01, and rank 1 of 33 on \\|A\\| at every c ≥ 0.03; ≥ 6 of 7 prompts |
| G4a | nudge J[0] by R[0]; every later layer's input and the loss, at every i < j | A = S = 0 exactly |
| G4b | loss at j, every j in J; d = unit ∇ₓL at the same-index position of the next prompt in V1_SCORED (cyclic) | at c = 0.01, over the ranked (prompt, j), the nudged rank-1 set equals the linear response's (rank of \\|∇ₓL_j·d\\| among \\|∇ₓL_j·R_k\\|, no nudge). The rank-1 rate and the old bound 1/33 + 2·sqrt(p₀(1−p₀)/N) are reported at every c, not judged |

**G4b changed after output** (Blocked 43 (a), the user's, 2026-10-14): the frozen rank-1 bound
failed at 143000 (17/224, bound 12) and the gradients alone, with no nudge, give the same 17
(`status-B.md` "B0 (2026-10-14)"), so the bound tested the model's gradient geometry, not the
harness. What G4b can test is that the nudged ranking is the one the linear response predicts.
Its false-fail rate over 18 steps is not derived: a correct harness fails only on a near-tie at
rank 1 inside G3's linear error. It overlaps G2 and G3.

**Floor** (NUMERICS): every batch of one comparison has the same shape and carries a base row
(row 0); the four c's batches are four base passes of one shape, and the floor is their max
pairwise ‖Δ‖ for the readout. A readout with \\|A\\| under 10× the floor is *below floor* and not
ranked. Sequences are truncated at j + 1 tokens wherever only the loss at j is read (causal:
exact for every kept position). The B1 positions and directions are the ones B1 will use.
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

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
FIRST = 143000                       # run first: its record is the populated check
LAYER = 5
D_MODEL = 1024
V1_SCORED = ("camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
             "paper_excerpt", "sullivan_ballou", "wiki_paragraph")   # = tools/run/b_pre_check.py's
MAX_TOKENS = 512                     # Phase 1's truncation (core/models.py extract_activations)
G0_TOL = 1e-5
G1_ULPS = 8.0
G2_C, G2_REL, G2_SHARE, G2_POS = 0.01, 0.05, 0.90, 4
G3_C, G3_REL, G3_PROMPTS = 0.01, 0.10, 6
G4B_C = 0.01
U32 = float(np.finfo(np.float32).eps)


class GateError(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update((REPO / "core" / "nudge.py").read_bytes())
    return h.hexdigest()[:12]


def prompts_hash() -> str:
    from core.config import PROMPTS
    h = hashlib.sha256()
    for k in V1_SCORED:
        h.update(k.encode() + b"\0" + PROMPTS[k].encode() + b"\0")
    return h.hexdigest()[:12]


def token_ids(tok, key: str) -> List[int]:
    from core.config import PROMPTS
    return list(tok(PROMPTS[key], truncation=True, max_length=MAX_TOKENS)["input_ids"])


def load(step: int, device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.lm_loading import resolve_lm_entry
    from core.models import from_pretrained_eager
    e = resolve_lm_entry(f"pythia-410m-step{step}")
    m = from_pretrained_eager(AutoModelForCausalLM, e["repo_id"], revision=e["revision"],
                              torch_dtype=dtype).to(device).eval()
    if next(m.parameters()).dtype != dtype:
        raise GateError(f"step {step}: model is not {dtype}")
    tok = AutoTokenizer.from_pretrained(e["repo_id"], revision=e["revision"])
    return m, tok


def site(st: "nd.Stack", ids: Sequence[int], key: str):
    """Block 5's input (1, n, D), r, B1's positions J and the K random directions R."""
    import torch
    with torch.no_grad():
        x = st.run(st.embed(ids), 0, stop=LAYER, keep={LAYER})["resid"][LAYER]
    r = nd.rms_norm(x[0].double().cpu().numpy())
    J = nd.draw_positions(key, len(ids))
    R = nd.random_directions(D_MODEL, nd.K_RANDOM, nd.seed_for(key, "resid_pre", LAYER))
    return x, r, J, R


def floor_of(bases: Sequence[np.ndarray]) -> float:
    """Max pairwise ‖Δ‖ (last axis) among same-shape base readouts, max over any other axes."""
    f = 0.0
    for a in range(len(bases)):
        for b in range(a + 1, len(bases)):
            d = np.asarray(bases[a], np.float64) - np.asarray(bases[b], np.float64)
            f = max(f, float(np.max(np.abs(d))))
    return f


def loss_arm(st, x, ids, i: int, dirs: np.ndarray, r: float) -> Dict:
    """``dirs[0]`` and the rest nudged at token i, loss at i, every c; sequence cut at i + 1."""
    import torch
    xs = x[:, :i + 1]
    A, bases = {}, []
    with torch.no_grad():
        for c in nd.C_GRID:
            eps = c * r
            delta = st.deltas(dirs, eps, [i], i + 1)
            out = st.run(xs.expand(len(delta), -1, -1), LAYER, delta=delta, loss_ids=ids,
                         loss_at=[i])["loss"][:, 0].double().cpu().numpy()
            b, p, m = nd.split_rows(out)
            A[c] = nd.anti_sym(p, m, b)[0]
            bases.append(np.array([b]))
    fl = floor_of(bases)
    rec = {"floor": fl, "c": {}}
    for c in nd.C_GRID:
        a = np.abs(A[c])
        below = bool(a[0] <= nd.FLOOR_MULT * fl)
        rec["c"][str(c)] = {"A0": float(A[c][0]), "A0_over_eps": float(A[c][0] / (c * r)),
                            "null_median_absA": float(np.median(a[1:])),
                            "rank": None if below else nd.exact_rank(a[0], a[1:]),
                            "below_floor": below}
    rec["turn_on"] = nd.turn_on(nd.C_GRID, [abs(A[c][0]) for c in nd.C_GRID], [fl] * 4)
    rec["turn_on_null_median"] = nd.turn_on(
        nd.C_GRID, [float(np.median(np.abs(A[c][1:]))) for c in nd.C_GRID], [fl] * 4)
    return rec


def grad_at(st, x, ids, i: int) -> np.ndarray:
    """∇ of the loss at i w.r.t. block 5's input at token i (float64 copy)."""
    import torch
    xs = x[:, :i + 1].detach().clone().requires_grad_(True)
    loss = st.run(xs, LAYER, loss_ids=ids, loss_at=[i])["loss"][0, 0]
    (g,) = torch.autograd.grad(loss, xs)
    return g[0, i].double().cpu().numpy()


# ---------------------------------------------------------------- the GPU arms

def g0(st, model, ids) -> float:
    import torch
    with torch.no_grad():
        hf = model(torch.tensor([ids], device=st.device)).logits
        mine = st.logits(st.run(st.embed(ids), 0, keep={st.L})["resid"][st.L])
    return float((mine - hf).abs().max())


def g1(st, x, r, J, R) -> Dict:
    import torch
    worst_on, worst_off, worst_tol_ratio = 0.0, 0.0, 0.0
    n = x.shape[1]
    with torch.no_grad():
        for j in J:
            xj_inf = float(x[0, j].abs().max())
            for c in nd.C_GRID:
                eps = c * r
                got = st.run(x.expand(2 * len(R) + 1, -1, -1), LAYER, stop=LAYER, keep={LAYER},
                             delta=st.deltas(R, eps, [int(j)], n))["resid"][LAYER]
                b, p, m = nd.split_rows(got)
                A, S = nd.anti_sym(p, m, b)
                Ad = A.double().cpu().numpy()
                Sd = S.double().cpu().numpy()
                tol = G1_ULPS * U32 * (xj_inf + eps)
                on = max(float(np.abs(Ad[:, j] - eps * R).max()), float(np.abs(Sd[:, j]).max()))
                off = np.ones(n, bool)
                off[j] = False
                worst_off = max(worst_off, float(np.abs(Ad[:, off]).max()), float(np.abs(Sd[:, off]).max()))
                worst_on = max(worst_on, on)
                worst_tol_ratio = max(worst_tol_ratio, on / tol)
    return {"max_on_err": worst_on, "max_on_err_over_tol": worst_tol_ratio, "max_off": worst_off,
            "pass": worst_tol_ratio <= 1.0 and worst_off == 0.0}


def g4a(st, x, ids, r, J, R) -> Dict:
    import torch
    j, n = int(J[0]), x.shape[1]
    worst_before, least_after = 0.0, np.inf
    keep = set(range(LAYER + 1, st.L + 1))
    with torch.no_grad():
        for c in nd.C_GRID:
            out = st.run(x.expand(3, -1, -1), LAYER, keep=keep, loss_ids=ids,
                         delta=st.deltas(R[:1], c * r, [j], n))
            reads = [out["resid"][l] for l in sorted(keep)] + [out["loss"][..., None]]
            for t in reads:
                b, p, m = nd.split_rows(t)
                A, S = nd.anti_sym(p, m, b)
                worst_before = max(worst_before, float(A[:, :j].abs().max()), float(S[:, :j].abs().max()))
                least_after = min(least_after, float(A[:, j:].abs().max()))
    return {"j": j, "max_before": worst_before, "min_over_reads_max_after": least_after,
            "pass": worst_before == 0.0 and least_after > 0.0}


def run_step(step: int, out: Path) -> Dict:
    import torch
    t0 = time.monotonic()
    model, tok = load(step, "cuda", torch.float32)
    st = nd.Stack(model)
    rec = {"step": step, "code": code_sha(), "prompts_hash": prompts_hash(),
           "numerics": nd.numerics_record(st), "prompts": {}}
    sites = {}
    for key in V1_SCORED:
        ids = token_ids(tok, key)
        x, r, J, R = site(st, ids, key)
        sites[key] = (ids, x, r, J, R)
        rec["prompts"][key] = {"n": len(ids), "r": r, "J": [int(j) for j in J],
                               "G0_max_abs": g0(st, model, ids), "G1": g1(st, x, r, J, R),
                               "G4a": g4a(st, x, ids, r, J, R)}
    # G3: the planted effect, one position per prompt
    for key, (ids, x, r, J, R) in sites.items():
        i = int(J[0])
        g = grad_at(st, x, ids, i)
        arm = loss_arm(st, x, ids, i, np.vstack([nd.unit(g), R]), r)
        arm["i"], arm["grad_norm"] = i, float(np.linalg.norm(g))
        rec["prompts"][key]["G3"] = arm
    # G4b: another prompt's loss gradient, at every position
    grads = {key: [nd.unit(grad_at(st, x, ids, int(j))) for j in J]
             for key, (ids, x, r, J, R) in sites.items()}
    for a, key in enumerate(V1_SCORED):
        ids, x, r, J, R = sites[key]
        other = V1_SCORED[(a + 1) % len(V1_SCORED)]
        rec["prompts"][key]["G4b"] = {"other": other, "arms": [
            dict(loss_arm(st, x, ids, int(j), np.vstack([grads[other][k], R]), r), j=int(j),
                 lin_rank=nd.exact_rank(abs(grads[key][k] @ grads[other][k]), np.abs(R @ grads[key][k])))
            for k, j in enumerate(J)]}
    rec["seconds"] = time.monotonic() - t0
    del model
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- G2, CPU float64

def g2_step(step: int) -> Dict:
    import torch
    torch.set_default_dtype(torch.float64)
    model, tok = load(step, "cpu", torch.float64)
    st = nd.Stack(model)
    rec = {"step": step, "code": code_sha(), "prompts_hash": prompts_hash(),
           "numerics": nd.numerics_record(st), "prompts": {}}
    for key in V1_SCORED:
        ids = token_ids(tok, key)
        x, r, J, R = site(st, ids, key)
        d = torch.as_tensor(R[0], dtype=torch.float64)
        n = x.shape[1]
        per = []
        for j in J[:G2_POS]:
            j = int(j)
            t = torch.zeros_like(x)
            t[0, j] = d
            with torch.no_grad():
                _, Jd = torch.func.jvp(lambda z: st.block(LAYER, z), (x,), (t,))
                base = st.block(LAYER, x)
                row = {"j": j, "c": {}}
                for c in nd.C_GRID:
                    eps = c * r
                    plus = st.block(LAYER, x + eps * t)
                    minus = st.block(LAYER, x - eps * t)
                    A, _ = nd.anti_sym(plus, minus, base)
                    num = (A[0, j:] / eps - Jd[0, j:]).norm(dim=-1)
                    den = Jd[0, j:].norm(dim=-1)
                    ok = den > 0
                    rel = (num[ok] / den[ok]).numpy()
                    row["c"][str(c)] = {"n_tokens": int(ok.sum()), "n_ok": int((rel <= G2_REL).sum()),
                                        "median_rel": float(np.median(rel))}
            per.append(row)
        rec["prompts"][key] = {"n": n, "r": r, "positions": per}
    del model
    return rec


# ---------------------------------------------------------------- the verdict

def verdict_step(gpu: Dict, cpu: Optional[Dict]) -> Dict:
    P = gpu["prompts"]
    v = {"G0": all(p["G0_max_abs"] <= G0_TOL for p in P.values()),
         "G1": all(p["G1"]["pass"] for p in P.values()),
         "G4a": all(p["G4a"]["pass"] for p in P.values())}
    g3_ok = 0
    for p in P.values():
        a = p["G3"]
        lin = abs(a["c"][str(G3_C)]["A0_over_eps"] / a["grad_norm"] - 1) <= G3_REL
        top = all(a["c"][str(c)]["rank"] == 1 for c in nd.C_GRID if c >= 0.03)
        g3_ok += lin and top
    v["G3"], v["G3_prompts"] = g3_ok >= G3_PROMPTS, g3_ok
    arms = [arm for p in P.values() for arm in p["G4b"]["arms"]]
    lin = [(arm["c"][str(G4B_C)]["rank"], arm["lin_rank"]) for arm in arms
           if arm["c"][str(G4B_C)]["rank"] is not None]
    v["G4b_lin"] = {"N": len(lin), "below_floor": len(arms) - len(lin),
                    "rank1_nudged": sum(a == 1 for a, _ in lin), "rank1_linear": sum(b == 1 for _, b in lin),
                    "rank1_mismatch": sum((a == 1) != (b == 1) for a, b in lin),
                    "rank_exact": sum(a == b for a, b in lin), "rank_within_1": sum(abs(a - b) <= 1 for a, b in lin)}
    v["G4b_pass"] = bool(lin) and v["G4b_lin"]["rank1_mismatch"] == 0
    v["G4b"] = {}   # the rank-1 rate against the old bound: reported, not judged
    for c in nd.C_GRID:
        ranks = [arm["c"][str(c)]["rank"] for p in P.values() for arm in p["G4b"]["arms"]]
        ranked = [k for k in ranks if k is not None]
        N, p0 = len(ranked), 1 / (nd.K_RANDOM + 1)
        rate = sum(k == 1 for k in ranked) / N if N else None
        bound = p0 + 2 * np.sqrt(p0 * (1 - p0) / N) if N else None
        v["G4b"][str(c)] = {"N": N, "below_floor": len(ranks) - N, "rate": rate, "bound": bound,
                            "over_bound": bool(N and rate > bound)}
    if cpu is not None:
        n_tok = sum(r["c"][str(G2_C)]["n_tokens"] for p in cpu["prompts"].values() for r in p["positions"])
        n_ok = sum(r["c"][str(G2_C)]["n_ok"] for p in cpu["prompts"].values() for r in p["positions"])
        v["G2_share"] = n_ok / n_tok
        v["G2"] = v["G2_share"] >= G2_SHARE
    v["pass"] = all(v.get(k) is True for k in ("G0", "G1", "G2", "G3", "G4a")) and v["G4b_pass"]
    return v


# ---------------------------------------------------------------- CLI

def _steps(a) -> List[int]:
    steps = list(a.steps or STEPS)
    return sorted(steps, key=lambda s: s != FIRST)


def cmd_run(a) -> int:
    nd.set_numerics()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (B0's default is GPU float32)")
    a.out.mkdir(parents=True, exist_ok=True)
    for step in _steps(a):
        path = a.out / f"gpu_step{step}.json"
        if path.exists():
            continue
        rec = run_step(step, a.out)
        P = rec["prompts"]
        if not all(np.isfinite(p["G3"]["grad_norm"]) and p["G3"]["grad_norm"] > 0 for p in P.values()):
            raise GateError(f"step {step}: a zero or non-finite planted gradient; record not written")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.rename(path)
        v = verdict_step(rec, None)
        print(f"done step {step} in {rec['seconds']:.0f}s: G0 {v['G0']} G1 {v['G1']} G3 {v['G3_prompts']}/7 "
              f"G4a {v['G4a']} G4b {v['G4b_pass']} (rank-1 nudged/linear/mismatch "
              f"{v['G4b_lin']['rank1_nudged']}/{v['G4b_lin']['rank1_linear']}/{v['G4b_lin']['rank1_mismatch']}) "
              f"rate " + " ".join(f"{c}:{x['rate']:.3f}/{x['bound']:.3f}"
                                               for c, x in v["G4b"].items() if x["N"]), flush=True)
        if a.first_only:
            return 0
    return 0


def cmd_g2(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    for step in _steps(a):
        path = a.out / f"cpu_step{step}.json"
        if path.exists():
            continue
        t0 = time.monotonic()
        rec = g2_step(step)
        rec["seconds"] = time.monotonic() - t0
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.rename(path)
        share = [r["c"][str(G2_C)] for p in rec["prompts"].values() for r in p["positions"]]
        print(f"done G2 step {step} in {rec['seconds']:.0f}s: "
              f"{sum(s['n_ok'] for s in share)}/{sum(s['n_tokens'] for s in share)} tokens within "
              f"{G2_REL:.0%} at c={G2_C}", flush=True)
        if a.first_only:
            return 0
    return 0


def cmd_report(a) -> int:
    rows, ok = {}, True
    for step in STEPS:
        g, c = a.out / f"gpu_step{step}.json", a.out / f"cpu_step{step}.json"
        if not g.exists() or not c.exists():
            print(f"step {step}: missing {'gpu' if not g.exists() else 'cpu'} record")
            ok = False
            continue
        gpu, cpu = json.loads(g.read_text()), json.loads(c.read_text())
        if gpu["prompts_hash"] != prompts_hash() or cpu["prompts_hash"] != prompts_hash():
            raise GateError(f"step {step}: a record ran on another prompt set")
        if gpu["code"] != code_sha() or cpu["code"] != code_sha():
            raise GateError(f"step {step}: a record ran on other code ({gpu['code']}, {cpu['code']}; "
                            f"now {code_sha()}): re-run it")
        rows[step] = verdict_step(gpu, cpu)
        ok &= rows[step]["pass"]
    summary = {"code": code_sha(), "prompts_hash": prompts_hash(), "complete": len(rows) == len(STEPS),
               "pass": bool(ok and len(rows) == len(STEPS)), "steps": rows}
    (a.out / "gate.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("| step | G0 | G1 | G2 share | G3 | G4a | G4b (rank-1 nudged / linear / mismatch; exact, ±1 of N) "
          "| rank-1 rate (old bound, not judged) at c = " + ", ".join(map(str, nd.C_GRID)) + " | pass |")
    print("|---|---|---|---|---|---|---|---|---|")
    for step, v in rows.items():
        L = v["G4b_lin"]
        g4b = ", ".join("—" if not x["N"] else f"{x['rate']:.3f} ({x['bound']:.3f})"
                        for x in v["G4b"].values())
        print(f"| {step} | {v['G0']} | {v['G1']} | {v['G2_share']:.3f} | {v['G3_prompts']}/7 | {v['G4a']} | "
              f"{v['G4b_pass']} ({L['rank1_nudged']} / {L['rank1_linear']} / {L['rank1_mismatch']}; "
              f"{L['rank_exact']}, {L['rank_within_1']} of {L['N']}) | {g4b} | {v['pass']} |")
    print(f"gate: {'PASS' if summary['pass'] else 'FAIL'}" + ("" if summary["complete"] else " (incomplete)"))
    return 0 if summary["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "g2", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name != "report":
            p.add_argument("--steps", type=int, nargs="*")
            p.add_argument("--first-only", action="store_true")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "g2": cmd_g2, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
