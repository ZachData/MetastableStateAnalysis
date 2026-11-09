"""C7 (i): Isobe et al.'s three phases (2605.07772) on Pythia-410m's stored seed-0 runs.

`trackC_writeup/design-C7.md` (frozen on Blocked 154, all (a)). Three readings on the 18 steps x 7
v1 prompts, step 0 the no-train control; tier 1, description:

* **P**, the depth profile by the paper's fixed three-phase criterion (Appendix B.3), not retuned,
  on F = -E1 and F = -|m|^2 over states 0-23 (states 1-23 beside; the post-final-LN state 24 out);
* **F**, each block's FFN and attention signed first-order share of the change in |m|^2 (unit rows),
  exact inputs because Pythia's blocks are parallel-residual; the exact two-part Shapley split and
  the unsigned tangential speeds beside;
* **Q**, each layer's interaction form S = sum_h sym(W_Q^T W_K)/sqrt(64): its positive share
  (Assumption 3.1 holds iff 1) and gap 1 - l2/l1; per-head medians beside.

Gate (`gate`): G1 the criterion on planted profiles, G1e the embedding's blind spot, G1n noise
(reported), G2 F's split on a planted cloud. Run (`run`): refuses unless the gate passed with this
code, `c2_grid.check_a5` passes (k1), the hooked pass reproduces the stored states and the MLP
outputs (k2), and `b2.sym_eigs` matches a dense eigvalsh (k3).

    python -m trackC_writeup.c7_profile gate --out <dir>
    python -m trackC_writeup.c7_profile run --out <dir>
    python -m trackC_writeup.c7_profile report --out <dir>
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

from trackA_force.a1a import DATA, REPO, STEPS, V1, run_dir

T = 23                       # placed: the profile's last state, the input to block 23 (design "The inputs")
LO, HI = 0.4, 0.6            # placed: the paper's interior window [0.4T, 0.6T] (2605.07772 B.3)
INIT_MIN = 0.1               # placed: the paper's B.3 criterion, initial part
SD_MAX = 0.05                # placed: the paper's B.3 criterion, plateau part
TERM_MIN = 0.1               # placed: the paper's B.3 criterion, terminal part
ESC_FRAC = 0.1               # placed: the paper's B.3 t_esc (10 % of the plateau-to-terminal rise)
FIT_FROM = 0.7               # placed: the paper's B.3 fit window [0.7T, T]
TERMINAL_BLOCKS = (20, 21, 22)   # placed: F's label blocks (design "F"), fixed before output
STORED_TOL = 1e-4            # placed: k2, as 1e U2's v1 bound (stored on the CPU)
EIG_TOL = 1e-8               # placed: k3
G2_TOL = 1e-2                # placed: G2, first order at a 1e-3 plant
HEAD_DIM = 64
K2_BLOCKS = (2, 12, 22)      # placed: design k2
K2_STEPS = (512, 143000)     # placed: design k2
K2_PROMPT = "wiki_paragraph"


class Refused(SystemExit):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c7_profile.py", "trackB_circuit/b2.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- P: the criterion

def criterion(F: Sequence[float], t0: int = 0) -> Dict:
    """2605.07772 Appendix B.3 on F(s), s = t0..T (low = clustered, the paper's sign).

    The time axis is the state index; the window [0.4T', 0.6T'] and [0.5T', T'] and [0.7T', T'] are
    on t = s - t0 with T' = T - t0, so states 1-23 is the same rule on a horizon of 22."""
    F = np.asarray(F, dtype=float)
    Tp = len(F) - 1
    t = np.arange(len(F), dtype=float)
    R = float(F.max() - F.min())
    win = (t >= LO * Tp) & (t <= HI * Tp)
    Fpl = float(np.median(F[win]))
    out = {"t0": t0, "T": t0 + Tp, "R": R, "F_pl": Fpl, "window": [int(t0 + t[win][0]), int(t0 + t[win][-1])],
           "argmax": int(t0 + F.argmax()), "argmin": int(t0 + F.argmin())}
    if R <= 0:
        out.update(initial=False, plateau=False, terminal=False, full=False, init_ratio=None,
                   sd_ratio=None, term_ratio=None, t_esc=None, a=None)
        return out
    ir, sr, tr = (F[0] - Fpl) / R, float(np.std(F[win], ddof=1)) / R, (F[-1] - Fpl) / R
    out.update(init_ratio=float(ir), sd_ratio=sr, term_ratio=float(tr),
               initial=bool(ir >= INIT_MIN), plateau=bool(sr <= SD_MAX), terminal=bool(tr >= TERM_MIN))
    out["full"] = out["initial"] and out["plateau"] and out["terminal"]
    out["t_esc"], out["a"] = None, None
    if F[-1] > Fpl:
        lift = F[-1] - Fpl
        late = np.where((t >= 0.5 * Tp) & (F - Fpl >= ESC_FRAC * lift))[0]
        out["t_esc"] = int(t0 + late[0]) if late.size else None
        fit = (t >= FIT_FROM * Tp) & (F - Fpl > 0)
        if fit.sum() >= 3:
            slope = np.polyfit(t[fit], np.log(F[fit] - Fpl), 1)[0]
            out["a"] = float(slope)       # E - E_pl ~ C exp(-a (T - t)) => log slope = a
    return out


def template(a: float, n: int = T + 1) -> np.ndarray:
    s = np.arange(n, dtype=float)
    return np.exp(-a * s) + np.exp(-a * (n - 1 - s))


def plants() -> Dict[str, np.ndarray]:
    s = np.arange(T + 1, dtype=float)
    mid = np.where(s <= 10, 1 - s / 10, np.where(s <= 13, (s - 10) / 3, 1.0))   # falls to 10, rises 10-13
    emb = np.zeros(T + 1)
    emb[0] = 1.0
    emb[-3:] = [0.0, 0.025, 0.05]                                                  # terminal rise 5 % of R
    out = {f"template_a{a}": template(a) for a in (0.3, 0.6, 1.0)}
    out.update(linear=1 - s / T, logistic=1 / (1 + np.exp(s - 12)), midstack=mid, embedding=emb)
    return out


def gate_g1() -> Dict:
    P = plants()
    res = {k: criterion(v) for k, v in P.items()}
    ok = (all(res[f"template_a{a}"]["full"] for a in (0.3, 0.6, 1.0))
          and not res["linear"]["full"] and not res["logistic"]["full"]
          and not res["midstack"]["plateau"])
    emb = {"states_0_23": res["embedding"], "states_1_23": criterion(P["embedding"][1:], t0=1)}
    rng = np.random.default_rng(0)
    noise = {}
    base = template(0.6)
    Rb = base.max() - base.min()
    for sd in (0.01, 0.02):
        r = [criterion(base + rng.normal(0, sd * Rb, base.size)) for _ in range(200)]
        noise[str(sd)] = {k: sum(x[k] for x in r) / 200 for k in ("initial", "plateau", "terminal", "full")}
    return {"pass": bool(ok), "plants": res, "G1e": emb, "G1n": noise}


# ---------------------------------------------------------------- F: the split

def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def m2(X: np.ndarray) -> float:
    m = unit(X).mean(0)
    return float(m @ m)


def tangential(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """P_perp_{x_hat}(v) / |x| per row: a unit token's first-order motion."""
    nx = np.linalg.norm(X, axis=1, keepdims=True)
    Xh = X / nx
    return (V - (V * Xh).sum(1, keepdims=True) * Xh) / nx


def split(X: np.ndarray, Xn: Optional[np.ndarray], U: np.ndarray, pos0: bool = True) -> Dict:
    """Block split of the change in |m|^2 (all rows), and the unsigned speeds (row 0 out if pos0)."""
    X, U = np.asarray(X, float), np.asarray(U, float)
    m = unit(X).mean(0)
    tu = tangential(X, U)
    sl = slice(1, None) if pos0 else slice(None)
    out = {"d_u": float(2 * m @ tu.mean(0)), "speed_u": float(np.median(np.linalg.norm(tu[sl], axis=1)))}
    if Xn is None:
        return out
    Xn = np.asarray(Xn, float)
    A = Xn - X - U
    ta = tangential(X, A)
    f0, f1 = m2(X), m2(Xn)
    fu, fa = m2(X + U), m2(X + A)
    out.update(d_a=float(2 * m @ ta.mean(0)), delta=f1 - f0,
               shap_u=0.5 * ((fu - f0) + (f1 - fa)), shap_a=0.5 * ((fa - f0) + (f1 - fu)),
               speed_a=float(np.median(np.linalg.norm(ta[sl], axis=1))))
    out["remainder"] = out["delta"] - out["d_u"] - out["d_a"]
    out["rho"] = out["speed_u"] / out["speed_a"] if out["speed_a"] > 0 else None
    return out


def f_label(blocks: Dict[int, Dict]) -> Dict:
    """design F: on each of blocks 20-22, d_u < 0 and |d_u| > |d_a|; the Shapley form beside."""
    def holds(k_u, k_a):
        return all(blocks[b][k_u] < 0 and abs(blocks[b][k_u]) > abs(blocks[b][k_a]) for b in TERMINAL_BLOCKS)
    return {"first_order": bool(holds("d_u", "d_a")), "shapley": bool(holds("shap_u", "shap_a"))}


def gate_g2() -> Dict:
    rng = np.random.default_rng(1)
    n, d = 200, 64
    c = np.zeros(d)
    c[0] = 1.0
    X = (c + 0.5 * rng.normal(size=(n, d))) * rng.uniform(5, 10, (n, 1))
    m = unit(X).mean(0)
    toward = tangential(X, np.tile(m, (n, 1))) * np.linalg.norm(X, axis=1, keepdims=True) ** 2
    toward *= 1e-3 / np.linalg.norm(toward, axis=1, keepdims=True) * np.linalg.norm(X, axis=1, keepdims=True)
    out = {}
    for name, (U, A) in {"ffn_out": (-toward, 0.3 * toward), "ffn_in": (toward, -0.3 * toward)}.items():
        r = split(X, X + U + A, U)
        rel = abs(r["d_u"] + r["d_a"] - r["delta"]) / abs(r["delta"])
        sign_ok = (r["d_u"] < 0 < r["d_a"]) if name == "ffn_out" else (r["d_u"] > 0 > r["d_a"])
        out[name] = {**r, "rel_first_order": rel, "signs": bool(sign_ok),
                     "shapley_sum": abs(r["shap_u"] + r["shap_a"] - r["delta"])}
    out["pass"] = all(v["signs"] and v["rel_first_order"] <= G2_TOL and v["shapley_sum"] < 1e-12
                      for v in out.values() if isinstance(v, dict))
    return out


# ---------------------------------------------------------------- Q: the form

def layer_form(heads: Sequence[Dict]) -> Dict:
    """S = sum_h sym(W_Q^T W_K)/sqrt(64), dense; its positive share and gap; per-head medians."""
    from trackB_circuit import b2
    S = sum(h["Wq"].T @ h["Wk"] for h in heads) / np.sqrt(HEAD_DIM)
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))[::-1]
    gap = float(1 - ev[1] / ev[0]) if ev[0] > 0 else None
    hs = [b2.sym_eigs(*b2.m_factors(h))[::-1] for h in heads]
    hg = [float(1 - e[1] / e[0]) for e in hs if e[0] > 0]
    return {"share": float(ev[ev > 0].sum() / np.abs(ev).sum()), "gap": gap, "l1": float(ev[0]),
            "l2": float(ev[1]), "l_min": float(ev[-1]),
            "head_share_median": float(np.median([b2.share(e) for e in hs])),
            "head_gap_median": float(np.median(hg)) if hg else None}


def check_k3(heads: Sequence[Dict]) -> float:
    from trackB_circuit import b2
    worst = 0.0
    for h in heads[:3]:
        A, B = b2.m_factors(h)
        M = A @ B.T
        dense = np.linalg.eigvalsh(0.5 * (M + M.T))
        fac = np.sort(b2.sym_eigs(A, B))
        top = np.sort(dense[np.argsort(-np.abs(dense))[: fac.size]])   # the rest are zeros
        worst = max(worst, float(np.max(np.abs(top - fac)) / np.abs(fac).max()))
    return worst


# ---------------------------------------------------------------- the run (deps)

def states(step: int, prompt: str) -> np.ndarray:
    z = np.load(run_dir(step, prompt) / "activations.npz")
    return z["activations"].astype(np.float64) * z["norms"].astype(np.float64)[:, :, None]


def energies(step: int, prompt: str) -> np.ndarray:
    d = json.loads((run_dir(step, prompt) / "energies.json").read_text())["layers"]
    return np.array([l["energies"]["1.0"] for l in d], dtype=float)


def mlp_updates(model, X: np.ndarray) -> List[np.ndarray]:
    """u_s = mlp_s(LN2(x_s)) for s = 0..23, float32 as the stored pass."""
    import torch
    out = []
    with torch.no_grad():
        for s, blk in enumerate(model.gpt_neox.layers):
            x = torch.from_numpy(X[s].astype(np.float32))
            out.append(blk.mlp(blk.post_attention_layernorm(x)).double().numpy())
    return out


def check_k2(model, tok, step: int) -> Dict:
    """The hooked pass on the stored tokens reproduces the stored states (rel), and mlp(LN2(stored x))
    reproduces the hooked MLP outputs (rel), at K2_BLOCKS."""
    import torch
    from p1e_energy_field.u2_attn import token_ids
    rd = run_dir(step, K2_PROMPT)
    ids = torch.tensor([token_ids(tok, rd)])
    got = {}
    hooks = [model.gpt_neox.layers[b].mlp.register_forward_hook(
        lambda mod, inp, out, b=b: got.__setitem__(b, out[0].double().numpy())) for b in K2_BLOCKS]
    try:
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
    finally:
        for h in hooks:
            h.remove()
    X = states(step, K2_PROMPT)
    rel_state = max(float(np.linalg.norm(hs[s][0].double().numpy() - X[s]) / np.linalg.norm(X[s])) for s in range(24))
    U = mlp_updates(model, X)
    rel_mlp = max(float(np.linalg.norm(U[b] - got[b]) / np.linalg.norm(got[b])) for b in K2_BLOCKS)
    return {"step": step, "rel_state": rel_state, "rel_mlp": rel_mlp,
            "pass": rel_state <= STORED_TOL and rel_mlp <= STORED_TOL}


def read_step(model, step: int) -> Dict:
    from trackB_circuit import b2
    rec = {"step": step, "prompts": {}, "Q": []}
    Fsum = {}
    for p in V1:
        X = states(step, p)
        U = mlp_updates(model, X)
        e1 = energies(step, p)[: T + 1]
        mm = np.array([m2(X[s]) for s in range(T + 1)])
        blocks = {s: split(X[s], X[s + 1] if s < T else None, U[s]) for s in range(T + 1)}
        rec["prompts"][p] = {
            "E1": e1.tolist(), "m2": mm.tolist(),
            "P": {"E1": criterion(-e1), "m2": criterion(-mm),
                  "E1_1_23": criterion(-e1[1:], t0=1), "m2_1_23": criterion(-mm[1:], t0=1)},
            "F": {str(s): v for s, v in blocks.items()}, "F_label": f_label(blocks)}
        for s, v in blocks.items():
            Fsum.setdefault(s, []).append(v)
    e1m = np.mean([rec["prompts"][p]["E1"] for p in V1], 0)
    mmm = np.mean([rec["prompts"][p]["m2"] for p in V1], 0)
    rec["mean"] = {"P": {"E1": criterion(-e1m), "m2": criterion(-mmm),
                         "E1_1_23": criterion(-e1m[1:], t0=1), "m2_1_23": criterion(-mmm[1:], t0=1)}}
    Fm = {s: {k: float(np.mean([v[k] for v in vs])) for k in vs[0] if vs[0][k] is not None}
          for s, vs in Fsum.items()}
    rec["mean"]["F"] = {str(s): v for s, v in Fm.items()}
    rec["mean"]["F_label"] = f_label(Fm)
    for l in range(24):
        rec["Q"].append(layer_form(b2.layer_weights(model, l)))
    return rec


def check_populated(rec: Dict) -> None:
    p = rec["prompts"][V1[0]]
    if not np.isfinite(p["E1"]).all() or p["P"]["E1"]["R"] <= 0 or not rec["Q"] or rec["Q"][12]["l1"] == 0:
        raise Refused(f"refusing: step {rec['step']}'s first record is not populated")


def cmd_gate(a) -> int:
    out = {"code": code_sha(), "G1": gate_g1(), "G2": gate_g2()}
    out["pass"] = out["G1"]["pass"] and out["G2"]["pass"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pass": out["pass"], "G1": out["G1"]["pass"], "G2": out["G2"]["pass"],
                      "G1e": {k: v["terminal"] for k, v in out["G1"]["G1e"].items()},
                      "G1n": out["G1"]["G1n"]}, indent=1))
    return 0 if out["pass"] else 1


def cmd_run(a) -> int:
    import torch
    from trackB_circuit.b0_gate import load
    from trackC_writeup.c2_grid import check_a5
    torch.backends.cuda.matmul.allow_tf32 = False
    code = code_sha()
    g = json.loads((a.out / "gate.json").read_text())
    if not g["pass"] or g["code"] != code:
        raise Refused(f"refusing: gate.json is {g['code']} pass={g['pass']}, this code is {code}")
    k1 = check_a5()
    checks = {"k1": {"pass": k1["pass"], "energy_max_rel_diff": k1["energy_max_rel_diff"]}, "k2": [], "k3": None}
    (a.out / "records").mkdir(parents=True, exist_ok=True)
    order = list(K2_STEPS) + [s for s in STEPS if s not in K2_STEPS]
    for i, step in enumerate(order):
        path = a.out / "records" / f"step{step}.json"
        if path.exists() and json.loads(path.read_text())["code"] == code:
            print(f"have step {step}", flush=True)
            continue
        t0 = time.monotonic()
        model, tok = load(step, "cpu", torch.float32)
        if step in K2_STEPS:
            k2 = check_k2(model, tok, step)
            checks["k2"].append(k2)
            if not k2["pass"]:
                raise Refused(f"refusing: k2 at step {step}: {k2}")
        if checks["k3"] is None:
            from trackB_circuit import b2
            checks["k3"] = check_k3(b2.layer_weights(model, 12))
            if checks["k3"] > EIG_TOL:
                raise Refused(f"refusing: k3 {checks['k3']}")
        rec = read_step(model, step)
        rec.update(code=code, wall_s=round(time.monotonic() - t0, 1))
        if i == 0:
            check_populated(rec)
        path.write_text(json.dumps(rec) + "\n")
        (a.out / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
        print(f"step {step} {rec['wall_s']} s", flush=True)
        del model
    return 0


def summarise(out: Path) -> Dict:
    rows = []
    for step in STEPS:
        r = json.loads((out / "records" / f"step{step}.json").read_text())
        row = {"step": step}
        for stat in ("E1", "m2", "E1_1_23", "m2_1_23"):
            ps = [r["prompts"][p]["P"][stat] for p in V1]
            row[stat] = {k: sum(x[k] for x in ps) for k in ("initial", "plateau", "terminal", "full")}
            row[stat]["mean"] = {k: r["mean"]["P"][stat][k] for k in
                                 ("initial", "plateau", "terminal", "full", "argmax", "argmin", "t_esc", "a",
                                  "init_ratio", "sd_ratio", "term_ratio")}
        row["F_label"] = {k: sum(r["prompts"][p]["F_label"][k] for p in V1) for k in ("first_order", "shapley")}
        row["F_label_mean"] = r["mean"]["F_label"]
        row["F_terminal"] = {b: {k: r["mean"]["F"][str(b)].get(k) for k in
                                 ("d_u", "d_a", "shap_u", "shap_a", "delta", "remainder", "rho")}
                             for b in TERMINAL_BLOCKS}
        Q = r["Q"]
        row["Q"] = {"share_min": min(q["share"] for q in Q), "share_max": max(q["share"] for q in Q),
                    "gap_interior_median": float(np.median([Q[l]["gap"] for l in range(10, 14) if Q[l]["gap"] is not None])),
                    "head_share_median": float(np.median([q["head_share_median"] for q in Q]))}
        rows.append(row)
    return {"rows": rows}


def cmd_report(a) -> int:
    s = summarise(a.out)
    (a.out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    for r in s["rows"]:
        e, m = r["E1"], r["m2"]
        print(f"{r['step']:>6}  E1 i/p/t/full {e['initial']}/{e['plateau']}/{e['terminal']}/{e['full']}"
              f"  m2 {m['initial']}/{m['plateau']}/{m['terminal']}/{m['full']}"
              f"  E1[1-23] full {r['E1_1_23']['full']}  F {r['F_label']['first_order']}/{r['F_label']['shapley']}"
              f"  Q share {r['Q']['share_min']:.2f}-{r['Q']['share_max']:.2f} gap {r['Q']['gap_interior_median']:.3f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=DATA / "trackC" / "c7" / "2026-11-09-u1")
    a = ap.parse_args(argv)
    return {"gate": cmd_gate, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
