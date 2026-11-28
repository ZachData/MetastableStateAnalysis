"""
trackC_writeup/c8_sharp.py — C8 U2, candidate (ii): sharpness against Adam's stability threshold
(`design-C8.md` "U2"; the rule and its constants are `c8_sharp_rules`, pushed before any λ).

The order is enforced, so nothing is computed before what must precede it:

1. ``gate`` — G-quad (planted quadratics of known spectrum through the same Lanczos iteration and
   preconditioning code), G-hvp (on the released pythia-70m step143000, not a C8 arm: the HVP
   against a central difference of gradients, and its symmetry) and the timing that placed
   ``BATCH_ROWS``. No arm state is loaded.
   Planted in G-quad and the masking of v = 0 coordinates are the rules' (`c8_sharp_rules`).
2. ``records`` — C4's records (`c4_read._records_of`, unchanged) on each re-run's saved states.
3. ``reference`` — rule T (`c4pdf_rules.transient`) on each re-run's own records at every gated τ:
   the premise and the windows. **No λ exists yet**; ``measure`` refuses without this file.
4. ``measure`` — λ_P on every re-run state (``--what pre``), raw λ (``--what raw``) on the states
   `raw_states` lists (the originals only, as the design names them). Refuses until the gate passed, the reference exists and the user has
   registered or declined U2's entry (D4 (a); `c8_entries.json`, committed).
5. ``read`` — R_ii at each gated τ, joint, and the beside rows.

Each output names its inputs (sha256 of the gate, reference, records, moments files).

    METS_DATA=<main>/data python -m trackC_writeup.c8_sharp {gate,records,reference,measure,read} --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c8_rules as r8
from trackC_writeup import c8_sharp_rules as sr
from trackC_writeup import c8_train as t8

ENTRIES = REPO / "trackC_writeup/c8_entries.json"       # the user's, committed: {"ii": id | "declined"}
REGISTRY = REPO / "claims/registry.json"
HDD = Path("/run/media/system/HDD_1TB/mets_data")
U1_REF = Path("trackC/c8/2026-11-27-u1/reference.json")  # under DATA
RAW_ROOTS = {"F1": HDD / "c4pdf", "Fh": HDD / "c4pdf", "Ft": HDD / "c4pdf", "BP": HDD / "c4pdf",
             "BO": HDD / "c4pdf", "BK": HDD / "c4pdf", "A0": HDD / "c4"}
GATE_MODEL = ("EleutherAI/pythia-70m",)
GATE_REVS = ("step128", "step256", "step512")      # released, the arms' regime
GATE_BESIDE = "step143000"
GATE_ROWS, GATE_CTX = 2, 512
FD_STEP = 1e-7
HVP_TOL = {"fd": 1e-3, "prec": 1e-3, "sym": 1e-3}


class C8SharpRefused(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackC_writeup/c8_sharp.py").read_bytes()).hexdigest()[:12]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dump(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(p)


# ---------------------------------------------------------------- the operator and the iteration

def power(op: Callable, like: Sequence, seed: int = sr.SEED, n_min: int = sr.N_MIN, n_max: int = sr.N_MAX,
          tol: float = sr.CONV_TOL) -> dict:
    """Top algebraic eigenvalue of the symmetric ``op`` (a list of tensors shaped as ``like``), by
    Lanczos from a fixed-seed Gaussian (the same H·v products power iteration uses; the top Ritz
    value, so no shift is needed for a dominant negative eigenvalue). Three vectors are kept (no
    full reorthogonalisation: lost orthogonality duplicates Ritz values, it does not move the top
    one). Converged when the top Ritz value changes by ≤ tol relatively twice running after
    ``n_min`` steps; ``resid`` is the Ritz bound |β_j s_j| (exact arithmetic)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    v = _normed([torch.randn(t.shape, generator=g, dtype=torch.float32).to(t.device) for t in like])
    v_prev, beta, al, be, th, ok_run, conv, resid = None, 0.0, [], [], [], 0, False, None
    for j in range(1, n_max + 1):
        w = op(v)
        alpha = float(sum((a * b).sum() for a, b in zip(w, v)))
        w = [a - alpha * b for a, b in zip(w, v)]
        if v_prev is not None:
            w = [a - beta * b for a, b in zip(w, v_prev)]
        al.append(alpha)
        bnew = float(sum((a * a).sum() for a in w)) ** 0.5
        T = np.diag(al) + np.diag(be, 1) + np.diag(be, -1)
        ev, evec = np.linalg.eigh(T)
        th.append(float(ev[-1]))
        resid = abs(bnew * evec[-1, -1])
        if len(th) > 1 and abs(th[-1] - th[-2]) <= tol * abs(th[-1]):
            ok_run += 1
        else:
            ok_run = 0
        if (j >= n_min and ok_run >= 2) or bnew <= 1e-12 * max(abs(th[-1]), 1e-30):
            conv = True
            break
        v_prev, v, beta = v, [a / bnew for a in w], bnew
        be.append(bnew)
    return {"mu": th[-1], "iters": len(th), "converged": conv, "resid": resid,
            "last_rel": abs(th[-1] - th[-2]) / max(abs(th[-1]), 1e-30) if len(th) > 1 else None,
            "bottom_ritz": float(ev[0])}


def _normed(v):
    n = float(sum((t * t).sum() for t in v)) ** 0.5
    return [t / n for t in v]


def hvp_fn(model, x, mb: int = 1) -> Tuple[Callable, List]:
    """H·v for the mean next-token loss over the rows of ``x`` (float32, no autocast), accumulated
    over micro-batches of ``mb`` rows (the loss is a mean of equal-length rows, so H is too)."""
    import torch
    import torch.nn.functional as F
    params = [p for _, p in model.named_parameters()]
    n = x.shape[0]

    def op(v):
        out = [torch.zeros_like(p) for p in params]
        for k in range(0, n, mb):
            r = x[k:k + mb]
            logits = model(input_ids=r[:, :-1]).logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), r[:, 1:].reshape(-1))
            g = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((a * b).sum() for a, b in zip(g, v))
            hv = torch.autograd.grad(dot, params)
            for o, h in zip(out, hv):
                o.add_(h, alpha=r.shape[0] / n)
            del logits, loss, g, dot, hv
        return out
    return op, params


def precond_d(model, v_file: Path) -> Tuple[List, int, Dict[str, int]]:
    """D = P^{-1/2} per parameter, P = (1 − β₁^k)(√(v / (1 − β₂^k)) + ε) (`c8_sharp_rules`)."""
    import torch
    from safetensors import safe_open
    with safe_open(str(v_file), "pt") as f:
        k = int(f.metadata()["opt_step"])
        vs = {n: f.get_tensor(n) for n in f.keys()}
    names = [n for n, _ in model.named_parameters()]
    if set(names) != set(vs):
        raise C8SharpRefused(f"{v_file}: parameter names differ from the model's")
    if k < 1:
        raise C8SharpRefused(f"{v_file}: optimiser step {k}")
    b1, b2 = 1 - sr.BETA1 ** k, 1 - sr.BETA2 ** k
    dev = next(model.parameters()).device
    d, masked = [], {}
    for n in names:
        v = vs[n].to(dev)
        dn = (b1 * (torch.sqrt(v / b2) + sr.EPS)).rsqrt()
        z = v == 0
        if bool(z.any()):
            dn = dn.masked_fill(z, 0.0)                 # never given a gradient: excluded (c8_sharp_rules)
            masked[n] = int(z.sum())
        d.append(dn)
    return d, k, masked


def precond_op(op: Callable, d: List) -> Callable:
    return lambda v: [a * b for a, b in zip(op([x * y for x, y in zip(v, d)]), d)]


def held_batch(device, rows_n: int = sr.BATCH_ROWS):
    import torch
    from trackC_writeup import c4_train as ct
    rows = ct.open_rows(ct.ROOT)
    a = ct.HELD_OUT[0]
    x = np.asarray(rows[a:a + rows_n])[:, :sr.CTX + 1].astype(np.int64)
    return torch.from_numpy(x).to(device), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


# ---------------------------------------------------------------- 1. the gate

def _quad_case(n: int, eig: np.ndarray, dvec: np.ndarray, seed: int) -> dict:
    """A planted quadratic ½ θᵀAθ (A = Q diag(eig) Qᵀ) through the same `power` and `precond_op`."""
    import torch
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (q * eig) @ q.T
    At = torch.tensor(A, dtype=torch.float32, device="cuda")
    like = [torch.zeros(n // 2, device="cuda"), torch.zeros(n - n // 2, device="cuda")]

    def op(v):
        w = At @ torch.cat(v)
        return [w[:n // 2], w[n // 2:]]
    dt = torch.tensor(dvec, dtype=torch.float32, device="cuda")
    d = [dt[:n // 2], dt[n // 2:]]
    exact = float(np.max(np.linalg.eigvalsh((dvec[:, None] * A) * dvec[None, :])))
    raw_exact = float(np.max(eig))
    got_p, got_r = power(precond_op(op, d), like), power(op, like)
    return {"exact_pre": exact, "got_pre": got_p, "exact_raw": raw_exact, "got_raw": got_r,
            "rel_err_pre": abs(got_p["mu"] - exact) / abs(exact), "rel_err_raw": abs(got_r["mu"] - raw_exact) / abs(raw_exact)}


def g_quad() -> dict:
    n = 400
    rng = np.random.default_rng(1)
    cases = {
        # a clear top at the threshold's scale, with a gap ratio 0.9
        "gap0.9": (np.concatenate([[3.8e4 * 0.05], 3.8e4 * 0.05 * 0.9 * rng.random(n - 1)]), 10 ** rng.uniform(0, 1.5, n)),
        # a dominant negative eigenvalue (the iteration must shift)
        "negative": (np.concatenate([[-5.0, 2.0], rng.uniform(-1, 1.5, n - 2)]), np.ones(n)),
        # a heavy-tailed spectrum like a network's, preconditioner over 4 decades
        "bulk": (np.concatenate([[1.0], 0.3 * rng.random(20), 0.01 * rng.standard_normal(n - 21)]), 10 ** rng.uniform(0, 4, n)),
    }
    res = {k: _quad_case(n, e, dv, seed=i) for i, (k, (e, dv)) in enumerate(cases.items())}
    tol = 0.02
    ok = all(v["rel_err_pre"] <= tol and v["rel_err_raw"] <= tol for v in res.values())
    return {"cases": res, "tol": tol, "pass": bool(ok)}


def _released(device, rev: str):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    return from_pretrained_eager(GPTNeoXForCausalLM, GATE_MODEL[0], revision=rev,
                                 torch_dtype=torch.float32).to(device).eval()


def g_hvp(device) -> dict:
    """On released pythia-70m checkpoints (Pythia's own run, not a C8 arm), on ``GATE_ROWS`` rows of
    ``GATE_CTX`` predictions (fp64 at the measured context does not fit):

    - precision: the fp32 Rayleigh quotient uᵀHu against fp64's, along a high-curvature direction
      (u = H³r, normed), at each of ``GATE_REVS`` (the arms' regime, ≤ 512 updates); step143000
      beside (fp32 is off there by ~1e-2: the instrument is not claimed at the end of training);
    - correctness: fp64 H·u against an fp64 central difference of gradients at ``FD_STEP``, on
      ``GATE_REVS[1]``; symmetry: |wᵀHu − uᵀHw| / ‖Hu‖ in fp32 there;
    - the time of one H·v at ``BATCH_ROWS`` × ``CTX`` and the peak memory (what placed them)."""
    import copy
    import torch
    import torch.nn.functional as F
    xf, xh = held_batch(device, GATE_ROWS)
    x = xf[:, :GATE_CTX + 1]
    nrm = lambda t: float(sum((a.double() ** 2).sum() for a in t)) ** 0.5
    dot = lambda p_, q_: float(sum((a.double() * b.double()).sum() for a, b in zip(p_, q_)))
    out = {"model": GATE_MODEL[0], "rows": GATE_ROWS, "ctx": GATE_CTX, "batch_sha256": xh, "precision": {},
           "tol": HVP_TOL}
    for rev in GATE_REVS + (GATE_BESIDE,):
        torch.set_default_dtype(torch.float32)
        m = _released(device, rev)
        op, params = hvp_fn(m, x)
        g = torch.Generator().manual_seed(0)
        r1 = _normed([torch.randn(p.shape, generator=g).to(device) for p in params])
        r2 = _normed([torch.randn(p.shape, generator=g).to(device) for p in params])
        u = _normed(op(_normed(op(_normed(op(r1))))))
        hu = op(u)
        q32 = dot(u, hu)
        m64 = copy.deepcopy(m).double()
        torch.set_default_dtype(torch.float64)          # transformers builds its causal mask in the default dtype
        op64, p64 = hvp_fn(m64, x)
        u64 = [t.double() for t in u]
        hu64 = op64(u64)
        q64 = dot(u64, hu64)
        out["precision"][rev] = {"rayleigh_fp32": q32, "rayleigh_fp64": q64, "rel": abs(q32 - q64) / abs(q64),
                                 "beside": rev == GATE_BESIDE}
        if rev == GATE_REVS[1]:
            w = _normed(op(_normed(op(_normed(op(r2))))))
            out["sym_rel"] = abs(dot(w, hu) - dot(u, op(w))) / nrm(hu)

            def grad_at(eps):
                with torch.no_grad():
                    for p, d in zip(p64, u64):
                        p.add_(d, alpha=eps)
                tot = [torch.zeros_like(p) for p in p64]
                for k in range(x.shape[0]):
                    r = x[k:k + 1]
                    lg = m64(input_ids=r[:, :-1]).logits
                    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), r[:, 1:].reshape(-1))
                    for t, gg in zip(tot, torch.autograd.grad(loss, p64)):
                        t.add_(gg, alpha=1 / x.shape[0])
                with torch.no_grad():
                    for p, d in zip(p64, u64):
                        p.add_(d, alpha=-eps)
                return tot
            gp, gm = grad_at(FD_STEP), grad_at(-FD_STEP)
            out["fd_rel_err_fp64"] = nrm([(a - b) / (2 * FD_STEP) - c for a, b, c in zip(gp, gm, hu64)]) / nrm(hu64)
            out["fd_step"] = FD_STEP
            del gp, gm
        torch.set_default_dtype(torch.float32)
        if rev == GATE_REVS[1]:
            xb, _ = held_batch(device, sr.BATCH_ROWS)
            opb, _ = hvp_fn(m, xb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            opb(u)
            torch.cuda.synchronize()
            out.update({"hv_seconds_at_batch": time.time() - t0, "batch_rows": sr.BATCH_ROWS, "ctx_measured": sr.CTX,
                        "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
            out["max_minutes_per_state"] = out["hv_seconds_at_batch"] * sr.N_MAX / 60
        del m, m64, op, op64, p64, hu64
        torch.cuda.empty_cache()
    prec_ok = all(v["rel"] <= HVP_TOL["prec"] for k, v in out["precision"].items() if not v["beside"])
    out["pass"] = bool(prec_ok and out["fd_rel_err_fp64"] <= HVP_TOL["fd"] and out["sym_rel"] <= HVP_TOL["sym"])
    return out


def gate(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackC_writeup import c4_train as ct
    torch, device = ct._torch()
    q = g_quad()
    print(json.dumps({"G-quad": q["pass"], **{k: round(v["rel_err_pre"], 5) for k, v in q["cases"].items()}}), flush=True)
    h = g_hvp(device)
    print(json.dumps({**{k: h[k] for k in ("pass", "fd_rel_err_fp64", "sym_rel", "hv_seconds_at_batch", "peak_gib",
                                           "max_minutes_per_state")},
                      **{k: v["rel"] for k, v in h["precision"].items()}}), flush=True)
    out = {"G-quad": q, "G-hvp": h, "pass": bool(q["pass"] and h["pass"]),
           "code": {"c8_sharp": code_sha(), "c8_sharp_rules": sr.code_sha()}}
    _dump(Path(a.out) / "gate.json", out)
    return 0


# ---------------------------------------------------------------- 2. records, 3. reference

def records(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackA_force.a1a import V1
    from trackC_writeup import c4_read as cr
    from trackC_writeup import c4_train as ct
    from transformers import AutoTokenizer
    torch, device = cr.c._torch_setup()
    out, root = Path(a.out), Path(a.root)
    tok = AutoTokenizer.from_pretrained(cr.TOKENIZER[0], revision=cr.TOKENIZER[1])
    for arm in a.arm:
        first = True
        for s in t8.save_steps(arm):
            if all(cr.rec_path(out, arm, s, p).exists() for p in V1):
                continue
            d = ct.ckpt_dir(root, arm, s)
            if not d.exists():
                raise C8SharpRefused(f"{d}: missing")
            model = cr.load_local(d, device)
            cr._records_of(model, tok, device, out, arm, s, str(d))
            if first:
                r = json.loads(cr.rec_path(out, arm, s, V1[0]).read_text())
                print(f"first record populated: {arm} step {s}, n {r['n']}, e1c[0] {r['e1c'][0]:.6f}", flush=True)
                first = False
            del model
            torch.cuda.empty_cache()
        print(json.dumps({"arm": arm, "records": len(t8.save_steps(arm))}), flush=True)
    return 0


def reference(a) -> int:
    from trackC_writeup import c4pdf_read as pr
    from trackC_writeup import c4pdf_rules as rl
    from trackC_writeup import c8_read as c8r
    out = Path(a.out)
    rdir = out / "records_root"
    arms = pr.gated_arms()
    res = {"rules": {"c4pdf_rules": rl.code_sha(), "c8_rules": r8.code_sha(), "c8_sharp_rules": sr.code_sha()},
           "gated": list(arms), "per_tau": {k: {} for k in arms}, "inputs": {}}
    for arm in sr.RERUNS:
        st = list(t8.save_steps(arm))
        recs = pr.load_recs(rdir, arm, st)
        res["inputs"][f"{arm}_records"] = {"dir": str(rdir / arm), "sha256": c8r._records_sha(rdir, arm, st)}
        mats = pr.mats_of(recs, st, arms)
        br = None if arm == "F2" else rl.BRANCH_FROM
        for k in arms:
            tr = rl.transient(mats[k], st)
            v = {"transient": tr, "states": [st[0], st[-1]]}
            if tr["present"]:
                v["ref"] = [rl.previous(st, tr["onset"]), tr["onset"]]
                v["window"] = list(sr.window(tuple(v["ref"]), br))
            res["per_tau"][k][arm] = v
    _dump(out / "reference.json", res)
    print(json.dumps({k: {a_: (v.get("ref") if v["transient"]["present"] else None) for a_, v in d.items()}
                      for k, d in res["per_tau"].items()}), flush=True)
    return 0


# ---------------------------------------------------------------- 4. measure

def registration(entries: Path = ENTRIES, registry: Path = REGISTRY, check_git: bool = True) -> str:
    from trackC_writeup import c4pdf_train as pt
    if not entries.exists():
        return "pending"
    if check_git and not pt.committed(entries):
        raise C8SharpRefused(f"{entries}: counts only when committed (tracked and clean)")
    v = json.loads(entries.read_text()).get("ii")
    ids = {p.get("id") for p in json.loads(registry.read_text()).get("predictions", [])}
    return "declined" if v == "declined" else f"registered as {v}" if v in ids else "pending"


def raw_states(data: Path) -> Dict[str, List[int]]:
    """Raw λ's states: each original's U1 window(s) at every gated τ, ± 2 saved states; B_O and B_K
    at B_P's; A0's grid to 128."""
    from trackC_writeup import c4_train as ct
    from trackC_writeup import c4pdf_bk as bk
    from trackC_writeup import c4pdf_train as pt
    ref = json.loads((data / U1_REF).read_text())["per_tau"]
    out: Dict[str, List[int]] = {}

    def around(grid, wins):
        grid = sorted(grid)
        keep = set()
        for lo, hi in wins:
            idx = [i for i, s in enumerate(grid) if lo < s <= hi]
            keep |= {grid[j] for j in range(max(0, idx[0] - 2), min(len(grid), idx[-1] + 3))}
        return sorted(keep)
    for arm in ("F1", "Fh", "Ft", "BP"):
        out[arm] = around(pt.save_steps(arm), [tuple(ref[k][arm]["window"]) for k in ref])
    out["BO"] = [s for s in out["BP"] if s in pt.save_steps("BO")]
    out["BK"] = [s for s in out["BP"] if s in bk.save_steps("BK")]
    out["A0"] = [s for s in ct.GRID if s <= 128]
    return out


def _lam_done(f: Path) -> Dict[Tuple[str, str, int], dict]:
    if not f.exists():
        return {}
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return {(r["what"], r["arm"], r["step"]): r for r in rows}


def measure(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackA_force.a1a import DATA
    from trackC_writeup import c4_read as cr
    from trackC_writeup import c4_train as ct
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    if not g["pass"]:
        raise C8SharpRefused("the gate has not passed")
    if g["code"]["c8_sharp_rules"] != sr.code_sha():
        raise C8SharpRefused(f"gate ran on rules {g['code']['c8_sharp_rules']}, now {sr.code_sha()}")
    if not (out / "reference.json").exists():
        raise C8SharpRefused("no reference.json: rule T's onsets come before any λ")
    reg = registration()
    if reg == "pending":
        raise C8SharpRefused("U2's entry is neither registered nor declined (D4 (a)): no λ before the user's decision")
    torch, device = ct._torch()
    x, xh = held_batch(device)
    f = out / "lam.jsonl"
    done = _lam_done(f)
    if a.what == "pre":
        jobs = [(arm, s, Path(a.root)) for arm in (a.arm or sr.RERUNS) for s in t8.save_steps(arm)
                if t8.v_path(Path(a.root), arm, s).exists()]
    else:
        rs = raw_states(DATA)
        jobs = [(arm, s, RAW_ROOTS[arm]) for arm in (a.arm or rs) if arm in rs for s in rs[arm]]
    with f.open("a") as fh:
        for arm, s, root in jobs:
            if (a.what, arm, s) in done:
                continue
            d = ct.ckpt_dir(root, arm, s)
            m = cr.load_local(d, device)
            for p in m.parameters():
                p.requires_grad_(True)
            op, params = hvp_fn(m, x)
            rec = {"what": a.what, "arm": arm, "step": s, "ckpt": str(d), "batch_sha256": xh,
                   "registration": reg, "code": {"c8_sharp": code_sha(), "c8_sharp_rules": sr.code_sha()}}
            if a.what == "pre":
                vf = t8.v_path(root, arm, s)
                dd, k, masked = precond_d(m, vf)
                op = precond_op(op, dd)
                rec.update({"v_file": str(vf), "v_sha256": _sha(vf), "opt_step": k, "masked_v0": masked})
            t0 = time.time()
            r = power(op, params)
            rec.update({**r, "seconds": round(time.time() - t0, 1)})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps({k: rec[k] for k in ("what", "arm", "step", "iters", "converged", "seconds")}), flush=True)
            del m, op, params
            torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- 5. read

def read(a) -> int:
    from trackC_writeup import c8_read as c8r
    out = Path(a.out)
    ref = json.loads((out / "reference.json").read_text())
    lam = _lam_done(out / "lam.jsonl")
    pre = {arm: {s: r for (w, a_, s), r in lam.items() if w == "pre" and a_ == arm} for arm in sr.RERUNS}
    for arm in sr.RERUNS:
        want = [s for s in t8.save_steps(arm) if t8.v_path(Path(a.root), arm, s).exists()]
        if sorted(pre[arm]) != want:
            raise C8SharpRefused(f"{arm}: λ_P missing at {sorted(set(want) - set(pre[arm]))}")
    lams = {arm: {s: r["mu"] for s, r in pre[arm].items()} for arm in sr.RERUNS}
    unconv = {arm: [s for s, r in pre[arm].items() if not r["converged"]] for arm in sr.RERUNS}
    per_tau = {}
    for k, d in ref["per_tau"].items():
        prem = {arm: bool(d[arm]["transient"]["present"]) for arm in sr.RERUNS}
        wins = {arm: tuple(d[arm]["window"]) for arm in sr.WITH_TRANSIENT if prem[arm]}
        per_tau[k] = {"R_ii": sr.label_at_tau(lams, unconv, wins, prem),
                      "beside_theta_1": sr.label_at_tau(lams, unconv, wins, prem, theta=sr.THETA_BESIDE),
                      "windows": {a_: list(v) for a_, v in wins.items()}}
    joint = sr.joint({k: v["R_ii"]["label"] for k, v in per_tau.items()})
    ratio = {arm: {str(s): lams[arm][s] / sr.THRESH for s in sorted(lams[arm])} for arm in sr.RERUNS}
    raw = {}
    for (w, a_, s), r in sorted(lam.items()):
        if w == "raw":
            raw.setdefault(a_, {})[str(s)] = {"lam": r["mu"], "converged": r["converged"]}
    clip = {}
    for arm in sr.WITH_TRANSIENT:
        rows = r8.parse_log((Path(a.root) / arm / "train_log.jsonl").read_text().splitlines())
        for k, d in ref["per_tau"].items():
            if d[arm]["transient"]["present"]:
                lo, hi = d[arm]["window"]
                w_ = [float(rows[s]["grad_norm"]) for s in range(lo + 1, hi + 1)]
                clip.setdefault(arm, {})[k] = sum(g > r8.CLIP for g in w_) / len(w_)
    res = {"rule": sr.code_sha(), "theta": sr.THETA, "threshold": sr.THRESH, "per_tau": per_tau, "joint": joint,
           "beside": {"lambda_P_over_threshold": ratio, "raw_lambda": raw, "clip_share_in_window": clip,
                      "unconverged": unconv},
           "inputs": {"reference": {"path": str(out / "reference.json"), "sha256": _sha(out / "reference.json")},
                      "lam": {"path": str(out / "lam.jsonl"), "sha256": _sha(out / "lam.jsonl")},
                      "gate": {"path": str(out / "gate.json"), "sha256": _sha(out / "gate.json")},
                      "registration": registration()},
           "code": {"c8_sharp": code_sha(), "c8_sharp_rules": sr.code_sha()}}
    _dump(out / "reading.json", res)
    print(json.dumps({"joint": joint, **{k: v["R_ii"]["label"] for k, v in per_tau.items()}}), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="c8_sharp")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("gate")
    q.add_argument("--out", required=True)
    q = sub.add_parser("records")
    q.add_argument("--out", required=True, help="the records root (<u2 dir>/records_root)")
    q.add_argument("--root", default=str(t8.ROOT))
    q.add_argument("--arm", nargs="+", choices=t8.ARMS, required=True)
    q = sub.add_parser("reference")
    q.add_argument("--out", required=True)
    q = sub.add_parser("measure")
    q.add_argument("--out", required=True)
    q.add_argument("--root", default=str(t8.ROOT))
    q.add_argument("--what", choices=("pre", "raw"), required=True)
    q.add_argument("--arm", nargs="*")
    q = sub.add_parser("read")
    q.add_argument("--out", required=True)
    q.add_argument("--root", default=str(t8.ROOT))
    a = p.parse_args(argv)
    return {"gate": gate, "records": records, "reference": reference, "measure": measure, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
