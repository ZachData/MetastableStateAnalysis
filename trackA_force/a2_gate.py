"""
trackA_force/a2_gate.py — A2 unit 3's calibration gate, run before any real reading
(`design-A2.md` "Unit 3", "The gate"; `CLAUDE.md` "Calibration gate"). Torch float64 on the GPU.

**Synthetic stacks** (G1, G3, G4): 24 blocks, each with 16 identity-QK heads at β 3.5, an output
bias and an MLP, from a start with the measured start's spectrum (per v1 prompt: step 143000's
embedding rows ``x_0``, ``Q S Vᵀ`` with ``S``, ``V`` its own and ``Q`` a random orthonormal basis, so
the token identities are not the prompt's). LN1 is identity-affine, where a raw move ``|Cx|·g`` is
the frame move ``g`` to first order; G0 and G1 are also read on a copy with step 143000's LN1 per
block. A head with ``V = ±I`` adds ``±(s_l / 16)·|Cx_i|·(A U)_i`` at rows 1…n−1 (A the causal softmax
at β on ``U = f_l(x)[1:]``), ``s_l`` A1a's planted steps (log-uniform in [0.02, 0.5]). Unless an arm
says otherwise the MLP is off-field noise ``s_l·F̄·|Cx_i|·ĝ_i`` (ĝ a random unit direction per token,
F̄ the mean field) and ``b_O`` a constant ``0.1·s_l·F̄·med|Cx|·r̂`` (*placed*). The stack moves
``x ← x + Σ parts`` block by block.

- **G0** (quadrature): ``Σ_c Δ_c`` against the frame chord, ≤ 1e-6 relative, at every block of every
  stack (synthetic, 143000's LN1, every G2 seed, step 0).
- **G1** (planted effect): head ``l mod 16`` has ``V = −I``: at every block it reads ``proj < 0`` and
  ranks first in "against" (the most negative of 18).
- **G2** (null, family-level): Pythia's init redrawn (every weight matrix normal at its step-0
  σ, vectors as at step 0), 50 seeds + a 51st, v1's 7 prompts, β 3.5: per seed the 432
  components' median ``proj`` over prompts, ``|S⁻|`` and the top-43 share of the against mass; the
  51st seed's two values inside the 50 seeds' 5–95 % range. Real step 0 beside (not gated).
- **G3** (off-field): head ``(l + 5) mod 16`` moves its tokens by ``3·s_l·|F_i|`` along a unit
  direction ⊥ F_i in the tangent plane (a fixed plane's rotation ``J U_i``, F and U projected out):
  its off share ranks first and ``|proj| ≤ 0.05`` of its calibrated size ``disp / F̄`` (*placed*).
- **G4** (shared drift): the MLP adds one constant vector ``s_l·F̄·med|Cx|·r̂`` to every token;
  its drift-removed ``|proj_df| ≤ 0.1`` of its calibrated size (*placed*); its raw ``proj`` beside.

    python -m trackA_force.a2_gate --out <dir> [--device cuda] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import a2_comp as ac
from .a1a import DATA, N_BLOCKS, V1, prompts_hash, run_dir
from .a1a_gate import NS, planted_h

SEED = 20261016
BETA = 3.5
EPS = 1e-5
N_SEEDS = 50
TOL = {"G0": ac.G0_TOL, "G3_proj": 0.05, "G4_proj_df": 0.10, "G2_lo": 5.0, "G2_hi": 95.0}
MLP_NOISE, BIAS, ROT, DRIFT = 1.0, 0.1, 3.0, 1.0      # placed magnitudes (docstring)
START_STEP = 143000


def gate_sha() -> str:
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def measured_start(prompt: str, rng) -> np.ndarray:
    """(n, d): the prompt's step-143000 embedding rows' spectrum with random token directions."""
    z = np.load(run_dir(START_STEP, prompt) / "activations.npz")
    x0 = z["activations"][0].astype(np.float64) * z["norms"][0][:, None]
    _, S, Vt = np.linalg.svd(x0, full_matrices=False)
    Q, _ = np.linalg.qr(rng.standard_normal((x0.shape[0], S.size)))
    return (Q * S) @ Vt


def _unit(Z):
    return Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def synth_parts(x, s: float, arm: str, L: int, w, b, gen):
    """(18, n, d) raw parts at block L for ``arm`` ∈ {G1, G3, G4}; the planted index."""
    import torch
    n, d = x.shape
    U = ac.frame_t(x, w, b, EPS)
    k = (x - x.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
    Ur = U[1:]
    S = (BETA * (Ur @ Ur.T)).masked_fill(torch.triu(torch.ones(n - 1, n - 1, dtype=torch.bool, device=x.device), 1),
                                          -math.inf)
    AU = torch.zeros_like(x)
    AU[1:] = torch.softmax(S, dim=-1) @ Ur
    F = torch.zeros_like(x)
    F[1:] = ac.field_t(Ur, BETA)
    Fbar = float(F[2:].norm(dim=-1).mean())
    signs = torch.ones(ac.N_HEADS, dtype=x.dtype, device=x.device)
    plant = None
    if arm == "G1":
        plant = L % ac.N_HEADS
        signs[plant] = -1.0
    heads = signs[:, None, None] * (s / ac.N_HEADS) * k[None] * AU[None]
    rnd = lambda *sh: torch.randn(*sh, generator=gen, device=x.device, dtype=x.dtype)  # noqa: E731
    if arm == "G3":
        plant = (L + 5) % ac.N_HEADS
        e = torch.linalg.qr(rnd(d, 2))[0]                                 # a fixed plane, centred
        e = e - e.mean(0)
        JU = (U @ e[:, 0])[:, None] * e[:, 1] - (U @ e[:, 1])[:, None] * e[:, 0]
        xi = JU - (JU * U).sum(-1, keepdim=True) * U
        Fu = _unit(F)
        xi = xi - (xi * Fu).sum(-1, keepdim=True) * Fu
        xi = _unit(xi - xi.mean(-1, keepdim=True)) * F.norm(dim=-1, keepdim=True)
        xi[0] = 0.0
        heads[plant] = ROT * s * k * xi
    med_k = float(k.median())
    r = _unit(rnd(d))
    if arm == "G4":
        mlp = (DRIFT * s * Fbar * med_k) * r[None].expand(n, -1).clone()
    else:
        mlp = MLP_NOISE * s * Fbar * k * _unit(rnd(n, d))
    bO = (BIAS * s * Fbar * med_k) * _unit(rnd(d))[None].expand(n, -1)
    return torch.cat([heads, bO[None], mlp[None]], 0), plant


def stack(x0: np.ndarray, h: np.ndarray, arm: str, lnw, lnb, device: str, seed: int) -> List[Dict]:
    """One synthetic stack over 24 blocks; per block the read and the planted index."""
    import torch
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.as_tensor(x0, dtype=torch.float64, device=device)
    rows = []
    for L in range(N_BLOCKS):
        w = torch.as_tensor(lnw[L], dtype=torch.float64, device=device)
        b = torch.as_tensor(lnb[L], dtype=torch.float64, device=device)
        P, plant = synth_parts(x, float(h[L]), arm, L, w, b, gen)
        r = ac.read_block(x, P, len(ac.COMPS), w, b, EPS, BETA)
        rows.append({"block": L, "plant": plant, "g0": r["g0"], "proj": r["proj"].tolist(),
                     "proj_df": r["proj_df"].tolist(), "off": r["off"].tolist(), "disp": r["disp"].tolist(),
                     "field": r["total"]["field"], "total": r["total"]})
        x = x + P.sum(0)
    return rows


def judge_synth(arm: str, rows: List[Dict]) -> Dict:
    fails = []
    for r in rows:
        p, proj, F = r["plant"], np.array(r["proj"]), r["field"]
        cal = lambda c: r["disp"][c] / F                                  # noqa: E731
        if arm == "G1" and not (proj[p] < 0 and int(np.argmin(proj)) == p):
            fails.append((r["block"], "rank", int(np.argmin(proj)), proj[p]))
        if arm == "G3":
            if int(np.argmax(r["off"])) != p:
                fails.append((r["block"], "off_rank", int(np.argmax(r["off"])), r["off"][p]))
            if abs(proj[p]) > TOL["G3_proj"] * cal(p):
                fails.append((r["block"], "proj", proj[p] / cal(p)))
        if arm == "G4":
            m = len(ac.COMPS) - 1
            if abs(r["proj_df"][m]) > TOL["G4_proj_df"] * cal(m):
                fails.append((r["block"], "proj_df", r["proj_df"][m] / cal(m)))
    out = {"fails": fails, "g0_max": max(r["g0"] for r in rows)}
    if arm == "G3":
        out["proj_over_cal"] = [r["proj"][r["plant"]] / (r["disp"][r["plant"]] / r["field"]) for r in rows]
    if arm == "G4":
        m = len(ac.COMPS) - 1
        out["proj_df_over_cal"] = [r["proj_df"][m] / (r["disp"][m] / r["field"]) for r in rows]
        out["proj_over_cal"] = [r["proj"][m] / (r["disp"][m] / r["field"]) for r in rows]
    return out


def synthetic(device: str, prompts: Sequence[str]) -> Dict:
    from .a1a_gate import ln_all
    from .a1a import STEPS
    lns = ln_all()
    i = STEPS.index(START_STEP)
    ones, zeros = np.ones((N_BLOCKS, 1024)), np.zeros((N_BLOCKS, 1024))
    res = {}
    for j, p in enumerate(prompts):
        seed = SEED + 100 * j
        x0 = measured_start(p, np.random.default_rng(seed))
        if x0.shape[0] - 1 != NS[p]:
            raise SystemExit(f"refusing: {p}: start has {x0.shape[0] - 1} rows past 0, A1a's gate {NS[p]}")
        h = planted_h(seed, N_BLOCKS)
        cell = {"n": int(x0.shape[0]), "h": h.tolist()}
        for arm in ("G1", "G3", "G4"):
            rows = stack(x0, h, arm, ones, zeros, device, seed + 1)
            cell[arm] = {"rows": rows, **judge_synth(arm, rows)}
        rows = stack(x0, h, "G1", lns["w"][i], lns["b"][i], device, seed + 1)
        cell["G1_ln143000"] = {"rows": rows, **judge_synth("G1", rows)}
        res[p] = cell
        print(f"{p}: " + "; ".join(f"{a} fails {len(cell[a]['fails'])}, g0 {cell[a]['g0_max']:.1e}"
                                   for a in ("G1", "G3", "G4", "G1_ln143000")), flush=True)
    return res


# ---------------------------------------------------------------- G2: Pythia's init, redrawn

def init_sigmas(model) -> Dict[str, float]:
    return {n: float(p.detach().double().std()) for n, p in model.named_parameters() if p.ndim >= 2}


def redraw(model, sig: Dict[str, float], seed: int, device: str) -> None:
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in sig:
                p.normal_(0.0, sig[n], generator=g)


def read_model(hk, ids_by: Dict[str, "object"], ln: Dict, betas: Optional[Dict[str, Sequence[float]]] = None) -> Dict:
    """Per prompt the 432 proj values and off shares (at β 3.5, or each prompt's layer βs); the
    median over prompts' set statistics (`a2_comp.set_stats`); G0 and the pass's checks."""
    per, offs, g0, split, recon = {}, {}, 0.0, 0.0, 0.0
    for p, ids in ids_by.items():
        hs, rows = hk.run_read(ids, ln, betas[p] if betas else [BETA] * N_BLOCKS)
        per[p], offs[p] = ac.flatten(rows, "proj"), ac.off_global(rows)
        g0 = max(g0, max(r["g0"] for r in rows.values()))
        split = max(split, max(r["split_rel"] for r in rows.values()))
        recon = max(recon, ac.check_recon(hs, rows))
    med = np.median(np.stack([per[p] for p in ids_by]), axis=0)
    st = ac.set_stats(med, np.median(np.stack([offs[p] for p in ids_by]), axis=0))
    return {**{k: v for k, v in st.items() if not k.endswith("_idx")},
            "g0_max": g0, "split_max": split, "recon_max": recon, "median_proj": med.tolist()}


def stored_match(hs: np.ndarray, rd: Path) -> float:
    z = np.load(rd / "activations.npz")
    st = z["activations"] * z["norms"][..., None]
    unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)    # noqa: E731
    return float(np.abs(unit(hs) - unit(st)).max())


BAND_KEYS = ("S_minus", "top43_minus", "S_plus", "top43_plus", "S_zero", "top43_zero", "n_negative", "gini")


def g2(device: str, prompts: Sequence[str], n_seeds: int, betas: Optional[Dict[str, Sequence[float]]] = None) -> Dict:
    """G2's redraws; ``betas`` (per prompt, 24 layer βs) reads the field there instead of at 3.5
    (unit 4's init band, STATE Blocked 58 (a1)). The gated pair is ``S_minus``, ``top43_minus``."""
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1e_energy_field.u2_attn import ln_from, token_ids
    model, _ = load_model("pythia-410m-step0")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("refusing: model is not float32")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    dev = next(model.parameters()).device
    ids_by = {p: torch.tensor([token_ids(tok, run_dir(0, p))], device=dev) for p in prompts}
    ln = ln_from(model)
    sig = init_sigmas(model)
    hk = ac.comp_hooked(model)
    hs, _ = hk.run_read(ids_by[prompts[0]], ln, [BETA] * N_BLOCKS)
    match = stored_match(hs, run_dir(0, prompts[0]))
    if match > 1e-4:
        raise SystemExit(f"refusing: step 0's pass against its stored rows {match:.1e}")
    step0 = read_model(hk, ids_by, ln, betas)
    step0["stored_match"] = match
    print(f"step 0 (beside): |S-| {step0['S_minus']}, top43 {step0['top43_minus']:.3f}, "
          f"g0 {step0['g0_max']:.1e}, split {step0['split_max']:.1e}, recon {step0['recon_max']:.1e}", flush=True)
    seeds = []
    for k in range(n_seeds + 1):
        t0 = time.monotonic()
        redraw(model, sig, SEED + k, str(dev))
        r = read_model(hk, ids_by, ln, betas)
        r.pop("median_proj")
        seeds.append({"seed": SEED + k, **r})
        print(f"G2 seed {k}: |S-| {r['S_minus']}, top43 {r['top43_minus']:.3f}, g0 {r['g0_max']:.1e}, "
              f"recon {r['recon_max']:.1e} ({time.monotonic() - t0:.0f}s)", flush=True)
    hk.close()
    ref, test = seeds[:n_seeds], seeds[n_seeds]
    out = {"step0": step0, "seeds": seeds, "sigmas": sig, "range": {}}
    for key in BAND_KEYS:
        lo, hi = np.percentile([s[key] for s in ref], [TOL["G2_lo"], TOL["G2_hi"]])
        out["range"][key] = {"lo": float(lo), "hi": float(hi), "test": test[key], "step0": step0[key],
                             "inside": bool(lo <= test[key] <= hi), "step0_inside": bool(lo <= step0[key] <= hi)}
    out["g0_max"] = max([step0["g0_max"]] + [s["g0_max"] for s in seeds])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="one prompt, 3 seeds (a check, not the gate)")
    a = ap.parse_args(argv)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    prompts = V1[:1] if a.quick else V1
    n_seeds = 3 if a.quick else N_SEEDS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rec = {"code": ac.code_sha(), "gate_code": gate_sha(), "prompts_hash": prompts_hash(), "quick": a.quick, "beta": BETA,
           "nodes": ac.NODES, "tol": TOL, "seed": SEED, "device": a.device}
    rec["synthetic"] = synthetic(a.device, prompts)
    rec["G2"] = g2(a.device, prompts, n_seeds)
    syn = rec["synthetic"]
    g0 = max([c[arm]["g0_max"] for c in syn.values() for arm in ("G1", "G3", "G4", "G1_ln143000")]
             + [rec["G2"]["g0_max"]])
    v = {"G0": g0 <= TOL["G0"],
         "G1": all(not c["G1"]["fails"] for c in syn.values()),
         "G2": all(rec["G2"]["range"][k]["inside"] for k in ("S_minus", "top43_minus")),
         "G3": all(not c["G3"]["fails"] for c in syn.values()),
         "G4": all(not c["G4"]["fails"] for c in syn.values())}
    rec["verdict"] = {**v, "pass": all(v.values()), "g0_max": g0,
                      "beside": {"G1_ln143000": all(not c["G1_ln143000"]["fails"] for c in syn.values())}}
    rec["seconds"] = time.monotonic() - t0
    (out / "a2_gate.json").write_text(json.dumps(rec) + "\n")
    print(f"verdict {rec['verdict']} ({rec['seconds']:.0f}s)", flush=True)
    return 0 if rec["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
