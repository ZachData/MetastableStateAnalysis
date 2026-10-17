"""
trackA_force/a2_ablate.py — A2 unit 5: the ablation, its nulls and the reading (`design-A2.md`
"Unit 5"; frozen 2026-10-16, Blocked 56 (a); the null by STATE Blocked 59 (b), the user's).

- ``rank``: at a step, the 8 long passages (`LONG8_HASH` ``ba605f4e14b5``) through unit 4's hooked
  pass (`a2_comp`), the field at the step's layer β (*placed*: the median over v1's 7 prompts of
  A1a's ``fe_full`` layer βs, since A1a measured none on the long passages). Per passage the 432
  components' ``proj``, ``off_global``, ``size`` and each counted part's sum over positions 1…n−1.
- ``sets``: the median over the 8 passages per component: ``S⁻`` (the 50 % against set), ``S⁺``
  (the top ``|S⁻|`` along components: the opposite-sign control, same count) and ``S⁰`` (the 50 %
  off set, beside). The ablation constant is each component's mean output over the 8 passages'
  tokens (positions 1…n−1, pooled). **The null** (Blocked 59 (b)): 20 random sets per arm (seed
  20261016), each member replaced by a component of the same type (head / ``b_O`` / MLP) and
  decile of mean ``|v_c|`` over the 432; heads and ``b_O`` also in the same band (L1–8, L9–16,
  L17–24), MLPs in any band (each set's bands recorded). A draw never takes the arm's own members
  or repeats one; a member with no such component widens to any band (counted); an arm whose
  band-matched heads cannot give 20 distinct sets drops the band for heads too (*placed*,
  recorded as ``heads_any_band``). Refuses below 20 distinct sets.
- ``ablate``: on v1's 7 prompts, mean-ablation at positions 1…n−1 (position 0 untouched): a head's
  ``W_O^h`` output replaced by its mean (a hook on ``attention.dense``), an MLP's output by its
  mean. ``b_O`` is constant over tokens, so its mean-ablation is the identity (it is recorded as
  such, not dropped). Readouts per prompt, each block in its own LN1 frame ``f_l``, positions
  1…n−1, at A1a's layer β for that (step, prompt):
  (i) the mean cosine of ``f_23(x_23 + v_23)``, the last block's end (*placed*: A1b's ``cos_23``
  read at the block's end, so the last block's own components count);
  (iii) whether ``E_β(f_l(x_l + v_l)) > E_β(f_l(x_l))`` (Geshkovski's ``E_β``, ``core.metrics``),
  per block; ``rise`` the share over 24 blocks;
  (ii) manipulation check: ``Σ_{l ≤ 22}`` of the mean lifted (log) projected step over A1a's causal
  t* at p10; (iv) next-token loss, quoted.
- **The rule** (design, *placed* aggregation): per step, an arm's (i) is the mean over the 7
  prompts and its (iii) the share of all 7 × 24 blocks. (i) *passes* at a step if ablating ``S⁻``
  gives a (i) above the unablated model's, above ``S⁺``'s and above all 20 of its null sets';
  (iii) likewise. **Supported** if both pass at ≥ 2 of the 3 steps; **mixed** if not, and one of
  them passes at ≥ 2 (named); **refused** otherwise. ``S⁰`` is read by the same test, beside.
- ``calib``: the rule's own calibration, before the real steps: (a) step 0 end to end (rank on the
  long passages, ablate, readouts on v1) must not have both pass; (b) on unit 3's synthetic stack
  (`a2_gate`, arm G1: one ``V = −I`` head per block among 15 ``+I``), ranked on a separate start
  and read on the 7 prompts' starts at β 3.5, both must pass. ``real`` refuses unless both held.

    python -m trackA_force.a2_ablate calib  --out <dir> --gate <a2_gate.json>
    python -m trackA_force.a2_ablate real   --out <dir> --gate <a2_gate.json>
    python -m trackA_force.a2_ablate report --out <dir>

Tier 1: exploratory, unregistered. The pass float32 on the GPU, eager, TF32 off; reads float64.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import a2_comp as ac
from .a1a import N_BLOCKS, REPO, T_BLOCKS, V1, prompts_hash, run_dir
from .a2_batch import _model, _setup, a1a_betas, check_gate, A1A_DIR

REAL = (2000, 8000, 143000)          # design: the OV turn, the sum-over-heads ascent, the end
CAL_STEP = 0
N_NULL = 20
SEED = 20261016
K = len(ac.COMPS)                    # 18 per block
MLP, BIAS = K - 1, K - 2
BANDS = ((0, 8), (8, 16), (16, 24))  # L1–8, L9–16, L17–24 as blocks 0–7, 8–15, 16–23
PASS_STEPS = 2                       # design: at ≥ 2 of the 3 steps
SYN_BETA = 3.5


def code_sha() -> str:
    h = hashlib.sha256(ac.code_sha().encode())
    for f in ("trackA_force/a2_ablate.py", "trackA_force/a2_gate.py", "trackA_force/a2_batch.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: types, sizes, nulls

def ctype(i: int) -> str:
    c = i % K
    return "mlp" if c == MLP else ("bias" if c == BIAS else "head")


def cband(i: int) -> int:
    L = i // K
    return next(k for k, (a, b) in enumerate(BANDS) if a <= L < b)


def deciles(size: np.ndarray) -> np.ndarray:
    """(432,) decile 0…9 of each component's mean ``|v_c|`` over all 432 (ties by rank)."""
    r = np.argsort(np.argsort(size, kind="stable"), kind="stable")
    return (r * 10 // size.size).astype(int)


def pool(i: int, dec: np.ndarray, banned: set, band: bool) -> List[int]:
    t, d, b = ctype(i), dec[i], cband(i)
    return [j for j in range(dec.size) if j not in banned and ctype(j) == t and dec[j] == d
            and (not band or cband(j) == b)]


def null_sets(arm: Sequence[int], dec: np.ndarray, rng, n: int = N_NULL, tries: int = 5000) -> Dict:
    """``n`` distinct size-matched sets for ``arm`` (Blocked 59 (b): MLPs drop the band match). If the
    band-matched heads cannot give ``n`` distinct sets, heads drop it too (``heads_any_band``)."""
    for head_band in (True, False):
        r = _draw(arm, dec, rng, n, tries, head_band)
        if r is not None:
            return {**r, "heads_any_band": not head_band}
    raise SystemExit(f"refusing: fewer than {n} distinct null sets for an arm of {len(arm)}, any band")


def _draw(arm, dec, rng, n, tries, head_band: bool) -> Optional[Dict]:
    arm = [int(i) for i in arm]
    seen, sets, widened = set(), [], 0
    for _ in range(tries):
        if len(sets) == n:
            break
        banned, pick, wid = set(arm), [], 0
        for i in arm:
            P = pool(i, dec, banned, band=head_band and ctype(i) != "mlp")
            if not P:
                P, wid = pool(i, dec, banned, band=False), wid + 1
            if not P:
                return None
            j = int(rng.choice(P))
            pick.append(j)
            banned.add(j)
        key = frozenset(pick)
        if key in seen:
            continue
        seen.add(key)
        sets.append(sorted(pick))
        widened += wid
    if len(sets) < n:
        return None
    return {"sets": sets, "widened": widened,
            "bands": [[cband(j) for j in s] for s in sets], "types": [[ctype(j) for j in s] for s in sets]}


def arms_from(proj: np.ndarray, off: np.ndarray) -> Dict[str, List[int]]:
    sm = ac.mass_set(proj, -1)
    pos = np.flatnonzero(proj > 0)
    if pos.size < sm.size:
        raise SystemExit(f"refusing: {pos.size} along components for an S+ of {sm.size}")
    sp = pos[np.argsort(-proj[pos], kind="stable")][:sm.size]
    return {"S_minus": sm.tolist(), "S_plus": sp.tolist(), "S_zero": ac.mass_set(off, +1).tolist()}


def energy(U, beta: float) -> float:
    """``E_β`` on unit rows (torch), as `core.metrics.interaction_energy`; at β = 0 the mean cosine
    (its β → 0 limit, up to a constant), so "rises" stays defined."""
    G = U @ U.T
    m = U.shape[0]
    if beta == 0.0:
        return float(G.mean())
    return float(((beta * G).exp().sum() / (2.0 * beta * m * m)))


def readouts(xs, ln_w, ln_b, eps: float, betas: Sequence[float]) -> Dict:
    """``xs`` (25, n, d) torch float64: the 24 block inputs and the last block's end (pre-final-LN)."""
    import torch
    from .a1a_gate import _t_lifted
    rise, proj = [], []
    for L in range(N_BLOCKS):
        w = torch.as_tensor(ln_w[L], dtype=xs.dtype, device=xs.device)
        b = torch.as_tensor(ln_b[L], dtype=xs.dtype, device=xs.device)
        U0, U1 = ac.frame_t(xs[L], w, b, eps)[1:], ac.frame_t(xs[L + 1], w, b, eps)[1:]
        beta = float(betas[L])
        rise.append(bool(energy(U1, beta) > energy(U0, beta)))
        F = ac.field_t(U0, beta)
        xx = (F * F).sum(-1)
        Dm = _t_lifted(U0, U1, "log")
        v = (Dm * F).sum(-1) / xx
        ok = (xx >= 1e-18) & torch.isfinite(v)
        proj.append(float(v[ok].mean()))
        if L == N_BLOCKS - 1:
            n = U1.shape[0]
            s = U1.sum(0)
            cos = float((s @ s - n) / (n * (n - 1)))
    return {"cos": cos, "rise": rise, "proj": proj, "T_proj": float(sum(proj[l] for l in T_BLOCKS))}


def step_rule(res: Dict[str, Dict], arm: str = "S_minus", control: Optional[str] = "S_plus") -> Dict:
    """One step's test for ``arm``: each readout above the unablated, the control and all nulls."""
    def agg(rows: List[Dict]) -> Tuple[float, float]:
        return (float(np.mean([r["cos"] for r in rows])), float(np.mean([x for r in rows for x in r["rise"]])))
    base = agg(res["none"]["prompts"])
    a = agg(res[arm]["prompts"])
    ctl = agg(res[control]["prompts"]) if control else (-math.inf, -math.inf)
    nul = [agg(n["prompts"]) for n in res[f"{arm}_null"]]
    out = {"base": base, "arm": a, "control": None if control is None else ctl, "null_max": (max(x[0] for x in nul), max(x[1] for x in nul)),
           "null_n_above": (sum(x[0] >= a[0] for x in nul), sum(x[1] >= a[1] for x in nul))}
    for k, name in ((0, "cos"), (1, "rise")):
        out[f"{name}_pass"] = bool(a[k] > base[k] and a[k] > ctl[k] and all(a[k] > x[k] for x in nul))
    out["both"] = out["cos_pass"] and out["rise_pass"]
    return out


def reading(rules: Dict[int, Dict]) -> Dict:
    nc = sum(r["cos_pass"] for r in rules.values())
    nr = sum(r["rise_pass"] for r in rules.values())
    nb = sum(r["both"] for r in rules.values())
    if nb >= PASS_STEPS:
        v = "supported"
    elif nc >= PASS_STEPS or nr >= PASS_STEPS:
        v = "mixed: (i) re-clustering" if nc >= PASS_STEPS else "mixed: (iii) monotone energy"
    else:
        v = "refused"
    return {"verdict": v, "steps_cos": nc, "steps_rise": nr, "steps_both": nb}


# ---------------------------------------------------------------- rank (GPU)

def long_ids(tok) -> Dict[str, List[int]]:
    from p1e_energy_field.long_prompts_1e import expected_tokens, load8
    texts, exp = load8(), expected_tokens()
    out = {}
    for k, t in sorted(texts.items()):
        ids = tok(t)["input_ids"]
        if len(ids) != exp[k] or len(ids) > 2048:
            raise SystemExit(f"refusing: {k}: {len(ids)} tokens, provenance {exp[k]}")
        out[k] = ids
    return out


def step_beta(step: int) -> List[float]:
    b = a1a_betas(step)
    return [float(x) for x in np.median(np.array([b[p] for p in V1]), 0)]


def rank(step: int, out: Path, code: str) -> Dict:
    """The 8 passages at ``step``; returns the sets record (written to ``sets_step<step>.json``)."""
    import torch
    from transformers import AutoTokenizer
    from p1e_energy_field.long_prompts_1e import LONG8_HASH
    from p1e_energy_field.u2_attn import ln_from
    f = out / f"sets_step{step}.json"
    if f.exists():
        s = json.loads(f.read_text())
        if s.get("code") != code:
            raise SystemExit(f"refusing to resume: {f} was written by other code")
        return s
    rd = out / "rank"
    rd.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    ids_by = long_ids(tok)
    beta = step_beta(step)
    model = _model(step)
    ln = ln_from(model)
    hk = ac.comp_hooked(model)
    dev = next(model.parameters()).device
    t0 = time.monotonic()
    for k, ids in ids_by.items():
        g = rd / f"step{step}_{k}.npz"
        if g.exists() and json.loads(str(np.load(g)["meta"])).get("code") == code:
            continue
        hs, rows = hk.run_read(torch.tensor([ids], device=dev), ln, beta, part_sum=True)
        recon = ac.check_recon(hs, rows)
        if recon > 1e-4:
            raise SystemExit(f"refusing: {k} at {step}: counted parts against the hidden states {recon:.1e}")
        proj, off = ac.flatten(rows, "proj"), ac.off_global(rows)
        size = ac.flatten(rows, "size")
        psum = np.stack([rows[L]["part_sum"] for L in range(N_BLOCKS)]).reshape(N_BLOCKS * K, -1)
        if not (np.all(np.isfinite(proj)) and np.any(proj != 0) and np.all(np.isfinite(psum))):
            raise SystemExit(f"refusing: {k} at {step}: proj or part sums not populated")
        meta = {"code": code, "step": step, "passage": k, "n": len(ids), "long8": LONG8_HASH, "beta": beta,
                "recon": recon, "g0_max": max(r["g0"] for r in rows.values()),
                "split_max": max(r["split_rel"] for r in rows.values())}
        np.savez(g, proj=proj, off=off, size=size, psum=psum, meta=json.dumps(meta))
        print(f"rank step{step} {k}: n {len(ids)}, g0 {meta['g0_max']:.1e}, recon {recon:.1e} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
        del hs, rows
        gc.collect()
        torch.cuda.empty_cache()
    hk.close()
    del model, hk
    torch.cuda.empty_cache()
    Z = {k: np.load(rd / f"step{step}_{k}.npz") for k in ids_by}
    proj = np.median(np.stack([Z[k]["proj"] for k in ids_by]), 0)
    off = np.median(np.stack([Z[k]["off"] for k in ids_by]), 0)
    size = np.mean(np.stack([Z[k]["size"] for k in ids_by]), 0)
    ntok = sum(len(ids) - 1 for ids in ids_by.values())
    means = sum(Z[k]["psum"] for k in ids_by) / ntok                       # (432, d)
    np.save(out / f"means_step{step}.npy", means)
    arms = arms_from(proj, off)
    dec = deciles(size)
    rng = np.random.default_rng(SEED + step)
    nulls = {a: null_sets(arms[a], dec, rng) for a in arms}
    s = {"code": code, "step": step, "beta": beta, "long8": LONG8_HASH, "arms": arms,
         "names": {a: [ac.comp_name(i) for i in v] for a, v in arms.items()},
         "proj": proj.tolist(), "off": off.tolist(), "size": size.tolist(), "decile": dec.tolist(),
         "nulls": nulls, "stats": {k: v for k, v in ac.set_stats(proj, off).items() if not k.endswith("_idx")},
         "arm_deciles": {a: [int(dec[i]) for i in v] for a, v in arms.items()},
         "arm_bands": {a: [cband(i) for i in v] for a, v in arms.items()}}
    f.write_text(json.dumps(s) + "\n")
    return s


# ---------------------------------------------------------------- ablate (GPU)

class Ablate:
    """Mean-ablation hooks for one set of components (block-major indices into the 432)."""

    def __init__(self, model, means: np.ndarray):
        import torch
        self.model, self.handles = model, []
        self.layers = getattr(model, "gpt_neox", model).layers
        self.H = model.config.num_attention_heads
        dev = next(model.parameters()).device
        self.k = self.H + 2                                    # heads, b_O, MLP (a2_comp.COMPS)
        self.means = torch.as_tensor(means, dtype=torch.float32, device=dev).view(len(self.layers), self.k, -1)
        self.idx: Dict[int, List[int]] = {}
        self.end = None
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.attention.dense.register_forward_hook(
                lambda m, i, o, _l=l: self._heads(_l, m, i[0], o)))
            self.handles.append(layer.mlp.register_forward_hook(lambda m, i, o, _l=l: self._mlp(_l, o)))
        self.handles.append(self.layers[-1].register_forward_hook(
            lambda m, i, o: setattr(self, "end", o[0][0].detach())))

    def set(self, comps: Sequence[int]) -> None:
        self.idx = {}
        for c in comps:
            self.idx.setdefault(int(c) // self.k, []).append(int(c) % self.k)

    def _heads(self, l, m, z, o):
        hs = [c for c in self.idx.get(l, []) if c < self.H]
        if not hs:
            return o
        o = o.clone()
        hd = z.shape[-1] // self.H
        for h in hs:
            sl = slice(h * hd, (h + 1) * hd)
            o[:, 1:] += self.means[l, h] - z[:, 1:, sl] @ m.weight[:, sl].T
        return o

    def _mlp(self, l, o):
        if self.k - 1 not in self.idx.get(l, []):
            return o
        o = o.clone()
        o[:, 1:] = self.means[l, self.k - 1]
        return o

    def close(self):
        for h in self.handles:
            h.remove()


def unembed(step: int, dev):
    """The checkpoint's unembedding (`load_model` returns the bare model), for loss (iv)."""
    import torch
    from transformers import GPTNeoXForCausalLM
    lm = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m", revision=f"step{step}", torch_dtype=torch.float32)
    W = lm.embed_out.weight.detach().to(dev)
    del lm
    return W


def ablate_step(step: int, sets: Dict, out: Path, code: str) -> Dict:
    import torch
    from transformers import AutoTokenizer
    from core.holdout import refuse_held_out
    from p1e_energy_field.u2_attn import ln_from, token_ids
    f = out / f"ablate_step{step}.json"
    if f.exists():
        r = json.loads(f.read_text())
        if r.get("code") != code:
            raise SystemExit(f"refusing to resume: {f} was written by other code")
        return r
    refuse_held_out([run_dir(step, p) for p in V1], context="a2_ablate")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    betas = a1a_betas(step)
    tstar = {p: json.loads((A1A_DIR / f"step{step}_{p}.json").read_text())["tstar"]["causal"].get("p10") for p in V1}
    model = _model(step)
    ln = ln_from(model)
    W_U = unembed(step, next(model.parameters()).device)
    ab = Ablate(model, np.load(out / f"means_step{step}.npy"))
    dev = next(model.parameters()).device
    ids_by = {p: torch.tensor([token_ids(tok, run_dir(step, p))], device=dev) for p in V1}
    configs = [("none", [])] + [(a, sets["arms"][a]) for a in ("S_minus", "S_plus", "S_zero")]
    configs += [(f"{a}_null", s) for a in ("S_minus", "S_plus", "S_zero") for s in sets["nulls"][a]["sets"]]
    res: Dict = {}
    t0 = time.monotonic()
    for name, comps in configs:
        ab.set(comps)
        rows = []
        for p, ids in ids_by.items():
            with torch.no_grad():
                o = model(input_ids=ids, output_hidden_states=True, output_attentions=False, use_cache=False)
            xs = torch.stack([h[0] for h in o.hidden_states[:N_BLOCKS]] + [ab.end]).double()
            r = readouts(xs, ln["w"], ln["b"], ln["eps"], betas[p])
            lg = o.last_hidden_state[0, :-1] @ W_U.T                       # post-final-LN state
            r["loss"] = float(torch.nn.functional.cross_entropy(lg, ids[0, 1:]))
            r["T_over_tstar"] = r["T_proj"] / tstar[p] if tstar[p] else None
            r["prompt"] = p
            rows.append(r)
            del o, xs
        entry = {"comps": list(comps), "prompts": rows}
        if name.endswith("_null"):
            res.setdefault(name, []).append(entry)
        else:
            res[name] = entry
    ab.close()
    if len(res["S_minus"]["prompts"]) != len(V1) or not np.isfinite(res["none"]["prompts"][0]["cos"]):
        raise SystemExit(f"refusing: step {step}: readouts not populated")
    rec = {"code": code, "step": step, "prompts_hash": prompts_hash(), "res": res,
           "rule": step_rule(res), "beside_S_zero": step_rule(res, "S_zero", None),
           "seconds": time.monotonic() - t0}
    f.write_text(json.dumps(rec) + "\n")
    print(f"ablate step {step}: {rec['rule']} ({rec['seconds']:.0f}s)", flush=True)
    del model, ab
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- the synthetic calibration

def synth_run(x0: np.ndarray, h: np.ndarray, seed: int, device: str, comps: Sequence[int] = (),
              means: Optional[np.ndarray] = None, read: bool = False) -> Dict:
    """Unit 3's G1 stack (identity LN1, β 3.5), the parts in ``comps`` replaced by ``means`` at rows
    1…n−1. ``read``: the per-component ranking reads and the part means; the readouts always."""
    import torch
    from .a2_gate import EPS, synth_parts
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.as_tensor(x0, dtype=torch.float64, device=device)
    d = x.shape[1]
    w, b = torch.ones(d, dtype=torch.float64, device=device), torch.zeros(d, dtype=torch.float64, device=device)
    ab: Dict[int, List[int]] = {}
    for c in comps:
        ab.setdefault(int(c) // K, []).append(int(c) % K)
    xs, rows, pmean = [x], {}, []
    for L in range(N_BLOCKS):
        P, _ = synth_parts(x, float(h[L]), "G1", L, w, b, gen)
        for c in ab.get(L, []):
            P[c, 1:] = torch.as_tensor(means[L * K + c], dtype=P.dtype, device=device)
        if read:
            rows[L] = ac.read_block(x, P, K, w, b, EPS, SYN_BETA)
            pmean.append(P[:, 1:].mean(1).cpu().numpy())
        x = x + P.sum(0)
        xs.append(x)
    ones, zeros = np.ones((N_BLOCKS, d)), np.zeros((N_BLOCKS, d))
    out = {"read": readouts(torch.stack(xs), ones, zeros, EPS, [SYN_BETA] * N_BLOCKS)}
    if read:
        out["proj"], out["off"] = ac.flatten(rows, "proj"), ac.off_global(rows)
        out["size"], out["means"] = ac.flatten(rows, "size"), np.concatenate(pmean)
    return out


def synthetic(device: str) -> Dict:
    from .a1a_gate import planted_h
    from .a2_gate import SEED as GSEED, measured_start
    # rank and means on a start of its own (the first prompt's spectrum, another random basis and h)
    rs = GSEED + 999
    r0 = synth_run(measured_start(V1[0], np.random.default_rng(rs)), planted_h(rs, N_BLOCKS), rs + 1, device, read=True)
    arms = arms_from(r0["proj"], r0["off"])
    dec = deciles(r0["size"])
    nulls = {a: null_sets(arms[a], dec, np.random.default_rng(SEED)) for a in ("S_minus", "S_plus")}
    planted = [L * K + (L % ac.N_HEADS) for L in range(N_BLOCKS)]
    configs = [("none", []), ("S_minus", arms["S_minus"]), ("S_plus", arms["S_plus"])]
    configs += [(f"{a}_null", s) for a in ("S_minus", "S_plus") for s in nulls[a]["sets"]]
    res: Dict = {}
    for name, comps in configs:
        rows = []
        for j, p in enumerate(V1):                                       # read on the gate's 7 starts
            seed = GSEED + 100 * j
            x0 = measured_start(p, np.random.default_rng(seed))
            rows.append({**synth_run(x0, planted_h(seed, N_BLOCKS), seed + 1, device, comps, r0["means"])["read"],
                         "prompt": p})
        entry = {"comps": list(comps), "prompts": rows}
        if name.endswith("_null"):
            res.setdefault(name, []).append(entry)
        else:
            res[name] = entry
    rule = step_rule(res)
    return {"arms": arms, "names": {a: [ac.comp_name(i) for i in v] for a, v in arms.items()},
            "S_minus_planted": sum(i in planted for i in arms["S_minus"]), "nulls": nulls, "res": res,
            "rule": rule, "pass": rule["both"]}


# ---------------------------------------------------------------- commands

def calib(a) -> int:
    _setup("cuda")
    code = code_sha()
    check_gate(Path(a.gate), ac.code_sha())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "calib.json"
    t0 = time.monotonic()
    syn = synthetic("cuda")
    print(f"synthetic: {syn['rule']}; S- {len(syn['arms']['S_minus'])}, planted {syn['S_minus_planted']}", flush=True)
    sets0 = rank(CAL_STEP, out, code)
    print(f"step 0 sets: |S-| {len(sets0['arms']['S_minus'])}, nulls widened {sets0['nulls']['S_minus']['widened']}", flush=True)
    ab0 = ablate_step(CAL_STEP, sets0, out, code)
    rec = {"code": code, "synthetic": {k: v for k, v in syn.items() if k != "res"}, "synthetic_res": syn["res"],
           "step0_rule": ab0["rule"], "pass": {"synthetic": syn["pass"], "step0_not_supported": not ab0["rule"]["both"]},
           "seconds": time.monotonic() - t0}
    rec["pass"]["all"] = rec["pass"]["synthetic"] and rec["pass"]["step0_not_supported"]
    f.write_text(json.dumps(rec) + "\n")
    print(f"calibration {rec['pass']} ({rec['seconds']:.0f}s)", flush=True)
    return 0 if rec["pass"]["all"] else 1


def real(a) -> int:
    _setup("cuda")
    code = code_sha()
    check_gate(Path(a.gate), ac.code_sha())
    out = Path(a.out)
    c = json.loads((out / "calib.json").read_text())
    if c.get("code") != code or not c["pass"]["all"]:
        raise SystemExit(f"refusing: the rule's calibration is code {c.get('code')}, pass {c['pass']}; "
                         "the reading is not made (design \"Unit 5\")")
    rules = {}
    for step in REAL:
        s = rank(step, out, code)
        print(f"step {step} sets: |S-| {len(s['arms']['S_minus'])} {s['names']['S_minus']}", flush=True)
        rules[step] = ablate_step(step, s, out, code)["rule"]
    r = {"code": code, "rules": {str(k): v for k, v in rules.items()}, "reading": reading(rules)}
    (out / "reading.json").write_text(json.dumps(r) + "\n")
    print(f"reading: {r['reading']}", flush=True)
    return 0


def _agg(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def report(out: Path) -> int:
    code = code_sha()
    c = json.loads((out / "calib.json").read_text())
    L = [f"# A2 unit 5 (code {code})", "",
         f"Calibration: synthetic {c['pass']['synthetic']} (rule {c['synthetic']['rule']}; |S-| "
         f"{len(c['synthetic']['arms']['S_minus'])}, of them planted {c['synthetic']['S_minus_planted']}); "
         f"step 0 not supported {c['pass']['step0_not_supported']} (rule {c['step0_rule']})", ""]
    steps = [CAL_STEP] + [s for s in REAL if (out / f"ablate_step{s}.json").exists()]
    L += ["step | arm | |set| | cos (i) | rise (iii) | T/t* (ii) | loss (iv) | null max cos / rise | nulls ≥ arm (cos, rise) | widened",
          "---|---|---|---|---|---|---|---|---|---"]
    for s in steps:
        ab = json.loads((out / f"ablate_step{s}.json").read_text())
        st = json.loads((out / f"sets_step{s}.json").read_text())
        if ab.get("code") != code or st.get("code") != code:
            raise SystemExit(f"refusing: step {s} records from other code")
        res = ab["res"]
        for arm in ("none", "S_minus", "S_plus", "S_zero"):
            rows = res[arm]["prompts"]
            line = (f"{s} | {arm} | {len(res[arm]['comps'])} | {_agg(rows, 'cos'):.4f} | "
                    f"{np.mean([x for r in rows for x in r['rise']]):.3f} | {_agg(rows, 'T_over_tstar'):.3f} | "
                    f"{_agg(rows, 'loss'):.3f}")
            if arm != "none":
                nul = res[f"{arm}_null"]
                nc = [_agg(n["prompts"], "cos") for n in nul]
                nr = [np.mean([x for r in n["prompts"] for x in r["rise"]]) for n in nul]
                a_c, a_r = _agg(rows, "cos"), np.mean([x for r in rows for x in r["rise"]])
                line += (f" | {max(nc):.4f} / {max(nr):.3f} | {sum(x >= a_c for x in nc)}, {sum(x >= a_r for x in nr)} | "
                         f"{st['nulls'][arm]['widened']}")
            else:
                line += " | | |"
            L.append(line)
        L.append(f"{s} | rule | S- {ab['rule']['cos_pass']}/{ab['rule']['rise_pass']} (cos/rise) | "
                 f"S0 beside {ab['beside_S_zero']['cos_pass']}/{ab['beside_S_zero']['rise_pass']} | | | | | |")
        L.append(f"{s} | sets | S- {st['names']['S_minus']} | S+ {st['names']['S_plus']} | "
                 f"S0 {st['names']['S_zero']} | | | | |")
    if (out / "reading.json").exists():
        r = json.loads((out / "reading.json").read_text())
        L += ["", f"**Reading: {r['reading']['verdict']}** ({r['reading']})"]
    txt = "\n".join(L) + "\n"
    (out / "a2_unit5_report.txt").write_text(txt)
    print(txt)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("calib", "real", "report"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate")
    a = ap.parse_args(argv)
    if a.cmd == "calib":
        return calib(a)
    if a.cmd == "real":
        return real(a)
    return report(Path(a.out))


if __name__ == "__main__":
    raise SystemExit(main())
