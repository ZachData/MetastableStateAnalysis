"""
trackC_writeup/c5_massive.py — C5 unit 1, M2: the massive-activation coordinates projected out of
every token, both C2 counts recounted (`design-C5.md`, frozen 2026-10-23 on Blocked 104 (a)).

Per seed (0 the development set, 1–9 scored), on the 18 grid steps and the 7 v1 prompts:

- **pass1**: per step, each coordinate's max |x| over every token of every prompt at states 0–23
  (and S1's two halves, position 0 and the rest, for the known answer at seed 0's 143000);
- **select**: r = that max over the median coordinate's (S1's `over_median`); S = the coordinates
  at r ≥ 100 at any state at any step (the union); M = the next |S| outside S by their largest r;
  20 random sets of |S| outside S. Written to `set.json` before any projected energy exists;
- **pass2**: per state, E₁ uncentred and token-centred (`c2_aniso`'s two energies) on the cloud
  with each set's coordinates zeroed before normalising, for S, M and the random sets, and
  unprojected (the precision check); ‖m‖² and each set's ‖m_set‖² on the unprojected cloud;
- **preflight**: on seed 0's S, seed 0's off-grid step 7000 and unit 3s's 3 re-inits: τ_p and τ_M
  (the scale-matched arms) and unit 3s's centred gate on the projected counts;
- **read**: R0, R0′, R0b, P1, P2 and the tally.

Tier 1, description (nothing here is registered).
    python -m trackC_writeup.c5_massive pass1 --out <dir> --seeds 0      # HF_HOME=<main>/data/hf
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c5_massive pass1 --out <dir> --seeds 1 .. 9
    python -m trackC_writeup.c5_massive select --out <dir>
    python -m trackC_writeup.c5_massive preflight --out <dir>               # HF_HOME=<main>/data/hf
    python -m trackC_writeup.c5_massive pass2 --out <dir> --seeds ...       # as pass1
    python -m trackC_writeup.c5_massive read --out <dir>
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import REL_TOL, TOKENIZER, n_viol, repo_for
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_seeds as cs

SEEDS = tuple(range(1, 10))                     # scored; seed 0 is the development set
ALL_SEEDS = (0,) + SEEDS
N_STATES = 24                                   # states 0–23, the block inputs S1 read
R_MIN = 100.0                                   # S: r ≥ 100× the median coordinate's
MAX_S = 16                                      # more than this: refuse for that seed
N_RAND = 20
SEED = 20261023                                 # the random sets: default_rng(SEED + seed)
KNOWN_REL = 1e-3                                # S1's top-3 ratios at seed 0's 143000
MIN_NORM = c.MIN_NORM
P1_FIXED = (1e-4, 1e-3, 1e-2)                   # A5's D2 arms
TAU_C = cs.TAU_ARMS                             # unit 3s's centred arms
RAND_P2 = 7.58e-4                               # the B-matched arm: the random sets' P2 arm
TAU_P: Optional[float] = 1.15e-3                # 1e-3 × s_p / s_u at seed 0's 7000 (preflight, 3 s.f.)
TAU_M: Optional[float] = 1.02e-3                # 1e-3 × s_M / s_u, likewise
EARLY = c.EARLY
SHARE_STEPS = (128, 256, 512, 1000)             # R0′
KEPT = 7                                        # tally: ≥ 7 of 9
SPECIFIC_RAND = 18                              # of 20 random sets keeping their dates
DM0 = 1e-12                                     # R0′: |Δ‖m‖²| below this is not divided
SCALE_STEP = 7000
MASSIVE = "trackS/s1/2026-10-21-u3-massive/massive.json"
C2U3S = "trackC/c2u3s/2026-10-22-run"           # unit 3s's records and reading (code e7a1e159f247)


class M2Refused(RuntimeError):
    pass


@functools.lru_cache(maxsize=None)
def code_sha() -> str:
    """The hash of the code this process loaded: taken once, so an edit to the files during a run
    cannot relabel its later records (unit 1's first pass 1 did; `LESSONS.md`)."""
    h = hashlib.sha256()
    for f in ("trackC_writeup/c5_massive.py", "trackC_writeup/c2_seeds.py", "trackC_writeup/c2_aniso.py",
              "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the projection

def proj_energies(H, idx) -> Tuple[float, float]:
    """E₁ uncentred and token-centred of the cloud H (n, d) with the coordinates ``idx`` zeroed,
    each token then L2-normalised. From the raw float64 Gram by the rank-|idx| correction
    G − H_S H_Sᵀ; the centred Gram is K − r1ᵀ − 1rᵀ + s (r the row means of K), re-normalised.
    Works on numpy or torch (float64)."""
    import torch
    H = torch.as_tensor(H, dtype=torch.float64)
    G = H @ H.T
    if len(idx):
        HS = H[:, torch.as_tensor(list(idx), device=H.device)]
        G = G - HS @ HS.T
    return _energies_from_gram(G)


def _energies_from_gram(G) -> Tuple[float, float]:
    import torch
    n = G.shape[0]
    d = torch.diagonal(G)
    if float(d.min()) < MIN_NORM ** 2:
        raise M2Refused(f"a projected token of norm {float(d.min()) ** 0.5:.3g} < {MIN_NORM}")
    s = d.sqrt()
    K = G / (s[:, None] * s[None, :])
    r = K.mean(dim=1)
    C = K - r[:, None] - r[None, :] + r.mean()
    dc = torch.diagonal(C)
    if float(dc.min()) < MIN_NORM ** 2:
        raise M2Refused(f"a centred token of norm {float(dc.min()) ** 0.5:.3g} < {MIN_NORM}")
    sc = dc.sqrt()
    Kc = C / (sc[:, None] * sc[None, :])
    return (float(torch.exp(K).sum() / (2 * n * n)), float(torch.exp(Kc).sum() / (2 * n * n)))


def e1_a5(h: np.ndarray) -> float:
    """A5's E₁ exactly as `c2_aniso.state_row` computes its e1 (float32 Gram of the normalised states)."""
    import torch
    from core.metrics import interaction_energies_batched
    from core.models import layernorm_to_sphere
    U32 = layernorm_to_sphere(torch.from_numpy(h)).numpy()
    return interaction_energies_batched(U32 @ U32.T, [1.0])[1.0]


def state_quantities(h, sets: Dict[str, Sequence[int]]) -> dict:
    """One state: per set (and 'none') E₁ uncentred and centred; ‖m‖² of the unprojected
    normalised cloud and each set's ‖m_set‖² (m restricted to the set's coordinates)."""
    import torch
    H = torch.as_tensor(h, dtype=torch.float64)
    G = H @ H.T
    out = {"e": {}}
    for name, idx in {"none": [], **sets}.items():
        Gp = G
        if len(idx):
            HS = H[:, torch.as_tensor(list(idx), device=H.device)]
            Gp = G - HS @ HS.T
        out["e"][name] = _energies_from_gram(Gp)
    U = H / H.norm(dim=1, keepdim=True)
    m = U.mean(dim=0)
    out["m2"] = float(m @ m)
    out["m2_set"] = {name: float(m[list(idx)] @ m[list(idx)]) for name, idx in sets.items()}
    return out


# ---------------------------------------------------------------- pure: the sets

def ratios(maxabs: np.ndarray) -> np.ndarray:
    """S1's over_median per state: each coordinate's max |x| over the median coordinate's."""
    maxabs = np.asarray(maxabs, float)
    return maxabs / np.median(maxabs, axis=-1, keepdims=True)


def top3(row: np.ndarray) -> List[Tuple[int, float]]:
    r = ratios(row)
    return [(int(i), float(r[i])) for i in np.argsort(-r)[:3]]


def select_sets(r_by_step: Dict[int, np.ndarray], seed: int) -> dict:
    """S, M, the random sets and R0b's inputs from pass 1's ratios (step → (24, 1024))."""
    steps = sorted(r_by_step)
    above = {s: sorted({int(i) for i in np.nonzero((r_by_step[s] >= R_MIN).any(axis=0))[0]}) for s in steps}
    S = sorted(set().union(*above.values()))
    d = next(iter(r_by_step.values())).shape[-1]
    rmax = np.max([r_by_step[s].max(axis=0) for s in steps], axis=0)
    out = {"seed": seed, "S": S, "r_min": R_MIN, "above_by_step": {str(s): above[s] for s in steps},
           "S_max_ratio_by_step": {str(s): (float(r_by_step[s][:, S].max()) if S else None) for s in steps},
           "rmax_top20": [(int(i), float(rmax[i])) for i in np.argsort(-rmax)[:20]],
           "R0b_step": next((s for s in steps if above[s]), None)}
    if not S or len(S) > MAX_S:
        return {**out, "refused": f"|S| = {len(S)}: empty or more than {MAX_S}"}
    rest = np.array([i for i in range(d) if i not in set(S)])
    M = sorted(int(i) for i in rest[np.argsort(-rmax[rest], kind="stable")[:len(S)]])
    rng = np.random.default_rng(SEED + seed)
    R = [sorted(int(i) for i in rng.choice(rest, size=len(S), replace=False)) for _ in range(N_RAND)]
    return {**out, "refused": None, "M": M, "M_rmax": [float(rmax[i]) for i in M], "random": R}


def set_names(sel: dict) -> Dict[str, List[int]]:
    return {"S": sel["S"], "M": sel["M"], **{f"R{k:02d}": r for k, r in enumerate(sel["random"])}}


# ---------------------------------------------------------------- pure: the readings

def date(v: np.ndarray) -> dict:
    """`a3.d4a`'s half-rise; unreadable if the plateau is 0 or a step 0–128 count ≥ half of it."""
    v = np.asarray(v, float)
    r = a3.d4a(v)
    p = r["plateau_median"]
    bad = p <= 0 or max(float(v[STEPS.index(s)]) for s in EARLY) >= p / 2
    return {"step": r["step"], "plateau": p, "unreadable": bool(bad)}


def compare(proj: dict, ref: dict) -> str:
    """One arm: 'kept' (same step), 'later', 'earlier', or 'unreadable' (either side)."""
    if proj["unreadable"] or ref["unreadable"]:
        return "unreadable"
    return "kept" if proj["step"] == ref["step"] else "later" if proj["step"] > ref["step"] else "earlier"


def label(cmp: Dict[str, str]) -> dict:
    """P1 / P2 per seed: kept at every arm, moved (later) at every arm, else mixed; any arm
    unreadable reads mixed, flagged."""
    vals = list(cmp.values())
    lab = "kept" if all(v == "kept" for v in vals) else "moved" if all(v == "later" for v in vals) else "mixed"
    return {"label": lab, "flagged": "unreadable" in vals, "arms": cmp}


def specific(m_cmp: Dict[str, str], rand_kept: int) -> bool:
    """'Moved' is the set's only if M keeps its own dates at every arm and ≥ 18 of 20 random sets
    keep theirs."""
    return all(v == "kept" for v in m_cmp.values()) and rand_kept >= SPECIFIC_RAND


def tally(per: Dict[int, dict]) -> dict:
    kk = int(sum(r["label"] == "kept" for r in per.values()))
    km = int(sum(r["label"] == "moved" and r["specific"] for r in per.values()))
    lab = "kept" if kk >= KEPT else "the coordinates'" if km >= KEPT else "mixed"
    return {"K_kept": kk, "K_moved_specific": km, "n": len(per), "label": lab,
            "flagged": sorted(s for s, r in per.items() if r["flagged"])}


def shares(m2: np.ndarray, m2s: np.ndarray, e1: np.ndarray, tol: float = REL_TOL) -> Tuple[List[float], int]:
    """R0′ for one (prompt, step): at each transition whose E₁ drops by more than ``tol``, the
    signed share Δ‖m_S‖² / Δ‖m‖²; the number of drops with |Δ‖m‖²| < DM0 beside."""
    e1, m2, m2s = (np.asarray(x, float) for x in (e1, m2, m2s))
    drop = (e1[:-1] - e1[1:]) / np.maximum(np.abs(e1[:-1]), 1e-12) > tol
    dm, dms = np.diff(m2), np.diff(m2s)
    out, skipped = [], 0
    for t in np.nonzero(drop)[0]:
        if abs(dm[t]) < DM0:
            skipped += 1
        else:
            out.append(float(dms[t] / dm[t]))
    return out, skipped


# ---------------------------------------------------------------- model passes

def _seed_dir(out: Path, seed: int) -> Path:
    return out / f"seed{seed}"


def _p1_path(out: Path, seed: int, step: int) -> Path:
    return _seed_dir(out, seed) / "pass1" / f"step{step}.npz"


def _rec_path(out: Path, seed: int, step: int, prompt: str) -> Path:
    return _seed_dir(out, seed) / "records" / f"step{step}_{prompt}.json"


def _hidden(model, tok, prompt: str, device):
    """The 25 hidden states (n, d) as float32 torch tensors on the device, and the token ids."""
    import torch
    from core.config import PROMPTS
    enc = tok(PROMPTS[prompt], return_tensors="pt", truncation=True, max_length=c.MAX_TOKENS)
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    return [h[0].to(torch.float32) for h in o.hidden_states], enc["input_ids"][0].tolist()


def _meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C5.md", "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "tokenizer": list(TOKENIZER), "dtype": "float32 forward, float64 energies",
            "device": torch.cuda.get_device_name(0), "tf32": False, "torch": torch.__version__,
            "hf_home": os.environ.get("HF_HOME"), "tau_p": TAU_P, "tau_m": TAU_M, "tau_c": list(TAU_C)}


def pass1(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for seed in a.seeds:
        _p1_path(out, seed, 0).parent.mkdir(parents=True, exist_ok=True)
        (_seed_dir(out, seed) / "pass1" / "plan.json").write_text(json.dumps(_meta(torch), indent=1) + "\n")
        for step in STEPS:
            f = _p1_path(out, seed, step)
            if f.exists():
                continue
            model = cs.load(repo_for(seed), step, device)
            mx = {k: torch.zeros(N_STATES, 1024, dtype=torch.float64, device=device) for k in ("all", "rest", "pos0")}
            for p in V1:
                hs, _ = _hidden(model, tok, p, device)
                for l in range(N_STATES):
                    x = hs[l].abs().double()
                    mx["rest"][l] = torch.maximum(mx["rest"][l], x[1:].max(0).values)
                    mx["pos0"][l] = torch.maximum(mx["pos0"][l], x[0])
                    mx["all"][l] = torch.maximum(mx["all"][l], x.max(0).values)
            arr = {k: v.cpu().numpy() for k, v in mx.items()}
            if not all(np.all(np.isfinite(v)) and v.min() >= 0 and np.median(v) > 0 for v in arr.values()):
                raise M2Refused(f"seed {seed} step {step}: pass 1 not populated")
            np.savez(f, **arr, code=code_sha(), seed=seed, step=step)
            if first:
                r = ratios(arr["all"])
                print(f"first pass-1 record populated: seed {seed} step {step}, max r {r.max():.1f} "
                      f"at state {int(r.max(1).argmax())}", flush=True)
                first = False
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} pass 1 done ({time.time() - t0:.0f} s)", flush=True)
    return 0


def known_answer_s1(out: Path) -> dict:
    """Seed 0's 143000 against S1's `massive.json`: per state, both top-3 lists equal in coordinates
    and within KNOWN_REL in ratio."""
    ref = json.loads((DATA / MASSIVE).read_text())
    z = np.load(_p1_path(out, 0, 143000))
    worst = 0.0
    for l, L in enumerate(ref["layers"]):
        for half in ("rest", "pos0"):
            mine, theirs = top3(z[half][l]), [(x["coord"], x["over_median"]) for x in L[half]]
            if [i for i, _ in mine] != [i for i, _ in theirs]:
                raise M2Refused(f"S1's known answer: state {l} {half}: {mine} against {theirs}")
            worst = max(worst, max(abs(m - t) / t for (_, m), (_, t) in zip(mine, theirs)))
    if worst > KNOWN_REL:
        raise M2Refused(f"S1's known answer: a ratio off by {worst:.2g} > {KNOWN_REL}")
    return {"max_rel_dev": worst, "states": len(ref["layers"])}


def select(a) -> int:
    out = Path(a.out)
    ka = known_answer_s1(out)
    print(f"S1's known answer at seed 0's 143000: met (max rel dev {ka['max_rel_dev']:.1e})", flush=True)
    for seed in a.seeds:
        f = _seed_dir(out, seed) / "set.json"
        if f.exists():
            continue
        zs = {s: np.load(_p1_path(out, seed, s)) for s in STEPS}
        codes = {str(z["code"]) for z in zs.values()}
        if len(codes) != 1:
            raise M2Refused(f"seed {seed}: pass 1 from {codes}")
        sel = select_sets({s: ratios(z["all"]) for s, z in zs.items()}, seed)
        sel.update(code=code_sha(), pass1_code=codes.pop(), known_answer_s1=ka if seed == 0 else None)
        f.write_text(json.dumps(sel, indent=1) + "\n")
        print(f"seed {seed}: S {sel['S']} M {sel.get('M')} R0b {sel['R0b_step']} refused {sel['refused']}",
              flush=True)
    return 0


def _frozen() -> None:
    if TAU_P is None or TAU_M is None:
        raise M2Refused("TAU_P / TAU_M are not set: the preflight has not frozen the rule")


def pass2(a) -> int:
    _frozen()
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for seed in a.seeds:
        sel = json.loads((_seed_dir(out, seed) / "set.json").read_text())
        if sel["refused"]:
            print(f"seed {seed}: refused at select ({sel['refused']}); no pass 2", flush=True)
            continue
        sets = set_names(sel)
        ref_dir = (DATA / cs.A5_SEEDS / f"seed{seed}" / "records" if seed else DATA / c.A5_SEED0 / "records")
        c_dir = (DATA / C2U3S / f"seed{seed}" / "records" if seed else DATA / cs.C2U3 / "records")
        _rec_path(out, seed, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        (_seed_dir(out, seed) / "records" / "plan.json").write_text(
            json.dumps({**_meta(torch), "sets": sets}, indent=1) + "\n")
        for step in STEPS:
            todo = [p for p in V1 if not _rec_path(out, seed, step, p).exists()]
            if not todo:
                continue
            model = cs.load(repo_for(seed), step, device)
            for p in todo:
                hs, ids = _hidden(model, tok, p, device)
                e1 = [e1_a5(h.cpu().numpy()) for h in hs]
                rows = [state_quantities(h, sets) for h in hs]
                rec = {"seed": seed, "step": step, "prompt": p, "n": len(ids), "code": code_sha(),
                       "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:12], "e1": e1,
                       "m2": [r["m2"] for r in rows],
                       "m2_set": {k: [r["m2_set"][k] for r in rows] for k in sets},
                       "eu": {k: [r["e"][k][0] for r in rows] for k in ("none", *sets)},
                       "ec": {k: [r["e"][k][1] for r in rows] for k in ("none", *sets)}}
                ref = json.loads((ref_dir / f"step{step}_{p}.json").read_text())
                cref = json.loads((c_dir / f"step{step}_{p}.json").read_text())
                if rec["ids_sha"] != ref["ids_sha"] or rec["ids_sha"] != cref["ids_sha"]:
                    raise M2Refused(f"seed {seed} step {step} {p}: tokens differ from A5's / 3s's record")
                dev = float(np.max(np.abs(np.subtract(e1, ref["e1"])) / np.asarray(ref["e1"])))
                devc = float(np.max(np.abs(np.subtract(rec["ec"]["none"], cref["e1c"])) / np.asarray(cref["e1c"])))
                if dev > c.ENERGY_TOL or devc > c.ENERGY_TOL:
                    raise M2Refused(f"seed {seed} step {step} {p}: e1 off A5's by {dev:.2g}, "
                                    f"e1c off 3s's by {devc:.2g} (> {c.ENERGY_TOL})")
                rec.update(e1_dev_a5=dev, e1c_dev_3s=devc)
                bad = [k for k in ("e1", "m2") if len(rec[k]) != 25 or not np.all(np.isfinite(rec[k]))]
                bad += [f"{w}:{k}" for w in ("eu", "ec") for k, v in rec[w].items()
                        if len(v) != 25 or not np.all(np.isfinite(v)) or min(v) <= 0]
                if bad:
                    raise M2Refused(f"seed {seed} step {step} {p}: not populated: {bad}")
                _rec_path(out, seed, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: seed {seed} step {step} {p}, n {rec['n']}, e1 dev {dev:.1e}, "
                          f"e1c dev {devc:.1e}, eu S[12] {rec['eu']['S'][12]:.6f}, "
                          f"ec S[12] {rec['ec']['S'][12]:.6f}, q[12] {rec['m2_set']['S'][12] / rec['m2'][12]:.3f}",
                          flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} pass 2 done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- the preflight

def _proj_units(hs, S: Sequence[int], M: Sequence[int], uS: np.ndarray, u: np.ndarray,
                rng: np.random.Generator, device=None) -> dict:
    """One prompt's 25 states: relative drops per transition of the uncentred energy unprojected,
    S-projected and M-projected; of the S-projected centred energy; and unit 3s's plants (A on the
    earlier state, B on the later) read through the S-projected energies. A's vector ``uS`` is
    supported on S; B's spread is zeroed on S; A with unit 3s's full ``u`` beside."""
    import torch

    def pe(h, idx):                                         # proj_energies on the device
        return proj_energies(torch.as_tensor(h, dtype=torch.float64, device=device), idx)

    H = [np.asarray(h, np.float64) for h in hs]
    eS = [pe(h, S) for h in H]
    base = {"u": c.rel_drops([e[0] for e in eS]).tolist(), "c": c.rel_drops([e[1] for e in eS]).tolist(),
            "u_none": c.rel_drops([pe(h, [])[0] for h in H]).tolist(),
            "u_M": c.rel_drops([pe(h, M)[0] for h in H]).tolist()}
    keep = np.ones(H[0].shape[1])
    keep[list(S)] = 0.0
    out: Dict[str, Dict[str, List[float]]] = {k: {} for k in ("A_u", "A_c", "Afull_c", "B_u", "B_c")}
    for d in cs.DOSES_A:
        k = f"{d:g}"
        for key in ("A_u", "A_c", "Afull_c"):
            out[key][k] = []
        for t in range(len(H) - 1):
            ea = pe(c.plant_common(H[t], d, uS), S)
            out["A_u"][k].append((ea[0] - eS[t + 1][0]) / abs(ea[0]))
            out["A_c"][k].append((ea[1] - eS[t + 1][1]) / abs(ea[1]))
            ef = pe(c.plant_common(H[t], d, u), S)[1]
            out["Afull_c"][k].append((ef - eS[t + 1][1]) / abs(ef))
    for d in cs.DOSES_B:
        k = f"{d:g}"
        out["B_u"][k], out["B_c"][k] = [], []
        for t in range(len(H) - 1):
            m = float(np.median(np.linalg.norm(H[t + 1], axis=1)))
            g = rng.standard_normal(H[t + 1].shape) / np.sqrt(H[t + 1].shape[1]) * keep[None, :]
            eb = pe(H[t + 1] + d * m * g, S)
            out["B_u"][k].append((eS[t][0] - eb[0]) / abs(eS[t][0]))
            out["B_c"][k].append((eS[t][1] - eb[1]) / abs(eS[t][1]))
    return {"base": base, "plants": out}


def preflight(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import init_summary, reinit_model
    out = Path(a.out)
    pf = out / "preflight"
    pf.mkdir(parents=True, exist_ok=True)
    sel = json.loads((_seed_dir(out, 0) / "set.json").read_text())
    if sel["refused"]:
        raise M2Refused(f"seed 0's set refused: {sel['refused']}")
    S, M = sel["S"], sel["M"]
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rng = np.random.default_rng(cs.SEED)
    u = rng.standard_normal(1024)
    u /= np.linalg.norm(u)                                  # unit 3s's u, drawn first as there
    uS = np.zeros(1024)
    uS[S] = u[S]
    uS /= np.linalg.norm(uS)
    names = [f"reinit:{s}" for s in range(c.N_REINIT)] + [f"step{SCALE_STEP}"]
    pops: Dict[str, dict] = {}
    for name in names:
        if name.startswith("reinit"):
            cfg = AutoConfig.from_pretrained(c.MODEL, revision="step0")
            model = reinit_model(cfg, int(name.split(":")[1])).to(device)
            init = init_summary(model)
        else:
            model, init = cs.load(c.MODEL, SCALE_STEP, device), None
        per = {}
        for p in V1:
            hs, _ = _hidden(model, tok, p, device)
            per[p] = _proj_units([h.cpu().numpy() for h in hs], S, M, uS, u, rng, device)
        pops[name] = {"per_prompt": per, "init_summary": init}
        del model
        torch.cuda.empty_cache()
        print(f"{name} done", flush=True)

    def flat(ns, part, key, dose=None):
        return [x for n in ns for p in V1
                for x in (pops[n]["per_prompt"][p][part][key] if dose is None
                          else pops[n]["per_prompt"][p][part][key][dose])]

    reinits, trained = names[:-1], names[-1:]
    units = {pop: {"base": {w: flat(ns, "base", w) for w in ("u", "c", "u_none", "u_M")},
                   "plants": {k: {f"{d:g}": flat(ns, "plants", k, f"{d:g}")
                                  for d in (cs.DOSES_B if k[0] == "B" else cs.DOSES_A)}
                              for k in ("A_u", "A_c", "Afull_c", "B_u", "B_c")}}
             for pop, ns in (("reinit", reinits), ("trained", trained))}
    (pf / "preflight_units.json").write_text(json.dumps(units) + "\n")
    tr, nu = units["trained"], units["reinit"]
    s_u, s_p, s_m = (cs.scale(tr["base"][w]) for w in ("u_none", "u", "u_M"))
    derived = {"s_u": s_u, "s_p": s_p, "s_m": s_m, "tau_p": REL_TOL * s_p / s_u, "tau_m": REL_TOL * s_m / s_u}
    again = (float(f"{derived['tau_p']:.3g}"), float(f"{derived['tau_m']:.3g}"))
    if TAU_P is not None and (TAU_P, TAU_M) != again:
        raise M2Refused(f"the preflight derives {again}, not the frozen {(TAU_P, TAU_M)}")
    gate = cs.gate_arms(units, TAU_C)
    p1_null = {f"{t:.3g}": cs.rate(nu["base"]["u"], t) for t in (*P1_FIXED, again[0])}
    tool_in = {"statistic": "S-projected centred E_1 relative drop per (prompt, transition), "
                            "(E_t − E_t+1) / |E_t| (c5_massive, seed 0's S)",
               "direction": "greater", "nulls": {"reinit": nu["base"]["c"]}, "moved_marks": [],
               "trained": tr["base"]["c"], "trained_step": str(SCALE_STEP),
               "scored_steps": [str(s) for s in STEPS],
               "planted": {f"{d:g}": nu["plants"]["B_c"][f"{d:g}"] for d in cs.DOSES_B},
               "bounds": None, "decision_p": 0.05}
    (pf / "preflight_input.json").write_text(json.dumps(tool_in) + "\n")
    rec = {"code": code_sha(), "meta": _meta(torch), "S": S, "M": M, "derived": derived, "arms": again,
           "frozen": TAU_P is not None, "gate": gate, "p1_reinit_rates_uncentred_projected": p1_null,
           "A_in_S_max_abs_shift": max(abs(x - y) for d in cs.DOSES_A
                                       for x, y in zip(tr["plants"]["A_c"][f"{d:g}"], tr["plants"]["A_c"]["0"])),
           "Afull_share_beside": {f"{t:.3g}": cs.share(cs.rate(tr["plants"]["Afull_c"][cs.A_DOSE], t),
                                                       cs.rate(tr["base"]["c"], t)) for t in TAU_C},
           "init_summary": {n: pops[n]["init_summary"] for n in reinits}}
    (pf / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("derived", "arms", "gate", "p1_reinit_rates_uncentred_projected",
                                          "A_in_S_max_abs_shift", "Afull_share_beside")}, indent=1))
    return 0 if gate["pass"] else 1


# ---------------------------------------------------------------- reading

def _load(out: Path, seed: int) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = _rec_path(out, seed, s, p)
            if not f.exists():
                raise M2Refused(f"{f}: missing")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def vser(recs, which: str, name: Optional[str], tol: float) -> np.ndarray:
    """V on the 18 steps: violations summed over the 7 prompts; ``which`` 'e1' (A5's path), 'eu' or 'ec'."""
    return np.array([sum(n_viol(recs[(s, p)][which] if name is None else recs[(s, p)][which][name], tol)
                         for p in V1) for s in STEPS], float)


def seed_reading(recs, sel: dict, ref_c: dict) -> dict:
    """P1, P2, R0, R0′ and the scale ratios for one seed. ``ref_c`` unit 3s's per-arm centred dates."""
    names = set_names(sel)
    rand = [k for k in names if k.startswith("R")]
    # P1: each arm (projected tolerance, unprojected tolerance)
    p1_arms = {f"{t:g}": (t, t) for t in P1_FIXED}
    ref_u = {k: date(vser(recs, "e1", None, tu)) for k, (_, tu) in p1_arms.items()}
    ref_u["scale"] = ref_u["0.001"]

    def p1_cmp(name, tau_scale):
        arms = {**p1_arms, "scale": (tau_scale, REL_TOL)}
        return {k: compare(date(vser(recs, "eu", name, tp)), ref_u[k]) for k, (tp, _) in arms.items()}

    p1 = label(p1_cmp("S", TAU_P))
    p1_m = p1_cmp("M", TAU_M)
    p1_r = sum(compare(date(vser(recs, "eu", k, REL_TOL)), ref_u["0.001"]) == "kept" for k in rand)
    p1.update(specific=specific(p1_m, p1_r), M_arms=p1_m, random_kept=int(p1_r),
              dates={k: date(vser(recs, "eu", "S", tp))["step"] for k, (tp, _) in
                     {**p1_arms, "scale": (TAU_P, REL_TOL)}.items()},
              ref_dates={k: v["step"] for k, v in ref_u.items()})
    # P2: unit 3s's arms; the unprojected dates are 3s's
    ref_c2 = {f"{t:.3g}": {"step": ref_c[f"{t:.3g}"]["centred_step"],
                           "unreadable": ref_c[f"{t:.3g}"]["unreadable"]} for t in TAU_C}
    p2 = label({k: compare(date(vser(recs, "ec", "S", t)), ref_c2[k]) for k, t in zip(ref_c2, TAU_C)})
    p2_m = {k: compare(date(vser(recs, "ec", "M", t)), ref_c2[k]) for k, t in zip(ref_c2, TAU_C)}
    kr = f"{RAND_P2:.3g}"
    p2_r = sum(compare(date(vser(recs, "ec", k, RAND_P2)), ref_c2[kr]) == "kept" for k in rand)
    p2.update(specific=specific(p2_m, p2_r), M_arms=p2_m, random_kept=int(p2_r),
              dates={k: date(vser(recs, "ec", "S", t))["step"] for k, t in zip(ref_c2, TAU_C)},
              ref_dates={k: v["step"] for k, v in ref_c2.items()})
    # R0: q per (step, state), prompt-averaged; R0′ at steps 128–1000
    r0 = {k: {str(s): np.mean([np.asarray(recs[(s, p)]["m2_set"][k]) / np.asarray(recs[(s, p)]["m2"])
                               for p in V1], axis=0).tolist() for s in STEPS}
          for k in ("S", "M")}
    r0["random_mean"] = {str(s): np.mean([np.asarray(recs[(s, p)]["m2_set"][k]) / np.asarray(recs[(s, p)]["m2"])
                                          for p in V1 for k in rand], axis=0).tolist() for s in STEPS}
    r0p = {}
    for k in ("S", "M", "R00"):
        r0p[k] = {}
        for s in SHARE_STEPS:
            vals, skipped = [], 0
            for p in V1:
                r = recs[(s, p)]
                v, sk = shares(r["m2"], r["m2_set"][k], r["e1"])
                vals += v
                skipped += sk
            r0p[k][str(s)] = {"n": len(vals), "skipped": skipped,
                              "median": float(np.median(vals)) if vals else None,
                              "frac_above_half": float(np.mean(np.asarray(vals) > 0.5)) if vals else None}
    ratio = {str(s): cs.scale(np.concatenate([c.rel_drops(recs[(s, p)]["eu"]["S"]) for p in V1]))
             / cs.scale(np.concatenate([c.rel_drops(recs[(s, p)]["eu"]["none"]) for p in V1])) for s in STEPS}
    precision = {f"{t:g}": [int(x) for x in vser(recs, "eu", "none", t) - vser(recs, "e1", None, t)]
                 for t in P1_FIXED}
    return {"P1": p1, "P2": p2, "R0": r0, "R0prime": r0p, "R0b_step": sel["R0b_step"],
            "S": sel["S"], "M": sel["M"], "S_max_ratio_by_step": sel["S_max_ratio_by_step"],
            "s_p_over_s_u_by_step": ratio, "precision_V_none_minus_e1": precision}


def read(a) -> int:
    _frozen()
    out = Path(a.out)
    score = json.loads((DATA / cs.A5_SEEDS / "score.json").read_text())
    r3s = json.loads((DATA / C2U3S / "reading.json").read_text())
    per, refused, codes = {}, {}, {}
    for seed in ALL_SEEDS:
        sel = json.loads((_seed_dir(out, seed) / "set.json").read_text())
        if sel["refused"]:
            refused[seed] = sel["refused"]
            continue
        recs = _load(out, seed)
        codes[seed] = sorted({r["code"] for r in recs.values()})
        vu = vser(recs, "e1", None, REL_TOL)
        if seed and vu.tolist() != score["notes"][str(seed)]["V"]:
            raise M2Refused(f"seed {seed}: V from e1 is not A5's ({vu.tolist()})")
        ref_c = (r3s["per_seed"][str(seed)] if seed else r3s["seed0_development"])["arms"]
        per[seed] = seed_reading(recs, sel, ref_c)
    if len({tuple(v) for v in codes.values()}) != 1:
        raise M2Refused(f"records from more than one code version: {codes}")
    dev = per.pop(0, None)
    res = {"code": code_sha(), "record_codes": codes, "tau_p": TAU_P, "tau_m": TAU_M, "tau_c": list(TAU_C),
           "refused": refused,
           "tally_P1": tally({s: r["P1"] for s, r in per.items()}),
           "tally_P2": tally({s: r["P2"] for s, r in per.items()}),
           "per_seed": per, "seed0_development": dev}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    for s, r in sorted(per.items()):
        print(s, "S", r["S"], "R0b", r["R0b_step"], "| P1", r["P1"]["label"], r["P1"]["dates"], r["P1"]["ref_dates"],
              "spec", r["P1"]["specific"], "| P2", r["P2"]["label"], r["P2"]["dates"], r["P2"]["ref_dates"],
              "spec", r["P2"]["specific"])
    print(json.dumps({"P1": res["tally_P1"], "P2": res["tally_P2"], "refused": refused}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("pass1", "select", "preflight", "pass2", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd in ("pass1", "select", "pass2"):
            sp.add_argument("--seeds", type=int, nargs="+", default=list(ALL_SEEDS))
    a = ap.parse_args(argv)
    return {"pass1": pass1, "select": select, "preflight": preflight, "pass2": pass2, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
