"""
trackC_writeup/c4_read.py — C4: the records, V's gate and the reading on the two trained arms
(`design-C4.md`; frozen on Blocked 155 all (a)). The rule is the pure part below, written and
tested in U1 before any arm passes step 64 and before the user registers it (D4 (a)).

**Records** (`records`): C2 unit 3b's per-state row (`c2_dose.record`: e1, e1c, e1r, cos) on the
7 v1 prompts, float32 eager, for each saved checkpoint of an arm; the loader is checked against
unit 3b's records of the released `pythia-70m` through this file's own path (`known`).

**The counts**: V = the uncentred count at 1e-3 (`c2_aniso.n_viol` summed over the 7 prompts);
the centred count at each of 70m's four frozen τ arms (`c2_dose.TAU_ARMS["70m"]`, as C6 read
them). Each arm's series is read on the shared dense grid (`c4_train.GRID`).

**The date** (`date_on`): `a3.d4a`'s half-rise on the grid: plateau = the median at 1000, 2000,
4000; the date is the first grid step whose count is ≥ half of it. A plateau ≤ 0 is "not formed
by 4000" (a reading). A step-0 count ≥ half the plateau is unreadable (the init already holds
it). 92 (i)'s early check (any step ≤ 128 at half) is reported beside, **not** applied: on F
the cumulative hypothesis predicts V's date at ≈ 23, so applying it would refuse the very
reading that hypothesis predicts.

**The label** (`label`): for A0's date t (≤ W), the tolerance ρ = max(1.25, t / prev(t),
next(t) / t) on the dense grid (no A0 rerun exists); hypothesis h is consistent when F's date
lies in [m_h(t/ρ), m_h(t ρ)] with m_step(t) = t and m_cumulative(t) = t² / 2W (the flat maps,
`tools/math_checks/c4_flat_maps.py`); the label is "consistent with step", "consistent with
cumulative", "neither" or "both". The maps hold on the warmup only, so an A0 date with t ρ > W,
or whose two bands meet, labels nothing ("A0 past the separable range"; `/challenge-pr` on #346,
finding 5). For the centred count (C6's form), h is consistent when it is consistent at every τ
arm that is gated on F and dated on both arms; no such arm: unreadable.

**Divergence** (`divergence`): either arm's training losses (`c4_train.diverged`: 50
consecutive losses over 2× the running median) diverging is a refusal; nothing is read
(`gate` and `read`; finding 2).

**R1** (`r1`): A0's date against the released `pythia-70m`'s (unit 3b's seed-0 records), both on
Pythia's grid to 4000; A0 must lie within the released date's grid step ± 1 (per count, per τ
arm), else a refusal and F is not read.

**V's gate** (`gate`, U2): C6's gate (`c6_warmup.gate_one`, plant B's dose by rule 100 (a), the
same plants and bars) on F at the off-grid step 3500, with 70m's 3 re-inits; the re-inits' rates
must equal unit 3b's preflight. V is read only if 100 (a) finds B's dose; a centred τ arm only if
it passes. If 100 (a) refuses, the centred arms are gated at plant B's fixed dose 0.5 (C6 U1b's
form, `c6_warmup.U1B_B_DOSE`, which must be unit 3b's 70m dose) and V is not read (design
"Readings"; finding 1).

**Beside** (`induction`): the largest per-head induction score at each checkpoint, on 8
sequences of 64 random tokens repeated twice (seeded): the mean attention from each second-copy
position to the token after its first occurrence. Defined and run on the released `pythia-70m`
before any arm is read.

Tier 1 until the user registers the reading.
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read known --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read records --out <dir> --arm A0 --upto 1000
    python -m trackC_writeup.c4_read h6 --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read h4-controls --out <dir>   # beside
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read induction --out <dir> --model released
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read gate --out <dir>        # U2
    python -m trackC_writeup.c4_read read --out <dir>                                     # U3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackA_force.a5_seeds import REL_TOL, TOKENIZER
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as cd
from trackC_writeup import c4_train as ct

GRID = ct.GRID
PYTHIA = tuple(s for s in STEPS if s <= ct.HORIZON)       # 0, 2, 4, …, 4000 (13 steps)
PLATEAU = (1000, 2000, 4000)                              # a3.d4a's
RHO_MIN = 1.25                                            # placed (design)
TAU = cd.TAU_ARMS["70m"]                                  # 70m's four frozen centred arms
KEYS = ("u",) + tuple(f"{t:.3g}" for t in TAU)
HYPS = ("step", "cumulative")
GATE_STEP = 3500
SEED = 20261109
U3B_70M = DATA / "trackC/c2u3b/2026-10-22-run/70m/records"
U3B_PREFLIGHT = DATA / "trackC/c2u3b/2026-10-22-preflight/70m/preflight_plants.json"
IND_LEN, IND_N, IND_SEED = 64, 8, 20261109


class C4ReadRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4_read.py", "trackC_writeup/c2_dose.py", "trackC_writeup/c2_seeds.py",
              "trackC_writeup/c2_aniso.py", "trackC_writeup/c6_warmup.py", "trackA_force/a5_seeds.py",
              "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the rule

def date_on(v: Sequence[float], grid: Sequence[int] = GRID) -> dict:
    """The half-rise on ``grid`` (``v`` aligned with it)."""
    v = np.asarray(v, float)
    if len(v) != len(grid):
        raise C4ReadRefused(f"{len(v)} values on a grid of {len(grid)}")
    p = float(np.median([v[list(grid).index(s)] for s in PLATEAU]))
    early = {int(s): float(x) for s, x in zip(grid, v) if s <= 128}
    beside = {"early_92i_flag": bool(p > 0 and max(early.values()) >= p / 2), "early": early}
    if p <= 0:
        return {"step": None, "plateau": p, "status": "not formed by 4000", **beside}
    if v[0] >= p / 2:
        return {"step": None, "plateau": p, "status": "unreadable (step 0 at half)", **beside}
    i = int(np.argmax(v >= p / 2))
    return {"step": int(grid[i]), "plateau": p, "status": "dated", **beside}


def rho_at(t: int, grid: Sequence[int] = GRID) -> float:
    """max(RHO_MIN, the ratios of the grid intervals either side of ``t``)."""
    g = list(grid)
    i = g.index(t)
    r = [RHO_MIN]
    if i > 0 and g[i - 1] > 0:
        r.append(t / g[i - 1])
    if i + 1 < len(g):
        r.append(g[i + 1] / t)
    return float(max(r))


def m(h: str, t: float) -> float:
    """A0's date ``t`` mapped to F's under ``h`` (t ≤ W; past W the arms share the cosine)."""
    if t > ct.W:
        raise C4ReadRefused(f"t = {t} > W: the flat map is defined on the warmup only")
    if h == "step":
        return float(t)
    if h == "cumulative":
        return t * t / (2.0 * ct.W)
    raise C4ReadRefused(f"hypothesis {h}")


def separable(t_a0: int, rho: float) -> bool:
    """The flat maps are defined (t ρ ≤ W) and the two bands are disjoint."""
    return t_a0 * rho <= ct.W and band("cumulative", t_a0, rho)[1] < band("step", t_a0, rho)[0]


def band(h: str, t_a0: int, rho: float) -> Tuple[float, float]:
    return m(h, t_a0 / rho), m(h, min(t_a0 * rho, ct.W))


def consistent(t_f: int, t_a0: int, rho: float) -> Dict[str, bool]:
    return {h: band(h, t_a0, rho)[0] <= t_f <= band(h, t_a0, rho)[1] for h in HYPS}


def name(cons: Dict[str, bool]) -> str:
    s, cu = cons["step"], cons["cumulative"]
    return "both" if s and cu else "consistent with step" if s else "consistent with cumulative" if cu else "neither"


def label(f: dict, a0: dict) -> dict:
    """One count on one τ: F's date against A0's."""
    if a0["status"] != "dated":
        return {"label": f"A0 {a0['status']}", "consistent": None}
    if f["status"] == "not formed by 4000":
        return {"label": "not formed by 4000", "consistent": None}
    if f["status"] != "dated":
        return {"label": f"F {f['status']}", "consistent": None}
    rho = rho_at(a0["step"])
    if not separable(a0["step"], rho):
        return {"label": "A0 past the separable range", "consistent": None, "rho": rho, "t_a0": a0["step"]}
    cons = consistent(f["step"], a0["step"], rho)
    return {"label": name(cons), "consistent": cons, "rho": rho, "t_a0": a0["step"], "t_f": f["step"],
            "bands": {h: band(h, a0["step"], rho) for h in HYPS}}


def centred_label(per_tau: Dict[str, dict], gated: Dict[str, bool]) -> dict:
    """C6's form: h consistent iff consistent at every gated τ arm dated on both arms."""
    used = [k for k, r in per_tau.items() if gated.get(k) and r["consistent"] is not None]
    nf = [k for k, r in per_tau.items() if gated.get(k) and r["label"] == "not formed by 4000"]
    past = [k for k, r in per_tau.items() if gated.get(k) and r["label"] == "A0 past the separable range"]
    if not used:
        return {"label": "not formed by 4000" if nf else "unreadable", "arms": used, "past_arms": past}
    cons = {h: all(per_tau[k]["consistent"][h] for k in used) for h in HYPS}
    return {"label": name(cons), "consistent": cons, "arms": used, "not_formed_arms": nf, "past_arms": past}


def train_losses(lines: Iterable[str]) -> list:
    """An arm's losses by step from its `train_log.jsonl` (a resumed step's last entry wins)."""
    by = {}
    for ln in lines:
        d = json.loads(ln)
        if "step" in d and "loss" in d and "arm" not in d:
            by[int(d["step"])] = float(d["loss"])
    if sorted(by) != list(range(1, len(by) + 1)):
        raise C4ReadRefused(f"the training log's steps are not 1..{len(by)}")
    return [by[s] for s in sorted(by)]


def divergence(root: Path, arms: Sequence[str] = ct.ARMS) -> Dict[str, Optional[int]]:
    """Each arm's first diverged step by `c4_train.diverged` (None: none)."""
    out = {}
    for arm in arms:
        with open(root / arm / "train_log.jsonl") as fh:
            out[arm] = ct.diverged(train_losses(fh))
    return out


def r1_pass(a0_step: Optional[int], rel_step: Optional[int], grid: Sequence[int] = PYTHIA) -> bool:
    """A0's date within the released date's grid step ± 1 on Pythia's grid."""
    if a0_step is None or rel_step is None:
        return False
    g = list(grid)
    return abs(g.index(a0_step) - g.index(rel_step)) <= 1


def series(recs: Dict[Tuple[int, str], dict], key: str, steps: Sequence[int]) -> np.ndarray:
    """A count over ``steps``: violations summed over the 7 prompts (``key`` 'u' or a τ string)."""
    if key == "u":
        return np.array([sum(c.n_viol(recs[(s, p)]["e1"], REL_TOL) for p in V1) for s in steps], float)
    t = float(key)
    return np.array([sum(c.n_viol(recs[(s, p)]["e1c"], t) for p in V1) for s in steps], float)


def all_series(recs, steps) -> Dict[str, np.ndarray]:
    return {k: series(recs, k, steps) for k in KEYS}


def induction_scores(attn: np.ndarray, L: int) -> np.ndarray:
    """Per head, the mean attention from second-copy position i (L ≤ i < 2L, BOS-free) to i − L + 1.
    ``attn`` (layers, heads, n, n) for one sequence of 2L tokens."""
    idx = np.arange(L, 2 * L)
    return attn[:, :, idx, idx - L + 1].mean(-1)


# ---------------------------------------------------------------- records

def rec_path(out: Path, arm: str, step: int, prompt: str) -> Path:
    return out / arm / "records" / f"step{step}_{prompt}.json"


def load_local(path: Path, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    mdl = from_pretrained_eager(GPTNeoXForCausalLM, str(path), torch_dtype=torch.float32).to(device).eval()
    if next(mdl.parameters()).dtype != torch.float32:
        raise C4ReadRefused(f"{path}: not float32")
    return mdl


def _records_of(model, tok, device, out: Path, arm: str, step: int, src: str) -> None:
    n_states = model.config.num_hidden_layers + 1
    for p in V1:
        f = rec_path(out, arm, step, p)
        if f.exists():
            continue
        hs, ids = c.states(model, tok, p, device)
        rec = cd.record(hs, ids, p, n_states)
        rec.update(arm=arm, step=step, source=src, code=code_sha())
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec) + "\n")


def records(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out, root = Path(a.out), Path(a.root)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    steps = [s for s in ct.SAVE_STEPS if s <= a.upto and (a.steps is None or s in a.steps)]
    first = True
    for s in steps:
        d = ct.ckpt_dir(root, a.arm, s)
        if not d.exists():
            raise C4ReadRefused(f"{d}: missing")
        if all(rec_path(out, a.arm, s, p).exists() for p in V1):
            continue
        model = load_local(d, device)
        _records_of(model, tok, device, out, a.arm, s, str(d))
        if first:
            r = json.loads(rec_path(out, a.arm, s, V1[0]).read_text())
            print(f"first record populated: {a.arm} step {s}, n {r['n']}, e1[0] {r['e1'][0]:.6f}, "
                  f"e1c[0] {r['e1c'][0]:.6f}", flush=True)
            first = False
        del model
        torch.cuda.empty_cache()
    print(json.dumps({"arm": a.arm, "steps": steps}), flush=True)
    return 0


def load_arm(out: Path, arm: str, steps: Sequence[int]) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in steps:
        for p in V1:
            f = rec_path(out, arm, s, p)
            if not f.exists():
                raise C4ReadRefused(f"{f}: missing")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def load_released(steps: Sequence[int] = PYTHIA) -> Dict[Tuple[int, str], dict]:
    return {(s, p): json.loads((U3B_70M / f"step{s}_{p}.json").read_text()) for s in steps for p in V1}


def known(a) -> int:
    """The loader's known answer: released `pythia-70m` steps 0 and 1000 saved through
    `c4_train.save_ckpt` and read back through `load_local`, against unit 3b's records
    (|Δe1| / e1 ≤ c2_dose's ENERGY_TOL; the same token ids)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    worst, res = 0.0, {}
    for s in (0, 1000):
        d = out / "known" / "ckpt" / f"step{s}"
        if not d.exists():
            mdl = GPTNeoXForCausalLM.from_pretrained(ct.BASE, revision=f"step{s}", torch_dtype=torch.float32)
            ct.save_ckpt(mdl, d)
            del mdl
        model = load_local(d, device)
        _records_of(model, tok, device, out, "known", s, str(d))
        del model
        for p in V1:
            new = json.loads(rec_path(out, "known", s, p).read_text())
            old = json.loads((U3B_70M / f"step{s}_{p}.json").read_text())
            if new["ids_sha"] != old["ids_sha"]:
                raise C4ReadRefused(f"step {s} {p}: tokens differ from unit 3b's")
            for k in ("e1", "e1c"):
                dd = max(abs(x - y) / abs(y) for x, y in zip(new[k], old[k]) if y != 0)
                worst = max(worst, dd)
        res[str(s)] = "read"
    ok = worst <= c.ENERGY_TOL
    out_ = {"code": code_sha(), "steps": res, "max_rel": worst, "tol": c.ENERGY_TOL, "pass": bool(ok)}
    ct._jdump(out / "known.json", out_)
    print(json.dumps(out_), flush=True)
    return 0 if ok else 1


def h6(a) -> int:
    """A0's counts on Pythia's grid to ``--upto`` against the released run's (report; no bar),
    with R1's dates where the grid reaches 4000."""
    out = Path(a.out)
    steps = [s for s in PYTHIA if s <= a.upto]
    a0 = all_series(load_arm(out, "A0", steps), steps)
    rel = all_series(load_released(steps), steps)
    res = {"code": code_sha(), "steps": steps,
           "A0": {k: v.tolist() for k, v in a0.items()}, "released": {k: v.tolist() for k, v in rel.items()}}
    ct._jdump(out / f"h6_upto{a.upto}.json", res)
    for k in KEYS:
        print(k.ljust(9), "A0 ", " ".join(f"{int(x):4d}" for x in a0[k]))
        print(" " * 9, "rel", " ".join(f"{int(x):4d}" for x in rel[k]))
    print("steps    ", "    ", " ".join(f"{s:4d}" for s in steps))
    return 0


def induction(a) -> int:
    """The beside witness on the released run's grid (``--model released``) or an arm's saved steps."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    out = Path(a.out)
    g = np.random.default_rng(IND_SEED)
    half = g.integers(1000, 50000, (IND_N, IND_LEN))
    seqs = torch.from_numpy(np.concatenate([half, half], 1)).to(device)
    if a.model == "released":
        steps = [s for s in PYTHIA if s <= a.upto] + ([143000] if a.upto >= 143000 else [])
        def get(s):
            return from_pretrained_eager(GPTNeoXForCausalLM, ct.BASE, revision=f"step{s}",
                                         torch_dtype=torch.float32).to(device).eval()
    else:
        steps = [s for s in ct.SAVE_STEPS if s <= a.upto and ct.ckpt_dir(Path(a.root), a.model, s).exists()]
        def get(s):
            return load_local(ct.ckpt_dir(Path(a.root), a.model, s), device)
    res = {}
    for s in steps:
        mdl = get(s)
        with torch.no_grad():
            att = mdl(input_ids=seqs, output_attentions=True).attentions
        A = torch.stack(att, 1).float().cpu().numpy()             # (N, layers, heads, n, n)
        sc = np.mean([induction_scores(A[i], IND_LEN) for i in range(IND_N)], 0)
        li, hi = np.unravel_index(int(np.argmax(sc)), sc.shape)
        res[str(s)] = {"max": float(sc.max()), "head": [int(li), int(hi)]}
        print(json.dumps({"model": a.model, "step": s, **res[str(s)]}), flush=True)
        del mdl
        torch.cuda.empty_cache()
    ct._jdump(out / f"induction_{a.model}.json", {"code": code_sha(), "len": IND_LEN, "n": IND_N,
                                                  "seed": IND_SEED, "per_step": res})
    return 0


# ---------------------------------------------------------------- H4's power (beside)

H4C_MODELS = ("EleutherAI/pythia-70m", "EleutherAI/pythia-70m-warmup01") + tuple(
    f"EleutherAI/pythia-70m-seed{i}" for i in range(1, 10))


def h4_controls(a) -> int:
    """H4's held-out loss at ``--step`` on released variants (`/challenge-pr` on #346, finding 3):
    a 10x slower warmup (warmup01) and seeds 1-9 (other init and data order). Beside, no bar."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    rows = ct.open_rows(Path(a.root))
    res = {"code": code_sha(), "code_c4_train": ct.code_sha(), "rows": list(ct.HELD_OUT), "step": a.step, "loss": {}}
    for nm in H4C_MODELS:
        mdl = from_pretrained_eager(GPTNeoXForCausalLM, nm, revision=f"step{a.step}",
                                    torch_dtype=torch.float32).to(device).eval()
        res["loss"][nm] = ct.held_loss(mdl, rows, device)
        print(json.dumps({"model": nm, "loss": res["loss"][nm]}), flush=True)
        del mdl
        torch.cuda.empty_cache()
    ref = res["loss"][H4C_MODELS[0]]
    res["rel_diff"] = {k: (v - ref) / ref for k, v in res["loss"].items()}
    ct._jdump(Path(a.out) / f"h4_controls_{a.step}.json", res)
    return 0


# ---------------------------------------------------------------- V's gate on F (U2)

def gate(a) -> int:
    """C6's gate on F at GATE_STEP with 70m's 3 re-inits (rule 100 (a) for plant B's dose; if it
    refuses, the centred arms at plant B 0.5, V not read). Refuses if either arm has diverged."""
    from trackC_writeup import c6_warmup as c6
    div = divergence(Path(a.root))
    if any(v is not None for v in div.values()):
        raise C4ReadRefused(f"divergence {div}: nothing is read")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import reinit_model
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    cfg = AutoConfig.from_pretrained(ct.BASE, revision="step0")
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(cfg.hidden_size)
    u /= np.linalg.norm(u)
    pops, reinits = {}, [f"reinit:{s}" for s in range(c.N_REINIT)]
    for nm in reinits:
        mdl = reinit_model(cfg, int(nm.split(":")[1])).to(device)
        pops[nm] = c6._units(mdl, tok, device, rng, u)
        del mdl
    mdl = load_local(ct.ckpt_dir(Path(a.root), "F", GATE_STEP), device)
    pops[f"step{GATE_STEP}"] = c6._units(mdl, tok, device, rng, u)
    del mdl
    pf = json.loads(U3B_PREFLIGHT.read_text())
    g = c6.gate_one(pops, reinits, f"step{GATE_STEP}")
    as_frozen = None
    if g["b_dose"] is None:                                   # V refused: the centred count alone
        if pf["gate"]["b_dose"] != c6.U1B_B_DOSE:
            raise C4ReadRefused(f"U1B_B_DOSE {c6.U1B_B_DOSE} is not unit 3b's 70m dose {pf['gate']['b_dose']}")
        as_frozen = c6._jsonable(g)
        g = c6.gate_one(pops, reinits, f"step{GATE_STEP}", fixed_b=c6.U1B_B_DOSE)
    null_now = g["null_rates"]
    null_3b = {k: pf["gate"]["arms"][k]["null_rate"] for k in null_now}
    if any(abs(null_now[k] - null_3b[k]) > 0.005 for k in null_now):
        raise C4ReadRefused(f"the re-inits' rates {null_now} differ from unit 3b's {null_3b}")
    arms = {f"{t:.3g}": bool(g["arms"].get(f"{t:.3g}", {}).get("pass", False)) for t in TAU}
    res = {"code": code_sha(), "step": GATE_STEP, "gate": c6._jsonable(g), "v_refused_gate": as_frozen,
           "mode": "as frozen" if as_frozen is None else "centred only, plant B at 0.5 (C6 U1b's form)",
           "u_read": as_frozen is None,
           "arms_gated": arms, "null_vs_3b": [null_now, null_3b]}
    ct._jdump(out / "gate.json", res)
    print(json.dumps({"mode": res["mode"], "u_read": res["u_read"], "b_dose": g["b_dose"], "arms_gated": arms}), flush=True)
    return 0


# ---------------------------------------------------------------- the reading (U3)

def read(a) -> int:
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    div = divergence(Path(a.root))
    if any(v is not None for v in div.values()):
        ct._jdump(out / "reading.json", {"code": code_sha(), "refused": "divergence", "diverged": div})
        raise C4ReadRefused(f"divergence {div}: nothing is read")
    a0 = all_series(load_arm(out, "A0", GRID), GRID)
    f = all_series(load_arm(out, "F", GRID), GRID)
    rel = all_series(load_released(PYTHIA), PYTHIA)
    a0_py = {k: v[[GRID.index(s) for s in PYTHIA]] for k, v in a0.items()}
    r1 = {}
    for k in KEYS:
        da, dr = date_on(a0_py[k], PYTHIA), date_on(rel[k], PYTHIA)
        r1[k] = {"A0": da["step"], "released": dr["step"], "pass": r1_pass(da["step"], dr["step"])}
    gated = {"u": g["u_read"], **g["arms_gated"]}
    dates = {arm: {k: date_on(s[k]) for k in KEYS} for arm, s in (("A0", a0), ("F", f))}
    per = {k: label(dates["F"][k], dates["A0"][k]) for k in KEYS}
    refused = [k for k in KEYS if gated.get(k) and not r1[k]["pass"]]
    res = {"code": code_sha(), "code_c4_train": ct.code_sha(), "diverged": div, "gate_mode": g["mode"],
           "r1": r1, "r1_refused": refused, "gated": gated, "dates": dates, "per_key": per,
           "R2_centred": None if refused else centred_label({k: per[k] for k in KEYS[1:]}, gated),
           "R3_V": None if refused or not gated["u"] else per["u"],
           "beside": {"A0": {k: v.tolist() for k, v in a0.items()}, "F": {k: v.tolist() for k, v in f.items()}}}
    ct._jdump(out / "reading.json", res)
    print(json.dumps({k: res[k] for k in ("r1", "r1_refused", "R2_centred", "R3_V")}, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(q):
        q.add_argument("--out", required=True)
        q.add_argument("--root", default=str(ct.ROOT))
    q = sub.add_parser("known"); common(q)
    q = sub.add_parser("records"); common(q)
    q.add_argument("--arm", required=True, choices=ct.ARMS)
    q.add_argument("--upto", type=int, required=True)
    q.add_argument("--steps", type=int, nargs="*", default=None)
    q = sub.add_parser("h6"); common(q)
    q.add_argument("--upto", type=int, default=1000)
    q = sub.add_parser("induction"); common(q)
    q.add_argument("--model", required=True, choices=("released",) + ct.ARMS)
    q.add_argument("--upto", type=int, default=4000)
    q = sub.add_parser("h4-controls"); common(q)
    q.add_argument("--step", type=int, default=1000)
    q = sub.add_parser("gate"); common(q)
    q = sub.add_parser("read"); common(q)
    a = p.parse_args(argv)
    return {"known": known, "records": records, "h6": h6, "induction": induction, "gate": gate,
            "read": read, "h4-controls": h4_controls}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
