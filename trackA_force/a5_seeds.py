"""
trackA_force/a5_seeds.py — A5: the energy break on the PolyPythias 410m seeds (`design-A5.md`,
decided 2026-10-19, D10 (a) 2026-10-20; not registered).

Per (seed, step, v1 prompt), one GPU pass (float32, eager, TF32 off), one record:

- **e1**: E_1 on the unit sphere for the 25 hidden states, computed exactly as Phase 1 did
  (`core.models.layernorm_to_sphere`, float32 Gram, `core.metrics.interaction_energies_batched`),
  so that a violation is `energy_violation_severity`'s (rel_tol 1e-3) and V, summed over the 7
  v1 prompts, is the panel's `p1_energy_nviol_b1` (D2 (a), D3 (a)).
- **e35**: beside, not scored: E_3.5 in each block's unit LN1 frame (`a1a.frame`), positions
  1…n−1, blocks 0–23.
- **prev**, **ind**: per head, the previous-token score (mean attention from i to i − 1, i ≥ 1)
  and the induction score at +1 (Σ attention over `induction_candidates(strict=True)` and the
  pair count). D10 (a) picks one head of each kind per seed, the top score at 143000.

The statistic is D4 (a) (`a3.d4a`): the first grid step with V ≥ half the median of V over
1000, 2000 and 4000; inside is D5 (a), the step is 256 or 512. K counts the seeds inside, a seed
that fails to load or has a zero plateau counting as outside (D7 (a)).

`gate` is the calibration gate on seed 0 (`design-A5.md` "Before it can be registered" 2): the
known answer (tokens, per-layer energies, the panel's counts, A3's E3 / E4 rows), the planted
effect, the planted null (code, not the null), the sensitivity arms, and D10's known answer
(the rule's heads on seed 0, their dates, ties and shares).

Tier 1 until registered.
    python -m trackA_force.a5_seeds run --seed 0 --out <dir> [--steps 143000 ...] [--allow-download]
    python -m trackA_force.a5_seeds gate --out <dir>
    python -m trackA_force.a5_seeds score --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .a1a import DATA, REPO, STEPS, V1, frame, prompts_hash, run_dir
from . import a3

SEEDS = tuple(range(10))                       # 0 is `pythia-410m` itself, the development set
OUTLIERS = (3, 4)                              # PolyPythias's named 410m outliers, D7 (a): count, flagged
WINDOW = (256, 512)                            # D5 (a): the break in (128, 512]
PLATEAU = (1000, 2000, 4000)                   # D4 (a)
P0 = 1 / 6                                     # D6 (b): uniform over the 12 non-zero steps up to 4000
NULL_STEPS = tuple(s for s in STEPS if 0 < s <= 4000)
REL_TOL = 1e-3                                 # D2 (a), core.metrics.ENERGY_VIOLATION_REL_TOL
SENS_TOLS = (1e-4, 1e-3, 1e-2)
BETA_LN1 = 3.5                                 # beside (Blocked 9's β), not scored
MAX_TOKENS = 512                               # as core.models.extract_activations
TOKENIZER = ("EleutherAI/pythia-410m", "step143000")
ENERGY_TOL = 1e-4                              # placed: per-layer |ΔE| / E against Phase 1's CPU run
ATTN_TOL = 1e-4                                # placed: |Δ score| against A3's rows (stored attentions)
A3_ROWS = "trackA/a3/2026-10-20-A3u5/a3_rows.json"
N_NULL = 100_000
SEED = 20261020


class A5Refused(RuntimeError):
    pass


def repo_for(seed: int) -> str:
    return "EleutherAI/pythia-410m" if seed == 0 else f"EleutherAI/pythia-410m-seed{seed}"


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("a5_seeds.py", "a3.py", "a1a.py"):
        h.update((REPO / "trackA_force" / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def n_viol(energies: Sequence[float], rel_tol: float = REL_TOL) -> int:
    from core.metrics import energy_violation_severity
    return energy_violation_severity(list(energies), rel_tol=rel_tol)["n_violations"]


def series_v(recs: Dict[Tuple[int, str], dict], prompts: Sequence[str] = V1,
             rel_tol: float = REL_TOL, key: str = "e1") -> np.ndarray:
    """V on the 18 steps: violations summed over ``prompts``."""
    return np.array([sum(n_viol(recs[(s, p)][key], rel_tol) for p in prompts) for s in STEPS], float)


def inside(r: dict) -> bool:
    return r.get("reading") == "located" and r["step"] in WINDOW


def shift(v: np.ndarray, d: int) -> np.ndarray:
    """The series moved d grid steps later (d < 0: earlier), padded with its end values."""
    v = np.asarray(v, float)
    i = np.clip(np.arange(len(v)) - d, 0, len(v) - 1)
    return v[i]


def binom_tail(k: int, n: int = 9, p: Fraction = Fraction(1, 6)) -> float:
    """P(Bin(n, p) ≥ k), exact in rationals (a float sum can exceed 1 at k = 0)."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def head_units(recs: Dict[Tuple[int, str], dict], kind: str, L: int, h: int) -> np.ndarray:
    """(7 prompts, 18 steps, k): prev → (score,), ind → (Σ attention, pairs) for a3's statistics."""
    if kind == "prev":
        return np.array([[[recs[(s, p)]["prev"][L][h]] for s in STEPS] for p in V1])
    return np.array([[[recs[(s, p)]["ind"][L][h], recs[(s, p)]["n_ind"]] for s in STEPS] for p in V1])


def head_scores(recs: Dict[Tuple[int, str], dict], step: int = 143000) -> Dict[str, np.ndarray]:
    """(24, 16) per kind at ``step``: prev the mean over prompts, ind pooled over prompts."""
    prev = np.mean([np.asarray(recs[(step, p)]["prev"]) for p in V1], axis=0)
    ind = (np.sum([np.asarray(recs[(step, p)]["ind"]) for p in V1], axis=0)
           / sum(recs[(step, p)]["n_ind"] for p in V1))
    return {"prev": prev, "ind": ind}


def pick(scores: np.ndarray) -> dict:
    """D10 (a): the top head (ties to the lowest index), with the runner-up beside."""
    flat = scores.ravel()
    order = np.argsort(-flat, kind="stable")
    L, h = divmod(int(order[0]), scores.shape[1])
    L2, h2 = divmod(int(order[1]), scores.shape[1])
    return {"head": f"L{L}H{h}", "L": L, "h": h, "score": float(flat[order[0]]),
            "runner_up": f"L{L2}H{h2}", "runner_up_score": float(flat[order[1]])}


def rank_of(scores: np.ndarray, L: int, h: int) -> int:
    return int((scores.ravel() > scores[L, h]).sum()) + 1


def share(r: dict, i: int) -> Optional[float]:
    """D10 (a): the share of a row's move done at grid index i, against its own dating reference."""
    x = np.asarray(r["series"], float)
    if r["id"] == "break":
        return float(x[i] / r["plateau_median"])
    if not r.get("ref"):
        return None
    return float(a3.SIGN["up"] * (x[i] - x[0]) / r["ref"])


def tie_break(a: dict, b: dict) -> dict:
    """D10 (a) for one pair: the dates, and where the half-point rule ties them, each row's share
    at the grid step before, the LR schedule's share beside. Descriptive: no p, no E."""
    out = {"pair": f"{a['id']}–{b['id']}", "steps": [a.get("step"), b.get("step")]}
    if a.get("reading") != "located" or b.get("reading") != "located":
        out["tie"] = None
        return out
    out["tie"] = a["index"] == b["index"]
    if not out["tie"] or a["index"] == 0:
        return out
    i = a["index"] - 1
    sa, sb = share(a, i), share(b, i)
    out.update(before=STEPS[i], shares=[sa, sb], lr_share=a3.lr(STEPS[i]),
               further=(a["id"] if sa > sb else b["id"] if sb > sa else None),
               censored=[bool(a.get("censored")), bool(b.get("censored"))])
    return out


def rows_for(recs: Dict[Tuple[int, str], dict], heads: Dict[str, Tuple[int, int]],
             rng: np.random.Generator) -> Dict[str, dict]:
    """The three dated rows on one seed: the break (D4 (a)) and the two heads (A3's locator)."""
    v = series_v(recs)
    br = dict(a3.d4a(v), id="break", series=v.tolist(), censored=False)
    out = {"break": br}
    for kind, stat in (("prev", a3.stat_mean), ("ind", a3.stat_pooled)):
        L, h = heads[kind]
        u = head_units(recs, kind, L, h)
        r = a3.locate_units(u, stat, "up", rng)
        r.update(id=kind, head=f"L{L}H{h}", series=stat(u).tolist(), censored=a3.censored(r))
        out[kind] = r
    return out


def d10(rows: Dict[str, dict]) -> List[dict]:
    return [tie_break(rows["break"], rows["ind"]), tie_break(rows["prev"], rows["ind"])]


# ---------------------------------------------------------------- records

def rec_path(out: Path, step: int, prompt: str) -> Path:
    return out / "records" / f"step{step}_{prompt}.json"


def load_records(out: Path, need: bool = True) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = rec_path(out, s, p)
            if f.exists():
                recs[(s, p)] = json.loads(f.read_text())
            elif need:
                raise A5Refused(f"{f}: missing")
    return recs


def check_populated(rec: dict) -> None:
    """Refuse rather than degrade: a record must be finite and full-sized."""
    e1, e35 = np.asarray(rec["e1"]), np.asarray(rec["e35"])
    prev, ind = np.asarray(rec["prev"]), np.asarray(rec["ind"])
    bad = []
    if e1.shape != (25,) or not np.all(np.isfinite(e1)) or np.any(e1 <= 0):
        bad.append(f"e1 {e1.shape}")
    if e35.shape != (24,) or not np.all(np.isfinite(e35)):
        bad.append(f"e35 {e35.shape}")
    if prev.shape != (24, 16) or not np.all(np.isfinite(prev)) or prev.max() <= 0:
        bad.append(f"prev {prev.shape}")
    if ind.shape != (24, 16) or not np.all(np.isfinite(ind)) or rec["n_ind"] < 1:
        bad.append(f"ind {ind.shape}, n_ind {rec['n_ind']}")
    if bad:
        raise A5Refused(f"step {rec['step']} {rec['prompt']}: not populated: {', '.join(bad)}")


def one_pass(model, tok, prompt: str, ln: Dict, device) -> dict:
    import torch
    from core.battery_structure import induction_candidates
    from core.config import PROMPTS
    from core.metrics import interaction_energies_batched
    from core.models import layernorm_to_sphere
    enc = tok(PROMPTS[prompt], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    ids = enc["input_ids"][0].tolist()
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()},
                  output_hidden_states=True, output_attentions=True)
    hs = [h[0].to(torch.float32).cpu() for h in o.hidden_states]
    e1 = []
    for h in hs:                                   # as p1_mstate_tracking.analysis_p1
        U = layernorm_to_sphere(h).numpy()
        e1.append(interaction_energies_batched(U @ U.T, [1.0])[1.0])
    e35 = []
    for L in range(len(hs) - 1):                   # state 24 is post-final-LN: no block reads it
        U = frame(hs[L].numpy()[1:], ln["w"][L], ln["b"][L], ln["eps"])
        n = U.shape[0]
        e35.append(float(np.exp(BETA_LN1 * (U @ U.T)).sum() / (2 * BETA_LN1 * n * n)))
    A = torch.stack([a[0] for a in o.attentions]).to(torch.float64)   # (24, 16, n, n), on device
    n = A.shape[-1]
    i = torch.arange(1, n, device=A.device)
    prev = A[:, :, i, i - 1].mean(-1)
    pairs = induction_candidates(ids, strict=True)
    q = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=A.device)
    k = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=A.device)
    ind = A[:, :, q, k].sum(-1) if len(pairs) else torch.zeros(A.shape[:2], dtype=A.dtype)
    return {"prompt": prompt, "n": n, "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:12],
            "tokens": tok.convert_ids_to_tokens(ids), "e1": e1, "e35": e35,
            "prev": prev.cpu().numpy().tolist(), "ind": ind.cpu().numpy().tolist(),
            "n_ind": len(pairs)}


def run(a) -> int:
    if not a.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"         # no seed is downloaded before A5 registers
    import torch
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the passes are the GPU's, design-A5 'Not a choice')")
    device = torch.device("cuda")
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    repo = repo_for(a.seed)
    meta = {"design": "trackA_force/design-A5.md", "seed": a.seed, "repo": repo, "code": code_sha(),
            "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "dtype": "float32", "device": torch.cuda.get_device_name(0), "tf32": False,
            "torch": torch.__version__, "max_tokens": MAX_TOKENS}
    (out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    steps = list(a.steps or STEPS)
    steps = [143000] + [s for s in steps if s != 143000] if 143000 in steps else steps
    first = True
    t0 = time.time()
    for step in steps:
        todo = [p for p in V1 if not rec_path(out, step, p).exists()]
        if not todo:
            print(f"have step {step}", flush=True)
            first = False
            continue
        try:
            model = from_pretrained_eager(GPTNeoXForCausalLM, repo, revision=f"step{step}",
                                          torch_dtype=torch.float32).to(device).eval()
        except Exception as e:                     # D7 (a): a seed that fails to load counts as outside
            (out / f"load_failed_step{step}.txt").write_text(f"{type(e).__name__}: {e}\n")
            print(f"step {step}: load failed: {e}", flush=True)
            continue
        if next(model.parameters()).dtype != torch.float32:
            raise A5Refused(f"step {step}: weights are not float32")
        ln = {"w": [l.input_layernorm.weight.detach().double().cpu().numpy() for l in model.gpt_neox.layers],
              "b": [l.input_layernorm.bias.detach().double().cpu().numpy() for l in model.gpt_neox.layers],
              "eps": float(model.gpt_neox.layers[0].input_layernorm.eps)}
        for p in todo:
            rec = one_pass(model, tok, p, ln, device)
            rec.update(step=step, revision=f"step{step}", **{k: meta[k] for k in ("seed", "repo", "code")})
            check_populated(rec)
            rec_path(out, step, p).write_text(json.dumps(rec) + "\n")
            if first:
                print(f"first record populated: step {step} {p}, n {rec['n']}, "
                      f"e1[0] {rec['e1'][0]:.6f}, e1[-1] {rec['e1'][-1]:.6f}, n_ind {rec['n_ind']}", flush=True)
                first = False
        del model
        torch.cuda.empty_cache()
        print(f"step {step} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- the gate (seed 0)

def panel_v() -> np.ndarray:
    p = {int(r["step"]): r for r in csv.DictReader((DATA / "analysis" / "colocation_panel.csv").open())}
    return np.array([float(p[s]["p1_energy_nviol_b1"]) for s in STEPS])


def known_answer(recs: Dict[Tuple[int, str], dict]) -> dict:
    from tools.run.behavioural import _tokens_txt
    tok_bad, e_rel, margins = [], 0.0, []
    for s in STEPS:
        for p in V1:
            d = run_dir(s, p)
            if recs[(s, p)]["tokens"] != _tokens_txt(d):
                tok_bad.append(f"{s}|{p}")
            ref = [l["energies"]["1.0"] for l in json.loads((d / "energies.json").read_text())["layers"]]
            ours = np.asarray(recs[(s, p)]["e1"])
            e_rel = max(e_rel, float(np.max(np.abs(ours - ref) / np.abs(ref))))
            rd = -np.diff(ours) / np.maximum(np.abs(ours[:-1]), 1e-12)
            margins.append(float(np.min(np.abs(rd - REL_TOL))))
    v, want_v = series_v(recs), panel_v()
    br = a3.d4a(v)
    # A3's attention rows, from the stored CPU attentions: E4 is L5H2, E3 pools L8H6 and L11H14.
    a3r = json.loads((DATA / A3_ROWS).read_text())["rows"]
    e4 = a3.stat_mean(head_units(recs, "prev", *a3.PREV))
    e3 = a3.stat_pooled(np.concatenate([head_units(recs, "ind", L, h) for L, h in a3.MEMBERS_P1]))
    d_e4 = float(np.max(np.abs(e4 - np.asarray(a3r["E4"]["series"]))))
    d_e3 = float(np.max(np.abs(e3 - np.asarray(a3r["E3"]["series"]))))
    ok = (not tok_bad and e_rel <= ENERGY_TOL and np.array_equal(v, want_v)
          and br["step"] == 512 and br["plateau_median"] == 57.0 and max(d_e3, d_e4) <= ATTN_TOL)
    return {"pass": bool(ok), "tokens_mismatch": tok_bad, "energy_max_rel_diff": e_rel,
            "energy_tol": ENERGY_TOL, "closest_drop_to_tol": float(min(margins)),
            "V": v.tolist(), "panel_V": want_v.tolist(), "V_equal": bool(np.array_equal(v, want_v)),
            "break": br, "e4_max_abs_diff": d_e4, "e3_max_abs_diff": d_e3, "attn_tol": ATTN_TOL,
            "violations_step0": float(v[0])}


def planted_effect(v: np.ndarray) -> dict:
    r = a3.d4a(shift(v, 1))
    return {"pass": bool(r["step"] == 1000 and not inside(r)), "reading": r}


def planted_null(v: np.ndarray, rng: np.random.Generator) -> dict:
    """Seed 0's series moved so its half-rise sits at each of the null's 12 steps; then K over 9
    series with the step drawn from h0, against Bin(9, 1/6). Checks the code, not the null."""
    i0 = STEPS.index(512)
    table = []
    for s in NULL_STEPS:
        r = a3.d4a(shift(v, STEPS.index(s) - i0))
        table.append({"planted": s, "reads": r["step"], "plateau_median": r["plateau_median"],
                      "inside": inside(r)})
    exact = all(t["inside"] == (t["planted"] in WINDOW) for t in table)
    ins = np.array([t["inside"] for t in table])
    k = ins[rng.integers(0, len(NULL_STEPS), size=(N_NULL, 9))].sum(axis=1)
    pmf = np.bincount(k, minlength=10) / N_NULL
    want = np.array([math.comb(9, j) * P0 ** j * (1 - P0) ** (9 - j) for j in range(10)])
    se = np.sqrt(want * (1 - want) / N_NULL)
    z = float(np.max(np.abs(pmf - want) / np.maximum(se, 1e-12)))
    return {"pass": bool(exact and z < 4.0), "table": table, "inside_iff_window": exact,
            "pmf": pmf.tolist(), "binom_pmf": want.tolist(), "max_z": z, "n": N_NULL,
            "tail_ge7": float(pmf[7:].sum()), "binom_tail_ge7": binom_tail(7)}


def sensitivity(recs: Dict[Tuple[int, str], dict]) -> dict:
    tol = {f"{t:g}": a3.d4a(series_v(recs, rel_tol=t)) for t in SENS_TOLS}
    lopo = {p: a3.d4a(series_v(recs, [q for q in V1 if q != p])) for p in V1}
    every = list(tol.values()) + list(lopo.values())
    return {"pass": bool(all(inside(r) for r in every)),
            "rel_tol": {k: {"step": r["step"], "plateau_median": r["plateau_median"]} for k, r in tol.items()},
            "leave_one_out": {k: {"step": r["step"], "plateau_median": r["plateau_median"]} for k, r in lopo.items()}}


def d10_known(recs: Dict[Tuple[int, str], dict], rng: np.random.Generator) -> dict:
    """D10 (a) on seed 0: the rule's heads, their dates, the ties and shares; A3's heads beside."""
    sc = head_scores(recs)
    picks = {k: pick(sc[k]) for k in ("prev", "ind")}
    rule = rows_for(recs, {k: (p["L"], p["h"]) for k, p in picks.items()}, rng)
    a3h = rows_for(recs, {"prev": a3.PREV, "ind": a3.MEMBERS_P1[0]}, rng)
    keep = ("head", "reading", "step", "ref_step", "ref_over_sigma", "censored", "plateau_median")
    return {"picks": picks,
            "a3_heads_rank": {"L5H2 (prev)": rank_of(sc["prev"], *a3.PREV),
                              **{f"L{L}H{h} (ind)": rank_of(sc["ind"], L, h) for L, h in a3.MEMBERS_P1}},
            "rule": {"rows": {k: {f: r.get(f) for f in keep} for k, r in rule.items()}, "pairs": d10(rule)},
            "a3_heads_beside": {"rows": {k: {f: r.get(f) for f in keep} for k, r in a3h.items()},
                                "pairs": d10(a3h)}}


def gate(a) -> int:
    out = Path(a.out)
    recs = load_records(out)
    rng = np.random.default_rng(SEED)
    ka = known_answer(recs)
    v = np.asarray(ka["V"])
    res = {"code": code_sha(), "prompts_hash": prompts_hash(), "records": str(out),
           "known_answer": ka, "planted_effect": planted_effect(v), "planted_null": planted_null(v, rng),
           "sensitivity": sensitivity(recs), "d10": d10_known(recs, rng),
           "beside": {"V_ln1_b35": series_v(recs, key="e35").tolist()}}
    res["pass"] = all(res[k]["pass"] for k in ("known_answer", "planted_effect", "planted_null", "sensitivity"))
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: (res[k]["pass"] if isinstance(res[k], dict) and "pass" in res[k] else res[k])
                      for k in ("known_answer", "planted_effect", "planted_null", "sensitivity", "pass")}))
    return 0 if res["pass"] else 1


# ---------------------------------------------------------------- the score (unit 3)

def p_value_p_a5(k: int, n: int = 9) -> dict:
    """`P-A5`'s gate (`design-A5.md` "The draft entry"): the exact one-sided tail P(Bin(9, 1/6) ≥ K)
    and its e-value by `core.evalues.calibrate`, quoted "against a uniform-location null"."""
    from core.evalues import calibrate
    p = binom_tail(int(k), n)
    return {"K": int(k), "n": n, "p0": P0, "p": p, "e": float(calibrate(p))}


def score(a) -> int:
    root = Path(a.out)
    per = {}
    for seed in SEEDS[1:]:
        d = root / f"seed{seed}"
        recs = load_records(d, need=False) if d.exists() else {}
        if len(recs) < len(STEPS) * len(V1):
            per[seed] = {"reading": "failed to load", "inside": False, "outlier": seed in OUTLIERS}
            continue
        r = a3.d4a(series_v(recs))
        per[seed] = {**r, "inside": inside(r), "outlier": seed in OUTLIERS}
    g = p_value_p_a5(sum(r["inside"] for r in per.values()))
    res = {"code": code_sha(), **g, "per_seed": per}
    (root / "score.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(g))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True, choices=SEEDS)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, nargs="*")
    r.add_argument("--allow-download", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True, help="the directory holding seed1 … seed9")
    a = ap.parse_args(argv)
    return {"run": run, "gate": gate, "score": score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
