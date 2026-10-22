"""
trackC_writeup/c2_aniso.py — C2 unit 3: the anisotropy control (`design-C2.md` Blocked 92 (i),
the size label Blocked 95 (a); "Unit 3: the anisotropy control" there).

Per (step, v1 prompt) on `pythia-410m` (seed 0), one GPU pass (float32, eager, TF32 off), one record
holding, for each of the 25 hidden states:

- **e1**: A5's E_1, computed exactly as `trackA_force.a5_seeds.one_pass` does (float32 Gram of the
  L2-normalised states), so that the records reproduce A5's seed-0 records (the known answer);
- **e1c**: E_1 on the token-centred, re-normalised cloud: U the L2-normalised states (the cloud
  e1 reads), U − mean over tokens, each row re-normalised (float64). This is the scored energy;
- **e1r**: beside, not scored: E_1 on the raw-centred cloud, x − mean over tokens, then normalised;
- **cos**: the layer's mean off-diagonal cosine of U, (‖Σ U_i‖² − n) / (n (n − 1)) (F9's
  anisotropy, beside).

V_c is the count of violations of e1c (D2 (a)'s tolerance) summed over the 7 prompts. The reading
(`verdict`) is 92 (i)'s rule with 95 (a)'s size label first.

`preflight` reads only populations outside the scored family (the 18 grid steps): 3 matched
re-inits (`p1d_cluster_ensemble.arch_null.reinit_model`, Pythia's σ) and the off-grid step 7000.
It plants (A) a common direction on the earlier state of each transition, which the centred
energy should not see, and (B) token-independent spread on the later state, which it should; it
writes `tools/design_preflight.py`'s input and report.

Tier 1, description (C2 quotes it beside L1; nothing here is registered).
    python -m trackC_writeup.c2_aniso preflight --out <dir>
    python -m trackC_writeup.c2_aniso run --out <dir>
    python -m trackC_writeup.c2_aniso read --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import MAX_TOKENS, REL_TOL, SENS_TOLS, TOKENIZER, n_viol

MODEL = "EleutherAI/pythia-410m"
EARLY = tuple(s for s in STEPS if s <= 128)     # 92 (i): reported first; the unreadable check
WINDOW = (256, 512)                             # 92 (i) "survives": the half-rise in (128, 512]
SIZE = 0.5                                      # 95 (a): "mostly removed" below half the uncentred plateau
MIN_NORM = 1e-6                                 # a centred token this short has no direction: refuse
ENERGY_TOL = 1e-4                               # placed: |Δe1| / e1 against A5's seed-0 records (A5's own)
A5_SEED0 = "trackA/a5/2026-10-20-u1/seed0"      # A5's seed-0 records (code 7ae0bff62ada)
PREFLIGHT_STEP = 7000                           # off the grid, past the 1000–4000 plateau
N_REINIT = 3                                    # matched re-inits for the preflight's null
DOSES = (0.0, 0.25, 0.5, 1.0)                   # plant size, in units of the state's median token norm
SEED = 20261022


class AnisoRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_aniso.py", "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def unit(X: np.ndarray) -> np.ndarray:
    """Rows L2-normalised (float64); refuses a row shorter than MIN_NORM."""
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if n.min() < MIN_NORM:
        raise AnisoRefused(f"a row of norm {n.min():.3g} < {MIN_NORM}: no direction to re-normalise")
    return X / n


def centred(U: np.ndarray) -> np.ndarray:
    """The token-centred, re-normalised cloud: U − mean over tokens, rows re-normalised."""
    U = np.asarray(U, dtype=np.float64)
    return unit(U - U.mean(axis=0, keepdims=True))


def energy(U: np.ndarray) -> float:
    """E_1 of a unit-row cloud, as `core.metrics.interaction_energies_batched` at β = 1."""
    from core.metrics import interaction_energies_batched
    return interaction_energies_batched(U @ U.T, [1.0])[1.0]


def mean_cos(U: np.ndarray) -> float:
    """Mean off-diagonal cosine of a unit-row cloud: (‖Σ U_i‖² − n) / (n (n − 1))."""
    s = np.asarray(U, dtype=np.float64).sum(axis=0)
    n = U.shape[0]
    return float((s @ s - n) / (n * (n - 1)))


def state_row(h: np.ndarray) -> dict:
    """One hidden state (n, d), float32 as the model gave it: every quantity the record keeps."""
    import torch
    from core.metrics import interaction_energies_batched
    from core.models import layernorm_to_sphere
    U32 = layernorm_to_sphere(torch.from_numpy(h)).numpy()       # A5's e1, float32 throughout
    return {"e1": interaction_energies_batched(U32 @ U32.T, [1.0])[1.0],
            "e1c": energy(centred(U32)),
            "e1r": energy(unit(np.asarray(h, np.float64) - np.asarray(h, np.float64).mean(0))),
            "cos": mean_cos(unit(h))}


def rel_drops(e: Sequence[float]) -> np.ndarray:
    """(E_t − E_{t+1}) / |E_t| per transition: a violation is > REL_TOL (D2 (a))."""
    e = np.asarray(e, dtype=np.float64)
    return (e[:-1] - e[1:]) / np.maximum(np.abs(e[:-1]), 1e-12)


def series(recs: Dict[Tuple[int, str], dict], key: str, rel_tol: float = REL_TOL) -> np.ndarray:
    """V on the 18 steps for ``key``: violations summed over the 7 prompts."""
    return np.array([sum(n_viol(recs[(s, p)][key], rel_tol) for p in V1) for s in STEPS], float)


def verdict(vc: np.ndarray, vu: np.ndarray) -> dict:
    """92 (i)'s reading of the centred V, with 95 (a)'s size label first.

    ``vc`` the centred V and ``vu`` the uncentred V on the same records, both on STEPS.
    - plateau medians (1000, 2000, 4000) of each; ratio = centred / uncentred;
    - centred plateau 0: **removed** (the half-rise finds no step by 4000);
    - ratio < SIZE: **mostly removed** (95 (a)), the date beside;
    - any of steps 0–128 ≥ half the centred plateau: **unreadable** (92 (i));
    - else the half-rise (`a3.d4a`) in (128, 512]: **survives**; at 1000–4000: **moved**.
    With a positive plateau the half-rise always finds a step ≤ 4000, so these are exhaustive.
    """
    vc, vu = np.asarray(vc, float), np.asarray(vu, float)
    rc, ru = a3.d4a(vc), a3.d4a(vu)
    pc, pu = rc["plateau_median"], ru["plateau_median"]
    if pu <= 0:
        raise AnisoRefused("the uncentred plateau is 0: these are not the records A5 read")
    early = {s: float(vc[STEPS.index(s)]) for s in EARLY}
    out = {"early_centred": early, "plateau_centred": pc, "plateau_uncentred": pu,
           "ratio": pc / pu, "step": rc["step"], "step_uncentred": ru["step"]}
    if pc <= 0:
        return {**out, "label": "removed"}
    if pc / pu < SIZE:
        return {**out, "label": "mostly removed"}
    if max(early.values()) >= pc / 2:
        return {**out, "label": "unreadable"}
    if rc["step"] in WINDOW:
        return {**out, "label": "survives"}
    if rc["step"] <= 4000:
        return {**out, "label": "moved"}
    raise AssertionError(f"half-rise at {rc['step']} with a positive plateau")  # cannot happen


# ---------------------------------------------------------------- model passes

def _torch_setup():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the passes are the GPU's)")
    return torch, torch.device("cuda")


def load_step(step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, MODEL, revision=f"step{step}",
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise AnisoRefused(f"step {step}: weights are not float32")
    return m


def states(model, tok, prompt: str, device) -> Tuple[List[np.ndarray], List[int]]:
    """The 25 hidden states (n, d) as float32 numpy, and the token ids."""
    import torch
    from core.config import PROMPTS
    enc = tok(PROMPTS[prompt], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    return [h[0].to(torch.float32).cpu().numpy() for h in o.hidden_states], enc["input_ids"][0].tolist()


def record(hs: Sequence[np.ndarray], ids: List[int], prompt: str) -> dict:
    rows = [state_row(h) for h in hs]
    rec = {"prompt": prompt, "n": len(ids), "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:12],
           **{k: [r[k] for r in rows] for k in ("e1", "e1c", "e1r", "cos")}}
    check_populated(rec)
    return rec


def check_populated(rec: dict) -> None:
    """Refuse rather than degrade: 25 finite values per key, energies positive."""
    bad = [k for k in ("e1", "e1c", "e1r", "cos")
           if len(rec[k]) != 25 or not np.all(np.isfinite(rec[k]))
           or (k != "cos" and min(rec[k]) <= 0)]
    if bad:
        raise AnisoRefused(f"{rec['prompt']}: not populated: {bad}")


def rec_path(out: Path, step: int, prompt: str) -> Path:
    return out / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C2.md", "repo": MODEL, "code": code_sha(),
            "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "dtype": "float32", "device": torch.cuda.get_device_name(0), "tf32": False,
            "torch": torch.__version__, "max_tokens": MAX_TOKENS}


def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"            # the 18 steps are cached; nothing is downloaded
    torch, device = _torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    m = meta(torch)
    (out / "plan.json").write_text(json.dumps(m, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    a5 = DATA / A5_SEED0 / "records"
    first, t0 = True, time.time()
    for step in STEPS:
        todo = [p for p in V1 if not rec_path(out, step, p).exists()]
        if not todo:
            continue
        model = load_step(step, device)
        for p in todo:
            hs, ids = states(model, tok, p, device)
            rec = record(hs, ids, p)
            ref = json.loads((a5 / f"step{step}_{p}.json").read_text())
            if rec["ids_sha"] != ref["ids_sha"]:
                raise AnisoRefused(f"step {step} {p}: tokens differ from A5's record")
            dev = float(np.max(np.abs(np.subtract(rec["e1"], ref["e1"])) / np.asarray(ref["e1"])))
            if dev > ENERGY_TOL:
                raise AnisoRefused(f"step {step} {p}: e1 off A5's record by {dev:.2g} > {ENERGY_TOL}")
            rec.update(step=step, revision=f"step{step}", code=m["code"], e1_dev_a5=dev)
            rec_path(out, step, p).write_text(json.dumps(rec) + "\n")
            if first:
                print(f"first record populated: step {step} {p}, n {rec['n']}, e1 dev {dev:.1e}, "
                      f"e1c[0] {rec['e1c'][0]:.6f}, cos[12] {rec['cos'][12]:.4f}", flush=True)
                first = False
        del model
        torch.cuda.empty_cache()
        print(f"step {step} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- reading

def load_records(out: Path) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in STEPS:
        for p in V1:
            f = rec_path(out, s, p)
            if not f.exists():
                raise AnisoRefused(f"{f}: missing")
            recs[(s, p)] = json.loads(f.read_text())
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise AnisoRefused(f"records from {len(codes)} code versions: {sorted(codes)}")
    return recs


def read(a) -> int:
    out = Path(a.out)
    recs = load_records(out)
    vu, vc, vr = series(recs, "e1"), series(recs, "e1c"), series(recs, "e1r")
    v = verdict(vc, vu)
    cos = {s: [float(np.mean([recs[(s, p)]["cos"][L] for p in V1])) for L in range(25)] for s in STEPS}
    res = {"code": sorted({r["code"] for r in recs.values()})[0], "steps": list(STEPS),
           "early_centred_first": v["early_centred"],     # 92 (i): reported first
           "verdict": v, "V_uncentred": vu.tolist(), "V_centred": vc.tolist(),
           "e1_max_dev_a5": max(r["e1_dev_a5"] for r in recs.values()),
           "beside": {"V_centred_at_tol": {str(t): series(recs, "e1c", t).tolist() for t in SENS_TOLS},
                      "V_raw_centred": vr.tolist(), "raw_centred_d4a": a3.d4a(vr),
                      "mean_cos_by_layer": {str(s): c for s, c in cos.items()},
                      "mean_cos_layer_mean": [float(np.mean(cos[s])) for s in STEPS]}}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"early_centred": v["early_centred"]}))
    print(json.dumps({k: v[k] for k in ("label", "step", "plateau_centred", "plateau_uncentred", "ratio")}))
    return 0


# ---------------------------------------------------------------- the preflight (unit 3, before output)

def plant_common(h: np.ndarray, a: float, u: np.ndarray) -> np.ndarray:
    """(A) every token moved by the same vector a · m · u, m the median token norm."""
    m = float(np.median(np.linalg.norm(h, axis=1)))
    return np.asarray(h, np.float64) + a * m * u[None, :]


def plant_spread(h: np.ndarray, b: float, rng: np.random.Generator) -> np.ndarray:
    """(B) each token moved by its own Gaussian vector of norm about b · m."""
    m = float(np.median(np.linalg.norm(h, axis=1)))
    g = rng.standard_normal(h.shape) / np.sqrt(h.shape[1])
    return np.asarray(h, np.float64) + b * m * g


def _e(h: np.ndarray, which: str) -> float:
    U = unit(h)
    return energy(centred(U) if which == "c" else U)


def plants(hs: Sequence[np.ndarray], u: np.ndarray, rng: np.random.Generator) -> dict:
    """Per dose, the 24 transitions' relative drops, each transition planted on its own:
    (A) on the earlier state, read uncentred ('A_u') and centred ('A_c'); (B) on the later, centred."""
    base = {w: [_e(h, w) for h in hs] for w in ("u", "c")}
    out: Dict[str, Dict[str, List[float]]] = {"A_u": {}, "A_c": {}, "B_c": {}}
    for d in DOSES:
        k = str(d)
        out["A_u"][k], out["A_c"][k], out["B_c"][k] = [], [], []
        for t in range(len(hs) - 1):
            hp = plant_common(hs[t], d, u)
            for w in ("u", "c"):
                e0 = _e(hp, w)
                out[f"A_{w}"][k].append((e0 - base[w][t + 1]) / abs(e0))
            e1 = _e(plant_spread(hs[t + 1], d, rng), "c")
            out["B_c"][k].append((base["c"][t] - e1) / abs(base["c"][t]))
    out["base_u"], out["base_c"] = rel_drops(base["u"]).tolist(), rel_drops(base["c"]).tolist()
    return out


def preflight(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch_setup()
    from transformers import AutoConfig, AutoTokenizer
    from p1d_cluster_ensemble.arch_null import init_summary, reinit_model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(1024)
    u /= np.linalg.norm(u)
    pops: Dict[str, dict] = {}
    models = [(f"reinit:{s}", None) for s in range(N_REINIT)] + [(f"step{PREFLIGHT_STEP}", PREFLIGHT_STEP)]
    for name, step in models:
        if step is None:
            cfg = AutoConfig.from_pretrained(MODEL, revision="step0")
            model = reinit_model(cfg, int(name.split(":")[1])).to(device)
            init = init_summary(model)
        else:
            model, init = load_step(step, device), None
        per = {}
        for p in V1:
            hs, _ = states(model, tok, p, device)
            per[p] = plants(hs, u, rng)
            if not per[p]["base_c"] or not np.all(np.isfinite(per[p]["base_c"])):
                raise AnisoRefused(f"{name} {p}: centred drops not populated")
        pops[name] = {"per_prompt": per, "init_summary": init}
        del model
        torch.cuda.empty_cache()
        print(f"{name} done", flush=True)

    def flat(names, key, dose=None):
        return [x for n in names for p in V1
                for x in (pops[n]["per_prompt"][p][key] if dose is None else pops[n]["per_prompt"][p][key][dose])]

    reinits = [f"reinit:{s}" for s in range(N_REINIT)]
    trained = [f"step{PREFLIGHT_STEP}"]
    tool_in = {"statistic": "centred E_1 relative drop per (prompt, transition), (E_t − E_t+1) / |E_t| "
                            "on the token-centred, re-normalised cloud (c2_aniso)",
               "direction": "greater",
               "nulls": {"reinit": flat(reinits, "base_c")},
               "moved_marks": [],
               "trained": flat(trained, "base_c"),
               "trained_step": str(PREFLIGHT_STEP),
               "scored_steps": [str(s) for s in STEPS],
               "planted": {f"{d:g}": flat(reinits, "B_c", str(d)) for d in DOSES},
               "bounds": None,
               "decision_p": 0.05}
    (out / "preflight_input.json").write_text(json.dumps(tool_in) + "\n")

    def rate(names, key, d):
        return float(np.mean(np.asarray(flat(names, key, str(d))) > REL_TOL))

    table = {pop: {"A_uncentred": {f"{d:g}": rate(names, "A_u", d) for d in DOSES},
                   "A_centred": {f"{d:g}": rate(names, "A_c", d) for d in DOSES},
                   "B_centred": {f"{d:g}": rate(names, "B_c", d) for d in DOSES},
                   "A_centred_max_shift": max(
                       float(np.max(np.abs(np.subtract(pops[n]["per_prompt"][p]["A_c"][str(d)],
                                                       pops[n]["per_prompt"][p]["A_c"]["0.0"]))))
                       for n in names for p in V1 for d in DOSES)}
             for pop, names in (("reinit", reinits), ("trained", trained))}
    rec = {"code": code_sha(), "meta": meta(torch), "doses": list(DOSES), "rel_tol": REL_TOL,
           "u_seed": SEED, "violation_rate_at_rel_tol": table,
           "init_summary": {n: pops[n]["init_summary"] for n in reinits}}
    (out / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(table, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("preflight", "run", "read"):
        sub.add_parser(c).add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"preflight": preflight, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
