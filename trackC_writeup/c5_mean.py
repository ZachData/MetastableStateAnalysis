"""
trackC_writeup/c5_mean.py — C5 unit 2, M2's other half: do ‖m‖²'s falls at the break sit in a
few coordinates chosen by the mean? (`design-C5.md` "Unit 2", Blocked 105 (a), the user's.)

Per seed (0 the development set, 1–9 scored), on the 18 grid steps and the 7 v1 prompts:

- **gate**: on seed 0's off-grid step 7000, the reader on a planted null (each fall turned by a
  Haar rotation) and a planted effect (a fall of f·|Δ| added in 3 coordinates); refuses unless
  both sides pass; writes `tools/design_preflight`'s input;
- **pass**: m, the token mean of the normalised states, stored per coordinate (7 × 25 × 1024
  per step), with e1 (A5's path); known answers against unit 1's records;
- **read**: k½ (the fewest coordinates holding half a fall) at every transition V counts at
  128–1000, against 100 Haar rotations per seed; Q1 (the tally), Q2, Q3.

Tier 1, description (nothing here is registered).
    METS_DATA=<main>/data HF_HOME=<main>/data/hf python -m trackC_writeup.c5_mean gate --out <dir>
    HF_HOME=<main>/data/hf python -m trackC_writeup.c5_mean pass --out <dir> --seeds 0
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c5_mean pass --out <dir> --seeds 1 .. 9
    python -m trackC_writeup.c5_mean read --out <dir>
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

from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import REL_TOL, TOKENIZER, repo_for
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_seeds as cs
from trackC_writeup import c5_massive as cm

SEEDS = cm.SEEDS
ALL_SEEDS = cm.ALL_SEEDS
FAMILY_STEPS = (128, 256, 512, 1000)
FEW = 16                                        # k½ ≤ 16: "a few" (unit 1's MAX_S)
KEPT = 7                                        # the tally: ≥ 7 of 9
N_ROT = 100                                     # Haar rotations per seed
ROT_SEED = 20261024                             # ortho_group random_state: ROT_SEED + seed
N_ROT_LEVEL = 10                                # the level's rotated values: the seed's first 10
F_ROT = 0.05                                    # Q2: f_rot ≤ 0.05
RECUR = 0.25                                    # Q3: in the k½ set at ≥ ¼ of the transitions
DM0 = cm.DM0
SCALE_STEP = cm.SCALE_STEP                      # 7000: the gate's off-grid trained step
GATE_ROT_SEED = 20261025
GATE_PLANT_SEED = 20261026
DOSES = (0.0, 0.5, 1.0, 2.0)
GATE_DOSE = 2.0
GATE_NULL_FEW, GATE_NULL_ROT = 0.05, 0.10       # the planted null: at most
GATE_EFF = 0.9                                  # the planted effect at f = 2: at least
N_PLANT = 3
M2_TOL = (1e-9, 1e-15)                          # Σm² against unit 1's m2: relative, absolute
C5U1 = "trackC/c5/2026-10-23-u1"                # unit 1's records (code 7e45858c218d)


class MeanRefused(RuntimeError):
    pass


@functools.lru_cache(maxsize=None)
def code_sha() -> str:
    """The hash of the code this process loaded, taken once (`LESSONS.md` 3)."""
    h = hashlib.sha256()
    for f in ("trackC_writeup/c5_mean.py", "trackC_writeup/c5_massive.py", "trackC_writeup/c2_seeds.py",
              "trackC_writeup/c2_aniso.py", "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the statistic

def drops(e1: Sequence[float], tol: float = REL_TOL) -> np.ndarray:
    """The transitions ℓ → ℓ+1 whose E₁ drops by more than ``tol`` relative (unit 1's R0′ rule)."""
    e1 = np.asarray(e1, float)
    return np.nonzero((e1[:-1] - e1[1:]) / np.maximum(np.abs(e1[:-1]), 1e-12) > tol)[0]


def k_half(A, B) -> np.ndarray:
    """Per row, the fewest coordinates whose δᵢ = Bᵢ² − Aᵢ² (most negative first) sum to ≤ ½Δ,
    Δ = Σδᵢ < 0. A, B (T, d) or (d,); a row with Δ ≥ 0 refuses."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    d = B * B - A * A
    D = d.sum(axis=1)
    if np.any(D >= 0):
        raise MeanRefused("k½ asked of a transition at which ‖m‖² does not fall")
    cum = np.cumsum(np.sort(d, axis=1), axis=1)
    return np.argmax(cum <= 0.5 * D[:, None], axis=1) + 1


def k_half_level(M) -> np.ndarray:
    """Per row, the fewest coordinates holding half of ‖m‖²."""
    M = np.atleast_2d(np.asarray(M, float))
    q = M * M
    cum = np.cumsum(-np.sort(-q, axis=1), axis=1)
    return np.argmax(cum >= 0.5 * q.sum(axis=1, keepdims=True), axis=1) + 1


def participation(M) -> np.ndarray:
    """(Σmᵢ²)² / Σmᵢ⁴ per row: 1 for one coordinate, d for a flat vector."""
    M = np.atleast_2d(np.asarray(M, float))
    q = M * M
    return q.sum(axis=1) ** 2 / (q * q).sum(axis=1)


def k_half_set(a, b) -> List[int]:
    """The coordinates of one transition's k½ set."""
    d = np.asarray(b, float) ** 2 - np.asarray(a, float) ** 2
    return sorted(int(i) for i in np.argsort(d, kind="stable")[:int(k_half(a, b)[0])])


def rotations(n: int, d: int, random_state: int) -> List[np.ndarray]:
    """``n`` Haar rotations of R^d, reproducible."""
    from scipy.stats import ortho_group
    rng = np.random.default_rng(random_state)
    return [ortho_group.rvs(d, random_state=rng) for _ in range(n)]


def f_rot(A, B, Qs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per row, the fraction of rotations with k½ ≤ the observed; the rotated k½ (n_rot, T) beside."""
    A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
    k = k_half(A, B)
    kr = np.array([k_half(A @ Q.T, B @ Q.T) for Q in Qs])
    return (kr <= k[None, :]).mean(axis=0), kr


def plant(A, B, coords: Sequence[Sequence[int]], f: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add a fall of f·|Δ| in each row's ``coords`` (|coords| coordinates, an equal part each) by
    raising the before-state's coordinates: (aᵢ + α)² − aᵢ² = f|Δ| / |coords|, α ≥ 0."""
    A, B = np.array(np.atleast_2d(A), float), np.atleast_2d(np.asarray(B, float))
    D = (B * B - A * A).sum(axis=1)
    if f == 0:
        return A, B
    for t, cs_ in enumerate(coords):
        part = f * abs(D[t]) / len(cs_)
        for i in cs_:
            a = A[t, i]
            A[t, i] = (1.0 if a >= 0 else -1.0) * np.sqrt(a * a + part)     # |a| raised, sign kept
    return A, B


def recurring(sets: Sequence[Sequence[int]], frac: float = RECUR) -> List[Tuple[int, float]]:
    """Coordinates inside at least ``frac`` of the k½ sets, with their fraction."""
    if not sets:
        return []
    cnt: Dict[int, int] = {}
    for s in sets:
        for i in s:
            cnt[i] = cnt.get(i, 0) + 1
    n = len(sets)
    return sorted(((i, k / n) for i, k in cnt.items() if k / n >= frac), key=lambda x: -x[1])


def label_q1(med: float) -> str:
    return "few" if med <= FEW else "spread"


def tally(labels: Dict[int, str]) -> dict:
    few = sum(v == "few" for v in labels.values())
    spread = sum(v == "spread" for v in labels.values())
    lab = "in a few coordinates" if few >= KEPT else "spread" if spread >= KEPT else "mixed"
    return {"few": few, "spread": spread, "of": len(labels), "label": lab}


def family(ms: Dict[Tuple[int, str], np.ndarray], e1s: Dict[Tuple[int, str], np.ndarray],
           steps: Sequence[int] = FAMILY_STEPS) -> dict:
    """The transitions V counts at ``steps``: before / after m, their keys, and the counted skips."""
    A, B, keys, skipped_small, skipped_rise = [], [], [], 0, 0
    for s in steps:
        for p in V1:
            m, e1 = ms[(s, p)], e1s[(s, p)]
            for t in drops(e1):
                dm = float(m[t + 1] @ m[t + 1] - m[t] @ m[t])
                if abs(dm) < DM0:
                    skipped_small += 1
                elif dm >= 0:
                    skipped_rise += 1
                else:
                    A.append(m[t]), B.append(m[t + 1]), keys.append((s, p, int(t)))
    d = next(iter(ms.values())).shape[-1]
    return {"A": np.array(A).reshape(-1, d), "B": np.array(B).reshape(-1, d), "keys": keys,
            "skipped_small": skipped_small, "skipped_rise": skipped_rise}


# ---------------------------------------------------------------- model passes

def _npz(out: Path, seed: int, step: int) -> Path:
    return out / f"seed{seed}" / f"step{step}.npz"


def means(model, tok, device) -> dict:
    """Per v1 prompt: m (25, 1024) float64, e1 (25,) by A5's path, the tokens' hash."""
    import torch
    out = {"m": [], "e1": [], "ids_sha": [], "n": []}
    for p in V1:
        hs, ids = cm._hidden(model, tok, p, device)
        rows = []
        for h in hs:
            H = h.double()
            rows.append((H / H.norm(dim=1, keepdim=True)).mean(dim=0).cpu().numpy())
        out["m"].append(np.array(rows))
        out["e1"].append([cm.e1_a5(h.cpu().numpy()) for h in hs])
        out["ids_sha"].append(hashlib.sha256(str(ids).encode()).hexdigest()[:12])
        out["n"].append(len(ids))
        del hs
    torch.cuda.empty_cache()
    return {"m": np.array(out["m"]), "e1": np.array(out["e1"]), "ids_sha": out["ids_sha"], "n": out["n"]}


def check_against_unit1(seed: int, step: int, r: dict) -> dict:
    """Tokens, e1 and Σm² (and Σ over S) against unit 1's records; refuse on a miss."""
    sel = json.loads((DATA / C5U1 / f"seed{seed}" / "set.json").read_text())
    worst = {"e1": 0.0, "m2": 0.0, "m2_S": 0.0}
    for j, p in enumerate(V1):
        ref = json.loads((DATA / C5U1 / f"seed{seed}" / "records" / f"step{step}_{p}.json").read_text())
        if r["ids_sha"][j] != ref["ids_sha"]:
            raise MeanRefused(f"seed {seed} step {step} {p}: tokens differ from unit 1's record")
        m = r["m"][j]
        for key, mine, theirs in (("e1", r["e1"][j], ref["e1"]),
                                  ("m2", (m * m).sum(axis=1), ref["m2"]),
                                  ("m2_S", (m[:, sel["S"]] ** 2).sum(axis=1), ref["m2_set"]["S"])):
            mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
            tol = c.ENERGY_TOL if key == "e1" else M2_TOL[0]
            err = np.abs(mine - theirs)
            if np.any(err > tol * np.abs(theirs) + (0 if key == "e1" else M2_TOL[1])):
                raise MeanRefused(f"seed {seed} step {step} {p}: {key} off unit 1's by {err.max():.2g}")
            worst[key] = max(worst[key], float(np.max(err / np.maximum(np.abs(theirs), 1e-300))))
    return worst


def _meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C5.md \"Unit 2\"", "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "tokenizer": list(TOKENIZER), "dtype": "float32 forward, float64 m",
            "device": torch.cuda.get_device_name(0), "tf32": False, "torch": torch.__version__,
            "hf_home": os.environ.get("HF_HOME")}


def run_pass(a) -> int:
    if not (Path(a.out) / "gate" / "gate.json").exists() or \
            not json.loads((Path(a.out) / "gate" / "gate.json").read_text())["passed"]:
        raise MeanRefused("the gate has not passed: no grid-step pass")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for seed in a.seeds:
        _npz(out, seed, 0).parent.mkdir(parents=True, exist_ok=True)
        (out / f"seed{seed}" / "plan.json").write_text(json.dumps(_meta(torch), indent=1) + "\n")
        for step in STEPS:
            f = _npz(out, seed, step)
            if f.exists():
                continue
            model = cs.load(repo_for(seed), step, device)
            r = means(model, tok, device)
            del model
            if not (np.all(np.isfinite(r["m"])) and np.all(np.isfinite(r["e1"])) and r["m"].shape == (7, 25, 1024)
                    and np.all((r["m"] ** 2).sum(-1) > 0)):
                raise MeanRefused(f"seed {seed} step {step}: not populated")
            dev = check_against_unit1(seed, step, r)
            np.savez(f, m=r["m"], e1=r["e1"], ids_sha=np.array(r["ids_sha"]), n=np.array(r["n"]),
                     code=code_sha(), seed=seed, step=step, dev=json.dumps(dev))
            if first:
                m2 = (r["m"] ** 2).sum(-1)
                print(f"first record populated: seed {seed} step {step}, m2[0, 12] {m2[0, 12]:.6f}, "
                      f"devs {dev}", flush=True)
                first = False
        print(f"seed {seed} pass done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- the gate

def gate(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out = Path(a.out) / "gate"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    model = cs.load(c.MODEL, SCALE_STEP, device)
    r = means(model, tok, device)
    del model
    ms = {(SCALE_STEP, p): r["m"][j] for j, p in enumerate(V1)}
    e1s = {(SCALE_STEP, p): r["e1"][j] for j, p in enumerate(V1)}
    fam = family(ms, e1s, steps=(SCALE_STEP,))
    T = len(fam["keys"])
    if T < 20:
        raise MeanRefused(f"step {SCALE_STEP}: {T} falls, fewer than 20")
    A, B = fam["A"], fam["B"]
    k_trained = k_half(A, B)
    Qn = rotations(N_ROT, A.shape[1], ROT_SEED + 0)             # the null's, as for seed 0
    gate_rng = np.random.default_rng(GATE_ROT_SEED)
    from scipy.stats import ortho_group
    At, Bt = [], []
    for t in range(T):                                          # a fresh turn per transition
        Q = ortho_group.rvs(A.shape[1], random_state=gate_rng)
        At.append(Q @ A[t]), Bt.append(Q @ B[t])
    At, Bt = np.array(At), np.array(Bt)
    prng = np.random.default_rng(GATE_PLANT_SEED)
    coords = [sorted(prng.choice(A.shape[1], size=N_PLANT, replace=False).tolist()) for _ in range(T)]
    arms = {}
    for f in DOSES:
        Ap, Bp = plant(At, Bt, coords, f)
        k = k_half(Ap, Bp)
        fr, _ = f_rot(Ap, Bp, Qn)
        arms[f"{f:g}"] = {"k_half": k.tolist(), "few": float(np.mean(k <= FEW)),
                          "rot": float(np.mean(fr <= F_ROT)), "median_k": float(np.median(k))}
    null, eff = arms["0"], arms[f"{GATE_DOSE:g}"]
    passed = (null["few"] <= GATE_NULL_FEW and null["rot"] <= GATE_NULL_ROT
              and eff["few"] >= GATE_EFF and eff["rot"] >= GATE_EFF)
    fr_tr, _ = f_rot(A, B, Qn)
    res = {"code": code_sha(), "step": SCALE_STEP, "n_transitions": T,
           "skipped_small": fam["skipped_small"], "skipped_rise": fam["skipped_rise"],
           "trained": {"k_half": k_trained.tolist(), "median_k": float(np.median(k_trained)),
                       "rot": float(np.mean(fr_tr <= F_ROT))},
           "arms": arms, "passed": bool(passed),
           "bars": {"null_few_max": GATE_NULL_FEW, "null_rot_max": GATE_NULL_ROT, "effect_min": GATE_EFF,
                    "dose": GATE_DOSE}}
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    pf = {"statistic": "k_half: the fewest coordinates holding half a fall in |m|^2", "direction": "less",
          "nulls": {"turned": null["k_half"]}, "moved_marks": [],
          "trained": k_trained.tolist(), "trained_step": str(SCALE_STEP),
          "scored_steps": [str(s) for s in FAMILY_STEPS],
          "planted": {str(i): arms[f"{f:g}"]["k_half"] for i, f in enumerate(DOSES)},
          "bounds": [1, A.shape[1]]}
    (out / "preflight_input.json").write_text(json.dumps(pf) + "\n")
    print(json.dumps({"T": T, "trained_median_k": res["trained"]["median_k"], "trained_rot": res["trained"]["rot"],
                      **{f: {k: v for k, v in x.items() if k != "k_half"} for f, x in arms.items()},
                      "passed": passed}), flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------- reading

def _load(out: Path, seed: int):
    ms, e1s, codes = {}, {}, set()
    for s in STEPS:
        z = np.load(_npz(out, seed, s))
        codes.add(str(z["code"]))
        for j, p in enumerate(V1):
            ms[(s, p)], e1s[(s, p)] = z["m"][j], z["e1"][j]
    return ms, e1s, codes


def seed_reading(ms, e1s, seed: int) -> dict:
    fam = family(ms, e1s)
    A, B = fam["A"], fam["B"]
    Qs = rotations(N_ROT, A.shape[1], ROT_SEED + seed)
    k = k_half(A, B)
    fr, kr = f_rot(A, B, Qs)
    steps = np.array([s for s, _, _ in fam["keys"]])
    sets = [k_half_set(A[t], B[t]) for t in range(len(k))]
    sel = json.loads((DATA / C5U1 / f"seed{seed}" / "set.json").read_text())
    rec = recurring(sets)
    level = {}
    Ql = Qs[:N_ROT_LEVEL]
    for s in STEPS:
        M = np.array([ms[(s, p)] for p in V1])                  # (7, 25, d)
        flat = M.reshape(-1, M.shape[-1])
        kl = k_half_level(flat).reshape(7, 25)
        pr = participation(flat).reshape(7, 25)
        klr = np.median([k_half_level(flat @ Q.T) for Q in Ql])
        prr = np.median([participation(flat @ Q.T) for Q in Ql])
        level[str(s)] = {"k_half_median_by_state": np.median(kl, axis=0).tolist(),
                         "pr_median_by_state": np.median(pr, axis=0).tolist(),
                         "k_half_median": float(np.median(kl)), "pr_median": float(np.median(pr)),
                         "rotated_k_half_median": float(klr), "rotated_pr_median": float(prr)}
    med = float(np.median(k))
    return {"n_transitions": int(len(k)), "skipped_small": fam["skipped_small"], "skipped_rise": fam["skipped_rise"],
            "Q1": {"median_k_half": med, "label": label_q1(med),
                   "rotated_median_k_half": float(np.median(kr)),
                   "quartiles": np.percentile(k, [25, 75]).tolist(), "min": int(k.min()), "max": int(k.max()),
                   "frac_few": float(np.mean(k <= FEW)),
                   "by_step": {str(s): ({"n": int((steps == s).sum()), "median_k_half": float(np.median(k[steps == s])),
                                         "rotated_median_k_half": float(np.median(kr[:, steps == s]))}
                                        if (steps == s).any() else {"n": 0}) for s in FAMILY_STEPS}},
            "Q2": {"median_f_rot": float(np.median(fr)), "frac_f_rot_le": float(np.mean(fr <= F_ROT)),
                   "basis_aligned": bool(np.mean(fr <= F_ROT) >= 0.5)},
            "Q3": {"recurring": rec, "recurring_in_S": [i for i, _ in rec if i in sel["S"]],
                   "recurring_in_M": [i for i, _ in rec if i in sel.get("M", [])], "S": sel["S"], "M": sel.get("M"),
                   "level": level}}


def read(a) -> int:
    out = Path(a.out)
    per, codes = {}, {}
    for seed in ALL_SEEDS:
        ms, e1s, cds = _load(out, seed)
        codes[seed] = sorted(cds)
        per[seed] = seed_reading(ms, e1s, seed)
        q = per[seed]
        print(seed, "T", q["n_transitions"], "k½ med", q["Q1"]["median_k_half"], q["Q1"]["label"],
              "rot", q["Q1"]["rotated_median_k_half"], "| f_rot≤.05", round(q["Q2"]["frac_f_rot_le"], 3),
              "| recurring", q["Q3"]["recurring"][:5], flush=True)
    if len({tuple(v) for v in codes.values()}) != 1:
        raise MeanRefused(f"records from more than one code version: {codes}")
    dev = per.pop(0)
    res = {"code": code_sha(), "record_codes": codes, "tally_Q1": tally({s: r["Q1"]["label"] for s, r in per.items()}),
           "basis_aligned": sum(r["Q2"]["basis_aligned"] for r in per.values()),
           "per_seed": per, "seed0_development": dev}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"Q1": res["tally_Q1"], "basis_aligned_of_9": res["basis_aligned"]}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("gate", "pass", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd == "pass":
            sp.add_argument("--seeds", type=int, nargs="+", default=list(ALL_SEEDS))
    a = ap.parse_args(argv)
    return {"gate": gate, "pass": run_pass, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
