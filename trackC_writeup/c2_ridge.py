"""The energy ridgeline's record (the user's, 2026-10-28; figure 12 in `numbers.py paper-figures`).
Description only: nothing registered, no threshold, no date. No forward pass: an embedding lookup.

Two things, in one record:

- **the ridges**: seed 0's E₁ at the 25 states, every step of the grid and every prompt, from A5's
  seed-0 records (field `e1`); each transition's count of prompts that drop (A5's rule, REL_TOL);
  V per step. Refuses unless V equals A5's known answer at every step.
- **the identity reference**: each prompt's step-0 embedding rows (the records' state 0: the
  token ids must hash to the record's `ids_sha`, and E at t = 0 must equal its `e1[0]` to 1e-5) run
  under ẋᵢ = P_{xᵢ}(Σⱼ softmaxⱼ(β⟨xᵢ, xⱼ⟩) xⱼ), Q = K = V = I, β = 1 [Geshkovski et al. 2023], Euler
  steps of DT renormalised to the sphere, to t = T, read at 25 equally spaced times (as 410m's 25
  states; t is not a layer count). The full mask is the case E_β is proved non-decreasing in, and
  refuses if E ever falls at a fine step; the causal mask (the model's own) is not covered, and its
  smallest fine-step change is recorded. Rows are taken in an orthonormal basis of their span
  (exact: the velocity stays in the span), so each prompt is an n × n ODE.

CPU, seconds: an n × n ODE per prompt and mask, n ≤ 512 (no GPU needed for 2 × 7 small ODEs).

Run: METS_DATA=<main>/data HF_HUB_OFFLINE=1 python -m trackC_writeup.c2_ridge
"""
from __future__ import annotations

import hashlib
import json
from typing import List

import numpy as np

from core.config import PROMPTS
from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackA_force.a5_seeds import MAX_TOKENS, REL_TOL, TOKENIZER

RECS = DATA / "trackA" / "a5" / "2026-10-20-u1" / "seed0" / "records"
GATE0 = DATA / "trackA" / "a5" / "2026-10-20-u1" / "seed0" / "gate.json"
OUT = DATA / "trackC" / "c2ridge" / "2026-10-28"
BETA, T, N_FINE, STATES = 1.0, 8.0, 2400, 25    # T as figure 6's toy; 100 fine steps between reads
MODEL = ("EleutherAI/pythia-410m", "step0")


class RidgeRefused(SystemExit):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_ridge.py", "trackA_force/a1a.py", "trackA_force/a5_seeds.py", "core/config.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def inputs_sha() -> str:
    """sha256 over the seed-0 record files, in name order (as `c2_readings.inputs_sha`)."""
    h = hashlib.sha256()
    for f in sorted(RECS.glob("step*_*.json")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- pure

def drops(e: np.ndarray) -> np.ndarray:
    """D2 (a): transition t is a drop when E₁ falls by more than REL_TOL of its value."""
    return (-np.diff(e) / np.maximum(np.abs(e[:-1]), 1e-12)) > REL_TOL


def energy(c: np.ndarray, beta: float = BETA) -> float:
    return float(np.exp(beta * (c @ c.T)).sum() / (2 * beta * len(c) ** 2))


def identity_run(c: np.ndarray, causal: bool, beta: float = BETA, t_end: float = T,
                 n_fine: int = N_FINE, states: int = STATES):
    """E_β at `states` equally spaced times of the identity-weight flow from rows `c` (unit norm),
    and the smallest change of E over the fine steps."""
    n, dt = len(c), t_end / n_fine
    every = n_fine // (states - 1)
    if every * (states - 1) != n_fine:
        raise RidgeRefused(f"refusing: {n_fine} fine steps do not divide into {states - 1} reads")
    mask = np.tril(np.ones((n, n), bool)) if causal else np.ones((n, n), bool)
    fine: List[float] = []
    reads: List[float] = []
    for i in range(n_fine + 1):
        g = c @ c.T
        fine.append(float(np.exp(beta * g).sum() / (2 * beta * n * n)))
        if i % every == 0:
            reads.append(fine[-1])
        if i == n_fine:
            break
        s = np.where(mask, beta * g, -np.inf)
        w = np.exp(s - s.max(axis=1, keepdims=True))
        v = (w / w.sum(axis=1, keepdims=True)) @ c
        v -= (v * c).sum(axis=1, keepdims=True) * c
        c = c + dt * v
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    return reads, float(np.min(np.diff(fine)))


def span_coords(x: np.ndarray) -> np.ndarray:
    """Unit rows `x` (n × d) in an orthonormal basis of their span: C Cᵀ = x xᵀ, C is n × rank."""
    q, r = np.linalg.qr(x.T)
    return r.T.copy()


# ---------------------------------------------------------------- run

def ridges() -> dict:
    v_known = json.loads(GATE0.read_text())["known_answer"]["V"]
    e, cnt, v = {}, [], []
    for s in STEPS:
        rows = {p: json.loads((RECS / f"step{s}_{p}.json").read_text()) for p in V1}
        e[str(s)] = {p: rows[p]["e1"] for p in V1}
        d = np.sum([drops(np.asarray(rows[p]["e1"], float)) for p in V1], axis=0)
        cnt.append(d.astype(int).tolist())
        v.append(int(d.sum()))
    if v != list(v_known):
        raise RidgeRefused(f"refusing: V from the records {v} is not A5's known answer {v_known}")
    return {"e1": e, "drops": cnt, "V": v}


def identity() -> dict:
    import torch
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    emb = GPTNeoXForCausalLM.from_pretrained(MODEL[0], revision=MODEL[1], torch_dtype=torch.float32
                                             ).gpt_neox.embed_in.weight.detach()
    out = {}
    for p in V1:
        rec = json.loads((RECS / f"step0_{p}.json").read_text())
        ids = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"][0].tolist()
        if hashlib.sha256(str(ids).encode()).hexdigest()[:12] != rec["ids_sha"]:
            raise RidgeRefused(f"refusing: {p}'s token ids are not the record's")
        x = emb[ids].double().numpy()
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        c = span_coords(x)
        e0 = energy(c)
        if abs(e0 - rec["e1"][0]) > 1e-5:
            raise RidgeRefused(f"refusing: {p}'s E at t = 0 is {e0}, the record's e1[0] {rec['e1'][0]}")
        row = {"n": len(ids), "e0": e0, "e1_0_record": rec["e1"][0]}
        for m in ("full", "causal"):
            row[m], row[f"{m}_min_fine_diff"] = identity_run(c.copy(), m == "causal")
        if row["full_min_fine_diff"] < -1e-12:
            raise RidgeRefused(f"refusing: E fell under the full mask on {p} ({row['full_min_fine_diff']:.2e}), "
                               "which the theorem forbids: the integrator is wrong")
        out[p] = row
        print(f"{p}: n {len(ids)}, E0 {e0:.5f}, full {row['full'][-1]:.4f} (min dE {row['full_min_fine_diff']:.1e}), "
              f"causal {row['causal'][-1]:.4f} (min dE {row['causal_min_fine_diff']:.1e})", flush=True)
    return out


def main() -> int:
    rec = {"code": code_sha(), "producer": "python -m trackC_writeup.c2_ridge",
           "inputs": {"records": str(RECS.relative_to(DATA)), "sha256": inputs_sha(),
                      "V_checked_against": str(GATE0.relative_to(DATA)), "model": list(MODEL),
                      "tokenizer": list(TOKENIZER)},
           "steps": list(STEPS), "prompts": list(V1), "rel_tol": REL_TOL,
           "identity": {"beta": BETA, "T": T, "n_fine": N_FINE, "states": STATES},
           "ridges": ridges()}
    rec["identity"]["prompts"] = identity()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ridge.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {OUT / 'ridge.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
