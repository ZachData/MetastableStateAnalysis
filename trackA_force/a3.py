"""trackA_force/a3.py — A3 units 2–5: the producer, the rows, the locator and its gate, the
reading (`design-A3.md` §2–§4; frozen 2026-10-19 as revised, Blocked 70 (ii) (a); unit 3 by
Blocked 72, the user's: E3 at +1, the floor at max(4, the row's own), E7, E8, E11; unit 4: E6u,
E6b; unit 5 by Blocked 73 and 74, the user's: E6b′, the pairs, the order, the figure's lanes).

Rows built here, on the 18 distinct steps (`a1a.STEPS`; steps 0 and 1 are one point):

- **E1** energy violations, Σ nViolRel at β = 1 over the 7 v1 prompts, the panel's
  ``p1_energy_nviol_b1``; located by `design-A5.md` D4 (a) exactly (half the median over the
  1000–4000 plateau), not by §4. The per-prompt counts from each step's Phase 1 report must sum
  to the panel's.
- **E2** OV: median over 384 heads of ``attractive_energy_fraction_core`` (down; σ over heads).
- **E3** induction, re-defined by Blocked 72 (ii) (a): the pooled attention of ``L8H6`` and
  ``L11H14`` (the members the 2026-10-18 correction names as induction heads at the successor)
  at the standard induction pairs (``induction_candidates(strict=True)``: the key follows an
  earlier copy of the query's token), pair-count-weighted over the 7 prompts and averaged over
  the two (up; σ over the 14 (prompt, member) units). *Beside, refused as defined:* the frozen E3
  (five members at offset 0, ``strict=False``), which must still reproduce
  ``behavioural_series.json`` per member.
- **E4** previous-token score of ``L5H2``: mean attention from query i to key i − 1, i ≥ 1
  (`p7d_redundancy.prev_token_profile`'s ``prev_all``), mean over prompts (up; σ over prompts).
  B1's records hold a signature, not this score, so it is read from ``attentions.npz``.
- **E5** effective rank on the sphere (``MinRank``), mean over prompts (down; σ over prompts).
  Must reproduce the panel's ``p1_eff_rank_normed_mean``.
- **E7** weight change ``‖W_t − W_0‖_F / ‖W_0‖_F`` and **E8** the ESD tail exponent α, per matrix
  type and layer (`a3_weights.py`); located on the median over the four types of the median over
  24 layers (up / down; σ over layers, the same layers resampled for every type).
- **E6u, E6b** KL(q ‖ p_model) of a unigram and a Witten–Bell bigram predictor (from 2M tokens
  of the Pile's validation split) to the model's next-token distribution, mean over 64 blocks of
  1024 held-out tokens (`a3_ngram.py`; σ over blocks); turning-point rows, located at the minimum.
- **E6b′** (unit 5, Blocked 74 (a)): E6b with the bigram counted on table set′, every document
  of the split outside the scored set (``--ngram-full``); gated and located as E6b, after every
  other row so their gate draws stay the same. A second date for E6b, not a row of its own.
- **E11** the causal simulator's cluster count at t = 200 from the measured start at the prompt's
  β (`a3_sim.py`), mean over prompts (down; σ over prompts).
- **E10** the LR schedule (warmup to 1430, cosine to 0.1×), analytic, the confound row, located
  by §4 with no floor.

**The locator (§4).** Δ(s) = x(s) − x(0), signed in the declared direction; the reference is the
largest Δ over steps ≤ 4000; the change step is the first with Δ ≥ reference / 2. Turning-point
rows (E6u, E6b) are located at the minimum over ≤ 4000, the floor on the fall into it. **No
change** if the reference is ≤ 0 or below ``FLOOR`` σ, σ the bootstrap SE (``N_BOOT`` resamples
over the row's units) of Δ at the reference step; **low power** below ``LOW_POWER`` σ.
`tools/math_checks/a3_locator.py` states its floor in units of per-step noise; σ here is the SE of
Δ, which under its iid model is √2 times that, so the floor as built is the stricter of the two
readings. The gate measures it at each row's own design.

**The gate (§4), before any real row is located** (a failed arm refuses its rows; nothing is
tuned after it). Per floored row, at its own units, statistic and per-step spread (τ_s, the
across-unit SD of the row's real Δ_u(s); step 0 carries none, so the Δs are independent across
steps, a conservative null). As revised by Blocked 71 (the user's, 2026-10-19): the row's own
**floor** is the 95 % upper bound of the 95th percentile of ref / σ over ``N_GATE`` null series, and the *null* arm checks
it on ``N_GATE`` fresh ones (≥ 95 % "no change"); the *effect* arm plants a logistic in log2(step)
of scale 0.25 octave centred in each gap 2–4 … 2000–4000 at 9 × σ at the reference step (§4's σ,
read literally: (i) (a)), placed there in ≥ 90 %; from unit 3 the floor used is max(4, that
bound) (Blocked 72 (iii) (a)), and the null arm and the power are taken at the floor used; the
row's **power** per gap is the least amplitude in ``AMPS`` reaching 90 % (line 102, re-derived),
and a located row below it at its gap is **low power**. A located row whose reference is 4000 is marked **censored** ((ii)). *Anchors:*
E1 at 512, E2 at 512, E10 at 1000. Turning-point rows (unit 4) plant a dip instead: a fall
centred in the gap and a rise of half of it centred in the next, placed right at the planted
minimum (`design-A3.md` §4 "Placed before any unit 4 output").

Beside, described and not read: `core.changepoint_colocation.change_profile`'s change-mass
centroid per row, flagged where it and the rule are ≥ 2 grid steps apart.

**The reading (unit 5; §3, §4 "Placed before any unit 5 output").** The eleven pairs by grid
distance: ≤ 1 aligned, ≥ 2 misaligned with the order; unread if either row is "no change",
refused or censored. E6b and E8 carry two dates (E6b′; E8 from its peak, which must be unit 3's
512), and a pair with either is read only where both dates give one verdict. Then the order of
the located rows and the figure's lanes (Δ / ref on the 18 steps).

Tier 1: exploratory, unregistered; v1 prompts only. CPU: numpy bootstraps over at most 384 units,
minutes; no forward pass here (E7, E8 and E11 are read from their producers' records).
    python -m trackA_force.a3 run --out <dir> --weights <a3_weights dir> --sim <a3_sim dir> \
        --ngram <a3_ngram dir> [--ngram-full <a3_ngram --scored dir>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .a1a import DATA, REPO, STEPS, V1, prompts_hash, run_dir

MEMBERS = ((8, 6), (7, 8), (11, 14), (12, 5), (8, 9))  # L5H2's partners, design-A3 §2 E3 (frozen)
MEMBERS_P1 = ((8, 6), (11, 14))   # E3 at +1, Blocked 72 (ii) (a): the 2026-10-18 correction's two
PREV = (5, 2)
WIN = tuple(i for i, s in enumerate(STEPS) if s <= 4000)      # indices 0..12
FLOOR = 4.0          # calibrated (a3_locator.py: 3.93 for 95 %), in σ = SE of Δ
LOW_POWER = 9.0      # calibrated: 6σ (gap 2–4) to 9σ (gap 2000–4000) for 90 % placement
N_BOOT = 1000
N_GATE = 1000
GATE_NULL = 0.95
GATE_EFFECT = 0.90
WIDTH = 0.25         # placed: logistic scale, octaves
AMPS = (2, 3, 4, 6, 9, 12, 18, 27, 40, 60, 90)   # power grid, in units of σ̄_ref (Blocked 71)
WARMUP, TOTAL, LR_MIN = 1430, 143000, 0.1
SEED = 20261019
E2_REPRO = {0: 0.504, 512: 0.094}   # PROJECT.md §3.12 A, design-A3 §3 (ii)
X = np.log2(np.maximum(np.asarray(STEPS, float), 1.0)); X[0] = -1.0   # as a3_locator.py


class A3Refused(RuntimeError):
    pass


# ---------------------------------------------------------------- statistics on units
# units: (n, S, k). A statistic maps (..., n, S, k) -> (..., S).

def stat_mean(u):
    return u[..., 0].mean(axis=-2)


def stat_median(u):
    return np.median(u[..., 0], axis=-2)


def stat_pooled(u):
    """k = 2: (pair-sum, pair-count); Σ sums / Σ counts over units."""
    return u[..., 0].sum(axis=-2) / u[..., 1].sum(axis=-2)


def stat_medmed(u):
    """k = matrix types: the median over types of the median over units (layers)."""
    return np.median(np.median(u, axis=-3), axis=-1)


STATS: Dict[str, Callable] = {"mean": stat_mean, "median": stat_median, "pooled": stat_pooled,
                              "medmed": stat_medmed}
SIGN = {"up": 1.0, "down": -1.0}


# ---------------------------------------------------------------- the locator (pure)

def reference(x: np.ndarray, direction: str, kind: str = "half") -> int:
    """Index of the reference step in WIN (never 0)."""
    w = np.asarray(x, float)[list(WIN)]
    if kind == "half":
        d = SIGN[direction] * (w - w[0])
        return int(np.argmax(d[1:])) + 1
    if kind == "turn":
        return int(np.argmin(w[1:])) + 1
    raise A3Refused(f"unknown locator kind {kind!r}")


def place(x: np.ndarray, direction: str, sigma: Optional[float], kind: str = "half",
          floor: Optional[float] = FLOOR) -> dict:
    """The rule's reading. ``sigma`` None or ``floor`` None: no floor (E10)."""
    w = np.asarray(x, float)[list(WIN)]
    j = reference(x, direction, kind)
    if kind == "half":
        d = SIGN[direction] * (w - w[0])
    else:
        d = w[0] - w                                  # the fall into the minimum
    ref = float(d[j])
    out = {"ref_index": j, "ref_step": STEPS[j], "ref": ref, "sigma": sigma,
           "ref_over_sigma": (ref / sigma if sigma else None)}
    floored = floor is not None and sigma is not None
    if ref <= 0 or (floored and ref < floor * sigma):
        out.update(reading="no change", index=None, step=None, low_power=None)
        return out
    i = j if kind == "turn" else int(np.argmax(d[1:] >= ref / 2)) + 1
    out.update(reading="located", index=i, step=STEPS[i],
               low_power=(bool(ref < LOW_POWER * sigma) if floored else None))
    return out


def boot_se(units: np.ndarray, stat: Callable, j: int, rng: np.random.Generator,
            n_boot: int = N_BOOT) -> float:
    """Bootstrap SE of stat(j) − stat(0) over the units (paired across steps)."""
    n = units.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    sub = units[:, [0, j], :][idx]                   # (B, n, 2, k)
    s = stat(sub)                                     # (B, 2)
    return float(np.std(s[:, 1] - s[:, 0], ddof=1))


def locate_units(units: np.ndarray, stat: Callable, direction: str, rng, kind="half",
                 n_boot: int = N_BOOT, floor: Optional[float] = FLOOR) -> dict:
    x = stat(units)
    j = reference(x, direction, kind)
    return place(x, direction, boot_se(units, stat, j, rng, n_boot), kind, floor=floor)


def censored(r: dict) -> Optional[bool]:
    """Blocked 71 (ii): the window's maximum (or minimum) is at 4000, so the half-point is set by
    where the window was cut. None where the row was not located."""
    if r.get("reading") != "located":
        return None
    return r["ref_index"] == WIN[-1]


def d4a(v: np.ndarray) -> dict:
    """design-A5.md D4 (a): first grid step with V ≥ half the median of V over 1000–4000."""
    v = np.asarray(v, float)
    plateau = float(np.median([v[STEPS.index(s)] for s in (1000, 2000, 4000)]))
    if plateau <= 0:
        return {"reading": "no change", "plateau_median": plateau, "index": None, "step": None}
    i = int(np.argmax(v >= plateau / 2))
    return {"reading": "located", "plateau_median": plateau, "index": i, "step": STEPS[i]}


def lr(step) -> float:
    s = float(step)
    if s <= WARMUP:
        return s / WARMUP
    f = (s - WARMUP) / (TOTAL - WARMUP)
    return LR_MIN + (1 - LR_MIN) * 0.5 * (1 + math.cos(math.pi * f))


# ---------------------------------------------------------------- rows from records

def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def panel() -> Dict[int, dict]:
    p = DATA / "analysis" / "colocation_panel.csv"
    return {int(r["step"]): r for r in csv.DictReader(p.open())}


def report_rows(step: int) -> Dict[str, dict]:
    """Per v1 prompt: MinRank and nViolRel at β = 1, from the step's Phase 1 cross-run report."""
    rep = run_dir(step, V1[0]).parent / "llm_cross_run_report.txt"
    lines = rep.read_text().splitlines()
    out: Dict[str, dict] = {}
    i = next(k for k, l in enumerate(lines) if l.startswith("Model") and "MinRank" in l)
    for l in lines[i + 2:]:
        f = l.split()
        if len(f) < 14 or not f[0].startswith("pythia"):
            break
        if f[0] != f"pythia-410m-step{step}":
            raise A3Refused(f"{rep}: row for {f[0]}, expected step {step}")
        if f[1] in V1:
            out.setdefault(f[1], {})["min_rank"] = float(f[6])
    for l in lines:
        m = re.search(r"\|\s*(\w+)\s*\|\s*beta=1\.0:\s*nViolRel=(\d+)", l)
        if m and m.group(1) in V1:
            out.setdefault(m.group(1), {})["nviol_b1"] = int(m.group(2))
    miss = [p for p in V1 if set(out.get(p, {})) != {"min_rank", "nviol_b1"}]
    if miss:
        raise A3Refused(f"{rep}: no MinRank / nViolRel for {miss}")
    return out


def rows_e1_e5() -> dict:
    pan = panel()
    rep = {s: report_rows(s) for s in STEPS}
    v = np.array([float(pan[s]["p1_energy_nviol_b1"]) for s in STEPS])
    vsum = np.array([sum(rep[s][p]["nviol_b1"] for p in V1) for s in STEPS])
    if not np.array_equal(v, vsum):
        raise A3Refused(f"E1: the reports' per-prompt counts {vsum} do not sum to the panel's {v}")
    er = np.array([[rep[s][p]["min_rank"] for s in STEPS] for p in V1])[..., None]
    pm = np.array([float(pan[s]["p1_eff_rank_normed_mean"]) for s in STEPS])
    if not np.allclose(er[..., 0].mean(0), pm, rtol=5e-4, atol=0):
        raise A3Refused("E5: the reports' mean MinRank does not reproduce the panel's")
    return {"E1": v, "E5": er}


def row_e2() -> np.ndarray:
    o = json.loads((DATA / "analysis" / "ov_per_head_series.json").read_text())
    cols = [o["steps"].index(s) for s in STEPS]
    keys = [f"{l},{h}" for l in range(o["n_blocks"]) for h in range(o["n_heads"])]
    u = np.array([[o["per_head"][k]["attractive_energy_fraction_core"][c] for c in cols]
                  for k in keys])
    med = np.median(u, axis=0)
    for s, want in E2_REPRO.items():                  # §3.12 A's published medians
        if round(float(med[STEPS.index(s)]), 3) != want:
            raise A3Refused(f"E2: median at {s} is {med[STEPS.index(s)]:.4f}, §3.12 A has {want}")
    return u[..., None]


def _attn_one(step: int, prompt: str, tokenizer) -> dict:
    from core.battery_structure import induction_candidates
    from tools.run.behavioural import prompt_ids
    d = run_dir(step, prompt)
    a = np.load(d / "attentions.npz")["attentions"]
    n = a.shape[-1]
    ids = prompt_ids(tokenizer, prompt, n, d)
    pairs = induction_candidates(ids)
    q = np.array([p[0] for p in pairs]); k = np.array([p[1] for p in pairs])
    p1 = induction_candidates(ids, strict=True)
    q1 = np.array([p[0] for p in p1], int); k1 = np.array([p[1] for p in p1], int)
    i = np.arange(1, n)
    return {"member_sums": [float(a[l, h, q, k].astype(np.float64).sum()) for l, h in MEMBERS],
            "n_pairs": len(pairs),
            "p1_sums": [float(a[l, h, q1, k1].astype(np.float64).sum()) for l, h in MEMBERS_P1],
            "n_pairs_p1": len(p1),
            "prev": float(a[PREV[0], PREV[1], i, i - 1].astype(np.float64).mean())}


def rows_e3_e4(cache: Path) -> dict:
    """One pass over the 18 × 7 stored attention tensors, cached in ``cache``."""
    if cache.exists():
        rec = json.loads(cache.read_text())
    else:
        from transformers import AutoTokenizer
        from tools.run.behavioural import TOKENIZER_ID
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        rec = {}
        for s in STEPS:
            for p in V1:
                rec[f"{s}|{p}"] = _attn_one(s, p, tok)
            print(f"  attentions read: step {s}", flush=True)
        cache.write_text(json.dumps(rec) + "\n")
    for key in ("n_pairs", "n_pairs_p1"):
        n = {p: {rec[f"{s}|{p}"][key] for s in STEPS} for p in V1}
        if any(len(v) != 1 for v in n.values()):
            raise A3Refused(f"E3: a prompt's {key} changes across steps: {n}")
        if any(min(v) == 0 for v in n.values()):
            raise A3Refused(f"E3: a prompt has no pairs ({key}): {n}")
    e3 = np.array([[[rec[f"{s}|{p}"]["member_sums"][m], rec[f"{s}|{p}"]["n_pairs"]]
                    for s in STEPS] for p in V1 for m in range(len(MEMBERS))])
    beh = json.loads((DATA / "analysis" / "behavioural_series.json").read_text())
    cols = [beh["steps"].index(s) for s in STEPS]
    for m, (l, h) in enumerate(MEMBERS):
        mine = e3[m::len(MEMBERS)]
        got = mine[..., 0].sum(0) / mine[..., 1].sum(0)
        want = np.array(beh["series_excl_repeated"][f"{l},{h}"])[cols]
        if not np.allclose(got, want, rtol=1e-5, atol=1e-9):
            raise A3Refused(f"E3: L{l}H{h}'s pool does not reproduce behavioural_series.json")
    e4 = np.array([[rec[f"{s}|{p}"]["prev"] for s in STEPS] for p in V1])[..., None]
    p1 = np.array([[[rec[f"{s}|{p}"]["p1_sums"][m], rec[f"{s}|{p}"]["n_pairs_p1"]]
                    for s in STEPS] for p in V1 for m in range(len(MEMBERS_P1))])
    return {"E3": p1, "E3_offset0": e3, "E4": e4}


def rows_weights(wdir: Path) -> dict:
    """E7, E8 units (24 layers, 18 steps, 4 types); the 32-step medians beside."""
    from .a3_weights import ALL_STEPS, TYPES, load_steps
    recs = load_steps(wdir)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise A3Refused(f"weights rows: records from {len(codes)} codes {codes}")
    out = {"code": codes.pop(), "dense": {}}
    for rid, key in (("E7", "rel_change"), ("E8", "alpha")):
        out[rid] = np.array([[recs[s][key][t] for s in STEPS] for t in TYPES]).transpose(2, 1, 0)
        dense = np.array([[recs[s][key][t] for s in ALL_STEPS] for t in TYPES]).transpose(2, 1, 0)
        out["dense"][rid] = {"steps": list(ALL_STEPS),
                             "per_type": {t: np.median(dense[..., i], 0).tolist()
                                          for i, t in enumerate(TYPES)},
                             "row": stat_medmed(dense).tolist()}
    sr = {t: [None if s == 0 else float(np.median(recs[s]["dW_stable_rank"][t])) for s in ALL_STEPS]
          for t in TYPES}
    out["dense"]["dW_stable_rank"] = {"steps": list(ALL_STEPS), "per_type": sr}
    if not np.all(out["E7"][:, 0, :] == 0):
        raise A3Refused("E7: the weight change at step 0 is not 0")
    return out


def rows_sim(sdir: Path) -> dict:
    """E11 units (7 prompts, 18 steps, 1): the cluster count at T_FINAL; the trajectory beside."""
    from .a3_sim import T_FINAL, TIMES, load_cells
    recs = load_cells(sdir)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise A3Refused(f"E11: records from {len(codes)} codes {codes}")
    u = np.array([[recs[(s, p)]["traj"][str(T_FINAL)]["clusters"] for s in STEPS]
                  for p in V1], float)[..., None]
    traj = {str(T): np.mean([[recs[(s, p)]["traj"][str(T)]["clusters"] for s in STEPS]
                             for p in V1], 0).tolist() for T in TIMES}
    start = np.mean([[recs[(s, p)]["clusters_at_start"] for s in STEPS] for p in V1], 0).tolist()
    return {"E11": u, "code": codes.pop(), "beside": {"mean_clusters_at": traj,
                                                      "mean_clusters_at_start": start}}


def rows_ngram(ndir: Path) -> dict:
    """E6u, E6b units (64 blocks, 18 steps, 1), with the checks placed before output (refuse on
    failure); the model's loss per step beside."""
    from .a3_ngram import (LOSS_143000, STEP0_TOL, VOCAB, NgramRefused, load_steps, neg_entropy,
                           tables)
    recs = load_steps(ndir)
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise A3Refused(f"E6: records from {len(codes)} codes {codes}")
    u = {rid: np.array([recs[s][key] for s in STEPS]).T[..., None]
         for rid, key in (("E6u", "kl_u"), ("E6b", "kl_b"))}
    loss = np.array([recs[s]["loss"] for s in STEPS]).T                # (blocks, steps)
    lo, hi = LOSS_143000
    l_end = float(loss[:, STEPS.index(143000)].mean())
    if not lo <= l_end <= hi:
        raise A3Refused(f"E6: the loss at 143000 is {l_end:.3f}, outside {LOSS_143000}")
    try:
        z = np.load(ndir / "corpus.npz")
        man = json.loads((ndir / "corpus.json").read_text())
        for key in ("blocks", "table"):                    # the arrays the records were made on
            got = hashlib.sha256(z[key].tobytes()).hexdigest()[:12]
            if got != man[f"{key}_sha"]:
                raise A3Refused(f"E6: corpus.npz {key} is {got}, corpus.json has {man[f'{key}_sha']}")
        q_u = tables(z["table"])["q_u"]
    except (OSError, KeyError, NgramRefused) as e:
        raise A3Refused(f"E6: corpus tables: {e}")
    flat = neg_entropy(q_u) + math.log(VOCAB)                          # KL(q_u ‖ uniform)
    e6u0 = float(u["E6u"][:, 0, 0].mean())
    if abs(e6u0 - flat) > STEP0_TOL * flat:
        raise A3Refused(f"E6u at step 0 is {e6u0:.3f}, KL(q_u ‖ uniform) is {flat:.3f}")
    return {**u, "code": codes.pop(), "beside": {"loss": loss.mean(0).tolist(),
                                                 "kl_u_uniform": flat,
                                                 "device": sorted({r["device"] for r in recs.values()})}}


TURN = {"E6u", "E6b", "E6b_full"}   # turning-point rows (§4): the minimum, gated on a planted dip

ROWS = {  # id: (row, direction, statistic, σ over)
    "E2": ("OV: median attractive_energy_fraction_core, 384 heads", "down", "median", "heads"),
    "E3": ("induction at +1: L8H6, L11H14 pooled attention (strict pairs)", "up", "pooled",
           "prompts × members"),
    "E4": ("previous-token score of L5H2", "up", "mean", "prompts"),
    "E5": ("effective rank on the sphere (MinRank), mean over prompts", "down", "mean", "prompts"),
    "E7": ("weight change ‖W−W0‖/‖W0‖, median over types of median over layers", "up", "medmed",
           "layers"),
    "E8": ("ESD tail α, median over types of median over layers", "down", "medmed", "layers"),
    "E11": ("causal simulator's cluster count at t = 200, mean over prompts", "down", "mean",
            "prompts"),
    # after the unit 2–3 rows, so their gate draws stay the same
    "E6u": ("KL(unigram ‖ model), mean over 64 blocks", "turn", "mean", "blocks"),
    "E6b": ("KL(bigram ‖ model), mean over 64 blocks", "turn", "mean", "blocks"),
    # unit 5 (Blocked 74 (a)): last, so every earlier row's gate draws stay the same
    "E6b_full": ("E6b′: KL(bigram from table set′ ‖ model), mean over 64 blocks", "turn", "mean",
                 "blocks"),
}


def kind_of(rid: str) -> str:
    return "turn" if rid in TURN else "half"


# ---------------------------------------------------------------- the gate

def unit_scores(units: np.ndarray, stat_name: str) -> np.ndarray:
    """(n_units, S, k′) in the statistic's own units: per-pair means for the pooled row (k′ = 1);
    every type for the medmed rows."""
    if stat_name == "pooled":
        return (units[..., 0] / units[..., 1])[..., None]
    return units


def _synthetic(units: np.ndarray, direction: str, stat_name: str, rng, n: int,
               amp_shape: Optional[np.ndarray] = None) -> np.ndarray:
    """(n, n_units, S, k) flat noise at the row's per-step spread, plus a planted shape."""
    real = unit_scores(units, stat_name)
    tau = np.std(real - real[:, :1], axis=0, ddof=1)             # (S, k′), tau[0] = 0
    z = rng.standard_normal((n,) + real.shape) * tau
    if amp_shape is not None:
        z = z + amp_shape[:, None]
    if stat_name == "pooled":
        w = units[:, :1, 1]                                       # prompt pair counts
        z = z[..., 0]
        return np.stack([z * w, np.broadcast_to(w, z.shape)], axis=-1)
    return z


def boot_se_all(units: np.ndarray, stat: Callable, rng: np.random.Generator,
                n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap SE of stat(j) − stat(0) for every j in WIN, paired resamples; [0] = 0."""
    n = units.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    s = stat(units[:, list(WIN), :][idx])                        # (B, W)
    return np.std(s - s[:, :1], axis=0, ddof=1)


def _noise_draws(units: np.ndarray, rid: str, rng, n: int):
    """n flat series at the row's spread: their statistic (n, S) and Δ's SE at each WIN step.

    A planted shape is added to every unit alike, so it shifts each statistic (mean, median,
    pooled) by exactly the shape and leaves the bootstrap SE unchanged: one bootstrap per draw
    serves every gap and amplitude."""
    _, direction, sname, _ = ROWS[rid]
    stat = STATS[sname]
    syn = _synthetic(units, direction, sname, rng, n)
    xs = np.stack([stat(u) for u in syn])
    sig = np.stack([boot_se_all(u, stat, rng) for u in syn])
    return xs, sig


def _read(x: np.ndarray, sig: np.ndarray, direction: str, floor: float,
          kind: str = "half") -> dict:
    j = reference(x, direction, kind)
    return place(x, direction, float(sig[j]), kind, floor=floor)


def _ratio(x, sig, direction, kind: str = "half") -> float:
    """ref / σ at the reference step; −inf where ref ≤ 0 (always "no change")."""
    w = np.asarray(x, float)[list(WIN)]
    j = reference(x, direction, kind)
    ref = SIGN[direction] * (w[j] - w[0]) if kind == "half" else w[0] - w[j]
    return float(ref / sig[j]) if ref > 0 else -np.inf


def _shape(i: int) -> np.ndarray:
    centre = (X[i] + X[i + 1]) / 2
    sh = 1 / (1 + np.exp(-(X - centre) / WIDTH))
    return sh - sh[0]


def _dip(i: int) -> np.ndarray:
    """A turning-point row's planted shape (§4 "Placed before any unit 4 output"): a fall of 1
    centred in gap (i, i + 1), a rise of 1/2 centred in (i + 1, i + 2); its minimum is i + 1."""
    return -_shape(i) + 0.5 * _shape(i + 1)


def gate_row(rid: str, units: np.ndarray, rng) -> dict:
    """§4's gate as revised by Blocked 71 (decided 2026-10-19, the user's).

    (i) (a): the effect arm plants 9 × σ at the reference step (σ̄_ref, the largest mean null σ
    over the planted plateau's steps, where its reference can fall). Line 102, re-derived: the
    row's own floor is the 95 % upper bound of the 95th percentile of ref / σ over N_GATE
    null draws (derivation set),
    checked on an independent N_GATE (check set); its power is, per gap, the least amplitude in
    AMPS (units of σ̄_ref) placed right in ≥ 90 % at that floor (check set). The frozen 4σ null
    rate and unit 2's arm (9 × σ̄ at the step past the centre) are recomputed beside, as a check."""
    direction, kind = ROWS[rid][1], kind_of(rid)
    xd, sd = _noise_draws(units, rid, rng, N_GATE)
    xc, sc = _noise_draws(units, rid, rng, N_GATE)
    r = np.array([_ratio(x, s, direction, kind) for x, s in zip(xd, sd)])
    # the 95 % upper confidence bound of the 95th percentile, so a correct floor passes its check
    # on fresh draws with high probability (the bare percentile passes it about half the time:
    # unit 3's first run, E2 at 0.936)
    from scipy.stats import binom
    k = int(binom.ppf(GATE_NULL, len(r), GATE_NULL)) + 1
    own = float(np.ceil(np.sort(r)[min(k, len(r)) - 1] * 100) / 100)
    floor = max(FLOOR, own)                       # Blocked 72 (iii) (a), from unit 3 on
    rc = np.array([_ratio(x, s, direction, kind) for x, s in zip(xc, sc)])
    p_null = float(np.mean(rc < floor))
    p_null_frozen = float(np.mean(rc < FLOOR))
    sbar = sc.mean(axis=0)                                        # (W,)
    gaps = []
    for i in WIN[1:-1]:
        sref = float(sbar[i + 1:].max())
        sh = _dip(i) if kind == "turn" else SIGN[direction] * _shape(i)

        def placed(amp, fl):
            return float(np.mean([_read(x + amp * sh, s, direction, fl, kind)["index"] == i + 1
                                  for x, s in zip(xc, sc)]))
        ok = placed(LOW_POWER * sref, floor)
        curve = {a: placed(a * sref, floor) for a in AMPS}
        power = next((a for a in AMPS if curve[a] >= GATE_EFFECT), None)
        gaps.append({"gap": f"{STEPS[i]}-{STEPS[i + 1]}", "to_index": i + 1,
                     "sigma_ref": sref, "amplitude": LOW_POWER * sref, "placed_right": ok,
                     "pass": ok >= GATE_EFFECT, "power": power,
                     "curve": {str(a): v for a, v in curve.items()},
                     "unit2_arm": placed(LOW_POWER * float(sbar[i + 1]), FLOOR)})
    return {"floor": floor, "own_floor": own, "null_no_change": p_null, "null_pass": p_null >= GATE_NULL,
            "null_no_change_at_frozen_floor": p_null_frozen, "effect": gaps,
            "pass": p_null >= GATE_NULL and all(g["pass"] for g in gaps)}


def constant_row(units: np.ndarray, stat_name: str) -> bool:
    """Every unit equal to its step-0 value at every step (#235 finding 4): the gate cannot plant
    a move against zero noise, so the row is refused by name, not by a degenerate arm."""
    real = unit_scores(units, stat_name)
    return bool(np.all(real == real[:, :1]))


def low_power(r: dict, gate: dict) -> Optional[bool]:
    """Re-derived mark: ref / σ̄_ref (the gap's own unit, in which its power is stated) below the
    row's power at the gap it was located in; None where that gap was not gated (located at
    step 2). No amplitude in AMPS reaching 90 % marks it low power."""
    if r.get("reading") != "located":
        return None
    g = next((g for g in gate["effect"] if g["to_index"] == r["index"]), None)
    if g is None:
        return None
    if g["power"] is None:
        return True
    return bool(r["ref"] / g["sigma_ref"] < g["power"])


# ---------------------------------------------------------------- the reading (unit 5, pure)
# design-A3.md §3 and §4 "Placed before any unit 5 output" (Blocked 73, 74, the user's).

PAIRS = tuple(("E1", r) for r in ("E2", "E3", "E4", "E5", "E6u", "E6b", "E7", "E8", "E11")) + \
    (("E4", "E3"), ("E3", "E2"))
SECOND_DATE = {"E6b": "E6b_full", "E8": "E8_peak"}     # Blocked 74 (a), 73 (iii)
E8_PEAK_UNIT3 = 512                                    # unit 3's peak date; the code must give it
ORDERED = ("E1", "E2", "E3", "E4", "E5", "E6u", "E6b", "E7", "E8", "E11", "E10")


def from_peak(x: np.ndarray, direction: str) -> dict:
    """The half-move back from the row's opposite extreme (its peak for a down row) over ≤ 4000:
    the reference is the largest move in the declared direction after the peak within the
    window; the date the first step after the peak reaching half of it."""
    w = np.asarray(x, float)[list(WIN)]
    p = int(np.argmax(-SIGN[direction] * w))
    d = SIGN[direction] * (w[p + 1:] - w[p])
    if len(d) == 0 or d.max() <= 0:
        return {"reading": "no change", "index": None, "step": None, "peak_step": STEPS[p]}
    j = int(np.argmax(d)) + p + 1
    i = int(np.argmax(d >= d.max() / 2)) + p + 1
    return {"reading": "located", "index": i, "step": STEPS[i], "peak_step": STEPS[p],
            "ref_step": STEPS[j], "ref": float(d.max()), "censored": j == WIN[-1]}


def dates(rows: dict, rid: str):
    """(grid indices, None), one per date of the row, or (None, why it is unread)."""
    out = []
    for k in (rid,) + ((SECOND_DATE[rid],) if rid in SECOND_DATE else ()):
        r = rows.get(k)
        if r is None:
            return None, f"{k} not computed"
        if r["reading"] != "located":
            return None, f"{k} {r['reading']}"
        if r.get("censored"):
            return None, f"{k} censored"
        out.append(r["index"])
    return out, None


def verdict(i: int, j: int, a: str, b: str) -> tuple:
    """§3: equal or one grid step apart is aligned; two or more is misaligned, with the order."""
    if abs(i - j) <= 1:
        return ("aligned",)
    return ("misaligned", a if i < j else b)


def read_pair(rows: dict, a: str, b: str) -> dict:
    da, wa = dates(rows, a)
    db, wb = dates(rows, b)
    if da is None or db is None:
        return {"pair": [a, b], "verdict": "unread", "why": wa or wb}
    st = {a: [STEPS[i] for i in da], b: [STEPS[i] for i in db]}
    vs = {verdict(i, j, a, b) for i in da for j in db}
    if len(vs) > 1:
        two = a if a in SECOND_DATE else b
        return {"pair": [a, b], "steps": st, "verdict": "unread (table)" if two == "E6b"
                else "unread (two dates)", "why": f"{two}'s two dates give "
                + " and ".join(sorted(" ".join(v) for v in vs))}
    v = vs.pop()
    out = {"pair": [a, b], "steps": st, "verdict": v[0], "notes": []}
    if v[0] == "misaligned":
        out["first"] = v[1]
    else:
        lo, hi = min(da + db), max(da + db)
        out["notes"].append(f"aligned: one grid step either side, {STEPS[max(lo - 1, 0)]}–"
                            f"{STEPS[min(hi + 1, len(STEPS) - 1)]}")
    if max(st[a] + st[b]) <= 2000:
        out["notes"].append("inside warmup (E10)")
    if {a, b} == {"E1", "E2"}:
        out["notes"].append("a reproduction (§3 (ii)), not a finding")
    return out


def order(rows: dict) -> List[dict]:
    """The located, uncensored rows by step; equal date lists are ties (one entry)."""
    ents = {}
    for rid in ORDERED:
        d, _ = dates(rows, rid)
        if d:
            ents.setdefault(tuple(d), []).append(rid)
    return [{"steps": [STEPS[i] for i in k], "rows": v}
            for k, v in sorted(ents.items(), key=lambda kv: (min(kv[0]), max(kv[0])))]


def unigram_first(rows: dict) -> Optional[bool]:
    """Stated only where E6u is before both of E6b's dates; None where either is unread."""
    du, _ = dates(rows, "E6u")
    db, _ = dates(rows, "E6b")
    if not du or not db:
        return None
    return du[0] < min(db)


QUOTED = (  # design-A3 §2 "Quoted rows": drawn as bands, never located, never in a pair
    ("Belrose: n-gram-sequence loss trough", 64, 256),
    ("Belrose: rise after the trough", 256, 1000),
    ("RankMe collapse (410m)", 32, 256),
    ("RankMe recovery", 256, 1000),
    ("induction phase change (Olsson)", 1200, 2400),
    ("R6f embedding move (Phase 10)", 512, 2000),
    ("U2 regime change (1e)", 2000, 4000),
    ("A1: step 16 collapse at state 23", 16, 16),
    ("A2: MLP cancels attention", 1000, 4000),
    ("A2: fixed key, a period", 2000, 4000),
    ("A2: fixed key, a newline", 8000, 143000),
    ("A2: sink share rising", 1000, 143000),
)


def lane(r: dict, rid: str) -> dict:
    """The figure's lane: Δ / ref on the 18 steps (0 at step 0, 1 at the reference)."""
    x = np.asarray(r["series"], float)
    if rid == "E1":                                   # D4 (a): V / plateau median, half at 0.5
        y, how = x / r["plateau_median"], "V / plateau median (D4 (a))"
    elif r["dir"] == "turn":
        y, how = (x[0] - x) / r["ref"], "fall from step 0 / fall into the minimum"
    elif r.get("ref"):
        y, how = SIGN[r["dir"]] * (x - x[0]) / r["ref"], "Δ / ref"
    else:
        m = np.max(np.abs(x - x[0])) or 1.0
        y, how = SIGN[r["dir"]] * (x - x[0]) / m, "Δ / max |Δ| (not located)"
    out = {"id": rid, "row": r["row"], "y": y.tolist(), "how": how, "index": r.get("index"),
           "step": r.get("step"), "reading": r["reading"],
           "hollow": r["reading"] != "located" or bool(r.get("censored"))}
    if "dense" in r and r.get("ref"):
        dx = np.asarray(r["dense"]["row"], float)
        out["dense"] = {"steps": r["dense"]["steps"],
                        "y": (SIGN[r["dir"]] * (dx - dx[0]) / r["ref"]).tolist()}
    return out


def reading(rows: dict) -> dict:
    """Pairs, order and the figure's lanes, from the located rows."""
    lanes = [lane(rows[k], k) for k in ORDERED + ("E6b_full",) if k in rows]
    return {"pairs": [read_pair(rows, a, b) for a, b in PAIRS], "order": order(rows),
            "unigram_first": unigram_first(rows), "lanes": lanes,
            "quoted": [{"label": l, "from": a, "to": b} for l, a, b in QUOTED],
            "warmup_end": WARMUP, "steps": list(STEPS)}


# ---------------------------------------------------------------- run

def beside(x: np.ndarray, direction: str, idx: Optional[int]) -> dict:
    from core.changepoint_colocation import ColocationRefused, change_profile
    try:
        cp = change_profile(STEPS, list(x), "rise" if direction == "up" else "drop")
    except ColocationRefused as e:
        return {"centroid_step": None, "note": str(e)[:120]}
    c = cp["centroid_step"]
    near = int(np.argmin(np.abs(np.log2(np.maximum(STEPS, 1)) - math.log2(max(c, 1)))))
    return {"centroid_step": c, "nearest_grid": STEPS[near],
            "flag": (idx is not None and abs(near - idx) >= 2)}


def check_full(nf: dict, ndir: Path, fdir: Path) -> None:
    """design-A3 §4 "Placed before any unit 5 output": table set′ starts with unit 4's table, the
    blocks are unit 4's, the bigram′ is sharper on the scored set, and the forward pass is the
    same (the model's loss per block as unit 4's). Refuse on failure."""
    m4 = json.loads((ndir / "corpus.json").read_text())
    mf = json.loads((fdir / "corpus.json").read_text())
    if mf.get("first_n_table_sha") != m4["table_sha"] or mf["blocks_sha"] != m4["blocks_sha"]:
        raise A3Refused(f"E6b′: table set′ starts {mf.get('first_n_table_sha')}, blocks "
                        f"{mf['blocks_sha']}; unit 4 {m4['table_sha']}, {m4['blocks_sha']}")
    if set(sum(mf["block_docs"], [])) != set(sum(m4["block_docs"], [])):
        raise A3Refused("E6b′: the scored documents differ from unit 4's")
    l4 = json.loads((ndir / "predictor_losses.json").read_text())["bigram"]["loss"]
    lf = json.loads((fdir / "predictor_losses.json").read_text())["bigram"]["loss"]
    if not lf < l4:
        raise A3Refused(f"E6b′: the bigram′'s own loss {lf:.3f} is not below the 2M bigram's {l4:.3f}")
    from .a3_ngram import load_steps
    a, b = load_steps(ndir), load_steps(fdir)
    gap = max(abs(x - y) for s in STEPS for x, y in zip(a[s]["loss"], b[s]["loss"]))
    if gap > 1e-4:
        raise A3Refused(f"E6b′: the model's loss differs from unit 4's by {gap:.2e} nats")
    nf["beside"].update(bigram_loss=lf, bigram_loss_2m=l4, loss_gap_to_unit4=gap,
                        unigram_loss=json.loads((fdir / "predictor_losses.json").read_text())
                        ["unigram"]["loss"])


def run(out: Path, wdir: Path, sdir: Path, ndir: Path, fdir: Optional[Path] = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec = {"code": _sha(Path(__file__)), "prompts_hash": prompts_hash(), "steps": list(STEPS),
           "inputs": {f: _sha(DATA / "analysis" / f) for f in
                      ("colocation_panel.csv", "ov_per_head_series.json",
                       "behavioural_series.json")},
           "constants": {"FLOOR": FLOOR, "LOW_POWER": LOW_POWER, "N_BOOT": N_BOOT,
                         "N_GATE": N_GATE, "WIDTH": WIDTH, "SEED": SEED, "AMPS": list(AMPS)}}
    r15 = rows_e1_e5()
    wr, sr, nr = rows_weights(wdir), rows_sim(sdir), rows_ngram(ndir)
    nf = None
    if fdir is not None:                                   # unit 5: E6b′ (Blocked 74 (a))
        nf = rows_ngram(fdir)
        check_full(nf, ndir, fdir)
        rec["inputs"].update({"a3_ngram_full_code": nf["code"], "a3_ngram_full_dir": str(fdir)})
    rec["inputs"].update({"a3_weights_code": wr["code"], "a3_sim_code": sr["code"],
                          "a3_ngram_code": nr["code"], "a3_weights_dir": str(wdir),
                          "a3_sim_dir": str(sdir), "a3_ngram_dir": str(ndir)})
    units = {"E2": row_e2(), "E5": r15["E5"], **rows_e3_e4(out / "attn_rows.json"),
             "E7": wr["E7"], "E8": wr["E8"], "E11": sr["E11"], "E6u": nr["E6u"],
             "E6b": nr["E6b"]}
    if nf is not None:
        units["E6b_full"] = nf["E6b"]
    print(f"rows built ({time.time() - t0:.0f} s)", flush=True)

    # gate first: nothing real is located before every arm is recorded
    rec["gate"] = {}
    for rid in (k for k in ROWS if k in units):
        if constant_row(units[rid], ROWS[rid][2]):
            rec["gate"][rid] = {"pass": False, "refused": "constant: no spread at any step",
                                "floor": None, "null_no_change": None, "effect": []}
            print(f"gate {rid}: refused, constant row", flush=True)
            continue
        rec["gate"][rid] = gate_row(rid, units[rid], rng)
        g = rec["gate"][rid]
        print(f"gate {rid}: floor {g['floor']:.2f}, null {g['null_no_change']:.3f}, effect min "
              f"{min(e['placed_right'] for e in g['effect']):.3f} -> "
              f"{'pass' if g['pass'] else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    (out / "a3_gate.json").write_text(json.dumps(rec, indent=1) + "\n")

    rows = {}
    e1 = d4a(r15["E1"])
    rows["E1"] = {"row": "energy violations, Σ nViolRel at β = 1", "dir": "up",
                  "series": r15["E1"].tolist(), **e1, "located_by": "design-A5 D4 (a)"}
    e10 = np.array([lr(s) for s in STEPS])
    rows["E10"] = {"row": "LR schedule (confound)", "dir": "up", "series": e10.tolist(),
                   **place(e10, "up", None, floor=None), "located_by": "§4, no floor"}
    for rid, (name, direction, sname, over) in ROWS.items():
        if rid not in units:
            continue
        x = STATS[sname](units[rid])
        r = {"row": name, "dir": direction, "sigma_over": over, "n_units": int(units[rid].shape[0]),
             "series": x.tolist()}
        if not rec["gate"][rid]["pass"]:
            r.update(reading="refused", step=None, index=None,
                     why=rec["gate"][rid].get("refused", "a gate arm failed"))
        else:
            g = rec["gate"][rid]
            r.update(locate_units(units[rid], STATS[sname], direction, rng, kind_of(rid),
                                  floor=g["floor"]))
            r["low_power_frozen"] = r["low_power"]           # ref / σ < 9, as frozen
            r["low_power"] = low_power(r, g)
            r["censored"] = censored(r)
            r["reading_at_frozen_floor"] = ("located" if r["ref_over_sigma"] >= FLOOR
                                            else "no change")
        rows[rid] = r
    # beside, not located (Blocked 72 (ii) (a)): the frozen E3 at offset 0, refused as defined
    e30 = units["E3_offset0"]
    rows["E3_offset0"] = {"row": "frozen E3: five members at offset 0 (strict=False)", "dir": "up",
                          "series": stat_pooled(e30).tolist(), "reading": "refused (definition)",
                          "step": None, "index": None,
                          "per_member": {f"L{l}H{h}": stat_pooled(e30[m::len(MEMBERS)]).tolist()
                                         for m, (l, h) in enumerate(MEMBERS)}}
    rows["E3"]["per_member"] = {f"L{l}H{h}": stat_pooled(units["E3"][m::len(MEMBERS_P1)]).tolist()
                                for m, (l, h) in enumerate(MEMBERS_P1)}
    rows["E7"]["dense"], rows["E8"]["dense"] = wr["dense"]["E7"], wr["dense"]["E8"]
    rows["E7"]["dW_stable_rank_beside"] = wr["dense"]["dW_stable_rank"]
    rows["E11"]["trajectory_beside"] = sr["beside"]
    rows["E6u"]["model_loss_beside"] = nr["beside"]
    if nf is not None:
        rows["E6b_full"]["beside_full"] = {**nf["beside"], "E6u_full_series":
                                           stat_mean(nf["E6u"]).tolist()}
    # unit 5: E8's second date, from its peak (Blocked 73 (iii)); unit 3 gave 512
    rows["E8_peak"] = {"row": "E8 dated from its peak", "dir": "down",
                       **from_peak(np.asarray(rows["E8"]["series"]), "down")}
    if rows["E8_peak"]["step"] != E8_PEAK_UNIT3:
        raise A3Refused(f"E8 from its peak reads {rows['E8_peak']['step']}, unit 3 gave "
                        f"{E8_PEAK_UNIT3}")
    for rid, r in rows.items():
        if rid == "E8_peak":
            continue
        d = "down" if r["dir"] == "turn" else r["dir"]               # a turn row's fall
        r["beside"] = beside(np.asarray(r["series"]), d, r.get("index"))
    anchors = {"E1": 512, "E2": 512, "E10": 1000}
    rec["anchors"] = {k: {"want": v, "got": rows[k].get("step"), "pass": rows[k].get("step") == v}
                      for k, v in anchors.items()}
    rec["rows"] = rows
    rec["reading"] = reading(rows)
    rec["elapsed_s"] = round(time.time() - t0)
    (out / "a3_rows.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def table(rec: dict) -> str:
    h = ("| id | row | dir | reading | step | ref / σ | floor | low power | censored "
         "| centroid (beside) |")
    lines = [h, "|---|---|---|---|---|---|---|---|---|---|"]
    for rid in ("E1", "E2", "E3", "E4", "E5", "E6u", "E6b", "E6b_full", "E7", "E8", "E11", "E10",
                "E3_offset0"):
        if rid not in rec["rows"]:
            continue
        r = rec["rows"][rid]
        ros = r.get("ref_over_sigma")
        b = r["beside"]
        cen = (f"{b['centroid_step']:.0f}" + (" (≥ 2 steps)" if b.get("flag") else "")
               if b.get("centroid_step") is not None else "—")
        fl = rec["gate"].get(rid, {}).get("floor")
        dash = lambda v: "—" if v is None else v
        lines.append(f"| {rid} | {r['row']} | {r['dir']} | {r['reading']} | {r.get('step') or '—'} "
                     f"| {f'{ros:.1f}' if ros else '—'} | {f'{fl:.2f}' if fl else '—'} "
                     f"| {dash(r.get('low_power'))} | {dash(r.get('censored'))} | {cen} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--weights", type=Path, required=True)
    r.add_argument("--sim", type=Path, required=True)
    r.add_argument("--ngram", type=Path, required=True)
    r.add_argument("--ngram-full", type=Path, help="unit 5: table set′'s records (E6b′)")
    t = sub.add_parser("report"); t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rec = run(a.out, a.weights, a.sim, a.ngram, a.ngram_full)
    else:
        rec = json.loads((a.out / "a3_rows.json").read_text())
    print(table(rec))
    print("anchors:", {k: (v["got"], "pass" if v["pass"] else "FAIL") for k, v in rec["anchors"].items()})
    if "reading" in rec:
        print(pairs_table(rec["reading"]))


def pairs_table(rd: dict) -> str:
    lines = ["| pair | steps | verdict | first | notes |", "|---|---|---|---|---|"]
    for p in rd["pairs"]:
        st = "; ".join(f"{k} {' / '.join(map(str, v))}" for k, v in p.get("steps", {}).items())
        notes = "; ".join(p.get("notes", [])) or p.get("why", "")
        lines.append(f"| {p['pair'][0]}–{p['pair'][1]} | {st or '—'} | {p['verdict']} "
                     f"| {p.get('first', '—')} | {notes} |")
    lines.append("order: " + " < ".join(
        "=".join(o["rows"]) + f" ({' / '.join(map(str, o['steps']))})" for o in rd["order"]))
    lines.append(f"unigram first: {rd['unigram_first']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
