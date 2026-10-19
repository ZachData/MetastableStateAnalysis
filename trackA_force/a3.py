"""trackA_force/a3.py — A3 unit 2: the producer, the existing rows, the locator and its gate
(`design-A3.md` §2, §4; frozen 2026-10-19 as revised, Blocked 70 (ii) (a)).

Rows built here, on the 18 distinct steps (`a1a.STEPS`; steps 0 and 1 are one point):

- **E1** energy violations, Σ nViolRel at β = 1 over the 7 v1 prompts, the panel's
  ``p1_energy_nviol_b1``; located by `design-A5.md` D4 (a) exactly (half the median over the
  1000–4000 plateau), not by §4. The per-prompt counts from each step's Phase 1 report must sum
  to the panel's.
- **E2** OV: median over 384 heads of ``attractive_energy_fraction_core`` (down; σ over heads).
- **E3** induction: the five typed members' pooled induction attention (``behavioural_series``'s
  pair-count-weighted pool over the 7 prompts, averaged over members; up; σ over the 35
  (prompt, member) units, each carrying its prompt's pair count). Must reproduce
  ``behavioural_series.json`` per member.
- **E4** previous-token score of ``L5H2``: mean attention from query i to key i − 1, i ≥ 1
  (`p7d_redundancy.prev_token_profile`'s ``prev_all``), mean over prompts (up; σ over prompts).
  B1's records hold a signature, not this score, so it is read from ``attentions.npz``.
- **E5** effective rank on the sphere (``MinRank``), mean over prompts (down; σ over prompts).
  Must reproduce the panel's ``p1_eff_rank_normed_mean``.
- **E10** the LR schedule (warmup to 1430, cosine to 0.1×), analytic, the confound row, located
  by §4 with no floor.

**The locator (§4).** Δ(s) = x(s) − x(0), signed in the declared direction; the reference is the
largest Δ over steps ≤ 4000; the change step is the first with Δ ≥ reference / 2. Turning-point
rows (E6, unit 4) are located at the minimum over ≤ 4000, the floor on the fall into it. **No
change** if the reference is ≤ 0 or below ``FLOOR`` σ, σ the bootstrap SE (``N_BOOT`` resamples
over the row's units) of Δ at the reference step; **low power** below ``LOW_POWER`` σ.
`tools/math_checks/a3_locator.py` states its floor in units of per-step noise; σ here is the SE of
Δ, which under its iid model is √2 times that, so the floor as built is the stricter of the two
readings. The gate measures it at each row's own design.

**The gate (§4), before any real row is located** (a failed arm refuses its rows; nothing is
tuned after it). Per floored row, at its own units, statistic and per-step spread (τ_s, the
across-unit SD of the row's real Δ_u(s); step 0 carries none, so the Δs are independent across
steps, a conservative null): *null* ``N_GATE`` flat series read "no change" in ≥ 95 %; *effect*
a logistic in log2(step) of scale 0.25 octave centred in each gap 2–4 … 2000–4000, at 9 × the
mean null σ at the step past the centre, ``N_GATE`` each, placed there in ≥ 90 %. *Anchors:* E1
at 512, E2 at 512, E10 at 1000. Turning-point rows are gated in unit 4 on E6's own σ.

Beside, described and not read: `core.changepoint_colocation.change_profile`'s change-mass
centroid per row, flagged where it and the rule are ≥ 2 grid steps apart. No pair is read here
(that is unit 5).

Tier 1: exploratory, unregistered; v1 prompts only. CPU: numpy bootstraps over at most 384 units,
minutes; no forward pass.
    python -m trackA_force.a3 run --out <dir>
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

MEMBERS = ((8, 6), (7, 8), (11, 14), (12, 5), (8, 9))  # L5H2's partners, design-A3 §2 E3
PREV = (5, 2)
WIN = tuple(i for i, s in enumerate(STEPS) if s <= 4000)      # indices 0..12
FLOOR = 4.0          # calibrated (a3_locator.py: 3.93 for 95 %), in σ = SE of Δ
LOW_POWER = 9.0      # calibrated: 6σ (gap 2–4) to 9σ (gap 2000–4000) for 90 % placement
N_BOOT = 1000
N_GATE = 1000
GATE_NULL = 0.95
GATE_EFFECT = 0.90
WIDTH = 0.25         # placed: logistic scale, octaves
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


STATS: Dict[str, Callable] = {"mean": stat_mean, "median": stat_median, "pooled": stat_pooled}
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
                 n_boot: int = N_BOOT) -> dict:
    x = stat(units)
    j = reference(x, direction, kind)
    return place(x, direction, boot_se(units, stat, j, rng, n_boot), kind)


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
    pairs = induction_candidates(prompt_ids(tokenizer, prompt, n, d))
    q = np.array([p[0] for p in pairs]); k = np.array([p[1] for p in pairs])
    i = np.arange(1, n)
    return {"member_sums": [float(a[l, h, q, k].astype(np.float64).sum()) for l, h in MEMBERS],
            "n_pairs": len(pairs),
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
    n = {p: {rec[f"{s}|{p}"]["n_pairs"] for s in STEPS} for p in V1}
    if any(len(v) != 1 for v in n.values()):
        raise A3Refused(f"E3: a prompt's induction pair count changes across steps: {n}")
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
    return {"E3": e3, "E4": e4}


ROWS = {  # id: (row, direction, statistic, σ over)
    "E2": ("OV: median attractive_energy_fraction_core, 384 heads", "down", "median", "heads"),
    "E3": ("induction: five typed members' pooled induction attention", "up", "pooled",
           "prompts × members"),
    "E4": ("previous-token score of L5H2", "up", "mean", "prompts"),
    "E5": ("effective rank on the sphere (MinRank), mean over prompts", "down", "mean", "prompts"),
}


# ---------------------------------------------------------------- the gate

def unit_scores(units: np.ndarray, stat_name: str) -> np.ndarray:
    """(n_units, S) in the statistic's own units: per-pair means for the pooled row."""
    if stat_name == "pooled":
        return units[..., 0] / units[..., 1]
    return units[..., 0]


def _synthetic(units: np.ndarray, direction: str, stat_name: str, rng, n: int,
               amp_shape: Optional[np.ndarray] = None) -> np.ndarray:
    """(n, n_units, S, k) flat noise at the row's per-step spread, plus a planted shape."""
    real = unit_scores(units, stat_name)
    tau = np.std(real - real[:, :1], axis=0, ddof=1)             # (S,), tau[0] = 0
    z = rng.standard_normal((n,) + real.shape) * tau
    if amp_shape is not None:
        z = z + amp_shape
    if stat_name == "pooled":
        w = units[:, :1, 1]                                       # prompt pair counts
        return np.stack([z * w, np.broadcast_to(w, z.shape)], axis=-1)
    return z[..., None]


def gate_row(rid: str, units: np.ndarray, rng) -> dict:
    _, direction, sname, _ = ROWS[rid]
    stat = STATS[sname]
    null = _synthetic(units, direction, sname, rng, N_GATE)
    reads = [locate_units(u, stat, direction, rng) for u in null]
    p_null = float(np.mean([r["reading"] == "no change" for r in reads]))
    # the mean null σ at each step in WIN, for the planted amplitude
    sig = np.zeros(len(WIN))
    probe = null[:200]
    for j in WIN[1:]:
        sig[j] = np.mean([boot_se(u, stat, j, rng) for u in probe])
    gaps = []
    for i in WIN[1:-1]:
        centre = (X[i] + X[i + 1]) / 2
        shape = 1 / (1 + np.exp(-(X - centre) / WIDTH)); shape = shape - shape[0]
        amp = LOW_POWER * sig[i + 1]
        planted = _synthetic(units, direction, sname, rng, N_GATE,
                             amp_shape=SIGN[direction] * amp * shape)
        ok = float(np.mean([locate_units(u, stat, direction, rng)["index"] == i + 1
                            for u in planted]))
        gaps.append({"gap": f"{STEPS[i]}-{STEPS[i + 1]}", "amplitude": float(amp),
                     "placed_right": ok, "pass": ok >= GATE_EFFECT})
    return {"null_no_change": p_null, "null_pass": p_null >= GATE_NULL, "effect": gaps,
            "pass": p_null >= GATE_NULL and all(g["pass"] for g in gaps)}


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


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec = {"code": _sha(Path(__file__)), "prompts_hash": prompts_hash(), "steps": list(STEPS),
           "inputs": {f: _sha(DATA / "analysis" / f) for f in
                      ("colocation_panel.csv", "ov_per_head_series.json",
                       "behavioural_series.json")},
           "constants": {"FLOOR": FLOOR, "LOW_POWER": LOW_POWER, "N_BOOT": N_BOOT,
                         "N_GATE": N_GATE, "WIDTH": WIDTH, "SEED": SEED}}
    r15 = rows_e1_e5()
    units = {"E2": row_e2(), "E5": r15["E5"], **rows_e3_e4(out / "attn_rows.json")}
    print(f"rows built ({time.time() - t0:.0f} s)", flush=True)

    # gate first: nothing real is located before every arm is recorded
    rec["gate"] = {}
    for rid in ROWS:
        rec["gate"][rid] = gate_row(rid, units[rid], rng)
        g = rec["gate"][rid]
        print(f"gate {rid}: null {g['null_no_change']:.3f}, effect min "
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
        x = STATS[sname](units[rid])
        r = {"row": name, "dir": direction, "sigma_over": over, "n_units": int(units[rid].shape[0]),
             "series": x.tolist()}
        if not rec["gate"][rid]["pass"]:
            r.update(reading="refused", step=None, index=None)
        else:
            r.update(locate_units(units[rid], STATS[sname], direction, rng))
        rows[rid] = r
    for rid, r in rows.items():
        r["beside"] = beside(np.asarray(r["series"]), r["dir"], r.get("index"))
    anchors = {"E1": 512, "E2": 512, "E10": 1000}
    rec["anchors"] = {k: {"want": v, "got": rows[k].get("step"), "pass": rows[k].get("step") == v}
                      for k, v in anchors.items()}
    rec["rows"] = rows
    rec["elapsed_s"] = round(time.time() - t0)
    (out / "a3_rows.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def table(rec: dict) -> str:
    h = "| id | row | dir | reading | step | ref / σ | low power | centroid (beside) |"
    lines = [h, "|---|---|---|---|---|---|---|---|"]
    for rid in ("E1", "E2", "E3", "E4", "E5", "E10"):
        r = rec["rows"][rid]
        ros = r.get("ref_over_sigma")
        b = r["beside"]
        cen = (f"{b['centroid_step']:.0f}" + (" (≥ 2 steps)" if b.get("flag") else "")
               if b.get("centroid_step") is not None else "—")
        lines.append(f"| {rid} | {r['row']} | {r['dir']} | {r['reading']} | {r.get('step') or '—'} "
                     f"| {f'{ros:.1f}' if ros else '—'} | {r.get('low_power') if r.get('low_power') is not None else '—'} | {cen} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("report"); t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rec = run(a.out)
    else:
        rec = json.loads((a.out / "a3_rows.json").read_text())
    print(table(rec))
    print("anchors:", {k: (v["got"], "pass" if v["pass"] else "FAIL") for k, v in rec["anchors"].items()})


if __name__ == "__main__":
    main()
