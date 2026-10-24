"""
trackC_writeup/numbers.py — C2's numbers and figures, each from its record (`design-C2.md`
"Numbers and figures"; unit 2, 2026-10-22).

Three steps, so that only the first needs the main tree's `data/`:

- **produce** reads every record in `RECORDS`, computes each key in `KEYS` from it and writes
  `trackC_writeup/numbers.json` (tracked): every key's value and record, every record's sha256.
  A record whose sha256 differs from the table's is refused until named with `--reread` (the
  change has been read); a key with no record cannot be written.
- **render** fills `draft-C2.src.md`'s `{{key}}` / `{{key|fmt}}` slots from the table and writes
  `draft-C2.md`. An unknown key or format refuses. `check` renders in memory and fails if the
  tracked `draft-C2.md` differs, so CI checks the draft against the table on any checkout.
- **figures** draws figures 1–10 from the table alone (matplotlib): `fig/fig*.svg` for the
  Markdown draft and `tex/fig/fig*.pdf` for the LaTeX one (`to_tex.py`), both deterministic.
  Figure 6 is an illustration computed in `_toy`, not a record.

Numbers a paper reports are that paper's and carry its citation; numbers from this project's
records and from Pythia's released training configs are keys.

    METS_DATA=<main>/data python -m trackC_writeup.numbers produce [--reread data:<path> ...]
    python -m trackC_writeup.numbers render | check | figures
    python -m trackC_writeup.to_tex build | check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
HERE = Path(__file__).resolve().parent
TABLE = HERE / "numbers.json"
SOURCE = HERE / "draft-C2.src.md"
DRAFT = HERE / "draft-C2.md"
FIGDIR = HERE / "fig"
PDFDIR = HERE / "tex" / "fig"        # the LaTeX draft's copies (`to_tex.py`)

#: Record ids: `data:` under METS_DATA (the main tree's, never added), `repo:` tracked.
RECORDS = {
    "a5_seed0": "data:trackA/a5/2026-10-20-u1/seed0/gate.json",
    "a5_score": "data:trackA/a5/2026-10-20-u2/score.json",
    "a5_plan0": "data:trackA/a5/2026-10-20-u1/seed0/plan.json",
    "a5_audit": "repo:claims/audits/p_a5_real_run.json",
    "registry": "repo:claims/registry.json",
    "a3": "data:trackA/a3/2026-10-20-A3u5/a3_rows.json",
    "s2_readings": "data:trackS/s2/2026-10-21-u2/readings.json",
    "s2_gates": "data:trackS/s2/2026-10-21-u2/gate_read.json",
    "metrics": "repo:core/metrics.py",
    "evalues": "repo:core/evalues.py",
    # EleutherAI/pythia models/{70M,160M,410M}/pythia-*.yml, fetched 2026-10-22 from main at a19eecb807ec
    "cfg_70m": "data:trackC/configs/pythia-70m.yml",
    "cfg_160m": "data:trackC/configs/pythia-160m.yml",
    "cfg_410m": "data:trackC/configs/pythia-410m.yml",
    # C2's own runs (units 3, 3s, 3b, 3c), S1's ρ, and GPT-2 small's logged LR (unit 4)
    "c3": "data:trackC/c2u3/2026-10-22-run/reading.json",
    "c3s": "data:trackC/c2u3s/2026-10-22-run/reading.json",
    "c3b": "data:trackC/c2u3b/2026-10-22-run/reading.json",
    "c3c": "data:trackC/c2u3c/2026-10-22-run/reading.json",
    "s1_r1": "data:trackS/s1/2026-10-21-u2-r1/r1.json",
    "s1_qk": "data:trackS/s1/2026-10-21-u3-r1qk/r1_qk.json",
    "gpt2_lr": "repo:trackC_writeup/gpt2_small_lr_log.json",
    "c5mean": "data:trackC/c2u5/2026-10-23-mean/profile.json",
    "c5b": "data:trackC/c2u5b/2026-10-23/readings.json",
    "push": "repo:trackC_writeup/p_a5_push_record.json",
    # C5 on M2 (figures 9 and 10; C2p unit 1): each seed's massive set, and unit 2's reading
    "c5set0": "data:trackC/c5/2026-10-23-u1/seed0/set.json",
    "c5u1": "data:trackC/c5/2026-10-23-u1/reading.json",
    "c5u2": "data:trackC/c5/2026-10-24-u2/reading.json",
}
SEEDS = tuple(str(s) for s in range(1, 10))
#: The grid A5's V series are indexed on (`trackA_force.a1a.STEPS`); a claim checks A3's equals it.
GRID = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
S2_STEPS = ("0", "256", "512", "1000", "2000", "4000", "143000")
#: The rows §5's table and figure 2 show, in the order of their dates.
A3_ROWS = ("E5", "E6u", "E6b", "E1", "E2", "E8", "E3", "E4", "E10")
#: S2's scored steps (R1's cells after the break) and R2's (the steps where copying exists).
S2_SCORED = ("512", "1000", "2000", "4000", "143000")
S2_R2 = ("1000", "2000", "4000", "143000")
#: S1's families and steps (`design-S1.md`); QK's ρ is in its own record (S1 unit 3).
S1_FAMILIES = ("qk_query", "qk_key", "ov_write", "ov_cplx", "ov_read", "ov_neg", "ov_pos")
S1_STEPS = ("0", "128", "256", "512", "1000", "2000", "4000", "143000")
#: 3c's seeds, and its two fixed centred arms (the other two are derived; `design-C2.md` "Unit 3c").
C3C_SEEDS = ("alias", "battlestar", "caprica", "darkmatter", "expanse")
C3C_FIXED = ("0.0001", "0.0003")


class Refused(SystemExit):
    pass


def resolve(rid: str) -> Path:
    kind, _, rel = rid.partition(":")
    if kind == "data":
        return DATA / rel
    if kind == "repo":
        return REPO / rel
    raise Refused(f"refusing: record id {rid!r} is neither data: nor repo:")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yml(text: str) -> dict:
    """The Pythia configs are NeoX's YAML-with-comments; read only `"key": scalar` lines."""
    out = {}
    for m in re.finditer(r'^\s*"([\w-]+)":\s*([^,\n{\[]+?|\[[^\]]*\])\s*,?\s*$', text, re.M):
        v = m.group(2).strip()
        try:
            out[m.group(1)] = json.loads(v)
        except json.JSONDecodeError:
            out[m.group(1)] = v
    return out


def load(rid: str) -> object:
    p = resolve(rid)
    if not p.is_file():
        raise Refused(f"refusing: record {rid} is absent ({p})")
    if p.suffix == ".py":
        return p.read_text()
    return yml(p.read_text()) if p.suffix == ".yml" else json.loads(p.read_text())


def constant(src: str, name: str) -> float:
    """A module-level numeric constant, read from the source (`name: type = value` or `name = value`)."""
    m = re.search(rf"^{name}\s*(?::[^=]+)?=\s*([-+\d.eE]+)\s*$", src, re.M)
    if not m:
        raise Refused(f"refusing: no constant {name} in the source")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# The keys: name -> (record, function of that record's contents)
# ---------------------------------------------------------------------------

def _p_a5(reg: dict) -> dict:
    entries = reg if isinstance(reg, list) else reg.get("predictions", [])
    return next(e for e in entries if e.get("id") == "P-A5")


def _steps(a3: dict) -> List[int]:
    return list(a3["steps"])


def _i(step: int) -> int:
    return GRID.index(step)


def _upto(v: Sequence[float], step: int) -> List[float]:
    """V at every grid step up to and including `step`."""
    return list(v[:_i(step) + 1])


def _between(v: Sequence[float], lo: int, hi: int) -> List[float]:
    return list(v[_i(lo):_i(hi) + 1])


def _v(score: dict, seed: str) -> List[float]:
    v = score["notes"][seed]["V"]
    if len(v) != len(GRID):
        raise Refused(f"refusing: seed {seed}'s V has {len(v)} steps, not the {len(GRID)} of GRID")
    return v


def _late_rise(v: Sequence[float]) -> List[int]:
    """The consecutive pair of steps after 4000 across which V rises most."""
    j = max(range(_i(4000), len(GRID) - 1), key=lambda i: v[i + 1] - v[i])
    return [GRID[j], GRID[j + 1]]


def _at(a3: dict, series: Sequence[float], step: int) -> float:
    return series[_steps(a3).index(step)]


def _d10_pairs(score: dict) -> Dict[str, dict]:
    """Each seed's previous-token–induction pair, where both rows are located."""
    out = {}
    for s in SEEDS:
        p = next(x for x in score["notes"][s]["d10"]["pairs"] if x["pair"] == "prev–ind")
        if p.get("tie") is not None:
            out[s] = p
    return out


FALSIFIER_K = 3        # P-A5's registered falsifier, K <= 3; a claim checks it against the registry


def _binom_tail(n: int, p: float, k: int) -> float:
    """P(Bin(n, p) >= k), exact."""
    from math import comb
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def _lane_at(k: dict, row: str, step: int) -> float:
    return k["a3.lanes"][row]["y"][k["grid.steps"].index(step)]


def _utc(iso: str):
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


def _early_argmax(score: dict) -> int:
    """The grid step (of 0, 2, 4) at which the largest early count sits, on the seed that has it."""
    seed = max(SEEDS, key=lambda k: max(_upto(_v(score, k), 4)))
    v = _upto(_v(score, seed), 4)
    return GRID[v.index(max(v))]


def _lane(a3: dict, rid: str) -> dict:
    return next(x for x in a3["reading"]["lanes"] if x["id"] == rid)


def _verdicts(a3: dict, v: str) -> int:
    return sum(p["verdict"] == v for p in a3["reading"]["pairs"])


def _s2_cells(rec: dict, arm: str) -> dict:
    cells = rec["R1"][arm]["cells"]
    return {s: {"value": cells[s]["value"], **cells[s]["ctl"]} for s in S2_STEPS if s in cells}


def _g3p(rec: dict) -> dict:
    return {s: {"value": c["value"], **c["ctl"]} for s, c in rec["G3p"]["cells"].items()}


def _span(xs) -> List[float]:
    xs = list(xs)
    return [min(xs), max(xs)]


def _c3_matched(c3: dict) -> float:
    """The centred plateau at the matched tolerance (1e-4, post hoc in unit 3) over the uncentred one."""
    v = c3["beside"]["V_centred_at_tol"]["0.0001"]
    plat = sorted(v[_i(s)] for s in (1000, 2000, 4000))[1]
    return plat / c3["verdict"]["plateau_uncentred"]


def _c3s_arms(c3s: dict):
    """(seed, arm, the arm's row) over the 9 seeds and the four centred arms."""
    for s in SEEDS:
        for tau, arm in c3s["per_seed"][s]["arms"].items():
            yield s, tau, arm


def _cfrac(c3s: dict, brk: int, tau: Optional[float] = None) -> List[float]:
    """The centred count at the uncentred break over its own plateau, on the seeds (0 included)
    that break at `brk`, at every arm or at the one named."""
    runs = [c3s["seed0_development"]] + [c3s["per_seed"][s] for s in SEEDS]
    return [a["V_c_at_break"] / a["plateau_centred"] for r in runs if r["break_step"] == brk
            for t, a in r["arms"].items() if tau is None or float(t) == tau]


def _c3c_dates(c3c: dict, arms: Sequence[str]) -> List[int]:
    return [c3c["per_seed"][s]["centred"]["arms"][a]["step"] for s in C3C_SEEDS for a in arms]


def _c3c_step_arms(c3c: dict, seed: str) -> List[float]:
    """The centred arms whose date falls in step's centred band, on one seed."""
    band = c3c["predicted"]["centred"]["step"]
    return [float(t) for t, a in c3c["per_seed"][seed]["centred"]["arms"].items() if a["step"] in band]


def _c3c_at(c3c: dict, step: int) -> List[float]:
    i = c3c["grid"].index(step)
    return _span(c3c["beside"][s]["V_u"][i] for s in C3C_SEEDS)


def _lr(log: dict) -> Dict[str, list]:
    return {s: v["lr"] for s, v in log["seeds"].items()}


def _s1_fams(rec: dict) -> List[str]:
    """The families a record reads ρ for (S1 unit 2's record leaves QK's to unit 3's)."""
    return [f for f in S1_FAMILIES if "PR" in rec["cells"].get(f"0|{f}", {})]


def _s1_leaves(rec: dict) -> Dict[str, int]:
    """Each family's first step whose participation ratio leaves the null's band (S1's reading)."""
    out = {}
    for f in _s1_fams(rec):
        for s in S1_STEPS:
            c = rec["cells"][f"{s}|{f}"]
            if not c["band"]["PR_min"] <= c["PR"] <= c["band"]["PR_max"]:
                out[f] = int(s)
                break
    return out


def _s1_rho(rec: dict, step: str) -> Dict[str, float]:
    return {f: rec["cells"][f"{step}|{f}"]["rho"] for f in _s1_fams(rec)}


#: Unit 5b's steps: the break's window and induction's step (R6), and the steps after (R6, late).
C5B_EARLY = ("128", "256", "512", "1000")
C5B_LATE = ("2000", "4000", "8000", "16000", "32000", "54000", "143000")


def _r1_out(r: dict) -> List[tuple]:
    """The (seed, prompt, date) series R1 dates outside the window."""
    return [(int(s), p, d) for s, x in r["per_seed"].items() for p, d in x["R1"]["dates"].items()
            if d not in r["window"]]


def _c5seeds(r: dict) -> Dict[str, dict]:
    """C5's readings keep seed 0 (the development set) apart from the nine; the figures draw all ten."""
    return dict(r["per_seed"], **{"0": r["seed0_development"]})


def _c5b_s_exc(r: dict) -> int:
    """Drops at 128–1000 where the mean's term carries half or less (from each step's share)."""
    b = r["R6"]["by_step"]
    return sum(round(b[s]["drops"] * (1 - b[s]["s_gt_half_share"])) for s in C5B_EARLY)


KEYS: Dict[str, tuple] = {
    # Pythia's released training configs (§2.3, §3)
    "cfg.train_iters": ("cfg_410m", lambda c: c["train-iters"]),
    "cfg.warmup_pct": ("cfg_410m", lambda c: c["warmup"] * 100),
    "cfg.warmup_steps": ("cfg_410m", lambda c: round(c["warmup"] * c["train-iters"])),
    "cfg.lr_floor_ratio": ("cfg_410m", lambda c: c["min_lr"] / c["optimizer_lr"]),
    "cfg.lr_410m": ("cfg_410m", lambda c: c["optimizer_lr"]),
    "cfg.lr_160m": ("cfg_160m", lambda c: c["optimizer_lr"]),
    "cfg.lr_70m": ("cfg_70m", lambda c: c["optimizer_lr"]),
    "cfg.warmup_160m": ("cfg_160m", lambda c: round(c["warmup"] * c["train-iters"])),
    "cfg.warmup_70m": ("cfg_70m", lambda c: round(c["warmup"] * c["train-iters"])),
    "cfg.seq_len": ("cfg_410m", lambda c: c["seq-length"]),
    "cfg.layers": ("cfg_410m", lambda c: c["num-layers"]),
    "cfg.width": ("cfg_410m", lambda c: c["hidden-size"]),
    "cfg.heads": ("cfg_410m", lambda c: c["num-attention-heads"]),
    "cfg.heads_total": ("cfg_410m", lambda c: c["num-layers"] * c["num-attention-heads"]),
    "cfg.layers_m1": ("cfg_410m", lambda c: c["num-layers"] - 1),
    "cfg.warmup_half": ("cfg_410m", lambda c: round(c["warmup"] * c["train-iters"] / 2)),
    "cfg.states": ("cfg_410m", lambda c: c["num-layers"] + 1),
    "cfg.early_saves": ("cfg_410m", lambda c: c["extra-save-iters"]),
    "cfg.save_every": ("cfg_410m", lambda c: c["checkpoint-factor"]),
    # the grid and the prompts (§3)
    "grid.steps": ("a3", _steps),
    "grid.n": ("a3", lambda a: len(_steps(a))),
    "grid.n_to4000": ("a3", lambda a: sum(0 < s <= 4000 for s in _steps(a))),
    "a5.prompts_hash": ("a5_seed0", lambda g: g["prompts_hash"]),
    "a5.max_tokens": ("a5_plan0", lambda p: p["max_tokens"]),
    "a5.dtype": ("a5_plan0", lambda p: p["dtype"]),
    "a5.tf32": ("a5_plan0", lambda p: p["tf32"]),
    "rel_tol": ("metrics", lambda s: constant(s, "ENERGY_VIOLATION_REL_TOL")),
    "kappa": ("evalues", lambda s: constant(s, "DEFAULT_KAPPA")),
    "a5.n_prompts": ("a5_seed0", lambda g: len(g["sensitivity"]["leave_one_out"])),
    # seed 0 (§4; A5's gate record, whose V equals A3's E1 and the panel's)
    "s0.V": ("a5_seed0", lambda g: g["known_answer"]["V"]),
    "s0.V256": ("a5_seed0", lambda g: g["known_answer"]["V"][_i(256)]),
    "s0.V512": ("a5_seed0", lambda g: g["known_answer"]["V"][_i(512)]),
    "s0.Vmax_to128": ("a5_seed0", lambda g: max(_upto(g["known_answer"]["V"], 128))),
    "s0.late_rise": ("a5_seed0", lambda g: _late_rise(g["known_answer"]["V"])),
    "s0.plateau": ("a5_seed0", lambda g: g["known_answer"]["break"]["plateau_median"]),
    "s0.break": ("a5_seed0", lambda g: g["known_answer"]["break"]["step"]),
    "s0.V_end": ("a5_seed0", lambda g: g["known_answer"]["V"][-1]),
    "s0.ln1_break": ("a5_seed0", lambda g: g["d10"]["seed_reading"]["beside_ln1_b35"]["break"]["step"]),
    "s0.energy_rel_diff": ("a5_seed0", lambda g: g["known_answer"]["energy_max_rel_diff"]),
    "s0.V_equal": ("a5_seed0", lambda g: g["known_answer"]["V_equal"]),
    "s0.tols": ("a5_seed0", lambda g: sorted(float(t) for t in g["sensitivity"]["rel_tol"])),
    "s0.tol_steps": ("a5_seed0", lambda g: [g["sensitivity"]["rel_tol"][t]["step"]
                                            for t in sorted(g["sensitivity"]["rel_tol"], key=float)]),
    "s0.loo_steps": ("a5_seed0", lambda g: sorted({v["step"] for v in g["sensitivity"]["leave_one_out"].values()})),
    # the registration (§3)
    "reg.at": ("a5_audit", lambda a: a["order"]["registered_at"]),
    "reg.first_record": ("a5_audit", lambda a: a["order"]["first_record_at"]),
    "reg.commit": ("a5_audit", lambda a: a["order"]["registered_commit"][:7]),
    "reg.first": ("a5_audit", lambda a: a["order"]["registered_first"]),
    "reg.lag_min": ("a5_audit", lambda a: round((_utc(a["order"]["first_record_at"])
                                                 - _utc(a["order"]["registered_at"])).total_seconds() / 60)),
    "reg.falsifier_K": ("registry", lambda r: int(re.search(r"K <= (\d+)", _p_a5(r)["falsifier"]).group(1))),
    "reg.p0": ("a5_score", lambda s: str(Fraction(s["p0"]).limit_denominator(100))),
    "reg.falsifier_p": ("a5_score", lambda s: _binom_tail(s["n"], s["p0"], FALSIFIER_K + 1)),
    "reg.pushed_at": ("push", lambda r: r["events"][0]["timestamp"].replace("Z", "+00:00")),
    "reg.pushed_kind": ("push", lambda r: r["events"][0]["activity_type"]),
    "reg.pushed_has_commit": ("push", lambda r: r["contained_in"][r["events"][0]["after"][:7]]),
    # the mean's term ‖m‖² by layer, read after the dates (unit 5, c2_mean_profile.py)
    "mean.falls": ("c5mean", lambda r: {st: [r["seeds_data"][s]["falls_all"][st] for s in map(str, r["seeds"])]
                                        for st in ("128", "256", "512", "1000")}),
    "mean.frame_falls": ("c5mean", lambda r: sum(d["falls_all"][st] - d["falls_inner"][st]
                                                 for d in r["seeds_data"].values() for st in ("128", "256", "512", "1000"))),
    "mean.n_trans": ("c5mean", lambda r: r["transitions_all"]),
    # the referee's readings from records (unit 5b, c2_readings.py; design-C2.md "Unit 5b")
    "rd.map": ("c5b", lambda r: r["R2"]["map"]),
    "rd.depth": ("c5b", lambda r: r["R7"]["e1_prompt_mean"]),
    "rd.r1_n": ("c5b", lambda r: r["R1"]["series"]),
    "rd.r1_inside": ("c5b", lambda r: r["R1"]["inside"]),
    "rd.r1_out_n": ("c5b", lambda r: len(_r1_out(r))),
    "rd.r1_out_prompts": ("c5b", lambda r: sorted({p for _, p, _ in _r1_out(r)})),
    "rd.r1_out_steps": ("c5b", lambda r: sorted({d for _, _, d in _r1_out(r)})),
    "rd.r1_share": ("c5b", lambda r: _span(r["R1"]["max_prompt_share_at_break"])),
    "rd.map_median": ("c5b", lambda r: {s: v["median_t"] for s, v in r["R2"]["at"].items()}),
    "rd.map_top": ("c5b", lambda r: {s: sorted(t["t"] for t in v["top3"]) for s, v in r["R2"]["at"].items()}),
    "rd.frame_share": ("c5b", lambda r: r["R3"]["frame_share"]),
    "rd.inner_dates": ("c5b", lambda r: r["R3"]["dates"]),
    "rd.inner_inside": ("c5b", lambda r: r["R3"]["inside"]),
    "rd.net_n": ("c5b", lambda r: r["R4"]["0"]["series"]),
    "rd.net_min_pos": ("c5b", lambda r: min(v["net_positive"] for v in r["R4"].values())),
    "rd.net_median": ("c5b", lambda r: {s: r["R4"][s]["net_median"] for s in ("128", "512", "1000", "143000")}),
    "rd.net_inner_late": ("c5b", lambda r: _span(r["R4"][s]["net_inner_positive"] for s in C5B_LATE[2:])),
    "rd.net_inner_early": ("c5b", lambda r: min(r["R4"][str(s)]["net_inner_positive"] for s in GRID if s < 8000)),
    "rd.late_dates": ("c5b", lambda r: r["R5"]["dates"]),
    "rd.late_inside": ("c5b", lambda r: r["R5"]["inside"]),
    "rd.late_plateaus": ("c5b", lambda r: _span(r["R5"]["plateaus"])),
    "rd.late_moved": ("c5b", lambda r: [s for s in range(len(r["R5"]["dates"]))
                                        if r["R5"]["dates"][s] != r["per_seed"][str(s)]["break"]]),
    "rd.mfall_at": ("c5b", lambda r: {s: r["R6"]["by_step"][s]["m_fall_share_at_drops"] for s in C5B_EARLY}),
    "rd.mfall_late": ("c5b", lambda r: min(r["R6"]["by_step"][s]["m_fall_share_at_drops"] for s in C5B_LATE)),
    "rd.drops_early": ("c5b", lambda r: sum(r["R6"]["by_step"][s]["drops"] for s in C5B_EARLY)),
    "rd.s_exc": ("c5b", _c5b_s_exc),
    "rd.s_median": ("c5b", lambda r: _span(r["R6"]["by_step"][s]["s_median"] for s in C5B_EARLY[1:])),
    "rd.s_late": ("c5b", lambda r: _span(r["R6"]["by_step"][s]["s_gt_half_share"] for s in C5B_LATE)),
    # R6's base rate over every move (post hoc, `/challenge-pr` on #261 finding 1)
    "rd.same_dir": ("c5b", lambda r: _span(v["same_direction_share"] for v in r["R6"]["by_step"].values())),
    "rd.m_any_fall": ("c5b", lambda r: _span(v["m_fall_share_at_any_e_fall"] for v in r["R6"]["by_step"].values()
                                             if v["m_fall_share_at_any_e_fall"] is not None)),
    "rd.s_rises": ("c5b", lambda r: {s: r["R6"]["by_step"][s]["s_gt_half_share_rises"] for s in ("16", "32")}),
    "rd.s_rises_min_from128": ("c5b", lambda r: min(r["R6"]["by_step"][s]["s_gt_half_share_rises"]
                                                    for s in C5B_EARLY + C5B_LATE)),
    # the nine seeds (§4)
    "a5.K": ("a5_score", lambda s: s["K"]),
    "a5.n": ("a5_score", lambda s: s["n"]),
    "a5.p": ("a5_score", lambda s: s["p"]),
    "a5.E": ("a5_score", lambda s: s["e"]),
    "a5.steps": ("a5_score", lambda s: [s["per_seed"][k]["step"] for k in SEEDS]),
    "a5.n512": ("a5_score", lambda s: sum(s["per_seed"][k]["step"] == 512 for k in SEEDS)),
    "a5.at256": ("a5_score", lambda s: [int(k) for k in SEEDS if s["per_seed"][k]["step"] == 256]),
    "a5.outliers": ("a5_score", lambda s: [int(k) for k in SEEDS if s["per_seed"][k]["outlier"]]),
    "a5.V": ("a5_score", lambda s: {k: _v(s, k) for k in SEEDS}),
    "a5.plateau_min": ("a5_score", lambda s: min(s["per_seed"][k]["plateau_median"] for k in SEEDS)),
    "a5.plateau_max": ("a5_score", lambda s: max(s["per_seed"][k]["plateau_median"] for k in SEEDS)),
    "a5.plateaus": ("a5_score", lambda s: [s["per_seed"][k]["plateau_median"] for k in SEEDS]),
    "a5.V128_max": ("a5_score", lambda s: max(_v(s, k)[_i(128)] for k in SEEDS)),
    "a5.half_min": ("a5_score", lambda s: min(s["per_seed"][k]["plateau_median"] for k in SEEDS) / 2),
    "a5.V512_min": ("a5_score", lambda s: min(_v(s, k)[_i(512)] for k in SEEDS)),
    "a5.half_max": ("a5_score", lambda s: max(s["per_seed"][k]["plateau_median"] for k in SEEDS) / 2),
    "a5.step0_seeds": ("a5_score", lambda s: [int(k) for k in SEEDS if _v(s, k)[0] > 0]),
    "a5.step0_n": ("a5_score", lambda s: sum(_v(s, k)[0] > 0 for k in SEEDS)),
    "a5.step0_min": ("a5_score", lambda s: min(_v(s, k)[0] for k in SEEDS if _v(s, k)[0] > 0)),
    "a5.step0_max": ("a5_score", lambda s: max(_v(s, k)[0] for k in SEEDS)),
    "a5.early_max": ("a5_score", lambda s: max(max(_upto(_v(s, k), 4)) for k in SEEDS)),
    "a5.early_max_seed": ("a5_score", lambda s: int(max(SEEDS, key=lambda k: max(_upto(_v(s, k), 4))))),
    "a5.early_max_step": ("a5_score", lambda s: _early_argmax(s)),
    "a5.step0_clean": ("a5_score", lambda s: sum(_v(s, k)[0] == 0 for k in SEEDS)),
    "a5.mid_max": ("a5_score", lambda s: max(max(_between(_v(s, k), 8, 64)) for k in SEEDS)),
    "a5.ln1_512": ("a5_score", lambda s: sum(s["notes"][k]["beside_ln1_b35"]["break"]["step"] == 512
                                             for k in SEEDS)),
    "a5.d10_n": ("a5_score", lambda s: len(_d10_pairs(s))),
    "a5.d10_tie_steps": ("a5_score", lambda s: sorted({p["steps"][0] for p in _d10_pairs(s).values() if p["tie"]})),
    "a5.d10_tied": ("a5_score", lambda s: sum(p["tie"] for p in _d10_pairs(s).values())),
    "a5.d10_prev": ("a5_score", lambda s: sum(p.get("further") == "prev" for p in _d10_pairs(s).values())),
    "a5.d10_other": ("a5_score", lambda s: [int(k) for k, p in _d10_pairs(s).items()
                                            if p.get("further") not in (None, "prev")]),
    "a5.d10_unlocated": ("a5_score", lambda s: [int(k) for k in SEEDS if k not in _d10_pairs(s)]),
    "a5.d10_prev_share": ("a5_score", lambda s: [min(p["shares"][0] for p in _d10_pairs(s).values()
                                                     if p.get("further") == "prev"),
                                                 max(p["shares"][0] for p in _d10_pairs(s).values()
                                                     if p.get("further") == "prev")]),
    "a5.d10_ind_share": ("a5_score", lambda s: [min(p["shares"][1] for p in _d10_pairs(s).values()
                                                    if p.get("further") == "prev"),
                                                max(p["shares"][1] for p in _d10_pairs(s).values()
                                                    if p.get("further") == "prev")]),
    # the timeline (§5; seed 0)
    "a3.rows": ("a3", lambda a: {r: {"row": a["rows"][r]["row"], "step": a["rows"][r].get("step"),
                                     "reading": a["rows"][r]["reading"],
                                     "censored": a["rows"][r].get("censored", False)}
                                 for r in A3_ROWS + ("E7", "E11")}),
    "a3.lanes": ("a3", lambda a: {r: {"y": _lane(a, r)["y"], "step": _lane(a, r)["step"],
                                      "how": _lane(a, r)["how"]} for r in A3_ROWS}),
    "a3.E1_centroid": ("a3", lambda a: a["rows"]["E1"]["beside"]["centroid_step"]),
    "a3.E2_bottom": ("a3", lambda a: _steps(a)[a["rows"]["E2"]["series"].index(min(a["rows"]["E2"]["series"]))]),
    "a3.E8_peak": ("a3", lambda a: a["rows"]["E8_peak"]["step"]),
    "a3.E6b_full": ("a3", lambda a: a["rows"]["E6b_full"]["step"]),
    "a3.warmup_end": ("a3", lambda a: a["reading"]["warmup_end"]),
    "a3.share512": ("a3", lambda a: {r: _at(a, _lane(a, r)["y"], 512) for r in ("E1", "E2", "E3", "E4", "E10")}),
    "a3.n_pairs": ("a3", lambda a: len(a["reading"]["pairs"])),
    "a3.n_aligned": ("a3", lambda a: _verdicts(a, "aligned")),
    "a3.n_misaligned": ("a3", lambda a: _verdicts(a, "misaligned")),
    "a3.n_unread": ("a3", lambda a: _verdicts(a, "unread")),
    "a3.aligned_in_warmup": ("a3", lambda a: all("inside warmup (E10)" in p.get("notes", [])
                                                 for p in a["reading"]["pairs"] if p["verdict"] == "aligned")),
    "a3.ngram_blocks": ("a3", lambda a: a["rows"]["E6u"]["n_units"]),
    "a3.floor": ("a3", lambda a: a["constants"]["FLOOR"]),
    # S2 (figure 3; §6 is unit 4's)
    "s2.a": ("s2_readings", lambda r: _s2_cells(r, "a")),
    "s2.b": ("s2_readings", lambda r: _s2_cells(r, "b")),
    "s2.g3p": ("s2_gates", _g3p),
    # S2 in words (§6, appendix B)
    "s2.r1_answer": ("s2_readings", lambda r: {a: r["R1"][a]["answer"] for a in "ab"}),
    "s2.r1_lower": ("s2_readings", lambda r: {a: r["R1"][a]["counts"] for a in "ab"}),
    "s2.n_scored": ("s2_readings", lambda r: len(S2_SCORED)),
    "s2.r1a_higher": ("s2_readings", lambda r: [int(s) for s in S2_SCORED
                                                if r["R1"]["a"]["cells"][s]["raw"] == "higher"]),
    "s2.r1b_raw": ("s2_readings", lambda r: sorted({r["R1"]["b"]["cells"][s]["raw"] for s in S2_SCORED})),
    "s2.r2b_G": ("s2_readings", lambda r: _span(r["R2"]["b"]["cells"][s]["value"] for s in S2_R2)),
    "s2.r2b_raw": ("s2_readings", lambda r: sorted({r["R2"]["b"]["cells"][s]["raw"] for s in S2_R2})),
    "s2.n_r2": ("s2_readings", lambda r: len(S2_R2)),
    "s2.b_edit": ("s2_readings", lambda r: _span(r["beside"][s]["sizes"]["b"]["edit"]["median"]
                                                 for s in S2_R2)),
    "s2.g3p_span": ("s2_gates", lambda g: _span(g["G3p"]["cells"][s]["value"] for s in S2_SCORED)),
    "s2.g3p_raw": ("s2_gates", lambda g: {s: g["G3p"]["cells"][s]["raw"] for s in S2_SCORED}),
    "s2.g3p_v": ("s2_gates", lambda g: {s: g["G3p"]["cells"][s]["value"] for s in S2_SCORED}),
    "s2.g3p_ctl": ("s2_gates", lambda g: {s: [g["G3p"]["cells"][s]["ctl"]["min"], g["G3p"]["cells"][s]["ctl"]["max"]]
                                          for s in S2_SCORED}),
    "s2.g1_labelled": ("s2_gates", lambda g: sum(g["G1"]["labelled"].values())),
    "s2.g2_lost": ("s2_gates", lambda g: g["G2"]["lost"]),
    "s2.g2_G": ("s2_gates", lambda g: _span(g["G2"]["cells"][s]["value"] for s in S2_R2)),
    "s2.g3_full": ("s2_gates", lambda g: _span(g["G3"]["cells"][f"1.0@{s}"]["value"] for s in S2_SCORED[:-1])),
    "s2.g3_counts": ("s2_gates", lambda g: g["G3"]["counts_at_1"]),
    # the anisotropy control on seed 0 (§4; unit 3)
    "c3.label": ("c3", lambda c: c["verdict"]["label"]),
    "c3.plateau_c": ("c3", lambda c: c["verdict"]["plateau_centred"]),
    "c3.plateau_u": ("c3", lambda c: c["verdict"]["plateau_uncentred"]),
    "c3.step": ("c3", lambda c: c["verdict"]["step"]),
    "c3.early_max": ("c3", lambda c: max(c["verdict"]["early_centred"].values())),
    "c3.matched_ratio": ("c3", _c3_matched),
    "c3.e1_dev": ("c3", lambda c: c["e1_max_dev_a5"]),
    # the centred count on the seeds (§4; unit 3s)
    "c3s.K1": ("c3s", lambda c: c["tally"]["K1"]),
    "c3s.K2": ("c3s", lambda c: c["tally"]["K2"]),
    "c3s.n": ("c3s", lambda c: c["tally"]["n"]),
    "c3s.events": ("c3s", lambda c: [c["tally"]["event1"], c["tally"]["event2"]]),
    "c3s.tau": ("c3s", lambda c: c["tau_arms"]),
    "c3s.s0_dates": ("c3s", lambda c: sorted({a["centred_step"] for a in c["seed0_development"]["arms"].values()})),
    "c3s.s0_Vc": ("c3s", lambda c: _span(a["V_c_at_break"] for a in c["seed0_development"]["arms"].values())),
    "c3s.s0_Vu": ("c3s", lambda c: c["seed0_development"]["V_u_at_break"]),
    "c3s.s0_early": ("c3s", lambda c: max(v for a in c["seed0_development"]["arms"].values()
                                          for v in a["early_centred"].values())),
    "c3s.s0_cfrac": ("c3s", lambda c: _span(a["V_c_at_break"] / a["plateau_centred"]
                                            for a in c["seed0_development"]["arms"].values())),
    "c3s.s0_ufrac": ("c3s", lambda c: c["seed0_development"]["V_u_at_break"]
                     / c["seed0_development"]["plateau_uncentred"]),
    "c3s.s0_lane": ("c3s", lambda c: {"y": [v / c["seed0_development"]["arms"]["0.0001"]["plateau_centred"]
                                            for v in c["beside"]["0"]["V_c"]["0.0001"]],
                                      "step": c["seed0_development"]["arms"]["0.0001"]["centred_step"]}),
    "c3s.cfrac512": ("c3s", lambda c: _span(_cfrac(c, 512))),
    "c3s.cfrac512_null": ("c3s", lambda c: _span(_cfrac(c, 512, c["tau_arms"][0]))),
    "c3s.cfrac256": ("c3s", lambda c: _span(_cfrac(c, 256))),
    "c3s.off": ("c3s", lambda c: [{"seed": int(s), "tau": float(t), "step": a["centred_step"]}
                                  for s, t, a in _c3s_arms(c) if a["centred_step"] != 1000]),
    "c3s.miss": ("c3s", lambda c: [int(s) for s in SEEDS if not c["per_seed"][s]["event1"]]),
    "c3s.n_cells": ("c3s", lambda c: sum(1 for _ in _c3s_arms(c))),
    # the LR-dose arm (§7; unit 3b)
    "c3b.u": ("c3b", lambda c: {z: c["per_size"][z]["uncentred"]["step"] for z in ("70m", "160m")}),
    "c3b.c": ("c3b", lambda c: {z: c["per_size"][z]["centred"]["dates"] for z in ("70m", "160m")}),
    "c3b.plateau_u": ("c3b", lambda c: {z: c["per_size"][z]["uncentred"]["plateau"] for z in ("70m", "160m")}),
    "c3b.label": ("c3b", lambda c: c["joint"]["label"]),
    "c3b.survivors": ("c3b", lambda c: c["joint"]["survivors"]),
    "c3b.pred": ("c3b", lambda c: {z: c["beside"][z]["predicted"] for z in ("70m", "160m")}),
    "c3b.band_u": ("c3b", lambda c: {k: _span(v) for k, v in c["beside"]["160m"]["predicted"]["uncentred"].items()}),
    "c3b.band_c": ("c3b", lambda c: {k: _span(v) for k, v in c["beside"]["160m"]["predicted"]["centred"].items()}),
    "c3b.band_c70": ("c3b", lambda c: {k: _span(v) for k, v in c["beside"]["70m"]["predicted"]["centred"].items()}),
    "c3b.ref": ("c3b", lambda c: c["ref"]),
    "c3b.V": ("c3b", lambda c: {z: {"u": c["beside"][z]["V_u"], "c": c["beside"][z]["V_c"]["0.0001"],
                                    "pu": c["per_size"][z]["uncentred"]["plateau"],
                                    "pc": c["per_size"][z]["centred"]["arms"]["0.0001"]["plateau"]}
                                for z in ("70m", "160m")}),
    # the warmup arm (§7; unit 3c)
    "c3c.n": ("c3c", lambda c: len(C3C_SEEDS)),
    "c3c.grid": ("c3c", lambda c: c["grid"]),
    "c3c.breaks": ("c3c", lambda c: [c["per_seed"][s]["uncentred"]["step"] for s in C3C_SEEDS]),
    "c3c.plateaus": ("c3c", lambda c: [c["per_seed"][s]["uncentred"]["plateau"] for s in C3C_SEEDS]),
    # figure 4's right panel (unit 5b, R8): each centred arm's V_c over its own plateau, median over seeds
    "c3c.Vc_med": ("c3c", lambda c: {a: [median(c["beside"][s]["V_c"][a][i] / c["per_seed"][s]["centred"]["arms"][a]["plateau"]
                                                for s in C3C_SEEDS) for i in range(len(c["grid"]))]
                                     for a in c["per_seed"][C3C_SEEDS[0]]["centred"]["arms"]}),
    "c3c.alt": ("c3c", lambda c: [c["per_seed"][s]["uncentred"]["beside"]["half_rise_on_1000_4000"]["step"]
                                  for s in C3C_SEEDS]),
    "c3c.label": ("c3c", lambda c: c["joint"]["label"]),
    "c3c.survivors": ("c3c", lambda c: c["joint"]["survivors"]),
    "c3c.u_consistent": ("c3c", lambda c: sorted({x for s in C3C_SEEDS for x in c["per_seed"][s]["uncentred"]["consistent"]})),
    "c3c.c_span": ("c3c", lambda c: _span(d for s in C3C_SEEDS for d in c["per_seed"][s]["centred"]["dates"])),
    "c3c.c_agree": ("c3c", lambda c: sum(c["per_seed"][s]["centred"]["agree"] for s in C3C_SEEDS)),
    "c3c.fixed": ("c3c", lambda c: _span(_c3c_dates(c, C3C_FIXED))),
    "c3c.derived": ("c3c", lambda c: _span(_c3c_dates(c, [a for a in c["per_seed"]["alias"]["centred"]["arms"]
                                                          if a not in C3C_FIXED]))),
    "c3c.null_only": ("c3c", lambda c: sum(_c3c_step_arms(c, s) == [c["tau_arms"][0]] for s in C3C_SEEDS)),
    "c3c.pred_c": ("c3c", lambda c: {k: _span(v) for k, v in c["predicted"]["centred"].items()}),
    "c3c.pred": ("c3c", lambda c: {k: _span(v) for k, v in c["predicted"]["uncentred"].items()}),
    "c3c.V100": ("c3c", lambda c: _c3c_at(c, 100)),
    "c3c.V350": ("c3c", lambda c: _c3c_at(c, 350)),
    "c3c.early_max": ("c3c", lambda c: max(max(v["uncentred"].values()) for v in c["early_first"].values())),
    "c3c.Vu": ("c3c", lambda c: {s: c["beside"][s]["V_u"] for s in C3C_SEEDS}),
    "gpt2.lr_peak": ("gpt2_lr", lambda g: max(lr for _, lr in _lr(g)["alias"])),
    "gpt2.lr_frac": ("gpt2_lr", lambda g: [[t, lr / max(x for _, x in _lr(g)["alias"])] for t, lr in _lr(g)["alias"]]),
    "gpt2.warmup_end": ("gpt2_lr", lambda g: max(next(t for t, lr in v if lr >= 0.999 * max(x for _, x in v))
                                                 for v in _lr(g).values())),
    "gpt2.lr_half": ("gpt2_lr", lambda g: max(next(t for t, lr in v if lr >= max(x for _, x in v) / 2)
                                              for v in _lr(g).values())),
    # S1, the shared subspaces (appendix A)
    "s1.leave_ov": ("s1_r1", _s1_leaves),
    "s1.leave_qk": ("s1_qk", _s1_leaves),
    "s1.rho_end_ov": ("s1_r1", lambda r: _span(_s1_rho(r, "143000").values())),
    "s1.rho_end_qk": ("s1_qk", lambda r: _span(_s1_rho(r, "143000").values())),
    "s1.rho_null_ov": ("s1_r1", lambda r: max(abs(v - 1) for s in ("0", "128") for v in _s1_rho(r, s).values())),
    "s1.rho_null_qk": ("s1_qk", lambda r: max(abs(v - 1) for s in ("0", "128") for v in _s1_rho(r, s).values())),
    "s1.rho_write": ("s1_r1", lambda r: {s: r["cells"][f"{s}|ov_write"]["rho"] for s in S1_STEPS}),
    "s1.rho_query": ("s1_qk", lambda r: {s: r["cells"][f"{s}|qk_query"]["rho"] for s in S1_STEPS}),
    "s1.rho_ov": ("s1_r1", lambda r: {s: _s1_rho(r, s) for s in S1_STEPS}),
    "s1.rho_qk": ("s1_qk", lambda r: {s: _s1_rho(r, s) for s in S1_STEPS}),
    # the gates (appendix B)
    "a3.gate_rows": ("a3", lambda a: sorted(r for r, g in a["gate"].items() if g.get("null_pass"))),
    "a3.gate_null_min": ("a3", lambda a: min(g["null_no_change"] for g in a["gate"].values()
                                             if g.get("null_no_change") is not None)),
    "a3.gate_effect_pass": ("a3", lambda a: all(e["pass"] for g in a["gate"].values() for e in g.get("effect", []))),
    "a3.n_gate": ("a3", lambda a: a["constants"]["N_GATE"]),
    "a5g.planted_ok": ("a5_seed0", lambda g: [r["planted"] for r in g["planted_null"]["table"]
                                              if r["reads"] == r["planted"]]),
    "a5g.planted_off": ("a5_seed0", lambda g: [[r["planted"], r["reads"]] for r in g["planted_null"]["table"]
                                               if r["reads"] != r["planted"]]),
    "a5g.inside_iff": ("a5_seed0", lambda g: g["planted_null"]["inside_iff_window"]),
    "a5g.null_n": ("a5_seed0", lambda g: g["planted_null"]["n"]),
    "a5g.null_maxz": ("a5_seed0", lambda g: g["planted_null"]["max_z"]),
    "a5g.effect_step": ("a5_seed0", lambda g: g["planted_effect"]["reading"]["step"]),
    "a5g.energy_tol": ("a5_seed0", lambda g: g["known_answer"]["energy_tol"]),
    "a5g.pass": ("a5_seed0", lambda g: g["pass"]),
    # figures 6–10 (C2p unit 1): the series each draws
    "f7.series": ("c3s", lambda c: {s: {"u": b["V_u"], "c": b["V_c"]["0.0001"]} for s, b in c["beside"].items()}),
    "f7.plateaus": ("c3s", lambda c: {s: [d["plateau_uncentred"], d["arms"]["0.0001"]["plateau_centred"]]
                                      for s, d in dict(c["per_seed"], **{"0": c["seed0_development"]}).items()}),
    "f8.mean_depth": ("c5mean", lambda r: {st: r["seeds_data"]["0"]["profile"][st] for st in ("512", "1000")}),
    "f8.same_dir": ("c5b", lambda r: {s: v["same_direction_share"] for s, v in r["R6"]["by_step"].items()}),
    "f9.ratio": ("c5u1", lambda r: {s: [d["S_max_ratio_by_step"][str(t)] for t in GRID] for s, d in _c5seeds(r).items()}),
    "f9.r_min": ("c5set0", lambda r: r["r_min"]),
    "f9.window_max": ("c5u1", lambda r: max(d["S_max_ratio_by_step"][t] for d in _c5seeds(r).values()
                                            for t in C5B_EARLY)),
    "f9.r0b": ("c5u1", lambda r: _span(d["R0b_step"] for d in _c5seeds(r).values())),
    "f10.level": ("c5u2", lambda r: {s: [d["Q3"]["level"][str(t)]["k_half_median"] for t in GRID]
                                     for s, d in _c5seeds(r).items()}),
    "f10.rotated": ("c5u2", lambda r: {s: [d["Q3"]["level"][str(t)]["rotated_k_half_median"] for t in GRID]
                                       for s, d in _c5seeds(r).items()}),
    "f10.break_model": ("c5u2", lambda r: _span(d["Q3"]["level"][t]["k_half_median"] for t in ("512", "1000")
                                                for d in _c5seeds(r).values())),
    "f10.break_rot": ("c5u2", lambda r: _span(d["Q3"]["level"][t]["rotated_k_half_median"] for t in ("512", "1000")
                                              for d in _c5seeds(r).values())),
    "f10.at4000": ("c5u2", lambda r: _span(d["Q3"]["level"]["4000"]["k_half_median"] for d in _c5seeds(r).values())),
}


#: What the draft's sentences assert of the keys, beyond the values it prints. Checked by `produce`
#: (a failing claim refuses the table) and by CI on the tracked table, so a re-run that keeps the
#: render consistent but makes a sentence false cannot pass. Each names its sentence.
CLAIMS: List[tuple] = [
    ("§3.1 the grid is A5's", lambda k: tuple(k["grid.steps"]) == GRID),
    ("§3.2 one date at every tolerance and prompt left out",
     lambda k: len(set(k["s0.tol_steps"])) == 1 and len(set(k["s0.loo_steps"])) == 1
     and k["s0.tol_steps"][0] == k["s0.break"]),
    ("§3.3 the plateau is not the end", lambda k: k["s0.V_end"] != k["s0.plateau"]),
    ("§3.4 registered before the first record", lambda k: k["reg.first"] is True and k["reg.lag_min"] > 0),
    ("§3.4 the falsifier's level is read at the registered K", lambda k: k["reg.falsifier_K"] == FALSIFIER_K),
    ("§4 E₃.₅ dates 512 on every seed, so it differs from E₁ on the 256 seeds",
     lambda k: k["a5.ln1_512"] == k["a5.n"] and len(k["a5.at256"]) > 0),
    ("§5 E5 climbs back past its start between 512 and 1000",
     lambda k: _lane_at(k, "E5", 512) > 0 > _lane_at(k, "E5", 1000)),
    ("§5 E3 peaks at 1000 and falls after",
     lambda k: max(k["a3.lanes"]["E3"]["y"]) == _lane_at(k, "E3", 1000) > k["a3.lanes"]["E3"]["y"][-1]),
    ("§5 E5 and E6u run below the figure's cut, no other drawn row does",
     lambda k: {r for r, l in k["a3.lanes"].items() if min(l["y"]) < LANE_FLOOR} == {"E5", "E6u"}),
    ("§3.4 the branch GitHub first records holds the registration commit, before the first record",
     lambda k: k["reg.pushed_has_commit"] is True and k["reg.pushed_kind"] == "branch_creation"
     and _utc(k["reg.pushed_at"]) < _utc(k["reg.first_record"])),
    ("§4.1 ‖m‖² never falls at 128; falls at 512 on every seed; the 256 seeds fall most at 256",
     lambda k: max(k["mean.falls"]["128"]) == 0 and min(k["mean.falls"]["512"]) > 0
     and set(sorted(range(10), key=lambda s: -k["mean.falls"]["256"][s])[:2]) == set(k["a5.at256"])),
    ("§4.1 the frame-changing transitions hold no fall of ‖m‖² at 128–1000", lambda k: k["mean.frame_falls"] == 0),
    ("§6 arm a: step 0 inside its controls, 256 above all 40",
     lambda k: k["s2.a"]["0"]["min"] <= k["s2.a"]["0"]["value"] <= k["s2.a"]["0"]["max"]
     and k["s2.a"]["256"]["value"] > k["s2.a"]["256"]["max"]),
    ("§3.1 seed 0 reproduces exactly", lambda k: k["s0.V_equal"] is True),
    ("§4 seed 0: no drop up to 128, the break at 512", lambda k: k["s0.Vmax_to128"] == 0 and k["s0.break"] == 512),
    ("§4 all seeds inside, K = n", lambda k: k["a5.K"] == k["a5.n"] and set(k["a5.steps"]) <= {256, 512}),
    ("§4 the rest date 512, as seed 0", lambda k: k["a5.n512"] + len(k["a5.at256"]) == k["a5.n"]),
    ("§4 both outliers inside", lambda k: k["a5.K"] == k["a5.n"]),
    ("§4 a count moved by one stays in the window",
     lambda k: k["a5.V128_max"] + 1 < k["a5.half_min"] and k["a5.V512_min"] - 1 >= k["a5.half_max"]),
    ("§4 V is 0 at 8 to 64 on every seed", lambda k: k["a5.mid_max"] == 0),
    ("§4 seed 0 and the clean seeds", lambda k: k["a5.step0_clean"] + k["a5.step0_n"] == k["a5.n"]),
    ("§4 the second energy agrees", lambda k: k["a5.ln1_512"] == k["a5.n"] and k["s0.ln1_break"] == 512),
    ("§5 the order, strict where written <, tied where =",
     lambda k: _st(k, "E5") < _st(k, "E6u") < _st(k, "E6b") < _st(k, "E1") == _st(k, "E2")
     < _st(k, "E3") == _st(k, "E4") == _st(k, "E10")),
    ("§5 E8 between the break and the tie, 512 from its peak",
     lambda k: _st(k, "E1") <= _st(k, "E8") <= _st(k, "E3") and k["a3.E8_peak"] == _st(k, "E1")),
    ("§5 E6b′ dates as E6b", lambda k: k["a3.E6b_full"] == _st(k, "E6b")),
    ("§5 every aligned pair inside the warmup", lambda k: k["a3.aligned_in_warmup"] is True
     and _st(k, "E3") < k["a3.warmup_end"] == k["cfg.warmup_steps"]),
    ("§5 the pairs add up", lambda k: k["a3.n_aligned"] + k["a3.n_misaligned"] + k["a3.n_unread"] == k["a3.n_pairs"]),
    ("§5 E2's bottom is induction's step", lambda k: k["a3.E2_bottom"] == _st(k, "E3")),
    ("§5 the schedule beside previous-token at 512",
     lambda k: abs(k["a3.share512"]["E10"] - k["a3.share512"]["E4"]) < 0.05),
    ("§5 the shares order the tie", lambda k: k["a3.share512"]["E1"] > k["a3.share512"]["E2"]
     > k["a3.share512"]["E4"] > k["a3.share512"]["E3"]),
    ("§5 every located seed ties, one tie step", lambda k: k["a5.d10_tied"] == k["a5.d10_n"]
     and k["a5.d10_tie_steps"] == [_st(k, "E3")]),
    ("§5 the seeds' D10 counts add up", lambda k: k["a5.d10_prev"] + len(k["a5.d10_other"]) == k["a5.d10_n"]
     and k["a5.d10_n"] + len(k["a5.d10_unlocated"]) == k["a5.n"]),
    ("§5 previous-token's share above induction's on the seeds that say so",
     lambda k: k["a5.d10_prev_share"][0] > k["a5.d10_ind_share"][1]),
    # unit 4: §1, §4.1, §5's E1c, §6, §7, the appendices
    ("§4.1 seed 0's centred date is 1000 at every arm, E1c's step, nothing early",
     lambda k: k["c3s.s0_dates"] == [1000] == [k["c3.step"]] and k["c3s.s0_early"] <= 3
     and k["c3.e1_dev"] == 0 and k["c3s.s0_lane"]["step"] == 1000
     and k["c3s.s0_ufrac"] > 0.9 and k["c3s.s0_cfrac"][1] < 0.5),
    ("§4.1 the tally: K1 = K2, one seed off, at one arm, at its own break",
     lambda k: k["c3s.K1"] == k["c3s.K2"] == k["c3s.n"] - 1 and len(k["c3s.miss"]) == 1
     and [o["seed"] for o in k["c3s.off"]] == k["c3s.miss"]
     and k["c3s.off"][0]["step"] == k["a5.steps"][k["c3s.miss"][0] - 1]
     and k["c3s.events"] == ["reproduced", "reproduced"] and k["c3s.n_cells"] == 4 * k["c3s.n"]),
    ("§4.1 below half at the break on every seed but the tie",
     lambda k: k["c3s.cfrac512"][1] <= 0.5 and k["c3s.cfrac256"][1] < 0.5
     and sorted(k["a5.at256"]) == [3, 9]),
    ("§4.1 the size label withdrawn: matched ratio near 1, frozen ratio under half",
     lambda k: k["c3.label"] == "mostly removed" and k["c3.plateau_c"] < k["c3.plateau_u"] / 2
     and 0.9 <= k["c3.matched_ratio"] <= 1.1),
    ("§5 E1c joins the 1000 tie, its share at 512 between induction's and previous-token's",
     lambda k: k["c3.step"] == _st(k, "E3")
     and k["a3.share512"]["E3"] < k["c3s.s0_cfrac"][0] <= k["c3s.s0_cfrac"][1] < k["a3.share512"]["E4"]),
    ("§6 R1 not shown under both, (a) higher at 512 and 143000, (b) as its controls",
     lambda k: k["s2.r1_answer"] == {"a": "not shown", "b": "not shown"}
     and k["s2.r1_lower"] == {"a": 0, "b": 0} and k["s2.r1a_higher"] == [512, 143000]
     and k["s2.r1b_raw"] == ["as its controls"]),
    ("§6 (b) is about a tenth of a head's norm; copying survives it, far above G2's",
     lambda k: 0.05 < k["s2.b_edit"][0] and k["s2.b_edit"][1] < 0.15
     and k["s2.r2b_raw"] == ["as its controls"] and k["s2.r2b_G"][0] > 2 * k["s2.g2_G"][1]),
    ("§6 G3′ lower at 512, 1000, higher at 2000, never near monotone; G3 reaches 0",
     lambda k: k["s2.g3p_raw"]["512"] == k["s2.g3p_raw"]["1000"] == "lower"
     and k["s2.g3p_raw"]["2000"] == "higher" and k["s2.g3p_span"][0] > 20
     and k["s2.g3_full"] == [0, 0] and k["a3.E2_bottom"] == 1000),
    ("§1, §7 3b: only the centred date excludes the instantaneous rate; E₁'s 256 sits in all three bands",
     lambda k: all(k["c3b.band_u"][s][0] <= 256 <= k["c3b.band_u"][s][1] for s in ("step", "cumulative", "instantaneous"))
     and k["c3b.pred"]["70m"]["uncentred"]["instantaneous"][0] <= 256 <= k["c3b.pred"]["70m"]["uncentred"]["instantaneous"][-1]),
    ("§7 3b: both sizes' dates, the label, the bands",
     lambda k: k["c3b.u"] == {"70m": 256, "160m": 256} and k["c3b.c"] == {"70m": [1000], "160m": [1000]}
     and k["c3b.label"] == "consistent with step and cumulative"
     and k["c3b.band_c"]["instantaneous"][1] < 1000 and k["c3b.band_c70"]["instantaneous"][1] < 1000
     and k["c3b.band_c"]["step"][0] <= 1000 <= k["c3b.band_c"]["step"][1]
     and k["c3b.band_c"]["cumulative"][0] <= 1000 <= k["c3b.band_c"]["cumulative"][1]
     and all(k["c3b.band_u"][s][0] <= 256 <= k["c3b.band_u"][s][1] for s in ("step", "cumulative"))
     and k["c3b.ref"] == {"u": [256, 512], "c": [512, 1000]}),
    ("§7 3c: E₁ in step's band only on every seed; the two stages; the other window splits",
     lambda k: k["c3c.u_consistent"] == ["step"] and k["c3c.label"] == "consistent with step"
     and all(k["c3c.pred"]["step"][0] <= b <= k["c3c.pred"]["step"][1] for b in k["c3c.breaks"])
     and k["c3c.pred"]["cumulative"][0] <= 100 <= k["c3c.pred"]["cumulative"][1]
     and k["c3c.pred"]["step"][0] <= 350 <= k["c3c.pred"]["step"][1]
     and len(set(k["c3c.alt"])) > 1 and min(k["c3c.alt"]) <= k["c3c.pred"]["cumulative"][1]),
    ("§7 3c: no centred date; fixed arms in the cumulative band; step via the null arm only",
     lambda k: k["c3c.c_agree"] == 0
     and k["c3c.pred_c"]["cumulative"][0] <= k["c3c.fixed"][0] <= k["c3c.fixed"][1] <= k["c3c.pred_c"]["cumulative"][1]
     and k["c3c.null_only"] == 4 and k["c3c.early_max"] <= 1),
    ("§7 GPT-2 small's LR: half by 100, peak about 4000", lambda k: k["gpt2.lr_half"] == 100
     and 3900 <= k["gpt2.warmup_end"] <= 4100),
    ("A: the null to 128; who leaves at 256 and 512",
     lambda k: max(k["s1.rho_null_ov"], k["s1.rho_null_qk"]) < 0.002
     and k["s1.leave_qk"] == {"qk_query": 256, "qk_key": 256}
     and k["s1.leave_ov"] == {"ov_write": 256, "ov_cplx": 256, "ov_read": 512, "ov_neg": 512, "ov_pos": 512}),
    ("A: OV write most concentrated at 512 and 1000, query from 2000, QK from 4000",
     lambda k: all(k["s1.rho_ov"][s]["ov_write"] == min(list(k["s1.rho_ov"][s].values()) + list(k["s1.rho_qk"][s].values()))
                   for s in ("512", "1000"))
     and k["s1.rho_qk"]["2000"]["qk_query"] == min(list(k["s1.rho_ov"]["2000"].values()) + list(k["s1.rho_qk"]["2000"].values()))
     and all(max(k["s1.rho_qk"][s].values()) < min(k["s1.rho_ov"][s].values()) for s in ("4000", "143000"))),
    ("B: the gates passed as written",
     lambda k: k["a5g.pass"] is True and k["a5g.inside_iff"] is True and k["a5g.planted_off"] == [[4000, 2000]]
     and k["a5g.effect_step"] == 1000 and k["s0.energy_rel_diff"] < k["a5g.energy_tol"]
     and k["a3.gate_effect_pass"] is True and k["s2.g1_labelled"] == 0 and k["s2.g2_lost"] == 4
     and k["s2.g3_counts"] == 4),
    # unit 5b (design-C2.md "Unit 5b")
    ("§3.2 R1: every series outside the window is one prompt's, at 1000",
     lambda k: k["rd.r1_out_prompts"] == ["camus_letranger"] and k["rd.r1_out_steps"] == [1000]
     and k["rd.r1_inside"] + k["rd.r1_out_n"] == k["rd.r1_n"] == 70),
    ("§3.2 R3: the frame transitions hold no drop at 256–1000; without them every seed dates as V",
     lambda k: all(k["rd.frame_share"][s] == 0 for s in ("256", "512", "1000"))
     and k["rd.inner_dates"] == [k["s0.break"]] + k["a5.steps"] and k["rd.inner_inside"] == 10),
    ("§3.2 R2: the drops sit mid-depth at 256 and 512 and move shallower at 1000",
     lambda k: k["rd.map_median"]["1000"] < k["rd.map_median"]["512"] <= k["rd.map_median"]["256"]),
    ("§3.2, C R4: E₁ rises from first state to last on every series at every step; state 1 to 23 on all before 8000",
     lambda k: k["rd.net_min_pos"] == k["rd.net_n"] and k["rd.net_inner_early"] == k["rd.net_n"]
     and k["rd.net_inner_late"][1] < k["rd.net_n"]),
    ("§3.3 R5: under the later plateau every seed is in the window; only seed 9 moves, 256 to 512",
     lambda k: k["rd.late_inside"] == 10 and k["rd.late_moved"] == [9] and k["rd.late_dates"][9] == 512),
    ("§4.1 R6: every drop at 128–1000 comes with a fall of ‖m‖²; the mean's term carries more than half of all but one",
     lambda k: all(v == 1.0 for v in k["rd.mfall_at"].values())
     and k["rd.s_exc"] == 1 and k["rd.s_median"][0] > 0.5),
    ("abstract, §1, §4.1 R6′: at every step E₁ and ‖m‖² move together at ≥ 0.95 of transitions; ‖m‖² falls at ≥ 0.95 of any fall",
     lambda k: k["rd.same_dir"][0] > 0.95 and k["rd.m_any_fall"][0] > 0.95),
    ("§4.1 R6′: the mean's term is not most of the rises at 16 and 32, but is at every step from 128",
     lambda k: max(k["rd.s_rises"].values()) < 0.5 and k["rd.s_rises_min_from128"] > 0.5),
    ("appendix C, figure 9: the massive coordinates are ordinary at 128–1000 and reach the threshold only after",
     lambda k: k["f9.window_max"] < 10 and k["f9.r0b"][0] > 1000),
    ("appendix C, figure 10: the mean is spread at the break, below a random basis's count by 4000",
     lambda k: k["f10.break_model"][0] >= k["f10.break_rot"][0] and k["f10.at4000"][1] < k["f10.break_rot"][0]),
    ("figure 7's caption: the centred half-rise at 1000 on most seeds", lambda k: k["c3s.K2"] > k["c3s.n"] / 2),
]


def _st(k: Dict[str, object], row: str) -> int:
    return k["a3.rows"][row]["step"]


def check_claims(keys: Dict[str, dict]) -> List[str]:
    k = {n: v["value"] for n, v in keys.items()}
    return [name for name, ok in CLAIMS if not ok(k)]


def compute(records: Dict[str, object]) -> Dict[str, dict]:
    out = {}
    for key, (name, fn) in KEYS.items():
        if name not in RECORDS:
            raise Refused(f"refusing: key {key} names no record ({name})")
        out[key] = {"value": fn(records[name]), "record": RECORDS[name]}
    return out


def _flatten_cfg(c: dict) -> dict:
    """`optimizer.params.lr` sits inside a nested block the line reader also sees as `"lr"`."""
    if "lr" in c:
        c["optimizer_lr"] = c["lr"]
    return c


def produce(reread: Sequence[str] = ()) -> dict:
    hashes = {rid: sha256(resolve(rid)) for rid in RECORDS.values() if resolve(rid).is_file()}
    missing = [rid for rid in RECORDS.values() if rid not in hashes]
    if missing:
        raise Refused(f"refusing: records absent: {missing} (METS_DATA={DATA})")
    if TABLE.is_file():
        old = json.loads(TABLE.read_text())["records"]
        changed = [rid for rid, h in hashes.items() if rid in old and old[rid] != h and rid not in reread]
        if changed:
            raise Refused(f"refusing: records changed since the table was written: {changed}; "
                          f"read the change, then pass --reread for each")
    recs = {name: load(rid) for name, rid in RECORDS.items()}
    for name in recs:
        if name.startswith("cfg_"):
            recs[name] = _flatten_cfg(recs[name])
    audit, score_path = recs["a5_audit"], resolve(RECORDS["a5_score"])
    if audit["score_sha256"] != sha256(score_path):
        raise Refused("refusing: score.json is not the one P-A5's tracked real-run record hashes")
    keys = compute(recs)
    false = check_claims(keys)
    if false:
        raise Refused(f"refusing: the draft's sentences no longer hold: {false}")
    if TABLE.is_file():
        old = json.loads(TABLE.read_text())["keys"]
        moved = [n for n, v in keys.items() if n in old and old[n] != v
                 and n not in reread and v["record"] not in reread]
        if moved:
            raise Refused(f"refusing: keys changed value or record with no record's change named: {moved}; "
                          f"read the change, then pass --reread for each key or record")
    return {"producer": "python -m trackC_writeup.numbers produce",
            "code": sha256(Path(__file__))[:12],
            "records": hashes, "keys": keys}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _num(v) -> str:
    if isinstance(v, bool):
        raise Refused(f"refusing: {v!r} is a flag; name a format")
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        return _grouped(int(v))
    raise Refused(f"refusing: {v!r} is not an integer; name a format (f1, f2, f3, sci)")


def _grouped(n: int) -> str:
    """143 000, as the draft writes it; four-digit numbers stay whole (1430)."""
    s = str(abs(n))
    if len(s) > 4:
        s = f"{int(s):,}".replace(",", " ")
    return ("−" if n < 0 else "") + s


def _sci(v: float) -> str:
    m, e = f"{v:.1e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m} × 10{str(int(e)).translate(_SUP)}" if m != "1" else f"10{str(int(e)).translate(_SUP)}"


def _andstr(xs: Sequence[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _and(xs: Sequence) -> str:
    return _andstr([_num(x) for x in xs])


FORMATS: Dict[str, Callable] = {
    "f0": lambda v: _grouped(round(v)), "f1": lambda v: f"{v:.1f}", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}",
    "sci": _sci, "int": _num, "str": str,
    "list": lambda v: ", ".join(_num(x) for x in v),
    "and": _and,
    "ands": lambda v: _andstr(list(v)),
    "or": lambda v: _and(sorted(set(v))).replace(" and ", " or "),
    "range": lambda v: f"{_num(v[0])}–{_num(v[1])}",
    "range2": lambda v: f"{v[0]:.2f}–{v[1]:.2f}",
    "range3": lambda v: f"{v[0]:.3f}–{v[1]:.3f}",
    "date": lambda v: v[:10],
    "onoff": lambda v: "on" if v else "off",
    "tz": lambda v: "UTC" + ("+" if v[19] == "+" else "−") + str(int(v[20:22])),
    "scis": lambda v: _andstr([_sci(x) for x in v]),
    "word": lambda v: WORDS[int(v)], "Word": lambda v: WORDS[int(v)].capitalize(),
    "uniq": lambda v: _and(sorted(set(v))) if len(set(v)) > 1 else _num(v[0]),
    "step": lambda v: ("refused" if v["reading"].startswith("refused") else
                       f"censored at {_num(v['step'])}" if v["censored"] else _num(v["step"])),
    "time": lambda v: v[11:16],
    "utc": lambda v: _utc(v).strftime("%H:%M"),
    "v": lambda c: _num(c["value"]),
    "span": lambda c: f"{_num(c['min'])}–{_num(c['max'])}",
    "minmax": lambda v: f"{_num(min(v))}–{_num(max(v))}" if min(v) != max(v) else _num(v[0]),
}
WORDS = ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
SLOT = re.compile(r"\{\{([\w.]+)(?:\[(\w+)\])?(?:\|(\w+))?\}\}")


def render(source: str, keys: Dict[str, dict]) -> str:
    def one(m: re.Match) -> str:
        key, sub, fmt = m.groups()
        if key not in keys:
            raise Refused(f"refusing: {{{{{key}}}}} has no record (not in {TABLE.name})")
        v = keys[key]["value"]
        if sub is not None:
            v = v[sub] if isinstance(v, dict) else v[int(sub)]
        if fmt is None:
            return _num(v) if not isinstance(v, str) else v
        if fmt not in FORMATS:
            raise Refused(f"refusing: format {fmt!r} unknown ({sorted(FORMATS)})")
        return FORMATS[fmt](v)
    head = (f"<!-- trackC_writeup/draft-C2.md — rendered by `python -m trackC_writeup.numbers render` "
            f"from draft-C2.src.md and numbers.json; edit the source, not this file. -->\n")
    body = re.sub(r"\A<!--.*?-->\n", "", source, count=1, flags=re.S)
    return head + SLOT.sub(one, body)


# ---------------------------------------------------------------------------
# Figures (from the table alone)
# ---------------------------------------------------------------------------

INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d7d2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # dataviz reference palette, slots 1–3
WARM = "#efeee9"
LANE_FLOOR = -1.2      # figure 2 cuts a lane here: its late fall (E5, E6u) is not what it dates


def _style(plt) -> None:
    plt.rcParams.update({"svg.hashsalt": "c2", "svg.fonttype": "none", "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.6,
                         "lines.linewidth": 1.6, "legend.frameon": False})


def _xpos(steps: Sequence[int]) -> List[float]:
    """log2(step), with step 0 drawn one octave left of step 2 (the grid has no step 1)."""
    import math
    return [math.log2(s) if s else 0.0 for s in steps]


def _tick(s: int) -> str:
    return f"{s // 1000}k" if s >= 1000 else str(s)


def _xaxis(ax, steps: Sequence[int], warmup: int) -> None:
    import math
    ticks = [s for s in steps if s in (0, 2, 8, 32, 128, 512, 2000, 8000, 32000, 143000)]
    ax.set_xticks(_xpos(ticks), [_tick(s) for s in ticks])
    ax.axvspan(_xpos([2])[0] - 1, math.log2(warmup), color=WARM, zorder=0, lw=0)
    ax.set_xlim(-0.3, math.log2(steps[-1]) + 0.3)


def _save(fig, name: str) -> Path:
    """The SVG (the Markdown draft's) in `fig/`, and a PDF in `tex/fig/` (the LaTeX draft's, `to_tex.py`)."""
    FIGDIR.mkdir(exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    PDFDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDFDIR / path.with_suffix(".pdf").name, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    return path


def _toy(n: int = 40, beta: float = 1.0, dt: float = 0.02, t_end: float = 8.0, seed: int = 0):
    """Figure 6's illustration, not a record: n tokens on S² under the identity-weight dynamics
    ẋᵢ = P_{xᵢ}(Σⱼ softmaxⱼ(β⟨xᵢ, xⱼ⟩) xⱼ) [Geshkovski et al. 2023], Euler steps renormalised to
    the sphere, and E_β along it. Refuses if E_β ever falls (§2.1's monotonicity, the known answer)."""
    import numpy as np
    x = np.random.default_rng(seed).normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ts, xs, es = [], [], []
    for i in range(int(round(t_end / dt)) + 1):
        g = x @ x.T
        ts.append(i * dt), xs.append(x.copy()), es.append(np.exp(beta * g).sum() / (2 * beta * n * n))
        w = np.exp(beta * (g - g.max(axis=1, keepdims=True)))
        v = (w / w.sum(axis=1, keepdims=True)) @ x
        v -= (v * x).sum(axis=1, keepdims=True) * x
        x = x + dt * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    if min(np.diff(es)) < -1e-12:
        raise Refused("refusing: the toy's E_β fell, so figure 6 would not show §2.1's picture")
    return ts, xs, es


def figures(keys: Dict[str, dict]) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    k = {n: v["value"] for n, v in keys.items()}
    steps, warmup, x = k["grid.steps"], k["cfg.warmup_steps"], _xpos(k["grid.steps"])
    out = []

    # Figure 1: V by step on the 10 seeds, raw and against each seed's plateau
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    runs = {"0": (k["s0.V"], k["s0.plateau"])}
    runs.update({s: (k["a5.V"][s], k["a5.plateaus"][int(s) - 1]) for s in SEEDS})
    early = {str(s) for s in k["a5.at256"]}
    for s, (v, plat) in sorted(runs.items(), key=lambda r: (r[0] == "0", r[0] in early)):
        col, lw, z = ((SERIES[0], 2.2, 3) if s == "0" else (SERIES[1], 1.4, 2) if s in early
                      else ("#9a9893", 1.0, 1))
        axs[0].plot(x, v, color=col, lw=lw, zorder=z)
        axs[1].plot(x, [y / plat for y in v], color=col, lw=lw, zorder=z)
    axs[1].axhline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    axs[1].text(x[1], 0.53, "half the 1000–4000 plateau", color=MUTED, fontsize=8)
    for ax in axs:
        _xaxis(ax, steps, warmup)
        ax.axvspan(x[steps.index(128)], x[steps.index(512)], color=SERIES[0], alpha=0.08, lw=0)
        ax.set_xlabel("training step (log scale)")
    axs[0].set_ylabel("V, layer-to-layer drops in E₁")
    axs[1].set_ylabel("V / plateau median")
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], color=SERIES[0], lw=2.2), Line2D([], [], color=SERIES[1], lw=1.4),
                   Line2D([], [], color="#9a9893", lw=1.0)],
                  ["seed 0", f"seeds {_and(k['a5.at256'])} (dated {_num(256)})",
                   f"seeds {_and([int(s) for s in SEEDS if int(s) not in k['a5.at256']])}"],
                  loc="upper left", fontsize=8)
    out.append(_save(fig, "fig1_break.svg"))
    plt.close(fig)

    # Figure 2: A3's rows on seed 0, one lane each, with the centred cloud (unit 3) beside the 1000 tie
    lanes = dict(k["a3.lanes"], E1c=k["c3s.s0_lane"])
    rows = list(A3_ROWS[:-1]) + ["E1c", A3_ROWS[-1]]
    fig, axs = plt.subplots(len(rows), 1, figsize=(7.2, 0.62 * len(rows) + 0.6), sharex=True)
    for ax, r in zip(axs, rows):
        lane = lanes[r]
        ax.plot(x, lane["y"], color={"E10": SERIES[2], "E1c": SERIES[1]}.get(r, SERIES[0]), lw=1.4)
        if lane["step"] is not None:
            ax.plot([x[steps.index(lane["step"])]], [lane["y"][steps.index(lane["step"])]], "o",
                    ms=5, color=INK, zorder=4)
        ax.set_yticks([0, 1])
        ax.set_ylim(max(LANE_FLOOR, min(-0.25, min(lane["y"]) - 0.1)), max(1.35, max(lane["y"]) + 0.1))
        ax.grid(axis="x", visible=True)
        ax.text(1.01, 0.5, f"{r}  {_num(lane['step'])}", transform=ax.transAxes, va="center",
                fontsize=8, color=INK)
        _xaxis(ax, steps, warmup)
    axs[-1].set_xlabel("training step (log scale); dot: the row's dated step; shaded: the LR warmup")
    out.append(_save(fig, "fig2_timeline.svg"))
    plt.close(fig)

    # Figure 3: S2's R1 cells and G3′ against their 40 random edits, base V beside
    panels = (("s2.a", "a: symmetric part"), ("s2.b", "b: rotation removed"),
              ("s2.g3p", "every head's step-128 OV map"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (key, title) in zip(axs, panels):
        cells = k[key]
        st = [int(s) for s in S2_STEPS if s in cells]
        xs = [S2_STEPS.index(str(s)) for s in st]          # evenly spaced: the cells, not the axis
        base = [k["s0.V"][steps.index(s)] for s in st]
        ax.plot(xs, base, color="#9a9893", lw=1.0, marker="o", ms=3, label="unedited")
        for xi, s in zip(xs, st):
            c = cells[str(s)]
            ax.plot([xi, xi], [c["min"], c["max"]], color=SERIES[1], lw=3, alpha=0.45,
                    solid_capstyle="round")
        ax.plot(xs, [cells[str(s)]["value"] for s in st], color=SERIES[0], marker="o", ms=4, lw=1.4,
                label="edited")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(xs, [_tick(s) for s in st], fontsize=7.5)
        ax.set_xlim(-0.5, len(S2_STEPS) - 0.5)
        ax.set_xlabel("step")
    axs[0].set_ylabel("V")
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], color=SERIES[0], marker="o", ms=4), Line2D([], [], color="#9a9893", marker="o", ms=3),
                   Line2D([], [], color=SERIES[1], lw=3, alpha=0.45)],
                  ["edited", "unedited", "40 random edits of the same size (range)"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.02), bbox_transform=fig.transFigure,
                  ncol=3, fontsize=8)
    out.append(_save(fig, "fig3_ov.svg"))
    plt.close(fig)

    # Figure 4: the other schedules (§7): Pythia 70m / 160m against seed 0, and GPT-2 small's E₁
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    ax.plot(x, [v / k["s0.plateau"] for v in k["s0.V"]], color=SERIES[0], lw=2.0, label="410m, seed 0")
    for (size, col) in (("70m", SERIES[1]), ("160m", SERIES[2])):
        r = k["c3b.V"][size]
        ax.plot(x, [v / r["pu"] for v in r["u"]], color=col, lw=1.4, label=f"{size}, E₁")
        ax.plot(x, [v / r["pc"] for v in r["c"]], color=col, lw=1.2, ls=(0, (3, 2)), label=f"{size}, centred")
    ax.axhline(0.5, color=INK, lw=0.8, ls=(0, (1, 2)))
    _xaxis(ax, steps, warmup)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("V / plateau median")
    ax.legend(loc="upper left", fontsize=7)
    ax = axs[1]
    g = k["c3c.grid"]
    gx = _xpos(g)
    for lo_hi, col in ((k["c3c.pred"]["cumulative"], SERIES[1]), (k["c3c.pred"]["step"], SERIES[0])):
        ax.axvspan(*_xpos(lo_hi), color=col, alpha=0.10, lw=0)
    for i, s in enumerate(C3C_SEEDS):          # unit 5b (R8): over each seed's frozen plateau
        ax.plot(gx, [v / k["c3c.plateaus"][i] for v in k["c3c.Vu"][s]], color="#6f6d68", lw=1.0)
    for ys in k["c3c.Vc_med"].values():
        ax.plot(gx, ys, color=SERIES[1], lw=1.0, ls=(0, (3, 2)))
    lr = k["gpt2.lr_frac"]
    ax.plot(_xpos([t for t, _ in lr]), [f for _, f in lr], color=SERIES[2], lw=1.2)
    ax.axhline(0.5, color=INK, lw=0.8, ls=(0, (1, 2)))
    ticks = [s for s in g if s in (0, 20, 100, 350, 1000, 4000, 16000)]
    ax.set_xticks(_xpos(ticks), [_tick(s) for s in ticks])
    ax.set_xlim(-0.3, _xpos([g[-1]])[0] + 0.3)
    ax.text(_xpos([k["c3c.pred"]["cumulative"][0]])[0], ax.get_ylim()[1] * 0.92, "cumulative", fontsize=7, color=MUTED)
    ax.text(_xpos([k["c3c.pred"]["step"][0]])[0], ax.get_ylim()[1] * 0.80, "step", fontsize=7, color=MUTED)
    ax.legend([Line2D([], [], color="#6f6d68", lw=1.0), Line2D([], [], color=SERIES[1], lw=1.0, ls=(0, (3, 2))),
               Line2D([], [], color=SERIES[2], lw=1.2)],
              ["E₁, each seed", "centred, each tolerance (seed median)", "learning rate / peak"],
              loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6.5)
    ax.set_xlabel("GPT-2 small training step (log scale)")
    ax.set_ylabel("V / plateau median; LR / peak")
    out.append(_save(fig, "fig4_schedules.svg"))
    plt.close(fig)

    # Figure 5 (appendix C; unit 5b, R2 and R7): where the drops sit, and E₁ by depth
    from matplotlib.colors import LinearSegmentedColormap
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.5})
    ax = axs[0]
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", SERIES[0], "#0d2f5c"])
    im = ax.imshow(k["rd.map"], aspect="auto", cmap=seq, vmin=0, vmax=70, interpolation="nearest",
                   extent=(-0.5, len(steps) - 0.5, len(k["rd.map"]) + 0.5, 0.5))
    shown = [s for s in steps if s in (0, 8, 128, 512, 2000, 8000, 32000, 143000)]
    ax.set_xticks([steps.index(s) for s in shown], [_tick(s) for s in shown], fontsize=7.5)
    ax.set_yticks([1, 6, 12, 18, 24])
    ax.grid(False)
    ax.set_xlabel("training step (grid order)")
    ax.set_ylabel("transition (1 = embeddings → block 0)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("drops, of 70", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax = axs[1]
    ramp = ("#9cc3ee", "#5a9be0", SERIES[0], "#0d2f5c")    # one hue, light to dark with the step
    depth = k["rd.depth"]
    states = list(range(len(depth["0"]["128"])))
    for col, st in zip(ramp, ("128", "512", "1000", "143000")):
        others = [depth[s][st] for s in depth if s != "0"]
        ax.fill_between(states, [min(c) for c in zip(*others)], [max(c) for c in zip(*others)],
                        color=col, alpha=0.15, lw=0)
        ax.plot(states, depth["0"][st], color=col, lw=1.6, label=f"step {_tick(int(st))}")
    ax.set_xlabel("state (0 = embeddings)")
    ax.set_ylabel("E₁, mean over prompts")
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.legend(loc="upper left", fontsize=7)
    out.append(_save(fig, "fig5_count.svg"))
    plt.close(fig)

    # Figure 6 (§2.1; an illustration from `_toy`, no record): the identity-weight picture
    import numpy as np
    ts, xs, es = _toy()
    fig = plt.figure(figsize=(7.2, 2.3))
    snaps = (0.0, 2.0, 3.5)
    for j, t in enumerate(snaps):
        ax = fig.add_axes([0.01 + 0.19 * j, 0.08, 0.18, 0.80])
        p = xs[min(range(len(ts)), key=lambda i: abs(ts[i] - t))]
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=MUTED, lw=0.8))
        front = p[:, 2] >= 0
        ax.scatter(p[front, 0], p[front, 1], s=12, color=SERIES[0], zorder=3, lw=0)
        ax.scatter(p[~front, 0], p[~front, 1], s=12, facecolor="none", edgecolor=SERIES[0], lw=0.7, zorder=2)
        ax.set_xlim(-1.08, 1.08), ax.set_ylim(-1.08, 1.08), ax.set_aspect("equal"), ax.axis("off")
        ax.set_title(f"depth t = {t:g}", fontsize=8.5, color=INK)
    ax = fig.add_axes([0.66, 0.2, 0.32, 0.68])
    ax.plot(ts, es, color=SERIES[0], lw=1.6)
    for t in snaps:
        ax.axvline(t, color=RULE, lw=0.8, zorder=0)
    ax.set_xlabel("depth t (layers, read as time)")
    ax.set_ylabel("E₁")
    out.append(_save(fig, "fig6_picture.svg"))
    plt.close(fig)

    # Figure 7 (§4.1; units 3 and 3s): the two events on the ten seeds, uncentred and centred
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    ser, plat = k["f7.series"], k["f7.plateaus"]
    for s in sorted(ser, key=lambda s: s == "0"):
        col, lw, z = (INK, 2.0, 3) if s == "0" else ("#9a9893", 1.0, 1)
        for ax, part, pi in ((axs[0], "u", 0), (axs[1], "c", 1)):
            ax.plot(x, [v / plat[s][pi] for v in ser[s][part]], color=col if s == "0" else
                    (SERIES[0] if part == "u" else SERIES[1]), lw=lw, alpha=1 if s == "0" else 0.55, zorder=z)
    for ax, title in zip(axs, ("E₁ as read (uncentred)", "each prompt's mean token removed (centred, τ = 10⁻⁴)")):
        _xaxis(ax, steps, warmup)
        ax.axhline(0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(x[steps.index(512)], color=SERIES[0], lw=0.8, ls=(0, (1, 2)))
        ax.axvline(x[steps.index(1000)], color=SERIES[1], lw=0.8, ls=(0, (1, 2)))
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("training step (log scale)")
    axs[0].set_ylabel("V / plateau median")
    for ax, col in zip(axs, SERIES[:2]):
        ax.legend([Line2D([], [], color=INK, lw=2.0), Line2D([], [], color=col, lw=1.0, alpha=0.55)],
                  ["seed 0", "seeds 1–9"], loc="upper left", fontsize=7.5)
    out.append(_save(fig, "fig7_events.svg"))
    plt.close(fig)

    # Figure 8 (§2.1, appendix C; unit 5b): E₁ and its mean term by depth, and how often they move together
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})
    ax = axs[0]
    for col, st in ((SERIES[0], "512"), ("#0d2f5c", "1000")):
        e1, m2 = k["rd.depth"]["0"][st], k["f8.mean_depth"][st]
        mterm = [0.5 + 0.5 * v for v in m2]
        states = list(range(len(e1)))
        ax.plot(states, e1, color=col, lw=1.6, label=f"E₁, step {_tick(int(st))}")
        ax.plot(states, mterm, color=col, lw=1.2, ls=(0, (3, 2)), label=f"½ + ½‖m‖², step {_tick(int(st))}")
        falls = [i + 1 for i in range(len(e1) - 1) if e1[i + 1] < e1[i]]
        ax.plot(falls, [e1[i] for i in falls], "v", color=SERIES[1], ms=4, zorder=4)
    ax.plot([], [], "v", color=SERIES[1], ms=4, label="a drop in E₁")
    ax.set_xlabel("state (0 = embeddings)")
    ax.set_ylabel("mean over the 7 prompts")
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.legend(loc="upper left", fontsize=6.5, ncol=1)
    ax = axs[1]
    sd = k["f8.same_dir"]
    ax.plot(x, [sd[str(s)] for s in steps], color=SERIES[0], marker="o", ms=3, lw=1.4)
    _xaxis(ax, steps, warmup)
    ax.set_xticks(_xpos([0, 8, 128, 2000, 32000]), [_tick(s) for s in (0, 8, 128, 2000, 32000)])
    ax.set_ylim(0.9, 1.005)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("share of transitions where E₁\nand ‖m‖² move the same way")
    out.append(_save(fig, "fig8_mean.svg"))
    plt.close(fig)

    # Figure 9 (§4.1; C5 unit 1): the massive-activation coordinates arrive long after both events
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for s in range(10):
        ax.plot(x, k["f9.ratio"][str(s)], color=INK if s == 0 else "#9a9893", lw=2.0 if s == 0 else 1.0,
                zorder=3 if s == 0 else 1)
    ax.set_yscale("log")
    ax.axhline(k["f9.r_min"], color=SERIES[1], lw=0.9, ls=(0, (3, 2)))
    ax.text(x[1], k["f9.r_min"] * 1.15, f"{_num(k['f9.r_min'])}×, the set's threshold", fontsize=7.5, color=MUTED)
    _xaxis(ax, steps, warmup)
    ax.axvspan(x[steps.index(128)], x[steps.index(1000)], color=SERIES[0], alpha=0.08, lw=0)
    ax.set_xlabel("training step (log scale); blue band: 128–1000")
    ax.set_ylabel("largest coordinate of the set,\nover the median coordinate's")
    ax.legend([Line2D([], [], color=INK, lw=2.0), Line2D([], [], color="#9a9893", lw=1.0)],
              ["seed 0", "seeds 1–9"], loc="upper left", fontsize=7.5)
    out.append(_save(fig, "fig9_massive.svg"))
    plt.close(fig)

    # Figure 10 (§4.1; C5 unit 2): how many coordinates hold half of ‖m‖², against a random basis
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    lev, rot = k["f10.level"], k["f10.rotated"]
    ax.fill_between(x, [min(c) for c in zip(*rot.values())], [max(c) for c in zip(*rot.values())],
                    color="#9a9893", alpha=0.35, lw=0, label="the same means, random bases (range)")
    for s in sorted(lev, key=lambda s: s == "0"):
        ax.plot(x, lev[s], color=INK if s == "0" else SERIES[0], lw=2.0 if s == "0" else 1.0,
                alpha=1 if s == "0" else 0.6, zorder=3 if s == "0" else 2)
    ax.plot([], [], color=INK, lw=2.0, label="seed 0, the model's coordinates")
    ax.plot([], [], color=SERIES[0], lw=1.0, label="seeds 1–9")
    ax.set_yscale("log")
    _xaxis(ax, steps, warmup)
    ax.axvspan(x[steps.index(128)], x[steps.index(1000)], color=SERIES[0], alpha=0.08, lw=0)
    ax.set_xlabel("training step (log scale); blue band: 128–1000")
    ax.set_ylabel("coordinates holding half of ‖m‖²\n(median over states, of 1024)")
    ax.legend(loc="lower left", fontsize=7)
    out.append(_save(fig, "fig10_level.svg"))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="C2's numbers, draft and figures from their records")
    ap.add_argument("step", choices=("produce", "render", "check", "figures"))
    ap.add_argument("--reread", action="append", default=[],
                    help="a record id or key whose change has been read (produce only)")
    a = ap.parse_args(argv)
    if a.step == "produce":
        table = produce(a.reread)
        TABLE.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {TABLE.relative_to(REPO)}: {len(table['keys'])} keys from {len(table['records'])} records")
        return 0
    keys = json.loads(TABLE.read_text())["keys"]
    if a.step == "figures":
        for p in figures(keys):
            print(f"wrote {p.relative_to(REPO)}")
        return 0
    text = render(SOURCE.read_text(), keys)
    if a.step == "render":
        DRAFT.write_text(text)
        print(f"wrote {DRAFT.relative_to(REPO)}")
        return 0
    if DRAFT.read_text() != text:
        print(f"{DRAFT.relative_to(REPO)} is not the render of {SOURCE.name} and {TABLE.name}; "
              f"run `python -m trackC_writeup.numbers render`", file=sys.stderr)
        return 1
    print(f"{DRAFT.relative_to(REPO)} matches its render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
