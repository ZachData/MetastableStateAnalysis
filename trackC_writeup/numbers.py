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
- **figures** draws `fig/fig{1,2,3}_*.svg` from the table alone (matplotlib, deterministic SVG).

Numbers a paper reports are that paper's and carry its citation; numbers from this project's
records and from Pythia's released training configs are keys.

    METS_DATA=<main>/data python -m trackC_writeup.numbers produce [--reread data:<path> ...]
    python -m trackC_writeup.numbers render | check | figures
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
from typing import Callable, Dict, List, Optional, Sequence

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
HERE = Path(__file__).resolve().parent
TABLE = HERE / "numbers.json"
SOURCE = HERE / "draft-C2.src.md"
DRAFT = HERE / "draft-C2.md"
FIGDIR = HERE / "fig"

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
    # EleutherAI/pythia models/{70M,160M,410M}/pythia-*.yml, fetched 2026-10-22 from main
    "cfg_70m": "data:trackC/configs/pythia-70m.yml",
    "cfg_160m": "data:trackC/configs/pythia-160m.yml",
    "cfg_410m": "data:trackC/configs/pythia-410m.yml",
}
SEEDS = tuple(str(s) for s in range(1, 10))
S2_STEPS = ("0", "256", "512", "1000", "2000", "4000", "143000")
#: The rows §5's table and figure 2 show, in the order of their dates.
A3_ROWS = ("E5", "E6u", "E6b", "E1", "E2", "E8", "E3", "E4", "E10")


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


def _v(score: dict, seed: str) -> List[float]:
    return score["notes"][seed]["V"]


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


def _utc(iso: str):
    from datetime import datetime, timezone
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


def _early_argmax(score: dict) -> int:
    """The grid step (of 0, 2, 4) at which the largest early count sits, on the seed that has it."""
    seed = max(SEEDS, key=lambda k: max(_v(score, k)[:3]))
    v = _v(score, seed)[:3]
    return (0, 2, 4)[v.index(max(v))]


def _lane(a3: dict, rid: str) -> dict:
    return next(x for x in a3["reading"]["lanes"] if x["id"] == rid)


def _verdicts(a3: dict, v: str) -> int:
    return sum(p["verdict"] == v for p in a3["reading"]["pairs"])


def _s2_cells(rec: dict, arm: str) -> dict:
    cells = rec["R1"][arm]["cells"]
    return {s: {"value": cells[s]["value"], **cells[s]["ctl"]} for s in S2_STEPS if s in cells}


def _g3p(rec: dict) -> dict:
    return {s: {"value": c["value"], **c["ctl"]} for s, c in rec["G3p"]["cells"].items()}


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
    "grid.last_early": ("a3", lambda a: max(s for s in _steps(a) if s < 1000)),
    "a5.prompts_hash": ("a5_seed0", lambda g: g["prompts_hash"]),
    "a5.code": ("a5_seed0", lambda g: g["code"]),
    "a5.max_tokens": ("a5_plan0", lambda p: p["max_tokens"]),
    "a5.dtype": ("a5_plan0", lambda p: p["dtype"]),
    "a5.tf32": ("a5_plan0", lambda p: p["tf32"]),
    "rel_tol": ("metrics", lambda s: constant(s, "ENERGY_VIOLATION_REL_TOL")),
    "kappa": ("evalues", lambda s: constant(s, "DEFAULT_KAPPA")),
    "a5.n_prompts": ("a5_seed0", lambda g: len(g["sensitivity"]["leave_one_out"])),
    "a5.prompts": ("a5_seed0", lambda g: sorted(g["sensitivity"]["leave_one_out"])),
    # seed 0 (§4; A5's gate record, whose V equals A3's E1 and the panel's)
    "s0.V": ("a5_seed0", lambda g: g["known_answer"]["V"]),
    "s0.V128": ("a5_seed0", lambda g: g["known_answer"]["V"][7]),
    "s0.V256": ("a5_seed0", lambda g: g["known_answer"]["V"][8]),
    "s0.V512": ("a5_seed0", lambda g: g["known_answer"]["V"][9]),
    "s0.Vmax_to128": ("a5_seed0", lambda g: max(g["known_answer"]["V"][:8])),
    "s0.plateau": ("a5_seed0", lambda g: g["known_answer"]["break"]["plateau_median"]),
    "s0.half": ("a5_seed0", lambda g: g["known_answer"]["break"]["plateau_median"] / 2),
    "s0.break": ("a5_seed0", lambda g: g["known_answer"]["break"]["step"]),
    "s0.V_end": ("a5_seed0", lambda g: g["known_answer"]["V"][-1]),
    "s0.ln1_break": ("a5_seed0", lambda g: g["d10"]["seed_reading"]["beside_ln1_b35"]["break"]["step"]),
    "s0.energy_rel_diff": ("a5_seed0", lambda g: g["known_answer"]["energy_max_rel_diff"]),
    "s0.V_equal": ("a5_seed0", lambda g: g["known_answer"]["V_equal"]),
    "s0.planted_effect": ("a5_seed0", lambda g: g["planted_effect"]["reading"]["step"]),
    "s0.tol_steps": ("a5_seed0", lambda g: [g["sensitivity"]["rel_tol"][t]["step"]
                                            for t in sorted(g["sensitivity"]["rel_tol"], key=float)]),
    "s0.loo_steps": ("a5_seed0", lambda g: sorted({v["step"] for v in g["sensitivity"]["leave_one_out"].values()})),
    "s0.null_tail_ge7": ("a5_seed0", lambda g: g["planted_null"]["tail_ge7"]),
    # the registration (§3)
    "reg.at": ("a5_audit", lambda a: a["order"]["registered_at"]),
    "reg.first_record": ("a5_audit", lambda a: a["order"]["first_record_at"]),
    "reg.commit": ("a5_audit", lambda a: a["order"]["registered_commit"][:7]),
    "reg.first": ("a5_audit", lambda a: a["order"]["registered_first"]),
    "reg.lag_min": ("a5_audit", lambda a: round((_utc(a["order"]["first_record_at"])
                                                 - _utc(a["order"]["registered_at"])).total_seconds() / 60)),
    "reg.relevance": ("registry", lambda r: _p_a5(r)["relevance"]),
    "reg.falsifier_K": ("registry", lambda r: int(re.search(r"K <= (\d+)", _p_a5(r)["falsifier"]).group(1))),
    "reg.p0": ("a5_score", lambda s: str(Fraction(s["p0"]).limit_denominator(100))),
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
    "a5.V128_max": ("a5_score", lambda s: max(_v(s, k)[7] for k in SEEDS)),
    "a5.half_min": ("a5_score", lambda s: min(s["per_seed"][k]["plateau_median"] for k in SEEDS) / 2),
    "a5.V512_min": ("a5_score", lambda s: min(_v(s, k)[9] for k in SEEDS)),
    "a5.half_max": ("a5_score", lambda s: max(s["per_seed"][k]["plateau_median"] for k in SEEDS) / 2),
    "a5.step0_seeds": ("a5_score", lambda s: [int(k) for k in SEEDS if _v(s, k)[0] > 0]),
    "a5.step0_n": ("a5_score", lambda s: sum(_v(s, k)[0] > 0 for k in SEEDS)),
    "a5.step0_min": ("a5_score", lambda s: min(_v(s, k)[0] for k in SEEDS if _v(s, k)[0] > 0)),
    "a5.step0_max": ("a5_score", lambda s: max(_v(s, k)[0] for k in SEEDS)),
    "a5.early_max": ("a5_score", lambda s: max(max(_v(s, k)[:3]) for k in SEEDS)),
    "a5.early_max_seed": ("a5_score", lambda s: int(max(SEEDS, key=lambda k: max(_v(s, k)[:3])))),
    "a5.early_max_step": ("a5_score", lambda s: _early_argmax(s)),
    "a5.step0_clean": ("a5_score", lambda s: sum(_v(s, k)[0] == 0 for k in SEEDS)),
    "a5.mid_max": ("a5_score", lambda s: max(max(_v(s, k)[3:7]) for k in SEEDS)),
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
    "a3.code": ("a3", lambda a: a["code"]),
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
    "a3.floor": ("a3", lambda a: a["constants"]["FLOOR"]),
    # S2 (figure 3; §6 is unit 4's)
    "s2.code": ("s2_readings", lambda r: r["code"]),
    "s2.a": ("s2_readings", lambda r: _s2_cells(r, "a")),
    "s2.b": ("s2_readings", lambda r: _s2_cells(r, "b")),
    "s2.g3p": ("s2_gates", _g3p),
}


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
    return {"producer": "python -m trackC_writeup.numbers produce",
            "code": sha256(Path(__file__))[:12],
            "records": hashes, "keys": compute(recs)}


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


def _and(xs: Sequence) -> str:
    xs = [_num(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


FORMATS: Dict[str, Callable] = {
    "f0": lambda v: _grouped(round(v)), "f1": lambda v: f"{v:.1f}", "f2": lambda v: f"{v:.2f}", "f3": lambda v: f"{v:.3f}",
    "sci": _sci, "int": _num, "str": str,
    "list": lambda v: ", ".join(_num(x) for x in v),
    "and": _and,
    "range": lambda v: f"{_num(v[0])}–{_num(v[1])}",
    "range2": lambda v: f"{v[0]:.2f}–{v[1]:.2f}",
    "range3": lambda v: f"{v[0]:.3f}–{v[1]:.3f}",
    "date": lambda v: v[:10],
    "onoff": lambda v: "on" if v else "off",
    "word": lambda v: WORDS[int(v)], "Word": lambda v: WORDS[int(v)].capitalize(),
    "uniq": lambda v: _and(sorted(set(v))) if len(set(v)) > 1 else _num(v[0]),
    "step": lambda v: ("refused" if v["reading"].startswith("refused") else
                       f"censored at {_num(v['step'])}" if v["censored"] else _num(v["step"])),
    "time": lambda v: v[11:16],
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

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # dataviz reference palette, slots 1–3
WARM = "#efeee9"
LANE_FLOOR = -1.2      # figure 2 cuts a lane here: its late fall (E5, E6u) is not what it dates


def _style(plt) -> None:
    plt.rcParams.update({"svg.hashsalt": "c2", "svg.fonttype": "none", "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
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
    FIGDIR.mkdir(exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    return path


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
                  ["seed 0", f"seeds {_and(k['a5.at256'])} (dated {_num(256)})", "seeds 1–9, the rest"],
                  loc="upper left", fontsize=8)
    out.append(_save(fig, "fig1_break.svg"))
    plt.close(fig)

    # Figure 2: A3's rows on seed 0, one lane each
    rows = list(A3_ROWS)
    fig, axs = plt.subplots(len(rows), 1, figsize=(7.2, 0.62 * len(rows) + 0.6), sharex=True)
    for ax, r in zip(axs, rows):
        lane, meta = k["a3.lanes"][r], k["a3.rows"][r]
        ax.plot(x, lane["y"], color=SERIES[2] if r == "E10" else SERIES[0], lw=1.4)
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
              ("s2.g3p", "G3′: step-128 OV maps"))
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
    return out


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="C2's numbers, draft and figures from their records")
    ap.add_argument("step", choices=("produce", "render", "check", "figures"))
    ap.add_argument("--reread", action="append", default=[],
                    help="a record id whose change has been read (produce only)")
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
