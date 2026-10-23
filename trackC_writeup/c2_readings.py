"""C2 unit 5b: the referee's readings from records already on disk (`design-C2.md` "Unit 5b",
Blocked 102 (a)). No forward pass, no download.

Reads unit 3's (seed 0) and unit 3s's (seeds 1–9) records, which hold `e1` and `cos` at the 25
states per step and prompt, and writes R1–R7 of the design's table to one record:

- R1 per-prompt V, each series dated by the half-rise on its own 1000–4000 plateau;
- R2 the layer × step map of drops (figure 5, left);
- R3 V without the two frame-changing transitions (t = 1, 24), dated per seed;
- R4 E₁'s net change, first state to last (and state 1 to 23 beside);
- R5 the half-rise on a later plateau (8000–32000);
- R6 the mean's term at each drop: E₁ = ½ + ½‖m‖² + R (R ≥ 0; `tools/math_checks/
  c2_energy_moments.py` M4), so ΔE₁ = ½Δ‖m‖² + ΔR; and V_R, the drops of R alone;
- R7 E₁ by depth at four steps (figure 5, right).

Refuses unless each seed's V from these records equals A5's at every step. Description only:
nothing registered, no label, no threshold; no date here replaces the registered break.

Run: METS_DATA=<main>/data python -m trackC_writeup.c2_readings
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1
from trackA_force.a5_seeds import REL_TOL, WINDOW
from trackC_writeup.c2_seeds import A5_SEEDS, C2U3, SeedsRefused, load_seed

RUN = DATA / "trackC" / "c2u3s" / "2026-10-22-run"
GATE0 = DATA / "trackA" / "a5" / "2026-10-20-u1" / "seed0" / "gate.json"
OUT = DATA / "trackC" / "c2u5b" / "2026-10-23"
SEEDS = tuple(range(10))
N_T = 24                                    # transitions t = 1..24 (state t − 1 → t)
INNER = slice(1, 23)                        # t = 2..23: neither frame-changing transition
LATE = (8000, 16000, 32000)                 # R5, the referee's 8k–32k
MAP_STEPS = (128, 256, 512, 1000)           # R2
SHARE_STEPS = (256, 512, 1000, 143000)      # R3
DEPTH_STEPS = (128, 512, 1000, 143000)      # R7


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_readings.py", "trackC_writeup/c2_seeds.py", "trackA_force/a3.py",
              "trackA_force/a1a.py", "trackA_force/a5_seeds.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def inputs_sha(d) -> str:
    """sha256 over one seed's record files, in name order (`/challenge-pr` on #261, finding 4)."""
    h = hashlib.sha256()
    for f in sorted(d.glob("step*_*.json")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- pure

def drops(e: np.ndarray) -> np.ndarray:
    """D2 (a): transition t is a drop when E₁ falls by more than REL_TOL of its value."""
    return (-np.diff(e) / np.maximum(np.abs(e[:-1]), 1e-12)) > REL_TOL


def m2(r: dict) -> np.ndarray:
    """‖m‖² at each state from the record's mean off-diagonal cosine (math check M5)."""
    return (1 + (r["n"] - 1) * np.asarray(r["cos"], float)) / r["n"]


def half_rise(v: Sequence[float], plateau_steps: Sequence[int]) -> dict:
    """D4 (a)'s half-rise with the plateau taken over ``plateau_steps``."""
    v = np.asarray(v, float)
    plateau = float(np.median([v[STEPS.index(s)] for s in plateau_steps]))
    if plateau <= 0:
        return {"reading": "no change", "plateau_median": plateau, "step": None}
    return {"reading": "located", "plateau_median": plateau, "step": STEPS[int(np.argmax(v >= plateau / 2))]}


def one_series(r: dict) -> dict:
    """Everything R1–R6 need from one (step, prompt) record."""
    e = np.asarray(r["e1"], float)
    m = m2(r)
    rest = e - 0.5 - 0.5 * m
    if rest.min() < -1e-9:
        raise SeedsRefused(f"R < 0 ({rest.min():.2e}) in {r['prompt']} at {r['step']}: E₁'s series broken")
    d = drops(e)
    de, dm, dr = np.diff(e), 0.5 * np.diff(m), np.diff(rest)
    return {"d": d, "net": float((e[-1] - e[0]) / e[0]), "net_inner": float((e[23] - e[1]) / e[1]),
            "m_fall": dm < 0, "s": np.where(d, dm / np.where(d, de, 1.0), np.nan),
            "r_drop": (-dr / np.maximum(np.abs(e[:-1]), 1e-12)) > REL_TOL, "e": e,
            # R6′, added after `/challenge-pr` on #261 (post hoc): the base rate over all moves
            "same": np.sign(de) == np.sign(dm), "e_fall": de < 0,
            "move": np.abs(de) / np.maximum(np.abs(e[:-1]), 1e-12) > REL_TOL,
            "s_all": dm / np.where(de == 0, np.nan, de)}


def a5_v(seed: int) -> List[float]:
    if seed == 0:
        return json.loads(GATE0.read_text())["known_answer"]["V"]
    return json.loads((DATA / A5_SEEDS / "score.json").read_text())["notes"][str(seed)]["V"]


def a5_break(seed: int) -> int:
    if seed == 0:
        return json.loads(GATE0.read_text())["known_answer"]["break"]["step"]
    return json.loads((DATA / A5_SEEDS / "score.json").read_text())["per_seed"][str(seed)]["step"]


def _share(num: float, den: float):
    return None if den == 0 else float(num / den)


def _inside(r: dict) -> bool:
    return r["reading"] == "located" and r["step"] in WINDOW


# ---------------------------------------------------------------- the readings

def main() -> int:
    per_seed: Dict[str, dict] = {}
    drop_map = np.zeros((N_T, len(STEPS)), int)
    pooled = {st: {"drops": 0, "frame": 0, "m_fall_at": 0, "m_fall_all": 0, "trans": 0, "s": [],
                   "same": 0, "e_fall": 0, "m_fall_e_fall": 0, "moves": 0, "s_gt_half_moves": 0,
                   "rises": 0, "s_gt_half_rises": 0,
                   "net": [], "net_inner": []} for st in STEPS}
    depth = {}
    for seed in SEEDS:
        recs = load_seed(DATA / C2U3 / "records", None) if seed == 0 else \
            load_seed(RUN / f"seed{seed}" / "records", seed)
        ser = {(st, p): one_series(recs[(st, p)]) for st in STEPS for p in V1}
        v = [int(sum(ser[(st, p)]["d"].sum() for p in V1)) for st in STEPS]
        if v != [int(x) for x in a5_v(seed)]:
            raise SeedsRefused(f"seed {seed}: V from these records {v} is not A5's {a5_v(seed)}")
        brk = a5_break(seed)
        vp = {p: [int(ser[(st, p)]["d"].sum()) for st in STEPS] for p in V1}
        v_in = [int(sum(ser[(st, p)]["d"][INNER].sum() for p in V1)) for st in STEPS]
        v_r = [int(sum(ser[(st, p)]["r_drop"].sum() for p in V1)) for st in STEPS]
        for j, st in enumerate(STEPS):
            for p in V1:
                x, q = ser[(st, p)], pooled[st]
                drop_map[:, j] += x["d"]
                q["drops"] += int(x["d"].sum())
                q["frame"] += int(x["d"][0] + x["d"][-1])
                q["m_fall_at"] += int((x["d"] & x["m_fall"]).sum())
                q["m_fall_all"] += int(x["m_fall"].sum())
                q["trans"] += N_T
                q["s"] += x["s"][x["d"]].tolist()
                big = x["s_all"] > 0.5
                rise = x["move"] & ~x["d"]
                q["same"] += int(x["same"].sum())
                q["e_fall"] += int(x["e_fall"].sum())
                q["m_fall_e_fall"] += int((x["e_fall"] & x["m_fall"]).sum())
                q["moves"] += int(x["move"].sum())
                q["s_gt_half_moves"] += int((x["move"] & big).sum())
                q["rises"] += int(rise.sum())
                q["s_gt_half_rises"] += int((rise & big).sum())
                q["net"].append(x["net"])
                q["net_inner"].append(x["net_inner"])
        at = [ser[(brk, p)] for p in V1]
        n_at = sum(int(x["d"].sum()) for x in at)
        s_at = [s for x in at for s in x["s"][x["d"]]]
        per_seed[str(seed)] = {
            "break": brk, "V": v,
            "R1": {"V_p": vp, "dates": {p: a3.d4a(np.array(vp[p], float))["step"] for p in V1},
                   "max_prompt_share_at_break": _share(max(vp[p][STEPS.index(brk)] for p in V1),
                                                       v[STEPS.index(brk)])},
            "R3": {"V_in": v_in, "date": a3.d4a(np.array(v_in, float))},
            "R5": half_rise(v, LATE),
            "R6": {"drops_at_break": n_at,
                   "m_fall_share_at_break": _share(sum(int((x["d"] & x["m_fall"]).sum()) for x in at), n_at),
                   "s_gt_half_at_break": _share(sum(s > 0.5 for s in s_at), n_at),
                   "V_R": v_r, "V_R_date": half_rise(v_r, (1000, 2000, 4000))}}
        depth[str(seed)] = {str(st): np.mean([ser[(st, p)]["e"] for p in V1], axis=0).tolist()
                            for st in DEPTH_STEPS}
    r1_dates = [d for s in per_seed.values() for d in s["R1"]["dates"].values()]
    by_step = {}
    for st in STEPS:
        q = pooled[st]
        s = np.asarray(q["s"], float)
        by_step[str(st)] = {
            "drops": q["drops"], "frame_share": _share(q["frame"], q["drops"]),
            "m_fall_share_at_drops": _share(q["m_fall_at"], q["drops"]),
            "m_fall_share_all": _share(q["m_fall_all"], q["trans"]),
            "s_gt_half_share": _share(int((s > 0.5).sum()), q["drops"]),
            "s_median": (float(np.median(s)) if len(s) else None),
            "net_positive": int(sum(x > 0 for x in q["net"])), "net_median": float(np.median(q["net"])),
            "net_inner_positive": int(sum(x > 0 for x in q["net_inner"])),
            "net_inner_median": float(np.median(q["net_inner"])), "series": len(q["net"]),
            "same_direction_share": _share(q["same"], q["trans"]),
            "m_fall_share_at_any_e_fall": _share(q["m_fall_e_fall"], q["e_fall"]),
            "moves": q["moves"], "s_gt_half_share_moves": _share(q["s_gt_half_moves"], q["moves"]),
            "rises": q["rises"], "s_gt_half_share_rises": _share(q["s_gt_half_rises"], q["rises"])}
    map_read = {}
    for st in MAP_STEPS:
        col = drop_map[:, STEPS.index(st)]
        ts = np.repeat(np.arange(1, N_T + 1), col)
        top = sorted(range(N_T), key=lambda i: (-col[i], i))[:3]
        map_read[str(st)] = {"median_t": (float(np.median(ts)) if len(ts) else None),
                             "top3": [{"t": i + 1, "drops": int(col[i])} for i in top]}
    rec = {
        "code": code_sha(), "design": "trackC_writeup/design-C2.md \"Unit 5b\"",
        "inputs": {"seed0": str(C2U3) + "/records", "seeds1_9": "trackC/c2u3s/2026-10-22-run/seed<s>/records",
                   "V_checked_against": [str(GATE0.relative_to(DATA)), A5_SEEDS + "/score.json"],
                   "sha256": {str(s): inputs_sha(DATA / C2U3 / "records" if s == 0 else RUN / f"seed{s}" / "records")
                              for s in SEEDS}},
        "post_hoc": "R6's same_direction, any-fall and moves/rises shares were added after `/challenge-pr` on #261 "
                    "(finding 1), after R6's first output; description",
        "steps": list(STEPS), "prompts": list(V1), "window": list(WINDOW), "rel_tol": REL_TOL,
        "frame_t": [1, N_T], "late_plateau": list(LATE),
        "R1": {"series": len(r1_dates), "inside": sum(d in WINDOW for d in r1_dates),
               "no_change": sum(d is None for d in r1_dates),
               "dates": {str(d): r1_dates.count(d) for d in sorted({d for d in r1_dates if d is not None})},
               "max_prompt_share_at_break": [per_seed[str(s)]["R1"]["max_prompt_share_at_break"] for s in SEEDS]},
        "R2": {"map": drop_map.tolist(), "max_cell": len(SEEDS) * len(V1), "at": map_read},
        "R3": {"dates": [per_seed[str(s)]["R3"]["date"]["step"] for s in SEEDS],
               "inside": sum(_inside(per_seed[str(s)]["R3"]["date"]) for s in SEEDS),
               "frame_share": {str(st): by_step[str(st)]["frame_share"] for st in SHARE_STEPS}},
        "R4": {str(st): {k: by_step[str(st)][k] for k in ("net_positive", "net_median", "net_inner_positive",
                                                          "net_inner_median", "series")} for st in STEPS},
        "R5": {"dates": [per_seed[str(s)]["R5"]["step"] for s in SEEDS],
               "plateaus": [per_seed[str(s)]["R5"]["plateau_median"] for s in SEEDS],
               "inside": sum(_inside(per_seed[str(s)]["R5"]) for s in SEEDS)},
        "R6": {"by_step": {str(st): {k: by_step[str(st)][k] for k in
                                     ("drops", "m_fall_share_at_drops", "m_fall_share_all", "s_gt_half_share",
                                      "s_median", "same_direction_share", "m_fall_share_at_any_e_fall",
                                      "moves", "s_gt_half_share_moves", "rises", "s_gt_half_share_rises")}
                           for st in STEPS},
               "at_break": {str(s): {k: per_seed[str(s)]["R6"][k] for k in
                                     ("drops_at_break", "m_fall_share_at_break", "s_gt_half_at_break")}
                            for s in SEEDS},
               "V_R_dates": [per_seed[str(s)]["R6"]["V_R_date"]["step"] for s in SEEDS],
               "V_R_inside": sum(_inside(per_seed[str(s)]["R6"]["V_R_date"]) for s in SEEDS)},
        "R7": {"steps": list(DEPTH_STEPS), "e1_prompt_mean": depth},
        "per_seed": per_seed}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readings.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("R1", {k: rec["R1"][k] for k in ("series", "inside", "no_change", "dates")},
          "max share", [round(x, 2) for x in rec["R1"]["max_prompt_share_at_break"]])
    print("R2", rec["R2"]["at"])
    print("R3", rec["R3"])
    print("R4", {st: rec["R4"][str(st)] for st in (0, 128, 256, 512, 1000, 143000)})
    print("R5", rec["R5"])
    print("R6", {st: rec["R6"]["by_step"][str(st)] for st in (128, 256, 512, 1000, 143000)})
    print("R6 at break", rec["R6"]["at_break"])
    print("R6 V_R", rec["R6"]["V_R_dates"], rec["R6"]["V_R_inside"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
