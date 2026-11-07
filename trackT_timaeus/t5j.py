"""
trackT_timaeus/t5j.py — T5 U4's beside stage at 4369: on each arm, T2's counts (i)–(iii), U3's split,
and the sink / massive-activation onset (`trackT_timaeus/design-T5.md` "U4's beside stage at 4369",
Blocked 134 (a), fixed 2026-11-06 before any output). Description, no label: U4's labels are final.

- counts: `t2.py preflight` unchanged at trained step 4369, its trained population loaded from the
  arm's checkpoint (`t2.load` swapped in-process, so neither `t2.py`'s hash nor `t5h`'s moves);
  T2's re-inits, plants, `u` seed and doses, as U3. Read with `t5d` / `t5e`'s helpers.
- split: `t5f.read_model` on the arm at 4369.
- onsets: per checkpoint 0 … 4369, Gu et al.'s Sink^0.3 to the first token, Sun et al.'s massive-
  activation statistics on the block outputs 1 … L − 1, and n0 (checked against the read records).

Gates (refuse rather than degrade): `known` runs the counts' code path (released 4369 through
`t2.load`, at 1e-12) and load path (`t5u4/known/step4369.safetensors` through the swapped loader, at
1e-6) against U3's record, and the split's load path against U3's batch record (1e-6); `counts` and
`split` refuse until `known.json` passes; `onset` refuses a step whose n0 differs from the read
record by more than 1e-6.

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5j known
    ... python -m trackT_timaeus.t5j counts --arm A1     (split, onset likewise)
    ... python -m trackT_timaeus.t5j read
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1, prompts_hash

STEP = 4369
ARMS = ("A0", "A0p", "A0pp", "A1", "A2", "A3")
A0S = ("A0", "A0p", "A0pp")
PROMPTS_HASH = "976b62d677bc"
# the onsets (design "U4's beside stage at 4369"; fixed before output)
SINK_EPS = 0.3                          # Gu et al. 2410.10781's ε
MASSIVE_MAG, MASSIVE_RATIO = 100.0, 1000.0   # Sun et al. 2402.17762's criterion
# the gates
KNOWN_TOL_CODE, KNOWN_TOL_LOAD = 1e-12, 1e-6
SPLIT_TOL, N0_TOL = 1e-6, 1e-6
# the reading's cells (as U3's first read)
STOP_TAU, STOP_DOSE = "0.003", "0.5"
OUT = Path("/run/media/system/HDD_1TB/mets_data/trackT/t5u4/beside")
U3_PRE = "trackT/t5u3/2026-11-02"
U3_SPLIT = "trackT/t5u3s/2026-11-02-batch/32m/step4369.json"


class T5jRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5j.py", "trackT_timaeus/t2.py", "trackT_timaeus/t5h.py", "trackT_timaeus/t5f.py",
              "trackT_timaeus/t5.py", "trackT_timaeus/t5c.py", "trackT_timaeus/t5d.py", "trackT_timaeus/t5e.py",
              "trackC_writeup/c2_aniso.py", "trackC_writeup/c2_seeds.py", "trackC_writeup/c2_dose.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the onsets

def sink_metric(attn: Sequence[np.ndarray], eps: float = SINK_EPS) -> float:
    """Gu et al.'s Sink^ε to the first token: ``attn`` per block (H, n, n), rows the queries; the
    share of (block, head) whose attention to position 0, averaged over every query, exceeds ε."""
    hits = [np.asarray(a, np.float64)[:, :, 0].mean(1) > eps for a in attn]
    return float(np.mean(np.concatenate(hits)))


def massive_one(h: np.ndarray) -> dict:
    """One block output (n, d): the largest |coordinate|, its ratio to the median |coordinate|, and
    where it sits."""
    a = np.abs(np.asarray(h, np.float64))
    i, j = np.unravel_index(int(np.argmax(a)), a.shape)
    top, med = float(a[i, j]), float(np.median(a))
    return {"top": top, "ratio": top / med if med > 0 else float("inf"), "pos": int(i), "dim": int(j)}


def massive_step(per_prompt: Sequence[Sequence[dict]]) -> dict:
    """``per_prompt[p][ℓ]``: `massive_one` of block output ℓ + 1. Per ℓ the median over prompts of
    the magnitude and the ratio; M the largest ratio over ℓ, with that ℓ's magnitude; Sun et al.'s
    criterion; beside, the share of (prompt, ℓ) whose top sits at position 0 and the commonest dim."""
    nl = len(per_prompt[0])
    top = [float(np.median([p[l]["top"] for p in per_prompt])) for l in range(nl)]
    ratio = [float(np.median([p[l]["ratio"] for p in per_prompt])) for l in range(nl)]
    k = int(np.argmax(ratio))
    cells = [c for p in per_prompt for c in p]
    dim, n_dim = Counter(c["dim"] for c in cells).most_common(1)[0]
    return {"top_per_block": top, "ratio_per_block": ratio, "M": ratio[k], "M_block": k + 1, "M_top": top[k],
            "criterion": any(t >= MASSIVE_MAG and r >= MASSIVE_RATIO for t, r in zip(top, ratio)),
            "pos0_share": float(np.mean([c["pos"] == 0 for c in cells])),
            "dim_mode": int(dim), "dim_mode_share": n_dim / len(cells)}


def half_rise(series: Dict[int, float], base: Optional[int] = None) -> dict:
    """The first grid step s (after ``base``, default the first step) from which x ≥ x(base) +
    ½ (x(last) − x(base)) at s and every later step; "no rise" if x(last) ≤ x(base)."""
    steps = sorted(series)
    b = steps[0] if base is None else base
    later = [s for s in steps if s > b]
    x0, x1 = series[b], series[steps[-1]]
    if not later or x1 <= x0:
        return {"step": None, "reads": "no rise", "from": b, "x_base": x0, "x_last": x1}
    thr = x0 + 0.5 * (x1 - x0)
    first = None
    for s in reversed(later):
        if series[s] < thr:
            break
        first = s
    return {"step": first, "reads": f"half-rise at {first}", "from": b, "x_base": x0, "x_last": x1, "threshold": thr}


def held_from(flags: Dict[int, bool]) -> Optional[int]:
    """The first step from which ``flags`` holds at it and every later step (None if the last fails)."""
    first = None
    for s in sorted(flags, reverse=True):
        if not flags[s]:
            break
        first = s
    return first


def outcome_row(passes: Dict[str, Optional[bool]]) -> str:
    """The design's table, per count: any of A0, A0′, A0″ passes → row 3; else A1 passes → row 1;
    else row 2. A refused step (None) counts as not passing, and the row names it (T5's Parked
    item, carried to T5′ U1: a refused run is not silently a fail)."""
    refused = [a for a in ARMS if a in passes and passes[a] is None]
    tail = f" (refused, counted as not passing: {', '.join(refused)})" if refused else ""
    if any(passes.get(a) for a in A0S):
        return "an A0 seed passes: the count passes at A0's ρ in this recipe, unlike Dodecahedron at 0.81; ρ alone is not its stop" + tail
    if passes.get("A1"):
        return "A1 passes, no A0 seed does: the stop goes with ρ (or with what biases change), not with step or loss" + tail
    return "neither A1 nor an A0 seed passes: not ρ alone at 0.66; step, loss or something the arms share" + tail


# ---------------------------------------------------------------- models

def _setup():
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != PROMPTS_HASH:
        raise T5jRefused(f"prompts_hash {prompts_hash()} != {PROMPTS_HASH}")
    from trackC_writeup import c2_aniso as c
    return c._torch_setup()


def ckpt(arm: str, s: int = STEP) -> Path:
    from trackT_timaeus import t5h
    return t5h.ROOT / arm / "ckpt" / f"step{s}.safetensors"


def known_ckpt() -> Path:
    from trackT_timaeus import t5h
    return t5h.ROOT / "known" / f"step{STEP}.safetensors"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load(path: Path, arm: str, device):
    from trackT_timaeus import t5, t5h
    return t5h.load_weights(path, arm, t5h.dodeca_cfg(t5.DEFAULT_ROOTS), device)


@contextmanager
def trained_from(path: Optional[Path], arm: str):
    """`t2.load` answers ``step4369`` with the checkpoint at ``path`` (None: unchanged)."""
    from trackT_timaeus import t2
    orig = t2.load

    def load(repo, revision, device):
        if revision == f"step{STEP}":
            return _load(path, arm, device)
        return orig(repo, revision, device)
    if path is not None:
        t2.load = load
    try:
        yield
    finally:
        t2.load = orig


def _jdump(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(p)


def run_preflight(out: Path, path: Optional[Path], arm: str) -> dict:
    """T2's preflight at 4369 into ``out/32m-step4369/``; a refusal is written beside, not raised."""
    from trackC_writeup import c2_dose as d, c2_seeds as cs
    from trackT_timaeus import t2
    try:
        with trained_from(path, arm):
            rc = t2.preflight(SimpleNamespace(model="32m", out=str(out), trained_step=STEP))
        res = {"rc": rc}
    except (t2.T2Refused, d.DoseRefused, cs.SeedsRefused) as e:
        res = {"refused": f"{type(e).__name__}: {e}"}
    res.update({"code": code_sha(), "arm": arm, "ckpt": None if path is None else str(path),
                "ckpt_sha256": None if path is None else _sha(path)})
    _jdump(out / "source.json", res)
    return res


def _u3_record() -> dict:
    return json.loads((DATA / U3_PRE / f"32m-step{STEP}" / "preflight_plants.json").read_text())


def _split_diff(new: dict, ref: dict) -> float:
    vals = [abs(new["share"][k] - ref["share"][k]) for k in ref["share"]]
    vals += [abs(new["size"][k] - ref["size"][k]) for k in ref["size"]]
    vals += [abs(a - b) for a, b in zip(new["rho"], ref["rho"])] + [abs(new["size_write"] - ref["size_write"])]
    return max(vals)


def known(a) -> int:
    from trackT_timaeus import t2, t5d, t5f
    torch, device = _setup()
    out = Path(a.out) / "known"
    ref = _u3_record()
    res = {"code": code_sha(), "design": "trackT_timaeus/design-T5.md U4's beside stage at 4369"}
    run_preflight(out / "code_path", None, "A0")
    new = json.loads((out / "code_path" / f"32m-step{STEP}" / "preflight_plants.json").read_text())
    res["counts_code_path"] = t5d.known_answer(new, ref, KNOWN_TOL_CODE)
    run_preflight(out / "load_path", known_ckpt(), "A0")
    new = json.loads((out / "load_path" / f"32m-step{STEP}" / "preflight_plants.json").read_text())
    res["counts_load_path"] = t5d.known_answer(new, ref, KNOWN_TOL_LOAD)
    m = _load(known_ckpt(), "A0", device)
    sp = t5f.read_model(m, t2.tokenizer("32m"), device)
    diff = _split_diff(sp, json.loads((DATA / U3_SPLIT).read_text()))
    res["split_load_path"] = {"max_abs_diff": diff, "err_sum": sp["err_sum"],
                              "pass": diff <= SPLIT_TOL and sp["err_sum"] <= t5f.SUM_TOL}
    res["pass"] = all(res[k]["pass"] for k in ("counts_code_path", "counts_load_path", "split_load_path"))
    _jdump(out / "known.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "design"}, indent=1, default=str))
    return 0 if res["pass"] else 1


def _require_known(out: Path) -> None:
    p = out / "known" / "known.json"
    if not p.exists() or not json.loads(p.read_text()).get("pass"):
        raise T5jRefused(f"{p}: the known answers have not passed")
    if json.loads(p.read_text())["code"] != code_sha():
        raise T5jRefused(f"{p}: written at another code than {code_sha()}")


def counts(a) -> int:
    _setup()
    out = Path(a.out)
    _require_known(out)
    r = run_preflight(out / a.arm, ckpt(a.arm), a.arm)
    print(json.dumps(r))
    return 0


def split(a) -> int:
    from trackT_timaeus import t2, t5f
    torch, device = _setup()
    out = Path(a.out)
    _require_known(out)
    p = ckpt(a.arm)
    rec = t5f.read_model(_load(p, a.arm, device), t2.tokenizer("32m"), device)
    if rec["err_sum"] > t5f.SUM_TOL:
        raise T5jRefused(f"{a.arm}: err_sum {rec['err_sum']} > {t5f.SUM_TOL}")
    rec.update({"arm": a.arm, "step": STEP, "code": code_sha(), "ckpt_sha256": _sha(p)})
    _jdump(out / a.arm / f"split{STEP}.json", rec)
    print(json.dumps({"arm": a.arm, "share": rec["share"], "err_sum": rec["err_sum"]}))
    return 0


def onset_forward(model, tok, device) -> dict:
    """One checkpoint over the 7 prompts: S, the massive statistics, n0 per state (mean over prompts)."""
    import torch
    from core.config import PROMPTS
    from trackC_writeup import c2_aniso as c
    from trackT_timaeus import t5
    sinks, mass, n0, a0 = [], [], [], []
    for p in V1:
        enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=c.MAX_TOKENS)
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True, output_attentions=True)
        att = [x[0].to(torch.float64).cpu().numpy() for x in o.attentions]
        sinks.append(sink_metric(att))
        a0.append([x[:, :, 0].mean(1).tolist() for x in att])
        hs = [h[0].to(torch.float32).cpu().numpy() for h in o.hidden_states]
        n0.append([t5.state_stats(h)["n0"] for h in hs])
        mass.append([massive_one(h) for h in hs[1:-1]])
    return {"S": float(np.mean(sinks)), "S_per_prompt": sinks, "a0_mean": np.mean(a0, 0).tolist(),
            "massive": massive_step(mass), "n0": np.mean(n0, 0).tolist()}


def onset(a) -> int:
    from trackT_timaeus import t2, t5h
    torch, device = _setup()
    out = Path(a.out) / a.arm / "onset"
    tok = t2.tokenizer("32m")
    t0 = time.time()
    for s in t5h.CKPT_STEPS:
        p = out / f"step{s}.json"
        if p.exists() and not a.force:
            continue
        ck = ckpt(a.arm, s)
        rec = onset_forward(_load(ck, a.arm, device), tok, device)
        read = json.loads((t5h.ROOT / a.arm / "read" / f"step{s}.json").read_text())
        dn0 = float(np.max(np.abs(np.asarray(rec["n0"]) - np.asarray(read["n0"]))))
        if dn0 > N0_TOL:
            raise T5jRefused(f"{a.arm} step {s}: n0 differs from the read record by {dn0:.3g}")
        rec.update({"arm": a.arm, "step": s, "n0_vs_read": dn0, "code": code_sha(), "ckpt_sha256": _sha(ck)})
        _jdump(p, rec)
        print(json.dumps({"onset": f"{a.arm}/step{s}", "S": round(rec["S"], 3), "M": round(rec["massive"]["M"], 1),
                          "dn0": dn0, "t": round(time.time() - t0)}), flush=True)
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the reading

def counts_reading(pre: Path) -> dict:
    """(i)–(iii) at 4369 from one preflight directory (``pre/32m-step4369``)."""
    from trackT_timaeus import t5d, t5e
    d = pre / f"32m-step{STEP}"
    units = json.loads((d / "preflight_units.json").read_text())
    if (d / "preflight_plants.json").exists():
        rec = json.loads((d / "preflight_plants.json").read_text())
        if int(rec["trained_step"]) != STEP or not rec["checks_pass"]:
            raise T5jRefused(f"{d}: trained_step {rec['trained_step']}, checks_pass {rec['checks_pass']}")
        centred = t5d.step_reading(rec)
        centred["wide_post_hoc"] = t5e.wide_b_matched(units)
        i_pass: Optional[bool] = centred["any_trained_part"]
    else:
        src = json.loads((pre / "source.json").read_text()) if (pre / "source.json").exists() else {}
        centred = {"refused": src.get("refused", "no record"), "wide_post_hoc": t5e.wide_b_matched(units)}
        i_pass = None
    ft = t5e.fixed_tau(units)
    a_u = t5e.uncentred_a(units)
    return {"centred": centred, "fixed_tau": ft, "uncentred_A_share": a_u,
            "i": i_pass, "ii": ft[STOP_DOSE][STOP_TAU]["A_share"] <= t5d.cs.A_LEAK,
            "ii_A_share": ft[STOP_DOSE][STOP_TAU]["A_share"], "iii": a_u >= t5d.UNC_PYTHIA,
            "iii_reads": "as on Pythia" if a_u >= t5d.UNC_PYTHIA else "as at 9536" if a_u <= t5d.UNC_AS_9536 else "between"}


def onset_reading(arm: str, out: Path) -> dict:
    from trackT_timaeus import t5h
    recs = {s: json.loads((out / arm / "onset" / f"step{s}.json").read_text()) for s in t5h.CKPT_STEPS}
    reads = t5h.records(t5h.ROOT, arm)
    S = {s: r["S"] for s, r in recs.items()}
    M = {s: r["massive"]["M"] for s, r in recs.items()}
    n0 = {s: float(np.median(reads[s]["n0"][1:-1])) for s in t5h.CKPT_STEPS}
    rho = {s: reads[s]["rho_mid"] for s in t5h.CKPT_STEPS}
    T, Ts = t5h.trough(rho)
    last = recs[STEP]["massive"]
    return {"S": half_rise(S), "M": half_rise(M), "n0_mid": half_rise(n0), "rho_mid_from_trough": half_rise(rho, Ts),
            "sun_criterion_from": held_from({s: r["massive"]["criterion"] for s, r in recs.items()}),
            "at_4369": {"S": S[STEP], "M": M[STEP], "M_top": last["M_top"], "M_block": last["M_block"],
                        "pos0_share": last["pos0_share"], "dim_mode": last["dim_mode"],
                        "dim_mode_share": last["dim_mode_share"], "n0_mid": n0[STEP], "rho_mid": rho[STEP],
                        "v1_loss": reads[STEP]["v1_loss_mean"]},
            "trough": {"T": T, "step": Ts},
            "series": {"step": list(t5h.CKPT_STEPS), "S": [S[s] for s in t5h.CKPT_STEPS],
                       "M": [M[s] for s in t5h.CKPT_STEPS], "M_top": [recs[s]["massive"]["M_top"] for s in t5h.CKPT_STEPS],
                       "n0_mid": [n0[s] for s in t5h.CKPT_STEPS], "rho_mid": [rho[s] for s in t5h.CKPT_STEPS]},
            "codes": sorted({r["code"] for r in recs.values()}),
            "n0_vs_read_max": max(r["n0_vs_read"] for r in recs.values())}


def read(a) -> int:
    from trackT_timaeus import t5f
    out = Path(a.out)
    _require_known(out)
    res = {"design": "trackT_timaeus/design-T5.md U4's beside stage at 4369", "code": code_sha(),
           "gate": json.loads((out / "known" / "known.json").read_text()), "arms": {}}
    res["dodecahedron"] = {"4369": counts_reading(out / "known" / "code_path")}   # equals U3's record (the gate)
    sp_ref = json.loads((DATA / U3_SPLIT).read_text())
    res["dodecahedron"]["4369"]["split_share"] = sp_ref["share"]
    for arm in ARMS:
        sp = json.loads((out / arm / f"split{STEP}.json").read_text())
        res["arms"][arm] = {"counts": counts_reading(out / arm),
                            "split": {"share": sp["share"], "share_no0": sp["share_no0"], "size": sp["size"],
                                      "size_write": sp["size_write"], "token_constancy": sp["token_constancy"],
                                      "bias_norm_max": sp["bias_norm_max"], "err_sum": sp["err_sum"]},
                            "onset": onset_reading(arm, out)}
    res["outcome"] = {k: outcome_row({arm: res["arms"][arm]["counts"][k] for arm in ARMS}) for k in ("i", "ii", "iii")}
    res["sum_gate"] = {"max_err_sum": max(res["arms"][x]["split"]["err_sum"] for x in ARMS), "bar": t5f.SUM_TOL}
    _jdump(out / "reading.json", res)
    _print(res)
    return 0


def _print(res: dict) -> None:
    print("arm   rho   v1    (i)P   (ii)A@3e-3  (iii)A_u   S     M      n0   | S½   M½   n0½  rho½  Sun")
    for arm, r in res["arms"].items():
        c, o = r["counts"], r["onset"]
        z = o["at_4369"]
        print(f"{arm:5} {z['rho_mid']:.3f} {z['v1_loss']:.3f} {str(c['i']):6} {c['ii_A_share']:.3f}       "
              f"{c['uncentred_A_share']:.3f}     {z['S']:.2f} {z['M']:7.1f} {z['n0_mid']:5.1f} | "
              f"{o['S']['step']} {o['M']['step']} {o['n0_mid']['step']} {o['rho_mid_from_trough']['step']} "
              f"{o['sun_criterion_from']}")
    for k, v in res["outcome"].items():
        print(f"({k}) {v}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="t5j")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("known", "counts", "split", "onset", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", default=str(OUT))
        if cmd in ("counts", "split", "onset"):
            sp.add_argument("--arm", required=True, choices=ARMS)
        if cmd == "onset":
            sp.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"known": known, "counts": counts, "split": split, "onset": onset, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
