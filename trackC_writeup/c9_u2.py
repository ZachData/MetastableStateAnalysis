"""C9 unit 2a: A3's locator and gate parametrised by grid, the weights rows on every U2 run, their
gates, and the known answer (`design-C9.md` "U2" and "U2a as it runs"; Blocked 178 D1–D4, D6 (a),
confirmed by the user 2026-11-25). Tier 1, no forward pass, nothing located off 410m.

- **Grids.** 410m seed 0: Pythia's 18 (`a1a.STEPS`), window ≤ 4000. GPT-2 small × 5: C2 3c's 35
  tags (`c2_warmup.GRID`), window ≤ 16000 (8.4 B tokens, 410m's 4000). Harness arms A0, A0′:
  C4's 28-step `GRID`, window ≤ 4000. x = log₂ step, step 0 two octaves below the first nonzero
  step (A3's x(0) = −1 on Pythia's grid).
- **The locator and gate** are `a3.py`'s, with the grid an argument; the only change in rule is
  the plant's width, a quarter of its gap in octaves (A3's 0.25 on every one-octave gap). A turn
  row's dip needs a step past its rise, so its last gap is planted only where the grid runs past
  the window.
- **The weights rows** (`weights` command, one run a call, a record per step, resumable):
  E2 the attractive energy fraction of each head's OV core (`head_circuits.head_spectrum`'s
  ``attractive_energy_fraction_core``: Σ|λ|² over Re λ > 0 over Σ|λ|², λ the eigenvalues of the
  d_head × d_head core), median over heads; E7 ‖W − W₀‖_F / ‖W₀‖_F and E8 the ESD tail α
  (`a3_weights`' fit) per layer and matrix type (QKV, O, MLP in, MLP out), median over types of
  the median over layers. GPT-2's Conv1D weights are the transposes of Linear ones; Frobenius
  norms and the smaller Gram's eigenvalues do not see the transpose.
- **The gates** (`gate` command): every weights row on every run, at its own units and grid;
  nothing real is located off 410m (that is U2b's reading).
- **The known answer** (`known` command): (i) the generalised gate and locator on A3's own units,
  in A3's order and seed, give every date and floor of A3 unit 5's record; (ii) the weights
  producer on 410m seed 0 gives `a3_weights`' E7, E8 and `ov_per_head_series.json`'s E2, and
  swapped into (i) moves no date.

CPU throughout: the gate is numpy so that (i) can be compared draw for draw with `a3.py`; E8's fit
is a Python loop over tail lengths (`a3_weights.powerlaw_alpha`), which a GPU does not speed up.

    HF_HOME=<HDD_1TB>/mets_data/hf HF_HUB_OFFLINE=1 python -m trackC_writeup.c9_u2 weights --run alias
    HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 python -m trackC_writeup.c9_u2 weights --run 410m
    METS_DATA=<main>/data python -m trackC_writeup.c9_u2 known
    METS_DATA=<main>/data python -m trackC_writeup.c9_u2 gate --run alias
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS
from trackA_force.a3 import (A3Refused, AMPS, GATE_EFFECT, GATE_NULL, LOW_POWER, N_BOOT, N_GATE,
                             SIGN, STATS, FLOOR, boot_se, constant_row)

TC = DATA / "trackC"
OUT = TC / "c9" / "2026-11-26-u2a"
A3U5 = DATA / "trackA" / "a3" / "2026-10-20-A3u5"
A3W = DATA / "trackA" / "a3" / "2026-10-19-A3u3-weights"
HDD = Path("/run/media/system/HDD_1TB/mets_data")
WIDTH = 0.25                     # a quarter of the gap, in octaves (A3's 0.25 on one-octave gaps)
SEED_U2 = 20261126
TYPES = ("qkv", "o", "mlp_in", "mlp_out")
GPT2_GRID = (0, 10, 20, 40, 60, 80, 100, 150, 200, 250, 350, 400, 450, 500, 550, 600, 700, 800,
             900, 1000, 1100, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3600, 4000, 6000,
             8000, 12000, 16000)                                   # c2_warmup.GRID
HARNESS_GRID = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
                640, 768, 1000, 1250, 1500, 2000, 3000, 4000)      # c4_train.GRID
GPT2_SEEDS = ("alias", "battlestar", "caprica", "darkmatter", "expanse")
RUNS = ("410m",) + GPT2_SEEDS + ("A0", "A0p")                       # the gate's rng index
WROWS = {  # id: (direction, statistic)
    "E2": ("down", "median"), "E7": ("up", "medmed"), "E8": ("down", "medmed")}
E2_TOL = 1e-6
E78_RTOL = 1e-9        # set after the first 410m record: α differs from a3_weights' by 1e-13
                       # relative (BLAS threads, not the code); placed before the known answer ran


class C9Refused(RuntimeError):
    pass


# ---------------------------------------------------------------- grids

@dataclass(frozen=True)
class Grid:
    name: str
    steps: Tuple[int, ...]
    last: int                       # the window's last step

    @property
    def win(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.steps) if s <= self.last)

    @property
    def X(self) -> np.ndarray:
        x = np.log2(np.maximum(np.asarray(self.steps, float), 1.0))
        if self.steps[0] != 0 or self.steps[1] <= 0:
            raise C9Refused(f"{self.name}: a grid starts 0, then a positive step")
        x[0] = x[1] - 2.0
        return x


PYTHIA = Grid("pythia", tuple(STEPS), 4000)
GPT2 = Grid("gpt2", GPT2_GRID, 16000)
HARNESS = Grid("harness", HARNESS_GRID, 4000)


def grid_of(run: str) -> Grid:
    return PYTHIA if run == "410m" else GPT2 if run in GPT2_SEEDS else HARNESS


# ---------------------------------------------------------------- the locator, by grid (pure)

def reference(g: Grid, x, direction: str, kind: str = "half") -> int:
    w = np.asarray(x, float)[list(g.win)]
    if kind == "half":
        d = SIGN[direction] * (w - w[0])
        return int(np.argmax(d[1:])) + 1
    if kind == "turn":
        return int(np.argmin(w[1:])) + 1
    raise C9Refused(f"unknown locator kind {kind!r}")


def place(g: Grid, x, direction: str, sigma: Optional[float], kind: str = "half",
          floor: Optional[float] = FLOOR) -> dict:
    w = np.asarray(x, float)[list(g.win)]
    j = reference(g, x, direction, kind)
    d = SIGN[direction] * (w - w[0]) if kind == "half" else w[0] - w
    ref = float(d[j])
    out = {"ref_index": j, "ref_step": g.steps[j], "ref": ref, "sigma": sigma,
           "ref_over_sigma": (ref / sigma if sigma else None)}
    floored = floor is not None and sigma is not None
    if ref <= 0 or (floored and ref < floor * sigma):
        out.update(reading="no change", index=None, step=None, low_power=None)
        return out
    i = j if kind == "turn" else int(np.argmax(d[1:] >= ref / 2)) + 1
    out.update(reading="located", index=i, step=g.steps[i],
               low_power=(bool(ref < LOW_POWER * sigma) if floored else None))
    return out


def locate_units(g: Grid, units, stat: Callable, direction: str, rng, kind="half",
                 n_boot: int = N_BOOT, floor: Optional[float] = FLOOR) -> dict:
    x = stat(units)
    j = reference(g, x, direction, kind)
    return place(g, x, direction, boot_se(units, stat, j, rng, n_boot), kind, floor=floor)


def censored(g: Grid, r: dict) -> Optional[bool]:
    if r.get("reading") != "located":
        return None
    return r["ref_index"] == g.win[-1]


def boot_se_all(g: Grid, units, stat: Callable, rng, n_boot: int = N_BOOT) -> np.ndarray:
    n = units.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    s = stat(units[:, list(g.win), :][idx])
    return np.std(s - s[:, :1], axis=0, ddof=1)


def noise_draws(g: Grid, units, direction: str, sname: str, rng, n: int):
    stat = STATS[sname]
    syn = a3._synthetic(units, direction, sname, rng, n)
    xs = np.stack([stat(u) for u in syn])
    sig = np.stack([boot_se_all(g, u, stat, rng) for u in syn])
    return xs, sig


def _ratio(g: Grid, x, sig, direction: str, kind: str = "half") -> float:
    w = np.asarray(x, float)[list(g.win)]
    j = reference(g, x, direction, kind)
    ref = SIGN[direction] * (w[j] - w[0]) if kind == "half" else w[0] - w[j]
    return float(ref / sig[j]) if ref > 0 else -np.inf


def shape(g: Grid, i: int) -> np.ndarray:
    X = g.X
    centre = (X[i] + X[i + 1]) / 2
    sh = 1 / (1 + np.exp(-(X - centre) / (WIDTH * (X[i + 1] - X[i]))))
    return sh - sh[0]


def dip(g: Grid, i: int) -> np.ndarray:
    return -shape(g, i) + 0.5 * shape(g, i + 1)


def gaps(g: Grid, kind: str) -> Tuple[int, ...]:
    """The planted gaps' left indices: A3's WIN[1:-1]; a turn row's needs a step past i + 1."""
    w = g.win[1:-1]
    return tuple(i for i in w if kind != "turn" or i + 2 < len(g.steps))


def gate_row(g: Grid, direction: str, sname: str, kind: str, units, rng) -> dict:
    """`a3.gate_row` with the grid an argument (and the plant's width a quarter of its gap)."""
    xd, sd = noise_draws(g, units, direction, sname, rng, N_GATE)
    xc, sc = noise_draws(g, units, direction, sname, rng, N_GATE)
    r = np.array([_ratio(g, x, s, direction, kind) for x, s in zip(xd, sd)])
    from scipy.stats import binom
    k = int(binom.ppf(GATE_NULL, len(r), GATE_NULL)) + 1
    own = float(np.ceil(np.sort(r)[min(k, len(r)) - 1] * 100) / 100)
    floor = max(FLOOR, own)
    rc = np.array([_ratio(g, x, s, direction, kind) for x, s in zip(xc, sc)])
    p_null = float(np.mean(rc < floor))
    sbar = sc.mean(axis=0)
    out = []
    for i in gaps(g, kind):
        sref = float(sbar[i + 1:].max())
        sh = dip(g, i) if kind == "turn" else SIGN[direction] * shape(g, i)

        def placed(amp, fl):
            hits = []
            for x, s in zip(xc, sc):
                y = x + amp * sh
                hits.append(place(g, y, direction, float(s[reference(g, y, direction, kind)]),
                                  kind, floor=fl)["index"] == i + 1)
            return float(np.mean(hits))
        ok = placed(LOW_POWER * sref, floor)
        curve = {a: placed(a * sref, floor) for a in AMPS}
        power = next((a for a in AMPS if curve[a] >= GATE_EFFECT), None)
        out.append({"gap": f"{g.steps[i]}-{g.steps[i + 1]}", "to_index": i + 1,
                    "width_octaves": float(WIDTH * (g.X[i + 1] - g.X[i])),
                    "sigma_ref": sref, "amplitude": LOW_POWER * sref, "placed_right": ok,
                    "pass": ok >= GATE_EFFECT, "power": power,
                    "curve": {str(a): v for a, v in curve.items()}})
    return {"grid": g.name, "floor": floor, "own_floor": own, "null_no_change": p_null,
            "null_pass": p_null >= GATE_NULL, "effect": out,
            "pass": p_null >= GATE_NULL and all(e["pass"] for e in out)}


def gate_or_refuse(g: Grid, rid: str, direction: str, sname: str, kind: str, units, rng) -> dict:
    if constant_row(units, sname):
        return {"pass": False, "refused": "constant: no spread at any step", "floor": None,
                "null_no_change": None, "effect": []}
    return gate_row(g, direction, sname, kind, units, rng)


# ---------------------------------------------------------------- the weights rows (producer)

def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u2.py", "trackA_force/a3.py", "trackA_force/a3_weights.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def attractive_fraction(core: np.ndarray) -> float:
    eigs = np.linalg.eigvals(np.asarray(core, np.float64))
    e_all = float(np.sum(np.abs(eigs) ** 2))
    return float(np.sum(np.abs(eigs[eigs.real > 0.0]) ** 2) / max(e_all, 1e-300))


def layers_of(model):
    if hasattr(model, "gpt_neox"):
        return list(model.gpt_neox.layers), "neox"
    if hasattr(model, "transformer"):
        return list(model.transformer.h), "gpt2"
    raise C9Refused(f"{type(model).__name__}: neither GPT-NeoX nor GPT-2")


def _np(t) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def matrices(layer, arch: str) -> Dict[str, np.ndarray]:
    if arch == "neox":
        a, m = layer.attention, layer.mlp
        return {"qkv": _np(a.query_key_value.weight), "o": _np(a.dense.weight),
                "mlp_in": _np(m.dense_h_to_4h.weight), "mlp_out": _np(m.dense_4h_to_h.weight)}
    a, m = layer.attn, layer.mlp
    return {"qkv": _np(a.c_attn.weight), "o": _np(a.c_proj.weight),
            "mlp_in": _np(m.c_fc.weight), "mlp_out": _np(m.c_proj.weight)}


def head_cores(layer, arch: str, n_heads: int) -> list:
    """Each head's d_head × d_head OV core (its nonzero spectrum is the head's OV's)."""
    if arch == "neox":
        from core.pythia_weights import split_qkv_from_layer
        q = split_qkv_from_layer(layer)
        V = np.asarray(q["V"], np.float64)                         # (H·dh, d), Linear (out, in)
        O = _np(layer.attention.dense.weight)                      # (d, H·dh)
        dh = q["_geometry"]["head_size"]
        if q["_geometry"]["num_heads"] != n_heads:
            raise C9Refused(f"{q['_geometry']['num_heads']} heads, config has {n_heads}")
        return [V[h * dh:(h + 1) * dh] @ O[:, h * dh:(h + 1) * dh] for h in range(n_heads)]
    W = _np(layer.attn.c_attn.weight)                              # (d, 3d), Conv1D (in, out)
    P = _np(layer.attn.c_proj.weight)                              # (d, d)
    d = W.shape[0]
    dh = d // n_heads
    V = W[:, 2 * d:]
    return [P[h * dh:(h + 1) * dh] @ V[:, h * dh:(h + 1) * dh] for h in range(n_heads)]


def load(run: str, step: int):
    import torch
    if run == "410m":
        from trackB_circuit.b0_gate import load as l410
        return l410(step, "cpu", torch.float32)[0]
    if run in GPT2_SEEDS:
        from trackC_writeup.c2_warmup import load as lg
        return lg(run, step, "cpu")
    from trackC_writeup.c4_read import load_local
    root = {"A0": HDD / "c4", "A0p": HDD / "c4pc"}[run]
    return load_local(root / run / "ckpt" / f"step{step}", "cpu")


def weights(run: str, out: Path) -> None:
    from trackA_force.a3_weights import gram_eigs, powerlaw_alpha
    g = grid_of(run)
    d = out / "weights" / run
    d.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    m0 = load(run, 0)
    L0, arch = layers_of(m0)
    W0 = [matrices(l, arch) for l in L0]
    n_heads = m0.config.num_attention_heads if hasattr(m0.config, "num_attention_heads") \
        else m0.config.n_head
    del m0
    for s in g.steps:
        f = d / f"step{s}.json"
        if f.exists():
            if json.loads(f.read_text()).get("code") != code:
                raise C9Refused(f"refusing to resume: {f} was written by other code")
            continue
        t0 = time.time()
        m = load(run, s)
        Ls, arch_s = layers_of(m)
        if arch_s != arch or len(Ls) != len(W0):
            raise C9Refused(f"{run} step {s}: {arch_s} × {len(Ls)}, step 0 {arch} × {len(W0)}")
        rec = {"run": run, "step": s, "code": code, "arch": arch, "n_heads": n_heads,
               "rel_change": {t: [] for t in TYPES}, "alpha": {t: [] for t in TYPES},
               "e2": []}
        for l, layer in enumerate(Ls):
            for t, W in matrices(layer, arch).items():
                rec["rel_change"][t].append(float(np.linalg.norm(W - W0[l][t]) / np.linalg.norm(W0[l][t])))
                rec["alpha"][t].append(powerlaw_alpha(gram_eigs(W))["alpha"])
            rec["e2"].append([attractive_fraction(c) for c in head_cores(layer, arch, n_heads)])
        del m
        f.write_text(json.dumps(rec) + "\n")
        print(json.dumps({"run": run, "step": s, "e2_median": round(float(np.median(rec["e2"])), 4),
                          "s": round(time.time() - t0, 1)}), flush=True)


def weight_units(run: str, out: Path) -> Dict[str, np.ndarray]:
    """E2 (heads, S, 1); E7, E8 (layers, S, types), on the run's grid; refuses a missing step."""
    g = grid_of(run)
    d = out / "weights" / run
    recs = {}
    for s in g.steps:
        f = d / f"step{s}.json"
        if not f.exists():
            raise C9Refused(f"{run}: no weights record {f}")
        recs[s] = json.loads(f.read_text())
    codes = {r["code"] for r in recs.values()}
    if len(codes) != 1:
        raise C9Refused(f"{run}: weights records from {len(codes)} codes")
    e2 = np.array([np.ravel(recs[s]["e2"]) for s in g.steps]).T[..., None]
    u = {"E2": e2}
    for rid, key in (("E7", "rel_change"), ("E8", "alpha")):
        u[rid] = np.array([[recs[s][key][t] for s in g.steps] for t in TYPES]).transpose(2, 1, 0)
    if not np.all(u["E7"][:, 0, :] == 0):
        raise C9Refused(f"{run}: E7 at step 0 is not 0")
    u["_code"] = codes.pop()
    return u


# ---------------------------------------------------------------- the known answer

def a3_units() -> dict:
    """A3 unit 5's units, built by a3.py's own builders from its records (`a3.run`'s inputs)."""
    rec = json.loads((A3U5 / "a3_rows.json").read_text())
    inp = rec["inputs"]
    r15 = a3.rows_e1_e5()
    wr = a3.rows_weights(Path(inp["a3_weights_dir"]))
    sr = a3.rows_sim(Path(inp["a3_sim_dir"]))
    nr = a3.rows_ngram(Path(inp["a3_ngram_dir"]))
    nf = a3.rows_ngram(Path(inp["a3_ngram_full_dir"]))
    a3.check_full(nf, Path(inp["a3_ngram_dir"]), Path(inp["a3_ngram_full_dir"]))
    units = {"E2": a3.row_e2(), "E5": r15["E5"], **a3.rows_e3_e4(A3U5 / "attn_rows.json"),
             "E7": wr["E7"], "E8": wr["E8"], "E11": sr["E11"], "E6u": nr["E6u"],
             "E6b": nr["E6b"], "E6b_full": nf["E6b"]}
    return {"units": units, "record": rec}


def replay(units: dict) -> dict:
    """`a3.run`'s gate-then-locate loop, through this module's grid functions, A3's seed."""
    g, rng = PYTHIA, np.random.default_rng(a3.SEED)
    gate = {}
    for rid in (k for k in a3.ROWS if k in units):
        _, direction, sname, _ = a3.ROWS[rid]
        gate[rid] = gate_or_refuse(g, rid, direction, sname, a3.kind_of(rid), units[rid], rng)
    rows = {}
    for rid, (_, direction, sname, _) in a3.ROWS.items():
        if rid not in units:
            continue
        if not gate[rid]["pass"]:
            rows[rid] = {"reading": "refused", "step": None}
            continue
        r = locate_units(g, units[rid], STATS[sname], direction, rng, a3.kind_of(rid),
                         floor=gate[rid]["floor"])
        r["censored"] = censored(g, r)
        rows[rid] = r
    return {"gate": gate, "rows": rows}


def compare(rep: dict, rec: dict) -> dict:
    out, ok = {}, True
    for rid, r in rep["rows"].items():
        want = rec["rows"][rid]
        same = (r["reading"] == want["reading"] and r.get("step") == want.get("step")
                and r.get("censored") == want.get("censored")
                and rep["gate"][rid].get("floor") == rec["gate"][rid].get("floor"))
        out[rid] = {"got": [r["reading"], r.get("step"), r.get("censored"), rep["gate"][rid].get("floor")],
                    "want": [want["reading"], want.get("step"), want.get("censored"),
                             rec["gate"][rid].get("floor")], "equal": same}
        ok &= same
    return {"rows": out, "pass": ok}


def known(out: Path) -> dict:
    t0 = time.time()
    a = a3_units()
    rep = replay(a["units"])
    ki = compare(rep, a["record"])
    ki["null_no_change_equal"] = {rid: rep["gate"][rid].get("null_no_change")
                                  == a["record"]["gate"][rid].get("null_no_change")
                                  for rid in rep["gate"]}
    print(f"known (i): {'pass' if ki['pass'] else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    res = {"code": code_sha(), "a3_record": str(A3U5 / "a3_rows.json"),
           "a3_record_sha256": hashlib.sha256((A3U5 / "a3_rows.json").read_bytes()).hexdigest(),
           "i": ki}
    if ki["pass"]:
        w = weight_units("410m", out)
        from trackA_force.a3_weights import load_steps
        old = load_steps(A3W, STEPS)
        e78 = {}
        for rid, key in (("E7", "rel_change"), ("E8", "alpha")):
            ref = np.array([[old[s][key][t] for s in STEPS] for t in TYPES]).transpose(2, 1, 0)
            e78[rid] = float(np.max(np.abs(w[rid] - ref) / np.maximum(np.abs(ref), 1e-300)))
        ov = json.loads((DATA / "analysis" / "ov_per_head_series.json").read_text())
        cols = [ov["steps"].index(s) for s in STEPS]
        keys = [f"{l},{h}" for l in range(ov["n_blocks"]) for h in range(ov["n_heads"])]
        e2ref = np.array([[ov["per_head"][k]["attractive_energy_fraction_core"][c] for c in cols]
                          for k in keys])
        e2gap = float(np.max(np.abs(w["E2"][..., 0] - e2ref)))
        swapped = dict(a["units"], E2=w["E2"], E7=w["E7"], E8=w["E8"])
        kii = compare(replay(swapped), a["record"])
        kii.update(e7_max_rel_diff=e78["E7"], e8_max_rel_diff=e78["E8"], e2_max_abs_diff=e2gap,
                   values_pass=bool(e78["E7"] <= E78_RTOL and e78["E8"] <= E78_RTOL and e2gap <= E2_TOL),
                   weights_code=w["_code"])
        kii["pass"] = bool(kii["pass"] and kii["values_pass"])
        res["ii"] = kii
        print(f"known (ii): {'pass' if kii['pass'] else 'FAIL'} (E7 {e78['E7']:.2e}, E8 "
              f"{e78['E8']:.2e}, E2 {e2gap:.2e}; {time.time() - t0:.0f} s)", flush=True)
    res["pass"] = bool(ki["pass"] and res.get("ii", {}).get("pass"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "known.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ---------------------------------------------------------------- the gates (no location)

def gate(run: str, out: Path) -> dict:
    kn = out / "known.json"
    if not kn.exists() or not json.loads(kn.read_text()).get("pass"):
        raise C9Refused("the known answer has not passed: no run is gated before it")
    g = grid_of(run)
    rng = np.random.default_rng([SEED_U2, RUNS.index(run)])
    u = weight_units(run, out)
    t0 = time.time()
    res = {"code": code_sha(), "run": run, "grid": list(g.steps), "window_last": g.last,
           "weights_code": u["_code"], "seed": [SEED_U2, RUNS.index(run)], "rows": {}}
    for rid, (direction, sname) in WROWS.items():
        gr = gate_or_refuse(g, rid, direction, sname, "half", u[rid], rng)
        gr["n_units"] = int(u[rid].shape[0])
        res["rows"][rid] = gr
        print(f"{run} gate {rid}: floor {gr['floor']}, null {gr['null_no_change']}, effect min "
              f"{min((e['placed_right'] for e in gr['effect']), default=None)} -> "
              f"{'pass' if gr['pass'] else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    d = out / "gate"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("weights", "gate"):
        p = sub.add_parser(c)
        p.add_argument("--run", required=True, choices=RUNS)
        p.add_argument("--out", type=Path, default=OUT)
    k = sub.add_parser("known"); k.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "weights":
        weights(a.run, a.out)
    elif a.cmd == "gate":
        if a.run == "410m":
            raise C9Refused("410m's weights rows are A3's, gated by the known answer")
        gate(a.run, a.out)
    else:
        return 0 if known(a.out)["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
