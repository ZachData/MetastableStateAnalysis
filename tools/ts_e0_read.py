"""The labels of `MATH_TRAINING_SYMMETRY.md` §10 from `ts_e0_backward.py`'s gradients (tier 1).

Rules as frozen at `2cc19b5` (before any output): statistics size / persist / move on the last
layer's non-rotary key bias (KBn, dims 16-63 of each head's key bias), controls KBr (dims 0-15)
and QBn (non-rotary query bias); gates G0 (R64 invisible), G1 (planted), G2 (move readable);
per-arm "systematic" / "weak" / "random"; per-cell E0 label; Parked 1's label from Adam's step 2.

    python tools/ts_e0_read.py read      -> data/ts/e0/read.json (and a printed summary)
    python tools/ts_e0_read.py beside    -> data/ts/e0/beside_c4.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ts_e0_backward import (BAR_MOVE, BAR_RAND, BAR_SYS, BETAS, C4, CLIP, EPS, G0_BAR,  # noqa: E402
                                  G1_BAR, MOTION, OUT, PEAK, ROT, WARM, tensors)

LAYERS = {"pythia-70m": 6, "pythia-410m": 24}
ARMS = ("R32", "H16", "F16")
CELLS = [(m, s) for m in LAYERS for s in (256, 143000)]
UNIT = 2.0 ** -24


def qkv_name(model: str) -> str:
    return f"gpt_neox.layers.{LAYERS[model] - 1}.attention.query_key_value.bias"


def parts(v: np.ndarray) -> dict:
    """(H*3*64,) -> KBn, KBr, QBn as flat arrays (HF's (heads, 3, head) layout)."""
    x = v.reshape(-1, 3, 64)
    return {"KBn": x[:, 1, ROT:].ravel(), "KBr": x[:, 1, :ROT].ravel(), "QBn": x[:, 0, ROT:].ravel()}


def load(stem: str):
    p = OUT / f"{stem}.json"
    if not p.exists():
        return None, None
    z = np.load(OUT / f"{stem}.npz")
    return json.loads(p.read_text()), z


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def agree(a, b) -> dict:
    """Share of coordinates with sign(a) = sign(b), over those where both are nonzero."""
    ok = (a != 0) & (b != 0)
    n = int(ok.sum())
    return {"share": float(np.mean(np.sign(a[ok]) == np.sign(b[ok]))) if n else None, "n": n}


def arm_stats(model: str, step: int, arm: str, c: float, dB: dict) -> dict | None:
    rec, z = load(f"{model}_{step}_{arm}")
    if rec is None:
        return None
    n = qkv_name(model)
    gA, gB = parts(z[f"A|{n}"]), parts(z[f"B|{n}"])
    g = {k: (gA[k] + gB[k]) / 2 for k in gA}
    out = {"rows": rec["rows"], "scales": rec["scales"], "global_norm": rec["global_norm"],
           "size_ratio": rms(g["KBn"]) / rms(g["KBr"]),
           "size_over_eps": c * rms(g["KBn"]) / EPS,
           "persist": agree(gA["KBn"], gB["KBn"]),
           "persist_KBr": agree(gA["KBr"], gB["KBr"]),
           "move": agree(-g["KBn"], dB["KBn"]),
           "move_KBr": agree(-g["KBr"], dB["KBr"]),
           "move_QBn": agree(-g["QBn"], dB["QBn"]),
           "along_b": agree(-g["KBn"], dB["b_KBn"])}
    # every layer's persist (description, beside)
    L = LAYERS[model]
    out["persist_by_layer"] = []
    for l in range(L):
        nl = f"gpt_neox.layers.{l}.attention.query_key_value.bias"
        out["persist_by_layer"].append(agree(parts(z[f"A|{nl}"])["KBn"], parts(z[f"B|{nl}"])["KBn"])["share"])
    return out


def arm_label(st: dict, move_readable: bool) -> str:
    p = st["persist"]["share"]
    if p is None or p < BAR_RAND:
        return "random"
    mv = st["move"]["share"]
    if p >= BAR_SYS and st["size_over_eps"] >= 1.0 and (not move_readable or (mv or 0) >= BAR_SYS):
        return "systematic"
    return "weak"


def cell_label(lab: dict) -> str:
    if lab.get("R32") == "systematic":
        return "(c) R32 systematic: not specific to fp16"
    if lab.get("H16") == "systematic":
        return "(a-HF) fp16 rounding in HF's own graph"
    if lab.get("F16") == "systematic":
        return "(a-FA) the flash kernel's rounding"
    return "(b') none systematic"


def gates() -> dict:
    g = {}
    ratios = {}
    for s in (256, 143000):
        rec, z = load(f"pythia-70m_{s}_R64")
        if rec is None:
            ratios[s] = None
            continue
        n = qkv_name("pythia-70m")
        gg = {k: (parts(z[f"A|{n}"])[k] + parts(z[f"B|{n}"])[k]) / 2 for k in ("KBn", "KBr")}
        ratios[s] = rms(gg["KBn"]) / rms(gg["KBr"])
    g["G0"] = {"R64_size_ratio": ratios,
               "pass": all(r is not None and r <= G0_BAR for r in ratios.values())}
    rec, z = load("pythia-70m_256_R32_plant")
    if rec is not None:
        n = qkv_name("pythia-70m")
        w = np.random.default_rng(0).choice([-1.0, 1.0], size=(8, 64 - ROT)).ravel()
        gA, gB = parts(z[f"A|{n}"])["KBn"], parts(z[f"B|{n}"])["KBn"]
        a = agree((gA + gB) / 2, w)["share"]
        p = agree(gA, gB)["share"]
        g["G1"] = {"kappa": rec["plant"], "agree_w": a, "persist": p, "pass": a >= G1_BAR and p >= G1_BAR}
    else:
        g["G1"] = {"pass": False, "missing": True}
    return g


def motion(model: str, step: int) -> dict:
    n = qkv_name(model)
    t0, t1 = (step, MOTION[(model, step)]) if MOTION[(model, step)] > step else (MOTION[(model, step)], step)
    b0, b1, bs = tensors(model, t0, [n])[n], tensors(model, t1, [n])[n], tensors(model, step, [n])[n]
    d = parts(b1 - b0)
    d["b_KBn"] = parts(bs)["KBn"]
    d["interval"] = [t0, t1]
    return d


def parked1(model: str) -> dict | None:
    r0, z0 = load(f"{model}_0_R32_it0")
    r1, z1 = load(f"{model}_0_R32_it1")
    if r0 is None or r1 is None:
        return None
    L = LAYERS[model]
    n = f"gpt_neox.layers.{L - 1}.attention.dense.bias"
    g0 = (z0[f"A|{n}"] + z0[f"B|{n}"]) / 2
    g1 = (z1[f"A|{n}"] + z1[f"B|{n}"]) / 2
    lr1 = PEAK[model] * 1 / WARM                     # iteration 1's rate (iteration 0 ran at 0)
    obs = tensors(model, 2, [n])[n]
    b1_, b2_ = BETAS
    out = {"norms": [r0["global_norm"], r1["global_norm"]], "rows": r0["rows"], "rate_units": lr1 / UNIT,
           "observed_u_median": float(np.median(np.abs(obs)) / lr1),
           "observed_units_mode": float(np.bincount(np.abs(np.round(obs / UNIT)).astype(int)).argmax())}
    for tag, clip in (("clip", True), ("noclip", False)):
        c0 = min(1.0, CLIP / r0["global_norm"]) if clip else 1.0
        c1 = min(1.0, CLIP / r1["global_norm"]) if clip else 1.0
        m = (1 - b1_) * (b1_ * c0 * g0 + c1 * g1)
        v = (1 - b2_) * (b2_ * (c0 * g0) ** 2 + (c1 * g1) ** 2)
        u = (m / (1 - b1_ ** 2)) / (np.sqrt(v / (1 - b2_ ** 2)) + EPS)
        pred = -lr1 * u
        pu, ou = np.round(pred / UNIT), np.round(obs / UNIT)
        out[tag] = {"c": [c0, c1], "u_median": float(np.median(np.abs(u))),
                    "within_1_unit": float(np.mean(np.abs(pu - ou) <= 1)),
                    "sign_agree": float(np.mean(np.sign(pred) == np.sign(obs))),
                    "ratio_g1_g0_median": float(np.median(np.abs(c1 * g1) / np.maximum(np.abs(c0 * g0), 1e-30)))}
    pc = out["clip"]
    out["label"] = ("explained by the batches' gradients and the clip"
                    if 0.80 <= pc["u_median"] <= 0.90 and pc["within_1_unit"] >= 0.80
                    else "not explained" if pc["u_median"] >= 0.95 else "partial")
    h, _ = load(f"{model}_0_H16_it0")
    out["H16_it0_nonfinite_at_2^12"] = None if h is None else h["nonfinite_at_first_scale"]
    return out


def a0_first_step() -> dict:
    """C4's A0 (bf16, same batches and clip): logged grad norms at iterations 0-1, step-2 u."""
    log = [json.loads(x) for x in (C4 / "A0/train_log.jsonl").read_text().splitlines()]
    norms = {r["step"]: r["grad_norm"] for r in log if "step" in r and r["step"] in (1, 2)}
    from safetensors import safe_open
    n = "gpt_neox.layers.5.attention.dense.bias"
    with safe_open(str(C4 / "A0/ckpt/step2/model.safetensors"), "pt") as f:
        b = f.get_tensor(n).double().numpy()
    lr1 = PEAK["pythia-70m"] / WARM
    return {"grad_norm_it0_it1": [norms.get(1), norms.get(2)], "u_median": float(np.median(np.abs(b)) / lr1),
            "u_p10_p90": [float(np.percentile(np.abs(b), 10) / lr1), float(np.percentile(np.abs(b), 90) / lr1)]}


def read_all() -> dict:
    res = {"frozen_at": "2cc19b5", "gates": gates(), "cells": {}, "parked1": {}}
    for model, step in CELLS:
        rc, _ = load(f"{model}_{step}_R32")
        if rc is None:
            continue
        c = min(1.0, CLIP / rc["global_norm"])
        dB = motion(model, step)
        st = {a: arm_stats(model, step, a, c, dB) for a in ARMS}
        r = st["R32"]
        readable = (r["move_KBr"]["share"] or 0) >= BAR_MOVE and (r["move_QBn"]["share"] or 0) >= BAR_MOVE
        labs = {a: arm_label(s, readable) for a, s in st.items() if s is not None}
        res["cells"][f"{model}@{step}"] = {"c": c, "motion_interval": dB["interval"], "move_readable": readable,
                                          "arms": st, "arm_labels": labs,
                                          "label": cell_label(labs) if res["gates"]["G1"]["pass"] else "refused (G1)"}
    if not res["gates"]["G0"]["pass"]:
        res["unit_label"] = "visible in exact arithmetic (G0 failed)"
    else:
        ls = {v["label"] for v in res["cells"].values()}
        res["unit_label"] = ls.pop() if len(ls) == 1 else "mixed"
    for model in LAYERS:
        res["parked1"][model] = parked1(model)
    try:
        res["parked1"]["A0_beside"] = a0_first_step()
    except Exception as e:                       # beside; never blocks the read
        res["parked1"]["A0_beside"] = {"error": repr(e)}
    (OUT / "read.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def beside() -> dict:
    """C4's A0 and F against the released 70m: last layer's KBn rms over tau_1 (weights only)."""
    from safetensors import safe_open
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from trackC_writeup.c4_train import summed
    n = qkv_name("pythia-70m")
    out = {}
    for step in (256, 512, 1000, 4000):
        row = {}
        for arm in ("A0", "F"):
            p = C4 / arm / "ckpt" / f"step{step}" / "model.safetensors"
            if p.exists():
                with safe_open(str(p), "pt") as f:
                    row[arm] = rms(parts(f.get_tensor(n).double().numpy())["KBn"]) / summed(arm, step)
        try:
            row["released"] = rms(parts(tensors("pythia-70m", step, [n])[n])["KBn"]) / summed("A0", step)
        except Exception:
            row["released"] = None
        out[str(step)] = row
    (OUT / "beside_c4.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    r = read_all() if sys.argv[1] == "read" else beside()
    print(json.dumps(r, indent=1, default=str)[:6000])
