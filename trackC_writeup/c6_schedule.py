"""
trackC_writeup/c6_schedule.py — C6 U0: what `EleutherAI/pythia-70m-warmup01` shares with
`EleutherAI/pythia-70m`, read off the released weights (`design-C6.md` "What the run is").

The Hub has no model card for the run, and Belrose et al. (2402.04362 §5) give only "an extended
learning rate warmup period of 14,300 steps"; the peak rate, the initialisation and the data order
are not written anywhere found (`lit-C2p-fulltext.md` Parked 2). The weights carry them:

- **initialisation**: the two step-0 checkpoints are compared tensor by tensor; the control is
  `pythia-70m-seed1`'s step 0 (another seed), on the tensors that are not constant at init;
- **the first batch**: the first update (step 2; step 1 equals step 0 because NeoX's rate at
  iteration 0 is 0) is about lr · sign(g) per coordinate under Adam, so with one init and one
  batch the two runs' step-2 changes have the same sign wherever both are resolved in fp16; the
  control is seed 1's step-2 change against `pythia-70m`'s (another init and batch);
- **the schedule**: under Adam a coordinate's change over an interval scales with the summed rate
  while the two runs' gradients agree. (i) Early, t ≤ 512, the zero-initialised biases (fp16
  resolves them to 6e-8): the ratio of the two runs' |change| quantiles against the summed-rate
  ratio each hypothesis predicts; (ii) between grid steps from 512 to 143 000, the norm of each
  run's weight change against the same.

The hypotheses (NeoX's `AnnealingLR`: linear warmup over W iterations from 0, then cosine from
the peak to `min_lr` at 143 000; Pythia-70M's config: peak 1e-3, min 1e-4, W 1430): **H0** W 14 300
at Pythia's peak and floor; **Hp/2**, **H2p** W 14 300 at half and twice the peak; **Hw** W 1430
(no change). The reading, fixed before this file first ran (step 2's bias values had been seen,
one fp16 unit in `warmup01` against ten in `pythia-70m`): H0 is **confirmed** if its median
|log(measured / predicted)| over the readouts (i) and (ii) is below log 1.25 and below every other
hypothesis's; else **not confirmed**, and C6's bands are not frozen on Belrose's 14 300.

Beside, the data order after the first batch (added after the first run, descriptive): the sign
agreement of each run's bias increments over the grid intervals to 512 against `pythia-70m`'s, with
seed 1's as the control (another init and order; at step 2 it agrees at chance).

What this does not show: the data order once the runs' parameters part (their rates differ, so
their gradients do even on one batch); the schedule's exact form between grid steps; Adam's
betas or weight decay (taken from Pythia's config); the peak apart from the warmup's length
except through the late intervals, where the displacement proxy is approximate.

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_schedule --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from trackA_force.a1a import REPO, STEPS

RUN = "EleutherAI/pythia-70m-warmup01"
BASE = "EleutherAI/pythia-70m"
CONTROL = "EleutherAI/pythia-70m-seed1"
PEAK, MIN_LR, DECAY, W_BASE = 1e-3, 1e-4, 143000, 1430          # gpt-neox configs/pythia/70M.yml
HYPS = {"H0": (PEAK, 14300), "Hp/2": (PEAK / 2, 14300), "H2p": (PEAK * 2, 14300), "Hw": (PEAK, W_BASE)}
EARLY = tuple(s for s in STEPS if 4 <= s <= 512)
LATE = tuple(zip(STEPS[STEPS.index(512):-1], STEPS[STEPS.index(512) + 1:]))
QUANTS = (0.5, 0.9, 0.99)
CONFIRM = math.log(1.25)


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackC_writeup/c6_schedule.py").read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- pure: the schedule

def lr(i: int, peak: float, warmup: int, min_lr: float = None, decay: int = DECAY) -> float:
    """NeoX `AnnealingLR` at iteration i (0-based; checkpoint step t holds iterations 0 … t−1)."""
    mn = peak * MIN_LR / PEAK if min_lr is None else min_lr
    if i <= warmup:
        return peak * i / warmup
    return mn + (peak - mn) / 2 * (math.cos(math.pi * (i - warmup) / (decay - warmup)) + 1)


def cum(a: int, b: int, peak: float, warmup: int) -> float:
    """The summed rate over iterations a … b−1 (checkpoint step a to step b)."""
    return sum(lr(i, peak, warmup) for i in range(a, b))


def predicted(a: int, b: int, hyp: str) -> float:
    """The run's summed rate over (a, b] against `pythia-70m`'s."""
    return cum(a, b, *HYPS[hyp]) / cum(a, b, PEAK, W_BASE)


def verdict(readouts: List[dict]) -> dict:
    err = {h: float(np.median([abs(math.log(r["measured"] / r["predicted"][h])) for r in readouts]))
           for h in HYPS}
    best = min(err, key=err.get)
    ok = err["H0"] < CONFIRM and best == "H0"
    return {"median_abs_log_err": err, "best": best, "H0": "confirmed" if ok else "not confirmed",
            "threshold": CONFIRM}


# ---------------------------------------------------------------- the weights

def sd(repo: str, step: int) -> Dict[str, "np.ndarray"]:
    """A checkpoint's floating tensors as float32 (fp16 on the Hub, so exact), the -inf
    `masked_bias` buffers dropped; `pythia-70m` is cached as safetensors, the others as .bin."""
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        from safetensors.torch import load_file
        t = load_file(hf_hub_download(repo, "model.safetensors", revision=f"step{step}"))
    except (LocalEntryNotFoundError, OSError):
        t = torch.load(hf_hub_download(repo, "pytorch_model.bin", revision=f"step{step}"),
                       map_location="cpu", weights_only=True)
    return {k: v.float().numpy() for k, v in t.items()
            if v.dtype.is_floating_point and bool(torch.isfinite(v).all())}


def varying(p0: dict) -> List[str]:
    """Tensors not constant at init (layer norms start at 1 and 0, biases at 0)."""
    return [k for k, v in p0.items() if v.size > 1 and float(v.std()) > 0]


def zero_biases(p0: dict) -> List[str]:
    return [k for k, v in p0.items() if k.endswith(".bias") and "layernorm" not in k and not np.any(v)]


def sign_agree(da: np.ndarray, db: np.ndarray) -> dict:
    m = (da != 0) & (db != 0)
    return {"n_both": int(m.sum()), "n": int(da.size), "agree": float(np.mean(np.sign(da[m]) == np.sign(db[m])))}


def flat(d: dict, keys: List[str], ref: dict = None) -> np.ndarray:
    return np.concatenate([(d[k] - (ref[k] if ref is not None else 0)).ravel() for k in keys])


def main(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    w0, p0, c0 = sd(RUN, 0), sd(BASE, 0), sd(CONTROL, 0)
    var = varying(p0)
    init = {"run_vs_base": {"equal": sum(np.array_equal(w0[k], p0[k]) for k in p0), "of": len(p0),
                            "keys_only_in_run": sorted(set(w0) - set(p0)), "keys_only_in_base": sorted(set(p0) - set(w0))},
            "control_vs_base": {"equal_varying": sum(np.array_equal(c0[k], p0[k]) for k in var),
                                "of_varying": len(var)},
            "run_vs_base_varying": sum(np.array_equal(w0[k], p0[k]) for k in var)}
    zb = zero_biases(p0)
    w2, p2, c2 = sd(RUN, 2), sd(BASE, 2), sd(CONTROL, 2)
    w1, p1 = sd(RUN, 1), sd(BASE, 1)
    first = {"step1_equals_step0": {"run": all(np.array_equal(w1[k], w0[k]) for k in w0),
                                    "base": all(np.array_equal(p1[k], p0[k]) for k in p0)},
             "all": {"run_vs_base": sign_agree(flat(w2, list(p0), w0), flat(p2, list(p0), p0)),
                     "control_vs_base": sign_agree(flat(c2, list(p0), c0), flat(p2, list(p0), p0))},
             "zero_biases": {"run_vs_base": sign_agree(flat(w2, zb), flat(p2, zb)),
                             "control_vs_base": sign_agree(flat(c2, zb), flat(p2, zb)),
                             "max_abs": {"run": float(np.abs(flat(w2, zb)).max()),
                                         "base": float(np.abs(flat(p2, zb)).max())},
                             "lr_iter1": {h: lr(1, *HYPS[h]) for h in HYPS} | {"base": lr(1, PEAK, W_BASE)}}}
    readouts, order = [], []
    prev = {"run": w2, "base": p2, "control": c2}
    for t in EARLY:
        cur = {"run": sd(RUN, t), "base": sd(BASE, t), "control": sd(CONTROL, t)}
        inc = {k: flat(cur[k], zb, prev[k]) for k in cur}
        order.append({"interval": [STEPS[STEPS.index(t) - 1], t],
                      "run_vs_base": sign_agree(inc["run"], inc["base"]),
                      "control_vs_base": sign_agree(inc["control"], inc["base"])})
        prev = cur
        wb, pb = np.abs(flat(cur["run"], zb)), np.abs(flat(cur["base"], zb))
        for q in QUANTS:
            readouts.append({"kind": "early", "step": t, "q": q,
                             "measured": float(np.quantile(wb, q) / np.quantile(pb, q)),
                             "predicted": {h: predicted(0, t, h) for h in HYPS}})
    prev_w, prev_p = sd(RUN, LATE[0][0]), sd(BASE, LATE[0][0])
    for s, t in LATE:
        cur_w, cur_p = sd(RUN, t), sd(BASE, t)
        nw = math.sqrt(sum(float(np.sum((cur_w[k] - prev_w[k]) ** 2)) for k in var))
        npp = math.sqrt(sum(float(np.sum((cur_p[k] - prev_p[k]) ** 2)) for k in var))
        readouts.append({"kind": "late", "interval": [s, t], "measured": nw / npp,
                         "norm_run": nw, "norm_base": npp,
                         "predicted": {h: predicted(s, t, h) for h in HYPS}})
        prev_w, prev_p = cur_w, cur_p
    res = {"code": code_sha(), "run": RUN, "base": BASE, "control": CONTROL, "hyps": HYPS,
           "init": init, "first_update": first, "order_increments": order, "readouts": readouts, "verdict": verdict(readouts),
           "hf_home": os.environ.get("HF_HOME")}
    (out / "schedule.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"init": init, "first": first, "verdict": res["verdict"]}, indent=1))
    for o in order:
        print("increment", o["interval"], f"run {o['run_vs_base']['agree']:.4f} (n {o['run_vs_base']['n_both']})",
              f"control {o['control_vs_base']['agree']:.4f}")
    for r in readouts:
        print(r.get("step", r.get("interval")), r.get("q", ""), f"{r['measured']:.4f}",
              " ".join(f"{h} {v:.4f}" for h, v in r["predicted"].items()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    raise SystemExit(main(ap.parse_args()))
