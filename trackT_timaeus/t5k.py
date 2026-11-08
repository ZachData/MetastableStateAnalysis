"""
trackT_timaeus/t5k.py — T5′ U1: what forms with the rise, per U4 arm (`trackT_timaeus/design-T5p.md`,
frozen 2026-11-07 on Blocked 138, D1–D5 all (a), before any T5′ output).

Per checkpoint of each run (six arms × 49, released `-32m` × 53, `pythia-410m` × 18), on R = 32
random-token sequences (T = 256, seed 20261107, the model's ordinary vocabulary, no BOS):
    repeated  [r, r]     control  [r, r′]     null  [r, r″]
per-head prefix matching PM_h (attention from query T + j to key j + 1, j = 0 … T − 2), the
previous-token score on the control, the second half's token loss, and on 1024 held-out
sequences of `t5u4/data/tail.bin` (64–1087, trained by no arm; Blocked 139 (a), post hoc: the first
64, read by the 2026-11-07 gate, had no power) the token loss at positions 490–509 minus 40–59.
I = max_h (PM_h(rep) − PM_h(ctrl)); I_null the same with the null set (the planted null).
**I_text, beside** (Blocked 139 (a), the user's; post hoc, description only; fixed in
`design-T5p.md` "Blocked 139 decided" before it was built): I's formula on real text, R = 32 tail
rows each for x (1088–1119), x′ (1120–1151), x″ (1152–1183), each row's first T = 256 tokens:
    repeated  [x, x]     control  [x, x′]     null  [x, x″]
I_text_null is its planted null (max ≤ 0.05 described beside, not a gate). Dated by D2's rule
(floor 0.1), no label; I on random tokens stays the labelled statistic (D1 (a)).
The dating rule, the outcome table, the gate and the bridge are the design's; written here before
any output and tested on synthetic series (`tests/test_trackT_t5k.py`).

Implementation details fixed here before any output (the design leaves them open):
- P averages queries 1 … 2T − 1 of the control; B is the mean token loss predicting positions
  T + 1 … 2T − 1 (0-indexed), repeated minus control, per sequence pair.
- Bootstrap: the R sequence pairs are resampled jointly (repeated, control and null share r_i);
  ICL resamples its 1024 sequences. 1000 resamples, seed 20261030.
- The 4 SE floor for P, B, ICL: rise < 4 × the paired bootstrap SE of x(peak) − x(base), both steps
  resampled on the same sequences (/challenge-pr on #318, finding 4; was sqrt(SE² + SE²), before
  any dating). S, M, n0: no floor.
- A step's record is reused only if its inputs hash and code hash match (/challenge-pr on #318, 1).
- ρ's early peak is its first maximum on the window; its trough the first minimum after it.
- G2's measure: max over states of max|h_eager − h_sdpa| / max|h_eager| on the first 4 repeated
  sequences. G5's plant: 0.2 / (1 + exp(−(ln s − ln c) / 0.158)) (10 % → 90 % over a factor 2 in
  step), 0 at step 0. G6's SE: sqrt(SE(0)² + SE(143000)²).
- The noise rule's widened range [min − span, max + span] of the A0 seeds' d is applied at any span.
- Pythia's held-out text: each sequence decoded in Dodecahedron's tokenizer, re-encoded in Pythia's.
- I_text (fixed 2026-11-08, before any I_text output): Pythia's x, x′, x″ are the whole 2048-token
  row decoded and re-encoded, its first 256 tokens (as ICL's text; refused if shorter). Rows are
  kept as they are, `<|endoftext|>` included (real packed text; the share of x rows with one in
  their first 256 tokens is recorded in `gate.json`, beside). Its bootstrap resamples the 32 text
  triples jointly, from a second generator on the same seed (20261030), so I's SEs are unchanged by
  it and the two draw the same indices (unrelated sequence sets; harmless). Beside it, its top-5
  mean and argmax head. G7 also requires I_text and I_text_null populated. In `read`, G5's planted
  rise on I_text_null is described beside, not a gate (/challenge-pr on #324, finding 2; added before
  any batch output).

Tier 1, description (nothing here is registered).
    E="HF_HOME=<HDD_1TB>/mets_data/hf HF_HUB_OFFLINE=1 METS_DATA=<main>/data"
    $E python -m trackT_timaeus.t5k gate          # G1, G2, G3, G6, G7 (first output)
    $E python -m trackT_timaeus.t5k run --run A0  # every checkpoint of one run (resumable)
    $E python -m trackT_timaeus.t5k read          # G4, G5, the dating, the outcome table, the bridge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from trackA_force.a1a import DATA, REPO

R, T = 32, 256
DRAW_SEED = 20261107
HELD_LO, N_HELD, HELD_LEN = 64, 1024, 510   # tail rows 64–1087 (Blocked 139 (a), post hoc; was rows 0–63)
TEXT_LO = 1088                           # I_text: tail rows 1088–1119 (x), 1120–1151 (x′), 1152–1183 (x″)
LATE, EARLY = (490, 510), (40, 60)       # positions 490–509 and 40–59 (half-open)
N_BOOT, BOOT_SEED = 1000, 20261030
F_ON, F_HALF = 0.1, 0.5
I_FLOOR = 0.1                            # absolute, in I's units (design D2 (a))
SE_FLOOR = 4.0                           # P, B, ICL
CENS_FRAC = 0.1                          # censored: the last grid step carries ≥ 0.1 of the rise
NULL_MAX = 0.05                          # G4
PLANT_H, PLANT_W = 0.2, 0.158            # G5
PLANT_C = {"dodeca": (1072, 1465, 2002, 2735, 3738), "410m": (256, 512, 1000, 2000, 4000)}
G5_MIN = 38
G1_TOL, G2_TOL, G3_TOL = 1e-6, 1e-4, 1e-12
G6_I, G6_SE = 0.3, 4.0
ARMS = ("A0", "A0p", "A0pp", "A1", "A2", "A3")
A0S = ("A0", "A0p", "A0pp")
RELEASED, P410 = "dodecahedron-32m", "pythia-410m"
RUNS = ARMS + (RELEASED, P410)
END = {**{a: 4369 for a in ARMS}, RELEASED: 9536, P410: 4000}
OUT = DATA / "trackT" / "t5k" / "2026-11-07-n1024"   # the refused 64-sequence gate stays in 2026-11-07/
U2_DIR = "trackT/t5u2/2026-10-30"


class T5kRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5k.py", "trackT_timaeus/t5h.py", "trackT_timaeus/t5.py", "trackT_timaeus/t5c.py",
              "trackT_timaeus/t2.py", "trackT_timaeus/t5j.py", "trackA_force/a3.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the draws and the scorer

def draws(ordinary: np.ndarray, seed: int = DRAW_SEED, r: int = R, t: int = T) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b, c = (rng.choice(ordinary, (r, t)) for _ in range(3))
    return {"rep": np.concatenate([a, a], 1), "ctrl": np.concatenate([a, b], 1), "null": np.concatenate([a, c], 1)}


def text_sets(rows: np.ndarray, r: int = R, t: int = T) -> Dict[str, np.ndarray]:
    """I_text's sets from 3r token rows (each ≥ t long, in order x, x′, x″): [x, x], [x, x′], [x, x″]."""
    rows = np.asarray(rows)
    if rows.shape[0] != 3 * r or rows.shape[1] < t:
        raise T5kRefused(f"I_text needs {3 * r} rows of ≥ {t} tokens, got {rows.shape}")
    x, x1, x2 = (rows[k * r:(k + 1) * r, :t] for k in range(3))
    return {"rep_text": np.concatenate([x, x], 1), "ctrl_text": np.concatenate([x, x1], 1),
            "null_text": np.concatenate([x, x2], 1)}


def _net(rep: np.ndarray, ctrl: np.ndarray, null: np.ndarray, ix) -> tuple:
    """I and its null on the sequences ``ix``: max over heads of mean PM(rep) − PM(ctrl), and of null − ctrl."""
    c = ctrl[ix].mean(0)
    return (rep[ix].mean(0) - c).max(), (null[ix].mean(0) - c).max()


def pm_of(att, t: int = T):
    """Per (batch, head) prefix matching from attention (…, 2t, 2t): mean over j = 0 … t − 2 of the
    weight from query t + j to key j + 1 (torch or numpy-like tensors with ``diagonal``)."""
    d = att.diagonal(offset=-(t - 1), dim1=-2, dim2=-1)        # (q = t − 1 + i, k = i), i = 0 … t
    return d[..., 1:t].mean(-1)


def prev_of(att):
    return att.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)   # queries 1 … 2t − 1


def summarize(a: Dict[str, np.ndarray], seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> dict:
    """The step's statistics and bootstrap SEs from the per-sequence arrays."""
    rep, ctrl, null, prev = (np.asarray(a[k], float) for k in ("pm_rep", "pm_ctrl", "pm_null", "prev_ctrl"))
    dl = np.asarray(a["loss_rep"], float) - np.asarray(a["loss_ctrl"], float)
    icl = np.asarray(a["icl_late"], float) - np.asarray(a["icl_early"], float)

    def stat(ix):
        net = rep[ix].mean(0) - ctrl[ix].mean(0)
        return (net.max(), (null[ix].mean(0) - ctrl[ix].mean(0)).max(), prev[ix].mean(0).max(), dl[ix].mean())
    net = rep.mean(0) - ctrl.mean(0)
    I, In, P, B = stat(np.arange(len(rep)))
    rng = np.random.default_rng(seed)
    bs = np.array([stat(rng.integers(0, len(rep), len(rep))) for _ in range(n_boot)])
    bi = np.array([icl[rng.integers(0, len(icl), len(icl))].mean() for _ in range(n_boot)])
    h = int(np.argmax(net))
    out = {"I": float(I), "I_null": float(In), "P": float(P), "B": float(B), "ICL": float(icl.mean()),
           "I_top5": float(np.sort(net)[-5:].mean()), "argmax": h,
           "se": {"I": float(bs[:, 0].std(ddof=1)), "I_null": float(bs[:, 1].std(ddof=1)),
                  "P": float(bs[:, 2].std(ddof=1)), "B": float(bs[:, 3].std(ddof=1)), "ICL": float(bi.std(ddof=1))}}
    if "pm_rep_text" in a:                 # I_text, beside (Blocked 139 (a)); its own generator
        tr, tc, tn = (np.asarray(a[f"pm_{k}_text"], float) for k in ("rep", "ctrl", "null"))
        n = len(tr)
        It, Itn = _net(tr, tc, tn, np.arange(n))
        rt = np.random.default_rng(seed)
        bt = np.array([_net(tr, tc, tn, rt.integers(0, n, n)) for _ in range(n_boot)])
        nt = tr.mean(0) - tc.mean(0)
        out.update({"I_text": float(It), "I_text_null": float(Itn), "I_text_top5": float(np.sort(nt)[-5:].mean()),
                    "argmax_text": int(np.argmax(nt))})
        out["se"].update({"I_text": float(bt[:, 0].std(ddof=1)), "I_text_null": float(bt[:, 1].std(ddof=1))})
    return out


#: the beside statistics as functions of the per-sequence arrays and a resample (for the paired SE)
BESIDE = {"P": lambda a, ix: a["prev_ctrl"][ix].mean(0).max(),
          "B": lambda a, ix: (a["loss_rep"][ix] - a["loss_ctrl"][ix]).mean(),
          "ICL": lambda a, ix: (a["icl_late"][ix] - a["icl_early"][ix]).mean()}


def paired_se(stat, a_i: Dict[str, np.ndarray], a_j: Dict[str, np.ndarray], n: int,
              seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> float:
    """Bootstrap SE of stat(a_j) − stat(a_i), the same resampled sequences at both steps (they are
    read on the same sequences; /challenge-pr on #318, finding 4)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        d.append(stat(a_j, ix) - stat(a_i, ix))
    return float(np.std(d, ddof=1))


# ---------------------------------------------------------------- pure: the dating rule (D2 (a))

def _first(xs: Sequence[float], lo: int, hi: int, key) -> int:
    """Index of the first extreme of xs over [lo, hi] (key max)."""
    best = lo
    for i in range(lo, hi + 1):
        if key(xs[i]) > key(xs[best]):
            best = i
    return best


def date_from(steps: Sequence[int], x: Sequence[float], base: int, end: int, run_end: Optional[int] = None) -> dict:
    """on_f from index ``base`` over (base, end]: the first step with x − x(base) ≥ f · (peak − x(base));
    censored when the peak is the run's last index and the last grid step carries ≥ 0.1 of the rise."""
    run_end = end if run_end is None else run_end
    if end <= base:
        return {"base": steps[base], "peak": None, "rise": 0.0, "on_0.1": None, "on_0.5": None, "censored": False}
    pk = _first(x, base + 1, end, lambda v: v)
    rise = float(x[pk] - x[base])
    on = {}
    for f in (F_ON, F_HALF):
        on[f] = next((steps[i] for i in range(base + 1, end + 1) if rise > 0 and x[i] - x[base] >= f * rise), None)
    cens = bool(rise > 0 and pk == run_end and x[pk] - x[pk - 1] >= CENS_FRAC * rise)
    return {"base": steps[base], "peak": steps[pk], "rise": rise, "on_0.1": on[F_ON], "on_0.5": on[F_HALF],
            "censored": cens}


def date_series(steps: Sequence[int], x: Sequence[float], floor: Optional[float] = None,
                rise_se: Optional[Callable[[int, int], float]] = None) -> dict:
    """The generic rule: t = the minimum before the maximum, dated to the window's end. ``floor``
    (absolute) or ``rise_se`` (the paired bootstrap SE of x(peak) − x(t), from their indices; 4 SE)
    refuses a small rise."""
    end = len(steps) - 1
    pk = _first(x, 0, end, lambda v: v)
    t = _first(x, 0, pk, lambda v: -v)
    r = date_from(steps, x, t, end)
    thr = floor
    if rise_se is not None and r["peak"] is not None:
        thr = SE_FLOOR * rise_se(t, list(steps).index(r["peak"]))
    r["floor"] = thr
    r["refused"] = bool(thr is not None and r["rise"] < thr)
    return r


def rho_dates(steps: Sequence[int], rho: Sequence[float]) -> dict:
    """ρ_mid's fall (signed down, from its early peak to its trough) and rise (from the trough)."""
    end = len(steps) - 1
    p = _first(rho, 0, end, lambda v: v)
    t = _first(rho, p, end, lambda v: -v)
    fall = date_from(steps, [-v for v in rho], p, t, run_end=end)
    rise = date_from(steps, rho, t, end)
    return {"peak": steps[p], "trough": steps[t], "rho_peak": float(rho[p]), "rho_trough": float(rho[t]),
            "fall": fall, "rise": rise}


def phase(s: Optional[int], rd: dict) -> Optional[str]:
    if s is None:
        return None
    if s < rd["peak"]:
        return "early rise"
    if s <= rd["trough"]:
        return "fall"
    r10 = rd["rise"]["on_0.1"]
    return "trough" if r10 is None or s < r10 else "rise"


def gap(steps: Sequence[int], i_on: Optional[int], rho_on: Optional[int]) -> Optional[int]:
    if i_on is None or rho_on is None:
        return None
    return list(steps).index(i_on) - list(steps).index(rho_on)


def label(i_dates: dict, rho_rise: dict, d: Optional[int], by: int) -> str:
    """The outcome table (design, fixed): a label stands only where censoring cannot move it."""
    if i_dates["refused"] or i_dates["on_0.1"] is None:
        return f"no induction formed by {by}"
    if d is None:
        return "not ordered: ρ's rise not dated"
    ci, cr = i_dates["censored"], rho_rise["censored"]
    if d <= -2 and not ci:
        return "induction leads the rise"
    if -1 <= d <= 1 and not ci and not cr:
        return "induction with the rise"
    if d >= 2 and not cr:
        return "induction after the rise"
    return "censored, not ordered"


def noise(d: Dict[str, Optional[int]]) -> dict:
    """On the gap d: A0 seeds' span; arms inside A0's range widened by the span are not told apart."""
    a0 = [d[a] for a in A0S]
    if any(v is None for v in a0):
        return {"a0_d": a0, "span": None, "seed_dependent": None, "range": None, "not_told_apart": {},
                "not_run": f"A0 seed(s) without a gap: {', '.join(a for a in A0S if d[a] is None)}"}
    span = max(a0) - min(a0)
    lo, hi = min(a0) - span, max(a0) + span
    return {"a0_d": a0, "span": span, "seed_dependent": span >= 2, "range": [lo, hi],
            "not_told_apart": {a: (d[a] is not None and lo <= d[a] <= hi) for a in ARMS if a not in A0S}}


def plant(steps: Sequence[int], c: float, h: float = PLANT_H, w: float = PLANT_W) -> np.ndarray:
    return np.array([0.0 if s <= 0 else h / (1.0 + math.exp(-(math.log(s) - math.log(c)) / w)) for s in steps])


def g5_one(steps: Sequence[int], i_null: Sequence[float], c: int) -> dict:
    x = np.asarray(i_null, float) + plant(steps, c)
    r = date_series(steps, list(x), floor=I_FLOOR)
    ok = (not r["refused"]) and r["on_0.5"] is not None and abs(list(steps).index(r["on_0.5"]) - list(steps).index(c)) <= 1
    return {"centre": c, "on_0.5": r["on_0.5"], "refused": r["refused"], "pass": bool(ok)}


# ---------------------------------------------------------------- models and inputs

def _env():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackC_writeup import c2_aniso as c
    return c._torch_setup()


def ordinary_ids(tok) -> np.ndarray:
    special = set(tok.all_special_ids)
    special |= {i for i, t in getattr(tok, "added_tokens_decoder", {}).items() if getattr(t, "special", False)}
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    if eot is None or eot < 0:
        raise T5kRefused("no <|endoftext|> id")
    special.add(eot)
    return np.array([i for i in range(len(tok)) if i not in special], dtype=np.int64)


def tokenizer_of(run: str):
    from trackT_timaeus import t2
    return t2.p_tokenizer() if run == P410 else t2.tokenizer("32m")


def _tail_rows(run: str, lo: int, n: int, length: int, what: str) -> np.ndarray:
    """Tail rows lo … lo + n − 1, their first ``length`` tokens; Pythia's decoded in Dodecahedron's
    tokenizer and re-encoded in its own (refused if a row re-encodes shorter)."""
    from trackT_timaeus import t2, t5h
    tail = np.asarray(t5h.open_shard(t5h.ROOT)["tail"][lo:lo + n]).astype(np.int64)
    if len(tail) != n:
        raise T5kRefused(f"{what}: tail has {len(tail)} of rows {lo}–{lo + n - 1}")
    if run != P410:
        return tail[:, :length]
    dt, pt = t2.tokenizer("32m"), t2.p_tokenizer()
    out = []
    for row in tail:
        ids = pt(dt.decode(row.tolist()))["input_ids"]
        if len(ids) < length:
            raise T5kRefused(f"re-encoded {what} sequence has {len(ids)} < {length} tokens")
        out.append(ids[:length])
    return np.array(out, dtype=np.int64)


def held_out(run: str) -> np.ndarray:
    return _tail_rows(run, HELD_LO, N_HELD, HELD_LEN, "held-out")


def text_rows(run: str) -> np.ndarray:
    """I_text's 96 rows (x, x′, x″), each its first T tokens (Blocked 139 (a))."""
    return _tail_rows(run, TEXT_LO, 3 * R, T, "I_text")


def break_fraction(held: np.ndarray, eot: int, lo: int = EARLY[0], hi: int = LATE[1]) -> float:
    """Share of held-out rows with an <|endoftext|> in positions lo … hi − 1 (beside the gate; a break
    between the windows dilutes ICL, which more rows do not remove: /challenge-pr on #318, 6; #319, 2)."""
    return float((np.asarray(held)[:, lo:hi] == eot).any(1).mean())


def steps_of(run: str) -> tuple:
    from trackA_force import a3
    from trackT_timaeus import t5c, t5h
    if run in ARMS:
        return t5h.CKPT_STEPS
    return t5c.DODECA_STEPS if run == RELEASED else a3.STEPS


def load(run: str, s: int, device, dcfg=None):
    from trackT_timaeus import t2, t5, t5h
    if run in ARMS:
        ck = t5h.ROOT / run / "ckpt" / f"step{s}.safetensors"
        m = t5h.load_weights(ck, run, dcfg or t5h.dodeca_cfg(t5.DEFAULT_ROOTS), device)
        return m, {"ckpt_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()[:16]}
    repo = t2.MODELS["32m"] if run == RELEASED else "EleutherAI/pythia-410m"
    m = t5.load_real(repo, f"step{s}", t5.DEFAULT_ROOTS, device)
    return m, {"commit": t2.commit_of(m)}


def forward(model, sets: Dict[str, np.ndarray], held: np.ndarray, device, bs: int = 4) -> Dict[str, np.ndarray]:
    """Per-sequence arrays: PM per head on each set, the previous-token score on the control, the
    second half's loss on rep and ctrl, and the held-out losses at 490–509 and 40–59."""
    import torch
    import torch.nn.functional as F
    out: Dict[str, List] = {}
    for name in ("rep_text", "ctrl_text", "null_text"):          # I_text: PM only
        if name not in sets:
            continue
        pm = []
        for ch in torch.from_numpy(sets[name]).split(bs):
            with torch.no_grad():
                o = model(input_ids=ch.to(device), output_attentions=True)
            if len(o.attentions) != model.config.num_hidden_layers or o.attentions[0] is None:
                raise T5kRefused("attentions not returned (eager attention required)")
            pm.append(torch.cat([pm_of(a.float()) for a in o.attentions], 1).cpu().numpy())
            del o
        out[f"pm_{name}"] = np.concatenate(pm)
    for name in ("rep", "ctrl", "null"):
        pm, pv, ls = [], [], []
        for ch in torch.from_numpy(sets[name]).split(bs):
            ids = ch.to(device)
            with torch.no_grad():
                o = model(input_ids=ids, output_attentions=True)
            if len(o.attentions) != model.config.num_hidden_layers or o.attentions[0] is None:
                raise T5kRefused("attentions not returned (eager attention required)")
            pm.append(torch.cat([pm_of(a.float()) for a in o.attentions], 1).cpu().numpy())
            pv.append(torch.cat([prev_of(a.float()) for a in o.attentions], 1).cpu().numpy())
            lt = F.cross_entropy(o.logits.float()[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
            ls.append(lt[:, T:2 * T - 1].mean(1).cpu().numpy())        # positions T + 1 … 2T − 1
            del o
        out[f"pm_{name}"], out[f"loss_{name}"] = np.concatenate(pm), np.concatenate(ls)
        if name == "ctrl":
            out["prev_ctrl"] = np.concatenate(pv)
    late, early = [], []
    for ch in torch.from_numpy(held).split(8):
        ids = ch.to(device)
        with torch.no_grad():
            lg = model(input_ids=ids).logits.float()
        lt = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")   # lt[:, p − 1] = loss at p
        late.append(lt[:, LATE[0] - 1:LATE[1] - 1].mean(1).cpu().numpy())
        early.append(lt[:, EARLY[0] - 1:EARLY[1] - 1].mean(1).cpu().numpy())
    out["icl_late"], out["icl_early"] = np.concatenate(late), np.concatenate(early)
    return {k: np.asarray(v) for k, v in out.items()}


def _jdump(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(p)


class Inputs:
    def __init__(self, run: str):
        tok = tokenizer_of(run)
        self.ordinary = ordinary_ids(tok)
        self.sets = {**draws(self.ordinary), **text_sets(text_rows(run))}
        self.held = held_out(run)
        self.eot = int(tok.convert_tokens_to_ids("<|endoftext|>"))
        self.sha = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in
                                           (*self.sets.values(), self.held))).hexdigest()[:12]


def one_step(run: str, s: int, inp: Inputs, device, dcfg=None, force: bool = False) -> dict:
    import torch
    d = OUT / run
    p = d / f"step{s}.json"
    if p.exists() and not force:
        rec = json.loads(p.read_text())
        if rec.get("inputs_sha") == inp.sha and rec.get("code") == code_sha():
            return rec                     # reused only on the same inputs and code (/challenge-pr on #318, 1)
    m, src = load(run, s, device, dcfg)
    a = forward(m, inp.sets, inp.held, device)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"step{s}.npz", **a)
    rec = {"run": run, "step": s, **summarize(a), "layers": m.config.num_hidden_layers,
           "heads": m.config.num_attention_heads, "inputs_sha": inp.sha, "n_ordinary": int(len(inp.ordinary)),
           "code": code_sha(), **src}
    h = rec["argmax"]
    rec["argmax_lh"] = [h // rec["heads"], h % rec["heads"]]
    _jdump(p, rec)
    del m
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- commands

def g3() -> dict:
    """The scorer on synthetic attention, float64: a planted head gives PM 1; uniform causal gives net 0."""
    import torch
    n = 2 * T
    plant_ = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for j in range(T - 1):
        plant_[0, 0, T + j, j + 1] = 1.0
    uni = torch.tril(torch.ones(n, n, dtype=torch.float64)) / torch.arange(1, n + 1, dtype=torch.float64)[:, None]
    uni = uni[None, None]
    pm_p = float(pm_of(plant_)[0, 0])
    pm_u = float(pm_of(uni)[0, 0])
    want_u = float(np.mean([1.0 / (T + j + 1) for j in range(T - 1)]))
    # uniform causal attention on both sets through `summarize`: net I = 0
    u = np.full((R, 4), pm_u)
    z = np.zeros(R)
    net_u = summarize({"pm_rep": u, "pm_ctrl": u, "pm_null": u, "prev_ctrl": u, "loss_rep": z, "loss_ctrl": z,
                       "icl_late": z, "icl_early": z}, n_boot=10)["I"]
    return {"planted_pm": pm_p, "uniform_pm": pm_u, "uniform_pm_analytic": want_u, "uniform_net_I": net_u,
            "pass": abs(pm_p - 1) <= G3_TOL and abs(pm_u - want_u) <= G3_TOL and abs(net_u) <= G3_TOL}


def _load_sdpa(run: str, s: int, device, dcfg):
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM
    from trackT_timaeus import t5, t5h
    cfg = t5.config_of(t5h.SHAPE, "llama", t5h.ARMS[run][2], dcfg)
    cfg._attn_implementation = "sdpa"
    with torch.device(device):
        m = LlamaForCausalLM(cfg)
    m.load_state_dict(load_file(str(t5h.ROOT / run / "ckpt" / f"step{s}.safetensors"), device=str(device)), strict=False)
    m.tie_weights()
    return m.to(torch.float32).eval()


def gate(a) -> int:
    torch, device = _env()
    from trackT_timaeus import t2, t5, t5c, t5h
    dcfg = t5h.dodeca_cfg(t5.DEFAULT_ROOTS)
    res: dict = {"code": code_sha(), "design": "trackT_timaeus/design-T5p.md (frozen 2026-11-07)"}
    # G1: the read path
    tok = t2.tokenizer("32m")
    u2 = DATA / U2_DIR / RELEASED
    g1 = {}
    for run, s, ref in (("A0", 671, t5h.ROOT / "A0" / "read" / "step671.json"),
                        ("A0", 4369, t5h.ROOT / "A0" / "read" / "step4369.json"),
                        (RELEASED, 4369, u2 / "step4369.json")):
        m, _ = load(run, s, device, dcfg)
        got = t5c.read_model(m, tok, device)["rho_mid"]
        want = json.loads(ref.read_text())["rho_mid"]
        g1[f"{run}/{s}"] = {"rho_mid": got, "record": want, "diff": abs(got - want)}
        del m
    res["G1"] = {"cells": g1, "pass": all(v["diff"] <= G1_TOL for v in g1.values())}
    # G2: eager against sdpa
    inp = Inputs("A0")
    me, _ = load("A0", 4369, device, dcfg)
    ms = _load_sdpa("A0", 4369, device, dcfg)
    ids = torch.from_numpy(inp.sets["rep"][:4]).to(device)
    with torch.no_grad():
        he = me(input_ids=ids, output_hidden_states=True).hidden_states
        hs = ms(input_ids=ids, output_hidden_states=True).hidden_states
    rel = max(float((x - y).abs().max() / x.abs().max()) for x, y in zip(he, hs))
    res["G2"] = {"max_rel": rel, "sdpa_class": type(ms.model.layers[0].self_attn).__name__, "pass": rel <= G2_TOL}
    del me, ms
    # G3
    res["G3"] = g3()
    # G7: the first output
    r0 = one_step("A0", 4369, inp, device, dcfg, force=a.force)
    keys = ("I", "I_null", "P", "B", "ICL", "argmax", "I_text", "I_text_null", "argmax_text")
    res["G7"] = {"record": {k: r0.get(k) for k in keys}, "inputs_sha": r0["inputs_sha"],
                 "pass": all(r0.get(k) is not None and np.isfinite(r0[k]) for k in keys)}
    # G6: the positive control
    inp_p = Inputs(P410)
    r_end = one_step(P410, 143000, inp_p, device, force=a.force)
    r_0 = one_step(P410, 0, inp_p, device, force=a.force)
    se = math.hypot(r_end["se"]["ICL"], r_0["se"]["ICL"])
    res["G6"] = {"I_143000": r_end["I"], "ICL_0": r_0["ICL"], "ICL_143000": r_end["ICL"], "se": se,
                 "argmax_lh": r_end["argmax_lh"],
                 "pass": r_end["I"] >= G6_I and r_0["ICL"] - r_end["ICL"] >= G6_SE * se}
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    res["ICL_breaks"] = {"rows": [HELD_LO, HELD_LO + N_HELD], "positions": [EARLY[0], LATE[1]],
                         "fraction": break_fraction(inp.held, eot), "beside": "not a gate"}
    res["I_text_breaks"] = {"rows": [TEXT_LO, TEXT_LO + R], "positions": [0, T],
                            "fraction": {r: break_fraction(i.sets["rep_text"][:, :T], i.eot, 0, T)
                                         for r, i in (("dodecahedron", inp), (P410, inp_p))},
                            "beside": "share of x rows with <|endoftext|> in their first T tokens; not a gate"}
    res["pass"] = all(res[g]["pass"] for g in ("G1", "G2", "G3", "G6", "G7"))
    _jdump(OUT / "gate.json", res)
    print(json.dumps({g: res[g]["pass"] for g in ("G1", "G2", "G3", "G6", "G7")} | {"pass": res["pass"]}))
    return 0 if res["pass"] else 3


def _require_gate() -> dict:
    p = OUT / "gate.json"
    if not p.exists():
        raise T5kRefused("no gate.json: run `gate` first")
    g = json.loads(p.read_text())
    if not g["pass"]:
        raise T5kRefused(f"gate failed: {[k for k in ('G1', 'G2', 'G3', 'G6', 'G7') if not g[k]['pass']]}")
    return g


def run(a) -> int:
    torch, device = _env()
    from trackT_timaeus import t5, t5h
    _require_gate()
    dcfg = t5h.dodeca_cfg(t5.DEFAULT_ROOTS) if a.run != P410 else None
    inp = Inputs(a.run)
    t0 = time.time()
    for s in steps_of(a.run):
        r = one_step(a.run, s, inp, device, dcfg, force=a.force)
        print(json.dumps({"run": a.run, "step": s, "I": round(r["I"], 4), "I_null": round(r["I_null"], 4),
                          "I_text": round(r["I_text"], 4), "I_text_null": round(r["I_text_null"], 4),
                          "P": round(r["P"], 3), "B": round(r["B"], 3), "ICL": round(r["ICL"], 3),
                          "t": round(time.time() - t0)}), flush=True)
    return 0


# ---------------------------------------------------------------- the reading

def _series(run: str) -> dict:
    from trackT_timaeus import t5h, t5j
    steps = steps_of(run)
    recs = {s: json.loads((OUT / run / f"step{s}.json").read_text()) for s in steps}
    codes = {r["code"] for r in recs.values()}
    shas = {r["inputs_sha"] for r in recs.values()}
    if len(shas) != 1:
        raise T5kRefused(f"{run}: inputs differ across steps {shas}")
    if run in ARMS:
        reads = t5h.records(t5h.ROOT, run)
        on = {s: json.loads((t5j.OUT / run / "onset" / f"step{s}.json").read_text()) for s in steps}
        extra = {"S": [on[s]["S"] for s in steps], "M": [on[s]["massive"]["M"] for s in steps]}
    else:
        u2 = DATA / U2_DIR / run
        reads = {s: json.loads((u2 / f"step{s}.json").read_text()) for s in steps}
        extra = {}
    ser = {k: [recs[s][k] for s in steps] for k in ("I", "I_null", "P", "B", "ICL", "I_top5",
                                                     "I_text", "I_text_null", "I_text_top5")}
    se = {k: [recs[s]["se"][k] for s in steps] for k in ("I", "I_null", "P", "B", "ICL", "I_text", "I_text_null")}
    ser.update(extra)
    ser["rho_mid"] = [reads[s]["rho_mid"] for s in steps]
    ser["n0_mid"] = [float(np.median(reads[s]["n0"][1:-1])) for s in steps]
    ser["argmax_lh"] = [recs[s]["argmax_lh"] for s in steps]
    ser["argmax_text"] = [recs[s]["argmax_text"] for s in steps]
    return {"steps": list(steps), "series": ser, "se": se, "codes": sorted(codes), "inputs_sha": shas.pop()}


def _pair(run: str, steps: Sequence[int], k: str) -> Callable[[int, int], float]:
    def f(i: int, j: int) -> float:
        a_i, a_j = (dict(np.load(OUT / run / f"step{steps[q]}.npz")) for q in (i, j))
        n = len(a_i["icl_late"] if k == "ICL" else a_i["loss_rep"])
        return paired_se(BESIDE[k], a_i, a_j, n)
    return f


def read_run(run: str, data: dict, end: int) -> dict:
    """Every dated series of one run on its window to ``end``, the phases, d and the label."""
    steps = [s for s in data["steps"] if s <= end]
    n = len(steps)
    x = {k: v[:n] for k, v in data["series"].items()}
    se = {k: v[:n] for k, v in data["se"].items()}
    rd = rho_dates(steps, x["rho_mid"])
    out = {"end": end, "rho": rd,
           "I": date_series(steps, x["I"], floor=I_FLOOR),
           "I_null": date_series(steps, x["I_null"], floor=I_FLOOR),
           "P": date_series(steps, x["P"], rise_se=_pair(run, steps, "P")),
           "B": date_series(steps, [-v for v in x["B"]], rise_se=_pair(run, steps, "B")),
           "ICL": date_series(steps, [-v for v in x["ICL"]], rise_se=_pair(run, steps, "ICL")),
           # I_text, beside (Blocked 139 (a)): D2's rule and floor, no label
           "I_text": date_series(steps, x["I_text"], floor=I_FLOOR),
           "I_text_null": date_series(steps, x["I_text_null"], floor=I_FLOOR)}
    for k in ("S", "M", "n0_mid"):
        if k in x:
            out[k] = date_series(steps, x[k])
    out["phase"] = {k: {"on_0.1": phase(out[k]["on_0.1"], rd), "on_0.5": phase(out[k]["on_0.5"], rd)}
                    for k in ("I", "P", "B", "ICL", "I_text", "S", "M", "n0_mid") if k in out}
    out["d"] = {k: gap(steps, out[k]["on_0.1"], rd["rise"]["on_0.1"]) for k in out["phase"]}
    if run != P410:
        out["label"] = label(out["I"], rd["rise"], out["d"]["I"], end)
    out["max_I_null"] = float(max(x["I_null"]))
    out["max_I_text_null"] = float(max(x["I_text_null"]))
    return out


def read(a) -> int:
    from trackA_force import a3
    g = _require_gate()
    data = {r: _series(r) for r in RUNS}
    res: dict = {"design": "trackT_timaeus/design-T5p.md (frozen 2026-11-07)", "code": code_sha(), "gate": g,
                 "inputs_sha": {r: d["inputs_sha"] for r, d in data.items()},
                 "producer_codes": {r: d["codes"] for r, d in data.items()}, "runs": {}}
    for r in RUNS:
        res["runs"][r] = {str(END[r]): read_run(r, data[r], END[r])}
    res["runs"][RELEASED]["4369"] = read_run(RELEASED, data[RELEASED], 4369)
    # G4: the planted null on every run (dated to each run's end), max I_null ≤ 0.05 at every recorded step
    g4 = {r: {"refused": res["runs"][r][str(END[r])]["I_null"]["refused"],
              "max_all_steps": float(max(data[r]["series"]["I_null"]))} for r in RUNS}
    res["G4"] = {"runs": g4, "pass": all(v["refused"] and v["max_all_steps"] <= NULL_MAX for v in g4.values())}
    # I_text's null, beside (Blocked 139 (a)): described, not a gate, and gates nothing
    tn = {r: {"refused": res["runs"][r][str(END[r])]["I_text_null"]["refused"],
              "max_all_steps": float(max(data[r]["series"]["I_text_null"]))} for r in RUNS}
    res["I_text_null_beside"] = {"runs": tn, "all_within": all(v["refused"] and v["max_all_steps"] <= NULL_MAX
                                                               for v in tn.values()), "gate": False}
    # G5's planted rise on I_text's null, beside (/challenge-pr on #324, finding 2): I_text's noise is
    # larger than I's, so the rule's dating on it is described; not a gate, gates nothing
    tp = []
    for r in RUNS:
        steps = [s for s in data[r]["steps"] if s <= END[r]]
        xn = data[r]["series"]["I_text_null"][:len(steps)]
        for c in PLANT_C["410m" if r == P410 else "dodeca"]:
            tp.append({"run": r, **g5_one(steps, xn, c)})
    res["I_text_planted_beside"] = {"cells": tp, "n_pass": sum(v["pass"] for v in tp), "n": len(tp),
                                    "n_refused": sum(v["refused"] for v in tp), "gate": False}
    # G5: a planted rise on each run's I_null
    g5 = []
    for r in RUNS:
        steps = [s for s in data[r]["steps"] if s <= END[r]]
        xn = data[r]["series"]["I_null"][:len(steps)]
        for c in PLANT_C["410m" if r == P410 else "dodeca"]:
            g5.append({"run": r, **g5_one(steps, xn, c)})
    n_ok = sum(v["pass"] for v in g5)
    res["G5"] = {"cells": g5, "n_pass": n_ok, "n": len(g5),
                 "pass": n_ok >= G5_MIN and not any(v["refused"] for v in g5)}
    # the bridge (D3 (a)): C2's locator on 410m's I, reference to 4000; not a gate
    vi = data[P410]["series"]["I"]
    if list(data[P410]["steps"]) != list(a3.STEPS):
        raise T5kRefused("410m's grid is not C2's")
    br = a3.d4a(np.asarray(vi))
    res["bridge"] = {**br, "stands": br["step"] == 1000}
    if not (res["G4"]["pass"] and res["G5"]["pass"]):
        res["refused"] = "G4 or G5 failed: no outcome table (design: a G1–G7 failure refuses the reading)"
        res["outcome"] = None
    else:
        d = {r: res["runs"][r]["4369"]["d"]["I"] for r in ARMS}
        res["outcome"] = {"labels": {r: res["runs"][r]["4369"]["label"] for r in ARMS},
                          "released": {e: res["runs"][RELEASED][e]["label"] for e in ("4369", "9536")},
                          "d": d, "noise": noise(d)}
    _jdump(OUT / "reading.json", res)
    _print(res)
    return 0


def _print(res: dict) -> None:
    print(json.dumps({"G4": res["G4"]["pass"], "G5": f"{res['G5']['n_pass']}/{res['G5']['n']}",
                      "bridge": res["bridge"].get("step"), "stands": res["bridge"]["stands"]}))
    print("run          end   I_on.1 I_on.5 cens  rise  | rho fall½ trough rise.1 rise½ cens | d   phase.1   label")
    for r, ends in res["runs"].items():
        for e, o in ends.items():
            i, rd = o["I"], o["rho"]
            print(f"{r:12s} {e:>5s} {str(i['on_0.1']):>6s} {str(i['on_0.5']):>6s} {str(i['censored'])[0]:>4s} "
                  f"{i['rise']:.3f} | {str(rd['fall']['on_0.5']):>8s} {rd['trough']:>6d} {str(rd['rise']['on_0.1']):>6s} "
                  f"{str(rd['rise']['on_0.5']):>5s} {str(rd['rise']['censored'])[0]:>4s} | {str(o['d']['I']):>3s} "
                  f"{str(o['phase']['I']['on_0.1']):9s} {o.get('label', '')}")
    if res["outcome"]:
        print(json.dumps(res["outcome"]["noise"]))
    tp = res["I_text_planted_beside"]
    print(f"I_text's planted rise (beside, not a gate): {tp['n_pass']}/{tp['n']} dated within a step, "
          f"{tp['n_refused']} refused")
    print("I_text, beside (no label): run  end  on.1  on.5  cens  rise  phase.1  d  | max I_text_null")
    for r, ends in res["runs"].items():
        for e, o in ends.items():
            i = o["I_text"]
            print(f"  {r:12s} {e:>5s} {str(i['on_0.1']):>6s} {str(i['on_0.5']):>6s} {str(i['censored'])[0]:>4s} "
                  f"{i['rise']:.3f} {'refused' if i['refused'] else '':7s} {str(o['phase']['I_text']['on_0.1']):9s} "
                  f"{str(o['d']['I_text']):>3s} | {o['max_I_text_null']:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--run", required=True, choices=RUNS)
    r.add_argument("--force", action="store_true")
    sub.add_parser("read")
    a = ap.parse_args(argv)
    return {"gate": gate, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
