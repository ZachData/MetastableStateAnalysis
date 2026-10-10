"""OV1s: the sign at a fixed subspace (`p10_cluster_function/design-10.md` "OV1s", fixed before any pass).

OV1's cut (`tools/run/p10_ov1_cut.py`) with five arms: ``base``; ``att`` (each head's
``K = W_O W_V diag(γ)`` → ``S₊``) against ``neg`` (``−S₊``: the same eigenvectors and size, sign
reversed), both written in the weights; ``norep`` (``K + S₋``) against ``noatt`` (``K − S₊``),
which have rank up to 128 and so keep the base weights and add ``Σ_h Σ_j P^h_ij Δ_h x̂_j`` to each
block's attention output by a hook. Every stored c2a group's kind is kept, so c3x's groups can be
set against the rest. Tier 1: exploratory, unregistered.
    python tools/run/p10_ov1s_sign.py run --labels <R9 source> --ov1 <OV1 out> --out <dir> [--steps ...]
    python tools/run/p10_ov1s_sign.py report --out <dir>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from tools.run import p10_label_source as ls
from tools.run import p10_ov1_cut as ov
from tools.run.p10_ov1_cut import BANDS, LAYERS, PASSAGES, PRIMARY_STEPS, STEPS, OV1Error

ARMS = ("base", "att", "neg", "norep", "noatt")
WEIGHT_ARMS = ("base", "att", "neg")
HOOK_ARMS = ("norep", "noatt", "att_hook")               # att_hook: the first step's check only
LABELS = {"S1": ("att", "neg", "c3x"), "S2": ("norep", "noatt", "c3x"),
          "S3a": ("att", "neg", "x-rest"), "S3b": ("norep", "noatt", "x-rest")}
NAMES = {"c3x": ("merges", "separates"), "x-rest": ("c3x more", "c3x less")}
HOOK_TOL = 1e-3                                          # att by hook vs by weights, ‖Δhs‖/‖hs‖ (placed)
AGREE = 0.98                                             # kinds agreeing: hook check, OV1 reproduction (placed)
DISSOLVE = 0.2                                           # flipped arm keeps < this → said beside (OV1's)
FIRST = (16000, "wiki_paragraph")
OV1_READ = {(16000, "L9-16"), (16000, "L17-24")}         # where OV1 read "repulsion keeps them apart"


# ---------------------------------------------------------------- the arms

def pick(hd: Dict, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(U, λ)`` written into the head's weights under a weight arm."""
    lam, U = hd["lam"], hd["U"]
    pos = np.flatnonzero(lam > 0)
    if arm == "att":
        return U[:, pos], lam[pos]
    if arm == "neg":
        return U[:, pos], -lam[pos]
    raise OV1Error(f"{arm} is not a weight arm")


def delta(hd: Dict, arm: str, WO: np.ndarray, WVg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(L, R)`` with ``Δ = L Rᵀ`` added to the head's ``K`` under a hook arm (``WVg = W_V diag(γ)``)."""
    lam, U = hd["lam"], hd["U"]
    pos, neg = np.flatnonzero(lam > 0), np.flatnonzero(lam < 0)
    if arm == "norep":                                    # K + S₋, S₋ = U₋ |λ₋| U₋ᵀ
        return U[:, neg] * np.abs(lam[neg])[None, :], U[:, neg]
    if arm == "noatt":                                    # K − S₊
        return -U[:, pos] * lam[pos][None, :], U[:, pos]
    if arm == "att_hook":                                 # K + (S₊ − K) = S₊
        return (np.hstack([U[:, pos] * lam[pos][None, :], -WO]),
                np.hstack([U[:, pos], WVg.T]))
    raise OV1Error(f"{arm} is not a hook arm")


class SignCutter(ov.Cutter):
    """OV1's cutter with ``neg`` in the weights and the hook arms' added term."""

    def apply_weights(self, arm: str, step: int) -> Dict:
        """``att`` / ``neg`` written as OV1 writes its arms (each head's ``c`` to the dense bias)."""
        torch = self.torch
        self.restore()
        if arm == "base":
            return {"readback": 0.0}
        worst, pairs = 0.0, 0
        with torch.no_grad():
            for l, (lay, row) in enumerate(zip(self.layers, self.heads)):
                g, b = self.ln[l]
                att = lay.attention
                W = att.query_key_value.weight
                dev, dt = W.device, W.dtype
                qkv_w = W.view(self.H, 3, self.k, -1)
                qkv_b = att.query_key_value.bias.view(self.H, 3, self.k)
                c_sum = np.zeros(W.shape[1])
                for h, hd in enumerate(row):
                    U, lam = pick(hd, arm)
                    if (arm == "att" and (lam < 0).any()) or (arm == "neg" and (lam > 0).any()):
                        raise OV1Error(f"step {step} {arm} L{l} h{h}: a pair of the wrong sign")
                    WV, bV, WO = ov.write_back(U, lam, g, b, self.k)
                    qkv_w[h, 2] = torch.as_tensor(WV, device=dev, dtype=dt)
                    qkv_b[h, 2] = torch.as_tensor(bV, device=dev, dtype=dt)
                    att.dense.weight[:, h * self.k:(h + 1) * self.k] = torch.as_tensor(WO, device=dev, dtype=dt)
                    c_sum += hd["c"]
                    err = ov.readback_err(qkv_w[h, 2].double().cpu().numpy(),
                                          att.dense.weight[:, h * self.k:(h + 1) * self.k].double().cpu().numpy(),
                                          g, U, lam)
                    worst, pairs = max(worst, err), pairs + lam.size
                att.dense.bias.copy_(self.orig[l]["o_b"] + torch.as_tensor(c_sum, device=dev, dtype=dt))
        if worst > ov.READBACK_TOL:
            raise OV1Error(f"step {step} {arm}: written map off S' by {worst:.1e} of ‖S'‖")
        return {"readback": worst, "pairs": pairs}

    def factors(self, arm: str):
        """Per layer, ``(L, R)`` as (H, d, r) tensors on the model's device (zero-padded to r)."""
        torch = self.torch
        out = []
        for l, (lay, row) in enumerate(zip(self.layers, self.heads)):
            g, _ = self.ln[l]
            o = self.orig[l]
            qkv = o["qkv_w"].double().cpu().numpy().reshape(self.H, 3, self.k, -1)
            ow = o["o_w"].double().cpu().numpy()
            LR = [delta(hd, arm, ow[:, h * self.k:(h + 1) * self.k], qkv[h, 2] * g[None, :])
                  for h, hd in enumerate(row)]
            r = max(1, max(L.shape[1] for L, _ in LR))
            d = LR[0][0].shape[0]
            Ls, Rs = np.zeros((self.H, d, r)), np.zeros((self.H, d, r))
            for h, (L, R) in enumerate(LR):
                Ls[h, :, :L.shape[1]], Rs[h, :, :R.shape[1]] = L, R
            W = lay.attention.query_key_value.weight
            out.append((torch.as_tensor(Ls, device=W.device, dtype=W.dtype),
                        torch.as_tensor(Rs, device=W.device, dtype=W.dtype)))
        return out


class DeltaHook:
    """Adds ``Σ_h P^h x̂ Δ_hᵀ`` to every block's attention output (``Δ_h = L_h R_hᵀ``), reading each
    head's attention weights through ``u2_attn.Hooked``'s ``on_attn`` and ``x̂`` off LN1's input."""

    def __init__(self, hooked, factors):
        import torch
        self.hk, self.factors, self.handles, self.cap, self.on_p = hooked, factors, [], {}, {}
        for l, layer in enumerate(hooked.layers):
            ln = layer.input_layernorm

            def keep_xhat(m, i, o, _l=l, _eps=ln.eps):
                x = i[0][0]
                mu = x.mean(-1, keepdim=True)
                self.cap[("x", _l)] = (x - mu) / (x.var(-1, keepdim=True, unbiased=False) + _eps).sqrt()

            def keep_p(v, w, _l=l):
                self.cap[("p", _l)] = w[0]

            def add(m, i, o, _l=l):
                L, R = self.factors[_l]
                xr = torch.einsum("nd,hdr->hnr", self.cap.pop(("x", _l)), R)
                y = torch.einsum("hnm,hmr->hnr", self.cap.pop(("p", _l)), xr)
                extra = torch.einsum("hnr,hdr->nd", y, L)
                return (o[0] + extra[None].to(o[0].dtype),) + tuple(o[1:])

            self.handles.append(ln.register_forward_hook(keep_xhat))
            hooked.on_attn[l].append(keep_p)
            self.on_p[l] = keep_p
            self.handles.append(layer.attention.register_forward_hook(add))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        for l, f in self.on_p.items():
            self.hk.on_attn[l].remove(f)
        self.cap.clear()


# ---------------------------------------------------------------- the readout

def read_passage(by_arm: Dict[str, Dict[int, Tuple[np.ndarray, float]]], src: Path, step: int, passage: str,
                 cache: Dict) -> Dict:
    """Per layer: base against the stored c2a (refused if it differs), then every arm's kind for
    every stored c2a group (c3x's listed)."""
    layers, refused = {}, {}
    for L in LAYERS:
        _, stored = ls.load_column(src, f"step{step}", passage, L, "c2a", cache)
        _, labx = ls.load_column(src, f"step{step}", passage, L, "c3x", cache)
        ids = sorted(int(g) for g in np.unique(stored[stored >= 0]))
        x = sorted(int(g) for g in np.unique(labx[labx >= 0]))
        for g in x:
            if not np.array_equal(np.flatnonzero(labx == g), np.flatnonzero(stored == g)):
                raise OV1Error(f"step {step} {passage} L{L}: c3x group {g} is not c2a's")
        if not ov.same_partition(by_arm["base"][L][0], stored):
            refused[str(L)] = "base c2a differs from the stored c2a"
            continue
        cell = {"c3x": x, "groups": ids, "arms": {}}
        for arm, labs in by_arm.items():
            lab, rel = labs[L]
            k = ov.kinds(stored, lab, ids)
            cell["arms"][arm] = {**ov.layer_summary(lab), "rel": rel, "kind": {str(g): k[g][0] for g in ids}}
        layers[str(L)] = cell
    return {"layers": layers, "refused_layers": refused}


def kind_agreement(a: Dict, b: Dict, arm_a: str, arm_b: str, only_c3x: bool) -> Tuple[int, float]:
    """``(records, share with the same kind)`` between two records' layers."""
    n = same = 0
    for L, ca in a["layers"].items():
        cb = b["layers"].get(L)
        if cb is None:
            continue
        for g in (ca["c3x"] if only_c3x else ca["groups"]):
            ka, kb = ca["arms"][arm_a]["kind"].get(str(g)), cb["arms"][arm_b]["kind"].get(str(g))
            if ka is None or kb is None:
                continue
            n, same = n + 1, same + (ka == kb)
    return n, (same / n if n else float("nan"))


def check_populated(rec: Dict) -> None:
    """The rule's first-record check."""
    lay = rec["layers"]
    if not lay:
        raise SystemExit("refusing: first record has no readable layer")
    n_x = sum(len(c["c3x"]) for c in lay.values())
    n_all = sum(len(c["groups"]) for c in lay.values())
    if n_x == 0 or n_all == n_x:
        raise SystemExit(f"refusing: first record has {n_x} c3x and {n_all - n_x} other records")
    for arm in ARMS[1:]:
        if max(c["arms"][arm]["rel"] for c in lay.values()) <= ov.OFF_BASE:
            raise SystemExit(f"refusing: first record, {arm}'s hs is not off base")
        ks = [k for c in lay.values() for k in c["arms"][arm]["kind"].values()]
        if len(ks) != n_all:
            raise SystemExit(f"refusing: first record, {arm}: {len(ks)} kinds for {n_all} records")
    if all(k == "stable" for c in lay.values() for k in c["arms"]["att"]["kind"].values()):
        raise SystemExit("refusing: first record, att keeps every record stable")


def check_hook(rec: Dict, by_arm: Dict, hs_rel: Dict[int, float]) -> Dict:
    """The hook path on the real model: att by hook against att in the weights."""
    worst = max(hs_rel.values())
    n, agree = kind_agreement(rec, rec, "att", "att_hook", only_c3x=False)
    out = {"worst_rel": worst, "records": n, "agree": agree}
    if not worst <= HOOK_TOL or not agree >= AGREE:
        raise SystemExit(f"refusing: att by hook vs by weights: ‖Δhs‖/‖hs‖ {worst:.1e} (≤ {HOOK_TOL}), "
                         f"kinds agree {agree:.3f} of {n} (≥ {AGREE})")
    return out


def check_ov1(rec: Dict, ov1: Path) -> Dict:
    """Reproduction: att's c3x kinds against OV1's stored record."""
    f = ov1 / "records" / f"step{rec['step']}_{rec['passage']}.json"
    if not f.exists():
        raise SystemExit(f"refusing: no OV1 record {f}")
    old = json.loads(f.read_text())
    n, agree = kind_agreement(rec, old, "att", "att", only_c3x=True)
    out = {"ov1_record": str(f), "records": n, "agree": agree, "ov1_code": old.get("code")}
    if not n or not agree >= AGREE:
        raise SystemExit(f"refusing: att's c3x kinds agree with OV1's for {agree:.3f} of {n} (≥ {AGREE})")
    return out


# ---------------------------------------------------------------- the batch

def run(a) -> int:
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    from p1e_energy_field import u2_attn as ua
    from tools.run import p10_e1_energy as e1
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the rule's pass is the GPU's)")
    code = ov.code_sha()
    src = Path(a.labels)
    meta = {"rule": "design-10.md \"OV1s\"", "code": code, "labels": str(src), "ov1": str(a.ov1),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()[:16],
            "arms": ARMS, "hook_tol": HOOK_TOL, "agree": AGREE}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    steps = [FIRST[0]] + [s for s in (a.steps or STEPS) if s != FIRST[0]]
    first_done = (a.out / "records" / f"step{FIRST[0]}_{FIRST[1]}.json").exists()
    for step in steps:
        cache: Dict = {}
        d = ls.load_step(src, f"step{step}")
        todo = [p for p in PASSAGES if not (a.out / "records" / f"step{step}_{p}.json").exists()]
        if step == FIRST[0]:
            todo = sorted(todo, key=lambda p: p != FIRST[1])
        if not todo:
            print(f"have step {step}", flush=True)
            continue
        first_here = step == FIRST[0] and not first_done
        t0 = time.monotonic()
        model, _ = load_model(f"pythia-410m-step{step}")
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("refusing: model is not float32")
        cut = SignCutter(model)
        hk = ua.Hooked(model)
        dev = next(model.parameters()).device
        ids = {p: torch.tensor([ua.token_ids(tok, Path(d["prompts"][p]["stage0_run"]))], device=dev)
               for p in todo}
        kept = {p: np.asarray(d["prompts"][p]["kept"], dtype=int) for p in todo}
        base: Dict[str, np.ndarray] = {}
        att_hs: Dict[str, np.ndarray] = {}
        labels: Dict[str, Dict[str, Dict]] = {p: {} for p in todo}
        checks: Dict[str, Dict] = {}
        hook_rel: Dict[str, Dict[int, float]] = {}
        applied: Dict[str, Dict] = {}
        arms = ARMS + (("att_hook",) if first_here else ())
        for arm in arms:
            hook = None
            if arm in WEIGHT_ARMS:
                applied[arm] = cut.apply_weights(arm, step)
            else:
                cut.restore()
                hook = DeltaHook(hk, cut.factors(arm))
                applied[arm] = {"hook": True}
            for p in todo:
                if arm == "base":
                    rd = Path(d["prompts"][p]["stage0_run"])
                    hs, comps = hk.run(ids[p], tuple(range(24)))
                    chk = ua.check_pass(hs, {L: comps[L] for L in range(23)}, rd, "v1")
                    chk["last_block_rel"] = e1.check_last_block(hs, comps[23], model)
                    checks[p] = chk
                    base[p] = hs
                    del comps
                else:
                    hs, _ = hk.run(ids[p], ())
                if not np.isfinite(hs).all():
                    raise OV1Error(f"step {step} {p} {arm}: non-finite hidden states")
                if arm == "att" and first_here:
                    att_hs[p] = hs
                if arm == "att_hook":
                    hook_rel[p] = {L: float(np.linalg.norm(hs[L] - att_hs[p][L]) / np.linalg.norm(att_hs[p][L]))
                                   for L in LAYERS}
                labels[p][arm] = ov.arm_labels(hs, base[p], kept[p])
            if hook is not None:
                hook.close()
        cut.restore()
        print(f"step {step}: {len(arms)} arms × {len(todo)} passes, {time.monotonic() - t0:.0f}s; worst "
              f"read-back {max(v.get('readback', 0.0) for v in applied.values()):.1e}", flush=True)
        for p in todo:
            t1 = time.monotonic()
            body = read_passage(labels[p], src, step, p, cache)
            rec = {"step": step, "passage": p, "run": d["prompts"][p]["stage0_run"], "code": code,
                   "pass": "cuda:float32", "kept": int(kept[p].size), "checks": checks[p],
                   "applied": applied, **body}
            if first_here:
                rec["hook_check"] = check_hook(rec, labels[p], hook_rel[p])
                rec["ov1_check"] = check_ov1(rec, Path(a.ov1))
                for c in rec["layers"].values():           # the check arm is not an arm of the rule
                    c["arms"].pop("att_hook", None)
            if (step, p) == FIRST and not first_done:
                check_populated(rec)
                first_done = True
                print(f"first record populated: {len(rec['layers'])} layers, "
                      f"{sum(len(c['c3x']) for c in rec['layers'].values())} c3x / "
                      f"{sum(len(c['groups']) for c in rec['layers'].values())} records; hook "
                      f"{rec['hook_check']}; OV1 {rec['ov1_check']}", flush=True)
            path = a.out / "records" / f"step{step}_{p}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.rename(path)
            print(f"done step {step} {p}: {len(rec['layers'])} layers "
                  f"({len(rec['refused_layers'])} refused), {time.monotonic() - t1:.0f}s", flush=True)
            if a.first_only:
                hk.close()
                return 0
        hk.close()
        del model, hk, cut, base, labels, att_hs
        gc.collect()
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the reading

def shares(rec: Dict, band: Sequence[int], arm: str, side: str) -> Tuple[int, float, float]:
    """``(records, merged share, kept share)`` over the band's group-layer records, ``side`` =
    ``c3x`` or ``rest`` (stored c2a groups not in c3x)."""
    ks = []
    for L in band:
        c = rec["layers"].get(str(L))
        if c is None:
            continue
        xs = set(c["c3x"])
        gs = c["c3x"] if side == "c3x" else [g for g in c["groups"] if g not in xs]
        ks += [c["arms"][arm]["kind"][str(g)] for g in gs]
    if not ks:
        return 0, float("nan"), float("nan")
    return len(ks), float(np.mean([k in ov.MERGED for k in ks])), float(np.mean([k in ov.KEPT for k in ks]))


def rel(rec: Dict, band: Sequence[int], arm: str) -> float:
    v = [rec["layers"][str(L)]["arms"][arm]["rel"] for L in band if str(L) in rec["layers"]]
    return float(np.mean(v)) if v else float("nan")


def cell(recs: Dict, step: int, band: str, label: str) -> Dict:
    a1, a2, what = LABELS[label]
    up, down = NAMES[what]
    vals, kept_vals, beside = [], [], {k: [] for k in ("m1", "m2", "k1", "k2", "rest1", "single", "rel1", "rel2")}
    for p in PASSAGES:
        r = recs.get((step, p))
        if r is None:
            continue
        n1, m1, k1 = shares(r, BANDS[band], a1, "c3x")
        _, m2, k2 = shares(r, BANDS[band], a2, "c3x")
        if n1 < ov.MIN_RECORDS:
            continue
        if what == "c3x":
            vals.append(m1 - m2)
            kept_vals.append(k1 - k2)
        else:
            nr, r1, _ = shares(r, BANDS[band], a1, "rest")
            _, r2, _ = shares(r, BANDS[band], a2, "rest")
            if nr < ov.MIN_RECORDS:
                continue
            vals.append((m1 - r1) - (m2 - r2))
            beside["rest1"].append(r1)
            beside["single"].append(m1 - r1)
        beside["m1"].append(m1), beside["m2"].append(m2), beside["k1"].append(k1), beside["k2"].append(k2)
        beside["rel1"].append(rel(r, BANDS[band], a1)), beside["rel2"].append(rel(r, BANDS[band], a2))
    mean = {k: (float(np.mean(v)) if v else float("nan")) for k, v in beside.items()}
    return {"label": ov.sign_label(vals, up, down),
            "kept_label": ov.sign_label(kept_vals, "keeps", "loses") if what == "c3x" else "",
            "n_passages": len(vals), "values": vals, **mean,
            "dissolves": bool(np.isfinite(mean["k2"]) and mean["k2"] < DISSOLVE)}


def reading(table: Dict, step: int, band: str) -> List[str]:
    """The rule's readings for one window."""
    s1, s2 = table[(step, band, "S1")]["label"], table[(step, band, "S2")]["label"]
    out = []
    if s1 == "merges":
        out.append("the sign does it, at fixed eigenvectors and size")
    elif (step, band) in OV1_READ:
        out.append("OV1's reading is not the sign at fixed size")
    if s2 == "merges":
        out.append("removing only the repulsive part merges c3x more than removing only the attractive part")
    for s, x in (("S1", "S3a"), ("S2", "S3b")):
        if table[(step, band, s)]["label"] != "merges":
            continue
        lx = table[(step, band, x)]["label"]
        out.append({"c3x more": f"{x}: specific to c3x's groups", "c3x less": f"{x}: c3x's groups merge less "
                    "than the rest"}.get(lx, f"{x}: the stream, not c3x in particular ({lx})"))
    for s in ("S1", "S2"):
        c = table[(step, band, s)]
        if c["label"].startswith("leans"):
            out.append(f"{s} {c['label']} (a lean, not read)")
        if c["dissolves"]:
            out.append(f"{s}: the flipped cut dissolves the groups")
    c = table[(step, band, "S1")]
    if s1 == "merges" and c["rel2"] >= c["rel1"]:
        out.append("neg moves the stream as far as att: OV1's 'moved further' caveat does not apply")
    return out


def report(a) -> int:
    recs = {}
    for f in sorted((a.out / "records").glob("step*_*.json")):
        r = json.loads(f.read_text())
        recs[(r["step"], r["passage"])] = r
    if not recs:
        raise SystemExit(f"refusing: no records in {a.out}")
    steps = sorted({s for s, _ in recs})
    table = {(s, b, lab): cell(recs, s, b, lab) for s in steps for b in BANDS for lab in LABELS}
    ov.mark_isolated(table)
    lines = ["| step | band | S1 att−neg | S2 norep−noatt | S3a | S3b | merged att / neg | merged norep / noatt "
             "| rest merged att / norep | rel att / neg / norep / noatt | n |", "|" + "---|" * 11]
    readings = {}
    for s in steps:
        for b in BANDS:
            c = {lab: table[(s, b, lab)] for lab in LABELS}
            readings[f"{s}|{b}"] = reading(table, s, b)
            lines.append(
                f"| {s}{'' if s in PRIMARY_STEPS else ' (beside)'} | {b} | {c['S1']['label']} ({c['S1']['kept_label']})"
                f" | {c['S2']['label']} ({c['S2']['kept_label']}) | {c['S3a']['label']} | {c['S3b']['label']} "
                f"| {c['S1']['m1']:.2f} / {c['S1']['m2']:.2f} | {c['S2']['m1']:.2f} / {c['S2']['m2']:.2f} "
                f"| {c['S3a']['rest1']:.2f} / {c['S3b']['rest1']:.2f} "
                f"| {c['S1']['rel1']:.2f} / {c['S1']['rel2']:.2f} / {c['S2']['rel1']:.2f} / {c['S2']['rel2']:.2f} "
                f"| {c['S1']['n_passages']} |")
    refused = {f"{s}/{p}": r["refused_layers"] for (s, p), r in recs.items() if r["refused_layers"]}
    first = recs.get(FIRST, {})
    out = {"table": {f"{s}|{b}|{lab}": c for (s, b, lab), c in table.items()}, "readings": readings,
           "chance": {n: ov.chance(n) for n in (6, 7)}, "refused_layers": refused,
           "hook_check": first.get("hook_check"), "ov1_check": first.get("ov1_check"),
           "records": len(recs), "codes": sorted({r["code"] for r in recs.values()})}
    (a.out / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))
    print("\nreadings (rule):")
    for k, v in readings.items():
        if v:
            print(f"  {k}: " + "; ".join(v))
    print("chance (no effect, passages independent; they share the step's cut models):", out["chance"])
    print("hook check:", out["hook_check"], "\nOV1 reproduction:", out["ov1_check"])
    print("refused layers:", refused or "none")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--ov1", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="*")
    r.add_argument("--first-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
