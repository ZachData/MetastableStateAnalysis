"""OV1d: a dose curve at matched centred distance (`p10_cluster_function/design-10.md` "OV1d", fixed
before any pass).

OV1s's cut (`tools/run/p10_ov1s_sign.py`) as a curve: family ``w`` writes each head's map as
``t·S₊`` in the weights (``w+1`` = OV1s's ``att``, ``w-1`` its ``neg``, ``w+0`` the OV removed);
family ``z`` adds ``Δ = t·S₊ − K`` by OV1s's hook with column 0 of each head's ``P`` zeroed in
``Δ``'s term, so position 0's channel stays at base. Each arm's distance from base is ``dc``, the
centred ``‖Δhs‖`` over the kept rows; labels M (``w``) and M0 (``z``) read c3x's merged share on
the positive branch less the negative one at matched ``dc``. Tier 1: exploratory, unregistered.
    python tools/run/p10_ov1d_dose.py run --labels <R9 source> --ov1s <OV1s out> --out <dir> [--steps ...]
    python tools/run/p10_ov1d_dose.py report --out <dir>
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
from tools.run import p10_ov1s_sign as sg
from tools.run.p10_ov1_cut import BANDS, LAYERS, PASSAGES, OV1Error


def name(fam: str, t: float) -> str:
    return f"{fam}{t:+g}"                                 # w+1, w-0.5, w+0


W_T = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0)
Z_T = (-2.0, -1.0, 0.0, 0.25, 0.5, 1.0)
FAMILY = {"w": tuple(name("w", t) for t in W_T), "z": tuple(name("z", t) for t in Z_T)}
ARMS = ("base",) + FAMILY["w"] + FAMILY["z"]
T = {name(f, t): t for f, ts in (("w", W_T), ("z", Z_T)) for t in ts}
CHECK = "h+1"                                            # z+1 with column 0 kept: the first record's hook check
LABELS = {"M": "w", "M0": "z"}
STEPS = (4000, 8000, 16000)
S1_WINDOWS = {(4000, "L17-24"), (8000, "L9-16"), (8000, "L17-24"), (16000, "L9-16"), (16000, "L17-24")}
FIRST = (16000, "wiki_paragraph")
OV1S = {"w+1": "att", "w-1": "neg"}                      # reproduction against OV1s's arms
DISSOLVE = 0.2                                           # an arm keeping < this of c3x's records (OV1's)


# ---------------------------------------------------------------- the arms

def pick(hd: Dict, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(U, λ)`` written into the head for ``t·S₊`` (no pairs at t = 0)."""
    lam, U = hd["lam"], hd["U"]
    pos = np.flatnonzero(lam > 0) if t != 0 else np.zeros(0, int)
    return U[:, pos], t * lam[pos]


def delta(hd: Dict, t: float, WO: np.ndarray, WVg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(L, R)`` with ``L Rᵀ = t·S₊ − K`` (``K = W_O W_V diag(γ)``, ``WVg = W_V diag(γ)``)."""
    lam, U = hd["lam"], hd["U"]
    pos = np.flatnonzero(lam > 0)
    return (np.hstack([U[:, pos] * (t * lam[pos])[None, :], -WO]),
            np.hstack([U[:, pos], WVg.T]))


class DoseCutter(sg.SignCutter):
    """OV1s's cutter with ``t·S₊`` in the weights and ``t·S₊ − K`` as the hook's ``Δ``."""

    def apply_weights(self, arm: str, step: int) -> Dict:
        torch = self.torch
        self.restore()
        if arm == "base":
            return {"readback": 0.0}
        t = T[arm]
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
                    U, lam = pick(hd, t)
                    if (t > 0 and (lam <= 0).any()) or (t < 0 and (lam >= 0).any()):
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
        """Per layer, ``(L, R)`` as (H, d, r) tensors for ``Δ = t·S₊ − K`` (``z`` arms and ``h+1``)."""
        torch = self.torch
        t = T[arm] if arm != CHECK else 1.0
        out = []
        for l, (lay, row) in enumerate(zip(self.layers, self.heads)):
            g, _ = self.ln[l]
            o = self.orig[l]
            qkv = o["qkv_w"].double().cpu().numpy().reshape(self.H, 3, self.k, -1)
            ow = o["o_w"].double().cpu().numpy()
            LR = [delta(hd, t, ow[:, h * self.k:(h + 1) * self.k], qkv[h, 2] * g[None, :])
                  for h, hd in enumerate(row)]
            r = max(L.shape[1] for L, _ in LR)
            d = LR[0][0].shape[0]
            Ls, Rs = np.zeros((self.H, d, r)), np.zeros((self.H, d, r))
            for h, (L, R) in enumerate(LR):
                Ls[h, :, :L.shape[1]], Rs[h, :, :R.shape[1]] = L, R
            W = lay.attention.query_key_value.weight
            out.append((torch.as_tensor(Ls, device=W.device, dtype=W.dtype),
                        torch.as_tensor(Rs, device=W.device, dtype=W.dtype)))
        return out


class SinkHook:
    """OV1s's ``DeltaHook`` (adds ``Σ_h P^h x̂ Δ_hᵀ`` to every block's attention output), with
    column 0 of each head's ``P`` zeroed in that term when ``shut0``: what ``Δ`` sends through
    position 0 is removed, and position 0's own row gets nothing."""

    def __init__(self, hooked, factors, shut0: bool):
        import torch
        self.hk, self.factors, self.handles, self.cap, self.on_p = hooked, factors, [], {}, {}
        for l, layer in enumerate(hooked.layers):
            ln = layer.input_layernorm

            def keep_xhat(m, i, o, _l=l, _eps=ln.eps):
                x = i[0][0]
                mu = x.mean(-1, keepdim=True)
                self.cap[("x", _l)] = (x - mu) / (x.var(-1, keepdim=True, unbiased=False) + _eps).sqrt()

            def keep_p(v, w, _l=l):
                p = w[0]
                if shut0:
                    p = p.clone()
                    p[..., 0] = 0
                self.cap[("p", _l)] = p

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


# ---------------------------------------------------------------- the distances

def centred(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def dc(H: np.ndarray, B: np.ndarray) -> float:
    """``‖C(H) − C(B)‖_F / ‖C(B)‖_F`` (C: the token mean subtracted)."""
    cb = centred(B)
    return float(np.linalg.norm(centred(H) - cb) / np.linalg.norm(cb))


def frame_cos(Y: np.ndarray) -> np.ndarray:
    """c2a's cosine distances (its centred frame's unit rows), float64."""
    from p1d_cluster_ensemble.gaussian_null import frame_vectors
    Z, _ = frame_vectors(np.asarray(Y, dtype=np.float64), "centred")
    return 1.0 - Z @ Z.T


def arm_read(hs: np.ndarray, base: np.ndarray, kept: np.ndarray, base_cos: Dict[int, np.ndarray]) -> Dict[int, Dict]:
    """Per layer: the arm's c2a labels on the kept rows, ``rel`` (OV1s's, every row), ``dc``, ``dz``."""
    out = {}
    for L in LAYERS:
        H, B = hs[L][kept], base[L][kept]
        out[L] = {"lab": ov.c2a_labels(H),
                  "rel": float(np.linalg.norm(hs[L] - base[L]) / np.linalg.norm(base[L])),
                  "dc": dc(H, B),
                  "dz": float(np.linalg.norm(frame_cos(H) - base_cos[L]) / np.linalg.norm(base_cos[L]))}
    return out


# ---------------------------------------------------------------- the readout

def read_passage(by_arm: Dict[str, Dict[int, Dict]], src: Path, step: int, passage: str, cache: Dict) -> Dict:
    """OV1s's per-layer readout (base against the stored c2a, every stored group's kind per arm),
    with ``dc`` and ``dz`` beside ``rel``."""
    layers, refused = {}, {}
    for L in LAYERS:
        _, stored = ls.load_column(src, f"step{step}", passage, L, "c2a", cache)
        _, labx = ls.load_column(src, f"step{step}", passage, L, "c3x", cache)
        ids = sorted(int(g) for g in np.unique(stored[stored >= 0]))
        x = sorted(int(g) for g in np.unique(labx[labx >= 0]))
        for g in x:
            if not np.array_equal(np.flatnonzero(labx == g), np.flatnonzero(stored == g)):
                raise OV1Error(f"step {step} {passage} L{L}: c3x group {g} is not c2a's")
        if not ov.same_partition(by_arm["base"][L]["lab"], stored):
            refused[str(L)] = "base c2a differs from the stored c2a"
            continue
        cell = {"c3x": x, "groups": ids, "arms": {}}
        for arm, per in by_arm.items():
            r = per[L]
            k = ov.kinds(stored, r["lab"], ids)
            cell["arms"][arm] = {**ov.layer_summary(r["lab"]), "rel": r["rel"], "dc": r["dc"], "dz": r["dz"],
                                 "kind": {str(g): k[g][0] for g in ids}}
        layers[str(L)] = cell
    return {"layers": layers, "refused_layers": refused}


def check_populated(rec: Dict, z_off_w: Dict[str, float]) -> None:
    """The rule's first-record check."""
    lay = rec["layers"]
    if not lay:
        raise SystemExit("refusing: first record has no readable layer")
    n_x = sum(len(c["c3x"]) for c in lay.values())
    n_all = sum(len(c["groups"]) for c in lay.values())
    if n_x == 0:
        raise SystemExit("refusing: first record has no c3x record")
    for arm in ARMS[1:]:
        if max(c["arms"][arm]["rel"] for c in lay.values()) <= ov.OFF_BASE:
            raise SystemExit(f"refusing: first record, {arm}'s hs is not off base")
        ks = [k for c in lay.values() for k in c["arms"][arm]["kind"].values()]
        if len(ks) != n_all:
            raise SystemExit(f"refusing: first record, {arm}: {len(ks)} kinds for {n_all} records")
    for arm, v in z_off_w.items():
        if not v > ov.OFF_BASE:
            raise SystemExit(f"refusing: first record, {arm} is not off its w partner ({v:.1e}): zeroing did nothing")
    if all(k == "stable" for c in lay.values() for k in c["arms"]["w+1"]["kind"].values()):
        raise SystemExit("refusing: first record, w+1 keeps every record stable")


def check_hook(rec: Dict, hs_rel: Dict[int, float]) -> Dict:
    """The hook path: ``h+1`` (z+1 with column 0 kept) against ``w+1`` in the weights."""
    worst = max(hs_rel.values())
    n, agree = sg.kind_agreement(rec, rec, "w+1", CHECK, only_c3x=False)
    out = {"worst_rel": worst, "records": n, "agree": agree}
    if not worst <= sg.HOOK_TOL or not agree >= sg.AGREE:
        raise SystemExit(f"refusing: h+1 vs w+1: ‖Δhs‖/‖hs‖ {worst:.1e} (≤ {sg.HOOK_TOL}), "
                         f"kinds agree {agree:.3f} of {n} (≥ {sg.AGREE})")
    return out


def check_ov1s(rec: Dict, ov1s: Path) -> Dict:
    """Reproduction: ``w+1`` / ``w-1`` against OV1s's ``att`` / ``neg``, every stored record."""
    f = ov1s / "records" / f"step{rec['step']}_{rec['passage']}.json"
    if not f.exists():
        raise SystemExit(f"refusing: no OV1s record {f}")
    old = json.loads(f.read_text())
    out = {"ov1s_record": str(f), "ov1s_code": old.get("code")}
    for arm, theirs in OV1S.items():
        n, agree = sg.kind_agreement(rec, old, arm, theirs, only_c3x=False)
        out[arm] = {"records": n, "agree": agree}
        if not n or not agree >= sg.AGREE:
            raise SystemExit(f"refusing: {arm}'s kinds agree with OV1s's {theirs} for {agree:.3f} of {n} (≥ {sg.AGREE})")
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
    meta = {"rule": "design-10.md \"OV1d\"", "code": code, "labels": str(src), "ov1s": str(a.ov1s),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()[:16],
            "arms": ARMS, "steps": STEPS, "hook_tol": sg.HOOK_TOL, "agree": sg.AGREE}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    steps = [FIRST[0]] + [s for s in (a.steps or STEPS) if s != FIRST[0]]
    first_done = (a.out / "records" / f"step{FIRST[0]}_{FIRST[1]}.json").exists()
    for step in steps:
        if step not in STEPS:
            raise SystemExit(f"refusing: step {step} is not one of the rule's {STEPS}")
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
        cut = DoseCutter(model)
        hk = ua.Hooked(model)
        dev = next(model.parameters()).device
        ids = {p: torch.tensor([ua.token_ids(tok, Path(d["prompts"][p]["stage0_run"]))], device=dev)
               for p in todo}
        kept = {p: np.asarray(d["prompts"][p]["kept"], dtype=int) for p in todo}
        base: Dict[str, np.ndarray] = {}
        base_cos: Dict[str, Dict[int, np.ndarray]] = {}
        keep_hs: Dict[str, np.ndarray] = {}                    # the first record's w arms, for its checks
        reads: Dict[str, Dict[str, Dict]] = {p: {} for p in todo}
        checks: Dict[str, Dict] = {}
        applied: Dict[str, Dict] = {}
        hook_rel: Dict[int, float] = {}
        z_off_w: Dict[str, float] = {}
        arms = ARMS + ((CHECK,) if first_here else ())
        for arm in arms:
            hook = None
            if arm == "base" or arm.startswith("w"):
                applied[arm] = cut.apply_weights(arm, step)
            else:
                cut.restore()
                hook = SinkHook(hk, cut.factors(arm), shut0=arm.startswith("z"))
                applied[arm] = {"hook": True, "shut0": arm.startswith("z")}
            for p in todo:
                if arm == "base":
                    rd = Path(d["prompts"][p]["stage0_run"])
                    hs, comps = hk.run(ids[p], tuple(range(24)))
                    chk = ua.check_pass(hs, {L: comps[L] for L in range(23)}, rd, "v1")
                    chk["last_block_rel"] = e1.check_last_block(hs, comps[23], model)
                    checks[p] = chk
                    base[p] = hs
                    base_cos[p] = {L: frame_cos(hs[L][kept[p]]) for L in LAYERS}
                    del comps
                else:
                    hs, _ = hk.run(ids[p], ())
                if not np.isfinite(hs).all():
                    raise OV1Error(f"step {step} {p} {arm}: non-finite hidden states")
                if first_here and p == FIRST[1]:
                    if arm.startswith("w") and (arm == "w+1" or T[arm] in Z_T):
                        keep_hs[arm] = hs
                    if arm.startswith("z"):
                        w = keep_hs[name("w", T[arm])]
                        z_off_w[arm] = max(float(np.linalg.norm(hs[L] - w[L]) / np.linalg.norm(w[L])) for L in LAYERS)
                    if arm == CHECK:
                        w = keep_hs["w+1"]
                        hook_rel = {L: float(np.linalg.norm(hs[L] - w[L]) / np.linalg.norm(w[L])) for L in LAYERS}
                reads[p][arm] = arm_read(hs, base[p], kept[p], base_cos[p])
            if hook is not None:
                hook.close()
        cut.restore()
        print(f"step {step}: {len(arms)} arms × {len(todo)} passes, {time.monotonic() - t0:.0f}s; worst "
              f"read-back {max(v.get('readback', 0.0) for v in applied.values()):.1e}", flush=True)
        for p in todo:
            t1 = time.monotonic()
            body = read_passage(reads[p], src, step, p, cache)
            rec = {"step": step, "passage": p, "run": d["prompts"][p]["stage0_run"], "code": code,
                   "pass": "cuda:float32", "kept": int(kept[p].size), "checks": checks[p],
                   "applied": applied, **body}
            if first_here and p == FIRST[1]:
                rec["hook_check"] = check_hook(rec, hook_rel)
                rec["ov1s_check"] = check_ov1s(rec, Path(a.ov1s))
                rec["z_off_w"] = z_off_w
                for c in rec["layers"].values():           # the check arm is not an arm of the rule
                    c["arms"].pop(CHECK, None)
                check_populated(rec, z_off_w)
                first_done = True
                print(f"first record populated: {len(rec['layers'])} layers, "
                      f"{sum(len(c['c3x']) for c in rec['layers'].values())} c3x / "
                      f"{sum(len(c['groups']) for c in rec['layers'].values())} records; hook "
                      f"{rec['hook_check']}; OV1s {rec['ov1s_check']}; z off w {z_off_w}", flush=True)
            elif first_here:
                for c in rec["layers"].values():
                    c["arms"].pop(CHECK, None)
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
        del model, hk, cut, base, base_cos, reads, keep_hs
        gc.collect()
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the reading

def arm_shares(rec: Dict, band: Sequence[int], arm: str) -> Dict:
    """c3x's records in the band under ``arm``: count, merged / kept / death shares, and the mean
    ``dc``, ``rel``, ``dz`` over the band's readable layers."""
    ks, dist = [], {"dc": [], "rel": [], "dz": []}
    for L in band:
        c = rec["layers"].get(str(L))
        if c is None:
            continue
        a = c["arms"][arm]
        ks += [a["kind"][str(g)] for g in c["c3x"]]
        for k in dist:
            dist[k].append(a[k])
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in dist.items()}
    out["n"] = len(ks)
    for k, kinds in (("m", ov.MERGED), ("kept", ov.KEPT), ("death", ("death",))):
        out[k] = float(np.mean([x in kinds for x in ks])) if ks else float("nan")
    return out


def interp_first(branch: Sequence[Tuple[float, float, float]], d: float) -> Optional[float]:
    """``m`` at distance ``d`` on the first segment (outward from t = 0) whose end distances bracket
    ``d``, linear in ``d``; None if no segment does."""
    for (_, d1, m1), (_, d2, m2) in zip(branch, branch[1:]):
        if min(d1, d2) <= d <= max(d1, d2):
            return m1 if d2 == d1 else m1 + (m2 - m1) * (d - d1) / (d2 - d1)
    return None


def matched(pts: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """``(t, m positive side, m negative side)`` for every t ≠ 0 point matched onto the other branch;
    ``pts`` = (t, d, m) with a t = 0 point."""
    if not any(t == 0 for t, _, _ in pts):
        raise OV1Error("a family's points need t = 0")
    pos = sorted([p for p in pts if p[0] >= 0], key=lambda p: p[0])
    neg = sorted([p for p in pts if p[0] <= 0], key=lambda p: -p[0])
    out = []
    for t, d, m in pts:
        if t == 0 or not np.isfinite(d) or not np.isfinite(m):
            continue
        mi = interp_first(neg if t > 0 else pos, d)
        if mi is not None:
            out.append((t, m, mi) if t > 0 else (t, mi, m))
    return out


def cell(recs: Dict, step: int, band: str, label: str) -> Dict:
    fam = LABELS[label]
    vals, n_match = [], []
    curve = {arm: {k: [] for k in ("m", "kept", "death", "dc", "rel", "dz")} for arm in FAMILY[fam]}
    for p in PASSAGES:
        r = recs.get((step, p))
        if r is None:
            continue
        sh = {arm: arm_shares(r, BANDS[band], arm) for arm in FAMILY[fam]}
        for arm, s in sh.items():
            for k in curve[arm]:
                curve[arm][k].append(s[k])
        if sh[FAMILY[fam][0]]["n"] < ov.MIN_RECORDS:
            continue
        pairs = matched([(T[arm], s["dc"], s["m"]) for arm, s in sh.items()])
        if not pairs:
            continue
        vals.append(float(np.mean([mp - mn for _, mp, mn in pairs])))
        n_match.append(len(pairs))
    mean = {arm: {k: (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")) for k, v in c.items()}
            for arm, c in curve.items()}
    return {"label": ov.sign_label(vals, "merges", "separates"), "n_passages": len(vals), "values": vals,
            "matched_points": n_match, "curve": mean,
            "dissolves": [arm for arm, c in mean.items() if np.isfinite(c["kept"]) and c["kept"] < DISSOLVE]}


def reading(table: Dict, step: int, band: str) -> List[str]:
    """The rule's readings, in S1's five windows only."""
    if (step, band) not in S1_WINDOWS:
        return []
    m, m0 = table[(step, band, "M")]["label"], table[(step, band, "M0")]["label"]
    out = [{"merges": "at matched centred distance S₊ merges more than −S₊: S1 is not the distance moved",
            "separates": "at matched distance −S₊ merges more: S1 was the distance moved",
            "too few": "no matched distance (Blocked 33's fallback, (c))"}.get(
        m, "S1 not shown beyond the distance moved")]
    if m0 == "merges":
        out.append("with position 0's channel at base the sign still merges at matched distance: "
                   "not one shift through the sink")
    elif m == "merges":
        out.append("not shown without position 0's channel (the sink carries it, or z is too weak to read: its d beside)")
    for lab in LABELS:
        c = table[(step, band, lab)]
        if c["label"].startswith("leans"):
            out.append(f"{lab} {c['label']} (a lean, not read)")
        if c["dissolves"]:
            out.append(f"{lab}: dissolves ({', '.join(c['dissolves'])} keep < {DISSOLVE} of c3x's records)")
    return out


def report(a) -> int:
    recs = ov.load_records(a.out)
    if not recs:
        raise SystemExit(f"refusing: no records in {a.out}")
    steps = sorted({s for s, _ in recs})
    table = {(s, b, lab): cell(recs, s, b, lab) for s in steps for b in BANDS for lab in LABELS}
    ov.mark_isolated(table)
    lines = ["| step | band | M (w) | M0 (z) | n | matched / passage | read |", "|" + "---|" * 7]
    curves = ["| step | band | arm | t | m | kept | death | dc | rel | dz |", "|" + "---|" * 10]
    readings = {}
    for s in steps:
        for b in BANDS:
            c = {lab: table[(s, b, lab)] for lab in LABELS}
            readings[f"{s}|{b}"] = reading(table, s, b)
            lines.append(f"| {s} | {b} | {c['M']['label']} | {c['M0']['label']} "
                         f"| {c['M']['n_passages']} / {c['M0']['n_passages']} "
                         f"| {np.mean(c['M']['matched_points'] or [0]):.1f} / {np.mean(c['M0']['matched_points'] or [0]):.1f} "
                         f"| {'yes' if (s, b) in S1_WINDOWS else 'beside'} |")
            for lab in LABELS:
                for arm, v in c[lab]["curve"].items():
                    curves.append(f"| {s} | {b} | {arm} | {T[arm]:+g} | {v['m']:.2f} | {v['kept']:.2f} "
                                  f"| {v['death']:.2f} | {v['dc']:.2f} | {v['rel']:.2f} | {v['dz']:.2f} |")
    refused = {f"{s}/{p}": r["refused_layers"] for (s, p), r in recs.items() if r["refused_layers"]}
    first = recs.get(FIRST, {})
    out = {"table": {f"{s}|{b}|{lab}": c for (s, b, lab), c in table.items()}, "readings": readings,
           "chance": {n: ov.chance(n) for n in (6, 7)}, "refused_layers": refused,
           "hook_check": first.get("hook_check"), "ov1s_check": first.get("ov1s_check"),
           "z_off_w": first.get("z_off_w"), "records": len(recs), "codes": sorted({r["code"] for r in recs.values()})}
    (a.out / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))
    print("\ncurves (passage means; m = c3x merged share):")
    print("\n".join(curves))
    print("\nreadings (rule):")
    for k, v in readings.items():
        if v:
            print(f"  {k}: " + "; ".join(v))
    print("chance (no effect, passages independent; they share the step's cut models):", out["chance"])
    print("hook check:", out["hook_check"], "\nOV1s reproduction:", out["ov1s_check"], "\nz off w:", out["z_off_w"])
    print("refused layers:", refused or "none")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--ov1s", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="*")
    r.add_argument("--first-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
