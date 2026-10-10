"""OV1: the OV cut (`p10_cluster_function/design-10.md` "OV1", fixed before any cut-model pass).

With every attention head's OV cut to its attractive part, do c3x's groups merge (the OV's
repulsive part keeps them apart) or survive? Per head, in LN1's normalised frame,
``K = W_O W_V diag(γ)`` and ``S = sym(Π K Π)`` (``Π = I − 11ᵀ/d``), ``S = S₊ − S₋``; the arms
replace every head's ``K`` at once by ``S₊`` (``att``), ``−S₋`` (``rep``), or a random subset of
``S``'s eigenpairs of ``att``'s (``ctl+``) or ``rep``'s (``ctl−``) count and norm, 10 each. Each
head's constant ``c = W_O(W_V β + b_V)`` moves to the dense bias, so a cut head adds exactly
``Σ_j P_ij S' x̂_j + c`` (`tools/math_checks/ov_cut_ov1.py`). Per (step, passage, arm) one GPU
pass; c2a (level-set, centred, size 2) on the arm's ``hs[L]`` at R0's kept offsets; R9's stored
c2a linked to it by R6's ``link_layer_pair`` (containment ≥ 0.5): each c3x group record is kept
(stable), merged (merge / tangle), split or dead. Tier 1: exploratory, unregistered.
    python tools/run/p10_ov1_cut.py run --labels <R9 source> --out <dir> [--steps 512 ...]
    python tools/run/p10_ov1_cut.py report --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from p1d_cluster_ensemble.merge_tree import link_layer_pair
from tools.run import p10_label_source as ls

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
PRIMARY_STEPS = tuple(s for s in STEPS if s >= 64)       # E1's structure count
PASSAGES = ("wiki_paragraph", "sullivan_ballou", "paper_excerpt", "homer_iliad", "hdbscan_code",
            "camus_letranger", "latex_monograph")
LAYERS = tuple(range(1, 25))
BANDS = {"L1-8": tuple(range(1, 9)), "L9-16": tuple(range(9, 17)), "L17-24": tuple(range(17, 25))}
N_CTL = 10
ARMS = ("base", "att", "rep") + tuple(f"ctl+{i}" for i in range(N_CTL)) + tuple(f"ctl-{i}" for i in range(N_CTL))
PAIRS = {"att": "ctl+", "rep": "ctl-"}                    # arm → its controls
NONZERO = 1e-6                                           # |λ| > NONZERO · max|λ| (placed)
READBACK_TOL = 1e-4                                      # written map against S', of ‖S'‖ (placed)
OFF_BASE = 1e-3                                          # first record: an arm's hs moved (placed)
CONTROL_FLOOR = 0.2                                      # controls keep < this → said beside (placed)
MIN_RECORDS, MIN_PASSAGES = 3, 6                          # E1's
MIN_GAMMA = 1e-3
FIRST = (512, "wiki_paragraph")
MERGED, KEPT = ("merge", "tangle"), ("stable",)


class OV1Error(RuntimeError):
    """OV1 refuses."""


# ---------------------------------------------------------------- the cut, per head

def head_eig(W_O: np.ndarray, W_V: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(λ, U, ‖K‖_F)`` of ``S = sym(Π K Π)``, ``K = W_O W_V diag(γ)``, through its rank-2k factor:
    ``S = Z J Zᵀ``, ``Z = [Π W_O, Π diag(γ) W_Vᵀ]``, ``J = ½[[0, I], [I, 0]]``. Only the nonzero
    pairs (``|λ| > NONZERO · max|λ|``) are returned; ``U``'s columns are orthonormal, in the plane."""
    W_O, W_V, gamma = (np.asarray(a, dtype=np.float64) for a in (W_O, W_V, gamma))
    k = W_O.shape[1]
    B = W_O - W_O.mean(axis=0, keepdims=True)
    C = (gamma[:, None] * W_V.T)
    C = C - C.mean(axis=0, keepdims=True)
    Q, R = np.linalg.qr(np.hstack([B, C]))
    J = np.zeros((2 * k, 2 * k))
    J[:k, k:] = J[k:, :k] = 0.5 * np.eye(k)
    lam, w = np.linalg.eigh(R @ J @ R.T)
    keep = np.abs(lam) > NONZERO * np.abs(lam).max() if lam.size and np.abs(lam).max() > 0 else np.zeros(0, bool)
    K2 = float(np.trace((W_O.T @ W_O) @ ((W_V * gamma ** 2) @ W_V.T)))
    return lam[keep], Q @ w[:, keep], float(np.sqrt(max(K2, 0.0)))


def choose(lam: np.ndarray, arm: str, seed_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(indices into λ, the λ written)`` for one head under ``arm``."""
    pos, neg = np.flatnonzero(lam > 0), np.flatnonzero(lam < 0)
    if arm == "base":
        raise OV1Error("base is not cut")
    if arm == "att":
        return pos, lam[pos]
    if arm == "rep":
        return neg, lam[neg]
    side = pos if arm.startswith("ctl+") else neg
    if side.size == 0:
        return side, lam[side]
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    idx = np.sort(rng.choice(lam.size, size=side.size, replace=False))
    sel = lam[idx]
    return idx, sel * (np.linalg.norm(lam[side]) / np.linalg.norm(sel))


def write_back(U: np.ndarray, lam: np.ndarray, gamma: np.ndarray, beta: np.ndarray, k: int):
    """``(W_V' (k, d), b_V' (k,), W_O' (d, k))`` with ``W_O'(W_V'(γ ⊙ x + β) + b_V') = U diag(λ) Uᵀ x``,
    zero past the ``len(λ)`` pairs (math check C4)."""
    d, m = U.shape[0], lam.size
    if m > k:
        raise OV1Error(f"{m} pairs do not fit a {k}-wide head (math check C3)")
    root = np.sqrt(np.abs(lam))
    WV = np.zeros((k, d))
    WO = np.zeros((d, k))
    WV[:m] = (root[:, None] * U.T) / gamma[None, :]
    WO[:, :m] = U * (np.sign(lam) * root)[None, :]
    return WV, -(WV @ beta), WO


def readback_err(WV: np.ndarray, WO: np.ndarray, gamma: np.ndarray, U: np.ndarray, lam: np.ndarray) -> float:
    """``‖W_O' W_V' diag(γ) − U diag(λ) Uᵀ‖_F / ‖λ‖`` from the written (float32) factors, through
    k×k Grams (no d×d product)."""
    if lam.size == 0:
        return float(np.abs(WO).max() + np.abs(WV).max())
    X = np.asarray(WO, dtype=np.float64)
    Y = (np.asarray(WV, dtype=np.float64) * gamma[None, :]).T
    xy = np.trace((X.T @ X) @ (Y.T @ Y))
    cross = np.trace(((U.T @ X) @ (Y.T @ U)) * lam[None, :])
    e2 = xy - 2 * cross + float(lam @ lam)
    return float(np.sqrt(max(e2, 0.0)) / np.linalg.norm(lam))


class Cutter:
    """Every head's eigenpairs at one step, and the writer of an arm into the loaded model."""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.layers = getattr(model, "gpt_neox", model).layers
        if len(self.layers) == 0:
            raise OV1Error("the model has no layers (a stub?)")
        att0 = self.layers[0].attention
        self.H = model.config.num_attention_heads
        self.k = att0.head_size
        self.orig = [{"qkv_w": l.attention.query_key_value.weight.detach().clone(),
                      "qkv_b": l.attention.query_key_value.bias.detach().clone(),
                      "o_w": l.attention.dense.weight.detach().clone(),
                      "o_b": l.attention.dense.bias.detach().clone()} for l in self.layers]
        self.heads: List[List[Dict]] = []
        self.ln: List[Tuple[np.ndarray, np.ndarray]] = []     # LN1's (γ, β) per layer
        for l, lay in enumerate(self.layers):
            o = self.orig[l]
            g = lay.input_layernorm.weight.detach().double().cpu().numpy()
            b = lay.input_layernorm.bias.detach().double().cpu().numpy()
            if np.abs(g).min() < MIN_GAMMA:
                raise OV1Error(f"layer {l}: |γ| {np.abs(g).min():.1e} < {MIN_GAMMA} (W_V' divides by γ)")
            qkv = o["qkv_w"].double().cpu().numpy().reshape(self.H, 3, self.k, -1)
            bv = o["qkv_b"].double().cpu().numpy().reshape(self.H, 3, self.k)[:, 2]
            ow = o["o_w"].double().cpu().numpy()
            row = []
            for h in range(self.H):
                WO, WV = ow[:, h * self.k:(h + 1) * self.k], qkv[h, 2]
                lam, U, kn = head_eig(WO, WV, g)
                row.append({"lam": lam, "U": U, "K": kn, "c": WO @ (WV @ b + bv[h])})
            self.heads.append(row)
            self.ln.append((g, b))

    def stats(self) -> Dict:
        """Per (layer, head): n₊, n₋, ‖S₊‖, ‖S₋‖ and ‖K − S‖ as shares of ‖K‖."""
        out = {k: [] for k in ("n_pos", "n_neg", "s_pos", "s_neg", "off_sym")}
        for row in self.heads:
            for k in out:
                out[k].append([])
            for hd in row:
                lam, kn = hd["lam"], max(hd["K"], 1e-300)
                sp, sm = np.linalg.norm(lam[lam > 0]), np.linalg.norm(lam[lam < 0])
                out["n_pos"][-1].append(int((lam > 0).sum()))
                out["n_neg"][-1].append(int((lam < 0).sum()))
                out["s_pos"][-1].append(float(sp / kn))
                out["s_neg"][-1].append(float(sm / kn))
                out["off_sym"][-1].append(float(np.sqrt(max(kn ** 2 - sp ** 2 - sm ** 2, 0.0)) / kn))
        return out

    def restore(self) -> None:
        with self.torch.no_grad():
            for lay, o in zip(self.layers, self.orig):
                lay.attention.query_key_value.weight.copy_(o["qkv_w"])
                lay.attention.query_key_value.bias.copy_(o["qkv_b"])
                lay.attention.dense.weight.copy_(o["o_w"])
                lay.attention.dense.bias.copy_(o["o_b"])

    def apply(self, arm: str, step: int) -> Dict:
        """Write ``arm`` into the model (from the original weights); refuses a read-back off by
        more than ``READBACK_TOL``. Returns the worst read-back error and pairs written."""
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
                    idx, lam = choose(hd["lam"], arm, f"{step}|{arm}|{l}|{h}")
                    U = hd["U"][:, idx]
                    WV, bV, WO = write_back(U, lam, g, b, self.k)
                    qkv_w[h, 2] = torch.as_tensor(WV, device=dev, dtype=dt)
                    qkv_b[h, 2] = torch.as_tensor(bV, device=dev, dtype=dt)
                    att.dense.weight[:, h * self.k:(h + 1) * self.k] = torch.as_tensor(WO, device=dev, dtype=dt)
                    c_sum += hd["c"]
                    WV32 = qkv_w[h, 2].double().cpu().numpy()
                    WO32 = att.dense.weight[:, h * self.k:(h + 1) * self.k].double().cpu().numpy()
                    err = readback_err(WV32, WO32, g, U, lam)
                    worst = max(worst, err)
                    pairs += lam.size
                    if (arm == "att" and (lam < 0).any()) or (arm == "rep" and (lam > 0).any()):
                        raise OV1Error(f"step {step} {arm} L{l} h{h}: a pair of the wrong sign")
                att.dense.bias.copy_(self.orig[l]["o_b"] + torch.as_tensor(c_sum, device=dev, dtype=dt))
        if worst > READBACK_TOL:
            raise OV1Error(f"step {step} {arm}: written map off S' by {worst:.1e} of ‖S'‖")
        return {"readback": worst, "pairs": pairs}


# ---------------------------------------------------------------- the readout

def c2a_labels(Y: np.ndarray) -> np.ndarray:
    """c2a's labelling (level-set groups, centred, size 2) of the kept rows ``Y``."""
    groups, _ = ls._layer_groups(np.asarray(Y, dtype=np.float64), "centred", 2, shipped=False)
    return np.asarray(ls.labels_of(Y.shape[0], groups), dtype=int)


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    ga = sorted(tuple(np.flatnonzero(a == g)) for g in np.unique(a[a >= 0]))
    gb = sorted(tuple(np.flatnonzero(b == g)) for g in np.unique(b[b >= 0]))
    return ga == gb


def kinds(stored: np.ndarray, arm: np.ndarray, ids: Sequence[int]) -> Dict[int, Tuple[str, Optional[float]]]:
    """Each stored group id's MONIC kind against the arm's partition (R6's link), and its stable
    link's Jaccard."""
    out = link_layer_pair(stored, arm, min_overlap=0.5, measure="containment")
    jac = {a: j for a, b, j, _ in out["edges"]}
    kind = {}
    for comp in out["components"]:
        for a in comp["prev"]:
            kind[a] = comp["kind"]
    res = {}
    for g in ids:
        kd = kind.get(int(g), "death")
        res[int(g)] = (kd, float(jac[int(g)]) if kd == "stable" else None)
    return res


def layer_summary(lab: np.ndarray) -> Dict:
    n = lab.size
    sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(0, int)
    return {"n_groups": int((sizes > 0).sum()), "largest": float(sizes.max() / n) if sizes.size else 0.0,
            "noise": float((lab < 0).mean())}


def arm_labels(hs: np.ndarray, base: np.ndarray, kept: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
    """Per layer: the arm's c2a labels on the kept rows, and ``‖hs − base‖ / ‖base‖``."""
    return {L: (c2a_labels(hs[L][kept]), float(np.linalg.norm(hs[L] - base[L]) / np.linalg.norm(base[L])))
            for L in LAYERS}


def read_passage(by_arm: Dict[str, Dict[int, Tuple[np.ndarray, float]]], src: Path, step: int, passage: str,
                 cache: Dict) -> Dict:
    """Per layer: base against the stored c2a (refused if it differs), then every arm's kinds."""
    layers, refused = {}, {}
    for L in LAYERS:
        _, stored = ls.load_column(src, f"step{step}", passage, L, "c2a", cache)
        _, labx = ls.load_column(src, f"step{step}", passage, L, "c3x", cache)
        ids = sorted(int(g) for g in np.unique(labx[labx >= 0]))
        for g in ids:
            if not np.array_equal(np.flatnonzero(labx == g), np.flatnonzero(stored == g)):
                raise OV1Error(f"step {step} {passage} L{L}: c3x group {g} is not c2a's")
        if not same_partition(by_arm["base"][L][0], stored):
            refused[str(L)] = "base c2a differs from the stored c2a"
            continue
        cell = {"c3x": ids, "arms": {}}
        for arm, labs in by_arm.items():
            lab, rel = labs[L]
            k = kinds(stored, lab, ids)
            cell["arms"][arm] = {**layer_summary(lab), "rel": rel,
                                 "kind": {str(g): k[g][0] for g in ids},
                                 "jac": {str(g): k[g][1] for g in ids if k[g][1] is not None}}
        layers[str(L)] = cell
    return {"layers": layers, "refused_layers": refused}


def check_populated(rec: Dict) -> None:
    """The rule's first-record check."""
    lay = rec["layers"]
    if not lay:
        raise SystemExit("refusing: first record has no readable layer")
    n_c3x = sum(len(c["c3x"]) for c in lay.values())
    if n_c3x == 0:
        raise SystemExit("refusing: first record has no c3x group")
    for arm in ARMS[1:]:
        if max(c["arms"][arm]["rel"] for c in lay.values()) <= OFF_BASE:
            raise SystemExit(f"refusing: first record, {arm}'s hs is not off base")
        ks = [k for c in lay.values() for k in c["arms"][arm]["kind"].values()]
        if len(ks) != n_c3x:
            raise SystemExit(f"refusing: first record, {arm}: {len(ks)} kinds for {n_c3x} c3x records")
    for arm in ("att",) + tuple(a for a in ARMS if a.startswith("ctl")):
        if all(k == "stable" for c in lay.values() for k in c["arms"][arm]["kind"].values()):
            raise SystemExit(f"refusing: first record, {arm} keeps every c3x record stable")


# ---------------------------------------------------------------- the batch

def code_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


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
    code = code_sha()
    src = Path(a.labels)
    meta = {"rule": "design-10.md \"OV1\"", "code": code, "labels": str(src),
            "summary_sha256": hashlib.sha256((src / "summary.json").read_bytes()).hexdigest()[:16],
            "arms": ARMS, "nonzero": NONZERO, "readback_tol": READBACK_TOL}
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
        t0 = time.monotonic()
        model, _ = load_model(f"pythia-410m-step{step}")
        if next(model.parameters()).dtype != torch.float32:
            raise SystemExit("refusing: model is not float32")
        cut = Cutter(model)
        hpath = a.out / "heads" / f"step{step}.json"
        hpath.parent.mkdir(parents=True, exist_ok=True)
        hpath.write_text(json.dumps(cut.stats()) + "\n")
        hk = ua.Hooked(model)
        dev = next(model.parameters()).device
        ids = {p: torch.tensor([ua.token_ids(tok, Path(d["prompts"][p]["stage0_run"]))], device=dev)
               for p in todo}
        kept = {p: np.asarray(d["prompts"][p]["kept"], dtype=int) for p in todo}
        base: Dict[str, np.ndarray] = {}
        labels: Dict[str, Dict[str, Dict]] = {p: {} for p in todo}
        checks: Dict[str, Dict] = {}
        applied: Dict[str, Dict] = {}
        for arm in ARMS:
            applied[arm] = cut.apply(arm, step)
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
                labels[p][arm] = arm_labels(hs, base[p], kept[p])
        cut.restore()
        print(f"step {step}: {len(ARMS)} arms × {len(todo)} passes, {time.monotonic() - t0:.0f}s; "
              f"worst read-back {max(v['readback'] for v in applied.values()):.1e}", flush=True)
        for p in todo:
            t1 = time.monotonic()
            body = read_passage(labels[p], src, step, p, cache)
            rec = {"step": step, "passage": p, "run": d["prompts"][p]["stage0_run"], "code": code,
                   "pass": "cuda:float32", "kept": int(kept[p].size), "checks": checks[p],
                   "applied": applied, **body}
            if (step, p) == FIRST and not first_done:
                check_populated(rec)
                first_done = True
                print(f"first record populated: {len(rec['layers'])} layers, "
                      f"{sum(len(c['c3x']) for c in rec['layers'].values())} c3x records", flush=True)
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
        del model, hk, cut, base, labels
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- the reading

def sign_label(vals: Sequence[float], up: str, down: str) -> str:
    """The rule's label over the readable passages' values (E1's, renamed)."""
    v = [x for x in vals if np.isfinite(x)]
    n = len(v)
    if n < MIN_PASSAGES:
        return "too few"
    pos, neg = sum(x > 0 for x in v), sum(x < 0 for x in v)
    if pos == n:
        return up
    if neg == n:
        return down
    if pos == n - 1:
        return f"leans {up}"
    if neg == n - 1:
        return f"leans {down}"
    return "mixed"


def chance(n: int) -> Dict[str, float]:
    return {"full": 2 / 2 ** n, "leans": 2 * n / 2 ** n} if n else {"full": float("nan"), "leans": float("nan")}


def load_records(out: Path) -> Dict[Tuple[int, str], Dict]:
    recs = {}
    for f in sorted((out / "records").glob("step*_*.json")):
        r = json.loads(f.read_text())
        recs[(r["step"], r["passage"])] = r
    return recs


def shares(rec: Dict, band: Sequence[int], arm: str) -> Tuple[int, float, float]:
    """``(records, merged share, kept share)`` of the c3x group-layer records in a band."""
    ks = [c["arms"][arm]["kind"][str(g)] for L in band if str(L) in rec["layers"]
          for c in [rec["layers"][str(L)]] for g in c["c3x"]]
    if not ks:
        return 0, float("nan"), float("nan")
    return len(ks), float(np.mean([k in MERGED for k in ks])), float(np.mean([k in KEPT for k in ks]))


def cell(recs: Dict, step: int, band: str, arm: str) -> Dict:
    ctl = [a for a in ARMS if a.startswith(PAIRS[arm])]
    D, Dk, arm_m, ctl_m, arm_k, ctl_k = [], [], [], [], [], []
    for p in PASSAGES:
        r = recs.get((step, p))
        if r is None:
            continue
        n, m, k = shares(r, BANDS[band], arm)
        if n < MIN_RECORDS:
            continue
        cm = np.mean([shares(r, BANDS[band], c)[1] for c in ctl])
        ck = np.mean([shares(r, BANDS[band], c)[2] for c in ctl])
        D.append(m - cm)
        Dk.append(k - ck)
        arm_m.append(m), ctl_m.append(cm), arm_k.append(k), ctl_k.append(ck)
    return {"label": sign_label(D, "merges", "separates"), "kept_label": sign_label(Dk, "keeps", "loses"),
            "n_passages": len(D), "D": D,
            "merged": float(np.mean(arm_m)) if arm_m else float("nan"),
            "merged_ctl": float(np.mean(ctl_m)) if ctl_m else float("nan"),
            "kept": float(np.mean(arm_k)) if arm_k else float("nan"),
            "kept_ctl": float(np.mean(ctl_k)) if ctl_k else float("nan"),
            "control_dissolves": bool(ctl_k and np.mean(ctl_k) < CONTROL_FLOOR)}


def mark_isolated(table: Dict) -> None:
    """A lean not shared by an adjacent step (same band, arm, direction) is isolated."""
    for (step, band, arm), c in table.items():
        if not c["label"].startswith("leans"):
            continue
        word = c["label"].split()[-1]
        i = STEPS.index(step)
        nbrs = [table.get((STEPS[j], band, arm)) for j in (i - 1, i + 1) if 0 <= j < len(STEPS)]
        if not any(nb and word in nb["label"] for nb in nbrs):
            c["label"] += " (isolated)"


def reading(att: str, rep: str) -> str:
    """The rule's readings, per window."""
    a, r = att.split(" (")[0], rep.split(" (")[0]
    if "merges" in a and "merges" not in r:
        return "repulsion keeps them apart"
    if "merges" in a and "merges" in r:
        return "a sign-pure cut merges, either sign"
    if a in ("too few",):
        return "too few"
    return "not OV repulsion"


def report(a) -> int:
    recs = load_records(a.out)
    if not recs:
        raise SystemExit(f"refusing: no records in {a.out}")
    steps = sorted({s for s, _ in recs})
    table = {(s, b, arm): cell(recs, s, b, arm) for s in steps for b in BANDS for arm in PAIRS}
    mark_isolated(table)
    heads = {}
    for s in steps:
        f = a.out / "heads" / f"step{s}.json"
        if f.exists():
            h = json.loads(f.read_text())
            heads[s] = {k: float(np.median(np.asarray(v))) for k, v in h.items()}
    lines = ["| step | band | att (vs ctl+) | rep (vs ctl−) | reading | att merged / ctl+ | rep merged / ctl− "
             "| att kept / ctl+ | rep kept / ctl− | n |", "|" + "---|" * 10]
    for s in steps:
        for b in BANDS:
            ca, cr = table[(s, b, "att")], table[(s, b, "rep")]
            tag = " ‡" if ca["control_dissolves"] or cr["control_dissolves"] else ""
            lines.append(f"| {s}{'' if s in PRIMARY_STEPS else ' (beside)'} | {b} | {ca['label']} "
                         f"({ca['kept_label']}) | {cr['label']} ({cr['kept_label']}) | "
                         f"{reading(ca['label'], cr['label'])}{tag} | {ca['merged']:.2f} / {ca['merged_ctl']:.2f} "
                         f"| {cr['merged']:.2f} / {cr['merged_ctl']:.2f} | {ca['kept']:.2f} / {ca['kept_ctl']:.2f} "
                         f"| {cr['kept']:.2f} / {cr['kept_ctl']:.2f} | {ca['n_passages']} |")
    chance_line = {n: chance(n) for n in (6, 7)}
    refused = {f"{s}/{p}": r["refused_layers"] for (s, p), r in recs.items() if r["refused_layers"]}
    out = {"table": {f"{s}|{b}|{arm}": c for (s, b, arm), c in table.items()}, "heads_median": heads,
           "chance": chance_line, "refused_layers": refused,
           "records": len(recs), "codes": sorted({r["code"] for r in recs.values()})}
    (a.out / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))
    print(f"\n‡ a control keeps < {CONTROL_FLOOR} of the records: the random cut dissolves the groups too")
    print("chance (no effect, passages independent):", chance_line)
    print("heads (median over 384): ", {s: {k: round(v, 3) for k, v in h.items()} for s, h in heads.items()})
    print("refused layers:", refused or "none")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--steps", type=int, nargs="*")
    r.add_argument("--first-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
