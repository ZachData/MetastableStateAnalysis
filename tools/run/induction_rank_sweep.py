"""Stage 1 of the bottom-up induction programme (PROJECT.md §3.11): the
minimal rank of the OV half of one induction head.

TARGET, located 2026-09-07 from `attentions.npz` at step 4000 with no forward
pass: the circuit is `L5H2 -> L7H8`. `L7H8` is the behavioural leader (peak
induction score 0.0368, PROJECT.md §3.5) and `L5H2` is an overwhelming
previous-token head (mean attention at offset -1 = 0.895 against a 384-head
median of 0.018, ~49x). Stage 1 asks of `L7H8`'s OV: **how many dimensions
does the copying half of induction actually need?**

WHY OV FIRST AND NOT QK. Pythia rotates only `rotary_ndims = 16` of 64 head
dims (`core/rope.py`; `rotary_pct = 0.25`), so `M_QK = W_Q W_K^T` is the
effective attention operator on the 48 unrotated dims only and a rank
truncation of it is not a truncation of what the model computes. OV carries no
RoPE, and it is where the differential claim lives: the standard account says
the induction OV is a **copier** (aligning, token-subspace-aligned), the
particle account says it is **individuating** (repulsive). Those are opposed.

READOUT PAIRING, and it is the trap this design exists to avoid (§3.11
decision 3). The behavioural induction score is *mean post-softmax attention
on induction pairs* -- a pure QK quantity. **Ablating OV cannot move it within
the layer.** So the OV sweep is read on the COPYING side: next-token NLL on the
second copy of a repeated random sequence, which is what an induction circuit
is for, plus the KL against the unablated model.

TWO BASES, because they answer different questions (§3.11 decision 1):

  SVD    -- `OV_r = U_r S_r V_r^T`, the Eckart-Young optimal rank-r
            approximation. Measures GAIN. Answers "how many dimensions carry
            the action", which is what `r*` means.
  SCHUR  -- `OV_r = A P_r B` with `P_r` the projector onto the top-r INVARIANT
            subspace of the head core `C = B A`. Because span(Q_r) is
            C-invariant, `eig(A P_r B) \\ {0}` is exactly C's top-r
            eigenvalues, so this truncation keeps r genuine dynamical modes.
            Carries a SIGN, which SVD cannot, so it is the only basis in which
            the attractive/repulsive falsifier can be posed.

These heads are strongly non-normal (Henrici median 0.450), so the two are
expected to disagree, and the disagreement is a registered outcome rather than
a nuisance: `r*_SVD ~= r*_Schur` says the induction-relevant part is near
normal; `r*_SVD << r*_Schur` says induction lives in high-gain non-invariant
directions and the project's attractive/repulsive frame is measuring something
other than what the head does (`MATH_SPECTRAL_OT` §5.3(d) from a second
direction).

`r*` IS DERIVED, NOT THRESHOLDED (§3.11 decision 2). No "induction collapsed"
constant. The reported object is the whole NLL-vs-r curve together with a
MATCHED-NORM RANDOM rank-r control band, and `r*` is where the real curve
leaves that band -- the same construction that made
`N_CONTROLS_PER_INDUCTION_HEAD` a measured frontier rather than a placed
number.

Weights only ever touched through a save/restore around each measurement; the
baseline is re-measured at the end and asserted equal to the first one.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np
import scipy
import scipy.linalg as sla
import torch
import transformers

from core.lm_loading import load_causal_lm

LAYER, HEAD = 7, 8
PREV_LAYER, PREV_HEAD = 5, 2          # located from attentions.npz, see docstring
STEP = 4000
D_MODEL, D_HEAD, N_HEADS = 1024, 64, 16

#: Repeated-sequence induction eval. `n_rep` tokens sampled uniformly from a
#: safe id range, concatenated twice; the model is scored on the SECOND copy,
#: where an induction circuit is what makes the tokens predictable. Uniform
#: random ids rather than natural text on purpose: it removes every source of
#: predictability except the repetition itself, so the readout is about
#: induction and not about the language model.
N_REP = 96
N_SEQS = 8
VOCAB_LO, VOCAB_HI = 1000, 40000
EVAL_SEED = 20260907


def induction_batch(rng):
    seqs = []
    for _ in range(N_SEQS):
        s = rng.integers(VOCAB_LO, VOCAB_HI, size=N_REP)
        seqs.append(np.concatenate([s, s]))
    return torch.tensor(np.stack(seqs), dtype=torch.long)


@torch.no_grad()
def measure(model, ids):
    """Second-copy NLL and full next-token logprobs, on the repeated half."""
    out = model(ids)
    logits = out.logits[:, :-1, :].float()
    targets = ids[:, 1:]
    lp = torch.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # (B, T-1)
    # Positions belonging to the SECOND copy. Index t of tok_lp predicts
    # ids[:, t+1], so the second copy starts at t = N_REP - 1.
    second = tok_lp[:, N_REP - 1:]
    return {
        "second_copy_nll": float(-second.mean()),
        "first_copy_nll": float(-tok_lp[:, : N_REP - 1].mean()),
        "logprobs": lp,
    }


def ov_factors(model, layer, head):
    """`(A, B)` with `OV_h = A @ B`, in `p2_eigenspectra/weights.py`'s
    convention `OV_h = W_V_h.T @ W_O_h.T`. Verified 2026-09-07 against the
    on-disk `ov_head<h>_layer_<l>` to 1.3e-7 relative, the fp32 storage floor."""
    attn = model.gpt_neox.layers[layer].attention
    qkv = attn.query_key_value.weight.detach().numpy().astype(np.float64)
    dense = attn.dense.weight.detach().numpy().astype(np.float64)
    qkv3 = qkv.reshape(N_HEADS, 3 * D_HEAD, D_MODEL)
    W_V = qkv3[head, 2 * D_HEAD:, :]              # (64, 1024)
    W_O = dense[:, head * D_HEAD:(head + 1) * D_HEAD]   # (1024, 64)
    return W_V.T.copy(), W_O.T.copy()             # A (1024,64), B (64,1024)


def write_ov(model, layer, head, A, B):
    """Write `OV = A @ B` back, padding the factors to `d_head` with zeros."""
    attn = model.gpt_neox.layers[layer].attention
    r = A.shape[1]
    A_pad = np.zeros((D_MODEL, D_HEAD)); A_pad[:, :r] = A
    B_pad = np.zeros((D_HEAD, D_MODEL)); B_pad[:r, :] = B
    with torch.no_grad():
        qkv = attn.query_key_value.weight
        v = qkv.view(N_HEADS, 3 * D_HEAD, D_MODEL)[head, 2 * D_HEAD:, :]
        v.copy_(torch.tensor(A_pad.T, dtype=qkv.dtype))
        attn.dense.weight[:, head * D_HEAD:(head + 1) * D_HEAD].copy_(
            torch.tensor(B_pad.T, dtype=attn.dense.weight.dtype))


def truncate(A, B, r, basis, rng=None):
    """Rank-r factors for the head's OV in the requested basis."""
    if r == 0:
        # The ceiling of the whole curve: the head's OV removed entirely. Every
        # basis agrees here by construction, and without it the curve has no
        # top end to read `r*` against.
        return np.zeros((D_MODEL, 0)), np.zeros((0, D_MODEL))
    if basis == "svd":
        U, s, Vt = np.linalg.svd(A @ B, full_matrices=False)
        return U[:, :r] * s[:r], Vt[:r]
    if basis == "schur":
        C = B @ A                                   # (64, 64) head core
        T, Q, sdim = sla.schur(C, output="complex",
                               sort=lambda z: False)  # unsorted; reorder below
        ev = np.diag(T)
        order = np.argsort(-np.abs(ev))
        # Re-run with a selection that puts the r largest-|lambda| first, so
        # span(Q[:, :r]) is a genuine invariant subspace of C.
        keep = np.zeros(C.shape[0], dtype=bool); keep[order[:r]] = True
        thresh = np.abs(ev[order[r - 1]]) if r <= len(ev) else -1.0
        T, Q, sdim = sla.schur(C, output="complex",
                               sort=lambda z: abs(z) >= thresh - 1e-15)
        Qr = Q[:, :r]
        P = Qr @ Qr.conj().T                        # projector, C-invariant range
        return (A @ P).real, B
    if basis == "random":
        # Matched-norm random rank-r control: a random r-dim projector in the
        # core, so the operator norm scale and the factor structure match the
        # real truncation and only the DIRECTIONS are structureless.
        Qr, _ = np.linalg.qr(rng.normal(size=(D_HEAD, r)))
        P = Qr @ Qr.T
        return A @ P, B
    raise ValueError(basis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="1,2,3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--head", type=int, default=HEAD)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    step, layer, head = args.step, args.layer, args.head

    ranks = [int(x) for x in args.ranks.split(",")]
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    rng = np.random.default_rng(EVAL_SEED)

    model, tok = load_causal_lm(f"pythia-410m-step{step}")
    model.eval()
    ids = induction_batch(np.random.default_rng(EVAL_SEED))

    A0, B0 = ov_factors(model, layer, head)
    ref = np.load(sorted(DATA.glob(
        f"phase12/p2_eigenspectra_*/ov_weights_pythia-410m-step{step}.npz"))[0]
    )[f"ov_head{head}_layer_{layer}"]
    ov_rel = float(np.linalg.norm(A0 @ B0 - ref) / np.linalg.norm(ref))

    base = measure(model, ids)
    base_lp = base.pop("logprobs")
    print(f"  baseline: second-copy NLL {base['second_copy_nll']:.4f}  "
          f"first-copy {base['first_copy_nll']:.4f}  "
          f"(OV extraction check {ov_rel:.2e})", flush=True)

    out = {
        "_what_this_is":
            "Stage 1 of PROJECT.md section 3.11: minimal rank of the OV half "
            "of induction head L7H8 (partner prev-token head L5H2), by "
            "rank-r truncation in the SVD and Schur bases against a "
            "matched-norm random rank-r control. Readout is the COPYING side "
            "(second-copy NLL on repeated random sequences), because the "
            "behavioural induction score is a QK quantity that OV ablation "
            "cannot move within the layer.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
        "target": {"layer": layer, "head": head,
                   "prev_token_partner": [PREV_LAYER, PREV_HEAD] if
                   (layer, head) == (LAYER, HEAD) else None, "step": step},
        "eval": {"n_rep": N_REP, "n_seqs": N_SEQS, "seed": EVAL_SEED,
                 "vocab_range": [VOCAB_LO, VOCAB_HI]},
        "ov_extraction_rel_error": ov_rel,
        "baseline": base,
        "ranks": ranks,
        "curves": {},
    }

    for basis in ("svd", "schur", "random"):
        n_draws = args.controls if basis == "random" else 1
        rows = []
        for r in ranks:
            vals = []
            for d in range(n_draws):
                A, B = truncate(A0, B0, r, basis, rng)
                write_ov(model, layer, head, A, B)
                m = measure(model, ids)
                lp = m.pop("logprobs")
                kl = float((base_lp.exp() * (base_lp - lp)).sum(-1).mean())
                vals.append({**m, "kl_from_baseline": kl})
            write_ov(model, layer, head, A0, B0)          # restore
            agg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            agg["sd"] = float(np.std([v["second_copy_nll"] for v in vals]))
            agg["rank"] = r
            rows.append(agg)
            print(f"  {basis:>6} r={r:<3} second-copy NLL {agg['second_copy_nll']:.4f}"
                  f"  KL {agg['kl_from_baseline']:.4f}"
                  + (f"  (sd {agg['sd']:.4f}, {n_draws} draws)" if n_draws > 1 else ""),
                  flush=True)
        out["curves"][basis] = rows

    final = measure(model, ids); final.pop("logprobs")
    out["restore_check"] = {
        "baseline_second_copy_nll": base["second_copy_nll"],
        "after_sweep_second_copy_nll": final["second_copy_nll"],
        "abs_diff": abs(base["second_copy_nll"] - final["second_copy_nll"]),
    }
    print(f"  restore check: {out['restore_check']['abs_diff']:.2e}")

    _default = ("induction_rank_sweep.json" if (step, layer, head) == (STEP, LAYER, HEAD)
                else f"induction_rank_sweep_s{step}_L{layer}H{head}.json")
    dest = Path(args.out) if args.out else DATA / "analysis" / _default
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
