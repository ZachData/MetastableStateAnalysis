"""Do the members hold ONE subspace, and did they always? The geometric half.

WHY THIS EXISTS
---------------
The phase's central claim is that the members are **functionally redundant and
structurally distinct** (§3.12-S) -- they cover for one another without sharing
machinery. That claim has so far rested on **one scalar for one pair**: the
cosine between the two heads' mean residual-delta vectors, 0.868 at step 16000.

A cosine between two mean vectors is a **rank-1** summary of an effect that is
not rank-1. It cannot distinguish the two histories that matter here, and they
are the ones the redundancy question turns on:

- **Converged.** Each head starts with its own private effect subspace and they
  grow *into* a common one -- redundancy is acquired, and is a property the
  training process built.
- **Co-located from birth.** They occupy one subspace from the moment they have
  any effect at all, and what grows afterwards is the *extent* of that shared
  space, not its overlap -- redundancy is a by-product of where induction lives.

Both produce a high cosine at step 16000, which is the only place anyone has
looked. They differ everywhere before it.

THE INSTRUMENT. Ablate one head, keep the residual delta at **every** copied
position rather than averaging it, and take the row space of that
`(n_seqs*n_pos, d_model)` matrix. That is the head's **effect subspace** -- the
set of directions in the residual stream that this head is actually responsible
for. Then, per checkpoint:

1. **Effective rank** (participation ratio, and the 90 %-energy rank). Does one
   head's own subspace expand over training, or is it born its final size?
2. **Principal angles** between members' subspaces -- the real version of the
   rank-1 cosine, and it can see overlap that the mean vectors cancel out of.
3. **Coverage**: the fraction of head A's effect energy lying in the span of all
   the OTHER members. This is the redundancy question stated geometrically --
   "is what A does already done elsewhere" -- and it is the quantity the causal
   interaction matrix can only infer.
4. **Union rank vs sum of ranks.** If the members expanded into the same place,
   the union subspace stays about as large as one member's; if each kept its own,
   the union approaches the sum. This single comparison separates the two
   histories above, and it is threshold-free.

WHY GEOMETRIC AND NOT CAUSAL, HERE. `dNLL` has a hard ceiling at
`ln 50304 = 10.83` and §3.14.4-D shows the interesting checkpoints -- the
`(512, 4000]` window where every member forms -- are pressed against it, where
magnitudes become floors and interactions are compressed toward a false
sub-additivity. **Nothing in this file reads `dNLL`.** design-7d.md names the
residual-delta geometry as one of the two ceiling-immune instruments that must
carry the weight in exactly that window; this is that instrument, taken past
rank 1. `dNLL` is still recorded per arm, but only so rows can be cross-checked
against §3.12-U -- never as the measurement.

THE NULL, WHICH THIS MEASUREMENT IS USELESS WITHOUT. A member's effect subspace
runs to 120-200 of 1024 dimensions, so "80 % of head A's energy lies in the span
of the others" sounds decisive and may be nothing: `k/d_model` is the right
chance value only if the residual stream is isotropic, and it is emphatically
not -- its energy is concentrated in a few hundred directions that every head
writes into. Measured against that flat baseline, two heads with no relationship
at all would look redundant.

So the baseline is **measured, not assumed**: `--controls` ablates three
near-median heads from the catalogue (`L5H5` +0.00108, `L8H3` +0.00148, `L12H15`
+0.00099, against a median of +0.00107), **layer-matched** to the member spread.
They have no causal effect, but their deltas live in the same anisotropic
residual stream, so they give the honest null for every quantity here -- the
noise floor, the coverage fraction, and the principal-angle overlap. Each is
reported beside its null; where a member's number is not clear of it, the number
is not a finding. `sqrt(k/d)` is still recorded as `chance_isotropic` so the two
baselines can be compared, but it is the weaker of the two and should not be
quoted alone.

WHAT THE EARLY STEPS DO NOT SUPPORT. Below the formation window a member's
delta is numerical noise, and the row space of noise is a full-rank random
subspace whose overlap with anything is the chance value. Every row carries
`delta_norm` and `delta_rel` (delta against the baseline residual norm), and a
member within `--noise-mult` of the largest control's `delta_rel` at that
checkpoint is flagged `below_noise_floor`. **Do not read overlap where the
effect does not exist.**

Reported BOTH ways per §3.13: the mean-view (participation ratio, mean principal
angle) and the extremum-view (90 %-energy rank, max principal angle). Neither is
chosen after seeing the data.

COST: `1 + n + c` forward arms per checkpoint. The `1 + n` are the same arms as
`member_formation_curves.py`, which discards everything this file needs; only
the `c` control arms are new, and they are the reason any of it is readable.

NO p-value; nothing registered. pythia-410m is spent under `check_registry`
rule 3 -- exploratory, and not registrable on this data.
"""
import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
if not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import (
    D_MODEL, N_REP, EVAL_SEED, ov_factors, write_ov,
)

from p7d_redundancy.member_formation_curves import ALL_STEPS, batch, members

OUT = DATA / "analysis" / "member_subspace_geometry.json"

#: Near-median catalogue heads (median dNLL +0.00107), layer-matched to the
#: members' spread over layers 5-12. These carry the null for every overlap
#: quantity in this file -- see the docstring's NULL section. Chosen from
#: `redundancy_catalog.json` by distance to the median, before any geometry was
#: computed, so they are not a post-hoc baseline.
CONTROLS = "L5H5,L8H3,L12H15"


@torch.no_grad()
def resid_matrix(model, ids, chunk=4):
    """(second-copy NLL, final-layer residual at every copied position).

    The NLL is the same estimator as `member_formation_curves.probe` and rank 0
    of `induction_rank_sweep`, scored only from position `N_REP - 1`, so the
    `dNLL` column here is directly comparable with §3.12-U. The matrix is what
    that function throws away: `(n_seqs * n_pos, d_model)` rather than its
    column mean.
    """
    lc, lz, rows = [], [], []
    for i in range(0, len(ids), chunk):
        out = model(ids[i:i + chunk], output_hidden_states=True)
        lg = out.logits[:, :-1, :].float()
        t = ids[i:i + chunk, 1:]
        z = torch.logsumexp(lg, dim=-1)
        c = lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        lc.append(c[:, N_REP - 1:].numpy().astype(np.float64))
        lz.append(z[:, N_REP - 1:].numpy().astype(np.float64))
        h = out.hidden_states[-1][:, N_REP - 1:, :].float()
        rows.append(h.reshape(-1, D_MODEL).numpy().astype(np.float64))
        del out, lg, z, c, h
    return (float(-(np.concatenate(lc).mean() - np.concatenate(lz).mean())),
            np.concatenate(rows, axis=0))


def spectrum(D):
    """Singular values of an effect matrix, largest first."""
    return np.linalg.svd(D, compute_uv=False)


def basis(D, frac):
    """Orthonormal basis (d_model, k) for the `frac`-of-energy row space of `D`.

    The row space is the one that lives in the residual stream: `D` is
    (positions, d_model), so its right singular vectors are directions a reader
    downstream could see, and `k` is chosen by energy rather than fixed so a
    head whose effect is genuinely low-rank is not padded out with noise
    directions.
    """
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    e = s ** 2
    k = int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)
    return Vt[:k].T, k


def participation_ratio(s):
    """(Σσ²)² / Σσ⁴ -- effective dimension, insensitive to a threshold.

    1.0 for a rank-1 effect, `r` for `r` equal directions. The mean-view
    counterpart to the 90 %-energy rank, per §3.13.
    """
    e = s ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())


def angles(A, B):
    """(mean, max) principal-angle cosine between two orthonormal bases."""
    s = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1.0, 1.0)
    return float(s.mean()), float(s.max())


def cka(X, Y, center=False):
    """Linear CKA between two effect matrices sharing the same rows.

    The energy-weighted answer to "do these two heads do the same thing", and
    the measure the unweighted principal angles in `angles()` should be read
    beside rather than instead of. `angles()` orthonormalises first, so a
    direction carrying 40 % of a head's effect and one carrying 0.5 % count the
    same; CKA weights every direction by its singular value, is invariant to
    rotation, and needs no energy threshold.

    `center=False` keeps the mean delta, which is the rank-1 direction §3.12-S
    measured and is signal here rather than nuisance. `center=True` removes it
    and so reports only whether the two heads co-vary *across positions* --
    running both separates "writes the same direction" from "responds to the
    same tokens".
    """
    if center:
        X, Y = X - X.mean(0), Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy)) if xx and yy else float("nan")


def ambient_profile(D, V, s_amb, bins=(10, 50, 200)):
    """Where a head's effect sits along the residual stream's OWN spectrum.

    `V` are the baseline residual's right singular vectors, ordered by ambient
    variance, so index 0 is the direction the residual stream uses most. A head
    writing into the leading directions is competing for the trunk everything
    else uses; one writing far down the spectrum has found private bandwidth.

    This is also the mechanical explanation for `coverage_by_others` saturating:
    if the ambient spectrum is concentrated and every head writes into the same
    leading directions, any large pooled subspace captures everything.
    """
    e = (D @ V) ** 2
    per = e.sum(0)
    tot = per.sum()
    idx = np.arange(len(per))
    return {"energy_frac_in_ambient_top": {str(b): float(per[:b].sum() / tot)
                                           for b in bins},
            "weighted_mean_ambient_index": float((per * idx).sum() / tot),
            "ambient_participation_ratio": participation_ratio(s_amb)}


def coverage(D, Q):
    """Fraction of `D`'s energy lying in the subspace spanned by `Q`.

    The redundancy question stated geometrically: with `Q` the span of every
    OTHER member, this is how much of what this head does is already done
    elsewhere. Chance for a `k`-dimensional `Q` in `R^1024` is `k / 1024`, which
    the caller reports beside it -- a large coverage from a large `Q` is not a
    finding.
    """
    return float((np.linalg.norm(D @ Q) ** 2) / (np.linalg.norm(D) ** 2))


def orth(cols):
    """Orthonormal basis for the union of several subspaces."""
    M = np.concatenate(cols, axis=1)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > s[0] * 1e-10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in ALL_STEPS))
    ap.add_argument("--top", type=int, default=6,
                    help="catalogue members; 6 matches the §3.12-U curve, so "
                         "the dNLL column is a reproduction check")
    ap.add_argument("--heads", default="", help="explicit 'L5H2,L7H8' override")
    ap.add_argument("--seqs", type=int, default=16,
                    help="16 matches §3.12-S and §3.12-U")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--energy", type=float, default=0.90,
                    help="energy fraction defining a head's effect subspace")
    ap.add_argument("--controls", default=CONTROLS,
                    help="near-median heads carrying the measured null; the "
                         "isotropic k/d chance value is not a usable baseline "
                         "in this residual stream -- see __doc__")
    ap.add_argument("--noise-mult", type=float, default=2.0,
                    help="a member whose delta_rel is under this multiple of "
                         "the largest control's is flagged unreadable")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    def parse(spec):
        return [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:]))
                for h in spec.split(",") if h]

    if args.heads:
        heads, source = parse(args.heads), "--heads"
    else:
        heads, source = members(args.top)
    ctrls = [k for k in parse(args.controls) if k not in heads]
    steps = [int(x) for x in args.steps.split(",") if x]
    names = {k: f"L{k[0]}H{k[1]}" for k in heads + ctrls}

    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    res = {"_what_this_is": __doc__, "git_sha": git_sha,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "membership_source": source, "d_model": D_MODEL,
           "eval": {"n_seqs": args.seqs, "seed": EVAL_SEED, "n_rep": N_REP,
                    "energy_frac": args.energy, "noise_mult": args.noise_mult,
                    "scored_from_position": N_REP - 1},
           "heads": [names[k] for k in heads],
           "controls": [names[k] for k in ctrls],
           "steps": steps, "per_step": {}}
    if args.append and out.exists():
        prev = json.load(open(out))
        res["per_step"] = prev["per_step"]
        res["steps"] = sorted(set(prev["steps"]) | set(steps))

    print(f"members:  {', '.join(names[k] for k in heads)}   ({source})")
    print(f"controls: {', '.join(names[k] for k in ctrls)}   "
          f"(near-median dNLL; they carry the null)")
    print(f"grid:     {len(steps)} steps, {args.seqs} seqs, "
          f"{1 + len(heads) + len(ctrls)} arms per step, effect subspace = "
          f"{args.energy:.0%} of energy\n")

    Z = (np.zeros((D_MODEL, 1)), np.zeros((1, D_MODEL)))
    for s in steps:
        model, _ = load_causal_lm(f"pythia-410m-step{s}")
        model.eval()
        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs)

        nll0, R0 = resid_matrix(model, ids, args.chunk)
        base_norm = float(np.linalg.norm(R0))
        # The residual stream's own spectrum, so every head's effect can be
        # placed against the ambient importance ordering rather than treated as
        # living in an isotropic R^1024.
        _, s_amb, Vt_amb = np.linalg.svd(R0, full_matrices=False)
        V_amb = Vt_amb.T

        D, B, K, per_head = {}, {}, {}, {}
        for k in heads + ctrls:
            saved = ov_factors(model, *k)
            write_ov(model, *k, *Z)
            nll, R = resid_matrix(model, ids, args.chunk)
            write_ov(model, *k, *saved)

            D[k] = R - R0
            sv = spectrum(D[k])
            B[k], K[k] = basis(D[k], args.energy)
            dn = float(np.linalg.norm(D[k]))
            per_head[names[k]] = {
                "is_control": k in ctrls,
                "dnll": nll - nll0,
                "delta_norm": dn, "delta_rel": dn / base_norm,
                # §3.13, both views: PR is the mean-view effective dimension,
                # r_energy the extremum-view one.
                "participation_ratio": participation_ratio(sv),
                "r_energy": K[k],
                "top_singular_values": sv[:16].tolist(),
                "energy_in_top1": float(sv[0] ** 2 / (sv ** 2).sum()),
                **ambient_profile(D[k], V_amb, s_amb)}
            del R
            gc.collect()

        # The noise floor is measured at this checkpoint rather than assumed:
        # the largest control delta is what "no causal effect" looks like here.
        ctrl_rel = max(per_head[names[k]]["delta_rel"] for k in ctrls)
        for k in heads + ctrls:
            h = per_head[names[k]]
            h["control_delta_rel"] = ctrl_rel
            h["below_noise_floor"] = bool(
                k in heads and h["delta_rel"] < args.noise_mult * ctrl_rel)

        # --- pairwise geometry, member-member and member-control --------------
        # Every member-member cell is paired with the same quantity computed
        # against the controls, because a principal-angle cosine has no absolute
        # scale in an anisotropic residual stream.
        def cell(a, b):
            m, mx = angles(B[a], B[b])
            ma, mb = D[a].mean(0), D[b].mean(0)
            return {"subspace_mean_cos": m, "subspace_max_cos": mx,
                    # Energy-weighted, threshold-free, and the one to quote:
                    # the unweighted `subspace_*_cos` above counts a 0.5 %
                    # direction the same as a 40 % one.
                    "cka": cka(D[a], D[b]),
                    "cka_centered": cka(D[a], D[b], center=True),
                    # The rank-1 quantity §3.12-S reported, kept so this file is
                    # continuous with it rather than a separate scale.
                    "mean_delta_cosine": float(
                        np.dot(ma, mb) /
                        (np.linalg.norm(ma) * np.linalg.norm(mb))),
                    "chance_isotropic": float(np.sqrt(min(K[a], K[b]) / D_MODEL)),
                    "cross_coverage_a_in_b": coverage(D[a], B[b]),
                    "cross_coverage_b_in_a": coverage(D[b], B[a])}

        pair = {f"{names[a]}x{names[b]}": cell(a, b)
                for a, b in itertools.combinations(heads, 2)}
        null_pair = {f"{names[a]}x{names[b]}": cell(a, b)
                     for a in heads for b in ctrls}
        null_pair.update({f"{names[a]}x{names[b]}": cell(a, b)
                          for a, b in itertools.combinations(ctrls, 2)})
        null_mean_cos = float(np.mean([c["subspace_mean_cos"]
                                       for c in null_pair.values()]))

        # --- coverage by the rest of the set, against the measured null -------
        # SATURATES, and the guard below is not decoration. Pooling five members'
        # 90 %-energy bases spans 742-819 of 1024 dimensions from step 1000 on,
        # and *everything* scores ~0.97 against a subspace that large -- members
        # 0.970, controls 0.969. Read `cross_coverage_*` in `pairs` instead,
        # which is against a single head's ~150-dim basis and stays informative.
        for k in heads + ctrls:
            others = [B[j] for j in heads if j != k]
            Q = orth(others)
            h = per_head[names[k]]
            h["coverage_by_others"] = coverage(D[k], Q)
            h["coverage_by_others_dim"] = int(Q.shape[1])
            h["coverage_by_others_chance_isotropic"] = float(Q.shape[1] / D_MODEL)
            h["coverage_saturated"] = bool(Q.shape[1] > D_MODEL / 3)
        # A control's coverage by the members' span is what "already covered"
        # scores when there is nothing to cover -- the baseline the members must
        # clear to mean anything.
        cov_null = float(np.mean([per_head[names[k]]["coverage_by_others"]
                                  for k in ctrls]))
        for k in heads + ctrls:
            per_head[names[k]]["coverage_by_others_null"] = cov_null

        # --- the union, which is the whole question ---------------------------
        # If the members expanded into the SAME place the union subspace stays
        # about as large as one member; if each kept its own it approaches the
        # sum. `union_ratio` near 1/n is one shared subspace, near 1 is n
        # private ones.
        #
        # NOT COMPARABLE ACROSS CHECKPOINTS OVER A FIXED HEAD LIST, which the
        # first version of this file got wrong. A head that has not formed yet
        # contributes its full participation ratio to the denominator while
        # contributing almost nothing to the union, so the ratio falls as
        # members form and that fall is arithmetic, not geometry. At step 1000
        # four of the six are still at noise and the n=6 ratio reads 0.266 for
        # that reason alone. `union_formed` is therefore the one to read across
        # training: it is restricted to the members above the checkpoint's own
        # measured noise floor, with `n` recorded so a changing membership is
        # visible rather than silent.
        def union(group):
            st = np.concatenate([D[k] for k in group], axis=0)
            u_pr = participation_ratio(spectrum(st))
            _, u_r = basis(st, args.energy)
            s_pr = sum(per_head[names[k]]["participation_ratio"] for k in group)
            s_r = sum(K[k] for k in group)
            return {"n": len(group),
                    "union_participation_ratio": u_pr, "union_r_energy": u_r,
                    "sum_participation_ratio": s_pr, "sum_r_energy": s_r,
                    "union_ratio_pr": u_pr / s_pr if s_pr else None,
                    "union_ratio_r": u_r / s_r if s_r else None}

        formed = [k for k in heads if not per_head[names[k]]["below_noise_floor"]]
        u_all = union(heads)
        u_matched = union(heads[:len(ctrls)])
        u_null = union(ctrls)
        # The readable one: formed members only, and the controls cut to the
        # same n so the comparison is not between different-sized groups.
        u_formed = union(formed) if len(formed) > 1 else None
        u_formed_null = union(ctrls[:len(formed)]) if len(formed) > 1 else None

        nll_chk, _ = resid_matrix(model, ids, args.chunk)
        rec = {"baseline_nll": nll0, "baseline_resid_norm": base_norm,
               "restore_abs_diff": abs(nll_chk - nll0),
               "heads": per_head, "pairs": pair, "null_pairs": null_pair,
               "null_subspace_mean_cos": null_mean_cos,
               "coverage_null": cov_null, "control_delta_rel": ctrl_rel,
               "formed_members": [names[k] for k in formed],
               "union_formed": u_formed, "union_formed_null": u_formed_null,
               "union_members": u_all, "union_members_size_matched": u_matched,
               "union_controls": u_null}
        res["per_step"][str(s)] = rec

        flag = [n for n, h in per_head.items() if h["below_noise_floor"]]
        print(f"step {s:>6}  baseline NLL {nll0:>7.4f}  restore "
              f"{abs(nll_chk - nll0):.1e}")
        print(f"{'':>10}{'head':>8} {'dNLL':>8} {'|d|/|R|':>8} {'PR':>7} "
              f"{'r90':>5} {'top1':>6} {'cov(rest)':>10}")
        for k in heads:
            h = per_head[names[k]]
            print(f"{'':>10}{names[k]:>8} {h['dnll']:>+8.4f} "
                  f"{h['delta_rel']:>8.4f} {h['participation_ratio']:>7.2f} "
                  f"{h['r_energy']:>5} {h['energy_in_top1']:>6.3f} "
                  f"{h['coverage_by_others']:>10.3f}"
                  f"{'   (noise)' if h['below_noise_floor'] else ''}")
        for k in ctrls:
            h = per_head[names[k]]
            print(f"{'':>10}{names[k]:>8} {h['dnll']:>+8.4f} "
                  f"{h['delta_rel']:>8.4f} {h['participation_ratio']:>7.2f} "
                  f"{h['r_energy']:>5} {h['energy_in_top1']:>6.3f} "
                  f"{h['coverage_by_others']:>10.3f}   NULL")
        print(f"{'':>10}coverage null {cov_null:.3f}   |   "
              f"subspace cos: members "
              f"{np.mean([c['subspace_mean_cos'] for c in pair.values()]):.3f} "
              f"vs null {null_mean_cos:.3f}")
        print(f"{'':>10}CKA (energy-weighted): members "
              f"{np.mean([c['cka'] for c in pair.values()]):.3f} vs null "
              f"{np.mean([c['cka'] for c in null_pair.values()]):.3f}   |   "
              f"centered: members "
              f"{np.mean([c['cka_centered'] for c in pair.values()]):.3f} vs "
              f"null {np.mean([c['cka_centered'] for c in null_pair.values()]):.3f}")
        print(f"{'':>10}ambient: residual PR {participation_ratio(s_amb):.1f} "
              f"of {D_MODEL}; member energy in ambient top-10 "
              f"{np.mean([per_head[names[k]]['energy_frac_in_ambient_top']['10'] for k in heads]):.3f}, "
              f"top-50 "
              f"{np.mean([per_head[names[k]]['energy_frac_in_ambient_top']['50'] for k in heads]):.3f}")
        if u_formed:
            print(f"{'':>10}union ratio (PR), FORMED only "
                  f"[{', '.join(names[k] for k in formed)}]: "
                  f"{u_formed['union_ratio_pr']:.3f}   vs null(n="
                  f"{u_formed_null['n']}) {u_formed_null['union_ratio_pr']:.3f}")
        else:
            print(f"{'':>10}union ratio: fewer than two members formed -- "
                  f"not defined at this checkpoint")
        print(f"{'':>10}(fixed-list union, NOT comparable across steps: "
              f"n={u_all['n']} {u_all['union_ratio_pr']:.3f}, "
              f"controls {u_null['union_ratio_pr']:.3f})")
        if flag:
            print(f"{'':>10}below noise floor, overlap NOT readable: "
                  f"{', '.join(flag)}")
        print(flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)

        del model, ids, R0, D, B
        gc.collect()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
