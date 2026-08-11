"""
p5_single_mstate_analysis/sweep_geometry.py — Group A, re-cut as per-particle
geometry measured across the checkpoint sweep.

What changed from the original Group A
--------------------------------------
`cluster_profile.compute_profile` reports cluster-level compactness at one
checkpoint. Here the frozen token set from `token_sets.py` is measured at
every checkpoint, and every quantity is computed per particle first, with the
set-level number as an aggregate over it. The cluster-level view is then a
groupby rather than a separate code path, and the per-particle rows are what
`core/particles.py` and Phase 5c consume.

Three defects inherited from the old path, fixed here
-----------------------------------------------------

**Raw effective rank is not usable across a sweep (status-1 D1).**
`metrics.effective_rank(mode="raw")` runs the SVD on unnormalized vectors, so
its spectrum is dominated by whichever tokens have the largest norms.
Residual-stream norms grow across training, so a raw-rank trajectory over
143k steps measures norm growth as much as directional structure. Everything
here is computed in BOTH modes and reported side by side: `normed` is the
comparable quantity, `raw` is kept because raw-minus-normed divergence is
itself the norm-growth signal, and hiding it would make D1 invisible rather
than fixed.

**`dual_reading.effective_rank_contribution` hardcodes `mode="raw"` (B12).**
So the per-particle contribution it returns is largely "how big is this
token's norm." Correct for the degeneracy gate it was written for, wrong for
a developmental comparison. `particle_rank_contributions` below takes an
explicit mode and computes both.

**Position 0 is the NeoX attention sink.** No BOS is prepended, so position 0
carries a norm one to two orders above every other token and can single-
handedly set the raw spectrum. `core/frames.pos0_mask` exists for this. The
policy here is `excluded` by default for population aggregates — and the
exclusion is applied as a MASK, never by reindexing. Token positions are the
identity of a particle across the whole sweep; dropping row 0 from an array
would silently shift every position by one and there is no way to detect that
downstream. If position 0 is itself a member of a token set, that is reported
as a note rather than quietly handled.

Cost
----
Leave-one-out effective rank is computed from the (n, n) Gram's eigenvalues,
not from a fresh SVD of the (n, d) matrix — the nonzero spectra are identical,
and n is a few hundred while d is a thousand or more. `eff_rank_from_gram` is
checked against the SVD definition in the tests rather than assumed
equivalent.

Measured, not estimated: at Pythia-410M / wiki_paragraph scale (264 tokens,
d=1024, 55 particles across the three roles) one layer takes ~570 ms with
both contribution modes, so a 25-layer 27-checkpoint sweep is ~6.5 min per
token set — ~1.7 h for two anchors across eight prompts. `contribution_modes`
controls this directly: dropping to `("normed",)` halves it, and
`with_contributions=False` removes it entirely, leaving the set-level
geometry (which is milliseconds). The default keeps both modes because
raw-minus-normed divergence is the D1 signal; drop `raw` when that has
already been characterised for a given sweep, not before.

Pure numpy. Activations are passed in; the disk reader is a thin wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.run_discovery import RunRef, sweep_for_prompt
from .token_sets import TokenSet

__all__ = [
    "DEFAULT_POS0_POLICY",
    "l2_normalize_rows",
    "eff_rank_from_gram",
    "population_rank",
    "set_geometry",
    "particle_rank_contributions",
    "particle_geometry",
    "layer_geometry",
    "sweep_geometry",
    "make_frame_spec",
    "load_activations",
    "geometry_report_lines",
]


# Aggregates exclude the sink by default. This is one explicit decision
# applied identically at every checkpoint — a trained-vs-init contrast where
# the policy differed would be partly a sink contrast.
DEFAULT_POS0_POLICY = "excluded"

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Rows to unit norm. Zero rows stay zero rather than becoming NaN — a
    zero activation is a real (if rare) state and should not poison a whole
    layer's Gram."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms < _EPS, 1.0, norms)


def eff_rank_from_gram(G: np.ndarray) -> float:
    """
    Spectral-entropy effective rank from an (n, n) Gram matrix.

    exp(-sum p_i log p_i) with p_i the normalized eigenvalues of G. Identical
    to `metrics.effective_rank` on the corresponding (n, d) matrix, because
    the nonzero eigenvalues of X X^T are the squared singular values of X.
    Used so leave-one-out is an (n-1, n-1) eigendecomposition rather than a
    fresh SVD of an (n-1, d) matrix — with n in the hundreds and d in the
    thousands, that is the difference between a sweep taking seconds and
    taking minutes.

    Verified against metrics.effective_rank in tests, not assumed.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.shape[0] == 0:
        return float("nan")
    ev = np.linalg.eigvalsh(G)
    ev = np.clip(ev, 0.0, None)
    total = ev.sum()
    if total < _EPS:
        return 1.0
    p = np.clip(ev / total, 1e-12, None)
    return float(np.exp(-np.sum(p * np.log(p))))


def _gram(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T


def _pos0_keep(n: int, policy: str) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if policy == "excluded" and n > 0:
        keep[0] = False
    elif policy not in ("included", "excluded"):
        raise ValueError(f"pos0 policy must be 'included' or 'excluded', "
                         f"got {policy!r}")
    return keep


# ---------------------------------------------------------------------------
# Population-level
# ---------------------------------------------------------------------------

def population_rank(X: np.ndarray, pos0_policy: str = DEFAULT_POS0_POLICY) -> dict:
    """Effective rank of one layer's full token population, both modes."""
    X = np.asarray(X, dtype=np.float64)
    keep = _pos0_keep(X.shape[0], pos0_policy)
    Xk = X[keep]
    return {
        "n_population": int(Xk.shape[0]),
        "eff_rank_normed": eff_rank_from_gram(_gram(l2_normalize_rows(Xk))),
        "eff_rank_raw": eff_rank_from_gram(_gram(Xk)),
        "mean_norm": float(np.mean(np.linalg.norm(Xk, axis=1))) if Xk.size else float("nan"),
        "median_norm": float(np.median(np.linalg.norm(Xk, axis=1))) if Xk.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Set-level
# ---------------------------------------------------------------------------

def set_geometry(
    X: np.ndarray,
    positions: Sequence[int],
    pos0_policy: str = DEFAULT_POS0_POLICY,
) -> dict:
    """
    Compactness of one token set at one layer.

    All angular quantities are on L2-normalized vectors — the sphere frame —
    so they are comparable across checkpoints without a norm correction.

    `resultant_length` (the norm of the mean unit vector) is the primary
    compactness statistic: it is bounded in [0, 1], scale-free, and 1.0 iff
    every member points the same way. Mean pairwise cosine is reported
    alongside because the two come apart for a set spread over more than a
    hemisphere, and that disagreement is informative rather than noise.

    `separation` = mean within-set cosine minus mean member-to-nonmember
    cosine. Positive means the set is tighter than its surroundings; near zero
    means it is a region of a homogeneous cloud rather than an object.
    """
    X = np.asarray(X, dtype=np.float64)
    n_tokens = X.shape[0]
    pos = np.asarray(sorted(set(int(p) for p in positions)), dtype=np.int64)
    pos = pos[(pos >= 0) & (pos < n_tokens)]

    out: dict = {"n_set": int(pos.size)}
    if pos.size == 0:
        return {**out, "resultant_length": float("nan"),
                "mean_within_cos": float("nan"),
                "mean_between_cos": float("nan"),
                "separation": float("nan"),
                "eff_rank_normed": float("nan"),
                "eff_rank_raw": float("nan"),
                "mean_norm": float("nan"),
                "norm_ratio_to_population": float("nan")}

    U = l2_normalize_rows(X)
    Us = U[pos]

    resultant = float(np.linalg.norm(Us.mean(axis=0)))

    if pos.size >= 2:
        C = Us @ Us.T
        iu = np.triu_indices(pos.size, k=1)
        mean_within = float(np.mean(C[iu]))
    else:
        mean_within = float("nan")

    keep = _pos0_keep(n_tokens, pos0_policy)
    member = np.zeros(n_tokens, dtype=bool)
    member[pos] = True
    nonmember = keep & ~member
    if nonmember.any():
        mean_between = float(np.mean(Us @ U[nonmember].T))
    else:
        mean_between = float("nan")

    norms = np.linalg.norm(X, axis=1)
    pop_norms = norms[keep]

    return {
        **out,
        "resultant_length": resultant,
        "mean_within_cos": mean_within,
        "mean_between_cos": mean_between,
        "separation": (mean_within - mean_between
                       if np.isfinite(mean_within) and np.isfinite(mean_between)
                       else float("nan")),
        "eff_rank_normed": eff_rank_from_gram(_gram(Us)),
        "eff_rank_raw": eff_rank_from_gram(_gram(X[pos])),
        "mean_norm": float(np.mean(norms[pos])),
        "norm_ratio_to_population": (
            float(np.mean(norms[pos]) / np.median(pop_norms))
            if pop_norms.size and np.median(pop_norms) > _EPS else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Per-particle
# ---------------------------------------------------------------------------

def particle_rank_contributions(
    X: np.ndarray,
    positions: Sequence[int],
    mode: str = "normed",
    pos0_policy: str = DEFAULT_POS0_POLICY,
) -> Dict[int, float]:
    """
    Leave-one-out effective-rank contribution for each listed position.

    contribution(i) = eff_rank(population) - eff_rank(population without i)

    Positive means the particle was holding a direction the rest of the
    population did not span; negative means removing it *raised* the rank,
    i.e. it was pulling the spectrum toward one dominant direction. Both are
    real and the sign carries the meaning, so this is not clipped.

    `mode` is explicit and has no default that hides it. `mode="raw"` is what
    `core.dual_reading.effective_rank_contribution` hardcodes, which makes
    that function's output largely a norm statistic (B12) — fine for a
    degeneracy gate, wrong for a cross-checkpoint comparison.

    Only the listed positions are evaluated: the cost is one (n-1, n-1)
    eigendecomposition each, so restricting to the token set's ~60 members
    rather than all ~264 tokens is the difference between a fast sweep and a
    slow one. Pass `range(n)` if the whole population is wanted.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if mode == "normed":
        Xm = l2_normalize_rows(X)
    elif mode == "raw":
        Xm = X
    else:
        raise ValueError(f"mode must be 'normed' or 'raw', got {mode!r}")

    keep = _pos0_keep(n, pos0_policy)
    idx_all = np.where(keep)[0]
    if idx_all.size == 0:
        return {int(p): float("nan") for p in positions}

    G_full = _gram(Xm[idx_all])
    full_rank = eff_rank_from_gram(G_full)
    where = {int(orig): k for k, orig in enumerate(idx_all.tolist())}

    out: Dict[int, float] = {}
    for p in positions:
        p = int(p)
        k = where.get(p)
        if k is None:
            # Excluded by the pos0 policy, or out of range: NaN, not 0.0.
            # 0.0 would read as "contributes nothing", which is a different
            # claim from "was not measured".
            out[p] = float("nan")
            continue
        if G_full.shape[0] <= 1:
            out[p] = float("nan")
            continue
        sub = np.delete(np.delete(G_full, k, axis=0), k, axis=1)
        out[p] = float(full_rank - eff_rank_from_gram(sub))
    return out


def particle_geometry(
    X: np.ndarray,
    positions: Sequence[int],
    reference_positions: Optional[Sequence[int]] = None,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    with_contributions: bool = True,
    contribution_modes: Sequence[str] = ("normed", "raw"),
) -> Dict[int, dict]:
    """
    Per-particle row for each listed position.

    `reference_positions` defines the centroid a particle's angle is measured
    against — the token set it belongs to. Defaults to `positions` itself.
    Keeping it separate matters for the control and sibling roles, whose
    particles should be measured against their own centroid, not the
    primary's.

    Fields: cos_to_centroid, norm, norm_z (against the population's non-sink
    norm distribution), and the two rank contributions.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    U = l2_normalize_rows(X)
    ref = list(positions) if reference_positions is None else list(reference_positions)
    ref = [int(p) for p in ref if 0 <= int(p) < n]

    if ref:
        centroid = U[ref].mean(axis=0)
        cn = np.linalg.norm(centroid)
        centroid = centroid / cn if cn > _EPS else centroid
    else:
        centroid = None

    norms = np.linalg.norm(X, axis=1)
    keep = _pos0_keep(n, pos0_policy)
    pop_norms = norms[keep]
    mu = float(np.mean(pop_norms)) if pop_norms.size else float("nan")
    sd = float(np.std(pop_norms)) if pop_norms.size else float("nan")

    # A mode that was not requested yields NaN, never 0.0 — "not measured"
    # and "measured, contributes nothing" are different claims and must not
    # collapse into the same number.
    contrib: Dict[str, Dict[int, float]] = {
        m: {int(p): float("nan") for p in positions} for m in ("normed", "raw")
    }
    if with_contributions:
        for mode in contribution_modes:
            if mode not in ("normed", "raw"):
                raise ValueError(
                    f"contribution_modes entries must be 'normed' or 'raw', "
                    f"got {mode!r}"
                )
            contrib[mode] = particle_rank_contributions(
                X, positions, mode, pos0_policy)
    contrib_n, contrib_r = contrib["normed"], contrib["raw"]

    rows: Dict[int, dict] = {}
    for p in positions:
        p = int(p)
        if not (0 <= p < n):
            continue
        rows[p] = {
            "token_position": p,
            "cos_to_centroid": (float(U[p] @ centroid)
                                if centroid is not None else float("nan")),
            "norm": float(norms[p]),
            "norm_z": (float((norms[p] - mu) / sd)
                       if np.isfinite(sd) and sd > _EPS else float("nan")),
            "rank_contribution_normed": contrib_n.get(p, float("nan")),
            "rank_contribution_raw": contrib_r.get(p, float("nan")),
        }
    return rows


# ---------------------------------------------------------------------------
# One layer, all roles
# ---------------------------------------------------------------------------

_ROLES = ("primary", "sibling", "control")


def _role_positions(token_set: TokenSet) -> Dict[str, Tuple[int, ...]]:
    return {
        "primary": token_set.positions,
        "sibling": token_set.sibling_positions,
        "control": token_set.control_positions,
    }


def layer_geometry(
    X: np.ndarray,
    token_set: TokenSet,
    layer: int,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    with_contributions: bool = True,
    contribution_modes: Sequence[str] = ("normed", "raw"),
) -> dict:
    """
    One layer's record: population rank, per-role set geometry, per-particle
    rows for every role.

    All three roles are measured identically. Group G's three-tier ordering
    (primary > sibling > random) is only readable if the tiers were measured
    the same way, and the original Group A measured the primary alone.
    """
    X = np.asarray(X, dtype=np.float64)
    rec: dict = {
        "layer": int(layer),
        "population": population_rank(X, pos0_policy),
        "sets": {},
        "particles": [],
    }
    for role, pos in _role_positions(token_set).items():
        if not pos:
            continue
        rec["sets"][role] = set_geometry(X, pos, pos0_policy)
        rows = particle_geometry(X, pos, reference_positions=pos,
                                 pos0_policy=pos0_policy,
                                 with_contributions=with_contributions,
                                 contribution_modes=contribution_modes)
        for p, row in sorted(rows.items()):
            rec["particles"].append({**row, "role": role, "layer": int(layer)})
    return rec


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def make_frame_spec(model_rev: str, layer: Optional[int] = None,
                    pos0_policy: str = DEFAULT_POS0_POLICY):
    """
    FrameSpec for everything this module produces: kind="l2_sphere".

    Imported lazily so the module stays usable (and testable) without
    core.frames on the path. `model_rev` is the checkpoint revision, not the
    model name — distinguishing step1000 from step143000 is the entire point
    of the sweep.
    """
    from core.frames import FrameSpec
    return FrameSpec(kind="l2_sphere", layer_idx=layer, model_rev=model_rev,
                     pos0_policy=pos0_policy)


def load_activations(run_dir: Path) -> Optional[np.ndarray]:
    """(n_layers, n_tokens, d) from activations.npz, or None."""
    path = Path(run_dir) / "activations.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        key = "activations" if "activations" in data else list(data.keys())[0]
        return np.asarray(data[key])
    except Exception:
        return None


def sweep_geometry(
    token_set: TokenSet,
    refs: Sequence[RunRef],
    steps: Optional[Sequence[int]] = None,
    pos0_policy: str = DEFAULT_POS0_POLICY,
    with_contributions: bool = True,
    contribution_modes: Sequence[str] = ("normed", "raw"),
    loader: Callable[[Path], Optional[np.ndarray]] = load_activations,
) -> dict:
    """
    Measure one frozen token set at every checkpoint in the sweep.

    The token set is fixed; only the model changes. That is the whole design:
    "does this cluster still exist at step 512" is answered by measuring the
    same particles, not by re-selecting.

    Returns {"token_set": ..., "records": [...], "skipped": [...]}. A
    checkpoint whose activations are missing, or whose token count disagrees
    with the anchor's, is skipped with a reason — a token-count mismatch means
    the prompt was tokenized differently and the positions no longer refer to
    the same particles, which must abort that checkpoint rather than produce
    numbers about the wrong tokens.
    """
    sweep = sweep_for_prompt(refs, token_set.prompt_key)
    if steps is not None:
        want = set(int(s) for s in steps)
        sweep = [r for r in sweep if r.step in want]

    records: List[dict] = []
    skipped: List[dict] = []

    for ref in sweep:
        X_all = loader(Path(ref.run_dir))
        if X_all is None:
            skipped.append({"step": ref.step, "reason": "no activations.npz"})
            continue
        if X_all.ndim != 3:
            skipped.append({"step": ref.step,
                            "reason": f"activations have shape {X_all.shape}, "
                                      "expected (n_layers, n_tokens, d)"})
            continue
        n_tokens = int(X_all.shape[1])
        if token_set.n_tokens_prompt and n_tokens != token_set.n_tokens_prompt:
            skipped.append({
                "step": ref.step,
                "reason": (f"token count {n_tokens} != anchor's "
                           f"{token_set.n_tokens_prompt}; positions no longer "
                           "identify the same particles"),
            })
            continue

        layers = []
        for li in range(int(X_all.shape[0])):
            layers.append(layer_geometry(
                X_all[li], token_set, li, pos0_policy=pos0_policy,
                with_contributions=with_contributions,
                contribution_modes=contribution_modes))
        records.append({
            "step": ref.step,
            "model": ref.model,
            "hf_revision": ref.hf_revision,
            "run_dir": str(ref.run_dir),
            "n_layers": int(X_all.shape[0]),
            "n_tokens": n_tokens,
            "layers": layers,
        })

    notes = list(token_set.notes)
    if 0 in set(token_set.positions) and pos0_policy == "excluded":
        notes.append(
            "position 0 (the NeoX attention sink) is a member of this token "
            "set, but pos0_policy='excluded' removes it from every population "
            "aggregate — its per-particle rank contributions are NaN by "
            "construction. Re-run with pos0_policy='included' to measure it, "
            "and read the population ranks knowing the sink dominates them."
        )

    return {
        "token_set": token_set.name,
        "prompt_key": token_set.prompt_key,
        "anchor_step": token_set.anchor_step,
        "positions": list(token_set.positions),
        "pos0_policy": pos0_policy,
        "frame_kind": "l2_sphere",
        "contribution_modes": (list(contribution_modes)
                               if with_contributions else []),
        "records": records,
        "skipped": skipped,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def geometry_report_lines(sweep: dict, layer: Optional[int] = None) -> List[str]:
    """
    Per-checkpoint compactness at one layer (default: the deepest layer
    common to every checkpoint).

    Prints normed and raw effective rank side by side. Their divergence is
    the norm-growth signal D1 identifies, and reporting only one of them is
    how that confound stayed invisible.
    """
    recs = sweep.get("records", [])
    if not recs:
        return [f"(no checkpoints measured for {sweep.get('token_set')})"]

    if layer is None:
        layer = min(r["n_layers"] for r in recs) - 1

    lines = [
        f"{sweep['token_set']} / {sweep['prompt_key']} / layer {layer} / "
        f"frame={sweep['frame_kind']} pos0={sweep['pos0_policy']} / "
        f"n_primary={len(sweep['positions'])}",
        f"{'step':>8}  {'resultant':>9}  {'separation':>10}  "
        f"{'set_rank_n':>10}  {'pop_rank_n':>10}  {'pop_rank_raw':>12}",
    ]
    for r in recs:
        if layer >= len(r["layers"]):
            continue
        L = r["layers"][layer]
        prim = L["sets"].get("primary")
        if prim is None:
            continue
        lines.append(
            f"{r['step']:>8}  {prim['resultant_length']:>9.4f}  "
            f"{prim['separation']:>10.4f}  {prim['eff_rank_normed']:>10.3f}  "
            f"{L['population']['eff_rank_normed']:>10.3f}  "
            f"{L['population']['eff_rank_raw']:>12.3f}"
        )
    for s in sweep.get("skipped", []):
        lines.append(f"  [skip] step {s['step']}: {s['reason']}")
    for n in sweep.get("notes", []):
        lines.append(f"  [note] {n}")
    return lines
