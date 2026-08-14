"""
p1b_hemisphere/visualization/_fixture.py

A synthetic Phase 1b output directory, so the figure catalogue can be drawn
and exercised without a model, a GPU, or a Phase 1 run. Underscore-prefixed
because it is a test aid, not part of the figure pipeline.

The numbers are invented. The SHAPES are the ones `run_1b._save_run` writes:
same filenames, same JSON keys, same JSON-null-for-NaN convention, same
transition-fields-absent-at-the-last-layer convention, the same string-keyed
per-layer dicts JSON produces, and a real `ParticleTable` written through
`core.particles` rather than a hand-rolled npz. A figure that breaks against
this fixture breaks against a run.

What the synthetic model does, so a fixture figure is legible rather than
noise — all of it a GUESS, planted deliberately so a figure that fails to
show it is showing a plotting bug, and none of it a prediction:

  * Cone-collapse holds at every layer, with the normalized margin shrinking
    with depth — the phase's actual finding, drawn as a continuous quantity
    rather than a label so C1 has something to say.
  * The antipodal classifier reports `collapsed` almost everywhere while the
    relative one finds `separated` through mid-depth. That is status-1b R1
    made visible: the same geometry, two verdicts, only one of them
    reachable.
  * The Fiedler axis starts unrelated to PC1 and converges onto it with
    depth, so R2/A2's "Phase 5 may be using PC1 under a more expensive name"
    has a shape to look at.
  * Tokens are laminar early, mix through the middle third, and re-freeze —
    so the ribbon and flow figures have structure rather than a solid block.
  * `build_checkpoint_fixture` repeats the whole thing across a log-spaced
    Pythia-style step schedule, with the axis settling around step 1,000, so
    the checkpoint figures have a family to group.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ARTIFACT_PREFIX = "phase1b"

#: A token vocabulary with the surface classes X5 splits on — leading-space
#: words, bare subword continuations, punctuation, and digits — so the token
#: -class figure has all four populations rather than one.
_VOCAB = [
    " the", " model", " layer", "s", " token", " geometry", ",", " which",
    " collapses", " into", " a", " single", " point", ".", " In", " 1958",
    "ization", " of", " attention", " heads", ";", " 42", " and", " so",
    " forth", "—", " the", " residual", " stream", " is", " a", " sphere",
]


def _tokens(n: int) -> List[str]:
    return [_VOCAB[i % len(_VOCAB)] for i in range(n)]


def _ramp(step: int, midpoint: float = 1000.0, width: float = 1.2) -> float:
    """0 at init, 1 late, logistic in log10(step+1). Matches p2's fixture."""
    x = np.log10(step + 1.0)
    return float(1.0 / (1.0 + np.exp(-(x - np.log10(midpoint)) / width * 4.0)))


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def build_run(
    out_dir: Path,
    model: str = "gpt2-large",
    prompt: str = "wiki_paragraph",
    n_layers: int = 24,
    n_tokens: int = 128,
    d_model: int = 64,
    checkpoint_step: Optional[int] = None,
    seed: int = 0,
    with_nulls: bool = True,
    with_nesting: bool = True,
    trained: float = 1.0,
) -> Path:
    """
    Write one synthetic `phase1b_{stem}.*` triple into `out_dir`.

    `trained` in [0, 1] interpolates between an untrained-looking run (axis
    unrelated to PC1, no depth structure, tokens mixing at every layer) and a
    trained-looking one. `build_checkpoint_fixture` drives it off the step
    schedule; a single run leaves it at 1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stem = f"{model.replace('/', '_').replace('@', '_')}_{prompt}"

    depth = np.linspace(0.0, 1.0, n_layers)
    a = float(np.clip(trained, 0.0, 1.0))
    tokens = _tokens(n_tokens)

    # ---- the underlying object: a Fiedler value per (layer, token) --------
    # One elongated cone whose axis is stable, plus a mixing band through the
    # middle third. Everything else in the fixture is derived from this, so
    # the particle table and the per-layer scalars cannot disagree — which is
    # the property that makes the fixture worth trusting for figure tests.
    base = rng.normal(0.0, 1.0, n_tokens)
    base = base - base.mean()
    base = base / (np.abs(base).max() + 1e-12)
    minority = 0.18 + 0.22 * np.sin(depth * np.pi)          # 18% -> 40% -> 18%
    fiedler = np.zeros((n_layers, n_tokens))
    for L in range(n_layers):
        turbulence = 0.55 * np.exp(-((depth[L] - 0.5) ** 2) / 0.02) * a + 0.06
        shift = np.quantile(base, 1.0 - minority[L])
        fiedler[L] = (base - shift) + rng.normal(0.0, turbulence, n_tokens) * 0.35
        fiedler[L] /= (np.abs(fiedler[L]).max() + 1e-12)

    hemi = (fiedler > 0).astype(np.int8)
    sizes = np.stack([(hemi == 0).sum(1), (hemi == 1).sum(1)], axis=1)
    minority_frac = sizes.min(1) / float(n_tokens)
    boundary_frac = (np.abs(fiedler) < 0.05).mean(1)

    # ---- Block 0 scalars ---------------------------------------------------
    # Centroid angle stays well under pi/2 at every layer: two centroids
    # inside one open half-space essentially cannot be pi/2 apart, which is
    # exactly why the antipodal classifier can never fire here (R1).
    centroid_angle = 0.55 + 0.65 * np.sin(depth * np.pi) * a + rng.normal(0, 0.02, n_layers)
    within = np.stack([0.82 - 0.18 * depth * a + rng.normal(0, 0.01, n_layers),
                       0.79 - 0.16 * depth * a + rng.normal(0, 0.01, n_layers)], axis=1)
    between = 0.74 - 0.42 * np.sin(depth * np.pi) * a + rng.normal(0, 0.012, n_layers)
    sep_ratio = between / np.clip(within.mean(1), 1e-6, None)
    eigengap = 0.06 + 0.22 * np.sin(depth * np.pi) * a + rng.normal(0, 0.008, n_layers)
    asymmetry = np.abs(sizes[:, 0] - sizes[:, 1]) / float(n_tokens)

    # Labels come from the phase's own classifiers, not from a rule retyped
    # here. Two reasons: the vocabularies are wider than they look (`diffuse`
    # and `uniform` exist and a hand-rolled fixture would never emit them),
    # and a threshold change in bipartition_detect must move the fixture's
    # labels or the fixture stops being a test of anything.
    from p1b_hemisphere.bipartition_detect import (
        classify_regime, classify_regime_relative,
    )
    regime = [classify_regime(float(minority_frac[L]), float(centroid_angle[L]),
                              float(within[L, 0]), float(within[L, 1]))
              for L in range(n_layers)]
    regime_rel = [classify_regime_relative(float(minority_frac[L]),
                                           float(sep_ratio[L]))
                  for L in range(n_layers)]

    # ---- Block 1 transitions ----------------------------------------------
    crossing = np.array([int((hemi[L] != hemi[L + 1]).sum())
                         for L in range(n_layers - 1)], dtype=float)
    axis_rot = 0.04 + 0.30 * np.exp(-((depth[:-1] - 0.5) ** 2) / 0.02) * a \
        + rng.normal(0, 0.008, n_layers - 1)
    axis_rot = np.abs(axis_rot)
    overlap = np.clip(1.0 - crossing / float(n_tokens) * 1.4, 0.0, 1.0)

    # ---- Block 3 cone -------------------------------------------------------
    norm_margin = 0.34 - 0.26 * depth * a + rng.normal(0, 0.006, n_layers)
    norm_margin = np.clip(norm_margin, 0.008, None)
    cone_margin = norm_margin * (0.9 + 0.2 * rng.random(n_layers))
    cone_regime = ["cone_collapse"] * n_layers
    n_binding = np.clip((3 + 9 * depth * a + rng.normal(0, 0.8, n_layers)),
                        2, None).astype(int)
    d_eff = np.full(n_layers, min(d_model, n_tokens), dtype=int)

    # ---- Block A axis identity ----------------------------------------------
    # cos to the token mean stays at chance by construction: the Fiedler
    # vector is orthogonal to the Laplacian's trivial eigenvector, so X^T f
    # cancels the shared mean. A fixture that let this rise would hide the
    # one control Block A actually has.
    isotropic = 1.0 / np.sqrt(d_model)
    cos_mean = np.abs(rng.normal(0, isotropic * 0.8, n_layers))
    cos_pc1 = np.clip(0.42 + 0.52 * depth * a + rng.normal(0, 0.03, n_layers), 0, 0.999)
    cos_pc1_unc = np.clip(cos_pc1 + rng.normal(0, 0.02, n_layers), 0, 0.999)
    cos_cen_pc1 = np.clip(cos_pc1 + rng.normal(0, 0.015, n_layers), 0, 0.999)
    cos_mu_pc1 = np.clip(0.30 + 0.25 * depth * a + rng.normal(0, 0.03, n_layers), 0, 0.999)
    pc_frac = np.clip(cos_pc1 ** 2 + 0.25 * (1 - cos_pc1 ** 2), 0, 1)
    pc1_var = np.clip(0.18 + 0.34 * depth * a + rng.normal(0, 0.01, n_layers), 0, 1)
    redundancy = ["pc1" if cos_pc1[L] >= 0.9 else
                  ("top_pc_block" if pc_frac[L] >= 0.9 else "distinct")
                  for L in range(n_layers)]

    # ---- assemble per_layer -------------------------------------------------
    per_layer = []
    for L in range(n_layers):
        per_layer.append({
            "layer": L,
            "regime": regime[L],
            "regime_relative": regime_rel[L],
            "bipartition_eigengap": float(eigengap[L]),
            "centroid_angle": float(centroid_angle[L]),
            "within_half_ip": [float(within[L, 0]), float(within[L, 1])],
            "between_half_ip": float(between[L]),
            "separation_ratio": float(sep_ratio[L]),
            "fiedler_boundary_frac": float(boundary_frac[L]),
            "hemisphere_sizes": [int(sizes[L, 0]), int(sizes[L, 1])],
            "minority_fraction": float(minority_frac[L]),
            "asymmetry": float(asymmetry[L]),
            # Transition fields are defined for L -> L+1, so null at the last
            # layer — the writer's own convention, kept because a loader that
            # drops the row instead of NaN-ing it shifts every depth profile.
            "crossing_count": (int(crossing[L]) if L < n_layers - 1 else None),
            "axis_rotation": (float(axis_rot[L]) if L < n_layers - 1 else None),
            "match_overlap": (float(overlap[L]) if L < n_layers - 1 else None),
            "cone_regime": cone_regime[L],
            "cone_margin": float(cone_margin[L]),
            "normalized_margin": float(norm_margin[L]),
            "cone_escalated": False,
            "cone_n_binding": int(n_binding[L]),
            "cos_axis_mean": float(cos_mean[L]),
            "cos_axis_pc1": float(cos_pc1[L]),
            "axis_redundancy": redundancy[L],
        })

    # ---- per_token ----------------------------------------------------------
    stability = np.array([float((hemi[:, t] == np.bincount(hemi[:, t]).argmax()).mean())
                          for t in range(n_tokens)])
    border_index = np.abs(fiedler).mean(0)
    border_index = border_index / (border_index.max() + 1e-12)
    per_token = []
    for t in range(n_tokens):
        traj = [int(hemi[L, t]) for L in range(n_layers)]
        dom = int(np.bincount(hemi[:, t]).argmax())
        first_stable = next((L for L in range(n_layers)
                             if all(hemi[k, t] == hemi[-1, t]
                                    for k in range(L, n_layers))), None)
        per_token.append({
            "token_id": t,
            "token_str": tokens[t],
            "position": t,
            "hemisphere_trajectory": traj,
            "stability_score": float(stability[t]),
            "border_index": float(1.0 - border_index[t]),
            "first_assignment_layer": first_stable,
            "dominant_hemisphere": dom,
            "final_hemisphere": int(hemi[-1, t]),
        })

    # ---- events -------------------------------------------------------------
    # Sparse by design: under the antipodal regime_key these are foreclosed
    # (R4), and a fixture dense with events would hide that. Two shear events
    # in the mixing band is what the relative vocabulary would plausibly give.
    events = []
    if a > 0.5:
        mid = n_layers // 2
        events = [
            {"type": "shear", "layer": mid, "from_layer": mid - 1,
             "detail": {"axis_rotation": float(axis_rot[mid - 1])}},
            {"type": "drift", "layer": mid + 3, "from_layer": mid - 1,
             "detail": {"window_rotation": float(axis_rot[mid - 1:mid + 2].sum())}},
        ]

    # ---- summary ------------------------------------------------------------
    n_strong = sum(1 for r in regime if r == "strong_bipartition")
    n_sep = sum(1 for r in regime_rel if r == "separated")
    n_graded = sum(1 for r in regime_rel if r == "graded")
    nesting_overall = {
        "n_analyzed_layers": n_layers, "total_clusters": 14 * n_layers,
        "fully_nested_fraction": float(0.28 + 0.4 * a),
        "mixed_fraction": float(0.30 - 0.15 * a),
        "mean_r_c_distance_from_half": float(0.24 + 0.12 * a),
        "nesting_tolerance": 0.05, "mixed_half_width": 0.1,
    } if with_nesting else None

    summary = {
        "strong_bipartition_layer_fraction": n_strong / n_layers,
        "separated_layer_fraction": n_sep / n_layers,
        "graded_layer_fraction": n_graded / n_layers,
        "cone_collapse_layer_fraction": 1.0,
        "mean_normalized_cone_margin": float(norm_margin.mean()),
        "n_layers_escalated_to_full_d": 0,
        "mean_axis_rotation": float(axis_rot.mean()),
        "mean_asymmetry_strong": None,
        "event_counts": {e["type"]: 1 for e in events},
        "mean_stability_score": float(stability.mean()),
        "fraction_never_stable": float((np.array(
            [p["first_assignment_layer"] is None for p in per_token])).mean()),
        "hdbscan_nesting_overall": nesting_overall,
        "border_vs_noise_mean_auc": (float(0.52 + 0.14 * a) if with_nesting else None),
        "crossref_with_phase1": {
            "mean_axis_rotation_at_merge": float(axis_rot.mean() * 1.6),
            "mean_axis_rotation_off_merge": float(axis_rot.mean() * 0.9),
            "mean_crossing_at_violation": float(crossing.mean() * 1.3),
            "mean_crossing_off_violation": float(crossing.mean() * 0.95),
            "n_merges_in_run": 3, "n_violations_in_run": 5,
        },
        "axis_modal_redundancy": max(set(redundancy), key=redundancy.count),
        "mean_cos_axis_mean": float(cos_mean.mean()),
        "mean_cos_axis_pc1": float(cos_pc1.mean()),
        "mean_cos_mean_pc1": float(cos_mu_pc1.mean()),
    }
    if with_nulls:
        summary["mean_uniform_cone_fraction"] = float(0.05 + 0.1 * (1 - a))
        summary["mean_shuffled_cone_fraction"] = 1.0

    # ---- the JSON -----------------------------------------------------------
    data = {
        "model": model, "prompt": prompt,
        "n_layers": n_layers, "n_tokens": n_tokens,
        "checkpoint_step": checkpoint_step,
        "frame": {"kind": "l2_sphere", "pos0_policy": "keep",
                  "model_rev": model},
        "connectivity_floor": 0.0,
        "per_layer": per_layer,
        "per_token": per_token,
        "summary": summary,
        "events": events,
        "cone": {
            "n_layers": n_layers, "n_tokens": n_tokens,
            "regime_counts": {"cone_collapse": n_layers},
            "cone_collapse_fraction": 1.0, "split_fraction": 0.0,
            "first_split_layer": None,
            "n_cone_collapse_before_split": n_layers,
            "mean_normalized_margin": float(norm_margin.mean()),
            "min_normalized_margin": float(norm_margin.min()),
            "max_normalized_margin": float(norm_margin.max()),
            "mean_cone_margin": float(cone_margin.mean()),
            "n_lp_at_limit": 0, "n_escalated": 0,
            "mean_n_binding": float(n_binding.mean()),
            "n_null_layers": (n_layers if with_nulls else 0),
            "pca_n_components": 64, "dropped_indices": [], "tol": 1e-9,
        },
        "axis_identity": {
            "per_layer": [
                {"layer": L,
                 "cos_axis_mean": float(cos_mean[L]),
                 "cos_axis_pc1": float(cos_pc1[L]),
                 "cos_axis_pc1_uncentered": float(cos_pc1_unc[L]),
                 "cos_axis_centered_pc1": float(cos_cen_pc1[L]),
                 "cos_mean_pc1": float(cos_mu_pc1[L]),
                 "pc_subspace_fraction": float(pc_frac[L]),
                 "pc1_explained_variance": float(pc1_var[L]),
                 "redundancy": redundancy[L]}
                for L in range(n_layers)
            ],
            "summary": {
                "n_layers": n_layers,
                "modal_redundancy": summary["axis_modal_redundancy"],
                "redundancy_counts": {k: redundancy.count(k)
                                      for k in set(redundancy)},
                "mean_cos_axis_mean": float(cos_mean.mean()),
                "mean_cos_axis_pc1": float(cos_pc1.mean()),
                "mean_cos_mean_pc1": float(cos_mu_pc1.mean()),
                "mean_pc_subspace_fraction": float(pc_frac.mean()),
            },
        },
        "persistence_length": [float(x) for x in
                               np.clip(np.round(3 + 6 * np.cos(depth * np.pi) * a), 0, None)],
        "regime_key": "regime",
    }

    if with_nulls:
        data["cone"]["mean_z_vs_shuffled"] = 0.4
        data["cone"]["mean_z_vs_uniform"] = 6.1
        data["cone"]["mean_shuffled_cone_fraction"] = 1.0
        data["cone"]["mean_uniform_cone_fraction"] = float(0.05 + 0.1 * (1 - a))

    data["cone_per_layer"] = [
        {
            "layer": L, "cone_regime": cone_regime[L],
            "cone_margin": float(cone_margin[L]),
            "normalized_margin": float(norm_margin[L]),
            "solved": True, "lp_at_limit": False,
            "escalated_to_full_d": False,
            "d_eff": int(d_eff[L]), "n_binding": int(n_binding[L]),
            "binding_tokens": [int(t) for t in
                               np.argsort(np.abs(fiedler[L]))[:int(n_binding[L])]],
            **({
                "z_vs_shuffled": float(0.2 + 0.5 * rng.random()),
                "z_vs_uniform": float(4.5 + 3.0 * norm_margin[L] * 3),
                "pct_vs_uniform": float(99.0 + 0.9 * rng.random()),
                "null_mean_shuffled": float(norm_margin[L] * 0.95),
                "null_mean_uniform": float(norm_margin[L] * 0.18),
                "shuffled_cone_fraction": 1.0,
                "uniform_cone_fraction": float(0.05 + 0.1 * (1 - a)),
            } if with_nulls else {}),
        }
        for L in range(n_layers)
    ]

    if with_nesting:
        data["hdbscan_nesting"] = {
            "per_layer": {
                str(L): {
                    "clusters": [
                        {"cluster_id": c, "size": int(6 + 3 * ((c + L) % 4)),
                         "r_c": float(np.clip(rng.beta(0.6, 0.6), 0, 1)),
                         "nesting_class": ("nested_A" if (c + L) % 3 == 0 else
                                           ("nested_B" if (c + L) % 3 == 1 else "mixed"))}
                        for c in range(8)
                    ],
                    "summary": {"n_clusters": 8,
                                "fully_nested_fraction": float(0.3 + 0.5 * a * depth[L]),
                                "mixed_fraction": float(0.4 - 0.2 * a * depth[L]),
                                "mean_r_c_distance_from_half": float(0.2 + 0.2 * depth[L])},
                } for L in range(n_layers)
            },
            "overall": nesting_overall,
        }
        data["border_vs_noise"] = {
            "per_layer": {
                str(L): {"auc": float(np.clip(0.5 + 0.22 * a * np.sin(depth[L] * np.pi)
                                              + rng.normal(0, 0.02), 0, 1)),
                         "mean_abs_v_noise": float(0.18 + 0.05 * depth[L]),
                         "mean_abs_v_clustered": float(0.36 + 0.05 * depth[L]),
                         "n_noise": int(0.22 * n_tokens),
                         "n_clustered": int(0.78 * n_tokens)}
                for L in range(n_layers)
            },
            "overall": {"n_analyzed_layers": n_layers,
                        "mean_auc": float(0.52 + 0.14 * a),
                        "min_auc": 0.48, "max_auc": float(0.55 + 0.2 * a),
                        "fraction_layers_auc_above_0.6": float(0.2 * a)},
        }

    (out_dir / f"{ARTIFACT_PREFIX}_{stem}.json").write_text(
        json.dumps(data, indent=2))
    (out_dir / f"{ARTIFACT_PREFIX}_{stem}.md").write_text(
        f"# Phase 1b — {model} / {prompt}\n\n(synthetic fixture)\n")

    _write_particles(out_dir / f"{ARTIFACT_PREFIX}_{stem}_particles.npz",
                     model, prompt, checkpoint_step, fiedler, hemi, tokens,
                     stability, 1.0 - border_index, rng)

    # Axes: a stable direction rotating slowly with depth, in d_model dims.
    axes = np.zeros((n_layers, d_model))
    anchor = rng.normal(0, 1, d_model)
    anchor /= np.linalg.norm(anchor)
    drift = rng.normal(0, 1, d_model)
    drift -= drift.dot(anchor) * anchor
    drift /= np.linalg.norm(drift)
    for L in range(n_layers):
        # At a=0 the axis is a different random direction at every layer; at
        # a=1 it is one direction with a small depth-dependent lean.
        theta = (1.0 - a) * rng.uniform(0, np.pi / 2) + a * 0.25 * depth[L]
        v = np.cos(theta) * anchor + np.sin(theta) * drift
        axes[L] = v / np.linalg.norm(v)
    np.savez_compressed(out_dir / f"{ARTIFACT_PREFIX}_{stem}_axes.npz",
                        axes=axes.astype(np.float32),
                        valid=np.ones(n_layers, dtype=bool))
    return out_dir / f"{ARTIFACT_PREFIX}_{stem}.json"


def _write_particles(path: Path, model: str, prompt: str,
                     checkpoint_step: Optional[int], fiedler: np.ndarray,
                     hemi: np.ndarray, tokens: Sequence[str],
                     stability: np.ndarray, border: np.ndarray,
                     rng: np.random.Generator) -> None:
    """
    Write the ParticleTable through `core.particles`, not by hand.

    Going through the real writer is the point: it enforces the schema, the
    fixed-width-unicode string rule, and the `extra__` prefix, so a fixture
    that saves is a fixture the loader can read for the same reasons a real
    run is.
    """
    from core.particles import ParticleTable, default_population_tag

    n_L, n_T = fiedler.shape
    layer_ix, token_ix = np.meshgrid(np.arange(n_L), np.arange(n_T), indexing="ij")
    layer_ix, token_ix = layer_ix.ravel(), token_ix.ravel()

    # A plausible HDBSCAN labelling: a fifth unclustered, the rest in a
    # handful of clusters, with noise biased toward the Fiedler boundary —
    # which is the correlation border_vs_noise measures and M5 draws.
    cluster = np.zeros(layer_ix.size, dtype=np.int64)
    for j in range(layer_ix.size):
        v = abs(float(fiedler[layer_ix[j], token_ix[j]]))
        p_noise = float(np.clip(0.45 - 0.9 * v, 0.03, 0.6))
        cluster[j] = -1 if rng.random() < p_noise else int(token_ix[j] % 6)

    columns = {
        "model": np.array([model] * layer_ix.size),
        "checkpoint_step": np.full(layer_ix.size,
                                   -1 if checkpoint_step is None else int(checkpoint_step),
                                   dtype=np.int64),
        "prompt_key": np.array([prompt] * layer_ix.size),
        "layer": layer_ix.astype(np.int64),
        "token_position": token_ix.astype(np.int64),
        "cluster_label": cluster,
        "population": np.asarray(default_population_tag(cluster)),
        "token_str": np.array([tokens[t] for t in token_ix]),
    }
    extra = {
        "hemisphere": hemi.ravel().astype(np.int64),
        "fiedler_value": fiedler.ravel().astype(np.float64),
        "border_index": border[token_ix].astype(np.float64),
        "stability_score": stability[token_ix].astype(np.float64),
        "layer_valid": np.ones(layer_ix.size, dtype=np.int64),
        "layer_regime": np.array(["weak_bipartition"] * layer_ix.size),
    }
    ParticleTable(columns, extra).save(path)


# ---------------------------------------------------------------------------
# Whole directories
# ---------------------------------------------------------------------------

#: A short log-spaced schedule in the spirit of Pythia's, which is log-spaced
#: to 512 and linear after. Short enough that the fixture renders quickly,
#: spaced widely enough that log10(step+1) is not a straight line.
FIXTURE_STEPS = (0, 1, 16, 128, 512, 1000, 4000, 16000, 64000, 143000)


def build_fixture(out_dir: Path,
                  models: Sequence[str] = ("gpt2-large", "albert-base-v2"),
                  prompts: Sequence[str] = ("wiki_paragraph", "short_heterogeneous"),
                  checkpoints: bool = True,
                  seed: int = 0) -> Path:
    """
    A complete synthetic Phase 1b output directory.

    Two plain models across two prompts (one long, one short — so the
    `LONG_PROMPT_TOKENS` split in V5 has points on both sides), plus one
    checkpoint family when `checkpoints` is set, plus the cross-run digest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[dict] = []
    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            n_tokens = 148 if prompt == "wiki_paragraph" else 64
            n_layers = 24 if "large" in model else 12
            path = build_run(out_dir, model=model, prompt=prompt,
                             n_layers=n_layers, n_tokens=n_tokens,
                             seed=seed + 11 * i + j)
            runs.append(json.loads(path.read_text()))

    if checkpoints:
        for k, step in enumerate(FIXTURE_STEPS):
            path = build_run(out_dir, model=f"pythia-410m-step{step}",
                             prompt="wiki_paragraph", n_layers=24,
                             n_tokens=148, checkpoint_step=step,
                             seed=seed + 101 + k, trained=_ramp(step))
            runs.append(json.loads(path.read_text()))

    _write_cross_run(out_dir, runs)
    return out_dir


def _write_cross_run(out_dir: Path, runs: List[dict]) -> None:
    """
    The cross-run digest, built with the phase's own aggregator.

    `aggregate`, `aggregate_by_checkpoint`, and `global_verdict` are imported
    from `p1b_report` rather than reimplemented, so the fixture's digest is
    the shape the phase produces by construction — including whichever fields
    a future revision adds.
    """
    from p1b_hemisphere.p1b_report import (
        aggregate, aggregate_by_checkpoint, global_verdict,
    )

    by_model: Dict[str, list] = {}
    by_prompt: Dict[str, list] = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)
        by_prompt.setdefault(r["prompt"], []).append(r)

    cross_run = {
        "by_model": {m: aggregate(rs) for m, rs in by_model.items()},
        "by_prompt": {p: aggregate(rs) for p, rs in by_prompt.items()},
        "by_checkpoint": aggregate_by_checkpoint(runs),
        "global_verdict": global_verdict(runs),
    }
    (out_dir / f"{ARTIFACT_PREFIX}_cross_run.json").write_text(
        json.dumps(cross_run, indent=2))
