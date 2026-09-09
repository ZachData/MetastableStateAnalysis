"""
p7_motifs/p7_io.py — reading Phase 2 / 2b projectors into the channel
inputs the interaction table needs, and writing Phase 7's artifacts
against their registered contracts.

Why this module is where the risk is
-------------------------------------
Phase 7's two channels come from two different phases that store them in
two different, both-reasonable, mutually incompatible shapes:

  sign channel        p2_eigenspectra/weights.py writes `schur_attract_*`
  (attractive /       and `schur_repulse_*` (also `sym_*`) into
   repulsive)         ov_projectors_{stem}.npz as (d, d) SYMMETRIC
                      IDEMPOTENT PROJECTOR MATRICES, P = Z @ Z.T.

  rotational channel  p2b_imaginary/rotational_schur.py's
  (real / imaginary)  `top_rotation_planes` returns a LIST of (d, 2)
                      orthonormal plane bases and deliberately never forms
                      the (d, d) projector — that costs ~7 GB at d=1024 and
                      ~27 GB at d=2048, which its own docstring records as
                      the reason the earlier version was replaced.

`core.interactions.projection_fractions` accepts and validates both, plus
a plain orthonormal basis. This module's job is only to pull the right
arrays out of the right artifacts and hand them over with the frame and
the choice-of-decomposition recorded — not to reshape or re-derive
anything.

The schur / sym choice is not a default
----------------------------------------
Phase 2 stores two different attractive/repulsive splits:

  schur_*  from the real Schur form, splitting on Re(lambda) — the
           attractive/repulsive split of the FULL operator, which is what
           the particle dynamics are driven by.
  sym_*    from the eigendecomposition of the symmetric part alone,
           splitting on the sign of its real eigenvalues.

They answer different questions and Phase 2b's whole finding is that the
symmetric part carries 100% of the violation causality while the
antisymmetric part is dynamically neutral — so the two are not
interchangeable, and which one a Phase 7 result used changes what the
result means. `sign_channel` is therefore a required argument with no
default, and it is stamped into every record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from core.artifacts import get_spec

SIGN_CHANNEL_CHOICES = ("schur", "sym")


def _stem(model_name: str) -> str:
    """Phase 2's filename convention, matching weights.py's own."""
    return model_name.replace("/", "_")


def load_sign_channel(
    weights_dir: Union[str, Path],
    model_name: str,
    sign_channel: str,
    layer_name: Optional[str] = None,
) -> dict:
    """
    Load one layer's attractive/repulsive projectors from Phase 2's
    ov_projectors_{stem}.npz.

    sign_channel : "schur" or "sym" — see the module docstring. Required,
        because the two are not interchangeable and a silent default would
        make the choice invisible in the result.
    layer_name   : the layer key Phase 2 wrote (`layer_0`, ...). None
        selects the shared/global projector (`*_shared`), which is what a
        weight-tied model has and what a per-layer model does NOT.

    Returns {"U_pos", "U_neg", "provenance"}. Refuses rather than
    substituting when the requested key is absent: a Phase 7 run against a
    layer Phase 2 never decomposed must stop, not fall back to the shared
    projector, which would apply one layer's geometry to another's
    activations and produce numbers that look fine.
    """
    if sign_channel not in SIGN_CHANNEL_CHOICES:
        raise ValueError(
            f"sign_channel must be one of {list(SIGN_CHANNEL_CHOICES)}; got "
            f"{sign_channel!r}. There is no default: 'schur' splits the full "
            "operator on Re(lambda), 'sym' splits only the symmetric part, and "
            "which one a result used changes what it means."
        )

    weights_dir = Path(weights_dir)
    path = weights_dir / f"ov_projectors_{_stem(model_name)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 2 projectors not found at {path}. Phase 7's sign channel "
            "has no other source; run Phase 2 for this model first."
        )

    suffix = layer_name if layer_name is not None else "shared"
    data = np.load(path, allow_pickle=False)
    keys = (f"{sign_channel}_attract_{suffix}", f"{sign_channel}_repulse_{suffix}")
    missing = [k for k in keys if k not in data.files]
    if missing:
        raise KeyError(
            f"{path.name} has no {missing}. Present keys: "
            f"{sorted(data.files)[:8]}{'...' if len(data.files) > 8 else ''}. "
            "Refusing to substitute a different layer's projector."
        )

    return {
        "U_pos": data[keys[0]],
        "U_neg": data[keys[1]],
        "provenance": {
            "source": str(path),
            "sign_channel": sign_channel,
            "layer_name": suffix,
            "form": "symmetric_idempotent_projector",
        },
    }


def load_ov_circuits(weights_dir: Union[str, Path], model_name: str) -> dict:
    """
    Per-layer, per-head composed OV circuits from Phase 2's
    ov_weights_{stem}.npz — the third input `build_head_edges` needs,
    alongside the two channels above.

    Reads the array-name convention `weights.save_weight_decomposition`
    writes: `ov_head{h}_{lname}` with lname from summary["layers"], or the
    `_shared` suffix for a weight-tied model.

    Deliberately NOT `p2d_io.load_operators`, which reads the same file.
    That function raises when the `wq_head*` / `wk_head*` arrays are absent,
    because every Phase 2d sub-experiment needs M_h = W_Q W_K^T. Phase 7
    needs no such thing — the QK side enters here only through the pair
    types, which come from token identity in `core.battery_structure`, not
    from weights. Borrowing 2d's loader would make a Phase 2 run that is
    complete for this phase's purposes unreadable by it.

    Returns {"layers", "d_model", "n_layers", "n_heads", "source",
    "is_per_layer"}, with `layers[i]["heads"][h]["ov"]` the (d_model,
    d_model) matrix.
    """
    weights_dir = Path(weights_dir)
    stem = _stem(model_name)
    sum_p = weights_dir / f"ov_summary_{stem}.json"
    w_p = weights_dir / f"ov_weights_{stem}.npz"
    for path in (sum_p, w_p):
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 2 OV circuits not found at {path}. Phase 7's edges "
                "are typed by force, which is built from the composed OV "
                "circuit; there is no other source. Run Phase 2 for this "
                "model first."
            )

    with open(sum_p) as f:
        summary = json.load(f)
    w = np.load(w_p, allow_pickle=False)

    is_per_layer = bool(summary["is_per_layer"])
    lnames = list(summary["layers"]) if is_per_layer else ["shared"]

    layers = []
    for lname in lnames:
        heads = []
        h = 0
        while f"ov_head{h}_{lname}" in w.files:
            heads.append({"head": h, "ov": w[f"ov_head{h}_{lname}"]})
            h += 1
        if not heads:
            raise KeyError(
                f"{w_p.name} has no ov_head*_{lname} arrays. Present keys: "
                f"{sorted(w.files)[:8]}{'...' if len(w.files) > 8 else ''}."
            )
        layers.append({"layer_name": lname, "heads": heads})

    # Every layer must carry the same head count. A ragged decomposition
    # would otherwise be discovered as an IndexError against the attention
    # tensor, several loops in, with nothing naming the cause.
    counts = {len(l["heads"]) for l in layers}
    if len(counts) > 1:
        raise ValueError(
            f"{w_p.name} has differing head counts across layers ({sorted(counts)}); "
            "refusing rather than iterating the smallest."
        )

    # The summary is the authority for the geometry, not the arrays. Taking
    # d_model from the first array instead would make the one substitution
    # worth catching undetectable: `ov_head_core` is (d_head, d_head) for
    # every head, so an inferred d_model simply comes out as d_head and
    # every shape agrees. Its spectrum matches the composed circuit's and
    # its action on the residual stream does not, so the resulting force
    # vectors would be wrong in a way no shape reveals.
    declared = summary.get("d_model")
    d_head = summary.get("d_head")
    d_model = int(declared) if declared is not None else int(
        layers[0]["heads"][0]["ov"].shape[0])

    for layer in layers:
        for head in layer["heads"]:
            if head["ov"].shape != (d_model, d_model):
                extra = ""
                if d_head is not None and head["ov"].shape == (int(d_head), int(d_head)):
                    extra = (" That is (d_head, d_head) — the `ov_head_core` "
                             "factor B_h @ A_h, not the composed circuit "
                             "A_h @ B_h. Its spectrum is the same; what it "
                             "does to the residual stream is not.")
                raise ValueError(
                    f"{w_p.name}: ov_head{head['head']}_{layer['layer_name']} has "
                    f"shape {head['ov'].shape}, expected ({d_model}, {d_model})."
                    + extra
                )

    declared_heads = summary.get("n_heads")
    if declared_heads is not None and int(declared_heads) != len(layers[0]["heads"]):
        raise ValueError(
            f"{sum_p.name} declares n_heads={int(declared_heads)} but "
            f"{w_p.name} holds {len(layers[0]['heads'])} ov_head* arrays per "
            "layer. The summary and the arrays are from different runs."
        )

    return {
        "layers": layers,
        "is_per_layer": is_per_layer,
        "n_layers": len(layers),
        "n_heads": len(layers[0]["heads"]),
        "d_model": d_model,
        "source": str(w_p),
    }


def rotational_channel_from_blocks(block_data: dict, top_k: int = 32) -> dict:
    """
    Build the real/imaginary channel inputs from one layer's Schur block
    data (p2b_imaginary.rotational_schur.extract_schur_blocks output).

    `U_A` (imaginary) is the LIST of (d, 2) rotation-plane bases, passed
    through as a list rather than stacked into a projector — that is the
    memory decision top_rotation_planes exists to make, and undoing it
    here would reintroduce exactly the cost it avoids.

    `U_R` (real) is NOT returned. There is no stored basis for the real
    complement, and constructing one would mean a (d, d) eigendecomposition
    per layer. Callers that want the real fraction should use
    `1 - imag_frac` on the edges where imag_frac is finite, which is exact
    because the rotational planes and the real Schur directions are
    orthogonal by construction. Doing that subtraction in the caller keeps
    the identity visible instead of burying it in a second projector that
    would have to be kept consistent with the first.

    top_k caps how many planes are used, and is recorded: a truncated
    basis makes imag_frac a LOWER BOUND, not the full rotational fraction.
    """
    from p2b_imaginary.rotational_schur import top_rotation_planes

    planes = top_rotation_planes(block_data, top_k=top_k)
    n_available = int(planes["dim_rotation"] // 2)
    n_used = len(planes["bases"])

    return {
        "U_A": planes["bases"],
        "provenance": {
            "form": "list_of_2d_plane_bases",
            "top_k": int(top_k),
            "n_planes_used": n_used,
            "n_planes_available": n_available,
            "truncated": bool(n_used < n_available),
            # Stated rather than left for the reader to infer: with a
            # truncated basis the reported imaginary fraction is a lower
            # bound on the true rotational fraction.
            "imag_frac_is_lower_bound": bool(n_used < n_available),
            "dim_real": int(planes["dim_real"]),
            "d": int(planes["d"]),
        },
    }


def write_motif_counts(payload: dict, out_dir: Union[str, Path]) -> Path:
    """Write motif_counts.json, validating against its registered spec first."""
    spec = get_spec("phase7", "motif_counts")
    missing = set(spec.required_keys) - set(payload)
    if missing:
        raise ValueError(
            f"motif_counts payload is missing required keys {sorted(missing)} "
            "declared in core.artifacts REGISTRY['phase7']. Build it with "
            "p7_motifs.motif_stats.motif_counts_payload rather than by hand."
        )
    out = Path(out_dir) / spec.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return out


def write_formation_curve(payload: dict, out_dir: Union[str, Path]) -> Path:
    """
    Write formation_curve.json, validating against its registered spec.

    The spec requires `independence_source`, and that is deliberate: a
    formation curve correlating motif strength against the behavioural
    induction score without naming what makes the two independent has
    plotted one quantity against itself. See PREDICTIONS.md's Phase 7
    adjudication constraint 2.
    """
    spec = get_spec("phase7", "formation_curve")
    missing = set(spec.required_keys) - set(payload)
    if missing:
        raise ValueError(
            f"formation_curve payload is missing required keys "
            f"{sorted(missing)} declared in core.artifacts REGISTRY['phase7']."
        )
    out = Path(out_dir) / spec.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
