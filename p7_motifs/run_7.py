"""
p7_motifs/run_7.py — Phase 7 driver: typed interaction edges from a real run.

    python -m p7_motifs.run_7 \\
        --p2-dir results_p2/ --model pythia-410m-step143000 \\
        --prompt homer_iliad=results/<phase1 run>/pythia-410m-step143000_homer_iliad \\
        --prompt wiki_paragraph=results/<phase1 run>/pythia-410m-step143000_wiki_paragraph \\
        --sign-channel schur --out results_p7/

Build step 8 of `status-7.md`, which has read "NEXT" since 2026-08-22. Steps
1-7 built the artifact contract, the typed-edge primitive, the motif
alphabet, the statistics, the IO layer, the event level and the producer;
every one of them passes its oracle tier and none of them had a caller.
`interaction_table.npz` is declared in `core/artifacts.py` and consumed by
`motif_stats`, `formation_gate` and `cross_head_gate`, and until this module
existed nothing in the repository wrote one outside a test. Seven registered
predictions -- CLAIM-B's three requirements, P-AB1's two, P-I3 and P-ST1 --
were waiting on the file rather than on the hardware.

--model NAMES THE PHASE 2 ARTIFACT, i.e. the registry key Phase 2 saved
ov_summary_{stem}.json / ov_weights_{stem}.npz under, not the HF repo. Same
rule as run_2d, and the same reason: the stem is what the files are on disk.

NOTHING IS INFERRED FROM A DIRECTORY NAME. `--prompt KEY=DIR` pairs a
battery key to a Phase 1 run directory explicitly. A driver that read the
prompt out of the path would be wrong the first time a directory is renamed,
and the failure is silent -- the run would proceed with another prompt's
induction pairs against these activations.

The three frames, and why two of them are refusals rather than defaults
---------------------------------------------------------------------
1. **The activations are stored on the L2 sphere.** `p1_io._save_activations`
   writes unit vectors plus a separate `norms` array. Phase 2's projectors
   live in the residual-stream basis, and `build_head_edges` refuses a frame
   mismatch it can detect -- but a caller that passed the normalized array
   while declaring "raw" would be handing it something that looks raw only
   because the norms are gone. This module multiplies the norms back in and
   refuses outright when they are absent, which is the case for Phase 1 runs
   predating that field: for those the raw stream is unrecoverable and no
   substitute exists.

2. **Which stored state is the input to layer l depends on the extraction
   convention**, and getting it wrong applies one layer's OV circuit to
   another layer's activations with no shape error to show for it. Read from
   geometry.json via `p2d_io.extraction_convention`; when unrecorded, this
   module stops (exit 2) exactly as run_2d does rather than assuming.

3. **The rotational channel is not wired here, and its absence is recorded
   rather than defaulted.** `U_S`/`U_A` stay None, so `real_frac` and
   `imag_frac` are NaN -- which finding 2 of status-7.md establishes must
   stay distinguishable from an honest 0.0. Supplying them needs Phase 2b's
   `extract_schur_blocks` output for this checkpoint, which is a separate
   run; `p7_io.rotational_channel_from_blocks` is the seam when it exists.
   The manifest records `rotational_channel: "absent"` so a table missing
   the channel cannot be mistaken for one measured to have none.

A degenerate prompt is skipped and named, not silently dropped and not fatal
on its own. `check_prompt_admissible` decides, on `core.battery_structure`'s
verdict; the run fails only when fewer than `--require-usable` prompts
survive, which is `assert_battery_structure`'s rule one level down. A run
with no usable prompt yields a null indistinguishable from a real negative.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from core.battery_structure import analyze_prompt
from core.interactions import InteractionTable
from .interaction_graph import DEFAULT_TOP_K_PER_TARGET, build_head_edges
from .motif_stats import DegeneratePrompt, check_prompt_admissible
from .p7_io import load_ov_circuits, load_sign_channel


class RunRefused(RuntimeError):
    """A precondition this driver will not proceed past."""


# ---------------------------------------------------------------------------
# Frame resolution
# ---------------------------------------------------------------------------

def raw_activations(run: dict) -> np.ndarray:
    """
    The residual stream in the basis Phase 2's projectors live in.

    Phase 1 stores unit vectors and their norms separately. Multiplying them
    back is the whole of the reconstruction; the refusal is the point of the
    function. `p1c_io.load_run` returns norms=None for runs written before
    the field existed, and that is a hard limit on what those artifacts can
    answer rather than something to work around with a unit-norm stand-in.
    """
    A = run.get("activations")
    if A is None:
        raise RunRefused(f"no activations.npz under {run.get('run_dir')}")
    norms = run.get("norms")
    if norms is None:
        raise RunRefused(
            f"{run.get('run_dir')} has activations.npz but no `norms` array, so "
            "the raw residual stream cannot be recovered. Phase 2's projectors "
            "are in the residual basis; the stored vectors are on the L2 "
            "sphere. Re-extract this run rather than substituting unit norms — "
            "every force magnitude would be wrong by a per-token factor."
        )
    return np.asarray(norms)[..., None] * np.asarray(A)


def layer_input_index(layer: int, n_states: int, n_layers: int,
                      embedding_stripped: bool) -> int:
    """
    Index into the stored states of the activations ENTERING `layer`.

    With the embedding kept, stored[0] is the embedding, which is the input
    to layer 0, and stored[l] is the input to layer l throughout. With it
    stripped, stored[0] is already layer 0's OUTPUT, so the input to layer l
    is stored[l-1] and LAYER 0'S INPUT IS NOT IN THE ARTIFACT AT ALL. That
    is reported as a refusal for that layer rather than quietly starting the
    sweep at layer 1, because a formation curve missing its first layer and
    one that has it are not the same measurement.
    """
    idx = layer if not embedding_stripped else layer - 1
    if idx < 0:
        raise RunRefused(
            f"layer {layer}'s input is the embedding, which this Phase 1 run "
            "did not store (hidden_state_0_is_embedding is false). Re-extract "
            "with the embedding kept, or exclude layer 0 explicitly with "
            "--layers — but do not report the result as a full-depth sweep."
        )
    if idx >= n_states:
        raise RunRefused(
            f"layer {layer} needs stored state {idx} but the run has "
            f"{n_states} states for {n_layers} layers. The extraction "
            "convention and the Phase 2 decomposition disagree about depth."
        )
    return idx


# ---------------------------------------------------------------------------
# The edge build
# ---------------------------------------------------------------------------

def edges_for_prompt(*, model: str, prompt_key: str, X_all: np.ndarray,
                     attentions: np.ndarray, ov: dict, weights_dir: Path,
                     sign_channel: str, pairs: dict, checkpoint_step,
                     embedding_stripped: bool, layers=None, heads=None,
                     top_k_per_target=DEFAULT_TOP_K_PER_TARGET) -> list:
    """One prompt's edges, one table per (layer, head)."""
    n_layers = ov["n_layers"]
    n_heads = ov["n_heads"]
    want_layers = list(range(n_layers)) if layers is None else list(layers)
    want_heads = list(range(n_heads)) if heads is None else list(heads)

    if attentions.shape[0] < n_layers:
        raise RunRefused(
            f"attentions.npz has {attentions.shape[0]} layers but Phase 2 "
            f"decomposed {n_layers}. These are not the same run."
        )
    if attentions.shape[1] != n_heads:
        raise RunRefused(
            f"attentions.npz has {attentions.shape[1]} heads per layer but "
            f"Phase 2 decomposed {n_heads}."
        )

    tables = []
    for layer in want_layers:
        idx = layer_input_index(layer, X_all.shape[0], n_layers, embedding_stripped)
        X = np.asarray(X_all[idx], dtype=np.float64)
        layer_rec = ov["layers"][layer if ov["is_per_layer"] else 0]
        chan = load_sign_channel(
            weights_dir, model, sign_channel,
            layer_name=layer_rec["layer_name"] if ov["is_per_layer"] else None,
        )
        if attentions.shape[-1] != X.shape[0]:
            raise RunRefused(
                f"{prompt_key}: attention is {attentions.shape[-1]} tokens wide "
                f"but the activations have {X.shape[0]}. The two artifacts are "
                "from different tokenizations."
            )
        for head in want_heads:
            tables.append(build_head_edges(
                model=model,
                prompt_key=prompt_key,
                layer=layer,
                head=head,
                X=X,
                attention=np.asarray(attentions[layer, head], dtype=np.float64),
                OV=np.asarray(layer_rec["heads"][head]["ov"], dtype=np.float64),
                U_pos=chan["U_pos"],
                U_neg=chan["U_neg"],
                U_S=None,
                U_A=None,
                induction_pairs=pairs["induction"],
                strict_pairs=pairs["strict"],
                same_content_pairs=pairs["same_content"],
                checkpoint_step=checkpoint_step,
                top_k_per_target=top_k_per_target,
                declared_frame="raw",
            ))
    return tables


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_prompt_arg(spec: str) -> tuple:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--prompt takes KEY=DIR, got {spec!r}. The battery key is never "
            "inferred from the directory name."
        )
    key, _, path = spec.partition("=")
    if not key or not path:
        raise argparse.ArgumentTypeError(f"--prompt takes KEY=DIR, got {spec!r}")
    return key, Path(path)


def resolve_checkpoint_step(model_name: str):
    """The training step this model key names, or None outside the registry."""
    from core.config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(model_name)
    return None if cfg is None else cfg.get("checkpoint_step")


def load_tokenizer(model_name: str):
    """
    The tokenizer the pair sets are derived from.

    A named function rather than an inline import so the pure tier can
    substitute one: transformers is genuinely unimportable there, and the
    pair-set construction and every refusal in this module are otherwise
    plain numpy. The substitution point is this function, not the import.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(resolve_hf_repo(model_name))


def resolve_hf_repo(model_name: str) -> str:
    """core/models.py's mapping, not a second one — hf_repo, then
    pretrained_name, then the key itself, so a bare HF id passes through."""
    from core.config import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        return model_name
    return cfg.get("hf_repo") or cfg.get("pretrained_name") or model_name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2-dir", type=Path, required=True,
                    help="directory holding ov_weights_/ov_summary_/"
                         "ov_projectors_{stem}.npz from Phase 2")
    ap.add_argument("--model", required=True,
                    help="the Phase 2 artifact stem (a MODEL_CONFIGS key), "
                         "not the HF repo id")
    ap.add_argument("--prompt", action="append", required=True,
                    type=_parse_prompt_arg, metavar="KEY=DIR",
                    help="battery key and its Phase 1 run directory; repeatable")
    ap.add_argument("--sign-channel", required=True, choices=("schur", "sym"),
                    help="'schur' splits the full operator on Re(lambda), "
                         "'sym' only the symmetric part. No default: which "
                         "one a result used changes what it means.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--heads", nargs="+", type=int, default=None)
    ap.add_argument("--top-k-per-target", type=int,
                    default=DEFAULT_TOP_K_PER_TARGET,
                    help="edge retention cutoff; recorded in the table. An "
                         "absent edge is not a zero-force edge.")
    ap.add_argument("--require-usable", type=int, default=1,
                    help="minimum prompts surviving the admissibility gate")
    ap.add_argument("--revision", default=None,
                    help="HF revision, recorded in the manifest. Not used to "
                         "load anything here — the weights came from Phase 2.")
    from core.holdout import HoldoutError, add_holdout_args, refuse_held_out
    add_holdout_args(ap)
    args = ap.parse_args(argv)

    # The Phase 1 runs are data/phase12 inputs, screened like every other
    # reader's (core/holdout.py): a held-out prompt's run is refused, or
    # dropped under --v1-only.
    try:
        kept, holdout = refuse_held_out(
            [d for _, d in args.prompt], allow=args.allow_holdout,
            drop=args.v1_only, context="run_7")
    except HoldoutError as exc:
        print(exc, file=sys.stderr)
        return 1
    args.prompt = [(k, d) for k, d in args.prompt if Path(d) in set(kept)]

    t0 = time.time()
    from p1c_frames.p1c_io import load_run
    from p2d_operator_activation.p2d_io import extraction_convention

    try:
        ov = load_ov_circuits(args.p2_dir, args.model)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"cannot load Phase 2 OV circuits: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 2 OV: {ov['n_layers']} layers x {ov['n_heads']} heads, "
          f"d_model={ov['d_model']} ({ov['source']})")

    step = resolve_checkpoint_step(args.model)
    tokenizer = None
    tables, skipped, used, inputs = [], [], [], {}

    for prompt_key, run_dir in args.prompt:
        run = load_run(run_dir)

        conv = extraction_convention(run)
        if conv["source"] != "artifact" or conv["embedding_stripped"] is None:
            print(f"{run_dir}: geometry.json does not record "
                  f"hidden_state_0_is_embedding, so which stored state is the "
                  f"input to a layer cannot be resolved.\n"
                  f"  Guessing applies one layer's OV circuit to another "
                  f"layer's activations, silently. Re-extract.",
                  file=sys.stderr)
            return 2

        from core.config import PROMPTS
        if prompt_key not in PROMPTS:
            print(f"{prompt_key!r} is not a battery key; known: "
                  f"{sorted(PROMPTS)}", file=sys.stderr)
            return 1

        if tokenizer is None:
            tokenizer = load_tokenizer(args.model)

        from core.battery_structure import (
            TokenizationMismatch, induction_candidates, phase1_tokens,
            same_content_candidates, verified_prompt_ids)
        # The pairs index the run's activations, so they are built from the
        # tokens the run was extracted on, checked against the live text,
        # not from whatever the live text tokenises to today. Checked before
        # the admissibility gate, so a changed text is refused, not skipped.
        try:
            ids = verified_prompt_ids(tokenizer, PROMPTS[prompt_key],
                                      phase1_tokens(run_dir))
        except (OSError, TokenizationMismatch) as exc:
            print(f"{prompt_key}: {exc}. Pairs built from the live text would "
                  f"index a different tokenisation than {run_dir}.",
                  file=sys.stderr)
            return 1

        from core.io import load_manifest
        p1 = load_manifest(run_dir) or {}
        p1_step = p1.get("checkpoint_step")
        if p1_step is not None and step is not None and int(p1_step) != int(step):
            print(f"{run_dir}: its manifest says checkpoint {p1_step}, "
                  f"--model {args.model} is checkpoint {step}. Every edge would "
                  f"carry the wrong training step.", file=sys.stderr)
            return 1

        # On the ids the run holds: a run cut at extraction is judged on its
        # prefix, which is all the pairs can index.
        report = analyze_prompt(tokenizer, prompt_key, PROMPTS[prompt_key],
                                ids=ids)
        try:
            check_prompt_admissible(report, prompt_key)
        except DegeneratePrompt as exc:
            print(f"  SKIP {prompt_key}: {exc}")
            skipped.append({"prompt": prompt_key, "verdict": report["verdict"],
                            "flags": report["flags"]})
            continue

        ind = induction_candidates(ids)
        pairs = {"induction": ind,
                 "strict": induction_candidates(ids, strict=True),
                 "same_content": same_content_candidates(ids, ind)}

        attn_p = Path(run_dir) / "attentions.npz"
        if not attn_p.exists():
            print(f"{attn_p} not found. Phase 7 types edges by the force a "
                  f"head moves, which needs the post-softmax weights; Phase 1 "
                  f"writes them only when asked.", file=sys.stderr)
            return 1
        attentions = np.load(attn_p, allow_pickle=False)["attentions"]

        try:
            X_all = raw_activations(run)
            if X_all.shape[1] != len(ids):
                raise RunRefused(
                    f"tokens.txt holds {len(ids)} tokens but the activations "
                    f"have {X_all.shape[1]}. The run is not one extraction.")
            tables.extend(edges_for_prompt(
                model=args.model, prompt_key=prompt_key, X_all=X_all,
                attentions=attentions, ov=ov, weights_dir=args.p2_dir,
                sign_channel=args.sign_channel, pairs=pairs,
                checkpoint_step=step,
                embedding_stripped=bool(conv["embedding_stripped"]),
                layers=args.layers, heads=args.heads,
                top_k_per_target=args.top_k_per_target,
            ))
        except RunRefused as exc:
            print(f"{prompt_key}: {exc}", file=sys.stderr)
            return 1
        used.append(prompt_key)
        inputs[prompt_key] = {
            "run_dir": str(run_dir),
            "tokens_txt_sha256": hashlib.sha256(
                (Path(run_dir) / "tokens.txt").read_bytes()).hexdigest(),
            "n_tokens": len(ids),
            "p1_manifest_id": p1.get("manifest_id"),
            "p1_prompt_battery_hash": p1.get("prompt_battery_hash"),
        }
        print(f"  {prompt_key}: {report['n_tokens']} tokens, "
              f"{len(ind)} induction pairs")

    if len(used) < args.require_usable:
        print(f"only {len(used)} usable prompt(s), needed {args.require_usable}. "
              f"A run in this state yields a null result indistinguishable from "
              f"a real negative.", file=sys.stderr)
        return 1

    table = InteractionTable.concat(tables)
    args.out.mkdir(parents=True, exist_ok=True)
    out_p = args.out / "interaction_table.npz"
    table.save(out_p)
    print(f"\n{len(table)} edges over {len(used)} prompt(s) -> {out_p}")

    from core.config import PROMPTS
    from core.io import get_git_sha, write_manifest
    from core.prompts import compute_prompt_battery_hash
    write_manifest(
        args.out,
        model=args.model,
        # A hash of the texts actually used, not a key list: a changed text
        # under an unchanged key must change it.
        prompt_battery_hash=compute_prompt_battery_hash(
            {k: PROMPTS[k] for k in used}),
        wall_time_seconds=time.time() - t0,
        hf_revision=args.revision,
        checkpoint_step=step,
        git_sha=get_git_sha(),
        config={"sign_channel": args.sign_channel,
                "top_k_per_target": args.top_k_per_target,
                "layers": args.layers, "heads": args.heads},
        extra={"phase": "7",
               "rotational_channel": "absent",
               "prompts_used": used,
               # What each table row was read from: the run, the exact
               # tokens its pairs index, and the Phase 1 provenance.
               "inputs": inputs,
               "holdout": holdout,
               "prompts_skipped": skipped,
               "p2_source": ov["source"]},
    )
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
