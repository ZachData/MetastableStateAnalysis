"""
core/intervention.py — Merged intervention + logits runner (transition plan
v2, core analysis primitives, item 2 of 4).

Plan text: "Merged intervention + logits runner — consolidate
causal_tests.py's `_run_albert_with_hook` and dissociation.py's hooks into
one model-agnostic runner with logits/loss output. Unblocks phase 5b,
5c Group D, phase 6."

*** NOT RUNTIME-VERIFIED ***
This module requires torch and a real HuggingFace model to execute.
Neither is available in the sandbox this was written in (no network
access to install torch). Everything else in this pass (core/population.py
and its five consumers) was checked against a real, passing test run;
this module and its refactor of dissociation.py / p5_single_mstate_analysis/
causal_tests.py were not. Run tests/test_core_intervention.py for real
(pytest + torch + a small HF model, e.g. gpt2) before trusting this in a
pipeline. Read the code especially carefully as a result.

What existed before this module
--------------------------------
- p5_single_mstate_analysis/causal_tests.py::_run_albert_with_hook —
  manually reimplements the forward pass (embeddings, then a per-layer or
  per-iteration loop) so that a `hook_fn(step, hidden, layer)` callback can
  run *before* a specific layer/iteration computes its update. Returns
  (trajectory, attentions, tokens) — no logits, no loss. Two branches:
  GPT-2 (loop over `transformer.h`, one block per iteration) and ALBERT
  (loop `max_iterations` times over the single shared layer — `max_iterations`
  can exceed `model.config.num_hidden_layers`, the whole point of ALBERT's
  extended-iteration methodology from Blog 1).
- p6_subspace/dissociation.py::run_intervened_forward — uses the standard
  HuggingFace call (`model(**inputs, output_hidden_states=True,
  output_attentions=True)`) plus `register_forward_hook` on a list of
  target modules, firing on *every* invocation (no per-step gating — every
  layer, every iteration, uniformly, which is what the double-dissociation
  design actually wants). Also no logits, no loss.

Why this module is scoped to the *standard* (native forward pass) case
only, not a full merge of both
-------------------------------------------------------------------------
The standard HuggingFace call (`model(**inputs, output_hidden_states=True)`)
is sufficient for any model where the number of forward-pass layer
invocations is fixed by the model's own config — every per-layer model
(GPT-2, BERT, and Pythia's GPT-NeoX) and ALBERT run at its *native*
iteration count (`max_iterations == model.config.num_hidden_layers`). It is
NOT sufficient for ALBERT/BERT run for *more* iterations than
`config.num_hidden_layers` — the standard API has no parameter for that,
and `_run_albert_with_hook`'s manual reimplementation exists specifically
to support it. Per the transition plan's own scope decision ("Pythia-only,
one frozen exception... the multi-architecture comparison (ALBERT vs. BERT
vs. GPT-2) is closed as a reported finding"), extended-iteration ALBERT
causal testing is not part of any forward-going work this consolidation
needs to serve (design-5c.md and status-5c.md are explicit that Group D
targets GPT-2-large; Phase 5b and Phase 6 are likewise GPT-2/Pythia work).
This module therefore covers the standard case only. `_run_albert_with_hook`
is NOT deleted — p5_single_mstate_analysis/causal_tests.py still dispatches
to it for the ALBERT-extended-iteration case specifically; see that file's
`ablate_head` / `steer_residual` / `patch_activation` for the dispatch.

Layer-indexing convention (a correction, not a preference)
------------------------------------------------------------
`activations[0]` is the *embedding* layer here, matching
core/models.py's own extract_activations and extract_albert_extended —
the functions that actually produced every existing Phase 1 hdb_labels
array. dissociation.py's prior run_intervened_forward skipped the
embedding (`hidden_states[1:]`), which was internally self-consistent
within that one file (baseline and intervention runs were always compared
against each other, never against externally-supplied labels) but did NOT
match core/models.py's convention — a real misalignment risk if
`ctx["baseline_labels"]` were ever supplied from genuine Phase 1 output
rather than computed fresh inside the same call, since Phase 1's labels
are keyed to the embedding-included convention. This module adopts the
core/models.py convention; p6_subspace/dissociation.py's refactor to use
it fixes that latent misalignment as a natural side effect of the merge,
not as a separately-tracked bug.

Logits / loss (new capability, not present in either predecessor)
--------------------------------------------------------------------
Neither `_run_albert_with_hook` nor `run_intervened_forward` computed
logits or loss at all. design-5c.md is explicit about why Group D needs
it: "Readout is next-token cross-entropy delta and KL divergence, run on
both the trained model and its random-weight twin per arm." This module
adds `logits` to every run's return value (when the model has an LM head)
and an optional `loss` (standard causal-LM next-token cross-entropy — see
`run_model_with_hook`'s docstring for the masked-LM caveat), plus
`next_token_kl` / `next_token_kl_all_positions` for the KL half of that
readout.

Hook mechanics — the step-gating closure
-------------------------------------------
`_run_albert_with_hook`'s `step == target_layer` check worked because it
owned the loop and could count iterations itself. Using the standard API
means the loop is inside HuggingFace's own code, so this module's
`_step_gated_hook` counts invocations *on the hook itself* instead: a
closure counter incremented every time the hook fires, comparing against
a caller-supplied `steps` set. This generalizes both prior behaviors
uniformly: `steps=None` fires on every invocation of the target module
(dissociation.py's original, ungated behavior — the double-dissociation
design's "every layer, every head" intent); `steps=[k]` fires only on that
module's (k+1)-th invocation during this forward pass — for GPT-2 (or any
per-layer model), each layer's submodule is invoked exactly once, so
`steps=[0]` selects that layer; for ALBERT at its native iteration count,
the single shared submodule is invoked once per iteration, so `steps=[k]`
selects iteration k specifically, the same distinction
`_run_albert_with_hook`'s `step` argument made.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Step-gated hook wrapper
# ---------------------------------------------------------------------------

def _step_gated_hook(hook_fn: Callable, steps: Optional[Sequence[int]], is_pre: bool):
    """
    Wrap a plain PyTorch hook so it only executes when the invocation
    count of the module it's attached to (0-based, counted by this
    wrapper, reset per `register_*_hook` call) is in `steps`. `steps=None`
    means "every invocation" — see module docstring.

    hook_fn keeps the plain PyTorch hook signature for `is_pre`:
      is_pre=True  : hook_fn(module, inputs) -> modified inputs or None
      is_pre=False : hook_fn(module, inputs, output) -> modified output or None
    It is NOT given the step index — gating already restricts *when* it's
    called to the requested steps, so it doesn't need to re-check.
    """
    counter = {"n": 0}

    if is_pre:
        def _hook(module, inputs):
            i = counter["n"]
            counter["n"] += 1
            if steps is None or i in steps:
                return hook_fn(module, inputs)
            return None
        return _hook

    def _hook(module, inputs, output):
        i = counter["n"]
        counter["n"] += 1
        if steps is None or i in steps:
            return hook_fn(module, inputs, output)
        return None
    return _hook


# ---------------------------------------------------------------------------
# The merged runner
# ---------------------------------------------------------------------------

def run_model_with_hook(
    model,
    tokenizer,
    text: str,
    hooks: Optional[list] = None,
    device: str = "cpu",
    max_length: int = 512,
    compute_loss: bool = False,
    labels=None,
) -> dict:
    """
    Run one forward pass through any HuggingFace model that supports
    `output_hidden_states` / `output_attentions`, with zero or more
    intervention hooks, returning activations, attentions, logits, and
    (optionally) loss uniformly regardless of architecture.

    Parameters
    ----------
    hooks : list of dicts, each:
        {
          "module":  nn.Module,             # required
          "hook_fn": callable,              # required; see _step_gated_hook
          "type":    "forward" | "forward_pre",   # default "forward"
          "steps":   list[int] | None,      # default None (every invocation)
        }
        None or [] runs a plain baseline forward pass.
    device       : "cpu" or "cuda". Inputs are moved here before the call.
    max_length   : tokenizer truncation length.
    compute_loss : if True and the model has a `.logits` output, compute
        standard next-token cross-entropy (see below). If False, `loss`
        in the return value is always None regardless of `labels`.
    labels       : optional (n,) int array/tensor of target token ids for
        the loss. If None and compute_loss=True, defaults to the input's
        own `input_ids` shifted by one — ordinary self-supervised
        next-token language-modeling loss, matching design-5c.md's Group D
        readout ("next-token cross-entropy delta").

    Returns
    -------
    dict:
      activations : list of (n_tokens, d_model) float32 np.ndarray, one
          per `hidden_states` entry INCLUDING the embedding at index 0
          (see module docstring — this is core/models.py's convention,
          not dissociation.py's prior skip-the-embedding one).
      attentions  : list of (n_heads, n_tokens, n_tokens) float32
          np.ndarray, one per transformer layer (no embedding entry).
          [] if the model doesn't return attentions.
      logits      : (n_tokens, vocab) float32 np.ndarray, or None if the
          model has no `.logits` output (e.g. a bare encoder without an
          LM head).
      loss        : float, or None if compute_loss=False or logits is None.
          Standard SHIFTED next-token cross-entropy: logits[:-1] compared
          against input_ids[1:] (or `labels[1:]`-equivalent if `labels` is
          given — `labels` is taken as the full target sequence and
          shifted the same way). This is the causal-LM convention; it is
          NOT the right loss for a masked-LM readout (no shift, loss only
          at masked positions) — every model this consolidation actually
          needs to serve (GPT-2-large, Pythia) is a causal LM, so that
          convention is what's implemented. Passing a masked-LM model with
          compute_loss=True will not raise, but the number it returns
          isn't the right one for that architecture.
      tokens      : list[str].
    """
    import torch

    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    handles = []
    for spec in (hooks or []):
        module    = spec["module"]
        hook_fn   = spec["hook_fn"]
        hook_type = spec.get("type", "forward")
        steps     = spec.get("steps", None)

        if hook_type == "forward_pre":
            gated = _step_gated_hook(hook_fn, steps, is_pre=True)
            handles.append(module.register_forward_pre_hook(gated))
        elif hook_type == "forward":
            gated = _step_gated_hook(hook_fn, steps, is_pre=False)
            handles.append(module.register_forward_hook(gated))
        else:
            raise ValueError(
                f"run_model_with_hook: unknown hook type {hook_type!r}, "
                f"expected 'forward' or 'forward_pre'."
            )

    try:
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    activations = [h[0].float().cpu().numpy() for h in out.hidden_states]

    attentions = []
    if getattr(out, "attentions", None) is not None:
        attentions = [a[0].float().cpu().numpy() for a in out.attentions]

    logits = None
    out_logits = getattr(out, "logits", None)
    if out_logits is not None:
        logits = out_logits[0].float().cpu().numpy()

    loss = None
    if compute_loss and out_logits is not None:
        if labels is not None:
            target_ids = torch.as_tensor(labels, device=out_logits.device, dtype=torch.long)
        else:
            target_ids = inputs["input_ids"][0]
        shift_logits = out_logits[0, :-1, :]
        shift_targets = target_ids[1:]
        n = min(shift_logits.shape[0], shift_targets.shape[0])
        if n > 0:
            loss = torch.nn.functional.cross_entropy(
                shift_logits[:n], shift_targets[:n]
            ).item()

    return {
        "activations": activations,
        "attentions":  attentions,
        "logits":      logits,
        "loss":        loss,
        "tokens":      tokens,
    }


# ---------------------------------------------------------------------------
# Group D readout: next-token KL divergence
# ---------------------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def next_token_kl(logits_a: np.ndarray, logits_b: np.ndarray, position: int = -1) -> float:
    """
    KL(softmax(logits_a) || softmax(logits_b)) at one token position
    (default: the last position — the next-token prediction point).

    Parameters
    ----------
    logits_a, logits_b : (n_tokens, vocab) float arrays, e.g. baseline vs
        intervened logits from two run_model_with_hook calls on the same
        text. Must have the same vocab size; n_tokens may differ only if
        `position` indexes validly into both.
    position : token position to compare. Negative indices supported.

    Returns
    -------
    float >= 0 (KL divergence is never negative; 0.0 iff the two
    distributions at that position are identical).
    """
    a = logits_a[position].astype(np.float64)
    b = logits_b[position].astype(np.float64)
    log_p = a - _logsumexp(a)
    log_q = b - _logsumexp(b)
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def next_token_kl_all_positions(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    """
    KL(softmax(logits_a) || softmax(logits_b)) at every position shared by
    both. (min(n_tokens_a, n_tokens_b),) float array.
    """
    n = min(logits_a.shape[0], logits_b.shape[0])
    return np.array([next_token_kl(logits_a, logits_b, position=i) for i in range(n)])
