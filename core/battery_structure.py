"""
battery_structure.py — Does the prompt battery still have the structure it
was designed to have, under THIS tokenizer? (frames item 8, and item 12.)

Why this module exists
----------------------
`core/prompts.py` hashes the battery text, which guarantees two runs used
identical *strings*. It says nothing about whether those strings still have
the structure the analysis depends on after tokenization. A prompt built to
repeat a bigram under GPT-2 BPE can tokenize into non-repeating ids under
the NeoX tokenizer, silently yielding zero induction pairs — a null result
that looks like a finding and is actually a tokenizer artifact.

Nothing in the pipeline would error. That is the whole problem.

This module also closes item 12. `run_6.py` builds token identity as
`hash(token_string) % 2**31`, and Python salts string hashing per process:
the ids are not reproducible across runs, and the modulus admits birthday
collisions that manufacture false same-content pairs — which are P6-I2b's
null. Real tokenizer ids are already available at extraction time; this
module is where they get captured and recorded.

What "structure" means here
---------------------------
Two pair families, defined on token ids:

  induction     (query, key) with ids[key - 1] == ids[query - 1]
                — the repo's stated condition (qk_decompose docstring):
                the token BEFORE each position matches, so the head can
                use "one ahead of the match" to predict.

  strict        (query, key) with ids[key - 1] == ids[query]
                — the Anthropic formulation: the query token itself
                previously occurred at key - 1, so key is what comes next.

They are not the same set, and the repo tests the first while citing the
second. Both are reported; a large divergence between them is a signal that
the induction analysis is not measuring what its docstring claims.

Degeneracy
----------
The check that matters most is not "are there induction pairs" but "is the
comparison non-degenerate". Three ways it fails:

  uniform          one distinct token id — every causal pair is trivially an
                   induction pair and the same-content null is EMPTY
  empty_null       no same-content pairs survive the induction exclusion
  single_offset    all induction pairs share one offset, so N3 has no power
                   (see qk_offset_null.offset_shuffled_null)
  null_is_sink     the same-content null set consists (almost) entirely of
                   pairs whose key is position 0. On a uniform prompt the
                   only pairs excluded from the induction set are those with
                   key = 0, so the "null" collapses onto the attention sink
                   column and P6-I2b would be comparing induction structure
                   against sink behaviour (policy P1) rather than against
                   content-matched pairs

`config.PROMPTS["repeated_tokens"]` is a worked example of the first: it is
". " repeated ~264 times. Every token is identical, so every causal pair
qualifies as induction, the same-content set is empty, and P6-I2b cannot be
evaluated on it at all. That prompt is presumably in the battery to probe
uniform-input geometry, which is a legitimate different purpose — but it
must not be counted toward induction coverage.

See DESIGN_pythia_frames.md items 8 and 12.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


MIN_PAIRS_FOR_TEST = 3


# ---------------------------------------------------------------------------
# Tokenization capture
# ---------------------------------------------------------------------------

def tokenize_prompt(tokenizer, text: str) -> dict:
    """
    Real token ids, plus the BOS question.

    Duck-typed on the HF tokenizer call interface so a fake works in the
    stubbed session. Returns ids as a plain list of ints — NOT hashes of
    token strings (item 12).
    """
    enc = tokenizer(text)
    ids = list(enc["input_ids"] if isinstance(enc, dict) else enc)
    bos = getattr(tokenizer, "bos_token_id", None)
    has_bos = bool(ids and bos is not None and ids[0] == bos)
    return {
        "ids": ids,
        "n_tokens": len(ids),
        "n_distinct": len(set(ids)),
        "has_bos": has_bos,
    }


# ---------------------------------------------------------------------------
# Pair structure
# ---------------------------------------------------------------------------

def induction_candidates(ids, min_offset: int = 2, strict: bool = False) -> list:
    """
    (query, key) pairs, query > key, matching the induction pattern on ids.

    strict=False : ids[key - 1] == ids[query - 1]   (the repo's condition)
    strict=True  : ids[key - 1] == ids[query]       (the Anthropic condition)

    Both require key >= 1 so that key - 1 exists. Returned canonically with
    query > key, matching qk_decompose's convention and therefore indexable
    against a_frac_mat[query, key].
    """
    n = len(ids)
    out = []
    for key in range(1, n):
        left = ids[key - 1]
        for query in range(key + min_offset, n):
            probe = ids[query] if strict else (ids[query - 1] if query >= 1 else None)
            if probe is not None and probe == left:
                out.append((query, key))
    return out


def same_content_candidates(ids, induction, min_offset: int = 2) -> list:
    """
    (query, key) pairs with ids[query] == ids[key] and NOT already induction.

    This is P6-I2b's null set. Exact id equality only — the original
    implementation also admits a cosine-similarity fallback on activations,
    which makes the null set depend on the frame the activations are in and
    therefore on the bug this work stream is fixing. Id equality is
    frame-independent by construction.
    """
    ind = set(induction)
    n = len(ids)
    out = []
    for key in range(n):
        for query in range(key + min_offset, n):
            if (query, key) in ind:
                continue
            if ids[query] == ids[key]:
                out.append((query, key))
    return out


def pair_offsets(pairs) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.int64)
    a = np.asarray(pairs, dtype=np.int64)
    return a[:, 1] - a[:, 0]


# ---------------------------------------------------------------------------
# Per-prompt verdict
# ---------------------------------------------------------------------------

def analyze_prompt(tokenizer, name: str, text: str, min_offset: int = 2) -> dict:
    """
    Full structural report for one prompt under one tokenizer.

    Returns a dict carrying the counts, the degeneracy flags, and a verdict
    of "usable" / "degenerate" / "insufficient" for induction analysis.
    """
    tok = tokenize_prompt(tokenizer, text)
    ids = tok["ids"]

    ind = induction_candidates(ids, min_offset, strict=False)
    ind_strict = induction_candidates(ids, min_offset, strict=True)
    same = same_content_candidates(ids, ind, min_offset)

    ind_off = pair_offsets(ind)
    n_distinct_offsets = int(np.unique(ind_off).size) if ind_off.size else 0

    flags = []
    if tok["n_distinct"] <= 1:
        flags.append("uniform")
    if ind and not same:
        flags.append("empty_null")
    if ind_off.size and n_distinct_offsets < 2:
        flags.append("single_offset")

    # A null set that is really the sink column. The substantive criterion is
    # not the fraction — with few pairs a fraction is noise — but whether
    # enough null pairs survive once the sink column is removed.
    n_sink = sum(1 for _q, k in same if k == 0)
    n_nonsink = len(same) - n_sink
    sink_frac = (n_sink / len(same)) if same else 0.0
    if same and n_nonsink < MIN_PAIRS_FOR_TEST:
        flags.append("null_is_sink")

    # Divergence between the repo's condition and the cited one.
    overlap = len(set(ind) & set(ind_strict))
    denom = max(len(set(ind) | set(ind_strict)), 1)
    condition_agreement = overlap / denom

    if len(ind) < MIN_PAIRS_FOR_TEST or len(same) < MIN_PAIRS_FOR_TEST:
        verdict = "insufficient"
    elif {"uniform", "empty_null", "null_is_sink"} & set(flags):
        verdict = "degenerate"
    else:
        verdict = "usable"

    return {
        "name": name,
        "n_tokens": tok["n_tokens"],
        "n_distinct_tokens": tok["n_distinct"],
        "has_bos": tok["has_bos"],
        "n_induction": len(ind),
        "n_induction_strict": len(ind_strict),
        "condition_agreement": condition_agreement,
        "n_same_content": len(same),
        "null_sink_fraction": sink_frac,
        "n_same_content_nonsink": n_nonsink,
        "n_distinct_induction_offsets": n_distinct_offsets,
        "induction_offsets": sorted(set(int(o) for o in ind_off)),
        "flags": flags,
        "verdict": verdict,
        "structure_hash": _structure_hash(ids),
    }


def _structure_hash(ids) -> str:
    """
    Short hash of the tokenized structure, not the text.

    Recorded next to PROMPT_BATTERY_HASH so a manifest states both "the same
    strings" and "the same tokenization of those strings". A tokenizer swap
    changes this while leaving the text hash untouched — which is precisely
    the failure the text hash cannot see.
    """
    payload = json.dumps(list(map(int, ids)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Battery-level verification
# ---------------------------------------------------------------------------

def verify_battery_structure(tokenizer, prompts, min_offset: int = 2,
                             require_usable: int = 1) -> dict:
    """
    Structural report for the whole battery.

    `require_usable` is the number of prompts that must be usable for
    induction analysis. Zero usable prompts means P6-I2b cannot be evaluated
    at all, and that must surface as a loud precondition failure rather than
    as an empty result table downstream.

    Returns dict(per_prompt, n_usable, usable_names, battery_structure_hash,
    tokenizer_name, any_bos, ok).
    """
    per = [analyze_prompt(tokenizer, name, text, min_offset)
           for name, text in sorted(prompts.items())]
    usable = [p["name"] for p in per if p["verdict"] == "usable"]
    combined = hashlib.sha256(
        "".join(p["structure_hash"] for p in per).encode()
    ).hexdigest()[:16]
    return {
        "per_prompt": per,
        "n_usable": len(usable),
        "usable_names": usable,
        "battery_structure_hash": combined,
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", "")),
        "any_bos": any(p["has_bos"] for p in per),
        "ok": len(usable) >= require_usable,
    }


def assert_battery_structure(tokenizer, prompts, min_offset: int = 2,
                             require_usable: int = 1, context: str = "") -> dict:
    """
    Fail loud at run entry. Returns the report when it passes.

    Called from every run_*.py that depends on induction pairs. A battery
    that tokenizes into no usable induction structure produces a null result
    indistinguishable from a real negative, so it must stop the run.
    """
    rep = verify_battery_structure(tokenizer, prompts, min_offset, require_usable)
    if not rep["ok"]:
        where = f" [{context}]" if context else ""
        summary = ", ".join(
            f"{p['name']}={p['verdict']}" for p in rep["per_prompt"]
        )
        raise ValueError(
            f"Prompt battery has no usable induction structure under "
            f"{rep['tokenizer_name'] or 'this tokenizer'}{where}. "
            f"Needed {require_usable} usable prompt(s), found {rep['n_usable']}. "
            f"Per prompt: {summary}. A run in this state yields a null result "
            f"that is a tokenizer artifact, not a finding."
        )
    return rep


def battery_summary_lines(report: dict) -> list:
    lines = [
        "Prompt battery structure:",
        f"  tokenizer     {report['tokenizer_name'] or '(unnamed)'}",
        f"  structure     {report['battery_structure_hash']}",
        f"  BOS prepended {'yes' if report['any_bos'] else 'NO (position 0 is a content token)'}",
        f"  usable        {report['n_usable']} of {len(report['per_prompt'])}"
        f" — {', '.join(report['usable_names']) or 'none'}",
    ]
    for p in report["per_prompt"]:
        flag = f"  [{','.join(p['flags'])}]" if p["flags"] else ""
        lines.append(
            f"    {p['name']:22s} {p['verdict']:12s} "
            f"tok={p['n_tokens']:4d} distinct={p['n_distinct_tokens']:4d} "
            f"ind={p['n_induction']:5d} null={p['n_same_content']:5d}"
            f"(non-sink {p['n_same_content_nonsink']:4d}) "
            f"defn_agree={p['condition_agreement']:.2f}{flag}"
        )
    return lines
