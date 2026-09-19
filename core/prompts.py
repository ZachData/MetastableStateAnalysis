"""
core/prompts.py — Versioned prompt battery (transition plan v2, core
infrastructure item 5).

`core.config.PROMPTS` is the actual prompt text; this module is the
accountability layer on top of it. Cross-checkpoint comparison (the whole
point of the transition project) is only meaningful if every checkpoint
and both model sizes were run against the *identical* prompt set — a
silently-edited prompt would invalidate a comparison without either side
erroring. This module gives every run a short, deterministic hash of the
exact battery it used, which core.io.write_manifest records in every
manifest.json, so any two runs can be checked for prompt-battery identity
without diffing text by hand.

Usage
-----
    from core.prompts import PROMPT_BATTERY_HASH
    write_manifest(..., prompt_battery_hash=PROMPT_BATTERY_HASH)

Versioning policy: a change to the prompt text or the prompt set is a
deliberate act, so it must be paired with bumping PROMPT_BATTERY_VERSION.
This module does not try to detect an un-bumped change by comparing
against a hardcoded expected hash (that would require re-hardcoding the
expected value every time the battery legitimately changes, which is
exactly the kind of check that silently goes stale). Instead, the hash
itself is the accountability mechanism: it is recorded in every run's
manifest, and two runs meant to be comparable are checked for identical
prompt-battery hashes at analysis time (see verify_same_battery below).

THE v2 EXTENSION RULE — fixed 2026-09-19, BEFORE any text was chosen
-------------------------------------------------------------------
This paragraph is committed on its own, ahead of the commit that adds the
text, so that git shows the rule predates the prompts rather than being
written around them. That ordering is the whole point: the prompt is
`CLAIM-C`'s exchangeable unit, so choosing prompts after seeing how they
behave would be selection, and a gate whose rows were picked that way
reports nothing.

WHY TWELVE, and why it is not a fact about any prompt. `PROJECT.md` §3.41:
the gate ran on the eight v1 metastability prompts and could not express a
p below 0.0661, because six metrics is an EVEN number of cells per row and
four of the eight rows split exactly 3-3 — a tied row contributes the same
number to the observed sum and to every null pattern, so it is enumerated
without ever being counted. Twelve more rows at the same informative rate
(four in eight) is what brings the design's attainable p back under
alpha = 0.05. The arithmetic is about the FLOOR of the design, not about
any measured contrast, which is why it can be acted on without tainting
the test.

THE RULE.
  1. Twelve additions. v1's nine prompts keep their keys and their text
     exactly, so every v1 row remains the same row.
  2. Six genres, two prompts each — the genres already present among v1's
     eight metastability prompts: encyclopedic reference, first-person
     historical letter, technical/scientific exposition, literary
     narrative (one of the pair non-English, as v1 has), source code,
     structured markup.
  3. Per pair, one prompt in v1's short band (1000-1600 characters) and
     one in its long band (1800-2400). v1's 115-character
     `short_heterogeneous` stays the only prompt of its size; the
     extension does not change the battery's length profile.
  4. Sources fixed by genre in advance: Project Gutenberg public-domain
     texts for narrative and letters; English Wikipedia for encyclopedic;
     BSD/MIT-licensed source already installed in this venv for code; an
     open-access arXiv paper for technical prose; composed LaTeX for
     markup — matching where v1's own prompts come from.
  5. Within a source, the passage is the FIRST one meeting the length band
     from a predetermined anchor (the start of a named chapter, section or
     function). No candidate is run through a model before inclusion.
  6. Once any model has been run on v2, no prompt in it may be added,
     dropped or edited. A prompt that turns out uninformative stays.

CONSEQUENCE, stated rather than discovered later: v1 and v2 runs have
different battery hashes and `verify_same_battery` will refuse to compare
them. Every run on disk as of 2026-09-19 is v1, including Phase 1's
19-checkpoint sweep, so a v2 run is comparable only with other v2 runs.
This is the mechanism working; it is also a real cost, and it is the reason
the extension is a deliberate act rather than a convenience.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from core.config import PROMPTS

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

# Bump this any time PROMPTS's keys or text change, even by one character.
# The version string is folded into the hash below, so a bump alone changes
# every future manifest's prompt_battery_hash even if (by mistake) the text
# ended up identical to a prior version.
PROMPT_BATTERY_VERSION = "v2"   # v2 = v1 + twelve, 2026-09-19; see the rule above


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_prompt_battery_hash(
    prompts: Optional[dict] = None,
    version: Optional[str] = None,
) -> str:
    """
    Deterministic 12-hex-char hash over a prompt battery, same short-id
    convention as core.io.compute_manifest_id (sha256, truncated to 12
    hex chars) so both ids read consistently in a manifest.json.

    Deterministic in (sorted (key, text) pairs, version) — key order in
    the source dict doesn't matter, only the actual content does. Two
    processes loading the same PROMPTS dict in a different insertion
    order still get the same hash.

    Parameters
    ----------
    prompts : defaults to core.config.PROMPTS
    version : defaults to PROMPT_BATTERY_VERSION
    """
    if prompts is None:
        prompts = PROMPTS
    if version is None:
        version = PROMPT_BATTERY_VERSION

    parts = [str(version)]
    for key in sorted(prompts.keys()):
        parts.append(key)
        parts.append(prompts[key])

    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


# Computed once at import time — the single source of truth every run
# should pass to write_manifest. Recomputed (not hardcoded) so it always
# reflects whatever core.config.PROMPTS actually contains right now.
PROMPT_BATTERY_HASH = compute_prompt_battery_hash()


# ---------------------------------------------------------------------------
# Cross-run verification
# ---------------------------------------------------------------------------

def verify_same_battery(hash_a: str, hash_b: str) -> bool:
    """
    True iff two manifest.json prompt_battery_hash values match. Trivial
    on its own (`==` would do), but named so call sites read as an
    intentional check rather than an incidental string comparison, and so
    the one place this comparison happens can later grow additional
    checks (e.g. version compatibility) without touching every call site.
    """
    return hash_a == hash_b


def assert_same_battery(hash_a: str, hash_b: str, context: str = "") -> None:
    """
    Raise loudly if two runs meant to be compared used different prompt
    batteries. Use this before any cross-checkpoint or cross-model
    comparison that assumes identical prompts (which is nearly all of
    them) — a silent mismatch here would produce a plausible-looking but
    invalid comparison, which is worse than a crash.
    """
    if not verify_same_battery(hash_a, hash_b):
        where = f" ({context})" if context else ""
        raise ValueError(
            f"Prompt battery mismatch{where}: {hash_a!r} != {hash_b!r}. "
            "These runs were not produced with the same prompt battery — "
            "any comparison between them is invalid until this is resolved."
        )
