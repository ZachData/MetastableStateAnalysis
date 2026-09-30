"""
p1d_cluster_ensemble/long_prompts.py — six v1 prompts continued to Pythia's
context length, so Phase 1d can ask what length changes (`lit-1d.md` §4:
length first, count later; the user's direction of 2026-09-25 and
2026-09-29).

THE RULE — fixed 2026-09-29, BEFORE any continuation text was chosen
--------------------------------------------------------------------
This docstring is committed on its own, ahead of the commit that adds the
text, as `core/prompts.py`'s v2 rule was, so git shows the rule predates the
text. Nothing below is chosen after running a model on it.

  1. **Which prompts.** The six v1 metastability prompts that can be
     continued from their own source: `wiki_paragraph`, `repeated_tokens`,
     `sullivan_ballou`, `paper_excerpt`, `hdbscan_code`, `latex_monograph`.
     `homer_iliad` (it appears to be Fagles's 1990 translation) and
     `camus_letranger` (1942) are in copyright and are dropped, not
     replaced (user, 2026-09-29).
  2. **Prefix.** Each long prompt is the v1 text byte for byte, then the
     continuation. Attention is causal, so the long run's first ``n_v1``
     tokens are the v1 run's up to float noise, and length is the only
     change. The builder refuses unless the long prompt's first ``n_v1``
     token ids equal v1's.
  3. **Continuation.** The source's text in order, starting at the first
     character after the passage v1 ends on, with no skipping or
     selection, formatted by v1's own convention for that prompt: prose
     paragraphs joined by one space with headings, reference markers,
     footnotes, figures and displayed equations dropped (v1's prose has no
     newlines); code and LaTeX keep their lines.
  4. **Length.** Whole units (a paragraph for prose, a line for code and
     LaTeX) are appended while the total stays at or under ``MAX_TOKENS``
     (2048, Pythia's context, no special tokens added). If the source ends
     first, the prompt ends there and the shortfall is recorded, not
     padded.
  5. **Sources**, with a revision or version recorded at fetch time:
     - `wiki_paragraph`: English Wikipedia, "Charlotte Brontë" (CC BY-SA),
       the revision id the builder fetched.
     - `sullivan_ballou`: the rest of the same letter (14 July 1861, public
       domain) from a public-domain transcription, named in the provenance
       record. The letter is short, so this prompt is expected to stop
       below ``MAX_TOKENS``.
     - `paper_excerpt`: Geshkovski, Letrouit, Polyanskiy & Rigollet, "A
       mathematical perspective on Transformers" (arXiv 2312.10794), the
       arXiv version fetched. If the excerpt is not found verbatim in that
       text, the builder refuses rather than guessing a join.
     - `hdbscan_code`: `hdbscan/plots.py` from the hdbscan package
       installed in the `mets` env (BSD), version recorded.
     - `latex_monograph`: v1's text is composed, not sourced, so the
       continuation is composed too: written once, continuing the same
       document from its last word, before any model sees it, and never
       revised after a run.
     - `repeated_tokens`: v1's `". "` pattern repeated.
  6. **Separate from the battery.** Keys are ``<v1 key>_long``. They are
     *not* added to `core.config.PROMPTS`, so `PROMPT_BATTERY_HASH` and every
     battery check stay as they are; runs record ``LONG_PROMPTS_HASH``
     instead. The v2 held-out prompts are not involved.

Runs go to ``/run/media/system/HDD_1TB/mets_data`` (user, 2026-09-29): about
3 GB of attention per run at 2048 tokens.

BUILT 2026-09-29, before any model saw the text (the tokenizer only)
-------------------------------------------------------------------
Four of the six prompts pass; `provenance.json` has token counts and source
revisions. What the rule's application decided, each recorded rather than
worked around:

- `paper_excerpt` refused (rule 5's own clause): v1's excerpt is not
  verbatim in arXiv 2312.10794v5; inline math and citations were removed by
  hand, and the continuation is dense math.
- `repeated_tokens` refused (rule 2): v1 ends on a lone space token, and
  every continuation of its `". "` pattern merges with it.
- `latex_monograph`: the same lone space; a leading newline (whitespace to
  LaTeX) keeps it a token of its own, so the composed text starts on a new
  line. Chosen from the tokenizer, before any run.
- `sullivan_ballou`: v1's transcription differs from Wikisource's in
  punctuation, so the join is the longest ending of v1 found exactly once
  in the source (its last clause, "I have obeyed."), not a fixed 60
  characters. The letter ends at 1 032 tokens (rule 4's shortfall).

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

#: The v1 prompts continued (rule 1), in battery order.
SOURCE_KEYS = ("wiki_paragraph", "repeated_tokens", "sullivan_ballou",
               "paper_excerpt", "hdbscan_code", "latex_monograph")
#: Pythia's context length; the rule's cap (rule 4).
MAX_TOKENS = 2048
#: Where the long runs are written (user, 2026-09-29).
DATA_ROOT = "/run/media/system/HDD_1TB/mets_data"


def long_key(v1_key: str) -> str:
    return f"{v1_key}_long"


# ---------------------------------------------------------------------------
# The builder (rule 3-5). Sources are fetched once into ``<DATA_ROOT>/
# long_prompts_sources/``; the built texts and their provenance are
# committed under ``long_prompts/`` beside this file.
# ---------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent / "long_prompts"
UA = {"User-Agent": "Mets-research/1.0 (metastability study; Pythia)"}
WIKI_API = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts|revisions"
            "&rvprop=ids|timestamp&explaintext=1&exsectionformat=wiki"
            "&titles=Charlotte_Bront%C3%AB&format=json&formatversion=2")
BALLOU_API = ("https://en.wikisource.org/w/api.php?action=parse"
              "&page=Letter_from_Captain_Sullivan_Ballou_to_His_Wife_Sarah"
              "&prop=wikitext|revid&format=json&formatversion=2")
#: Rule 5's clause for `paper_excerpt`: v1's excerpt is not verbatim in
#: arXiv 2312.10794v5 (inline math, citations and section references were
#: removed by hand), so the builder refuses it. Recorded, not retried.
REFUSED = {"paper_excerpt": "v1's excerpt is not verbatim in arXiv 2312.10794v5 "
                            "(inline math and citations removed by hand); rule 5 refuses",
           "repeated_tokens": "v1 ends in a lone space token; every continuation of its "
                              "'. ' pattern merges with it, so rule 2 refuses"}
#: Shortest v1 ending accepted as the join point (see `_after`).
MIN_JOIN_CHARS = 12


def _get(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def fetch_sources(cache: Path) -> Dict:
    """Fetch each source once into ``cache``; returns their provenance."""
    import hdbscan
    import importlib.metadata as md
    cache.mkdir(parents=True, exist_ok=True)
    prov = {}
    for name, url in (("wiki_paragraph", WIKI_API), ("sullivan_ballou", BALLOU_API)):
        f = cache / f"{name}.json"
        if not f.exists():
            f.write_bytes(_get(url))
        prov[name] = {"url": url, "file": str(f)}
    d = json.loads((cache / "wiki_paragraph.json").read_text())["query"]["pages"][0]
    prov["wiki_paragraph"]["revision"] = d["revisions"][0]
    prov["sullivan_ballou"]["revision"] = json.loads(
        (cache / "sullivan_ballou.json").read_text())["parse"]["revid"]
    prov["hdbscan_code"] = {"file": str(Path(hdbscan.__file__).with_name("plots.py")),
                            "version": md.version("hdbscan")}
    prov["latex_monograph"] = {"file": str(HERE / "latex_monograph_continuation.tex"),
                               "note": "composed (rule 5)"}
    prov["repeated_tokens"] = {"note": "v1's '. ' unit repeated"}
    return prov


def _after(src: str, v1: str, key: str) -> str:
    """
    The source after the passage v1 ends on: the longest ending of v1 that
    occurs exactly once in the source (at least ``MIN_JOIN_CHARS``) marks the
    join. v1's transcription of the Ballou letter differs in punctuation, so
    only its last clause matches; the join is still unique. Refuses otherwise.
    """
    body = v1.rstrip("\n")
    for n in range(min(len(body), 400), MIN_JOIN_CHARS - 1, -1):
        tail = body[-n:]
        i = src.find(tail)
        if i >= 0 and src.find(tail, i + 1) < 0:
            rest = src[i + n:]
            return rest[1:] if v1.endswith("\n") and rest.startswith("\n") else rest
    raise ValueError(f"{key}: no ending of v1 of {MIN_JOIN_CHARS}+ characters occurs exactly "
                     "once in the source; refusing to guess a join")


def _units(key: str, v1: str, cache: Path) -> Tuple[str, List[str]]:
    """(joiner, continuation units in order) for one prompt (rule 3)."""
    if key == "wiki_paragraph":
        t = json.loads((cache / "wiki_paragraph.json").read_text())["query"]["pages"][0]["extract"]
        rest = _after(t, v1, key)
        first, _, more = rest.partition("\n")
        paras = [first.strip()] + [ln.strip() for ln in more.split("\n")]
        return " ", [p for p in paras if p and not p.startswith("==")]
    if key == "sullivan_ballou":
        t = json.loads((cache / "sullivan_ballou.json").read_text())["parse"]["wikitext"]
        rest = _after(t, v1, key)
        paras = [p.strip() for p in rest.split("\n\n")]
        return " ", [p for p in paras if p and not p.startswith("{{")]
    if key == "hdbscan_code":
        import hdbscan
        src = Path(hdbscan.__file__).with_name("plots.py").read_text()
        return "", [ln + "\n" for ln in _after(src, v1, key).split("\n")]
    if key == "latex_monograph":
        # v1 ends on a lone space token; a leading newline (whitespace to
        # LaTeX) is the one join that leaves it a token of its own (rule 2).
        lines = (HERE / "latex_monograph_continuation.tex").read_text().rstrip("\n").split("\n")
        return "", ["\n" + lines[0] + "\n"] + [ln + "\n" for ln in lines[1:]]
    if key == "repeated_tokens":
        return "", [". "] * 4000
    raise KeyError(key)


def build_one(key: str, v1: str, cache: Path, tok: Callable[[str], List[int]]) -> Dict:
    """Append whole units while the prompt fits MAX_TOKENS (rule 4); check the prefix (rule 2)."""
    joiner, units = _units(key, v1, cache)
    ids_v1 = tok(v1)
    text, used = v1, 0
    for u in units:
        cand = text + joiner + u
        if len(tok(cand)) > MAX_TOKENS:
            break
        text, used = cand, used + 1
    ids = tok(text)
    if ids[:len(ids_v1)] != ids_v1:
        k = next(i for i, (a, b) in enumerate(zip(ids, ids_v1)) if a != b)
        raise ValueError(f"{key}: the long prompt's token {k} differs from v1's; "
                         "the prefix does not reproduce v1 (rule 2)")
    return {"key": long_key(key), "text": text, "n_tokens": len(ids), "n_v1_tokens": len(ids_v1),
            "units_used": used, "units_available": len(units),
            "source_exhausted": used == len(units)}


def build_all(cache: Path, out: Path = HERE) -> Dict:
    """Build every non-refused prompt, write ``<key>.txt`` and ``provenance.json``."""
    from transformers import AutoTokenizer
    from core.config import PROMPTS
    t = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    prov = fetch_sources(cache)
    rec = {"rule_commit": "ca787de", "max_tokens": MAX_TOKENS, "tokenizer": "EleutherAI/pythia-410m",
           "refused": REFUSED, "prompts": {}}
    for k in SOURCE_KEYS:
        if k in REFUSED:
            continue
        r = build_one(k, PROMPTS[k], cache, lambda s: t(s)["input_ids"])
        (out / f"{r['key']}.txt").write_text(r.pop("text"))
        rec["prompts"][r["key"]] = {**r, "v1_key": k, "source": prov[k]}
    (out / "provenance.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def load_provenance() -> Dict:
    """``provenance.json`` as built: counts and sources per long key."""
    return json.loads((HERE / "provenance.json").read_text())


def load() -> Dict[str, str]:
    """The built long prompts, ``{key: text}``."""
    return {k: (HERE / f"{k}.txt").read_text() for k in load_provenance()["prompts"]}


def long_prompts_hash() -> str:
    """Short hash of the built texts, recorded in every long run's manifest (rule 6)."""
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(load().items()):
        h.update(k.encode() + b"\0" + v.encode() + b"\0")
    return h.hexdigest()[:12]
