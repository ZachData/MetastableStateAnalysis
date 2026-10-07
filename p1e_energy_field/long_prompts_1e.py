"""
p1e_energy_field/long_prompts_1e.py — four new long passages, so Phase 1e
reads 8 long passages instead of 1d's 4 (`design-1e.md` "Inputs, fixed
here"; the user's direction of 2026-10-06: move towards only the larger
prompts, and add passages now).

THE RULE — fixed 2026-10-06, BEFORE any source text was fetched
---------------------------------------------------------------
This docstring is committed on its own, ahead of the commit that adds the
text, as `p1d_cluster_ensemble/long_prompts.py`'s rule was, so git shows the
rule predates the text. Nothing below is chosen after running a model on it.

  1. **Which passages** (the user chose these four, 2026-10-06). Each
     replaces a genre 1d's long set lost, or adds a structure v1 lacks:

     | key | source | stands in for |
     |---|---|---|
     | `odyssey_butler_long` | Homer, *The Odyssey*, Samuel Butler's 1900 prose translation, Project Gutenberg eBook #1727; **Book VII** from its first body paragraph | `homer_iliad` (v1's passage, despite its key, appears to be Odysseus's account in *Odyssey* 7, in Fagles's translation, still in copyright) |
     | `horla_long` | Guy de Maupassant, *Le Horla*, the 1887 version (the journal, opening "8 mai"), French, from fr.wikisource | `camus_letranger` (first-person French narrative) |
     | `darwin_origin_long` | Charles Darwin, *On the Origin of Species*, 1st edition (1859), Project Gutenberg eBook #1228; **Chapter IV, "Natural Selection"**, from its first body paragraph | `paper_excerpt` (expository argument) |
     | `hamlet_long` | William Shakespeare, *Hamlet*, Project Gutenberg eBook #1524; **Act I, Scene I** from its first line | none: dialogue with speaker tags, a structure v1 does not have |

     None is a v2 held-out source (`core/holdout.py`: not Moby-Dick, the
     Quijote, the Lincoln letters, the two Wikipedia articles, the two
     papers, the sklearn / scipy code or the two LaTeX files), and no key
     carries a held-out key.
  2. **Start.** The first body text after the named heading. Headings, the
     chapter's or book's summary line, and Gutenberg's header and licence
     are not text. The builder refuses if the heading is not found exactly
     once, or the fetched file's title / author (and, for #1727, the
     translator) is not the one named above.
  3. **Continuation.** The source's text in order from the start, with no
     skipping or selection, through the next book / chapter / scene if the
     named one ends first (its heading dropped as in rule 2). Formatting by
     v1's own convention: prose (`odyssey_butler_long`, `horla_long`,
     `darwin_origin_long`) is paragraphs joined by one space, a paragraph's
     own line breaks turned into spaces, footnote markers and footnotes
     dropped; the journal's dates stay, as the text's own words. Drama
     (`hamlet_long`) keeps its lines, as code and LaTeX do in 1d's set:
     speaker tags and stage directions as the edition prints them, with
     leading indentation stripped.
  4. **Length.** Whole units (a paragraph for prose, a line for drama) are
     appended while the total stays at or under ``MAX_TOKENS`` (2048,
     Pythia's context, no special tokens added). A source that ends first
     is recorded as a shortfall, not padded.
  5. **Provenance.** Each fetched file is cached under
     ``DATA_ROOT/long_prompts_sources_1e/``, with its URL, sha256, and the
     Gutenberg "Release date" / "Most recently updated" lines or the
     Wikisource revision id, in ``long_prompts_1e/provenance.json``.
  6. **Separate from the battery.** Keys are not added to
     `core.config.PROMPTS`, so `PROMPT_BATTERY_HASH` stays as it is. The 8
     long passages (1d's 4 and these 4) carry their own hash,
     ``LONG8_HASH``, which every 1e run records.
  7. **Order.** Built with the tokenizer only; the texts are committed
     before any model sees them, and never revised after a run.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Tuple

MAX_TOKENS = 2048
DATA_ROOT = "/run/media/system/HDD_1TB/mets_data"
CACHE = Path(DATA_ROOT) / "long_prompts_sources_1e"
HERE = Path(__file__).resolve().parent / "long_prompts_1e"
UA = {"User-Agent": "Mets-research/1.0 (metastability study; Pythia)"}
RULE_COMMIT = "6de234e"

GUTENBERG = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
#: key: (eBook number, header lines that must be present, start heading lines, rule 2)
PG = {
    "odyssey_butler_long": (1727, ("Title: The Odyssey", "Author: Homer", "Translator: Samuel Butler"),
                            ("BOOK VII", "RECEPTION OF ULYSSES AT THE PALACE OF KING ALCINOUS.")),
    "darwin_origin_long": (1228, ("Title: On the Origin of Species By Means of Natural Selection",
                                  "Author: Charles Darwin"),
                           ("CHAPTER IV.", "NATURAL SELECTION.")),
    "hamlet_long": (1524, ("Title: Hamlet", "Author: William Shakespeare"),
                    ("SCENE I. Elsinore. A platform before the Castle.",)),
}
HORLA_PAGE = "Le_Horla_(recueil,_Ollendorff_1895)/Le_Horla"
HORLA_API = ("https://fr.wikisource.org/w/api.php?action=parse&page=" + HORLA_PAGE
             + "&prop=text|revid&format=json&formatversion=2")
KEYS = ("odyssey_butler_long", "horla_long", "darwin_origin_long", "hamlet_long")
#: Butler's footnote markers are digits glued to the word or stop before them ("god.57").
FOOTNOTE = re.compile(r"(?<=[^\s\d])\d{1,3}(?=[\s’”)\]]|$)")
HEADING = re.compile(r"^(BOOK [IVXL]+|CHAPTER [IVXL]+\.|ACT [IVX]+|SCENE [IVX]+\..*)$")


def _get(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def fetch_sources(cache: Path = CACHE) -> Dict:
    """Fetch (once) and describe every source (rule 5)."""
    cache.mkdir(parents=True, exist_ok=True)
    prov = {}
    for key, (n, _, _) in PG.items():
        f, url = cache / f"pg{n}.txt", GUTENBERG.format(n=n)
        if not f.exists():
            f.write_bytes(_get(url))
        raw = f.read_text(encoding="utf-8")
        dates = [ln.strip() for ln in raw[:3000].splitlines()
                 if ln.strip().startswith(("Release date:", "Most recently updated:"))]
        prov[key] = {"url": url, "file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                     "gutenberg": dates}
    f = cache / "horla.json"
    if not f.exists():
        f.write_bytes(_get(HORLA_API))
    prov["horla_long"] = {"url": HORLA_API, "file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                          "revision": json.loads(f.read_text())["parse"]["revid"],
                          "note": "fr.wikisource, validated; the 1887 text in Ollendorff's 1895 printing"}
    return prov


class _Paras(HTMLParser):
    """The text of every ``<p>`` in a Wikisource page, tags dropped."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paras, self._cur = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._cur = []

    def handle_endtag(self, tag):
        if tag == "p" and self._cur is not None:
            self.paras.append("".join(self._cur))
            self._cur = None

    def handle_data(self, data):
        if self._cur is not None:
            self._cur.append(data)


def _gutenberg_units(key: str, cache: Path) -> List[str]:
    n, header, heading = PG[key]
    raw = (cache / f"pg{n}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    for h in header:
        if h not in raw[:3000]:
            raise ValueError(f"{key}: pg{n}.txt's header lacks {h!r}; refusing (rule 2)")
    lines = raw.split("\n")
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    at = [nonblank[j] for j in range(len(nonblank) - len(heading) + 1)
          if [lines[nonblank[j + k]].strip() for k in range(len(heading))] == list(heading)]
    if len(at) != 1:
        raise ValueError(f"{key}: start heading {heading[0]!r} found {len(at)} times; refusing (rule 2)")
    i = at[0]
    seen = 0
    while seen < len(heading):          # skip the heading's own lines
        seen += bool(lines[i].strip())
        i += 1
    body = "\n".join(lines[i:]).split("*** END OF THE PROJECT GUTENBERG")[0]
    if key == "hamlet_long":
        out = [ln.strip() + "\n" for ln in body.split("\n")]
        while out and out[0] == "\n":
            out.pop(0)
        return [u for u in out if not HEADING.match(u.strip())]
    paras = [" ".join(ln.strip() for ln in p.split("\n") if ln.strip()) for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if p]
    if key == "darwin_origin_long":
        paras = paras[1:]               # the chapter's summary paragraph (rule 2)
    if key == "odyssey_butler_long":
        paras = [FOOTNOTE.sub("", p) for p in paras]
    # a later book's / chapter's heading and its summary line are not text (rules 2-3)
    out, skip = [], 0
    for p in paras:
        if HEADING.match(p):
            skip = 1
            continue
        if skip:
            skip -= 1
            continue
        out.append(p)
    return out


def _horla_units(cache: Path) -> List[str]:
    d = json.loads((cache / "horla.json").read_text())["parse"]["text"]
    p = _Paras()
    p.feed(d)
    # whitespace runs (a page break's span leaves several) as the page renders them: one space
    paras = [re.sub(r"[ \t\n]+", " ", x).strip() for x in p.paras]
    paras = [x for x in paras if x]
    starts = [i for i, x in enumerate(paras) if x.startswith("8 mai.")]
    if len(starts) != 1:
        raise ValueError(f"horla_long: '8 mai.' opens {len(starts)} paragraphs; refusing (rule 2)")
    return paras[starts[0]:]


def units(key: str, cache: Path = CACHE) -> Tuple[str, List[str]]:
    """(joiner, units in order) for one passage (rule 3)."""
    if key == "horla_long":
        return " ", _horla_units(cache)
    u = _gutenberg_units(key, cache)
    return ("", u) if key == "hamlet_long" else (" ", u)


def build_one(key: str, tok: Callable[[str], List[int]], cache: Path = CACHE) -> Dict:
    """Whole units while the passage fits MAX_TOKENS (rule 4)."""
    joiner, us = units(key, cache)
    text, used = "", 0
    for u in us:
        cand = (text + joiner + u) if text else u
        if len(tok(cand)) > MAX_TOKENS:
            break
        text, used = cand, used + 1
    if key == "hamlet_long":
        text = text.rstrip("\n")
    return {"key": key, "text": text, "n_tokens": len(tok(text)), "units_used": used,
            "units_available": len(us), "source_exhausted": used == len(us)}


def build_all(cache: Path = CACHE, out: Path = HERE) -> Dict:
    """Build the four passages; write ``<key>.txt`` and ``provenance.json``."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    prov = fetch_sources(cache)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"rule_commit": RULE_COMMIT, "max_tokens": MAX_TOKENS, "tokenizer": "EleutherAI/pythia-410m",
           "prompts": {}}
    for k in KEYS:
        r = build_one(k, lambda s: t(s)["input_ids"], cache)
        (out / f"{k}.txt").write_text(r.pop("text"))
        rec["prompts"][k] = {**r, "source": prov[k]}
    (out / "provenance.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return rec


def load_provenance() -> Dict:
    return json.loads((HERE / "provenance.json").read_text())


def load() -> Dict[str, str]:
    """The four built passages, ``{key: text}``."""
    return {k: (HERE / f"{k}.txt").read_text() for k in load_provenance()["prompts"]}


#: 1d's four long passages, read by path so this module needs nothing beyond the stdlib.
P1D_HERE = Path(__file__).resolve().parents[1] / "p1d_cluster_ensemble" / "long_prompts"
#: The 8 texts' hash, pinned (rule 6; `/challenge-pr` on #157, finding 2). Every run of the
#: 2026-10-06 batch records it; `load8` refuses texts that do not hash to it.
LONG8_HASH = "ba605f4e14b5"


def _provenances() -> List[Dict]:
    return [json.loads((d / "provenance.json").read_text()) for d in (P1D_HERE, HERE)]


def _texts8() -> Dict[str, str]:
    out = {k: (d / f"{k}.txt").read_text()
           for d, prov in zip((P1D_HERE, HERE), _provenances()) for k in prov["prompts"]}
    if len(out) != 8:
        raise ValueError(f"expected 8 long passages, have {sorted(out)}")
    return out


def long8_hash() -> str:
    """Short hash of the 8 texts on disk (compare with ``LONG8_HASH``)."""
    h = hashlib.sha256()
    for k, v in sorted(_texts8().items()):
        h.update(k.encode() + b"\0" + v.encode() + b"\0")
    return h.hexdigest()[:12]


def load8() -> Dict[str, str]:
    """1e's 8 long passages: 1d's 4 and these 4 (`design-1e.md` "Inputs"); refuses a changed text."""
    got = long8_hash()
    if got != LONG8_HASH:
        raise ValueError(f"the 8 long passages hash to {got}, not the pinned {LONG8_HASH}: a text "
                         "changed after the runs (rule 7); refusing")
    return _texts8()


def expected_tokens() -> Dict[str, int]:
    """Token count per long passage, as each provenance file recorded it."""
    return {k: v["n_tokens"] for prov in _provenances() for k, v in prov["prompts"].items()}


if __name__ == "__main__":
    r = build_all()
    for k, v in r["prompts"].items():
        print(k, v["n_tokens"], f"{v['units_used']}/{v['units_available']}", v["source_exhausted"])
    print("LONG8_HASH", long8_hash())
