"""
trackC_writeup/to_tex.py — the LaTeX draft, converted from the rendered Markdown (C2p unit 1,
2026-10-24; `docs/ROADMAP.md` C2p "the LaTeX conversion").

`draft-C2.md` (itself rendered from `draft-C2.src.md` and `numbers.json` by `numbers.py`) is
converted to `tex/main.tex` and `tex/refs.bib`, an Overleaf-ready folder with the figures' PDFs
that `numbers.py figures` writes into `tex/fig/`. Nothing here reads a record: every number in
the LaTeX is the Markdown's, so the render step's checks cover both.

The conversion targets pdfLaTeX (Overleaf's default): the draft's Unicode mathematics (E₁,
‖m‖², 10⁻⁴, ⟨xᵢ, xⱼ⟩, …) becomes math mode, citations become natbib author–year keys, and
§N, figure N, table 1 and appendix X become \\ref's. It refuses rather than guess: a display
equation it has no LaTeX for, a citation with no reference, a figure reference with no figure,
or a character pdfLaTeX cannot typeset.

    python -m trackC_writeup.to_tex build | check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DRAFT = HERE / "draft-C2.md"
TEXDIR = HERE / "tex"
MAIN = TEXDIR / "main.tex"
BIB = TEXDIR / "refs.bib"


class Refused(SystemExit):
    pass


#: The draft's two display equations, set by hand; a third, or an edited one, refuses.
DISPLAY = {
    "E_β(X) = (1 / 2βn²) Σᵢ Σⱼ exp(β ⟨xᵢ, xⱼ⟩).":
        r"E_\beta(X) = \frac{1}{2\beta n^2}\sum_{i}\sum_{j}\exp\bigl(\beta\,\langle x_i, x_j\rangle\bigr).",
    "E₁ = ½ + ½ ‖m‖² + Σ_{k≥2} (1 / 2n² k!) Σᵢ Σⱼ ⟨xᵢ, xⱼ⟩ᵏ,":
        r"E_1 = \tfrac12 + \tfrac12\,\|m\|^2 + \sum_{k\ge 2}\frac{1}{2n^2\,k!}\sum_i\sum_j \langle x_i, x_j\rangle^k,",
}

_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉ᵢⱼₖ.", "0123456789ijk."))
_SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵏᵀ", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "k", r"\top"]))
#: Single characters that are mathematics, in math mode.
_MATHCHAR = {
    "β": r"\beta", "ρ": r"\rho", "κ": r"\kappa", "α": r"\alpha", "Δ": r"\Delta", "Σ": r"\sum",
    "δ": r"\delta", "τ": r"\tau", "≥": r"\geq", "≤": r"\leq", "×": r"\times", "−": "-", "∈": r"\in",
    "½": r"\tfrac12", "·": r"\cdot", "′": "'", "√": r"\surd", "ẋ": r"\dot{x}",
}
#: Text characters pdfLaTeX (utf8, T1) sets directly, and those it is given as commands.
_TEXTCHAR = {"§": r"\S{}", "…": r"\dots{}", "–": "--", "—": "---", "“": "``", "”": "''", "’": "'", "‘": "`"}
_SAFE = set("éëÉèàöüäçñ")


def _math_inner(s: str) -> str:
    """A math fragment's Unicode, as LaTeX (inside $…$)."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        w = re.match(r"[A-Za-z]{2,}(?=[₀-₉ᵢⱼₖ])", s[i:]) if i == 0 else None
        if w:                                           # softmaxⱼ: a word with a subscript
            out.append(r"\operatorname{" + w.group(0) + "}")
            i += len(w.group(0))
            continue
        if c in _SUB:                                   # a run of subscripts
            j = i
            while j < len(s) and s[j] in _SUB and not (s[j] == "." and (j + 1 >= len(s) or s[j + 1] not in _SUB)):
                j += 1
            out.append("_{" + "".join(_SUB[ch] for ch in s[i:j]) + "}")
            i = j
            continue
        if c in _SUP:
            j = i
            while j < len(s) and s[j] in _SUP:
                j += 1
            out.append("^{" + "".join(_SUP[ch] for ch in s[i:j]) + "}")
            i = j
            continue
        if c == "‖":
            out.append(r"\|")
        elif c == "⟨":
            out.append(r"\langle ")
        elif c == "⟩":
            out.append(r"\rangle ")
        elif c in _MATHCHAR:
            m = _MATHCHAR[c]
            out.append(m + (" " if m[0] == "\\" and m[-1].isalpha() else ""))
        else:
            out.append(c)
        i += 1
    return "".join(out)


#: A math fragment: an identifier with sub/superscripts, a norm, a bracket, or a math character.
_MATH = re.compile(
    r"‖[^‖\s][^‖]{0,30}‖[⁰-⁹²³¹⁻ᵏᵀ]*"                                  # ‖m‖², ‖W − W₀‖
    r"|⟨[^⟩]{1,30}⟩[⁰-⁹²³¹⁻ᵏᵀ]*"                                      # ⟨xᵢ, xⱼ⟩ᵏ
    r"|(?<![A-Za-z])[A-Za-zΣΔβδ](?:_\{[^}]{1,20}\}|_[A-Za-zβ0-9]|\^\{[^}]{1,20}\})+"  # E_β, S^{d−1}, Σ_{k≥2}
    r"|(?<![A-Za-z0-9.])(?:[A-Za-z]+(?=[₀-₉ᵢⱼₖ])|\d+(?:\.\d+)?|[A-Za-zΔΣβδẋ)])(?:[₀-₉ᵢⱼₖ]+(?:\.[₀-₉]+)?|[⁰-⁹²³¹⁻ᵏᵀ]+)+"  # E₁, E₃.₅, 10⁻⁴, softmaxⱼ
    r"|[βρκαΔΣδτ≥≤×−∈½·′√]"
)
_OP_MERGE = re.compile(r"\$([^$]+)\$(\s*[=+<>/,]\s*|\s*[=+<>]\s*[A-Za-z0-9.]{1,3}\s*[=+<>]\s*|)\$([^$]+)\$")
_REL = r"(?:\\(?:in|geq|leq|times|cdot) ?|[=<>]|\\sum _\{[^}]*\})"
#: "$x$ $\in$ $S$" -> "$x \in S$": a lone relation merges with its neighbours across a space.
_REL_MERGE = (re.compile(r"\$([^$]+)\$\s+\$(" + _REL + r")\$"), re.compile(r"\$([^$]*" + _REL + r")\$\s+\$([^$]+)\$"))


def _protect(text: str, store: List[str], latex: str) -> str:
    store.append(latex)
    return f"\x00{len(store) - 1}\x00"


def _escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = re.sub(r"([&%#_{}$])", r"\\\1", s)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


class Converter:
    def __init__(self, md: str):
        self.text = re.sub(r"\A<!--.*?-->\n", "", md, count=1, flags=re.S)
        self.bib: Dict[str, dict] = {}            # key -> entry
        self.by_cite: Dict[Tuple[str, str], str] = {}
        self.figs: Dict[str, str] = {}            # "3" -> label
        self.secs: Dict[str, str] = {}            # "4.1" / "A" -> label
        self.last_cite: Optional[str] = None
        self._scan()

    # -- the references and the labels, first, so the body can point forward ------------------
    def _scan(self) -> None:
        refs = self.text.split("\n## References", 1)
        if len(refs) != 2:
            raise Refused("refusing: the draft has no '## References' section")
        for item in re.findall(r"^- (.+(?:\n  .+)*)", refs[1], re.M):
            self._bib_entry(" ".join(item.split()))
        for m in re.finditer(r"^!\[Figure (\d+)\]\(fig/([\w-]+)\.svg\)", self.text, re.M):
            self.figs[m.group(1)] = "fig:" + m.group(2)
        for m in re.finditer(r"^#{2,3} (?:(\d+(?:\.\d+)?)\.? |Appendix ([A-Z])\. )", self.text, re.M):
            n = m.group(1) or m.group(2)
            self.secs[n] = "sec:" + n

    def _bib_entry(self, item: str) -> None:
        m = re.match(r"(.+?) \((\d{4})\)\. (.+?)(\.|\?) arXiv:(\d{4}\.\d{4,5})\.?\s*$", item)
        if m:
            authors, year, title, end, eprint = m.groups()
            title += "?" if end == "?" else ""
            toks = [t.strip() for t in authors.split(", ")]
            if len(toks) % 2:
                raise Refused(f"refusing: cannot pair authors in {authors!r}")
            names = [f"{toks[i]}, {toks[i + 1]}" for i in range(0, len(toks), 2)]
            first = toks[0]
            key = re.sub(r"[^a-z]", "", first.lower()) + year
            if key in self.bib:
                raise Refused(f"refusing: two references key to {key}")
            self.bib[key] = {"type": "misc", "author": " and ".join(names), "title": title, "year": year,
                             "eprint": eprint, "archivePrefix": "arXiv",
                             "howpublished": f"\\href{{https://arxiv.org/abs/{eprint}}}{{arXiv:{eprint}}}"}
            self.by_cite[(first.split()[-1] if first.startswith("van ") else first, year)] = key
            self.by_cite[(first, year)] = key
            return
        m = re.match(r"Stanford CRFM\. (.+)$", item)
        if m:
            self.bib["crfm"] = {"type": "misc", "author": "{Stanford CRFM}", "title": "Mistral {GPT-2} small checkpoints",
                                "howpublished": m.group(1).rstrip(".")}
            self.by_cite[("Stanford CRFM", "")] = "crfm"
            return
        raise Refused(f"refusing: a reference this converter cannot read: {item[:80]!r}")

    def _key(self, surname: str, year: str) -> str:
        k = self.by_cite.get((surname, year))
        if k is None:
            raise Refused(f"refusing: no reference for {surname} {year}")
        return k

    # -- inline text ----------------------------------------------------------------------------
    def inline(self, s: str) -> str:
        store: List[str] = []
        s = re.sub(r"`([^`]+)`", lambda m: _protect(s, store, r"\texttt{" + _escape(m.group(1)) + "}"), s)
        s = self._cites(s, store)
        s = self._refs(s, store)
        s = _MATH.sub(lambda m: _protect(s, store, "$" + _math_inner(m.group(0)) + "$"), s)
        s = _escape(s)
        for c, t in _TEXTCHAR.items():
            s = s.replace(c, t)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
        s = s.replace('"', "\x01")
        s = re.sub(r"\x01([^\x01]*)\x01", r"``\1''", s)
        while "\x00" in s:
            s = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], s)
        while True:
            t = _OP_MERGE.sub(lambda m: "$" + m.group(1) + m.group(2) + m.group(3) + "$", s)
            for r in _REL_MERGE:
                t = r.sub(lambda m: "$" + m.group(1).rstrip() + " " + m.group(2) + "$", t)
            if t == s:
                break
            s = t
        return s

    def _cites(self, s: str, store: List[str]) -> str:
        def note(n: Optional[str]) -> str:
            return "" if not n else "[" + _escape(n.strip()).replace("§", r"\S{}").replace("–", "--") + "]"

        def paren(m: re.Match) -> str:
            body = m.group(1)
            if body.startswith("ibid"):
                if self.last_cite is None:
                    raise Refused("refusing: 'ibid.' with no citation before it")
                return _protect(s, store, r"\citep" + note(body.partition(",")[2]) + "{" + self.last_cite + "}")
            if body == "Stanford CRFM":
                self.last_cite = "crfm"
                return _protect(s, store, r"\citep{crfm}")
            keys, notes = [], None
            for part in body.split("; "):
                mm = re.match(r"(.+?)(?: et al\.| and [^\d]+?)? (\d{4})(?:, (.+))?$", part)
                if not mm:
                    raise Refused(f"refusing: a citation this converter cannot read: [{body}]")
                keys.append(self._key(mm.group(1), mm.group(2)))
                notes = mm.group(3) or notes
            self.last_cite = keys[-1]
            return _protect(s, store, r"\citep" + note(notes) + "{" + ",".join(keys) + "}")

        def textual(m: re.Match) -> str:
            name, years = m.group(1), m.group(3)
            keys = [self._key(name, y.strip()) for y in years.split(",")]
            self.last_cite = keys[-1]
            return _protect(s, store, r"\citet{" + ",".join(keys) + "}")

        s = re.sub(r"([A-Z][\w-]+)( et al\.| and (?:van der )?[A-Z][\w-]+)? \[((?:\d{4}, )*\d{4})\]", textual, s)
        return re.sub(r"\[((?:ibid\.|Stanford CRFM|(?:van der )?[A-Z][^\]]*?\d{4})[^\]]*)\]", paren, s)

    def _refs(self, s: str, store: List[str]) -> str:
        def sec(m: re.Match) -> str:
            n = m.group(1)
            if n not in self.secs:
                raise Refused(f"refusing: §{n} names no section of the draft")
            return _protect(s, store, r"\S\ref{" + self.secs[n] + "}")

        def fig(m: re.Match) -> str:
            word, a, sep, b = m.group(1), m.group(2), m.group(3), m.group(4)
            for n in filter(None, (a, b)):
                if n not in self.figs:
                    raise Refused(f"refusing: figure {n} is not in the draft")
            r = r"~\ref{" + self.figs[a] + "}"
            if b:
                r += ("--" if sep == "–" else " and ") + r"\ref{" + self.figs[b] + "}"
            return _protect(s, store, word + r)

        def app(m: re.Match) -> str:
            return _protect(s, store, m.group(1) + r"~\ref{" + self.secs[m.group(2)] + "}"
                            + ("" if not m.group(3) else r" and~\ref{" + self.secs[m.group(3)] + "}"))

        s = re.sub(r"§(\d+(?:\.\d+)?)(?!\.\d)", sec, s)
        s = re.sub(r"\b([Ff]igures?)\s(\d+)(?:(–| and )(\d+))?\b", fig, s)
        s = re.sub(r"\b([Tt]able)\s1\b", lambda m: _protect(s, store, m.group(1) + r"~\ref{tab:rows}"), s)
        return re.sub(r"\b([Aa]ppendix|[Aa]ppendices)\s([A-C])(?: and ([A-C]))?\b", app, s)

    # -- blocks ---------------------------------------------------------------------------------
    def blocks(self) -> List[str]:
        return [b for b in re.split(r"\n\s*\n", self.text.split("\n## References", 1)[0]) if b.strip()]

    def convert(self) -> str:
        out: List[str] = []
        title = None
        appendix = False
        pending_fig: Optional[Tuple[str, str]] = None
        pending_table_caption: Optional[str] = None
        for b in self.blocks():
            lines = b.split("\n")
            first = lines[0]
            if first.startswith("# "):
                title = self.inline(first[2:])
                continue
            if first.startswith("## ") or first.startswith("### "):
                level = "section" if first.startswith("## ") else "subsection"
                head = first.split(" ", 1)[1]
                m = re.match(r"(?:(\d+(?:\.\d+)?)\.? |Appendix ([A-Z])\. )?(.*)$", head)
                num, letter, text = m.groups()
                if letter and not appendix:
                    out.append(r"\appendix")
                    appendix = True
                lab = f"\\label{{sec:{num or letter}}}" if (num or letter) else ""
                out.append(f"\\{level}{{{self.inline(text)}}}{lab}")
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    out.append(self.para(rest))
                continue
            if first.startswith("**Abstract.**"):
                out.append("\\begin{abstract}\n" + self.inline(" ".join(lines)[len("**Abstract.**"):].strip())
                           + "\n\\end{abstract}")
                continue
            m = re.match(r"!\[Figure (\d+)\]\(fig/([\w-]+)\.svg\)$", first)
            if m:
                pending_fig = (m.group(1), m.group(2))
                continue
            if pending_fig and first.startswith(f"*Figure {pending_fig[0]}."):
                cap = " ".join(lines).strip()
                cap = re.sub(rf"^\*Figure {pending_fig[0]}\.\s*", "", cap).rstrip("*").strip()
                out.append("\\begin{figure}[tbp]\n\\centering\n"
                           f"\\includegraphics[width=\\linewidth]{{fig/{pending_fig[1]}.pdf}}\n"
                           f"\\caption{{{self.inline(cap)}}}\\label{{fig:{pending_fig[1]}}}\n\\end{{figure}}")
                pending_fig = None
                continue
            if pending_fig:
                raise Refused(f"refusing: figure {pending_fig[0]} has no caption right after it")
            if first.startswith("*Table 1."):
                pending_table_caption = re.sub(r"^\*Table 1\.\s*", "", " ".join(lines)).rstrip("*").strip()
                continue
            if first.startswith("|"):
                out.append(self.table(lines, pending_table_caption))
                pending_table_caption = None
                continue
            if first.startswith("  ") and not first.startswith("  -"):
                key = " ".join(l.strip() for l in lines)
                if key not in DISPLAY:
                    raise Refused(f"refusing: a display equation with no LaTeX in DISPLAY: {key!r}")
                out.append("\\begin{equation}\n" + DISPLAY[key] + "\n\\end{equation}")
                continue
            if re.match(r"(- |\d+\. )", first):
                out.append(self.lst(lines))
                continue
            out.append(self.para(b))
        if pending_fig or pending_table_caption:
            raise Refused("refusing: a figure or table caption with nothing after it")
        return self.document(title or "", "\n\n".join(out))

    def para(self, b: str) -> str:
        return self.inline(" ".join(l.strip() for l in b.split("\n")))

    def lst(self, lines: List[str]) -> str:
        env = "itemize" if lines[0].startswith("- ") else "enumerate"
        items: List[str] = []
        for l in lines:
            m = re.match(r"(- |\d+\. )(.*)$", l)
            if m:
                items.append(m.group(2))
            else:
                items[-1] += " " + l.strip()
        return f"\\begin{{{env}}}\n" + "\n".join("  \\item " + self.inline(i) for i in items) + f"\n\\end{{{env}}}"

    def table(self, lines: List[str], caption: Optional[str]) -> str:
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[-|\s]+\|$", l)]
        head, body = rows[0], rows[1:]
        spec = "@{}l X l@{}" if len(head) == 3 else "@{}" + "X" * len(head) + "@{}"
        tex = ["\\begin{table}[tbp]", "\\centering\\small"]
        if caption:
            tex.append(f"\\caption{{{self.inline(caption)}}}\\label{{tab:rows}}")
        tex += [f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}", "\\toprule",
                " & ".join(self.inline(c) for c in head) + r" \\", "\\midrule"]
        tex += [" & ".join(self.inline(c) for c in r) + r" \\" for r in body]
        tex += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
        return "\n".join(tex)

    def document(self, title: str, body: str) -> str:
        return "\n".join([
            "% main.tex: converted from trackC_writeup/draft-C2.md by `python -m trackC_writeup.to_tex build`.",
            "% Upload this folder (main.tex, refs.bib, fig/) to Overleaf; it compiles with pdfLaTeX.",
            r"\documentclass[11pt]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{lmodern}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs,tabularx}",
            r"\usepackage{caption}",
            r"\captionsetup[table]{position=top}",
            r"\usepackage[section]{placeins}",
            r"\usepackage[round,sort]{natbib}",
            r"\usepackage[hidelinks]{hyperref}",
            "",
            f"\\title{{{title}}}",
            r"\author{}",
            r"\date{}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            body,
            "",
            r"\bibliographystyle{plainnat}",
            r"\bibliography{refs}",
            "",
            r"\end{document}",
            "",
        ])

    def bibtex(self) -> str:
        out = ["% refs.bib: from the References list of trackC_writeup/draft-C2.md (`to_tex build`).", ""]
        for key in sorted(self.bib):
            e = dict(self.bib[key])
            typ = e.pop("type")
            fields = ",\n".join(f"  {k} = {{{_bibesc(v)}}}" for k, v in e.items())
            out.append(f"@{typ}{{{key},\n{fields}\n}}\n")
        return "\n".join(out)


def _bibesc(v: str) -> str:
    v = re.sub(r"`([^`]+)`", r"\\texttt{\1}", v)
    v = v.replace("–", "--").replace("&", r"\&")
    v = re.sub(r"(?<!\\texttt{)([A-Z]{2,}[\w-]*)", r"{\1}", v)     # keep acronyms' case (GPT-2, LLM)
    return v


def build_text(md: str) -> Tuple[str, str]:
    c = Converter(md)
    tex = c.convert()
    bad = sorted({ch for ch in tex if ord(ch) > 127 and ch not in _SAFE})
    if bad:
        raise Refused(f"refusing: characters pdfLaTeX would not set: {bad}")
    return tex, c.bibtex()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="the LaTeX draft, from the rendered Markdown")
    ap.add_argument("step", choices=("build", "check"))
    a = ap.parse_args(argv)
    tex, bib = build_text(DRAFT.read_text())
    if a.step == "build":
        TEXDIR.mkdir(exist_ok=True)
        MAIN.write_text(tex)
        BIB.write_text(bib)
        print(f"wrote {MAIN.relative_to(HERE.parent)}, {BIB.relative_to(HERE.parent)}")
        return 0
    stale = [p.name for p, t in ((MAIN, tex), (BIB, bib)) if not p.is_file() or p.read_text() != t]
    if stale:
        print(f"{', '.join(stale)} not the conversion of {DRAFT.name}; run `python -m trackC_writeup.to_tex build`",
              file=sys.stderr)
        return 1
    print("tex/main.tex and tex/refs.bib match the draft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
