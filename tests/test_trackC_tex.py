"""C2's LaTeX draft (`trackC_writeup/to_tex.py`, C2p unit 1): the tracked `tex/` is the conversion
of the rendered draft, the conversion sets the draft's Unicode mathematics as LaTeX, and it refuses
what it cannot convert."""
import pytest

pytestmark = pytest.mark.pure
from trackC_writeup import numbers as n
from trackC_writeup import to_tex as t

REFS = "\n## References\n\n- Geshkovski, B., Rigollet, P. (2023). A mathematical perspective. arXiv:2312.10794.\n"


def conv(body: str) -> str:
    return t.Converter("# T\n\n" + body + "\n" + REFS).convert()


def test_tracked_tex_is_the_conversion_of_the_draft():
    tex, bib = t.build_text(t.DRAFT.read_text())
    assert t.MAIN.read_text() == tex
    assert t.BIB.read_text() == bib


def test_every_figure_in_the_draft_has_its_pdf():
    for line in t.DRAFT.read_text().splitlines():
        if line.startswith("![Figure"):
            stem = line.split("fig/")[1].split(".svg")[0]
            assert (t.TEXDIR / "fig" / f"{stem}.pdf").is_file(), stem
            assert (n.FIGDIR / f"{stem}.svg").is_file(), stem


def test_unicode_mathematics_becomes_math_mode():
    c = t.Converter("# T\n" + REFS)
    assert c.inline("E₁ falls") == "$E_{1}$ falls"
    assert c.inline("‖m‖² of 10⁻⁴") == r"$\|m\|^{2}$ of $10^{-4}$"
    assert c.inline("E_β and S^{d−1}") == r"$E_\beta $ and $S^{d-1}$"
    assert c.inline("xᵢ ∈ S^{d−1}") == r"$x_{i} \in S^{d-1}$"
    assert c.inline("E = κ p^(κ−1) with") == r"E = $\kappa p^{\kappa -1}$ with"
    assert c.inline("softmaxⱼ") == r"$\operatorname{softmax}_{j}$"


def test_text_is_escaped_and_marked_up():
    c = t.Converter("# T\n" + REFS)
    assert c.inline("50 % of `a_b` and **bold** and *it*") == r"50 \% of \texttt{a\_b} and \textbf{bold} and \emph{it}"


def test_citations_become_natbib_keys():
    c = t.Converter("# T\n" + REFS)
    assert c.inline("[Geshkovski et al. 2023, §3.2]") == r"\citep[\S{}3.2]{geshkovski2023}"
    assert c.inline("Geshkovski et al. [2023]") == r"\citet{geshkovski2023}"
    assert c.inline("[ibid., Theorem 17]") == r"\citep[Theorem 17]{geshkovski2023}"


def test_refuses_an_unknown_citation_figure_or_display():
    with pytest.raises(t.Refused):
        conv("As shown [Nobody 2020].")
    with pytest.raises(t.Refused):
        conv("See figure 3.")
    with pytest.raises(t.Refused):
        conv("  E = mc².")
    with pytest.raises(t.Refused):
        conv("A bare caret, a^b.")


def test_sections_and_figures_get_labels():
    tex = conv("## 2. Background\n\nSee §2.\n\n![Figure 1](fig/fig1_x.svg)\n\n*Figure 1. A caption.*\n\nAs figure 1 shows.")
    assert r"\section{Background}\label{sec:2}" in tex
    assert r"\S\ref{sec:2}" in tex
    assert r"\includegraphics[width=\linewidth]{fig/fig1_x.pdf}" in tex
    assert r"figure~\ref{fig:fig1_x}" in tex
