"""Render docs/README.md to docs/README.pdf.

No pandoc/LaTeX on this box, so we build the PDF from what's available:
  - reportlab  -> page layout, text flow, automatic pagination
  - matplotlib -> renders each display equation ($$...$$) via mathtext to a
                  crisp transparent PNG that is embedded as an image
  - DejaVu TTF (bundled with matplotlib) -> registered with reportlab so inline
                  Greek/math glyphs (alpha, lambda, Sigma, norms, arrows) render

Inline math ($...$) is converted to Unicode + <super>/<sub> markup. Display math
is imaged. Run:  python docs/build_pdf.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted, HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

HERE = Path(__file__).parent
SRC = HERE / "README.md"
OUT = HERE / "README.pdf"
EQDIR = HERE / "_eq"
EQDIR.mkdir(exist_ok=True)

# ---- fonts: use matplotlib's DejaVu (broad Unicode coverage) ---------------
FONTDIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONTDIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONTDIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONTDIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONTDIR / "DejaVuSansMono.ttf")))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")

INK = HexColor("#1a1a1a")
ACCENT = HexColor("#0b4f6c")
CODEBG = HexColor("#f2f3f5")

H1 = ParagraphStyle("H1", fontName="DejaVu-Bold", fontSize=18, leading=23,
                    textColor=ACCENT, spaceBefore=6, spaceAfter=12)
H2 = ParagraphStyle("H2", fontName="DejaVu-Bold", fontSize=13.5, leading=18,
                    textColor=ACCENT, spaceBefore=16, spaceAfter=6)
H3 = ParagraphStyle("H3", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                    textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", fontName="DejaVu", fontSize=10, leading=15,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=16, bulletIndent=4,
                        spaceAfter=3)
CODE = ParagraphStyle("Code", fontName="DejaVuMono", fontSize=8.5, leading=11,
                      textColor=INK, backColor=CODEBG, borderPadding=6,
                      spaceBefore=4, spaceAfter=8)

# ---- inline LaTeX -> Unicode / reportlab markup ----------------------------
GREEK = {
    r"\varepsilon": "ε", r"\epsilon": "ε", r"\alpha": "α", r"\lambda": "λ",
    r"\gamma": "γ", r"\sigma": "σ", r"\mu": "μ", r"\delta": "δ", r"\Delta": "Δ",
    r"\Sigma": "Σ", r"\sum": "Σ", r"\Pi": "Π", r"\cdot": "·", r"\geq": "≥",
    r"\leq": "≤", r"\le": "≤", r"\ge": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\times": "×", r"\to": "→", r"\leftarrow": "←", r"\in": "∈",
    r"\infty": "∞", r"\nabla": "∇", r"\partial": "∂", r"\pm": "±",
    r"\dots": "…", r"\,": " ", r"\;": " ", r"\ ": " ", r"\!": "",
    r"\tfrac12": "½", r"\top": "T",
}


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def conv_math(s: str) -> str:
    s = s.replace(r"\{", "{").replace(r"\}", "}")  # literal braces
    s = re.sub(r"\\t?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)  # inline fractions
    for k in sorted(GREEK, key=len, reverse=True):  # longest first (\leftarrow before \le)
        s = s.replace(k, GREEK[k])
    s = s.replace(r"\|", "‖")
    # escape literal comparison/entity chars BEFORE inserting real markup tags
    s = esc(s)
    s = re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", s)
    s = re.sub(r"\^(\\?\w)", r"<super>\1</super>", s)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\\?\w)", r"<sub>\1</sub>", s)
    s = s.replace(r"\mathrm", "").replace(r"\text", "")
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return s


def inline(text: str) -> str:
    # Stash math first so **bold**/`code` spanning across $...$ still matches,
    # then escape the prose, apply emphasis, and restore converted math.
    maths: list[str] = []

    def stash(m):
        maths.append(m.group(1))
        return f"\x00{len(maths) - 1}\x00"

    tmp = re.sub(r"\$([^$]*)\$", stash, text)
    e = esc(tmp)
    e = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", e)          # bold first
    e = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", e)  # then italic
    e = re.sub(r"`([^`]+)`", r'<font face="DejaVuMono" size=8.5>\1</font>', e)
    e = re.sub(r"\x00(\d+)\x00", lambda m: conv_math(maths[int(m.group(1))]), e)
    return e


# ---- display equation ($$...$$) -> image -----------------------------------
def render_eq(latex: str, idx: int, fontsize: int = 15) -> Path:
    path = EQDIR / f"eq_{idx:03d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center", va="center",
             color="#1a1a1a")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06,
                transparent=True)
    plt.close(fig)
    return path


def eq_flowable(path: Path, dpi: int = 220, max_w: float = 5.3 * inch) -> Image:
    iw, ih = ImageReader(str(path)).getSize()
    w_pt = iw * 72.0 / dpi
    h_pt = ih * 72.0 / dpi
    if w_pt > max_w:
        r = max_w / w_pt
        w_pt, h_pt = max_w, h_pt * r
    img = Image(str(path), width=w_pt, height=h_pt)
    img.hAlign = "CENTER"
    return img


# ---- markdown -> flowables --------------------------------------------------
def build():
    lines = SRC.read_text().splitlines()
    flow = []
    buf: list[str] = []
    kind = None          # "para" | "bullet" | "olist"
    prefix = ""          # ordered-list number prefix
    eq_lines: list[str] = []
    code_lines: list[str] = []
    in_eq = False
    in_code = False

    def flush():
        nonlocal buf, kind, prefix
        if buf:
            txt = inline(" ".join(buf).strip())
            if kind == "bullet":
                flow.append(Paragraph("•&nbsp;&nbsp;" + txt, BULLET))
            elif kind == "olist":
                flow.append(Paragraph(prefix + txt, BULLET))
            else:
                flow.append(Paragraph(txt, BODY))
        buf = []
        kind = None
        prefix = ""

    for raw in lines:
        line = raw.rstrip("\n")

        if in_code:
            if line.strip().startswith("```"):
                flow.append(Preformatted("\n".join(code_lines), CODE))
                code_lines = []
                in_code = False
            else:
                code_lines.append(line)
            continue

        if in_eq:
            if line.strip() == "$$":
                flow.append(Spacer(1, 4))
                flow.append(eq_flowable(render_eq(" ".join(eq_lines).strip(), len(flow))))
                flow.append(Spacer(1, 4))
                eq_lines = []
                in_eq = False
            else:
                eq_lines.append(line)
            continue

        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            in_code = True
            continue

        if stripped == "$$":
            flush()
            in_eq = True
            continue

        m_one = re.match(r"^\$\$(.+)\$\$$", stripped)  # single-line $$...$$
        if m_one:
            flush()
            flow.append(Spacer(1, 4))
            flow.append(eq_flowable(render_eq(m_one.group(1).strip(), len(flow))))
            flow.append(Spacer(1, 4))
            continue

        if stripped == "---":
            flush()
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=HexColor("#c9ced4")))
            flow.append(Spacer(1, 4))
            continue

        if stripped.startswith("# "):
            flush()
            flow.append(Paragraph(inline(stripped[2:]), H1))
            continue
        if stripped.startswith("## "):
            flush()
            flow.append(Paragraph(inline(stripped[3:]), H2))
            continue
        if stripped.startswith("### "):
            flush()
            flow.append(Paragraph(inline(stripped[4:]), H3))
            continue

        m_ul = re.match(r"^[-*]\s+(.*)", stripped)
        if m_ul:
            flush()
            kind = "bullet"
            buf = [m_ul.group(1)]
            continue
        m_ol = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m_ol:
            flush()
            kind = "olist"
            prefix = f"{m_ol.group(1)}.&nbsp;&nbsp;"
            buf = [m_ol.group(2)]
            continue

        if stripped == "":
            flush()
            continue

        # plain text: continuation of the active block, or a new paragraph
        if kind is None:
            kind = "para"
        buf.append(stripped)

    flush()
    return flow


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="Decentralized Gossip-SDCA: Problem and Fix",
    )
    doc.build(build())
    # remove the equation-image scratch dir (images are embedded in the PDF)
    import shutil
    shutil.rmtree(EQDIR, ignore_errors=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
