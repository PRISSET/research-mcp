"""Parse full texts into ordered blocks: arXiv HTML (LaTeXML) first, PDF (PyMuPDF) as fallback.

A block is (section_path, kind, text, page). Math keeps its LaTeX alttext, tables are flattened row by row,
references / acknowledgements are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SKIP_SECTIONS = re.compile(r"^(references|bibliography|acknowledg|author contributions|broader impact)", re.IGNORECASE)


@dataclass
class Block:
    section: str
    kind: str          # text | table | figure
    text: str
    page: int | None = None


def _clean(s: str) -> str:
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()


def _heading(node) -> str:
    h = node.css_first("h1, h2, h3, h4, h5, h6")
    if not h:
        return ""
    # LaTeXML puts the number in <span class="ltx_tag">; keep "4.2 Distillation".
    tag = h.css_first(".ltx_tag")
    num = tag.text(strip=True) if tag else ""
    full = h.text(separator=" ", strip=True)
    title = full[len(num):].strip() if num and full.startswith(num) else full
    num = re.sub(r"^Appendix\s*", "", num)
    if num and title.startswith(num):  # some sources repeat the number inside the title ("5.1 5.1 Model ...")
        title = title[len(num):].strip()
    return _clean(f"{num} {title}")


def _math_to_text(tree) -> None:
    for m in tree.css("math"):
        alt = m.attributes.get("alttext") or ""
        m.replace_with(f" ${alt}$ " if alt else " ")


def _table_text(fig) -> str:
    caption = fig.css_first("figcaption")
    cap = _clean(caption.text(separator=" ")) if caption else ""
    rows = []
    for tr in fig.css("tr"):
        cells = [_clean(td.text(separator=" ")) for td in tr.css("td, th")]
        cells = [c for c in cells]
        if any(cells):
            rows.append(" | ".join(cells))
    body = "\n".join(rows)
    return f"{cap}\n{body}".strip()


def parse_arxiv_html(html: str) -> tuple[str, list[Block]]:
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(html)
    for sel in ("nav", "header", "footer", ".ltx_bibliography", ".ltx_page_footer", "script", "style",
                ".ltx_authors", ".ltx_dates", "button"):
        for n in tree.css(sel):
            n.decompose()
    _math_to_text(tree)
    title_node = tree.css_first("h1.ltx_title_document")
    title = _clean(title_node.text(separator=" ")) if title_node else ""
    blocks: list[Block] = []

    abstract = tree.css_first("div.ltx_abstract")
    if abstract:
        h = abstract.css_first("h6")
        if h:
            h.decompose()
        blocks.append(Block("Abstract", "text", _clean(abstract.text(separator=" "))))

    def walk(node, path: list[str]) -> None:
        for child in node.iter():
            tag = child.tag
            cls = child.attributes.get("class") or ""
            if tag == "section":
                head = _heading(child)
                if head and SKIP_SECTIONS.match(re.sub(r"^[\dA-Z.]+\s+", "", head)):
                    continue
                walk(child, path + ([head] if head else []))
            elif tag == "figure" and "ltx_table" in cls:
                t = _table_text(child)
                if t:
                    blocks.append(Block(" > ".join(path) or "Body", "table", t))
            elif tag == "figure":
                cap = child.css_first("figcaption")
                if cap:
                    blocks.append(Block(" > ".join(path) or "Body", "figure", _clean(cap.text(separator=" "))))
            elif tag == "div" and ("ltx_para" in cls or "ltx_theorem" in cls or "ltx_proof" in cls):
                t = _clean(child.text(separator=" "))
                if t:
                    blocks.append(Block(" > ".join(path) or "Body", "text", t))
            elif tag in ("div", "article", "span") and "ltx_para" not in cls:
                walk(child, path)

    article = tree.css_first("article") or tree.body
    if article is not None:
        walk(article, [])
    return title, [b for b in blocks if len(b.text) > 20]


def parse_pdf(data: bytes) -> list[Block]:
    """Fallback: page text split on blank lines; section = last line that looks like a numbered heading."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    blocks: list[Block] = []
    section = "Body"
    head_re = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[A-Z]\.?)\s+[A-Z][\w ,:&-]{2,80}$")
    stop = False
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("blocks"):
            text = b[4].strip()
            if not text:
                continue
            first = text.split("\n", 1)[0].strip()
            if head_re.match(first) and len(first) < 90:
                section = first
                if SKIP_SECTIONS.match(re.sub(r"^[\dA-Z.]+\s+", "", first)):
                    stop = True
                text = text[len(first):].strip()
            elif re.match(r"^(references|bibliography)$", first, re.IGNORECASE):
                stop = True
            if stop:
                break
            t = _clean(re.sub(r"-\n(\w)", r"\1", text))
            if len(t) > 40:
                kind = "figure" if re.match(r"^(Figure|Fig\.)\s*\d+", t) else "table" if re.match(r"^Table\s*\d+", t) else "text"
                blocks.append(Block(section, kind, t, pno))
        if stop:
            break
    return blocks
