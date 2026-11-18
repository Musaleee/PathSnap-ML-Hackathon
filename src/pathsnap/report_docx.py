"""Build reports/PathSnap_Team26_Report.docx from the HTML that report_pdf.py renders from PathSnap_Report.md.

Usage:  python -m pathsnap.report_pdf     (writes reports/PathSnap_Report.html)
        python -m pathsnap.report_docx    (writes reports/PathSnap_Team26_Report.docx)
"""
import re
from html import unescape
from html.parser import HTMLParser

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

from .config import REPO_ROOT

REP = REPO_ROOT / "reports"
INK, ACCENT, BLUE, SOFT = RGBColor(0x14, 0x21, 0x3D), RGBColor(0xB4, 0x53, 0x1A), RGBColor(0x2F, 0x5F, 0x9E), RGBColor(0x5B, 0x65, 0x78)
VOID = {"img", "br", "hr", "meta", "link", "input"}
TEXT_W_CM = 17.0          # A4 (21 cm) minus 2 x 2 cm margins


class Node:
    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.parent, self.children = tag, dict(attrs or {}), parent, []

    def cls(self):
        return self.attrs.get("class", "").split()


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def find(node, tag):
    if isinstance(node, Node):
        if node.tag == tag:
            return node
        for c in node.children:
            r = find(c, tag)
            if r is not None:
                return r
    return None


def text_of(node):
    if isinstance(node, str):
        return node
    return "".join(text_of(c) for c in node.children)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def keep_with_next(p):
    p.paragraph_format.keep_with_next = True


def bottom_border(p, color="B4531A", sz="12"):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single"); b.set(qn("w:sz"), sz); b.set(qn("w:space"), "1"); b.set(qn("w:color"), color)
    bdr.append(b); pPr.append(bdr)


class Writer:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2))
        sec.top_margin = sec.bottom_margin = Cm(1.8)
        st = self.doc.styles
        st["Normal"].font.name = "Calibri"; st["Normal"].font.size = Pt(10.5)
        st["Normal"].paragraph_format.space_after = Pt(4)
        for name, size, color in [("Heading 1", 17, INK), ("Heading 2", 13, INK), ("Heading 3", 11.5, BLUE)]:
            s = st[name]
            s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = color
            s.paragraph_format.space_before = Pt(14 if name != "Heading 3" else 10)
            s.paragraph_format.space_after = Pt(6); s.paragraph_format.keep_with_next = True
            rpr = s.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts"); rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(a), "Calibri")
        self.add_page_numbers()

    def add_page_numbers(self):
        p = self.doc.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("PathSnap · Team 26 · page "); r.font.size = Pt(8); r.font.color.rgb = SOFT
        for kind, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
            run = p.add_run(); run.font.size = Pt(8); run.font.color.rgb = SOFT
            if kind:
                f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), kind); run._r.append(f)
            else:
                t = OxmlElement("w:instrText"); t.set(qn("xml:space"), "preserve"); t.text = txt; run._r.append(t)

    # ---------- inline ----------
    def inline(self, p, node, fmt=None, size=None):
        fmt = dict(fmt or {})
        if isinstance(node, str):
            t = re.sub(r"\s+", " ", unescape(node))
            if not t:
                return
            r = p.add_run(t)
            r.bold = fmt.get("b"); r.italic = fmt.get("i")
            if fmt.get("code"):
                r.font.name = "Consolas"; r.font.size = Pt(9 if size is None else size - 0.5)
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            elif size:
                r.font.size = Pt(size)
            if fmt.get("sub"):
                r.font.subscript = True
            if fmt.get("color"):
                r.font.color.rgb = fmt["color"]
            return
        t = node.tag
        if t == "br":
            p.add_run().add_break(); return
        if t in ("strong", "b"):
            fmt["b"] = True
        elif t in ("em", "i"):
            fmt["i"] = True
        elif t == "code":
            fmt["code"] = True
        elif t == "sub":
            fmt["sub"] = True
        for c in node.children:
            self.inline(p, c, fmt, size)

    def strip_edges(self, p):
        if p.runs:
            p.runs[0].text = p.runs[0].text.lstrip()
            p.runs[-1].text = p.runs[-1].text.rstrip()

    def paragraph(self, node, style=None, align=None, size=None, fmt=None):
        p = self.doc.add_paragraph(style=style)
        if align:
            p.alignment = align
        for c in node.children:
            self.inline(p, c, fmt, size)
        self.strip_edges(p)
        return p

    # ---------- blocks ----------
    def block(self, node, list_level=0):
        if isinstance(node, str):
            if node.strip():
                p = self.doc.add_paragraph(); self.inline(p, node); self.strip_edges(p)
            return
        t, cls = node.tag, node.cls()
        if t == "div" and "cover" in cls:
            return self.cover(node)
        if t == "div" and "pb" in cls:
            return self.page_break()
        if t == "div" and "figrow" in cls:
            return self.figrow(node)
        if t == "div" and "summary" in cls:
            for c in node.children:
                self.block(c)
            return
        if t in ("h1", "h2", "h3"):
            p = self.doc.add_heading(level=int(t[1]))
            for c in node.children:
                self.inline(p, c)
            self.strip_edges(p)
            if t == "h1":
                bottom_border(p)
            return
        if t == "p":
            imgs = [c for c in node.children if isinstance(c, Node) and c.tag == "img"]
            if imgs and not text_of(node).strip():
                for im in imgs:
                    self.figure_img(im.attrs["src"], im.attrs.get("alt", ""))
                return
            p = self.paragraph(node)
            if text_of(node).strip().endswith(":"):
                keep_with_next(p)
            return
        if t in ("ul", "ol"):
            return self.list_block(node, t, list_level)
        if t == "table":
            return self.table(node)
        if t == "figure":
            img = find(node, "img"); cap = find(node, "figcaption")
            return self.figure_img(img.attrs["src"], text_of(cap) if cap is not None else "")
        for c in node.children:          # div, section, body ...
            self.block(c, list_level)

    def list_block(self, node, kind, level):
        n = int(node.attrs.get("start", 1))
        for li in [c for c in node.children if isinstance(c, Node) and c.tag == "li"]:
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            indent = 0.6 + 0.6 * level
            pf.left_indent = Cm(indent); pf.first_line_indent = Cm(-0.5); pf.space_after = Pt(2)
            p.add_run(f"{n}.\t" if kind == "ol" else ("•\t" if level == 0 else "–\t"))
            pf.tab_stops.add_tab_stop(Cm(indent))
            n += 1
            sub = []
            first = True
            for c in li.children:
                if isinstance(c, Node) and c.tag in ("ul", "ol", "table", "figure"):
                    sub.append(c)
                elif isinstance(c, Node) and c.tag == "p":
                    if first:
                        for cc in c.children:
                            self.inline(p, cc)
                        first = False
                    else:
                        sub.append(c)
                else:
                    if sub:
                        sub.append(c)
                    else:
                        self.inline(p, c)
            self.strip_edges(p)
            for s in sub:
                if isinstance(s, Node) and s.tag in ("ul", "ol"):
                    self.list_block(s, s.tag, level + 1)
                elif isinstance(s, Node) and s.tag == "p":
                    q = self.paragraph(s)
                    q.paragraph_format.left_indent = Cm(0.6 + 0.6 * level)
                elif isinstance(s, Node):
                    self.block(s, level + 1)

    def table(self, node, font=8.5, header_fill="EEF1F6"):
        rows = []
        for tr in self.iter_tag(node, "tr"):
            cells = [c for c in tr.children if isinstance(c, Node) and c.tag in ("th", "td")]
            rows.append(cells)
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        tbl = self.doc.add_table(rows=len(rows), cols=ncol)
        tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, r in enumerate(rows):
            for j, cell in enumerate(r):
                dc = tbl.cell(i, j)
                p = dc.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                is_head = cell.tag == "th"
                for c in cell.children:
                    self.inline(p, c, {"b": True} if is_head else None, size=font)
                self.strip_edges(p)
                if is_head:
                    shade(dc, header_fill)
        # keep rows together where possible
        for row in tbl.rows:
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return tbl

    def iter_tag(self, node, tag):
        for c in node.children:
            if isinstance(c, Node):
                if c.tag == tag:
                    yield c
                else:
                    yield from self.iter_tag(c, tag)

    def img_size(self, path, max_w, max_h):
        w, h = Image.open(path).size
        width = max_w
        if width * h / w > max_h:
            width = max_h * w / h
        return Cm(width)

    def figure_img(self, src, caption, max_w=TEXT_W_CM, max_h=11.5):
        path = REP / src
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        keep_with_next(p)
        p.add_run().add_picture(str(path), width=self.img_size(path, max_w, max_h))
        if caption:
            c = self.doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = c.add_run(caption); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = SOFT
            c.paragraph_format.space_after = Pt(8)

    def figrow(self, node):
        figs = [f for f in self.iter_tag(node, "figure")]
        if not figs:
            figs_p = [im for im in self.iter_tag(node, "img")]
            items = [(im.attrs["src"], im.attrs.get("alt", "")) for im in figs_p]
        else:
            items = [(find(f, "img").attrs["src"], text_of(find(f, "figcaption"))) for f in figs]
        tbl = self.doc.add_table(rows=1, cols=len(items)); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        w = TEXT_W_CM / len(items) - 0.3
        for j, (src, cap) in enumerate(items):
            cell = tbl.cell(0, j)
            p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(REP / src), width=self.img_size(REP / src, w, 7.5))
            c = cell.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = c.add_run(cap); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = SOFT
        self.doc.add_paragraph()

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def cover(self, node):
        def line(text, size, bold=False, color=INK, before=0, after=4):
            p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after)
            r = p.add_run(text); r.font.size = Pt(size); r.bold = bold; r.font.color.rgb = color
            return p
        get = lambda c: text_of(next(n for n in node.children if isinstance(n, Node) and c in n.cls())).strip()
        line(get("uni"), 22, True, before=40)
        line(get("dept"), 11)
        line(get("course"), 14, True, after=70)
        line(text_of(find(node, "h1")).strip(), 40, True, ACCENT)
        line(get("subtitle"), 16)
        line(get("kind"), 11, color=SOFT, after=50)
        t = find(node, "table")
        rows = [[text_of(c).strip() for c in tr.children if isinstance(c, Node)] for tr in self.iter_tag(t, "tr")]
        tbl = self.doc.add_table(rows=len(rows), cols=2); tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        head = tbl.cell(0, 0).merge(tbl.cell(0, 1))
        hp = head.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr = hp.add_run(rows[0][0]); hr.bold = True; hr.font.size = Pt(13); hr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(head, "14213D")
        for i, r in enumerate(rows[1:], start=1):
            for j in range(2):
                run = tbl.cell(i, j).paragraphs[0].add_run(r[j]); run.font.size = Pt(11)
        for row in tbl.rows:
            row.cells[0].width = Cm(7); row.cells[-1].width = Cm(5)
        repo = next(n for n in node.children if isinstance(n, Node) and "repo" in n.cls())
        p = line("", 10, before=50)
        self.inline(p, repo)
        line(get("date"), 10, color=SOFT)
        self.page_break()


def main():
    html = (REP / "PathSnap_Report.html").read_text()
    tb = TreeBuilder(); tb.feed(html)
    body = find(tb.root, "body")
    w = Writer()
    for c in body.children:
        w.block(c)
    out = REP / "PathSnap_Team26_Report.docx"
    w.doc.core_properties.title = "PathSnap - Team 26 - Technical Report"
    w.doc.core_properties.author = "Team 26"
    w.doc.save(out)
    print("written", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
