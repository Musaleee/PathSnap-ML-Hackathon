"""Render reports/PathSnap_Report.md to reports/PathSnap_Report.pdf (Markdown -> HTML -> headless Chrome).

Usage:  python -m pathsnap.report_pdf [path/to/chrome]
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

from .config import REPO_ROOT

REP = REPO_ROOT / "reports"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1b2333; }
h1 { font-size: 17pt; color: #14213D; border-bottom: 2px solid #B4531A; padding-bottom: 4px; margin: 0 0 10px; }
h2 { font-size: 13pt; color: #14213D; margin: 18px 0 6px; }
h3 { font-size: 11pt; color: #2F5F9E; margin: 14px 0 4px; }
p, li { margin: 4px 0; }
ul, ol { padding-left: 20px; margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 10px; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd4dc; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eef1f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #f2f4f7; padding: 0 3px; border-radius: 3px; }
figure { margin: 8px 0 12px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 108mm; border: 1px solid #e3e6eb; }
figcaption { font-size: 8.4pt; color: #5b6578; margin-top: 3px; font-style: italic; }
.figrow { display: flex; gap: 10px; page-break-inside: avoid; }
.figrow figure { flex: 1; }
.figrow figure img { max-height: 70mm; }
.pb { page-break-after: always; }
.cover { text-align: center; padding-top: 18mm; page-break-after: always; }
.cover .uni { font-size: 22pt; font-weight: 700; letter-spacing: 2px; color: #14213D; margin: 0; }
.cover .dept { font-size: 11pt; margin: 4px 0; }
.cover .course { font-size: 14pt; font-weight: 600; margin: 6px 0 30mm; }
.cover .title { font-size: 40pt; border: none; color: #B4531A; margin: 0; }
.cover .subtitle { font-size: 16pt; margin: 4px 0; color: #14213D; }
.cover .kind { font-size: 11pt; color: #5b6578; margin: 4px 0 22mm; }
.cover table.team { width: 60%; margin: 0 auto; font-size: 11pt; }
.cover table.team th { text-align: center; font-size: 13pt; background: #14213D; color: #fff; }
.cover .repo { margin-top: 16mm; font-size: 10pt; }
.cover .date { font-size: 10pt; color: #5b6578; }
.summary { border: 1px solid #cfd4dc; border-left: 4px solid #B4531A; padding: 6px 12px; background: #fbfaf7; margin-bottom: 10px; }
.toc { padding: 0 12px; }
"""


def figures(html):
    """<p><img alt=X src=Y></p>  ->  <figure><img><figcaption>X</figcaption></figure>"""
    return re.sub(r'<p>\s*<img alt="([^"]*)" src="([^"]+)"\s*/?>\s*</p>',
                  r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', html)


def main(chrome=CHROME):
    src = (REP / "PathSnap_Report.md").read_text()
    body = markdown.markdown(src, extensions=["tables", "md_in_html", "sane_lists"])
    body = figures(body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap - Team 26 - Technical Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out_html = REP / "PathSnap_Report.html"
    out_html.write_text(html)
    out_pdf = REP / "PathSnap_Team26_Report.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=180)
    print("written", out_pdf, f"{out_pdf.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
