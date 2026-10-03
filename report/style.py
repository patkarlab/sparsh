"""
Style sheets of the report pages, written into each page so that it needs no other file.

A sober layout: the system font stack, dark text on white, one accent color (a dark blue, for headings and rules),
and color otherwise only in the class and lineage chips and in the bars of the chart. The print style sheet sets
A4 pages with 15 mm margins, keeps the result panel, the chart and tables on one page where they fit, and, in
browsers that support page-margin boxes, prints the sample name, the research-use statement and the page number at
the foot of every page. The footer line at the end of a page is left out of print, because the header already
states the research use and the time of the report.
"""

ACCENT = "#1f4e79"

BASE = """
:root { --text: #1f2328; --muted: #57606a; --faint: #6e7781; --rule: #d0d7de; --soft: #f6f8fa;
        --accent: %(accent)s; }
* { box-sizing: border-box; }
html { font-size: 15px; -webkit-text-size-adjust: 100%%; text-size-adjust: 100%%; }
body { margin: 0; background: #ffffff; color: var(--text);
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans",
                    "Liberation Sans", sans-serif;
       font-size: 1rem; line-height: 1.55; }
.page { max-width: 800px; margin: 0 auto; padding: 28px 32px 64px; }
h1 { font-size: 1.4rem; font-weight: 600; line-height: 1.3; margin: 0 0 4px; letter-spacing: -0.005em; }
h2 { font-size: 1.05rem; font-weight: 600; color: var(--accent); margin: 36px 0 12px; padding-bottom: 6px;
     border-bottom: 1px solid var(--rule); }
h3 { font-size: 0.95rem; font-weight: 600; margin: 20px 0 6px; }
p { margin: 0 0 10px; }
a { color: var(--accent); }
ul { margin: 4px 0 12px; padding-left: 22px; }
li { margin: 2px 0; }
.muted { color: var(--muted); }
.small { font-size: 0.87rem; }
.mono, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
              font-size: 0.92em; }
.nav { font-size: 0.87rem; margin-bottom: 18px; }
header.report { border-top: 4px solid var(--accent); padding-top: 16px; }
.ruo { border: 1.5px solid var(--text); padding: 7px 12px; margin: 14px 0 18px; font-weight: 600; }
dl.meta { display: grid; grid-template-columns: max-content 1fr; gap: 3px 18px; margin: 0; font-size: 0.93rem; }
dl.meta dt { color: var(--muted); }
dl.meta dd { margin: 0; overflow-wrap: anywhere; }
.chip { display: inline-block; vertical-align: -1px; margin-right: 7px; flex: none; }
table { border-collapse: collapse; width: 100%%; font-size: 0.92rem; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.84rem; border-bottom: 1px solid var(--muted); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
footer.report { margin-top: 40px; padding-top: 10px; border-top: 1px solid var(--rule); font-size: 0.82rem;
                color: var(--faint); }
"""

CASE = """
.panel { border: 1px solid var(--rule); border-left: 4px solid var(--accent); padding: 18px 20px 14px;
         margin: 26px 0 8px; }
.fields { display: grid; grid-template-columns: 1fr 1.7fr 1.1fr; gap: 18px 26px; }
.field .label { font-size: 0.8rem; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase;
                margin-bottom: 3px; }
.field .value { font-size: 1.28rem; font-weight: 500; line-height: 1.3; overflow-wrap: anywhere;
                display: flex; align-items: baseline; }
.field .value .chip { align-self: center; }
.field .value.plain { display: block; }
.field .value .threshold { font-size: 0.8rem; color: var(--muted); font-weight: 400; white-space: nowrap; }
.field .sub { font-size: 0.88rem; color: var(--muted); margin-top: 4px; overflow-wrap: anywhere; }
.level { margin: 16px 0 0; padding-top: 12px; border-top: 1px solid var(--rule); }
.level p { margin: 0 0 4px; }
figure { margin: 8px 0 14px; }
figure svg.chart { display: block; max-width: 100%%; height: auto; }
figcaption { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }
table.sums { width: auto; min-width: 50%%; }
table.sums td:first-child { padding-right: 24px; }
.caveats { margin-top: 14px; }
.caveats p { padding-left: 12px; border-left: 2px solid var(--rule); }
.iscn { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 0.95rem; background: var(--soft); border: 1px solid var(--rule); padding: 10px 12px;
        margin: 4px 0 14px; white-space: pre-wrap; overflow-wrap: break-word; }
table.events { font-size: 0.86rem; }
table.events td { white-space: nowrap; padding-right: 12px; }
table.events th { padding-right: 12px; vertical-align: bottom; }
table.events td:last-child, table.events th:last-child { padding-right: 0; }
img.karyotype { display: block; max-width: 100%%; height: auto; margin: 10px 0 4px; border: 1px solid var(--rule); }
.table-wrap { overflow-x: auto; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; }
@media (max-width: 640px) {
  .page { padding: 18px 16px 48px; }
  .fields, .two { grid-template-columns: 1fr; }
  figure.probabilities { overflow-x: auto; }
  figure.probabilities svg.chart { max-width: none; }
  dl.meta { grid-template-columns: 1fr; gap: 0; }
  dl.meta dd { margin-bottom: 6px; }
}
"""

INDEX = """
.page { max-width: 1120px; }
.tools { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin: 22px 0 10px; }
.tools input { font: inherit; font-size: 0.93rem; padding: 6px 10px; width: 280px; max-width: 100%%;
               border: 1px solid var(--rule); border-radius: 4px; }
.tools input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.table-wrap { overflow-x: auto; }
table.samples td, table.samples th { white-space: nowrap; }
table.samples td.wrap { white-space: normal; min-width: 220px; }
table.samples th button { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer;
                          text-align: inherit; width: 100%%; }
table.samples th button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
table.samples th[aria-sort] button::after { content: ""; display: inline-block; width: 0.9em; }
table.samples th[aria-sort="ascending"] button::after { content: " \\2191"; }
table.samples th[aria-sort="descending"] button::after { content: " \\2193"; }
table.samples tbody tr:hover { background: var(--soft); }
.flag { display: inline-block; font-size: 0.78rem; color: var(--muted); border: 1px solid var(--rule);
        border-radius: 3px; padding: 0 5px; margin: 1px 4px 1px 0; }
.lin { display: inline-flex; align-items: center; }
"""

PRINT = """
@page { size: A4; margin: 15mm 15mm 17mm; }
@media print {
  html { font-size: 9.6pt; }
  body { line-height: 1.45; }
  .page { max-width: none; padding: 0; }
  .nav, .tools, footer.report { display: none; }
  h2 { margin-top: 20px; break-after: avoid; page-break-after: avoid; }
  h3 { break-after: avoid; page-break-after: avoid; }
  .panel, figure, table, .iscn, dl.meta, .ruo, img { break-inside: avoid; page-break-inside: avoid; }
  tr { break-inside: avoid; page-break-inside: avoid; }
  p, li { orphans: 3; widows: 3; }
  a { color: inherit; text-decoration: none; }
  .table-wrap { overflow: visible; }
  table.samples td, table.samples th { white-space: normal; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def css_string(text: str) -> str:
    """Text as a quoted CSS string that is safe inside a style element (no quote, backslash, < or > survives)."""
    out = []
    for ch in str(text):
        if ch in '"\\<>' or ord(ch) < 32:
            out.append(f"\\{ord(ch):x} ")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


FOOT = "font-size: 7.5pt; color: #57606a; font-family: Arial, 'Helvetica Neue', 'Liberation Sans', sans-serif;"


def case_css(sample: str) -> str:
    """The style sheet of a case page; the page foot names the sample."""
    left = css_string(f"Sample {sample}. For research use only.")
    foot = (f"\n@page {{ @bottom-left {{ content: {left}; {FOOT} }}\n"
            f"        @bottom-right {{ content: \"Page \" counter(page) \" of \" counter(pages); {FOOT} }} }}\n")
    return (BASE + CASE) % {"accent": ACCENT} + PRINT + foot


def index_css() -> str:
    """The style sheet of the index page, printed in landscape."""
    foot = (f"\n@page {{ size: A4 landscape;\n"
            f"        @bottom-left {{ content: \"For research use only.\"; {FOOT} }}\n"
            f"        @bottom-right {{ content: \"Page \" counter(page) \" of \" counter(pages); {FOOT} }} }}\n")
    return (BASE + INDEX) % {"accent": ACCENT} + PRINT + foot
