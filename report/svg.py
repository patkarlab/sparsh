"""
Class-probability chart of the per-case reports, drawn as inline SVG.

The chart is written element by element, so a page needs no plotting library and no image file, and it stays sharp
on screen and on paper. It is 680 units wide, the printable width of an A4 page with 15 mm margins, so it prints at
its own size; on a narrower screen it scales down with the page.

Layout: one row per class, most probable first. The class names form a column on the left, right-aligned against the
bars. SVG text does not wrap and its width is not known in advance, so widths are estimated from the character
widths of Helvetica and Arial, which the system sans-serif fonts roughly share, with a margin. The column is made as
wide as the longest name needs, up to about a third of the chart; a longer name is wrapped at underscores, hyphens,
colons or spaces over two or more lines, and never shortened. Each bar has its probability at its end, on a white
halo so that it stays legible where it crosses the dashed threshold line; the axis runs from 0 to 1.
"""

import re
from html import escape
from typing import List, Optional, Sequence, Tuple

WIDTH = 680
FONT_SIZE = 12.5
SMALL_SIZE = 12
LINE_HEIGHT = 15
BAR_HEIGHT = 18
LABEL_MAX = 260
FONT_FAMILY = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, "
               "'Noto Sans', sans-serif")
TEXT = "#1f2328"
MUTED = "#57606a"
GRID = "#e1e4e8"
AXIS = "#8c959f"
THRESHOLD = "#24292f"
OUTLINE = "rgba(0,0,0,0.28)"

# advance widths of Helvetica/Arial in units of the font size (a character not listed counts as 0.6)
_WIDTHS = {" ": 0.278, "!": 0.278, '"': 0.355, "#": 0.556, "$": 0.556, "%": 0.889, "&": 0.667, "'": 0.191,
           "(": 0.333, ")": 0.333, "*": 0.389, "+": 0.584, ",": 0.278, "-": 0.333, ".": 0.278, "/": 0.278,
           ":": 0.278, ";": 0.278, "<": 0.584, "=": 0.584, ">": 0.584, "?": 0.556, "@": 1.015, "[": 0.278,
           "\\": 0.278, "]": 0.278, "^": 0.469, "_": 0.556, "`": 0.333, "{": 0.334, "|": 0.26, "}": 0.334,
           "~": 0.584}
_WIDTHS.update({d: 0.556 for d in "0123456789"})
_WIDTHS.update(dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        (0.667, 0.667, 0.722, 0.722, 0.667, 0.611, 0.778, 0.722, 0.278, 0.5, 0.667, 0.556, 0.833,
                         0.722, 0.778, 0.667, 0.778, 0.722, 0.667, 0.611, 0.722, 0.667, 0.944, 0.667, 0.667, 0.611))))
_WIDTHS.update(dict(zip("abcdefghijklmnopqrstuvwxyz",
                        (0.556, 0.556, 0.5, 0.556, 0.556, 0.278, 0.556, 0.556, 0.222, 0.222, 0.5, 0.222, 0.833,
                         0.556, 0.556, 0.556, 0.556, 0.333, 0.5, 0.278, 0.556, 0.5, 0.722, 0.5, 0.5, 0.5))))


def text_width(text: str, size: float = FONT_SIZE) -> float:
    """Estimated width of a line of text, with 12% to spare for fonts wider than Arial (such as DejaVu Sans)."""
    return 1.12 * size * sum(_WIDTHS.get(ch, 0.6) for ch in str(text))


def wrap_label(text: str, max_width: float, size: float = FONT_SIZE) -> List[str]:
    """
    A name split into lines no wider than max_width, at underscores, hyphens, colons or spaces (the separator stays
    at the end of its line). A part without a break point that is wider than max_width on its own is split between
    characters, so no name is ever shortened or drawn outside the chart.
    """
    text = str(text)
    if text_width(text, size) <= max_width:
        return [text]
    parts = re.findall(r"[^_\-:\s]+[_\-:\s]*|[_\-:\s]+", text)
    lines, current = [], ""
    for part in parts:
        if current and text_width(current + part, size) > max_width:
            lines.append(current)
            current = part
        else:
            current += part
    if current:
        lines.append(current)
    out = []
    for line in (line.rstrip() for line in lines):
        while text_width(line, size) > max_width and len(line) > 1:
            cut = len(line) - 1
            while cut > 1 and text_width(line[:cut], size) > max_width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)
    return out


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _text(x: float, y: float, content: str, size: float = FONT_SIZE, anchor: str = "start", fill: str = TEXT,
          extra: str = "") -> str:
    return (f'<text x="{_num(x)}" y="{_num(y)}" font-size="{_num(size)}" text-anchor="{anchor}" fill="{fill}"'
            f'{extra}>{escape(content)}</text>')


def probability_chart(rows: Sequence[Tuple[str, float, str, str]], threshold: float, threshold_label: str,
                      title: str, caution: Optional[float] = None, caution_label: str = "") -> str:
    """
    Horizontal bar chart. rows: (class name shown, probability, color, probability as text), most probable first.
    threshold_label is the text of the threshold line ("Threshold 0.90"); title names the chart for screen readers.
    caution, when given, adds a dotted line at the caution threshold, labelled caution_label to its left, and the
    threshold label then stands to the right of its line.
    """
    wrapped = [wrap_label(name, LABEL_MAX) for name, _, _, _ in rows]
    label_w = max([text_width(line) for lines in wrapped for line in lines] + [40.0])
    left = 6 + min(label_w, LABEL_MAX) + 12
    right_pad = 60
    plot_w = WIDTH - left - right_pad
    top = 28
    heights = [max(30, LINE_HEIGHT * len(lines) + 14) for lines in wrapped]
    plot_h = sum(heights)
    bottom = top + plot_h
    height = bottom + 44

    def x_of(p: float) -> float:
        return left + max(0.0, min(1.0, p)) * plot_w

    out = [f'<svg class="chart" role="img" aria-labelledby="chart-title chart-desc" width="{WIDTH}" '
           f'height="{_num(height)}" viewBox="0 0 {WIDTH} {_num(height)}" font-family="{FONT_FAMILY}" '
           'overflow="visible">',
           f'<title id="chart-title">{escape(title)}</title>',
           '<desc id="chart-desc">' + escape("; ".join(f"{name}: {text}" for name, _, _, text in rows)
                                              + f". {threshold_label}."
                                              + (f" {caution_label}." if caution is not None else "")) + "</desc>"]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = round(x_of(tick)) + 0.5
        out.append(f'<line x1="{_num(x)}" y1="{top - 4}" x2="{_num(x)}" y2="{_num(bottom)}" stroke="{GRID}" '
                   'stroke-width="1" shape-rendering="crispEdges"/>')
        out.append(_text(x, bottom + 17, f"{tick:.2f}" if 0 < tick < 1 else f"{tick:.0f}", SMALL_SIZE, "middle",
                         MUTED))
    axis_y = round(bottom) + 0.5
    out.append(f'<line x1="{_num(left)}" y1="{_num(axis_y)}" x2="{_num(left + plot_w)}" y2="{_num(axis_y)}" '
               f'stroke="{AXIS}" stroke-width="1" shape-rendering="crispEdges"/>')
    out.append(_text(left + plot_w / 2, bottom + 36, "Probability", SMALL_SIZE, "middle", MUTED))

    y = top
    for (name, p, colour, text), lines, h in zip(rows, wrapped, heights):
        bar_y = y + (h - BAR_HEIGHT) / 2
        bar_w = max(p * plot_w, 1.5)
        out.append(f'<rect x="{_num(left)}" y="{_num(bar_y)}" width="{_num(bar_w)}" height="{BAR_HEIGHT}" '
                   f'fill="{escape(colour)}" stroke="{OUTLINE}" stroke-width="0.6"/>')
        first = y + h / 2 - (len(lines) - 1) * LINE_HEIGHT / 2 + FONT_SIZE * 0.35
        for k, line in enumerate(lines):
            out.append(_text(left - 10, first + k * LINE_HEIGHT, line, anchor="end"))
        out.append(_text(left + bar_w + 6, y + h / 2 + FONT_SIZE * 0.35, text,
                         extra=' paint-order="stroke" stroke="#ffffff" stroke-width="3" stroke-linejoin="round"'))
        y += h

    tx = round(x_of(threshold)) + 0.5
    out.append(f'<line x1="{_num(tx)}" y1="{top - 8}" x2="{_num(tx)}" y2="{_num(bottom)}" stroke="{THRESHOLD}" '
               'stroke-width="1.2" stroke-dasharray="4 3"/>')
    if caution is None:
        anchor, lx = "middle", tx
        half = text_width(threshold_label, SMALL_SIZE) / 2
        if tx + half > WIDTH - 2:
            anchor, lx = "end", WIDTH - 2
    else:
        cx = round(x_of(caution)) + 0.5
        out.append(f'<line x1="{_num(cx)}" y1="{top - 8}" x2="{_num(cx)}" y2="{_num(bottom)}" stroke="{MUTED}" '
                   'stroke-width="1.2" stroke-dasharray="1.5 2.5"/>')
        out.append(_text(cx - 4, top - 12, caution_label, SMALL_SIZE, "end", MUTED))
        anchor, lx = "start", tx + 4
        if lx + text_width(threshold_label, SMALL_SIZE) > WIDTH - 2:
            anchor, lx = "end", WIDTH - 2
    out.append(_text(lx, top - 12, threshold_label, SMALL_SIZE, anchor, THRESHOLD))
    out.append("</svg>")
    return "\n".join(out)


def chip(colour: str, size: int = 12) -> str:
    """A small square in a class or lineage color, as inline SVG so that it also prints."""
    return (f'<svg class="chip" width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-hidden="true" '
            f'focusable="false"><rect x="0.5" y="0.5" width="{size - 1}" height="{size - 1}" rx="2" '
            f'fill="{escape(colour)}" stroke="{OUTLINE}" stroke-width="0.6"/></svg>')
