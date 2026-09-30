"""
HTML pages of the per-case reports, and report_summary.csv.

write_report() writes, into the output folder:
  cases/<SAMPLE>.html   one self-contained page per sample: header, result, class probabilities, interpretation,
                        evidence from cross-validation, copy-number karyotype, input and quality, and a note on
                        the method, in that order;
  index.html            all samples in one table that sorts by any column and filters as one types;
  report_summary.csv    one row per sample with the fields of the result panel and the interpretation headline.

The pages are plain HTML5 with inline CSS and inline SVG; a karyotype figure is embedded as a data URI. They load
nothing from outside themselves, and the index page's only script is inline. Every value taken from an input file
is escaped, so a sample name, a sample-sheet entry or a karyotype string cannot change the structure of a page.
"""

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import style, text
from .inputs import Case, Context, Karyotype
from .svg import chip, probability_chart

RUO = "For research use only. Not for use in diagnostic procedures."
LEVEL_ORDER = {"subtype": 0, "no_subtype": 1, "family": 2, "lineage": 3, "none": 4}
EVENT_COLUMNS = (("type", "Type"), ("level", "Extent"), ("chrom", "Chromosome"), ("bands", "Bands"),
                 ("start", "Start"), ("end", "End"), ("size_mb", "Size (Mb)"), ("copy_number", "Copy number"),
                 ("cell_fraction", "Cell fraction"), ("whole_chromosome", "Whole chromosome"),
                 ("genes", "Genes of interest"))
LEVEL_WORDS = {"whole": "whole chromosome", "arm": "chromosome arm", "segment": "segment"}


def e(value) -> str:
    """Escaped text of any value, for element content and attribute values alike."""
    return escape("" if value is None else str(value), quote=True)


def when(moment: datetime) -> str:
    """30 September 2026, 10:15 (UTC+05:30)."""
    offset = moment.strftime("%z")
    zone = f" (UTC{offset[:3]}:{offset[3:]})" if offset else ""
    return f"{moment.day} {moment:%B %Y}, {moment:%H:%M}{zone}"


def recorded_time(value) -> str:
    """A time recorded as ISO text, written like when(); anything unreadable is shown as it is."""
    try:
        return when(datetime.fromisoformat(str(value)))
    except ValueError:
        return str(value)


def model_line(ctx: Context) -> str:
    """Model folder, networks, classes and threshold, as one line of the header."""
    m = ctx.model
    parts = []
    if m.n_networks:
        nets = f"{m.n_networks} network{'s' if m.n_networks != 1 else ''}"
        if m.models_used == "ensemble":
            nets += " (cross-validation fold ensemble)"
        elif m.models_used == "final":
            nets += " (final model)"
        parts.append(nets)
    parts.append(f"{m.n_classes} classes")
    threshold = f"reporting threshold {text.fmt_threshold(ctx.threshold)}"
    if ctx.recomputed:
        threshold += f" (the prediction run used {text.fmt_threshold(ctx.prediction_threshold)})"
    parts.append(threshold)
    lead = f"{m.name}: " if m.name else "Model folder not recorded; "
    return lead + ", ".join(parts)


def version_line(ctx: Context) -> str:
    """The version that wrote the report, and the one that made the predictions when it differs."""
    line = ctx.report_version
    if ctx.prediction_version and ctx.prediction_version != ctx.report_version:
        line += f" (predictions made with {ctx.prediction_version})"
    return line


def dl(rows: Sequence[Tuple[str, str]], raw: bool = False, cls: str = "meta") -> str:
    """A definition list; values are escaped unless raw (then they are HTML built here)."""
    items = "".join(f"<dt>{e(k)}</dt><dd>{v if raw else e(v)}</dd>" for k, v in rows)
    return f'<dl class="{cls}">{items}</dl>'


def document(title: str, css: str, body: str, script: str = "") -> str:
    """A complete page. Its content security policy lets it load nothing from outside itself."""
    tail = f"<script>\n{script}\n</script>\n" if script else ""
    policy = ("default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
              + ("script-src 'unsafe-inline'" if script else "script-src 'none'"))
    return ("<!DOCTYPE html>\n<html lang=\"en-GB\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"{policy}\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<meta name=\"referrer\" content=\"no-referrer\">\n"
            f"<title>{e(title)}</title>\n<style>{css}</style>\n</head>\n<body>\n{body}\n{tail}</body>\n</html>\n")


# ---------------------------------------------------------------------------------------------- case page parts
def header(case: Case, ctx: Context) -> str:
    """Title, research-use statement, sample, time of the report, version and model."""
    rows = [("Sample", f'<span class="mono">{e(case.sample)}</span>'),
            ("Report generated", e(when(ctx.generated))),
            ("SPARSH version", e(version_line(ctx))),
            ("Model", e(model_line(ctx)))]
    return (f'<header class="report">\n<h1>{e(ctx.title)}</h1>\n<p class="ruo">{e(RUO)}</p>\n'
            f'{dl(rows, raw=True)}\n</header>')


def result_panel(case: Case, res: text.Result, ctx: Context) -> str:
    """Lineage, result and confidence, with the reporting level in words beneath them."""
    if res.lineage is not None:
        lineage = f"{chip(ctx.lineage_colour(res.lineage))}<span>{e(res.lineage)}</span>"
    else:
        lineage = "<span>Not determined</span>"
    result_chip = chip(ctx.colour(res.result_class)) if res.result_class else ""
    result_subs = "".join(f'<div class="sub">{e(s)}</div>' for s in (res.result_name, res.result_note) if s)
    if res.confidence is not None:
        confidence = (f"{e(text.fmt_prob(res.confidence, ctx.threshold))} "
                      f'<span class="threshold">(reporting threshold {e(text.fmt_threshold(ctx.threshold))})</span>')
    else:
        confidence = e(res.confidence_text)
    fields = [
        ("Lineage", lineage, f'<div class="sub">{e(res.lineage_note)}</div>', "value"),
        ("Result", f"{result_chip}<span>{e(res.result)}</span>", result_subs, "value"),
        ("Confidence", confidence, "", "value plain"),
    ]
    cells = "".join(f'<div class="field"><div class="label">{label}</div><div class="{cls}">{value}</div>{sub}</div>'
                    for label, value, sub, cls in fields)
    level = f"<p>{e(res.level_text)}</p>" + (f"<p>{e(res.most_likely)}</p>" if res.most_likely else "")
    return (f'<section class="panel" aria-label="Result">\n<div class="fields">{cells}</div>\n'
            f'<div class="level">{level}</div>\n</section>')


def probabilities(case: Case, ctx: Context) -> str:
    """The chart of the five most probable classes, the lineage sums and the family sums."""
    t = ctx.threshold
    top = text.top_classes(case, ctx, 5)
    rows = [(ctx.shown(c), p, ctx.colour(c), text.fmt_prob(p, t)) for c, p in top]
    svg = probability_chart(rows, t, f"Threshold {text.fmt_threshold(t)}",
                            f"Class probabilities of sample {case.sample}: the {len(rows)} most probable classes")
    n = len(ctx.classes)
    caption = (f"The {len(rows)} most probable of the model's {n} classes, in the class colours. The dashed line "
               f"marks the reporting threshold of {text.fmt_threshold(t)}. The probabilities of all {n} classes add "
               "up to 1.")
    sums = text.lineage_sums(case, ctx)
    table = "".join(f'<tr><td><span class="lin">{chip(ctx.lineage_colour(lin))}{e(lin)}</span></td>'
                    f'<td class="num">{e(text.fmt_prob(p, t))}</td></tr>' for lin, p in sums)
    parts = [f'<section>\n<h2>Class probabilities</h2>\n<figure class="probabilities">\n{svg}\n'
             f'<figcaption>{e(caption)}</figcaption>\n'
             '</figure>\n<h3>Summed by lineage</h3>\n<table class="sums"><thead><tr><th>Lineage</th>'
             f'<th class="num">Probability</th></tr></thead><tbody>{table}</tbody></table>',
             '<p class="small muted">Lineages with a summed probability of at least 0.01.</p>']
    families = []
    own = ctx.hierarchy.family_of.get(case.top)
    if own:
        families.append((own, "Family of the most probable class"))
    if case.level == "family" and case.family and case.family != own:
        families.append((case.family, "Reported family"))
    for family, heading in families:
        p, members = text.family_sum(case, ctx, family)
        parts.append(f"<h3>{e(heading)}</h3>\n" + dl([
            ("Family", family), ("Classes in this model", ", ".join(ctx.shown(m) for m in members)),
            ("Summed probability", text.fmt_prob(p, t))]))
    parts.append("</section>")
    return "\n".join(parts)


def interpretation(case: Case, res: text.Result, ctx: Context) -> str:
    """The interpretation text of the reported entity, then the caveats."""
    it = text.interpret(case, res, ctx)
    parts = ["<section>", "<h2>Interpretation</h2>", f"<h3>{e(it.heading)}</h3>"]
    parts += [f"<p>{e(p)}</p>" for p in it.paragraphs]
    if it.confirm:
        parts.append(f'<p><span class="muted">Recommended confirmation:</span> {e(it.confirm)}</p>')
    if it.note:
        parts.append(f'<p><span class="muted">Note:</span> {e(it.note)}</p>')
    if it.caveats or it.not_in_model:
        parts.append('<h3>Caveats</h3>\n<div class="caveats">')
        parts += [f"<p>{e(c)}</p>" for c in it.caveats]
        if it.not_in_model:
            items = "".join(f"<li>{e(x)}</li>" for x in it.not_in_model)
            parts.append(f"<p>{e(it.not_in_model_intro)}</p>\n<ul>{items}</ul>")
        parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def evidence(case: Case, res: text.Result, ctx: Context) -> str:
    """What cross-validation showed at a similar coverage; empty without the model's tables."""
    ev = text.cv_evidence(case, res, ctx)
    if ev is None:
        return ""
    m = ctx.model
    scored = (f"; this sample was scored by the average of all {m.n_networks} networks"
              if m.n_networks and m.n_networks > 1 else "")
    source = (f"Source: condition {ev.condition} in cv_metrics_by_condition.csv"
              + (" and cv_recall_by_class.csv" if ctx.cv.recall is not None else "")
              + f" of the model folder {ctx.cv.folder.name}. In cross-validation each training array was scored by "
              f"the network of its own fold, which had not seen it{scored}.")
    return ("<section>\n<h2>Evidence from cross-validation</h2>\n" + "\n".join(f"<p>{e(s)}</p>" for s in ev.sentences)
            + f'\n<p class="small muted">{e(source)}</p>\n</section>')


def _number(value, decimals: int = 0) -> str:
    """A value from a karyotype file as text: numbers with thousands separators, flags as yes or no."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else ("yes" if value else "no")
    if isinstance(value, (int, float)) and np.isfinite(value):
        if decimals:
            return f"{value:,.{decimals}f}"
        return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


def _items(value) -> List[str]:
    """A list of flags or warnings from a karyotype file (a single text counts as one item)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if isinstance(v, dict):                     # {"code": ..., "text": ...} from the copy-number module
                v = v.get("text") or v.get("code")
            if v is not None and str(v).strip():
                out.append(str(v))
        return out
    return [str(value)] if str(value).strip() else []


def _alternatives(d: dict) -> str:
    """
    The other readings of the profile that the copy-number step noted (in files without that mark, the best fit
    at each other baseline whose cost is within 20 of the chosen one), in words: chromosome count and ploidy
    class, baseline, leukaemic fraction and the difference in cost.
    """
    fit = d.get("fit") if isinstance(d.get("fit"), dict) else {}
    chosen = d.get("baseline_copies")
    alternatives = [a for a in fit.get("alternatives") or [] if isinstance(a, dict)]
    marked = any("noted" in a for a in alternatives)
    words = []
    for a in alternatives:
        delta = a.get("delta")
        if a.get("baseline") == chosen or not isinstance(delta, (int, float)):
            continue
        if (marked and not a.get("noted")) or (not marked and delta > 20):
            continue
        text = f"{a['modal_number']} chromosomes ({a['ploidy_class']}), " if "modal_number" in a else ""
        words.append(f"{text}baseline {a['baseline']} copies, leukaemic fraction {a['purity']:.2f} "
                     f"(cost higher by {delta:.1f})")
    return "; ".join(words)


def _changed_arms(d: dict) -> str:
    """'5q x1 (50% of the arm), 7p x1, 7q x1' from the arm table of the copy-number module."""
    words = []
    for a in d.get("arms") or []:
        if not isinstance(a, dict) or a.get("call") in (None, "normal"):
            continue
        share = a.get("share_of_arm")
        part = f" ({share:.0%} of the arm)" if isinstance(share, (int, float)) and share < 0.9 else ""
        words.append(f"{a['arm']} x{a['copy_number']}{part}")
    return ", ".join(words)


def _gene_list(genes, limit: int = 6) -> str:
    """At most `limit` genes of interest, then how many more there are."""
    if not isinstance(genes, (list, tuple)):
        return "" if genes is None else str(genes)
    genes = [str(g) for g in genes]
    if len(genes) <= limit:
        return ", ".join(genes)
    return ", ".join(genes[:limit]) + f" and {len(genes) - limit} more"


def _detection_text(limit) -> str:
    """From {'5 Mb': 0.45, '20 Mb': 0.22}: 'a one-copy gain of 5 Mb is detectable in at least 45% of cells, one of
    20 Mb in at least 22%'; other values as text."""
    if isinstance(limit, dict):
        parts = [(k, v) for k, v in limit.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not parts:
            return ""
        words = [f"a one-copy gain of {parts[0][0]} is detectable when at least {parts[0][1]:.0%} of cells carry it"]
        words += [f"one of {k} when at least {v:.0%} do" for k, v in parts[1:]]
        return ", ".join(words)
    return f"detection limit {limit}"


def karyotype(case: Case, res: text.Result, ctx: Context) -> str:
    """The copy-number result of the sample, whatever of it the file holds; empty without one."""
    kt: Karyotype = case.karyotype
    if kt is None:
        return ""
    d = kt.data
    parts = ["<section>", "<h2>Copy-number karyotype</h2>"]
    if kt.error:
        parts.append(f"<p>The copy-number result could not be read: {e(kt.error)}.</p>")
    if d.get("iscn"):
        breakable = ",<wbr>".join(e(part) for part in str(d["iscn"]).split(","))   # lines break after commas
        parts.append(f'<div class="iscn">{breakable}</div>')
    rows = []
    for key, label in (("karyotype", "Karyotype (whole chromosomes)"), ("modal_number", "Modal chromosome number"),
                       ("ploidy_class", "Ploidy class"), ("sex", "Sex")):
        if d.get(key) not in (None, ""):
            value = _number(d[key]) if key == "modal_number" else str(d[key])
            if key == "sex" and d.get("sex_source"):
                value += f" ({d['sex_source']})"
            rows.append((label, value))
    if isinstance(d.get("leukaemic_fraction"), (int, float)) and not isinstance(d.get("leukaemic_fraction"), bool):
        rows.append(("Leukaemic fraction (from copy number)", f"{d['leukaemic_fraction']:.2f}"))
        alternatives = _alternatives(d)
        if alternatives:
            rows.append(("Other ploidies that fit", alternatives))
    elif "leukaemic_fraction" in d:
        rows.append(("Leukaemic fraction (from copy number)", "not estimable"))
    arms = _changed_arms(d)
    if arms:
        rows.append(("Arms changed (majority copy number)", arms))
    if rows:
        parts.append(dl(rows))
    if d.get("confirmation"):
        parts.append(f'<p class="small muted">{e(d["confirmation"])}</p>')
    lists = [("Flags", _items(d.get("flags"))), ("Warnings", _items(d.get("warnings")))]
    shown = [(title, items) for title, items in lists if items]
    if shown:
        blocks = "".join(f"<div><h3>{title}</h3><ul>{''.join(f'<li>{e(i)}</li>' for i in items)}</ul></div>"
                         for title, items in shown)
        parts.append(f'<div class="two">{blocks}</div>' if len(shown) == 2 else blocks)
    events = d.get("events")
    if isinstance(events, list) and events:
        events = [ev for ev in events if isinstance(ev, dict)]
        module = any("level" in ev for ev in events)
        if module:                                      # the copy-number module states the extent and the bands
            events = [dict(ev, genes=_gene_list(ev.get("genes")) if ev.get("level") != "whole" else "")
                      for ev in events]
        cols = [(k, label) for k, label in EVENT_COLUMNS if any(ev.get(k) not in (None, "") for ev in events)]
        if module:
            cols = [(k, label) for k, label in cols if k not in ("whole_chromosome", "start", "end")]
        numeric = {"start", "end", "size_mb", "copy_number", "cell_fraction"}
        num = ' class="num"'
        head = "".join(f"<th{num if k in numeric else ''}>{label}</th>" for k, label in cols)
        body = []
        for ev in events:
            cells = []
            for k, _ in cols:
                v = ev.get(k)
                if k == "cell_fraction":
                    v = "not estimated" if v is None else _number(v, 2) + (" (subclonal)"
                                                                            if ev.get("subclonal") else "")
                elif k in ("start", "end", "copy_number"):
                    v = _number(v)
                elif k == "size_mb":
                    v = _number(v, 1)
                elif k == "level":
                    v = LEVEL_WORDS.get(v, v)
                elif k == "genes":
                    v = ", ".join(v) if isinstance(v, list) else ("" if v is None else str(v))
                elif k == "whole_chromosome":
                    v = _number(v) if isinstance(v, bool) else ("" if v is None else str(v))
                cells.append(f"<td{num if k in numeric else ''}>{e(v)}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        parts.append(f'<h3>Events</h3>\n<div class="table-wrap"><table class="events"><thead><tr>{head}</tr></thead>'
                     f"<tbody>{''.join(body)}</tbody></table></div>")
    elif isinstance(events, list):
        parts.append("<p>No gains or losses were reported.</p>")
    if kt.figure:
        alt = e(f"Copy-number profile of {case.sample}")
        caption = kt.figure_source
        if "fit" in d or "method" in d:                 # drawn by the copy-number module
            caption = ("Read depth along the genome against the reference of other samples: grey points are bins, "
                       "lines are segments, red for gains and blue for losses, and zero is two copies in the "
                       "leukaemic cells. The right-hand axis gives the copy number in the leukaemic cells at the "
                       f"fitted fraction. File {kt.figure_source}.")
        parts.append(f'<figure><img class="karyotype" src="{kt.figure}" alt="{alt}">'
                     f"<figcaption>{e(caption)}</figcaption></figure>")
    if kt.chromosome_figure:
        alt = e(f"Copy-number profile of {case.sample}, chromosome by chromosome")
        caption = ("The same profile chromosome by chromosome, with the bands beneath, the centromere as a dashed line, "
                   "zero at two copies and the other levels of 1 to 4 copies as dotted lines; changed arms are named "
                   f"in each panel. File {kt.chromosome_figure_source}.")
        parts.append(f'<figure><img class="karyotype" src="{kt.chromosome_figure}" alt="{alt}">'
                     f"<figcaption>{e(caption)}</figcaption></figure>")
    reported = case.top if case.level in ("subtype", "no_subtype") else None
    note = ctx.interpretation["classes"].get(reported, {}).get("copy_number_note") if reported else ""
    if note:
        parts.append(f'<p><span class="muted">For {e(ctx.shown(reported))}:</span> {e(note)}</p>')
    qc = d.get("qc") if isinstance(d.get("qc"), dict) else {}
    qc_parts = []
    if qc.get("bin_size") not in (None, ""):
        size = qc["bin_size"]
        qc_parts.append(f"bin size {_number(size)} base pairs" if isinstance(size, (int, float)) and
                        not isinstance(size, bool) else f"bin size {size}")
    if qc.get("median_reads_per_bin") not in (None, ""):
        qc_parts.append(f"median reads per bin {_number(qc['median_reads_per_bin'], 1)}")
    if qc.get("noise_log2") not in (None, ""):
        qc_parts.append(f"bin-to-bin noise {_number(qc['noise_log2'], 2)} (log2 ratio)")
    if qc.get("detection_limit") not in (None, ""):
        text_limit = _detection_text(qc["detection_limit"])
        if text_limit:
            qc_parts.append(text_limit)
    if qc_parts:
        parts.append(f'<p class="small muted">Quality: {e("; ".join(qc_parts))}.</p>')
    if not kt.error and not (rows or shown or d.get("iscn") or isinstance(events, list) or kt.figure):
        parts.append("<p>The copy-number file holds no results for this sample.</p>")
    parts.append(f'<p class="small muted">Source: {e(kt.source)}.</p>')
    parts.append("</section>")
    return "\n".join(parts)


def _sequencing_rows(case: Case) -> List[Tuple[str, str]]:
    """Whole-genome reads, read lengths and depth, from the copy-number result of the sample."""
    kt = case.karyotype
    seq = kt.data.get("sequencing") if kt is not None and isinstance(kt.data.get("sequencing"), dict) else {}
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)          # noqa: E731
    rows = []
    if number(seq.get("reads")):
        rows.append(("Whole-genome reads used", f"{int(seq['reads']):,} (primary alignments, mapping quality "
                                                "20 or more)"))
    if number(seq.get("read_length_median")):
        text = f"median {int(seq['read_length_median']):,} bases"
        if number(seq.get("read_length_n50")):
            text += f", N50 {int(seq['read_length_n50']):,}"
        rows.append(("Read length", text))
    if number(seq.get("mean_depth")):
        text = f"{seq['mean_depth']:.2f}x"
        if number(seq.get("bases_gb")):
            text += f" ({seq['bases_gb']:.2f} Gb of reads over the genome)"
        rows.append(("Genome depth", text))
    return rows


def input_quality(case: Case, ctx: Context) -> str:
    """The input file, its CpGs and their coverage, the note of predict.py, the whole-genome depth when a
    copy-number result gives it, and the sample information."""
    n_cpgs = ctx.model.n_cpgs
    of = f" of {text.fmt_int(n_cpgs)}" if n_cpgs else ""
    rule = text.DUPLICATE_RULES.get(ctx.duplicate_probes or "", "")
    if np.isfinite(case.n_repeated_probes):
        repeated = text.fmt_int(case.n_repeated_probes)
        if case.n_repeated_probes > 0 and rule:
            repeated += f" ({rule})"
    else:
        repeated = "not recorded"
    rows = [("Input file", f'<span class="mono">{e(case.file)}</span>'),
            ("CpG columns in the file", e(text.fmt_int(case.n_columns))),
            ("Model CpGs among them", e(f"{text.fmt_int(case.n_matched)}{of}")),
            ("Model CpGs observed", e(f"{text.fmt_int(case.n_observed)}{of} ({text.fmt_pct(case.coverage_pct)})")),
            ("Repeated probe IDs combined", e(repeated))]
    if case.note:
        rows.append(("Note", e(case.note)))
    rows += [(label, e(value)) for label, value in _sequencing_rows(case)]
    parts = ["<section>", "<h2>Input and quality</h2>", dl(rows, raw=True)]
    if ctx.has_sheet:
        parts.append("<h3>Sample information</h3>")
        if case.info is None:
            parts.append("<p>The sample sheet has no row for this sample.</p>")
        elif not case.info:
            parts.append("<p>The sample sheet gives no further information for this sample.</p>")
        else:
            parts.append(dl(case.info))
    parts.append("</section>")
    return "\n".join(parts)


def about(ctx: Context) -> str:
    """How the classifier works, and which interpretation text and hierarchy were used."""
    sentences = text.about_text(ctx)
    made = []
    if ctx.prediction_created:
        made.append(f"Predictions made on {recorded_time(ctx.prediction_created)}")
    source = ctx.hierarchy_source
    made.append(f"class hierarchy: {Path(source).name if source.endswith('.json') else source}")
    return ("<section>\n<h2>About this result</h2>\n<p>" + e(" ".join(sentences)) + "</p>\n"
            f'<p class="small muted">{e(text.interpretation_source(ctx))} {e("; ".join(made))}.</p>\n</section>')


def case_page(case: Case, res: text.Result, ctx: Context) -> str:
    """The complete page of one sample."""
    body = "\n".join(part for part in (
        '<div class="page">',
        '<nav class="nav"><a href="../index.html">All samples</a></nav>',
        header(case, ctx),
        "<main>",
        result_panel(case, res, ctx),
        probabilities(case, ctx),
        interpretation(case, res, ctx),
        evidence(case, res, ctx),
        karyotype(case, res, ctx),
        input_quality(case, ctx),
        about(ctx),
        "</main>",
        f'<footer class="report">{e(RUO)} Sample {e(case.sample)}. Report generated {e(when(ctx.generated))} with '
        f'SPARSH {e(ctx.report_version)}.</footer>',
        "</div>") if part)
    return document(f"{case.sample}: {ctx.title}", style.case_css(case.sample), body)


# ---------------------------------------------------------------------------------------------- index page
INDEX_SCRIPT = """(function () {
  var table = document.getElementById('samples'), body = table.tBodies[0];
  var heads = table.tHead.rows[0].cells, input = document.getElementById('filter');
  var count = document.getElementById('count'), total = body.rows.length;
  function value(row, i, numeric) {
    var cell = row.cells[i], v = cell.getAttribute('data-sort');
    if (v === null) { v = cell.textContent.trim(); }
    if (!numeric) { return v; }
    v = parseFloat(v);
    return isNaN(v) ? -Infinity : v;
  }
  Array.prototype.forEach.call(heads, function (th, i) {
    var button = th.querySelector('button');
    if (!button) { return; }
    button.addEventListener('click', function () {
      var numeric = th.getAttribute('data-type') === 'num';
      var dir = th.getAttribute('aria-sort') === 'ascending' ? -1 : 1;
      Array.prototype.forEach.call(heads, function (h) { h.setAttribute('aria-sort', 'none'); });
      th.setAttribute('aria-sort', dir === 1 ? 'ascending' : 'descending');
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {
        var x = value(a, i, numeric), y = value(b, i, numeric);
        if (numeric) { return x === y ? 0 : (x < y ? -dir : dir); }
        return dir * x.localeCompare(y, undefined, {numeric: true, sensitivity: 'base'});
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase(), shown = 0;
    Array.prototype.forEach.call(body.rows, function (row) {
      var hit = !q || row.textContent.toLowerCase().indexOf(q) !== -1;
      row.hidden = !hit;
      if (hit) { shown += 1; }
    });
    count.textContent = shown + ' of ' + total + ' samples shown';
  });
})();"""


def index_page(cases: List[Case], results: Dict[str, text.Result], ctx: Context) -> str:
    """The table of all samples, with sorting and filtering by the inline script above."""
    t = ctx.threshold
    levels = pd.Series([results[c.sample].level for c in cases]).value_counts()
    counts = "; ".join(f"{text.LEVEL_COUNTS[lev]} {int(levels.get(lev, 0))}" for lev in LEVEL_ORDER
                       if levels.get(lev, 0))
    rows_meta = [("Report generated", e(when(ctx.generated))), ("SPARSH version", e(version_line(ctx))),
                 ("Model", e(model_line(ctx))),
                 ("Samples", e(f"{len(cases)} (reported at level: {counts})")),
                 ("Predictions", f'<span class="mono">{e(ctx.predictions_dir.name)}/predictions.csv</span>')]
    heads = [("Sample", "text", ""), ("Coverage", "num", "num"), ("Reported result", "text", ""),
             ("Reporting level", "num", ""), ("Confidence", "num", "num"), ("Lineage", "text", ""),
             ("Flags", "text", "")]
    th = "".join(f'<th scope="col" data-type="{kind}" aria-sort="none"' + (f' class="{cls}"' if cls else "")
                 + f'><button type="button">{label}</button></th>' for label, kind, cls in heads)
    body = []
    for c in cases:
        r = results[c.sample]
        conf = e(text.fmt_prob(r.confidence, t)) if r.confidence is not None else '<span class="muted">none</span>'
        lineage = (f'<span class="lin">{chip(ctx.lineage_colour(r.lineage), 10)}{e(r.lineage)}</span>'
                   if r.lineage else '<span class="muted">not determined</span>')
        flags = "".join(f'<span class="flag">{e(f)}</span>' for f in r.flags)
        body.append(
            "<tr>"
            f'<td data-sort="{e(c.sample)}"><a href="cases/{e(c.page)}">{e(c.sample)}</a></td>'
            f'<td class="num" data-sort="{c.coverage_pct:.4f}">{e(text.fmt_pct(c.coverage_pct))}</td>'
            f'<td class="wrap">{e(r.result)}</td>'
            f'<td data-sort="{LEVEL_ORDER.get(r.level, 9)}">{e(text.LEVEL_NAMES.get(r.level, r.level))}</td>'
            f'<td class="num" data-sort="{-1 if r.confidence is None else r.confidence:.6f}">{conf}</td>'
            f'<td data-sort="{e(r.lineage or "")}">{lineage}</td>'
            f"<td>{flags}</td>"
            "</tr>")
    note = (f"Confidence is the probability of the reported result: of the class for a subtype, summed over the "
            f"classes of the family or lineage otherwise. Flags: low coverage (under 5% of the model's CpGs "
            f"observed), below threshold (the most probable class is below {text.fmt_threshold(t)}), not callable "
            "(nothing reported). Select a column heading to sort; type in the box to filter.")
    html = f"""<div class="page">
<header class="report">
<h1>{e(ctx.title)}</h1>
<p class="ruo">{e(RUO)}</p>
{dl(rows_meta, raw=True)}
</header>
<main>
<div class="tools"><label for="filter">Filter</label><input id="filter" type="search" autocomplete="off"
placeholder="Sample, result, lineage or flag"><span id="count" class="small muted"
aria-live="polite">{len(cases)} of {len(cases)} samples shown</span></div>
<div class="table-wrap">
<table class="samples" id="samples">
<thead><tr>{th}</tr></thead>
<tbody>
{chr(10).join(body)}
</tbody>
</table>
</div>
<p class="small muted">{e(note)}</p>
</main>
<footer class="report">{e(RUO)} Report generated {e(when(ctx.generated))} with SPARSH {e(ctx.report_version)}.</footer>
</div>"""
    return document(ctx.title, style.index_css(), html, INDEX_SCRIPT)


# ---------------------------------------------------------------------------------------------- summary and writing
def summary_rows(cases: List[Case], results: Dict[str, text.Result], ctx: Context) -> pd.DataFrame:
    """One row per sample: the fields of the result panel and the first sentence of the interpretation."""
    rows = []
    for c in cases:
        r = results[c.sample]
        it = text.interpret(c, r, ctx)
        rows.append({
            "sample": c.sample, "page": f"cases/{c.page}", "lineage": r.lineage or "",
            "lineage_probability": r.lineage_prob if r.lineage else np.nan,
            "result": r.result, "result_name": r.result_name, "result_detail": r.result_note,
            "confidence": r.confidence if r.confidence is not None else np.nan,
            "threshold": ctx.threshold, "confidence_text": r.confidence_text,
            "reported_level": r.level, "level_text": r.level_text,
            "most_likely_class": ctx.shown(c.top), "most_likely_probability": c.confidence,
            "coverage_pct": c.coverage_pct, "flags": "; ".join(r.flags),
            "interpretation_headline": it.headline,
        })
    return pd.DataFrame(rows)


def write_report(ctx: Context, cases: List[Case], out: Path) -> Dict[str, text.Result]:
    """Write the case pages, the index page and report_summary.csv into out (which must exist)."""
    out = Path(out)
    (out / "cases").mkdir(exist_ok=True)
    results = {c.sample: text.describe_result(c, ctx) for c in cases}
    written = set()
    for c in cases:
        (out / "cases" / c.page).write_text(case_page(c, results[c.sample], ctx), encoding="utf-8")
        written.add(c.page)
    (out / "index.html").write_text(index_page(cases, results, ctx), encoding="utf-8")
    summary_rows(cases, results, ctx).to_csv(out / "report_summary.csv", index=False, float_format="%.5f")
    stale = sorted(f.name for f in (out / "cases").glob("*.html") if f.name not in written)
    if stale:
        print(f"Note: {len(stale)} page(s) in {out / 'cases'} are from an earlier report and not part of this one: "
              f"{stale[:5]}")
    return results
