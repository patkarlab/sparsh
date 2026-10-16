"""
Sentences of the per-case reports.

Every sentence is built from the data of one sample (report.inputs.Case) and from the context shared by all pages
(report.inputs.Context): the reported result and its level, the interpretation text of the reported entity, the
caveats that follow from the data, the evidence from the model's cross-validation and the explanation of the method.
Nothing here writes HTML; report/pages.py escapes and places the text.

Numbers. Probabilities are shown with two decimals. A probability just below the threshold is never rounded up to
it (0.8996 is shown as 0.8996, not 0.90, when the threshold is 0.90), and probabilities that would round to 1.00 or
0.00 are shown as "> 0.99" and "< 0.01", because a rounded 1.00 or 0.00 would read as certainty.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .inputs import LOW_COVERAGE, NORMAL_CLASS, NORMAL_LINEAGE, Case, Context

LEVEL_NAMES = {"subtype": "Subtype", "no_subtype": "Lineage, no specific subtype", "family": "Family",
               "provisional": "Provisional subtype", "lineage": "Lineage", "none": "Not reported"}
LEVEL_COUNTS = {"subtype": "subtype", "no_subtype": "lineage without a specific subtype", "family": "family",
                "provisional": "provisional subtype", "lineage": "lineage", "none": "not reported"}
DUPLICATE_RULES = {"mean": "mean of the copies with a value", "majority": "majority of the copies with a value",
                   "first": "first column only",
                   "error": "files with repeated probes refused"}


# ---------------------------------------------------------------------------------------------- numbers
def fmt_threshold(t: float) -> str:
    """0.9 -> '0.90'; a threshold with more decimals keeps them."""
    s = f"{t:.2f}"
    return s if abs(float(s) - t) < 1e-9 else f"{t:.4f}".rstrip("0")


def fmt_prob(p, threshold: Optional[float] = None) -> str:
    """
    A probability with two decimals, or more where two would put it on the other side of the threshold; values
    that would round to 0 or 1 are shown as "< 0.01" and "> 0.99" (see the module docstring).
    """
    if p is None or not np.isfinite(p):
        return "not available"
    p = min(max(float(p), 0.0), 1.0)
    if p >= 0.995 and (threshold is None or threshold <= 0.99):
        return "> 0.99"
    if p < 0.005 and (threshold is None or threshold >= 0.01):
        return "< 0.01"
    for decimals in (2, 3, 4, 5):
        s = f"{p:.{decimals}f}"
        shown = float(s)
        crosses = threshold is not None and (p >= threshold) != (shown >= threshold)
        certain = (shown == 1.0 and p < 1.0) or (shown == 0.0 and p > 0.0)
        if not crosses and not certain:
            return s
    if threshold is not None and p < threshold:
        return f"just below {fmt_threshold(threshold)}"
    return "> 0.99999" if p > 0.5 else "< 0.00001"


def fmt_pct(value: float) -> str:
    """A percentage already on the 0-100 scale: 27.5 -> '27.5%'."""
    if not np.isfinite(value):
        return "not available"
    if 0 < value < 0.1:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def fmt_fraction(value: float) -> str:
    """A share on the 0-1 scale as a percentage: 0.85 -> '85%', 0.9861 -> '98.6%'."""
    pct = 100.0 * value
    return f"{pct:.0f}%" if abs(pct - round(pct)) < 0.05 else f"{pct:.1f}%"


def fmt_int(value) -> str:
    """A count with thousands separators: 354551 -> '354,551'."""
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "not available"


def first_sentence(text: str) -> str:
    """The first sentence of a text (up to the first full stop or semicolon followed by a space)."""
    text = " ".join(str(text).split())
    for end in (". ", "; "):
        if end in text:
            return text.split(end)[0] + "."
    return text


def join_words(items: List[str], word: str = "and") -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


# ---------------------------------------------------------------------------------------------- probabilities
def top_classes(case: Case, ctx: Context, n: int = 5) -> List[Tuple[str, float]]:
    """The n most probable classes (internal names) with their probabilities, most probable first."""
    order = np.argsort(-case.probs, kind="stable")[:n]
    return [(ctx.classes[i], float(case.probs[i])) for i in order]


def lineage_sums(case: Case, ctx: Context, minimum: float = 0.01) -> List[Tuple[str, float]]:
    """Summed probability of every lineage of the hierarchy that reaches `minimum`, largest first."""
    sums: Dict[str, float] = {}
    for cls, p in zip(ctx.classes, case.probs):
        lin = ctx.hierarchy.lineage_of[cls]
        sums[lin] = sums.get(lin, 0.0) + float(p)
    return sorted(((k, v) for k, v in sums.items() if v >= minimum), key=lambda kv: -kv[1])


def family_sum(case: Case, ctx: Context, family: str) -> Tuple[float, List[str]]:
    """Summed probability of a family's classes in this model, and those classes."""
    members = ctx.hierarchy.families.get(family, [])
    return float(sum(case.probs[ctx.class_index(m)] for m in members)), members


# ---------------------------------------------------------------------------------------------- the result
@dataclass
class Result:
    """What the result panel of a case page shows, and what report_summary.csv and the index repeat."""
    lineage: Optional[str]
    lineage_prob: float                   # summed probability of that lineage (or of the most probable one)
    lineage_note: str
    result: str
    result_name: str
    result_note: str
    result_class: Optional[str]           # the reported class (internal name) at subtype level
    confidence: Optional[float]
    confidence_text: str
    level: str
    level_text: str
    most_likely: str
    flags: List[str] = field(default_factory=list)


def describe_result(case: Case, ctx: Context) -> Result:
    """The reported entity, its probability and its level, in words (see the module docstring for numbers)."""
    t = ctx.threshold
    ts = fmt_threshold(t)
    h = ctx.hierarchy
    top = ctx.shown(case.top)
    top_p = fmt_prob(case.confidence, t)
    threshold_text = f"(reporting threshold {ts})"
    level = case.level
    entry = ctx.interpretation["classes"].get(case.top, {})
    most_likely = ""
    result_class = None
    if level == "subtype":
        lineage = h.lineage_of[case.top]
        result, result_name, result_note = top, entry.get("name", ""), ""
        confidence, result_class = case.confidence, case.top
        level_text = "Reported at subtype level."
    elif level == "no_subtype":
        lineage = h.lineage_of[case.top]
        result, result_name, result_note = f"{lineage}, no specific subtype", entry.get("name", ""), ""
        confidence = case.confidence
        level_text = (f"Reported at lineage level without a specific subtype: the most probable class, {top}, "
                      f"reached the threshold, and it stands for {lineage} without a specific subtype.")
        most_likely = f"Most likely class: {top} ({top_p})."
    elif level == "family":
        family = case.family
        members = h.families.get(family, [])
        lineage = h.lineage_of[members[0]] if members else case.lineage
        result = family
        result_name = ctx.interpretation["families"].get(family, {}).get("name", "")
        result_note = ("Classes: " + ", ".join(ctx.shown(m) for m in members)) if members else ""
        confidence = case.family_prob
        level_text = "Reported at family level: no single subtype reached the threshold."
        if case.top in members:
            most_likely = f"Most likely member of the family: {top} ({top_p}), below the reporting threshold of {ts}."
        else:
            most_likely = f"Most likely class: {top} ({top_p}), below the reporting threshold of {ts}."
    elif level == "provisional":
        cs = fmt_threshold(ctx.caution_threshold) if ctx.caution_threshold is not None else "the caution threshold"
        lineage = h.lineage_of[case.top]
        result, result_name = top, entry.get("name", "")
        result_note = f"Caution: confidence between {cs} and {ts}"
        confidence, result_class = case.confidence, case.top
        level_text = ""                   # the label of the result says it all
    elif level == "lineage":
        lineage = case.lineage
        result, result_note = f"{lineage}, subtype undetermined", ""
        result_name = ctx.interpretation["lineages"].get(lineage, {}).get("name", "")
        confidence = case.lineage_prob
        level_text = ("Reported at lineage level: neither a single subtype nor a family of related subtypes "
                      "reached the threshold.")
        most_likely = f"Most likely class: {top} ({top_p}), below the reporting threshold of {ts}."
    else:
        lineage, result, result_name, confidence = None, "Not callable", "", None
        if not case.coverage_ok:
            result_note = "Too few CpGs observed"
            level_text = (f"Not reported: too few of the model's CpGs were observed ({fmt_pct(case.coverage_pct)}; "
                          f"at least {fmt_pct(100.0 * ctx.min_coverage)} are needed).")
            below = f", below the reporting threshold of {ts}" if case.confidence < t else ""
            most_likely = (f"Most likely class: {top} ({top_p}){below}; not reported, because too few CpGs were "
                           "observed.")
        else:
            result_note = "Nothing reached the threshold"
            level_text = "Not reported: no subtype, family of subtypes or lineage reached the threshold."
            most_likely = f"Most likely class: {top} ({top_p}), below the reporting threshold of {ts}."

    if confidence is not None and level == "provisional":
        confidence_text = (f"{fmt_prob(confidence, t)} (provisional range "
                           f"{fmt_threshold(ctx.caution_threshold)} to {ts})")
    elif confidence is not None:
        confidence_text = f"{fmt_prob(confidence, t)} {threshold_text}"
    elif not case.coverage_ok:
        confidence_text = "Not assessed"
    else:
        confidence_text = f"None reached the threshold of {ts}"
    if lineage is not None:
        lineage_prob = float(sum(p for c, p in zip(ctx.classes, case.probs) if h.lineage_of[c] == lineage))
        full = ctx.interpretation["lineages"].get(lineage, {}).get("name", "")
        lineage_note = (f"{full}; summed probability {fmt_prob(lineage_prob, t)}" if full and full != lineage
                        else f"Summed probability {fmt_prob(lineage_prob, t)}")
    elif not case.coverage_ok:
        lineage_prob = case.lineage_prob
        lineage_note = "Not assessed: too few CpGs observed"
    else:
        lineage_prob = case.lineage_prob
        lineage_note = f"Most probable: {case.lineage} ({fmt_prob(case.lineage_prob, t)}), below the threshold"

    flags = []
    if case.coverage_pct < 100.0 * LOW_COVERAGE:
        flags.append("low coverage")
    if level == "provisional":
        flags.append("provisional")
    elif case.coverage_ok and case.confidence < t:
        flags.append("below threshold")
    if level == "none":
        flags.append("not callable")
    return Result(lineage=lineage, lineage_prob=lineage_prob, lineage_note=lineage_note, result=result,
                  result_name=result_name,
                  result_note=result_note, result_class=result_class, confidence=confidence,
                  confidence_text=confidence_text, level=level, level_text=level_text, most_likely=most_likely,
                  flags=flags)


# ---------------------------------------------------------------------------------------------- interpretation
@dataclass
class Interpretation:
    """The interpretation text of the reported entity, and the caveats generated from the data."""
    heading: str
    paragraphs: List[str]
    confirm: str
    note: str
    caveats: List[str]
    not_in_model_intro: str
    not_in_model: List[str]
    headline: str


def _class_text(ctx: Context, cls: str) -> Tuple[str, str, str, str]:
    """Name, summary, confirmation and note of a class, with neutral text where the file has none."""
    entry = ctx.interpretation["classes"].get(cls, {})
    shown = ctx.shown(cls)
    name = entry.get("name") or shown
    summary = entry.get("summary") or (f"The methylation profile of this sample matches the class {shown} of the "
                                       "training arrays. No interpretation text has been written for this class.")
    return name, summary, entry.get("confirm", ""), entry.get("note", "")


def interpret(case: Case, res: Result, ctx: Context) -> Interpretation:
    """The text of the interpretation section: the entity's own text first, then the caveats."""
    text = ctx.interpretation
    h = ctx.hierarchy
    t = fmt_threshold(ctx.threshold)
    confirm = note = ""
    if case.level in ("subtype", "provisional"):
        heading, summary, confirm, note = _class_text(ctx, case.top)
        if case.level == "provisional":
            cs = fmt_threshold(ctx.caution_threshold) if ctx.caution_threshold is not None else "the caution threshold"
            heading = f"{heading} (Caution: confidence between {cs} and {t})"
            confirm = ""
        paragraphs = [summary]
    elif case.level == "no_subtype":
        if case.top in text["classes"]:
            heading, summary, confirm, note = _class_text(ctx, case.top)
        else:
            entry = text["lineages"].get(res.lineage, {})
            heading = f"{entry.get('name') or res.lineage}, no specific subtype"
            summary = (f"The most probable class, {ctx.shown(case.top)}, groups {res.lineage} without a specific "
                       "methylation subtype.")
        paragraphs = [summary]
    elif case.level == "family":
        entry = text["families"].get(case.family, {})
        members = [ctx.shown(m) for m in h.families.get(case.family, [])]
        heading = entry.get("name") or case.family
        which = f"the related subtypes {join_words(members)}" if members else "a family of related subtypes"
        summary = entry.get("summary") or (
            f"The summed probability of {which} reached the threshold. No interpretation text has been written for "
            "this family.")
        confirm = entry.get("confirm", "")
        paragraphs = [summary]
    elif case.level == "lineage":
        entry = text["lineages"].get(case.lineage, {})
        heading = f"{entry.get('name') or case.lineage}, subtype undetermined"
        summary = entry.get("undetermined") or (
            f"The methylation profile is that of {case.lineage}, but no subtype or family of related subtypes "
            "reached the threshold.")
        paragraphs = [summary]
    else:
        heading = "Not callable"
        key = "coverage" if not case.coverage_ok else "none"
        default = ("Too few of the model's CpGs were observed to classify this sample." if key == "coverage" else
                   "No subtype, family of subtypes or lineage reached the reporting threshold.")
        paragraphs = [text["levels"].get(key) or default]

    caveats = []
    if case.coverage_ok and case.coverage_pct < 100.0 * LOW_COVERAGE:
        observed = f"Only {fmt_pct(case.coverage_pct)} of the model's CpGs were observed in this sample."
        caution = "" if case.level == "none" else ", and the result should be read with caution"
        cv = ctx.cv
        if cv is None:
            caveats.append(f"{observed} This is below {fmt_fraction(LOW_COVERAGE)}, the lower end of the range of "
                           "coverage at which the model is tested in cross-validation, so its performance at this "
                           f"coverage has not been measured{caution}.")
        elif case.coverage_pct / 100.0 < float(cv.binary["fraction"].min()):
            lo, hi = float(cv.binary["fraction"].min()), float(cv.binary["fraction"].max())
            caveats.append(f"{observed} This is below the range of coverage at which the model was tested in "
                           f"cross-validation ({fmt_fraction(lo)} to {fmt_fraction(hi)} of its CpGs), so its "
                           f"performance at this coverage has not been measured{caution}.")
        else:
            caveats.append(f"{observed} This is below {fmt_fraction(LOW_COVERAGE)}, a low coverage for the model"
                           f"{caution}; the evidence from cross-validation below describes how the model performed at "
                           "a similar coverage.")
    if case.coverage_ok and case.confidence < ctx.threshold and case.level != "provisional":
        caveats.append(f"The most probable class, {ctx.shown(case.top)}, has a probability of "
                       f"{fmt_prob(case.confidence, ctx.threshold)}, below the reporting threshold of {t}. It is "
                       "indicative only and is not a result.")
    if case.level == "family":
        members = [ctx.shown(m) for m in h.families.get(case.family, [])]
        listed = f" ({join_words(members)})" if members else ""
        caveats.append(f"The subtype could not be resolved. The summed probability of the classes of this family"
                       f"{listed} reached the threshold, but none of them did on its own. A family "
                       "groups subtypes whose methylation profiles overlap; the result places the sample among them "
                       "without deciding between them, so the subtype has to be established by other methods.")
    elif case.level == "lineage":
        caveats.append(f"The subtype could not be resolved. The probability is spread over several classes, and "
                       f"only the summed probability of the {case.lineage} lineage reached the threshold; the "
                       "subtype has to be established by other methods.")
    reported_normal = ((case.level in ("subtype", "provisional") and case.top == NORMAL_CLASS)
                       or (case.level in ("subtype", "no_subtype", "provisional", "lineage")
                           and res.lineage == NORMAL_LINEAGE))
    if reported_normal:
        caveats.append("No leukemia methylation profile was detected. If leukemia is suspected, consider a low "
                       "blast percentage in the sequenced sample, a problem with the sample such as an aspirate "
                       "diluted with blood, or a disease that the model does not contain.")
    lineage = res.lineage if res.lineage is not None else case.lineage
    intro, not_in_model = "", []
    missing = text["not_in_model"].get(lineage, []) if lineage else []
    # shown for every AML result, since common AML entities are outside the model, and for other lineages only when
    # no subtype or family was reported
    if case.coverage_ok and missing and (lineage == "AML" or case.level == "lineage"):
        intro = ("The model does not contain every leukemia. A sample of an entity outside the model is reported "
                 "as the most similar class, family or lineage, or not at all. Entities not in the model include:")
        not_in_model = list(missing)
    headline = first_sentence(paragraphs[0]) if paragraphs else ""
    return Interpretation(heading=heading, paragraphs=paragraphs, confirm=confirm, note=note, caveats=caveats,
                          not_in_model_intro=intro, not_in_model=not_in_model, headline=headline)


# ---------------------------------------------------------------------------------------------- cross-validation
@dataclass
class Evidence:
    condition: str
    fraction: float
    sentences: List[str]


def cv_evidence(case: Case, res: Result, ctx: Context) -> Optional[Evidence]:
    """
    What cross-validation on the training arrays showed at the undiluted binary condition whose observed fraction
    is nearest to the sample's coverage: the share of arrays whose most probable class reached the threshold, the
    share of those that were right and, for a subtype-level result, the recall of the reported class.
    """
    cv = ctx.cv
    if cv is None:
        return None
    rows = cv.binary
    coverage = case.coverage_pct / 100.0
    row = rows.loc[(rows["fraction"] - coverage).abs().idxmin()]
    cond, frac = str(row["condition"]), float(row["fraction"])
    t = fmt_threshold(ctx.threshold)
    key = f"{ctx.threshold:.2f}"
    share = row.get(f"callable_share_{key}", np.nan)
    acc = row.get(f"accuracy_callable_{key}", np.nan)
    opening = (f"In cross-validation on the training arrays, with nanopore reads simulated at {fmt_fraction(frac)} "
               f"of the model's CpGs (the tested coverage nearest to this sample's {fmt_pct(case.coverage_pct)}"
               + (f"; the figures are those recorded at the threshold {key}" if abs(float(key) - ctx.threshold) > 1e-9
                  else "") + "), ")
    share = float(share) if share is not None and share == share else np.nan
    acc = float(acc) if acc is not None and acc == acc else np.nan
    if np.isfinite(share) and share == 0:
        first = opening + f"the most probable class reached the threshold of {t} for none of the arrays."
    elif np.isfinite(share):
        first = opening + (f"the most probable class reached the threshold of {t} for {fmt_fraction(share)} of the "
                           "arrays")
        first += f", and {fmt_fraction(acc)} of those calls were correct." if np.isfinite(acc) else "."
    else:
        first = opening + f"the share of arrays called at the threshold of {t} was not recorded."
    sentences = [first]
    if case.level == "provisional" and ctx.caution_threshold is not None:
        cs = fmt_threshold(ctx.caution_threshold)
        band = cv.bands.loc[cond] if cv.bands is not None and cond in cv.bands.index else None
        b_share = float(band["band_share"]) if band is not None else np.nan
        b_acc = float(band["band_accuracy"]) if band is not None else np.nan
        if np.isfinite(b_share) and np.isfinite(b_acc) and b_share > 0:
            sentences.append(f"For {fmt_fraction(b_share)} of the arrays the most probable class lay from {cs} up to "
                             f"{t}, as in this sample, and {fmt_fraction(b_acc)} of those calls were correct.")
        else:
            sentences.append(f"How often calls from {cs} up to {t} were correct at that coverage is not recorded for "
                             "this model.")
    if res.result_class is not None and cv.recall is not None and res.result_class in cv.recall.index \
            and cond in cv.recall.columns:
        recall = cv.recall.at[res.result_class, cond]
        n = cv.recall.at[res.result_class, "n_samples"] if "n_samples" in cv.recall.columns else np.nan
        if np.isfinite(recall):
            # the table holds recall to three decimals: give it as a count when only one count fits
            k = int(round(recall * n)) if np.isfinite(n) and 0 < n <= 1000 else None
            if k is not None and abs(k / n - recall) <= 0.0005 + 1e-9:
                share = f"{k:,} of its {fmt_int(n)} training arrays"
            elif np.isfinite(n):
                share = f"{fmt_fraction(recall)} of its {fmt_int(n)} training arrays"
            else:
                share = f"{fmt_fraction(recall)} of its training arrays"
            sentences.append(f"For {ctx.shown(res.result_class)}, the recall at that coverage was {recall:.2f}: the "
                             f"class was the most probable one for {share}.")
    lo, hi = float(rows["fraction"].min()), float(rows["fraction"].max())
    if coverage < lo or coverage > hi:
        sentences.append(f"This sample's coverage lies outside the tested range ({fmt_fraction(lo)} to "
                         f"{fmt_fraction(hi)}), so these figures may not apply to it.")
    if case.level in ("family", "lineage"):
        sentences.append(f"They concern calls of a single class, whereas this sample was reported at {case.level} "
                         "level.")
    sentences.append("These figures come from cross-validation on training arrays with simulated nanopore reads, "
                     "not from nanopore samples.")
    return Evidence(condition=cond, fraction=frac, sentences=sentences)


# ---------------------------------------------------------------------------------------------- method
def about_text(ctx: Context) -> List[str]:
    """Four or five sentences on how the classifier works, and the interpretation text used."""
    m = ctx.model
    t = fmt_threshold(ctx.threshold)
    normal = NORMAL_LINEAGE in ctx.hierarchy.lineages
    what = "acute leukemia and normal bone marrow" if normal else "acute leukemia"
    size = (f"{fmt_int(m.n_training_arrays)} Illumina methylation arrays of {what}, in {m.n_classes} classes"
            if m.n_training_arrays else f"Illumina methylation arrays of {what}, in {m.n_classes} classes")
    if m.train_sim in ("reads", "binary", "oneread"):
        trained = (f"SPARSH is a neural network classifier trained on {size}; during training, sparse nanopore "
                   "reads were simulated from the arrays at many levels of coverage.")
    elif m.train_sim == "mask":
        trained = (f"SPARSH is a neural network classifier trained on {size}; during training, most CpGs of each "
                   "array were hidden, to resemble the sparse data of low-coverage sequencing.")
    else:
        trained = f"SPARSH is a neural network classifier trained on {size}."
    sentences = [
        trained,
        "For this sample it was applied to the methylation values of the CpGs covered by nanopore reads, and CpGs "
        "without reads were treated as missing.",
    ]
    if m.n_networks and m.n_networks > 1:
        source = f"The class probabilities are the average of {m.n_networks} networks"
    elif m.n_networks == 1:
        source = "The class probabilities come from one network"
    else:
        source = "The class probabilities come from the model's networks"
    if m.calibrated is False:
        sentences.append(f"{source}; they were not calibrated for this model.")
    else:
        if m.n_networks and m.n_networks > 1:
            held_out = ", each calibrated on training arrays held out from its training"
        elif m.n_networks == 1:
            held_out = ", calibrated on training arrays held out from its training"
        else:
            held_out = ", calibrated on training arrays held out from their training"
        sentences.append(f"{source}{held_out}, so that on training arrays a probability approximates how often such "
                         "a call is correct.")
    if ctx.hierarchy.families and ctx.caution_threshold is not None:
        cs = fmt_threshold(ctx.caution_threshold)
        sentences.append(f"A result is reported at the most specific level that reaches the threshold of {t}: a "
                         "single subtype, then a family of related subtypes, using the summed probability of the "
                         f"family's classes. Failing both, a subtype whose probability reaches {cs} is reported as "
                         "provisional, to be confirmed by other tests, and otherwise a lineage whose summed "
                         f"probability reaches {t}; when none of these applies, no result is reported.")
    elif ctx.caution_threshold is not None:
        cs = fmt_threshold(ctx.caution_threshold)
        sentences.append(f"A result is reported as a single subtype when its probability reaches the threshold of "
                         f"{t}. Failing that, a subtype whose probability reaches {cs} is reported as provisional, "
                         f"to be confirmed by other tests, and otherwise a lineage whose summed probability reaches "
                         f"{t}; when none of these applies, no result is reported.")
    elif ctx.hierarchy.families:
        sentences.append(f"A result is reported at the most specific level that reaches the threshold of {t}: a "
                         "single subtype, then a family of related subtypes, then a lineage, using the summed "
                         "probabilities of the family's or lineage's classes; when none reaches it, no result is "
                         "reported.")
    else:
        sentences.append(f"A result is reported at the most specific level that reaches the threshold of {t}: a "
                         "single subtype, then a lineage, using the summed probability of the lineage's classes; "
                         "when neither reaches it, no result is reported.")
    sentences.append("The model knows only the classes it was trained on, so a leukemia of another kind is reported "
                     "as the most similar class, family or lineage, or not at all, and every result needs "
                     "confirmation by established diagnostic methods.")
    return sentences


def interpretation_source(ctx: Context) -> str:
    """File name and version of the interpretation text, with a note when the version is a draft."""
    version = ctx.interpretation["version"]
    line = f"Interpretation text: {ctx.interpretation_source.name}, version {version}."
    if version.strip().lower().startswith("draft"):
        line += " This version is a draft and has not been reviewed."
    return line
