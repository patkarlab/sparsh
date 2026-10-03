"""
Inputs of the per-case reports: what scripts/predict.py wrote, and what the report adds to it.

load_inputs() reads the output folder of scripts/predict.py (predictions.csv, and prediction_info.json,
display_names_used.json and class_hierarchy_used.json when present), the model folder named there or on the
command line (config.json, class_mapping.json and the cross-validation tables, all optional), the interpretation
text, the class colors, an optional sample sheet and an optional folder of copy-number karyotypes. It checks
them and returns one Context, shared by all pages, and one Case per sample.

Class names. predictions.csv shows the names of configs/display_names.json (for example T-ALL_HOXA_ETP for the
internal class T-ALL_HOXA9_ETP). Everything here is looked up by internal name, as in the model folder, the class
hierarchy, the colors and the interpretation text; evaluation/display.py maps between the two, and the pages
show the shown names again.

Threshold. The report uses the threshold of the prediction run (from prediction_info.json; for a prediction folder
without that file, the one stored with the model, else 0.90). When another threshold is asked for, the reported calls are
worked out again from the class probabilities with evaluation/hierarchy.py, as predict.py does, so that every
page agrees with the threshold it states. Samples that predict.py found to have too few observed CpGs stay not
callable. At the prediction run's own threshold the calls of predictions.csv are used as they are, because its
probabilities are rounded to five decimals and a recomputation could move a sample that lies on the threshold.

Coverage. A sample with fewer than 5% of the model's CpGs observed is flagged as having low coverage: the deployed
recipe is tested in cross-validation from 5% of its CpGs upwards. The caveat on its page states the range the
model's own cross-validation tables cover, when they are available.
"""

import base64
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.display import internal_names, load_display_names, show
from evaluation.hierarchy import LEVELS, Hierarchy, default_hierarchy_path, hierarchical_calls, load_hierarchy

REPO = Path(__file__).resolve().parent.parent
VERSION = (REPO / "VERSION").read_text().strip()
DEFAULT_INTERPRETATION = REPO / "configs" / "interpretation.json"
COLOURS_FILE = REPO / "configs" / "class_colours.json"
DEFAULT_TITLE = "SPARSH methylation classification report"
DEFAULT_THRESHOLD = 0.90
LOW_COVERAGE = 0.05
DEFAULT_MIN_COVERAGE = 0.01
GREY = "#969696"
COVERAGE_CALL = "not callable (coverage)"
NORMAL_CLASS = "Normal_Control_BM"
NORMAL_LINEAGE = "Normal marrow"

COLOUR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
BINARY_CONDITION = re.compile(r"^binary_(\d+(?:\.\d+)?)$")
KARYOTYPE_FILE = re.compile(r"^(.+)\.(karyotype\.json|karyotype\.svg|karyotype\.png|chromosomes\.svg|"
                            r"chromosomes\.png)$", re.IGNORECASE)
REQUIRED_COLUMNS = ("sample", "file", "n_columns", "n_matched", "n_observed", "coverage_pct", "prediction",
                    "confidence", "callable", "note", "family", "family_prob", "lineage", "lineage_prob",
                    "reported_call", "reported_level")
NUMERIC_COLUMNS = ("n_columns", "n_matched", "n_observed", "coverage_pct", "n_repeated_probes", "confidence",
                   "family_prob", "lineage_prob")
NEVER_BLANK = ("n_columns", "n_matched", "n_observed", "coverage_pct", "confidence", "lineage_prob")


class ReportInputError(Exception):
    """A problem with the inputs that stops the report; the message says what to fix."""


def warn(message: str) -> None:
    """A problem the report works around; printed so that it can be fixed before the next report."""
    print(f"Warning: {message}", file=sys.stderr)


@dataclass
class Karyotype:
    """The copy-number result of one sample: the JSON content, or why it could not be read, and the figure."""
    source: str
    data: dict = field(default_factory=dict)
    error: str = ""
    figure: Optional[str] = None          # data URI, so the page stays self-contained
    figure_source: str = ""
    chromosome_figure: Optional[str] = None   # one panel per chromosome, when the copy-number module drew it
    chromosome_figure_source: str = ""


@dataclass
class ModelInfo:
    """What is known about the model that made the predictions; None where nothing records it."""
    name: Optional[str]
    models_used: Optional[str]
    n_networks: Optional[int]
    n_classes: int
    n_cpgs: Optional[int]
    n_training_arrays: Optional[int]
    calibrated: Optional[bool]
    train_sim: Optional[str]


@dataclass
class CrossValidation:
    """The model's cross-validation tables: undiluted binary conditions, sorted by observed fraction, and recall."""
    binary: pd.DataFrame
    recall: Optional[pd.DataFrame]
    folder: Path


@dataclass
class Case:
    """One sample: the columns of predictions.csv that the report uses, with class names made internal."""
    sample: str
    page: str
    file: str
    n_columns: float
    n_matched: float
    n_observed: float
    coverage_pct: float
    n_repeated_probes: float
    note: str
    probs: np.ndarray                     # in the order of Context.classes
    top: str                              # most probable class (internal name)
    confidence: float                     # its probability
    coverage_ok: bool                     # predict.py found enough observed CpGs to call the sample
    level: str                            # subtype, no_subtype, family, lineage or none
    family: str                           # the reported family, or else the family of the top class, or ""
    family_prob: float
    lineage: str                          # the most probable lineage
    lineage_prob: float
    info: Optional[List[Tuple[str, str]]] = None      # sample-sheet columns; None when the sheet has no row
    karyotype: Optional[Karyotype] = None


@dataclass
class Context:
    """Everything the pages share: names, hierarchy, threshold, colors, interpretation text, model and sources."""
    title: str
    generated: datetime
    predictions_dir: Path
    classes: List[str]
    display: Dict[str, str]
    hierarchy: Hierarchy
    hierarchy_source: str
    threshold: float
    prediction_threshold: float
    recomputed: bool
    min_coverage: float
    colours: Dict[str, str]
    lineage_colours: Dict[str, str]
    interpretation: dict
    interpretation_source: Path
    model: ModelInfo
    cv: Optional[CrossValidation]
    prediction_version: Optional[str]
    prediction_created: Optional[str]
    duplicate_probes: Optional[str]
    has_sheet: bool = False
    has_karyotypes: bool = False
    report_version: str = VERSION

    def shown(self, cls: str) -> str:
        """The name shown for an internal class name."""
        return show(cls, self.display)

    def colour(self, cls: str) -> str:
        """Color of a class (internal name), as fitted by load_colours."""
        return self.colours.get(cls, GREY)

    def lineage_colour(self, lineage: str) -> str:
        """Color of a lineage; 'Other' in the color file, else gray, for a lineage without one."""
        return self.lineage_colours.get(lineage, self.lineage_colours.get("Other", GREY))

    def class_index(self, cls: str) -> int:
        """Position of a class (internal name) in the prob_ columns."""
        return self.classes.index(cls)


# ---------------------------------------------------------------------------------------------- small readers
def _read_json(path: Path, what: str):
    """The content of a JSON file, or a ReportInputError that names the file and its role."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ReportInputError(f"{path} ({what}) is not valid JSON: {e}")
    except OSError as e:
        raise ReportInputError(f"{path} ({what}) cannot be read: {e}")


def _read_csv_as_text(path: Path, what: str) -> pd.DataFrame:
    """A CSV file with every cell as stripped text (a byte-order mark, as spreadsheet programs write, is dropped)."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
        raise ReportInputError(f"{path} ({what}) cannot be read as a CSV file: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _normalise(name) -> str:
    """Sample names are compared as predict.py writes them: stripped and upper-cased."""
    return str(name).strip().upper()


def _label(column: str) -> str:
    """A sample-sheet column name as a label: 'blast_percent' -> 'Blast percent'."""
    text = re.sub(r"[_\s]+", " ", str(column).strip())
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------------------------------------- predictions
def read_predictions(path: Path) -> pd.DataFrame:
    """predictions.csv as written by scripts/predict.py, with numbers and flags converted and names normalized."""
    df = _read_csv_as_text(path, "predictions")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ReportInputError(f"{path} lacks the columns {missing}; it must be a predictions.csv written by "
                               "scripts/predict.py")
    if "n_repeated_probes" not in df.columns:            # optional: shown as not recorded
        df["n_repeated_probes"] = ""
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ReportInputError(f"{path} has no class-probability columns (prob_<class>)")
    if df.empty:
        raise ReportInputError(f"{path} lists no samples")
    for col in list(NUMERIC_COLUMNS) + prob_cols:
        text = df[col]
        values = pd.to_numeric(text.where(text != "", np.nan), errors="coerce")
        bad = (text != "") & values.isna()
        if bad.any():
            raise ReportInputError(f"{path}: column {col} holds a value that is not a number, "
                                   f"{text[bad].iloc[0]!r} (sample {df.loc[bad, 'sample'].iloc[0]})")
        if col in NEVER_BLANK or col in prob_cols:
            blank = values.isna()
            if blank.any():
                raise ReportInputError(f"{path}: column {col} is empty for sample {df.loc[blank, 'sample'].iloc[0]}")
        df[col] = values
    flags = df["callable"].str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if flags.isna().any():
        raise ReportInputError(f"{path}: column callable must be True or False "
                               f"(found {df.loc[flags.isna(), 'callable'].iloc[0]!r})")
    df["callable"] = flags.astype(bool)
    unknown = sorted(set(df["reported_level"]) - set(LEVELS))
    if unknown:
        raise ReportInputError(f"{path}: reported_level {unknown} is not one of {list(LEVELS)}")
    df["sample"] = df["sample"].map(_normalise)
    if (df["sample"] == "").any():
        raise ReportInputError(f"{path}: a row has no sample name")
    twice = sorted(set(df.loc[df["sample"].duplicated(), "sample"]))
    if twice:
        raise ReportInputError(f"{path}: samples listed more than once: {twice[:10]}")
    return df


def _class_mapping(model_dir: Optional[Path]) -> Optional[List[str]]:
    """The model's classes in index order, from class_mapping.json, or None without a model folder."""
    if model_dir is None or not (model_dir / "class_mapping.json").exists():
        return None
    content = _read_json(model_dir / "class_mapping.json", "class mapping")
    try:
        return [str(content[k]) for k in sorted(content, key=int)]
    except (TypeError, ValueError) as e:
        raise ReportInputError(f"{model_dir / 'class_mapping.json'} must map class indices to names: {e}")


def resolve_classes(shown_names: Sequence[str], info: dict, predictions_dir: Path,
                    model_classes: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    """
    Internal class names of the prob_ columns, in their order, and the internal -> shown map used by the
    prediction run. prediction_info.json records both; without it the map is read from display_names_used.json
    (fitted to the model's classes when the model folder is at hand), and a folder without either was written with
    the internal names.
    """
    shown_names = list(shown_names)
    if isinstance(info.get("classes"), list) and isinstance(info.get("display_names_shown"), dict):
        classes = [str(c) for c in info["classes"]]
        display = {str(k): str(v) for k, v in info["display_names_shown"].items()}
        if [show(c, display) for c in classes] != shown_names:
            raise ReportInputError(
                f"The prob_ columns of {predictions_dir / 'predictions.csv'} do not match the classes recorded in "
                "prediction_info.json of the same folder; the two files come from different prediction runs")
        return classes, display
    display = {}
    used = predictions_dir / "display_names_used.json"
    if used.exists():
        if model_classes is not None:
            try:
                display = load_display_names(str(used), model_classes)
            except ValueError as e:
                raise ReportInputError(str(e))
        else:
            content = _read_json(used, "display names")
            raw = content.get("display", {}) if isinstance(content, dict) else None
            if not isinstance(raw, dict):
                raise ReportInputError(f"{used}: 'display' must map internal class names to the names shown")
            present = set(shown_names)
            display = {str(k).strip(): str(v).strip() for k, v in raw.items()
                       if str(v).strip() in present and str(k).strip() != str(v).strip()}
    back = internal_names(display)
    classes = [back.get(s, s) for s in shown_names]
    if len(set(classes)) != len(classes):
        raise ReportInputError(f"{predictions_dir / 'predictions.csv'}: two prob_ columns name the same class")
    return classes, display


# ---------------------------------------------------------------------------------------------- other inputs
def load_colours(classes: Sequence[str], h: Hierarchy) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Color of every model class (its own color in configs/class_colours.json, else its lineage's, else gray)
    and the lineage colors. Colors must be written as #rgb or #rrggbb; others are ignored with a warning.
    """
    content = {}
    if COLOURS_FILE.exists():
        content = _read_json(COLOURS_FILE, "class colors")
    else:
        warn(f"{COLOURS_FILE} not found; every class is drawn in gray")

    def valid(part: str) -> Dict[str, str]:
        out = {}
        entries = content.get(part, {}) if isinstance(content, dict) else {}
        if not isinstance(entries, dict):
            raise ReportInputError(f"{COLOURS_FILE}: '{part}' must map names to colors")
        for name, value in entries.items():
            if isinstance(value, str) and COLOUR.match(value.strip()):
                out[str(name)] = value.strip()
            else:
                warn(f"{COLOURS_FILE}: color {value!r} of {name!r} is not written as #rrggbb; ignored")
        return out

    lineage_colours, class_colours = valid("lineages"), valid("classes")
    colours, by_lineage, grey = {}, [], []
    for c in classes:
        if c in class_colours:
            colours[c] = class_colours[c]
        elif h.lineage_of.get(c) in lineage_colours:
            colours[c] = lineage_colours[h.lineage_of[c]]
            by_lineage.append(c)
        else:
            colours[c] = lineage_colours.get("Other", GREY)
            grey.append(c)
    if by_lineage:
        print(f"Classes drawn in the color of their lineage (no color of their own in {COLOURS_FILE.name}): "
              + ", ".join(by_lineage))
    if grey:
        warn(f"no color for the classes {grey} in {COLOURS_FILE}, nor for their lineages; they are drawn in gray. "
             "Add them to 'classes', or their lineage to 'lineages'")
    return colours, lineage_colours


def load_interpretation(path: Path, classes: Sequence[str], display: Dict[str, str], h: Hierarchy) -> dict:
    """
    The interpretation file (configs/interpretation.json), checked and fitted to the model: class entries may be
    keyed by the internal or the shown name, and 'not_in_model' maps lineages to the entities the model lacks (a
    plain list is taken as the list for AML). Missing entries are allowed, and the pages then fall back to the
    class name and a neutral sentence; a warning lists the model's classes, families and lineages without text.
    """
    path = Path(path)
    if not path.exists():
        raise ReportInputError(f"Interpretation file {path} does not exist")
    content = _read_json(path, "interpretation text")
    if not isinstance(content, dict):
        raise ReportInputError(f"{path} must hold a JSON object with 'classes', 'families', 'lineages', 'levels' "
                               "and 'not_in_model'")
    parts = {}
    for key in ("classes", "families", "lineages"):
        part = content.get(key, {})
        if not isinstance(part, dict) or not all(isinstance(v, dict) for v in part.values()):
            raise ReportInputError(f"{path}: '{key}' must map names to objects of text fields")
        for name, entry in part.items():
            bad = sorted(k for k, v in entry.items() if not isinstance(v, str))
            if bad:
                raise ReportInputError(f"{path}: the fields {bad} of {key} entry {name!r} must be text")
        parts[key] = {str(k): {f: v.strip() for f, v in e.items()} for k, e in part.items()}
    levels = content.get("levels", {})
    if not isinstance(levels, dict) or not all(isinstance(v, str) for v in levels.values()):
        raise ReportInputError(f"{path}: 'levels' must map 'none' and 'coverage' to text")
    not_in_model = content.get("not_in_model", {})
    if isinstance(not_in_model, list):                  # a plain list is read as the list for AML
        not_in_model = {"AML": not_in_model}
    if not isinstance(not_in_model, dict) or not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v) for v in not_in_model.values()):
        raise ReportInputError(f"{path}: 'not_in_model' must map lineage names to lists of text")

    present = set(classes)
    back = internal_names(display)
    class_entries = {}
    for name, entry in parts["classes"].items():
        internal = name if name in present else back.get(name, name)
        if internal in class_entries and name != internal:
            continue                      # an entry under the internal name wins over one under the shown name
        class_entries[internal] = entry
    incomplete = [c for c in classes if c in class_entries
                  and not (class_entries[c].get("name") and class_entries[c].get("summary"))]
    if incomplete:
        warn(f"{path.name}: the entries of {incomplete} lack 'name' or 'summary'; the class name or a neutral "
             "sentence is used instead")
    no_entry = [c for c in classes if c not in class_entries]
    if no_entry:
        warn(f"{path.name} has no entry for {len(no_entry)} of the model's {len(classes)} classes; their pages use "
             "the class name and a neutral sentence: " + ", ".join(no_entry))
    families = [f for f in h.families if f not in parts["families"]]
    lineages = [lin for lin in h.lineages if lin not in parts["lineages"]]
    if families or lineages:
        warn(f"{path.name} has no text for the families {families} and lineages {lineages} of this model; "
             "neutral sentences are used")
    return {"version": str(content.get("version", "not stated")), "classes": class_entries,
            "families": parts["families"], "lineages": parts["lineages"],
            "levels": {str(k): v.strip() for k, v in levels.items()},
            "not_in_model": {str(lin): [v.strip() for v in items if v.strip()] for lin, items in not_in_model.items()}}


def read_sample_sheet(path: Path, predicted: Sequence[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Sample information by sample: every column of the sheet except 'sample', as (label, value) pairs, empty
    values left out. Names are matched as predict.py names samples (stripped, upper case). A sheet row whose
    sample was not predicted is an error, because it usually means that a name is spelled differently.
    """
    path = Path(path)
    if not path.exists():
        raise ReportInputError(f"Sample sheet {path} does not exist")
    sheet = _read_csv_as_text(path, "sample sheet")
    by_lower = {c.lower(): c for c in sheet.columns}
    if "sample" not in by_lower:
        raise ReportInputError(f"Sample sheet {path} needs a column 'sample' (its columns: {list(sheet.columns)})")
    sample_col = by_lower["sample"]
    sheet = sheet[(sheet != "").any(axis=1)]
    names = sheet[sample_col].map(_normalise)
    if (names == "").any():
        raise ReportInputError(f"Sample sheet {path}: {int((names == '').sum())} row(s) have values but no sample name")
    twice = sorted(set(names[names.duplicated()]))
    if twice:
        raise ReportInputError(f"Sample sheet {path} lists these samples more than once (case is ignored): "
                               f"{twice[:10]}")
    unknown = sorted(set(names) - set(predicted))
    if unknown:
        raise ReportInputError(
            f"Sample sheet {path} lists {len(unknown)} sample(s) that are not in predictions.csv: {unknown[:10]}. "
            "Names are matched ignoring case and surrounding spaces, as predict.py names samples after their "
            "files; correct the names or remove these rows")
    columns = [c for c in sheet.columns if c != sample_col and c and not re.match(r"^Unnamed: \d+$", c)]
    info = {}
    for name, (_, row) in zip(names, sheet.iterrows()):
        info[name] = [(_label(c), row[c]) for c in columns if row[c] != ""]
    return info


def _figure_uri(path: Path) -> Tuple[Optional[str], str]:
    """A figure file as a data URI, or None with the reason when it is not a PNG or SVG image."""
    try:
        content = path.read_bytes()
    except OSError as e:
        return None, f"cannot be read: {e}"
    if path.suffix.lower() == ".png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return None, "is not a PNG image"
        mime = "image/png"
    else:
        head = content[:65536].decode("utf-8", errors="replace")
        if "<svg" not in head:
            return None, "is not an SVG image"
        mime = "image/svg+xml"
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}", ""


def _file_key(name) -> str:
    """A sample name as the copy-number module writes it into file names: every run of characters other than
    letters, digits, '.', '_' and '-' becomes '_', and the name is upper-cased."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).upper()


def read_karyotypes(folder: Path, predicted: Sequence[str], wanted: Sequence[str]) -> Dict[str, Karyotype]:
    """
    Copy-number results of the wanted samples, from <SAMPLE>.karyotype.json and optionally <SAMPLE>.karyotype.svg
    or .png (the SVG is preferred when both exist). File names are matched ignoring case, and with spaces and other
    characters that cannot be part of a file name read as '_', as scripts/karyotype.py writes them. Files of
    samples that are not in predictions.csv are reported. A file that cannot be read gives a note on the page and
    a warning, not an error, so the other pages are still written.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ReportInputError(f"Karyotype folder {folder} does not exist")
    found: Dict[str, Dict[str, Path]] = {}
    for f in sorted(folder.iterdir()):
        m = KARYOTYPE_FILE.match(f.name)
        if not m or not f.is_file():
            continue
        key, ext = _file_key(m.group(1)), m.group(2).lower().replace("karyotype.", "")
        if ext in found.setdefault(key, {}):
            raise ReportInputError(f"Karyotype folder {folder}: {found[key][ext].name} and {f.name} belong to the "
                                   f"same sample ({key}); keep one")
        found[key][ext] = f
    by_key: Dict[str, List[str]] = {}
    for name in predicted:
        by_key.setdefault(_file_key(name), []).append(name)
    shared = {k: v for k, v in by_key.items() if len(v) > 1 and k in found}
    if shared:
        warn(f"the sample names {sorted(n for v in shared.values() for n in v)} share karyotype files; their "
             "karyotypes are left out")
    unused = sorted(k for k in found if k not in by_key)
    if unused:
        warn(f"{len(unused)} sample(s) in the karyotype folder are not in predictions.csv and are ignored: "
             f"{unused[:10]}")
    wanted = set(wanted)
    out = {}
    for key, files in found.items():
        names = by_key.get(key, [])
        if len(names) != 1 or names[0] not in wanted:
            continue
        name = names[0]
        if "json" not in files:
            warn(f"karyotype figure of {name} without {key}.karyotype.json; ignored")
            continue
        kt = Karyotype(source=files["json"].name)
        try:
            data = json.loads(files["json"].read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            kt.data = data
        except (ValueError, OSError) as e:
            kt.error = f"{files['json'].name} could not be read ({e})"
            warn(f"karyotype of {name}: {kt.error}")
        stated = kt.data.get("sample")
        if stated is not None and _file_key(stated) != key:
            warn(f"{files['json'].name} names the sample {stated!r}; it is shown for {name}, after its file name")
        for ext in ("svg", "png"):
            if ext in files:
                uri, problem = _figure_uri(files[ext])
                if uri:
                    kt.figure, kt.figure_source = uri, files[ext].name
                    break
                warn(f"karyotype figure {files[ext].name} {problem}; left out")
        for ext in ("chromosomes.svg", "chromosomes.png"):
            if ext in files:
                uri, problem = _figure_uri(files[ext])
                if uri:
                    kt.chromosome_figure, kt.chromosome_figure_source = uri, files[ext].name
                    break
                warn(f"chromosome figure {files[ext].name} {problem}; left out")
        out[name] = kt
    return out


def load_cross_validation(model_dir: Path, threshold: float) -> Optional[CrossValidation]:
    """
    The rows of cv_metrics_by_condition.csv named binary_<fraction> (undiluted, without call errors) and
    cv_recall_by_class.csv. Missing or unusable tables leave the section out, with a warning.
    """
    metrics_path = model_dir / "cv_metrics_by_condition.csv"
    if not metrics_path.exists():
        warn(f"{metrics_path} not found; the cross-validation section is left out")
        return None
    try:
        metrics = pd.read_csv(metrics_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
        warn(f"{metrics_path} cannot be read ({e}); the cross-validation section is left out")
        return None
    if "condition" not in metrics.columns:
        warn(f"{metrics_path} has no column 'condition'; the cross-validation section is left out")
        return None
    names = metrics["condition"].astype(str)
    match = names.str.match(BINARY_CONDITION.pattern)
    binary = metrics[match].copy()
    if binary.empty:
        warn(f"{metrics_path} has no undiluted binary condition (binary_<fraction>); the cross-validation section "
             "is left out")
        return None
    from_name = names[match].str.extract(BINARY_CONDITION.pattern)[0].astype(float)
    if "observed_fraction" in binary.columns:
        binary["fraction"] = pd.to_numeric(binary["observed_fraction"], errors="coerce").fillna(from_name)
    else:
        binary["fraction"] = from_name
    binary = binary.sort_values("fraction", kind="stable").reset_index(drop=True)
    wanted = f"callable_share_{threshold:.2f}"
    if wanted not in binary.columns:
        recorded = sorted(c[len("callable_share_"):] for c in binary.columns if c.startswith("callable_share_"))
        warn(f"{metrics_path} has no figures at the threshold {threshold:.2f} (only at {recorded}); the share of "
             "samples called and the accuracy of the calls are left out")
    elif abs(round(threshold, 2) - threshold) > 1e-9:
        warn(f"the cross-validation figures are recorded at the threshold {threshold:.2f}, the nearest to the "
             f"{threshold:g} in force, and the pages say so")
    recall = None
    recall_path = model_dir / "cv_recall_by_class.csv"
    if recall_path.exists():
        try:
            recall = pd.read_csv(recall_path, index_col=0)
            recall.index = recall.index.astype(str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
            warn(f"{recall_path} cannot be read ({e}); class recall is left out")
    else:
        warn(f"{recall_path} not found; class recall is left out of the cross-validation section")
    return CrossValidation(binary=binary, recall=recall, folder=model_dir)


def _min_coverage(info: dict, df: pd.DataFrame) -> float:
    """
    The minimum observed share of model CpGs below which predict.py marks a sample not callable: recorded in
    prediction_info.json, else read from the note of such a sample ("coverage below 1.0%"), else predict.py's
    default.
    """
    if info.get("min_coverage") is not None:
        return float(info["min_coverage"])
    for note in df["note"]:
        m = re.match(r"^coverage below (\d+(?:\.\d+)?)%$", str(note))
        if m:
            return float(m.group(1)) / 100.0
    return DEFAULT_MIN_COVERAGE


def page_names(samples: Sequence[str]) -> Dict[str, str]:
    """
    A safe, distinct file name for each sample's page: characters other than letters, digits, dot, hyphen and
    underscore become underscores, a leading dot is replaced (no hidden files), and names that would clash on a
    file system that ignores case get a numbered suffix.
    """
    used, out = set(), {}
    for sample in samples:
        base = re.sub(r"[^A-Za-z0-9._-]", "_", sample)
        base = re.sub(r"^\.+", lambda m: "_" * len(m.group(0)), base)[:120] or "sample"
        name, k = base, 2
        while name.lower() in used:
            name, k = f"{base}-{k}", k + 1
        used.add(name.lower())
        out[sample] = f"{name}.html"
    return out


# ---------------------------------------------------------------------------------------------- everything
def load_inputs(predictions_dir, model_dir=None, interpretation=None, sample_sheet=None, karyotype_dir=None,
                samples=None, title=None, threshold=None) -> Tuple[Context, List[Case]]:
    """Read and check every input of a report; see the module docstring. Raises ReportInputError."""
    pred_dir = Path(predictions_dir).expanduser()
    if not pred_dir.is_dir():
        raise ReportInputError(f"Predictions folder {pred_dir} does not exist")
    csv_path = pred_dir / "predictions.csv"
    if not csv_path.exists():
        raise ReportInputError(f"No predictions.csv in {pred_dir}; --predictions_dir must be an output folder of "
                               "scripts/predict.py")
    info = {}
    if (pred_dir / "prediction_info.json").exists():
        info = _read_json(pred_dir / "prediction_info.json", "record of the prediction run")
        if not isinstance(info, dict):
            raise ReportInputError(f"{pred_dir / 'prediction_info.json'} must hold a JSON object")
    else:
        print(f"No prediction_info.json in {pred_dir}; the model is described from its folder where possible")
    df = read_predictions(csv_path)

    # ------------------------------------------------------------ model folder
    given = Path(model_dir).expanduser() if model_dir else None
    recorded = Path(info["model_dir"]).expanduser() if info.get("model_dir") else None
    folder = given or recorded
    available = folder is not None and (folder / "config.json").exists()
    if folder is None:
        print("No model folder is named (--model_dir or prediction_info.json); the cross-validation section is "
              "left out")
    elif not available:
        warn(f"model folder {folder} (config.json) is not available; the cross-validation section is left out")
    elif given and recorded and given.resolve() != recorded.resolve():
        warn(f"--model_dir {given} is not the model folder of the prediction run ({recorded}); the "
             f"cross-validation section describes {given}")
    config = _read_json(folder / "config.json", "model configuration") if available else {}
    model_classes = _class_mapping(folder) if available else None

    # ------------------------------------------------------------ classes, names, hierarchy
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    classes, display = resolve_classes([c[len("prob_"):] for c in prob_cols], info, pred_dir, model_classes)
    if model_classes is not None and set(model_classes) != set(classes):
        extra = sorted(set(classes) - set(model_classes))
        lacking = sorted(set(model_classes) - set(classes))
        raise ReportInputError(
            f"The model folder {folder} has other classes than {csv_path}: {len(extra)} predicted class(es) are "
            f"not in the model (for example {extra[:3]}) and {len(lacking)} model class(es) were not predicted "
            f"(for example {lacking[:3]}). Give the --model_dir that made these predictions")
    recorded_hierarchy = Path(info["hierarchy_file"]) if info.get("hierarchy_file") else None
    if (pred_dir / "class_hierarchy_used.json").exists():
        hierarchy_path = pred_dir / "class_hierarchy_used.json"
    elif recorded_hierarchy is not None and recorded_hierarchy.exists():
        hierarchy_path = recorded_hierarchy
    elif "hierarchy_file" in info and info["hierarchy_file"] is None:
        hierarchy_path = None                     # the prediction run used lineage prefixes only
    else:
        hierarchy_path = default_hierarchy_path()
        if recorded_hierarchy is not None:
            warn(f"the class hierarchy of the prediction run ({recorded_hierarchy}) is not available; "
                 f"{hierarchy_path} is used")
    try:
        h = load_hierarchy(str(hierarchy_path) if hierarchy_path else None, classes)
    except (ValueError, OSError) as e:
        raise ReportInputError(f"Class hierarchy {hierarchy_path}: {e}")

    # ------------------------------------------------------------ threshold and calls
    prediction_threshold = info.get("threshold")
    if prediction_threshold is None:
        prediction_threshold = config.get("inference", {}).get("threshold", DEFAULT_THRESHOLD)
    prediction_threshold = float(prediction_threshold)
    use = float(threshold) if threshold is not None else prediction_threshold
    if not 0.0 < use <= 1.0:
        raise ReportInputError(f"The threshold must be above 0 and at most 1 (got {use})")
    recomputed = abs(use - prediction_threshold) > 1e-9
    back = internal_names(display)
    P = df[prob_cols].to_numpy(dtype=float)
    coverage_ok = (df["reported_call"] != COVERAGE_CALL).to_numpy()
    if not info:
        off = ((df["callable"] & (df["confidence"] < prediction_threshold - 1e-4))
               | (~df["callable"] & coverage_ok & (df["confidence"] >= prediction_threshold + 1e-4)))
        if off.any():
            warn(f"predictions.csv does not look as if it was made at the threshold {prediction_threshold:.2f} "
                 f"(sample {df.loc[off, 'sample'].iloc[0]}); give the prediction run's threshold with --threshold")
    if recomputed:
        calls = hierarchical_calls(P, classes, h, use, coverage_ok)
        top_idx = P.argmax(axis=1)
        tops = [classes[i] for i in top_idx]
        confidence = P[np.arange(len(P)), top_idx]
        source = calls
        print(f"Reported calls worked out again at the threshold {use:.2f} (the prediction run used "
              f"{prediction_threshold:.2f})")
    else:
        tops = [back.get(p, p) for p in df["prediction"]]
        unknown = sorted({t for t in tops if t not in classes})
        if unknown:
            raise ReportInputError(f"{csv_path}: predicted classes {unknown} have no prob_ column")
        confidence = df["confidence"].to_numpy()
        source = df
        odd = sorted({f for f in df["family"] if f and f not in h.families})
        if odd:
            warn(f"the families {odd} of predictions.csv are not families of the hierarchy {hierarchy_path}; "
                 "their members cannot be listed")

    levels, families, lineages = (source[c].tolist() for c in ("reported_level", "family", "lineage"))
    family_p, lineage_p = source["family_prob"].to_numpy(dtype=float), source["lineage_prob"].to_numpy(dtype=float)

    # ------------------------------------------------------------ which samples
    names = df["sample"].tolist()
    keep = list(range(len(df)))
    if samples:
        wanted = [_normalise(s) for s in samples]
        missing = [s for s in wanted if s not in set(names)]
        if missing:
            raise ReportInputError(f"--samples: {missing} are not in {csv_path} (names are matched ignoring case)")
        keep = [i for i, n in enumerate(names) if n in set(wanted)]

    sheet = read_sample_sheet(sample_sheet, names) if sample_sheet else {}
    if sample_sheet:
        without = [names[i] for i in keep if names[i] not in sheet]
        if without:
            warn(f"{len(without)} sample(s) have no row in the sample sheet: {without[:10]}")
    karyotypes = read_karyotypes(karyotype_dir, names, [names[i] for i in keep]) if karyotype_dir else {}

    colours, lineage_colours = load_colours(classes, h)
    interp_path = Path(interpretation).expanduser() if interpretation else DEFAULT_INTERPRETATION
    interp = load_interpretation(interp_path, classes, display, h)
    cv = load_cross_validation(folder, use) if available else None

    inference = config.get("inference", {})
    n_cpgs = info.get("n_cpgs") or config.get("model", {}).get("input_dim")
    model = ModelInfo(
        name=str(info.get("model_name") or (folder.resolve().name if folder else "")) or None,
        models_used=info.get("models_used"),
        n_networks=info.get("n_networks"), n_classes=len(classes),
        n_cpgs=int(n_cpgs) if n_cpgs else None,
        n_training_arrays=config.get("data", {}).get("n_samples"),
        calibrated=config.get("training", {}).get("calibrate") if config else None,
        train_sim=config.get("training", {}).get("train_sim") if config else None)
    if model.n_networks is None and inference.get("fold_models") and not inference.get("final_model"):
        model.n_networks, model.models_used = len(inference["fold_models"]), "ensemble"

    pages = page_names([names[i] for i in keep])
    cases = []
    for i in keep:
        row = df.iloc[i]
        family = families[i] if isinstance(families[i], str) else ""
        cases.append(Case(
            sample=names[i], page=pages[names[i]], file=row["file"], n_columns=row["n_columns"],
            n_matched=row["n_matched"], n_observed=row["n_observed"], coverage_pct=row["coverage_pct"],
            n_repeated_probes=row["n_repeated_probes"], note=row["note"], probs=P[i], top=tops[i],
            confidence=float(confidence[i]), coverage_ok=bool(coverage_ok[i]),
            level=str(levels[i]), family=family, family_prob=float(family_p[i]), lineage=str(lineages[i]),
            lineage_prob=float(lineage_p[i]), info=sheet.get(names[i]), karyotype=karyotypes.get(names[i])))

    ctx = Context(
        title=title or DEFAULT_TITLE, generated=datetime.now().astimezone(), predictions_dir=pred_dir.resolve(),
        classes=classes, display=display, hierarchy=h,
        hierarchy_source=str(hierarchy_path) if hierarchy_path else "lineage prefixes only",
        threshold=use, prediction_threshold=prediction_threshold, recomputed=recomputed,
        min_coverage=_min_coverage(info, df), colours=colours,
        lineage_colours=lineage_colours,
        interpretation=interp, interpretation_source=interp_path, model=model, cv=cv,
        prediction_version=info.get("sparsh_version"), prediction_created=info.get("created"),
        duplicate_probes=info.get("duplicate_probes"), has_sheet=bool(sample_sheet),
        has_karyotypes=bool(karyotype_dir))
    return ctx, cases
