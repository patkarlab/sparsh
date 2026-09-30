"""
Names shown for model classes in the outputs of scripts/predict.py.

A class can be shown under a different name without retraining: the frozen run folders
(class_mapping.json), the training data, the cross-validation files and the class hierarchy keep
the internal name, and predict.py computes everything with it. Only what predict.py writes uses
the shown name: prediction, top2_class, top3_class, reported_call, the prob_ column headers and,
with ground truth, true_label, the per-class table and the confusion matrix. Probabilities, calls
and scores are unchanged. A ground-truth file may use either name.

configs/display_names.json:
    {"display": {"<internal class name>": "<name shown>", ...}}
Names of classes a model does not have are ignored, so one file serves every model version.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

CLASS_COLUMNS = ("prediction", "top2_class", "top3_class", "reported_call", "true_label")


def default_display_path() -> Optional[Path]:
    path = Path(__file__).resolve().parent.parent / "configs" / "display_names.json"
    return path if path.exists() else None


def load_display_names(path: Optional[str], classes: Sequence[str]) -> Dict[str, str]:
    """
    Internal name -> shown name, for the classes of this model only. An empty dict (path None)
    shows the internal names. Refused: two classes shown under one name, and a shown name that
    is another class's internal name (a ground-truth label would then be ambiguous).
    """
    if path is None:
        return {}
    with open(path) as f:
        content = json.load(f)
    names = content.get("display", {})
    if not isinstance(names, dict):
        raise ValueError(f"{path}: 'display' must map internal class names to the names shown")
    present = set(classes)
    display = {}
    for internal, shown in names.items():
        internal, shown = str(internal).strip(), str(shown).strip()
        if not shown:
            raise ValueError(f"{path}: empty name shown for {internal!r}")
        if internal not in present:
            logger.info(f"  display names: {internal!r} is not a class of this model, ignored")
            continue
        if shown != internal:
            display[internal] = shown
    shown_names = list(display.values())
    twice = sorted({s for s in shown_names if shown_names.count(s) > 1})
    if twice:
        raise ValueError(f"{path}: several classes would be shown as {twice}")
    clash = sorted(s for s in shown_names if s in present)
    if clash:
        raise ValueError(f"{path}: {clash} would be shown for one class but is the internal name of another")
    return display


def internal_names(display: Dict[str, str]) -> Dict[str, str]:
    """Shown name -> internal name, to read ground-truth files that use the shown names."""
    return {shown: internal for internal, shown in display.items()}


def show(name, display: Dict[str, str]):
    """The shown name of one label (labels that are not renamed classes pass through)."""
    return display.get(name, name) if isinstance(name, str) else name


def show_columns(df: pd.DataFrame, display: Dict[str, str], columns: Iterable[str] = CLASS_COLUMNS) -> pd.DataFrame:
    """A copy of df with class names replaced by the shown names in the given columns (those present)."""
    if not display:
        return df
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: show(v, display))
    return out
