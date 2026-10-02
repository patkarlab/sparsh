"""
Names, colours and order of classes and lineages in the figures.

Classes are drawn in the colours of configs/class_colours.json and listed in its order, which groups them by
lineage; classes the file does not list follow in alphabetical order, in the colour of their lineage. Lineages
come from the prefix rules of configs/class_hierarchy.json, which also place classes that are not in the model.

The names shown are the short names of configs/figure_names.json, which also names the lineages for the legend
headings and the groups of the signature heatmaps. A class it does not list is shown under its name in
configs/display_names.json, or else its internal name. Where two classes of a figure share a short name, as
KMT2A-rearranged does in acute myeloid and B-lymphoblastic leukemia, maps and axes add the adjective of the
lineage in brackets (unique_name); legends, which group the classes under their lineage, do not need it.
"""

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.display import load_display_names
from evaluation.hierarchy import load_hierarchy

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
GREY = "#9e9e9e"


@dataclass
class Style:
    order: List[str]                    # internal class names, in legend order
    colour: Dict[str, str]
    shown: Dict[str, str]               # internal name -> name shown
    lineage: Dict[str, str]             # class -> lineage
    lineage_order: List[str]
    lineage_colour: Dict[str, str]
    lineage_shown: Dict[str, str] = field(default_factory=dict)
    lineage_adjective: Dict[str, str] = field(default_factory=dict)
    heatmap_groups: Dict[str, List[str]] = field(default_factory=dict)

    def name(self, cls: str) -> str:
        return self.shown.get(cls, cls)

    def lineage_name(self, lineage: str) -> str:
        return self.lineage_shown.get(lineage, lineage)

    def unique_name(self, cls: str, among: Optional[Sequence[str]] = None) -> str:
        """The short name, with the lineage adjective added when another class (of 'among') has the same one."""
        others = self.order if among is None else among
        mine = self.name(cls)
        if any(o != cls and self.name(o) == mine for o in others):
            lineage = self.lineage[cls]
            return f"{mine} ({self.lineage_adjective.get(lineage, self.lineage_name(lineage))})"
        return mine

    def grouped(self, present: Sequence[str]) -> List[Tuple[Optional[str], List[str]]]:
        """
        Classes by lineage in legend order, each lineage under its name as heading; a lineage whose only class
        is shown under the lineage's own name (blastic plasmacytoid dendritic cell neoplasm, normal bone marrow)
        gets no heading (None), and legends leave a gap before it instead.
        """
        present = [c for c in self.order if c in set(present)]
        groups = []
        for lin in self.lineage_order:
            members = [c for c in present if self.lineage[c] == lin]
            if members:
                own = len(members) == 1 and self.name(members[0]) == self.lineage_name(lin)
                groups.append((None if own else self.lineage_name(lin), members))
        return groups

    def axis_labels(self, classes: Sequence[str], counts: Optional[Sequence[int]] = None, width: int = 24) -> List[str]:
        """Names for a class axis: unique among the classes, with the count in brackets, wrapped onto two lines."""
        labels = []
        for k, c in enumerate(classes):
            text = self.unique_name(c, classes) + (f" ({counts[k]})" if counts is not None else "")
            lines = textwrap.wrap(text, width) if len(text) > width else [text]
            labels.append(lines[0] if len(lines) == 1 else lines[0] + "\n" + " ".join(lines[1:]))
        return labels

    def legend_entries(self, present: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
        """(label, class) rows of a grouped legend: headings and gaps have no class; a gap has an empty label."""
        rows = []
        for k, (heading, members) in enumerate(self.grouped(present)):
            if heading:
                rows.append((heading, None))
            elif k:
                rows.append(("", None))
            rows.extend((self.name(c), c) for c in members)
        return rows


def load_style(classes: Sequence[str], colours: Optional[str] = None, display: Optional[str] = None,
               hierarchy: Optional[str] = None, names: Optional[str] = None) -> Style:
    classes = sorted(set(classes))
    colours_path = Path(colours) if colours else CONFIGS / "class_colours.json"
    content = json.loads(colours_path.read_text()) if colours_path.exists() else {}
    class_colour = content.get("classes", {})
    lineage_colour = content.get("lineages", {})

    hierarchy_path = hierarchy or (CONFIGS / "class_hierarchy.json")
    h = load_hierarchy(str(hierarchy_path) if Path(hierarchy_path).exists() else None, classes)
    display_path = display or (CONFIGS / "display_names.json")
    shown = load_display_names(str(display_path), classes) if Path(display_path).exists() else {}
    names_path = Path(names) if names else CONFIGS / "figure_names.json"
    figure_names = json.loads(names_path.read_text()) if names_path.exists() else {}
    for cls, short in figure_names.get("classes", {}).items():
        if cls in classes:
            if not str(short).strip():
                raise ValueError(f"{names_path}: empty name for {cls!r}")
            shown[cls] = str(short).strip()

    listed = [c for c in class_colour if c in classes]
    order = listed + sorted(c for c in classes if c not in class_colour)
    colour = {c: class_colour.get(c) or lineage_colour.get(h.lineage_of[c]) or GREY for c in classes}
    lineage_order = list(dict.fromkeys(h.lineage_of[c] for c in order))
    lineage_colour = {lin: lineage_colour.get(lin, GREY) for lin in lineage_order}
    return Style(order=order, colour=colour, shown=shown, lineage=dict(h.lineage_of), lineage_order=lineage_order,
                 lineage_colour=lineage_colour, lineage_shown=dict(figure_names.get("lineages", {})),
                 lineage_adjective=dict(figure_names.get("lineage_adjectives", {})),
                 heatmap_groups={k: list(v) for k, v in figure_names.get("heatmap_groups", {}).items()})
