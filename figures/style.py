"""
Names, colours and order of classes and lineages in the figures.

Classes are drawn in the colours of configs/class_colours.json and listed in its order, which groups them by
lineage; classes the file does not list follow in alphabetical order, in the colour of their lineage. Names are
shown as configs/display_names.json gives them. Lineages come from the prefix rules of
configs/class_hierarchy.json, which also place classes that are not in the model, such as those added to a
supplementary figure.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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

    def name(self, cls: str) -> str:
        return self.shown.get(cls, cls)


def load_style(classes: Sequence[str], colours: Optional[str] = None, display: Optional[str] = None,
               hierarchy: Optional[str] = None) -> Style:
    classes = sorted(set(classes))
    colours_path = Path(colours) if colours else CONFIGS / "class_colours.json"
    content = json.loads(colours_path.read_text()) if colours_path.exists() else {}
    class_colour = content.get("classes", {})
    lineage_colour = content.get("lineages", {})

    hierarchy_path = hierarchy or (CONFIGS / "class_hierarchy.json")
    h = load_hierarchy(str(hierarchy_path) if Path(hierarchy_path).exists() else None, classes)
    display_path = display or (CONFIGS / "display_names.json")
    shown = load_display_names(str(display_path), classes) if Path(display_path).exists() else {}

    listed = [c for c in class_colour if c in classes]
    order = listed + sorted(c for c in classes if c not in class_colour)
    colour = {c: class_colour.get(c) or lineage_colour.get(h.lineage_of[c]) or GREY for c in classes}
    lineage_order = list(dict.fromkeys(h.lineage_of[c] for c in order))
    lineage_colour = {lin: lineage_colour.get(lin, GREY) for lin in lineage_order}
    return Style(order=order, colour=colour, shown=shown, lineage=dict(h.lineage_of), lineage_order=lineage_order,
                 lineage_colour=lineage_colour)
