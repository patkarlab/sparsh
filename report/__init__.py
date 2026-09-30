"""
SPARSH per-case reports: one self-contained HTML page per sample from the output folder of scripts/predict.py.

The package turns predictions.csv, and whatever else is at hand, into pages a haematologist can read and print:

  inputs.py  reads and checks the inputs (predictions, the record of the prediction run, class names, hierarchy,
             colours, interpretation text, sample sheet, copy-number karyotypes and the model's cross-validation
             tables) and gathers them into one Case per sample and one Context shared by all pages;
  text.py    writes the sentences of a report from those data: the result, the reporting level, the caveats,
             the evidence from cross-validation and the explanation of the method;
  svg.py     draws the class-probability chart as inline SVG, without a plotting library;
  style.py   holds the style sheets written into every page, for the screen and for A4 print;
  pages.py   assembles the case pages, the index page and report_summary.csv.

scripts/report.py is the command-line entry point. Everything is computed from the files given; nothing is fetched
from a network, and the pages load no fonts, scripts or images from outside themselves.
"""

from .inputs import Case, Context, ReportInputError, load_inputs
from .pages import write_report

__all__ = ["Case", "Context", "ReportInputError", "load_inputs", "write_report"]
