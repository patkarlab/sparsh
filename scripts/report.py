#!/usr/bin/env python3
r"""
SPARSH per-case reports: one self-contained HTML page per sample, from the output folder of scripts/predict.py.

Examples
    python scripts/report.py --predictions_dir predictions/batch1 --output_dir reports/batch1

    python scripts/report.py --predictions_dir predictions/batch1 --output_dir reports/batch1 \
        --sample_sheet samples.csv --karyotype_dir copy_number/ --samples S01 S07 --title "Batch 1"

Written into --output_dir (a new folder; an existing non-empty folder only with --overwrite):
  index.html          every sample in one table, sortable by any column and filtered as one types, each sample
                      linked to its page;
  cases/<SAMPLE>.html one page per sample: the result (lineage, reported entity, its probability and the level at
                      which it is reported), a chart of the five most probable classes, the interpretation text and
                      caveats, what cross-validation on the training arrays showed at a similar coverage, the
                      copy-number karyotype when one is given, the input and its quality, and a note on the method.
                      Pages are self-contained (inline CSS and SVG, figures embedded) and print on A4;
  report_summary.csv  one row per sample with the fields of the result panel and the interpretation headline.

Inputs
  --predictions_dir  output folder of scripts/predict.py: predictions.csv, and prediction_info.json,
                     display_names_used.json and class_hierarchy_used.json when present.
  --model_dir        the training run that made the predictions (default: the one named in prediction_info.json).
                     Its config.json, cv_metrics_by_condition.csv and cv_recall_by_class.csv give the model details
                     and the evidence from cross-validation; without them that section is left out.
  --interpretation   text per class, family and lineage (default configs/interpretation.json; see that file).
  --sample_sheet     CSV with a column 'sample' and any other columns (for example centre, sample_type,
                     blast_percent, collection_date), shown as sample information. Names are matched as predict.py
                     names samples: stripped and upper-cased. Do not put names of patients in it.
  --karyotype_dir    folder of <SAMPLE>.karyotype.json files from the copy-number module, each optionally with
                     <SAMPLE>.karyotype.svg or .png; whatever the JSON holds is shown, missing keys are left out.
  --threshold        reporting threshold (default: the prediction run's, else 0.90). Another value recomputes the
                     reported calls from the class probabilities (see report/inputs.py).

Class names are shown as in predictions.csv (configs/display_names.json); colours come from
configs/class_colours.json. The reports are for research use only and say so on every page.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from report import ReportInputError, load_inputs, write_report  # noqa: E402
from report.inputs import DEFAULT_INTERPRETATION  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SPARSH per-case HTML reports from the outputs of scripts/predict.py")
    p.add_argument("--predictions_dir", required=True, help="Output folder of scripts/predict.py")
    p.add_argument("--output_dir", required=True, help="New folder for the reports")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output folder")
    p.add_argument("--model_dir", default=None,
                   help="Training run folder (default: model_dir recorded in prediction_info.json)")
    p.add_argument("--interpretation", default=str(DEFAULT_INTERPRETATION), help="Interpretation text (JSON)")
    p.add_argument("--sample_sheet", default=None, help="CSV with a column 'sample' and further columns to show")
    p.add_argument("--karyotype_dir", default=None, help="Folder of <SAMPLE>.karyotype.json (and .svg or .png)")
    p.add_argument("--samples", nargs="+", default=None, help="Only these samples (names as in predictions.csv)")
    p.add_argument("--title", default=None, help="Report title (default: SPARSH methylation classification report)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Reporting threshold (default: the prediction run's, else 0.90)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).expanduser()
    if out.exists() and not out.is_dir():
        sys.exit(f"Output path {out} is a file. Choose a new --output_dir.")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out} is not empty. Choose a new --output_dir or pass --overwrite.")
    if args.title is not None and not args.title.strip():
        sys.exit("--title must not be empty")
    try:
        ctx, cases = load_inputs(args.predictions_dir, model_dir=args.model_dir, interpretation=args.interpretation,
                                 sample_sheet=args.sample_sheet, karyotype_dir=args.karyotype_dir,
                                 samples=args.samples, title=args.title.strip() if args.title else None,
                                 threshold=args.threshold)
    except ReportInputError as e:
        sys.exit(f"Error: {e}")
    out.mkdir(parents=True, exist_ok=True)
    results = write_report(ctx, cases, out)

    levels = {}
    for r in results.values():
        levels[r.level] = levels.get(r.level, 0) + 1
    print(f"Model: {ctx.model.name or 'folder not recorded'} ({ctx.model.n_classes} classes); reporting threshold "
          f"{ctx.threshold:.2f}"
          + ("" if ctx.cv is not None else "; no cross-validation tables, that section is left out"))
    print(f"Reports for {len(cases)} samples: " + ", ".join(
        f"{lev} {levels.get(lev, 0)}" for lev in ("subtype", "no_subtype", "family", "lineage", "none")))
    with_kt = sum(1 for c in cases if c.karyotype is not None)
    if ctx.has_karyotypes:
        print(f"Copy-number karyotypes shown for {with_kt} sample(s)")
    print(f"Written to {out}: index.html, cases/ ({len(cases)} pages), report_summary.csv")


if __name__ == "__main__":
    main()
