#!/usr/bin/env python3
"""
Threshold analysis of a finished training run, from its out-of-fold cross-validation predictions only.

    python scripts/threshold_analysis.py --run_dir runs/v04_labels --output threshold_analysis.csv

    # also record the figures of provisional calls for the report pages (configs/cv_caution_bands.csv)
    python scripts/threshold_analysis.py --run_dir runs/v04_labels --model_dir /path/to/sparsh_model_0.4.0 \
        --write_bands configs/cv_caution_bands.csv

For every simulated condition the run saved (cv_predictions_<condition>.csv), it gives:
  share and accuracy of the most probable class at or above the threshold (0.90 by default, the run's own);
  share and accuracy from the caution threshold (0.80 by default) up to the threshold, the calls that
  scripts/predict.py reports as provisional when no family reaches the threshold;
  share and accuracy at or above the caution threshold;
  a receiver operating characteristic analysis of the maximum probability, each array labelled correctly classified
  or misclassified, with the area under the curve and the cut-off that maximizes the Youden index (sensitivity +
  specificity - 1), with its sensitivity and specificity.
Accuracy is that of the most probable class against the array's class, without the family and lineage fallback
(scripts/hierarchy_report.py gives that).

It reads no nanopore sample and no label of one. Its figures describe the arrays of cross-validation with simulated
nanopore reads.

--write_bands adds the share and accuracy from the caution threshold up to the threshold, for every condition, to
the table that scripts/report.py quotes on the page of a provisional call. Rows are keyed by the SHA-256 checksum of
the config.json of --model_dir (default --run_dir), so that the report uses them only for that model; give the
folder of the model as it is deployed. Rows of the same model and thresholds already in the file are replaced.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.hierarchy import DEFAULT_CAUTION_THRESHOLD, check_thresholds  # noqa: E402

BAND_COLUMNS = ["model_config_sha256", "model_name", "condition", "threshold", "caution_threshold", "arrays",
                "band_share", "band_accuracy"]


def band(conf: np.ndarray, correct: np.ndarray, low: float, high: float = None):
    """Number of arrays with conf in [low, high) (or >= low), and the accuracy of their most probable class."""
    mask = conf >= low if high is None else (conf >= low) & (conf < high)
    n = int(mask.sum())
    return n, (float(correct[mask].mean()) if n else float("nan"))


def analyse(path: Path, threshold: float, caution: float) -> dict:
    p = pd.read_csv(path, usecols=["true_label", "prediction", "confidence"],
                    dtype={"true_label": str, "prediction": str})
    conf = p["confidence"].to_numpy(dtype=float)
    correct = (p["true_label"] == p["prediction"]).to_numpy()
    total = len(p)
    row = {"condition": path.name[len("cv_predictions_"):-len(".csv")], "arrays": total,
           "accuracy": float(correct.mean()) if total else float("nan")}
    for name, low, high in (("at_threshold", threshold, None), ("band", caution, threshold),
                            ("at_caution", caution, None)):
        n, acc = band(conf, correct, low, high)
        row[f"share_{name}"] = n / total if total else float("nan")
        row[f"accuracy_{name}"] = acc
    if total == 0 or correct.all() or not correct.any():
        row.update(auc=float("nan"), youden_cutoff=float("nan"), sensitivity=float("nan"),
                   specificity=float("nan"))
        return row
    fpr, tpr, cut = roc_curve(correct.astype(int), conf)
    j = np.where(np.isfinite(cut), tpr - fpr, -np.inf)
    best = int(np.argmax(j))
    row.update(auc=float(roc_auc_score(correct.astype(int), conf)), youden_cutoff=float(cut[best]),
               sensitivity=float(tpr[best]), specificity=float(1 - fpr[best]))
    return row


def sort_key(name: str):
    """Conditions grouped by simulation, then by coverage."""
    m = re.search(r"_(\d+(?:\.\d+)?)$", name)
    return (name[:m.start()] if m else name, float(m.group(1)) if m else 0.0)


def write_bands(table: pd.DataFrame, model_dir: Path, threshold: float, caution: float, out: Path) -> None:
    config = model_dir / "config.json"
    if not config.exists():
        sys.exit(f"{config} not found: --model_dir must be a model folder")
    checksum = hashlib.sha256(config.read_bytes()).hexdigest()
    new = pd.DataFrame({"model_config_sha256": checksum, "model_name": model_dir.resolve().name,
                        "condition": table["condition"], "threshold": threshold, "caution_threshold": caution,
                        "arrays": table["arrays"], "band_share": table["share_band"],
                        "band_accuracy": table["accuracy_band"]})[BAND_COLUMNS]
    if out.exists():
        old = pd.read_csv(out, dtype={"model_config_sha256": str, "condition": str})
        same = ((old["model_config_sha256"] == checksum) & ((old["threshold"] - threshold).abs() < 1e-6)
                & ((old["caution_threshold"] - caution).abs() < 1e-6))
        new = pd.concat([old[~same], new], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False, float_format="%.5f")
    print(f"\nFigures of provisional calls for {model_dir.resolve().name} (config.json {checksum[:12]}...) "
          f"written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="Folder of a finished training run (cv_predictions_*.csv)")
    ap.add_argument("--threshold", type=float, default=None, help="Default: the run's stored threshold, else 0.90")
    ap.add_argument("--caution_threshold", type=float, default=DEFAULT_CAUTION_THRESHOLD)
    ap.add_argument("--output", default=None, help="CSV file for the full table (optional)")
    ap.add_argument("--write_bands", default=None, help="Table of provisional calls to update for the reports")
    ap.add_argument("--model_dir", default=None, help="Model folder whose config.json keys the rows "
                                                      "(default --run_dir)")
    args = ap.parse_args()

    run = Path(args.run_dir).expanduser()
    threshold = args.threshold
    if threshold is None:
        threshold = 0.90
        if (run / "config.json").exists():
            threshold = json.loads((run / "config.json").read_text()).get("inference", {}).get("threshold", 0.90)
    try:
        caution = check_thresholds(float(threshold), args.caution_threshold)
    except ValueError as e:
        sys.exit(str(e))
    if caution is None:
        sys.exit("--caution_threshold 0 leaves nothing to analyse between the two thresholds")
    files = sorted(run.glob("cv_predictions_*.csv"), key=lambda f: sort_key(f.name[len("cv_predictions_"):-4]))
    if not files:
        sys.exit(f"{run}: no cv_predictions_<condition>.csv files")
    table = pd.DataFrame([analyse(f, float(threshold), caution) for f in files])
    print(f"Run {run.name}: threshold {float(threshold):.2f}, caution threshold {caution:.2f} "
          "(out-of-fold predictions of cross-validation; accuracy of the most probable class)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(table.round(3).to_string(index=False))
    if args.output:
        table.to_csv(args.output, index=False, float_format="%.5f")
        print(f"\nWritten to {args.output}")
    if args.write_bands:
        model_dir = Path(args.model_dir).expanduser() if args.model_dir else run
        write_bands(table, model_dir, float(threshold), caution, Path(args.write_bands).expanduser())


if __name__ == "__main__":
    main()
