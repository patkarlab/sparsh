"""
Per-class F1 score in cross-validation, drawn as in Figure 1c of Sturgeon (Vermeulen and colleagues, Nature 2023).

A training run (scripts/train.py) scores every array once, with the network of the fold that held it out, under
fixed conditions, and writes the predictions of each condition to cv_predictions_<condition>.csv. For each class
and condition, precision is the share of the arrays predicted as the class that belong to it, recall the share of
the class's arrays predicted as the class, and F1 their harmonic mean, 2 TP / (2 TP + FP + FN), all on the most
probable class without a threshold. The figure shows the conditions side by side for every class, the first one
lightest. By default these are the full arrays ('dense', the array profile as measured) and simulated nanopore
reads covering 30% of the model's CpGs without dilution ('binary_0.30'), the reference condition of the model card.
"""

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .style import Style

SIMULATION_NAMES = {"binary": "Simulated nanopore", "reads": "Simulated nanopore reads",
                    "oneread": "Simulated single reads", "mask": "Masked arrays"}


def parse(name: str):
    """'binary-blast30_0.30' -> ('binary', 0.30, 0.30, 0.0): simulation, covered share, blasts, call error
    (the names of evaluation/conditions.py, read here without importing the training code)."""
    label, _, cov = str(name).rpartition("_")
    try:
        fraction = float(cov)
    except ValueError:
        return str(name), None, 1.0, 0.0
    sim, *parts = label.split("-")
    blast, error = 1.0, 0.0
    for part in parts:
        if part.startswith("blast"):
            blast = float(part[5:]) / 100.0
        elif part.startswith("err"):
            error = float(part[3:]) / 100.0
    return sim, fraction, blast, error


def condition_label(name: str) -> str:
    if name == "dense":
        return "Full arrays"
    sim, fraction, blast, error = parse(name)
    text = SIMULATION_NAMES.get(sim, sim)
    if fraction is not None:
        text += f", {fraction:.0%} of CpGs"
    if blast < 1.0:
        text += f", {blast:.0%} blasts"
    if error:
        text += f", {error:.0%} call errors"
    return text


def read_predictions(model_dir: Path, condition: str) -> pd.DataFrame:
    path = Path(model_dir) / f"cv_predictions_{condition}.csv"
    if not path.exists():
        found = sorted(p.name[len("cv_predictions_"):-4] for p in Path(model_dir).glob("cv_predictions_*.csv"))
        raise FileNotFoundError(f"{path} not found; the run has the conditions {found}")
    table = pd.read_csv(path, usecols=["sample_id", "true_label", "prediction"], dtype=str)
    return table


def f1_table(model_dir: Path, conditions: Sequence[str], classes: Sequence[str], style: Style) -> pd.DataFrame:
    """One row per class: arrays, and precision, recall and F1 for each condition."""
    rows = {c: {"class": c, "shown": style.name(c), "lineage": style.lineage_name(style.lineage[c])}
            for c in classes}
    sizes = None
    for cond in conditions:
        table = read_predictions(model_dir, cond)
        truth, pred = table["true_label"].to_numpy(), table["prediction"].to_numpy()
        if sizes is None:
            sizes = {c: int(np.sum(truth == c)) for c in classes}
        elif {c: int(np.sum(truth == c)) for c in classes} != sizes:
            raise ValueError(f"cv_predictions_{cond}.csv does not hold the same arrays as the other conditions")
        for c in classes:
            tp = int(np.sum((truth == c) & (pred == c)))
            fp = int(np.sum((truth != c) & (pred == c)))
            fn = int(np.sum((truth == c) & (pred != c)))
            rows[c][f"precision_{cond}"] = round(tp / (tp + fp), 4) if tp + fp else 0.0
            rows[c][f"recall_{cond}"] = round(tp / (tp + fn), 4) if tp + fn else np.nan
            rows[c][f"f1_{cond}"] = round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else np.nan
    frame = pd.DataFrame([rows[c] for c in classes])
    frame.insert(3, "arrays", [sizes[c] for c in classes])
    return frame


def plot_f1(table: pd.DataFrame, conditions: Sequence[str], style: Style, path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    classes: List[str] = list(table["class"])
    n, k = len(classes), len(conditions)
    shades = np.linspace(0.3, 1.0, k) if k > 1 else np.array([1.0])
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.31)
    width = 0.84 / k
    x = np.arange(n)
    for j, cond in enumerate(conditions):
        values = table[f"f1_{cond}"].to_numpy(dtype=float)
        offsets = x - 0.42 + width * (j + 0.5)
        ax.bar(offsets, np.nan_to_num(values), width=width * 0.92, color=[style.colour[c] for c in classes],
               alpha=float(shades[j]), edgecolor="none")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("F1 score", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(style.axis_labels(classes, list(table["arrays"])), rotation=90, ha="center", va="top",
                       fontsize=5, linespacing=1.1)
    ax.tick_params(axis="y", labelsize=5)
    ax.grid(axis="y", color="#e0e0e0", lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor="#525252", alpha=float(shades[j]), edgecolor="none") for j in range(k)]
    ax.legend(handles, [condition_label(c) for c in conditions], fontsize=5, frameon=False, ncol=k,
              loc="lower left", bbox_to_anchor=(0.0, 1.02), handlelength=1.2, borderaxespad=0)
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)
