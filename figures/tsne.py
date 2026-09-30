"""
t-SNE map of the training arrays.

The CpGs are the most variable ones across the arrays (5,000 by default, ranked by the standard deviation of
their beta values), leaving out CpGs on the X and Y chromosomes, whose methylation follows sex, and CpGs missing
in more than 5% of the arrays. A CpG's few missing values are filled with its mean. The distance between two
arrays is one minus the Pearson correlation of their values over those CpGs, and the map is made from that
distance matrix with t-SNE (scikit-learn, perplexity 30, a fixed seed). The CpG list, the coordinates and the
settings are written next to the figures, so a map can be redrawn or checked.
"""

import inspect
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .style import Style


def variable_cpgs(betas: np.ndarray, annotation: pd.DataFrame, n: int = 5000, max_missing: float = 0.05,
                  log=print) -> np.ndarray:
    """Indices of the n CpGs with the largest standard deviation, off the sex chromosomes and rarely missing."""
    missing = np.isnan(betas).mean(axis=0)
    sd = np.nanstd(betas, axis=0)
    ok = (missing <= max_missing) & np.isfinite(sd)
    sex = annotation["chrom"].isin(["chrX", "chrY"]).to_numpy()
    ok &= ~sex
    log(f"  CpGs considered: {int(ok.sum())} (left out: {int(sex.sum())} on X or Y, "
        f"{int((missing > max_missing).sum())} missing in more than {max_missing:.0%} of arrays)")
    order = np.argsort(-np.where(ok, sd, -np.inf), kind="stable")
    return order[:min(n, int(ok.sum()))]


def correlation_distance(x: np.ndarray) -> np.ndarray:
    """One minus the Pearson correlation between rows; missing values are first filled with the column mean."""
    x = np.asarray(x, dtype=np.float64)
    means = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), means, x)
    z = x - x.mean(axis=1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    d = 1.0 - z @ z.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0)


def run_tsne(distance: np.ndarray, perplexity: float = 30.0, seed: int = 42, iterations: int = 2000) -> np.ndarray:
    from sklearn.manifold import TSNE
    options = dict(n_components=2, metric="precomputed", init="random", perplexity=perplexity,
                   learning_rate="auto", random_state=seed)
    # scikit-learn 1.5 renamed n_iter to max_iter
    options["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = iterations
    return TSNE(**options).fit_transform(distance)


def plot_map(coords: np.ndarray, classes: Sequence[str], style: Style, path_stem, labelled: bool = False,
             colour_by: str = "class", highlight: Optional[np.ndarray] = None, title: str = "") -> None:
    """
    Arrays as points in the class (or lineage) colours; the legend lists classes by lineage. With labelled, each
    class name is also written at the median of its arrays. highlight marks arrays (for example of classes outside
    the model) with a black outline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.lines import Line2D

    classes = np.asarray(classes, dtype=object)
    present = [c for c in style.order if c in set(classes)]
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    fig.subplots_adjust(left=0.02, right=0.70 if colour_by == "class" else 0.80, top=0.95, bottom=0.03)
    order = np.argsort([-np.sum(classes == c) for c in classes], kind="stable")   # small classes drawn on top
    colours = np.array([style.colour[c] if colour_by == "class" else style.lineage_colour[style.lineage[c]]
                        for c in classes], dtype=object)
    ax.scatter(coords[order, 0], coords[order, 1], s=5, c=list(colours[order]), lw=0, rasterized=True)
    if highlight is not None and highlight.any():
        ax.scatter(coords[highlight, 0], coords[highlight, 1], s=9, facecolors="none", edgecolors="black",
                   linewidths=0.4)
    if labelled:
        halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
        for c in present:
            x, y = np.median(coords[classes == c], axis=0)
            ax.text(x, y, style.name(c), fontsize=5, ha="center", va="center", path_effects=halo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, loc="left")

    handles, labels = [], []
    if colour_by == "class":
        for lin in style.lineage_order:
            members = [c for c in present if style.lineage[c] == lin]
            if not members:
                continue
            handles.append(Line2D([], [], lw=0))
            labels.append(lin)
            for c in members:
                handles.append(Line2D([], [], marker="o", lw=0, markersize=3.5, markerfacecolor=style.colour[c],
                                      markeredgewidth=0))
                labels.append(f"{style.name(c)} ({int(np.sum(classes == c))})")
    else:
        for lin in style.lineage_order:
            n = sum(int(np.sum(classes == c)) for c in present if style.lineage[c] == lin)
            if n:
                handles.append(Line2D([], [], marker="o", lw=0, markersize=3.5,
                                      markerfacecolor=style.lineage_colour[lin], markeredgewidth=0))
                labels.append(f"{lin} ({n})")
    legend = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.705 if colour_by == "class" else 0.805,
                                                                           0.5),
                        fontsize=5, frameon=False, handletextpad=0.3, borderaxespad=0, labelspacing=0.35)
    if colour_by == "class":
        for text, handle in zip(legend.get_texts(), handles):
            if isinstance(handle, Line2D) and handle.get_marker() in (None, "None", ""):
                text.set_fontweight("bold")
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)
