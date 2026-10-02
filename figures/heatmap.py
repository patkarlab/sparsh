"""
Signature heatmap, and global methylation by class.

The heatmap is drawn once for each group of lineages of configs/figure_names.json (heatmap_groups): by default
acute myeloid leukemia with acute leukemia of ambiguous lineage, blastic plasmacytoid dendritic cell neoplasm and
normal bone marrow, and B- and T-lymphoblastic leukemia. Within a group, each class is compared with the other
arrays of the group, so the heatmap shows what separates the classes of those lineages rather than the lineages
themselves. Signature CpGs are, for each class, the CpGs that separate it most clearly from those arrays, in
both directions: among CpGs off the sex chromosomes whose mean beta differs by at least min_delta (0.2 by
default), the per_class (25) with the largest positive t statistic of stats.py and the per_class with the
largest negative one. A CpG chosen for several classes is shown once, in the block of the first class in the
legend order.

The heatmap shows the beta values of these CpGs (rows, in blocks by the class they mark, hypermethylated CpGs of
a block first) in every array (columns, grouped by class in the legend order, which groups the lineages).
Within a class the arrays are ordered by their signature score, the mean beta of the class's hypermethylated
signature CpGs minus that of its hypomethylated ones. Bars above the columns give lineage and class, a bar
beside the rows the class each CpG marks.

Global methylation is the mean beta over all CpGs of an array, drawn per class as a box plot; the
hypermethylation of IDH-mutated leukaemia is the best-known difference.
"""

import warnings
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .style import Style


def signature_cpgs(statistics: Dict[str, pd.DataFrame], order: Sequence[str], annotation: pd.DataFrame,
                   per_class: int = 25, min_delta: float = 0.2) -> pd.DataFrame:
    sex = annotation["chrom"].isin(["chrX", "chrY"]).to_numpy()
    cpgs = annotation.index.to_numpy()
    used, rows = set(), []
    for cls in order:
        st = statistics[cls]
        ok = st["t"].notna().to_numpy() & ~sex
        for direction, sign in (("hyper", 1.0), ("hypo", -1.0)):
            mask = ok & (sign * st["delta"].to_numpy() >= min_delta)
            cand = np.flatnonzero(mask)
            cand = cand[np.argsort(-sign * st["t"].to_numpy()[cand], kind="stable")]
            chosen = [i for i in cand if i not in used][:per_class]
            used.update(chosen)
            for i in chosen:
                rows.append({"cpg": cpgs[i], "class": cls, "direction": direction,
                             "delta": round(float(st["delta"].iat[i]), 4), "t": round(float(st["t"].iat[i]), 2),
                             "fdr": float(st["fdr"].iat[i]), "mean_in": round(float(st["mean_in"].iat[i]), 4),
                             "mean_out": round(float(st["mean_out"].iat[i]), 4), "chrom": annotation["chrom"].iat[i],
                             "pos": annotation["pos"].iat[i], "genes": annotation["genes"].iat[i], "index": int(i)})
    return pd.DataFrame(rows)


def column_order(betas: np.ndarray, classes: np.ndarray, order: Sequence[str], signatures: pd.DataFrame) -> np.ndarray:
    """Arrays grouped by class in the legend order; within a class, by signature score (highest first)."""
    columns = []
    for cls in order:
        idx = np.flatnonzero(classes == cls)
        if not len(idx):
            continue
        mine = signatures[signatures["class"] == cls]
        hyper = mine.loc[mine["direction"] == "hyper", "index"].to_numpy(dtype=int)
        hypo = mine.loc[mine["direction"] == "hypo", "index"].to_numpy(dtype=int)
        score = np.zeros(len(idx))
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # arrays with no value at these CpGs
            if len(hyper):
                score += np.nan_to_num(np.nanmean(betas[np.ix_(idx, hyper)], axis=1))
            if len(hypo):
                score -= np.nan_to_num(np.nanmean(betas[np.ix_(idx, hypo)], axis=1))
        columns.extend(idx[np.argsort(-score, kind="stable")])
    return np.array(columns, dtype=int)


def plot_heatmap(betas: np.ndarray, classes: np.ndarray, signatures: pd.DataFrame, columns: np.ndarray,
                 style: Style, path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
    from matplotlib.lines import Line2D

    rows = signatures["index"].to_numpy(dtype=int)
    data = np.asarray(betas[np.ix_(columns, rows)], dtype=np.float32).T        # CpGs x arrays
    col_classes = classes[columns]
    present = [c for c in style.order if c in set(col_classes)]

    fig = plt.figure(figsize=(7.1, 8.2))
    left, width = 0.06, 0.63
    ax = fig.add_axes([left, 0.05, width, 0.88])
    ax_lin = fig.add_axes([left, 0.955, width, 0.012])
    ax_cls = fig.add_axes([left, 0.94, width, 0.012])
    ax_row = fig.add_axes([left - 0.015, 0.05, 0.01, 0.88])
    ax_bar = fig.add_axes([left + width + 0.015, 0.06, 0.012, 0.12])

    beta_map = LinearSegmentedColormap.from_list("beta", ["#2166ac", "#f7f7f7", "#b2182b"])
    beta_map.set_bad("#d9d9d9")
    image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", interpolation="nearest", cmap=beta_map, vmin=0,
                      vmax=1, rasterized=True)
    names = present
    class_index = np.array([names.index(c) for c in col_classes])
    ax_cls.imshow(class_index[None, :], aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([to_rgb(style.colour[c]) for c in names]), vmin=-0.5, vmax=len(names) - 0.5)
    lineages = [lin for lin in style.lineage_order if any(style.lineage[c] == lin for c in names)]
    lin_index = np.array([lineages.index(style.lineage[c]) for c in col_classes])
    ax_lin.imshow(lin_index[None, :], aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([to_rgb(style.lineage_colour[lin]) for lin in lineages]), vmin=-0.5,
                  vmax=len(lineages) - 0.5)
    row_classes = signatures["class"].to_numpy()
    row_index = np.array([names.index(c) if c in names else 0 for c in row_classes])
    ax_row.imshow(row_index[:, None], aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([to_rgb(style.colour[c]) for c in names]), vmin=-0.5, vmax=len(names) - 0.5)
    for a in (ax_lin, ax_cls, ax_row):
        a.set_xticks([])
        a.set_yticks([])
        for side in a.spines.values():
            side.set_visible(False)
    ax_lin.set_ylabel("Lineage", fontsize=5, rotation=0, ha="right", va="center")
    ax_cls.set_ylabel("Class", fontsize=5, rotation=0, ha="right", va="center")
    bounds = np.flatnonzero(col_classes[1:] != col_classes[:-1]) + 0.5
    for x in bounds:
        ax.axvline(x, color="white", lw=0.3)
    row_bounds = np.flatnonzero(row_classes[1:] != row_classes[:-1]) + 0.5
    for y in row_bounds:
        ax.axhline(y, color="white", lw=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(columns)} arrays, by class", fontsize=6)
    ax.set_ylabel(f"{len(rows)} signature CpGs, by the class they mark", fontsize=6, labelpad=10)

    handles, labels = [], []
    for label, c in style.legend_entries(names):
        if c is None:                                      # a lineage heading, or a gap
            handles.append(Line2D([], [], lw=0))
            labels.append(label)
            continue
        handles.append(Line2D([], [], marker="s", lw=0, markersize=4, markerfacecolor=style.colour[c],
                              markeredgewidth=0))
        labels.append(f"{label} ({int(np.sum(col_classes == c))})")
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left + width + 0.01, 0.97), fontsize=5,
                        frameon=False, handletextpad=0.3, labelspacing=0.3, borderaxespad=0)
    for text, handle in zip(legend.get_texts(), handles):
        if handle.get_marker() in (None, "None", ""):
            text.set_fontweight("bold")
    bar = fig.colorbar(image, cax=ax_bar)
    bar.set_label("beta value", fontsize=5)
    bar.ax.tick_params(labelsize=5)
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)


def global_methylation(betas: np.ndarray, block: int = 10000) -> np.ndarray:
    """Mean beta over all CpGs of every array (missing values left out)."""
    total = np.zeros(betas.shape[0])
    count = np.zeros(betas.shape[0])
    for a in range(0, betas.shape[1], block):
        b = np.asarray(betas[:, a:a + block], dtype=np.float64)
        ok = np.isfinite(b)
        total += np.where(ok, b, 0.0).sum(axis=1)
        count += ok.sum(axis=1)
    return total / np.maximum(count, 1)


def plot_global(values: np.ndarray, classes: np.ndarray, style: Style, path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in style.order if c in set(classes)]
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.33)
    data = [values[classes == c] for c in names]
    parts = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops={"color": "black", "lw": 0.8},
                       whiskerprops={"lw": 0.6}, capprops={"lw": 0.6}, boxprops={"lw": 0.6})
    for patch, c in zip(parts["boxes"], names):
        patch.set_facecolor(style.colour[c])
        patch.set_alpha(0.85)
    rng = np.random.default_rng(1)
    for k, d in enumerate(data, start=1):
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(d)), d, s=1.2, c="#252525", lw=0, alpha=0.6, rasterized=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(style.axis_labels(names, [len(d) for d in data]), rotation=90, ha="center", va="top",
                       fontsize=5, linespacing=1.1)
    ax.set_ylabel("Mean beta over all CpGs", fontsize=6)
    ax.tick_params(axis="y", labelsize=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)


def summary_table(statistics: Dict[str, pd.DataFrame], order: List[str], annotation: pd.DataFrame,
                  min_delta: float = 0.2, max_fdr: float = 0.05) -> pd.DataFrame:
    """Per class: arrays, and CpGs hyper- and hypomethylated by at least min_delta at the given false discovery rate."""
    autosome = ~annotation["chrom"].isin(["chrX", "chrY"]).to_numpy()
    rows = []
    for cls in order:
        st = statistics[cls]
        sig = (st["fdr"].to_numpy() <= max_fdr) & autosome
        rows.append({"class": cls, "arrays": int(st["n_in"].max()),
                     "hypermethylated_cpgs": int(np.sum(sig & (st["delta"].to_numpy() >= min_delta))),
                     "hypomethylated_cpgs": int(np.sum(sig & (st["delta"].to_numpy() <= -min_delta)))})
    return pd.DataFrame(rows)
