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


def _cross(p, q, r, s) -> bool:
    """True when segment pq properly crosses segment rs."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return d1 * d2 < 0 and d3 * d4 < 0


def _hits_box(p, q, box) -> bool:
    """True when segment pq passes through the box (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    if any(x0 < a[0] < x1 and y0 < a[1] < y1 for a in (p, q)):
        return True
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(_cross(p, q, corners[i], corners[(i + 1) % 4]) for i in range(4))


def _along(p, q, r, s, tolerance: float) -> bool:
    """True when segments pq and rs run nearly the same way (within 12 degrees) closer than tolerance."""
    u, v = np.subtract(q, p), np.subtract(s, r)
    nu, nv = np.hypot(*u), np.hypot(*v)
    if nu == 0 or nv == 0 or abs(np.dot(u, v)) / (nu * nv) < np.cos(np.deg2rad(12)):
        return False

    def gap(a, b, c):                                       # distance from point a to segment bc
        bc, ba = np.subtract(c, b), np.subtract(a, b)
        f = np.clip(np.dot(ba, bc) / max(np.dot(bc, bc), 1e-12), 0, 1)
        return np.hypot(*(ba - f * bc))
    return min(gap(p, r, s), gap(q, r, s), gap(r, p, q), gap(s, p, q)) < tolerance


def place_labels(ax, fig, coords: np.ndarray, classes: np.ndarray, present: Sequence[str], texts: dict,
                 fontsize: float = 5.0, rings: int = 14) -> None:
    """
    Writes each class name near the median of its arrays without overlapping another name. Classes are placed
    from the largest down; each tries positions on rings around its median and takes the one with the lowest
    cost: its distance from the median, the arrays of other classes it covers, and any crossing of its leader
    line with names and lines already placed. A name placed away from its median gets a thin line to it.
    Deterministic.
    """
    from matplotlib import patheffects

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(coords)
    frame = ax.get_window_extent(renderer)
    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    boxes, lines = [], []
    order = sorted(present, key=lambda c: -int(np.sum(classes == c)))
    angles = np.deg2rad(np.arange(0, 360, 20))
    for c in order:
        anchor = tuple(np.median(pts[classes == c], axis=0))
        probe = ax.text(0, 0, texts[c], fontsize=fontsize, ha="center", va="center")
        size = probe.get_window_extent(renderer)
        probe.remove()
        w, h = size.width + 2.0, size.height + 1.0
        others = pts[classes != c]
        best, best_cost = None, np.inf
        for k in range(rings + 1):
            radius = k * 0.9 * h
            for a in (angles if k else [0.0]):
                x, y = anchor[0] + radius * np.cos(a), anchor[1] + radius * np.sin(a)
                box = (x - w / 2, x + w / 2, y - h / 2, y + h / 2)
                if box[0] < frame.x0 or box[1] > frame.x1 or box[2] < frame.y0 or box[3] > frame.y1:
                    continue
                if any(box[0] < q[1] and box[1] > q[0] and box[2] < q[3] and box[3] > q[2] for q in boxes):
                    continue
                covered = int(np.sum((others[:, 0] > box[0]) & (others[:, 0] < box[1]) & (others[:, 1] > box[2])
                                     & (others[:, 1] < box[3])))
                cost = radius / h + 1.5 * covered
                cost += 3.0 * sum(_hits_box(seg[0], seg[1], box) for seg in lines)
                if radius > 0.9 * h:
                    end = (float(np.clip(anchor[0], box[0], box[1])), float(np.clip(anchor[1], box[2], box[3])))
                    cost += 3.0 * sum(_cross(anchor, end, seg[0], seg[1]) or _along(anchor, end, seg[0], seg[1], h)
                                      for seg in lines)
                    cost += 3.0 * sum(_hits_box(anchor, end, q) for q in boxes)
                if cost < best_cost:
                    best, best_cost = (x, y, box), cost
            if best is not None and best_cost <= 0.9 * (k + 1):
                break
        if best is None:                                   # no free place on the rings: write it at the median
            best = (anchor[0], anchor[1], (anchor[0] - w / 2, anchor[0] + w / 2, anchor[1] - h / 2,
                                           anchor[1] + h / 2))
        x, y, box = best
        boxes.append(box)
        inverse = ax.transData.inverted()
        tx, ty = inverse.transform((x, y))
        ax.text(tx, ty, texts[c], fontsize=fontsize, ha="center", va="center", path_effects=halo, zorder=5)
        if np.hypot(x - anchor[0], y - anchor[1]) > 0.9 * h:
            end = (float(np.clip(anchor[0], box[0], box[1])), float(np.clip(anchor[1], box[2], box[3])))
            lines.append((anchor, end))
            (ex, ey), (bx, by) = inverse.transform([end, anchor])
            ax.plot([ex, bx], [ey, by], color="#525252", lw=0.3, zorder=4)


def plot_map(coords: np.ndarray, classes: Sequence[str], style: Style, path_stem, labelled: bool = False,
             colour_by: str = "class", highlight: Optional[np.ndarray] = None, title: str = "") -> None:
    """
    Arrays as points in the class (or lineage) colours; the legend lists classes by lineage. With labelled, each
    class is also named beside its arrays (place_labels). highlight marks arrays (for example of classes outside
    the model) with a black outline, in the map and in the legend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    classes = np.asarray(classes, dtype=object)
    present = [c for c in style.order if c in set(classes)]
    outlined = set(classes[highlight]) if highlight is not None and highlight.any() else set()
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    fig.subplots_adjust(left=0.02, right=0.66 if colour_by == "class" else 0.72, top=0.95, bottom=0.03)
    order = np.argsort([-np.sum(classes == c) for c in classes], kind="stable")   # small classes drawn on top
    colours = np.array([style.colour[c] if colour_by == "class" else style.lineage_colour[style.lineage[c]]
                        for c in classes], dtype=object)
    ax.scatter(coords[order, 0], coords[order, 1], s=5, c=list(colours[order]), lw=0, rasterized=True)
    if outlined:
        ax.scatter(coords[highlight, 0], coords[highlight, 1], s=9, facecolors="none", edgecolors="black",
                   linewidths=0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, loc="left")
    if labelled:
        place_labels(ax, fig, coords, classes, present, {c: style.unique_name(c, present) for c in present})

    handles, labels = [], []
    if colour_by == "class":
        for label, c in style.legend_entries(present):
            if c is None:                                  # a lineage heading, or a gap
                handles.append(Line2D([], [], lw=0))
                labels.append(label)
                continue
            handles.append(Line2D([], [], marker="o", lw=0, markersize=3.5, markerfacecolor=style.colour[c],
                                  markeredgecolor="black" if c in outlined else "none",
                                  markeredgewidth=0.5 if c in outlined else 0))
            labels.append(f"{label} ({int(np.sum(classes == c))})")
    else:
        for lin in style.lineage_order:
            n = sum(int(np.sum(classes == c)) for c in present if style.lineage[c] == lin)
            if n:
                handles.append(Line2D([], [], marker="o", lw=0, markersize=3.5,
                                      markerfacecolor=style.lineage_colour[lin], markeredgewidth=0))
                labels.append(f"{style.lineage_name(lin)} ({n:,})")
    legend = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.665 if colour_by == "class" else 0.725,
                                                                           0.5),
                        fontsize=5, frameon=False, handletextpad=0.3, borderaxespad=0, labelspacing=0.35)
    if colour_by == "class":
        for text, handle in zip(legend.get_texts(), handles):
            if isinstance(handle, Line2D) and handle.get_marker() in (None, "None", ""):
                text.set_fontweight("bold")
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)
