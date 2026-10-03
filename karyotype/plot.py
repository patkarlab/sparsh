"""
Copy-number figures for one sample: the whole genome, and one panel per chromosome.

Gray points are the log ratios of the analysis bins; lines are the segments, red for gains and blue for losses
against the normal copy number of the chromosome, dark gray where unchanged. The log ratios are drawn so that
zero is two copies in the leukemic cells, at the fitted fraction and baseline: gains lie above the line and
losses below it, whatever the ploidy of the leukemia. (The files keep the log ratios as computed, centered on
the median autosomal bin.) The vertical range follows the segments, from -1 to 1 at least and -2 to 2 at most,
and bins beyond it are drawn on its edge. The right-hand axis of the genome figure marks the levels expected
for 0 to 8 copies, so a segment can be read against it directly.

The chromosome panels show the same bins and segments chromosome by chromosome, with the bands beneath, the
centromere as a dashed line and the levels of 1 to 4 copies as lines, so that the calls can be checked by eye,
arm by arm. Changed arms are named in each panel.
"""

from pathlib import Path

import matplotlib.ticker
import numpy as np

from .fit import expected_log2
from .genome import CHROMOSOMES, chrom_label
from .normalise import normal_copies

GAIN, LOSS, NEUTRAL, POINTS = "#b2182b", "#2166ac", "#404040", "#9e9e9e"
STAIN = {"gneg": "#ffffff", "gpos25": "#d9d9d9", "gpos33": "#cccccc", "gpos50": "#a6a6a6", "gpos66": "#8c8c8c",
         "gpos75": "#737373", "gpos100": "#404040", "acen": "#d6604d", "gvar": "#c6dbef", "stalk": "#9ecae1"}


def _subtitle(fit) -> str:
    if fit.purity is None:
        return "No change away from the baseline, so the leukemic fraction is not estimable."
    return f"Leukemic fraction {fit.purity:.2f}, baseline {fit.baseline} copies, computed from read depth."


def diploid_level(fit) -> float:
    """Log2 ratio (as computed, against the median autosomal bin) of two copies in the leukemic cells."""
    if fit.purity is None:
        return 0.0
    return float(expected_log2(2, 2, fit.purity, fit.baseline)) + fit.offset


def copy_levels(fit, chrom: str, sex: str, numbers, zero: float) -> list:
    """(copies, level) for the given copy numbers of one chromosome, on the plotted scale (0 = two copies)."""
    if fit.purity is None:
        return []
    n = normal_copies(chrom, sex)
    return [(c, float(expected_log2(c, n, fit.purity, fit.baseline)) + fit.offset - zero) for c in numbers]


def _top(means) -> float:
    """Upper end of the vertical range: at least 1, at most 2 (log2 ratio), following the segments."""
    means = np.asarray(means, dtype=float)
    means = means[np.isfinite(means)] if means.size else means
    return float(np.clip(np.max(np.abs(means)) + 0.4, 1.0, 2.0)) if means.size else 1.0


def plot_genome(path_stem: Path, sample: str, bins, log_ratio, seg, copies, normals, fit, sex: str,
                lengths: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chroms = [c for c in CHROMOSOMES if not (c == "chrY" and sex == "XX")]
    offsets, pos = {}, 0
    for c in chroms:
        offsets[c] = pos
        pos += lengths[c]
    total = pos

    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    for k, c in enumerate(chroms):
        if k % 2 == 0:
            ax.axvspan(offsets[c], offsets[c] + lengths[c], color="#f2f2f2", lw=0, zorder=0)
    x_mid = np.array([offsets.get(c, np.nan) for c in bins.chrom]) + (bins.start + bins.end) / 2.0
    use = np.isfinite(log_ratio) & np.isin(bins.chrom, chroms)
    zero = diploid_level(fit)                           # two copies in the leukemic cells are drawn at 0
    top = _top(seg["mean"].to_numpy(dtype=float) - zero)    # bins beyond the range sit on its edge
    y = np.clip(log_ratio - zero, -top, top)
    ax.scatter(x_mid[use], y[use], s=1.2, c=POINTS, lw=0, zorder=2, rasterized=True)
    for row, cn, normal in zip(seg.itertuples(), copies, normals):
        if row.chrom not in offsets:
            continue
        colour = GAIN if cn > normal else LOSS if cn < normal else NEUTRAL
        x0, x1 = offsets[row.chrom] + row.start, offsets[row.chrom] + row.end
        ax.plot([x0, x1], [np.clip(row.mean - zero, -top, top)] * 2, color=colour, lw=1.6,
                solid_capstyle="butt", zorder=3)
    ax.axhline(0, color="#bdbdbd", lw=0.5, zorder=1)
    ax.set_xlim(0, total)
    ax.set_ylim(-top - 0.1, top + 0.1)
    ax.set_ylabel("log2 ratio (0 = two copies)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xticks([offsets[c] + lengths[c] / 2 for c in chroms])
    ax.set_xticklabels([chrom_label(c) for c in chroms], fontsize=6)
    ax.tick_params(axis="x", length=0)
    for k, (c, tick) in enumerate(zip(chroms, ax.xaxis.get_major_ticks())):
        if CHROMOSOMES.index(c) >= 12 and k % 2 == 1:      # the small chromosomes alternate between two rows
            tick.set_pad(9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if fit.purity is not None:
        right = ax.twinx()
        ticks = [(c, v) for c, v in copy_levels(fit, "chr1", sex, range(0, 9), zero) if -top - 0.1 <= v <= top + 0.1]
        right.set_ylim(ax.get_ylim())
        right.set_yticks([v for _, v in ticks])
        right.set_yticklabels([str(c) for c, _ in ticks], fontsize=6)
        right.set_ylabel("copies (autosomes)", fontsize=7)
        right.spines["top"].set_visible(False)
    ax.set_title(sample, fontsize=8, loc="left", pad=14)
    ax.text(0.0, 1.02, _subtitle(fit), transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom",
            color="#404040")
    fig.tight_layout()
    fig.savefig(str(path_stem) + ".svg")
    fig.savefig(str(path_stem) + ".png", dpi=200)
    plt.close(fig)


def plot_chromosomes(path_stem: Path, sample: str, bins, log_ratio, seg, copies, normals, fit, sex: str,
                     lengths: dict, bands: dict, arms: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    chroms = [c for c in CHROMOSOMES if not (c == "chrY" and sex == "XX")]
    ncol = 6
    nrow = -(-len(chroms) // ncol)
    zero = diploid_level(fit)                           # two copies in the leukemic cells are drawn at 0
    top = _top(seg["mean"].to_numpy(dtype=float) - zero)
    changed = {}
    by_chrom = {}
    for a in arms:
        by_chrom.setdefault(a["arm"][:-1], []).append(a)
    for label, both in by_chrom.items():
        if all(a["call"] != "normal" and a["share_of_arm"] >= 0.9 for a in both) and \
                len({a["copy_number"] for a in both}) == 1:
            changed[label] = [f"x{both[0]['copy_number']}"]           # the whole chromosome
            continue
        for a in both:
            if a["call"] != "normal":
                share = f" ({a['share_of_arm']:.0%})" if a["share_of_arm"] < 0.9 else ""
                changed.setdefault(label, []).append(f"{a['arm'][-1]} x{a['copy_number']}{share}")
    height = 1.6 * nrow + 0.55
    fig = plt.figure(figsize=(7.1, height))
    grid = GridSpec(nrow * 2, ncol, figure=fig, height_ratios=[7, 1] * nrow, hspace=0.35, wspace=0.18,
                    left=0.06, right=0.98, top=1 - 0.5 / height, bottom=0.03)
    for k, chrom in enumerate(chroms):
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(grid[2 * r, c])
        ideo = fig.add_subplot(grid[2 * r + 1, c], sharex=ax)
        on = (bins.chrom == chrom) & np.isfinite(log_ratio)
        mid = (bins.start[on] + bins.end[on]) / 2e6
        ax.scatter(mid, np.clip(log_ratio[on] - zero, -top, top), s=1.0, c=POINTS, lw=0, rasterized=True, zorder=2)
        for row, cn, normal in zip(seg.itertuples(), copies, normals):
            if row.chrom != chrom:
                continue
            colour = GAIN if cn > normal else LOSS if cn < normal else NEUTRAL
            ax.plot([row.start / 1e6, row.end / 1e6], [np.clip(row.mean - zero, -top, top)] * 2, color=colour,
                    lw=1.4, solid_capstyle="butt", zorder=3)
        ax.axhline(0, color="#bdbdbd", lw=0.5, zorder=1)
        for cn, v in copy_levels(fit, chrom, sex, (1, 2, 3, 4), zero):
            if abs(v) <= top:
                if abs(v) > 0.02:                       # the solid line at 0 already marks two copies
                    ax.axhline(v, color="#bdbdbd", lw=0.4, ls=":", zorder=1)
                ax.text(lengths[chrom] / 1e6, v, f" {cn}", fontsize=4.5, va="center", ha="left", color="#737373")
        cen = [b for b in bands[chrom] if b.stain == "acen"]
        if cen:
            ax.axvline((cen[0].start + cen[-1].end) / 2e6, color="#9e9e9e", lw=0.5, ls="--", zorder=1)
        ax.set_xlim(0, lengths[chrom] / 1e6)
        ax.set_ylim(-top - 0.1, top + 0.1)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.tick_params(axis="x", labelbottom=False)
        if c:
            ax.tick_params(axis="y", labelleft=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        label = chrom_label(chrom)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=6.5, ha="left", va="top", fontweight="bold")
        if label in changed:
            ax.text(0.98, 0.98, ", ".join(changed[label]), transform=ax.transAxes, fontsize=5, ha="right",
                    va="top", color="#b2182b" if any(a["arm"].rstrip("pq") == label and a["call"] == "gain"
                                                     for a in arms) else "#2166ac")
        for b in bands[chrom]:
            ideo.axvspan(b.start / 1e6, b.end / 1e6, color=STAIN.get(b.stain, "#ffffff"), lw=0)
        ideo.set_ylim(0, 1)
        ideo.set_yticks([])
        ideo.tick_params(axis="x", labelsize=4, length=1.5, pad=0.5)
        ideo.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        for side in ("top", "right", "left"):
            ideo.spines[side].set_visible(False)
    fig.text(0.06, 1 - 0.08 / height, sample, fontsize=8, ha="left", va="top")
    fig.text(0.06, 1 - 0.26 / height, _subtitle(fit) + " Log2 ratio with 0 at two copies; dotted lines mark the "
             "other copy numbers.", fontsize=6.5, ha="left", va="top", color="#404040")
    fig.savefig(str(path_stem) + ".svg")
    fig.savefig(str(path_stem) + ".png", dpi=200)
    plt.close(fig)

