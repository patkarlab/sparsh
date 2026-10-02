"""
Differentially methylated regions: the input for figures/dmr.R, and the maps and locus plots of its results.

The region analysis uses DMRcate (Peters and colleagues, Epigenetics and Chromatin 2015) on the statistics of
stats.py: each class against all other arrays, CpG by CpG. figures/dmr.R places the CpGs on the genome (hg19),
smooths their t statistics with a Gaussian kernel (lambda 1000 bases, C 2, the DMRcate defaults for arrays) and
joins neighbouring significant CpGs into regions. A CpG counts as individually significant when its false
discovery rate is below 0.05 and its mean beta differs by at least 0.1; the second condition matters with
thousands of arrays, where very small differences reach significance. Only CpGs on the autosomes are used.

The maps keep regions of at least three CpGs whose mean difference in beta is at least 0.2 (both can be changed),
and show, class by class, where along the genome the class is hyper- or hypomethylated against all other arrays.
The locus plots show, for the strongest regions of each class, the mean beta of every class at each CpG of the
region and 2 kb around it, with the class in its colour and the others in grey.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .style import Style

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
HG19_LENGTHS = {"chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276, "chr5": 180915260,
                "chr6": 171115067, "chr7": 159138663, "chr8": 146364022, "chr9": 141213431, "chr10": 135534747,
                "chr11": 135006516, "chr12": 133851895, "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
                "chr16": 90354753, "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
                "chr21": 48129895, "chr22": 51304566}


def file_stem(cls: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cls)


def write_input(statistics: Dict[str, pd.DataFrame], annotation: pd.DataFrame, classes: Sequence[str],
                style: Style, out_dir: Path) -> pd.DataFrame:
    """One table per class for dmr.R (autosomal CpGs with a statistic), and classes.tsv listing them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    placed = annotation["chrom"].isin(AUTOSOMES).to_numpy() & np.isfinite(annotation["pos"].to_numpy(dtype=float))
    rows = []
    for cls in classes:
        st = statistics[cls]
        keep = placed & np.isfinite(st["t"].to_numpy())
        table = pd.DataFrame({"probe": annotation.index[keep], "chrom": annotation["chrom"].to_numpy()[keep],
                              "pos": annotation["pos"].to_numpy()[keep].astype(np.int64),
                              "t": st["t"].to_numpy()[keep], "p": st["p"].to_numpy()[keep],
                              "fdr": st["fdr"].to_numpy()[keep], "delta": st["delta"].to_numpy()[keep]})
        name = f"{file_stem(cls)}.tsv.gz"
        table.to_csv(out_dir / name, sep="\t", index=False, float_format="%.6g")
        rows.append({"class": cls, "shown": style.name(cls), "lineage": style.lineage[cls],
                     "arrays": int(st["n_in"].max()), "file": name, "cpgs": int(keep.sum())})
    listing = pd.DataFrame(rows)
    listing.to_csv(out_dir / "classes.tsv", sep="\t", index=False)
    return listing


def read_regions(dmr_dir: Path, listing: pd.DataFrame, annotation: pd.DataFrame, min_cpgs: int = 3,
                 min_delta: float = 0.2) -> pd.DataFrame:
    """All regions written by dmr.R (<class>.dmrs.tsv), with direction and the genes of their CpGs."""
    frames = []
    for cls in listing["class"]:
        path = Path(dmr_dir) / f"{file_stem(cls)}.dmrs.tsv"
        if not path.exists():
            continue
        d = pd.read_csv(path, sep="\t")
        if d.empty:
            continue
        d.insert(0, "class", cls)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no <class>.dmrs.tsv files in {dmr_dir}: run figures/dmr.R first")
    regions = pd.concat(frames, ignore_index=True)
    regions["direction"] = np.where(regions["meandiff"] > 0, "hyper", "hypo")
    regions["kept"] = (regions["no.cpgs"] >= min_cpgs) & (regions["meandiff"].abs() >= min_delta)
    by_chrom = {c: g.sort_values("pos") for c, g in annotation[annotation["chrom"].isin(AUTOSOMES)]
                .assign(pos=lambda a: a["pos"].astype(np.int64)).groupby("chrom")}
    genes = []
    for _, r in regions.iterrows():
        g = by_chrom.get(r["chrom"])
        if g is None:
            genes.append("")
            continue
        inside = g[(g["pos"] >= r["start"]) & (g["pos"] <= r["end"])]["genes"]
        names = dict.fromkeys(x for v in inside if isinstance(v, str) for x in v.split(";") if x)
        genes.append(", ".join(names))
    regions["genes"] = genes
    return regions


def plot_counts(regions: pd.DataFrame, style: Style, order: Sequence[str], path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = regions[regions["kept"]]
    names = [c for c in order if c in set(regions["class"])]
    hyper = [int(((kept["class"] == c) & (kept["direction"] == "hyper")).sum()) for c in names]
    hypo = [int(((kept["class"] == c) & (kept["direction"] == "hypo")).sum()) for c in names]
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.33)
    x = np.arange(len(names))
    ax.bar(x, hyper, color="#b2182b", width=0.8, label="hypermethylated in the class")
    ax.bar(x, [-v for v in hypo], color="#2166ac", width=0.8, label="hypomethylated in the class")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(style.axis_labels(names), rotation=90, ha="center", va="top", fontsize=5, linespacing=1.1)
    ax.set_ylabel("Regions (hypomethylated below zero)", fontsize=6)
    ax.tick_params(axis="y", labelsize=5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.legend(fontsize=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)


def plot_genome_map(regions: pd.DataFrame, style: Style, order: Sequence[str], path_stem) -> None:
    """Every kept region as a tick at its place in the genome, one row per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept = regions[regions["kept"]]
    names = [c for c in order if c in set(regions["class"])]
    offsets, pos = {}, 0
    for c in AUTOSOMES:
        offsets[c] = pos
        pos += HG19_LENGTHS[c]
    height = 0.16 * len(names) + 0.7
    fig, ax = plt.subplots(figsize=(7.1, height))
    fig.subplots_adjust(left=0.22, right=0.99, top=1 - 0.1 / height, bottom=0.45 / height)
    for k, c in enumerate(AUTOSOMES):
        if k % 2 == 0:
            ax.axvspan(offsets[c], offsets[c] + HG19_LENGTHS[c], color="#f2f2f2", lw=0, zorder=0)
    for row, cls in enumerate(names):
        mine = kept[kept["class"] == cls]
        for direction, colour, shift in (("hyper", "#b2182b", 0.2), ("hypo", "#2166ac", -0.2)):
            part = mine[mine["direction"] == direction]
            x = part["chrom"].map(offsets).to_numpy(dtype=float) + (part["start"] + part["end"]).to_numpy() / 2.0
            y = np.full(len(x), -row + shift)
            ax.scatter(x, y, marker="|", s=6, c=colour, lw=0.4, alpha=0.7, rasterized=True, zorder=2)
    ax.set_yticks([-r for r in range(len(names))])
    ax.set_yticklabels([style.unique_name(c, names) for c in names], fontsize=5)
    ax.set_ylim(-len(names) + 0.4, 0.6)
    ax.set_xlim(0, pos)
    ax.set_xticks([offsets[c] + HG19_LENGTHS[c] / 2 for c in AUTOSOMES])
    ax.set_xticklabels([c[3:] for c in AUTOSOMES], fontsize=5)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Chromosome (red: hypermethylated in the class, blue: hypomethylated)", fontsize=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)


def top_regions(regions: pd.DataFrame, order: Sequence[str], per_class: int = 1) -> pd.DataFrame:
    """The strongest kept regions of each class: smallest smoothed false discovery rate, then largest difference."""
    kept = regions[regions["kept"]].copy()
    kept["abs_diff"] = kept["meandiff"].abs()
    picks = []
    for cls in order:
        mine = kept[kept["class"] == cls].sort_values(["min_smoothed_fdr", "abs_diff"], ascending=[True, False])
        picks.append(mine.head(per_class))
    return pd.concat(picks, ignore_index=True) if picks else kept.head(0)


def plot_loci(picks: pd.DataFrame, betas: np.ndarray, classes: np.ndarray, annotation: pd.DataFrame, style: Style,
              order: Sequence[str], path_stem, flank: int = 2000) -> None:
    """One panel per region: mean beta of every class at each CpG of the region and its flanks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [c for c in order if c in set(classes)]
    members = {c: np.flatnonzero(classes == c) for c in names}
    n = len(picks)
    ncol = 6
    nrow = max(1, -(-n // ncol))
    height = 1.3 * nrow + 0.4
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.1, height), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 0.2 / height, bottom=0.4 / height, hspace=0.85,
                        wspace=0.3)
    fig.supxlabel("Distance from the start of the region (kb, hg19 coordinates in the titles); the region is shaded",
                  fontsize=5.5)
    fig.supylabel("Mean beta of each class", fontsize=5.5)
    chrom = annotation["chrom"].to_numpy()
    posn = annotation["pos"].to_numpy(dtype=float)
    for k, (_, r) in enumerate(picks.iterrows()):
        ax = axes[k // ncol][k % ncol]
        cls = r["class"]
        idx = np.flatnonzero((chrom == r["chrom"]) & (posn >= r["start"] - flank) & (posn <= r["end"] + flank))
        idx = idx[np.argsort(posn[idx])]
        x = (posn[idx] - r["start"]) / 1e3                          # kb from the start of the region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)            # a class with no value at a CpG
            for c in names:
                if c != cls:
                    ax.plot(x, np.nanmean(betas[np.ix_(members[c], idx)], axis=0), color="#bdbdbd", lw=0.5,
                            zorder=1)
            ax.plot(x, np.nanmean(betas[np.ix_(members[cls], idx)], axis=0), color=style.colour[cls], lw=1.3,
                    marker="o", markersize=1.5, zorder=3)
        ax.axvspan(0, (r["end"] - r["start"]) / 1e3, color="#fff2cc", lw=0, zorder=0)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=4, length=1.5, pad=1)
        genes = r["genes"].split(", ")[0] if isinstance(r["genes"], str) and r["genes"] else "intergenic"
        ax.set_title(f"{style.unique_name(cls)}\n{genes}, {r['chrom']}:{int(r['start']):,}", fontsize=4.5, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)
