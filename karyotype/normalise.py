"""
From read counts to log ratios: bin aggregation, GC correction, the cohort reference, and sex.

Each sample's counts at 100 kb are added up into larger analysis bins. The bin
size is chosen per sample from the depth: the smallest of 0.5, 1, 2 and 5 Mb
at which the median autosomal bin holds at least `min_reads` reads, because
the counting noise of a bin falls with the square root of its read count.
Excluded 100 kb bins (see reference.py) do not contribute, and each analysis
bin is scaled by the number of 100 kb bins it actually holds.

GC correction is done per sample: the median read density of autosomal bins
is taken within narrow classes of GC content, a cubic curve is laid through
those medians, and every bin is divided by the value of the curve at its GC. Medians keep the correction robust to real gains and losses.
The one per cent of bins with the most extreme GC content (for example the
GC-rich ends of chromosomes 16, 17 and 19) are left out, because there the
correction rests on too few bins.

What GC does not explain (mappability, repeats, the preferences of the
library chemistry, the aligner) is the same in every sample sequenced the
same way, so it is removed by dividing by a reference built from the cohort:
the median, bin by bin, of the GC-corrected profiles of the other samples.
The sample being called is always left out of its own reference. The median
is robust to the cohort's aberrations as long as no single bin is changed in
most samples, which holds for a mixed leukaemia cohort. Bins whose reference
is very low or whose spread across the cohort is large are not used.

The sex chromosomes need the constitutional sex of each sample: two X
chromosomes in women, one X and one Y in men. It is taken from a sample sheet
when one is given. Otherwise it is inferred from the read density on the
usable part of chromosome Y (outside the pseudoautosomal and X-transposed
regions) relative to the autosomes: near half in men, near zero in women.
Where that density lies depends on the aligner and the library, so the
threshold is taken from the cohort itself when its samples fall into two
clearly separated groups, midway between them on a log scale, and otherwise
from fixed values. A sample close to the threshold is decided by the X
chromosome and gets a warning, since loss of the Y chromosome in the
leukaemic cells, or loss of an X, makes a sample harder to read.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .counts import count_paths, read_counts, safe_name
from .genome import AUTOSOMES, CHROMOSOMES
from .reference import FINE_BIN, load_bin_table

BIN_SIZES = (500_000, 1_000_000, 2_000_000, 5_000_000)
# Read density on chromosome Y relative to the autosomes (both per usable 100 kb bin). A man's single Y gives
# about 0.5; a woman's reads there are misplaced ones.
Y_MALE = 0.10            # fixed thresholds, used when the cohort does not show two clear groups
Y_FEMALE = 0.02
Y_MARGIN = 2.0           # with a cohort threshold t: male above t * margin, female below t / margin
X_MALE = 0.72            # X density relative to the autosomes below which an ambiguous sample is taken as male


@dataclass
class Sample:
    name: str
    meta: dict
    counts: np.ndarray                  # reads per 100 kb bin, in the order of the bin table
    sex: Optional[str] = None           # 'XX' or 'XY'
    sex_source: str = ""
    warnings: List[str] = field(default_factory=list)


def load_sample(counts_dir: Path, sample: str, table: pd.DataFrame) -> Sample:
    tsv, js = count_paths(Path(counts_dir), sample)
    if not js.exists():
        raise FileNotFoundError(f"{js} not found: {sample} has not been counted")
    meta = json.loads(js.read_text())
    header, df = read_counts(tsv)
    if int(header.get("bin_size", FINE_BIN)) != FINE_BIN:
        raise ValueError(f"{tsv}: counts must be at {FINE_BIN} bases per bin")
    if len(df) != len(table) or not (df["chrom"].to_numpy() == table["chrom"].to_numpy()).all() \
            or not (df["start"].to_numpy() == table["start"].to_numpy()).all():
        raise ValueError(f"{tsv}: bins do not match the {meta.get('build')} bin table")
    return Sample(name=meta.get("sample", sample), meta=meta, counts=df["reads"].to_numpy(dtype=np.float64))


def list_counted(counts_dir: Path) -> List[str]:
    out = []
    for js in sorted(Path(counts_dir).glob("*.counts.json")):
        try:
            out.append(json.loads(js.read_text())["sample"])
        except (ValueError, KeyError):
            continue
    return out


# ------------------------------------------------------------------ bins
@dataclass
class Bins:
    """Analysis bins of one size: coordinates, GC, and which fine bins they hold."""
    size: int
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    gc: np.ndarray
    n_fine: np.ndarray                  # usable 100 kb bins inside
    usable: np.ndarray                  # bins that can be used at all (enough usable fine bins)
    groups: np.ndarray                  # index of the analysis bin for every fine bin (-1 when excluded)


def make_bins(table: pd.DataFrame, size: int) -> Bins:
    if size % FINE_BIN:
        raise ValueError(f"bin size must be a multiple of {FINE_BIN}")
    k = size // FINE_BIN
    chrom_col = table["chrom"].to_numpy()
    groups = np.full(len(table), -1, dtype=np.int64)
    chroms, starts, ends, gcs, n_fine, usable = [], [], [], [], [], []
    idx = 0
    for chrom in CHROMOSOMES:
        rows = np.flatnonzero(chrom_col == chrom)
        for first in range(0, len(rows), k):
            block = rows[first:first + k]
            ok = block[~table["excluded"].to_numpy()[block]]
            groups[ok] = idx
            chroms.append(chrom)
            starts.append(int(table["start"].iat[block[0]]))
            ends.append(int(table["end"].iat[block[-1]]))
            gcs.append(float(table["gc"].to_numpy()[ok].mean()) if len(ok) else np.nan)
            n_fine.append(len(ok))
            usable.append(len(ok) >= max(1, k // 2))
            idx += 1
    return Bins(size, np.array(chroms), np.array(starts), np.array(ends), np.array(gcs), np.array(n_fine),
                np.array(usable), groups)


def aggregate(bins: Bins, fine_counts: np.ndarray) -> np.ndarray:
    """Reads per usable 100 kb bin within each analysis bin (NaN where the analysis bin is not usable)."""
    ok = bins.groups >= 0
    total = np.bincount(bins.groups[ok], weights=fine_counts[ok], minlength=len(bins.chrom))
    with np.errstate(invalid="ignore", divide="ignore"):
        density = total / bins.n_fine
    density[~bins.usable] = np.nan
    return density


def choose_bin_size(bins_by_size: Dict[int, Bins], fine_counts: np.ndarray, min_reads: float) -> int:
    """Smallest analysis bin whose median autosomal bin has at least min_reads reads."""
    for size in sorted(bins_by_size):
        b = bins_by_size[size]
        density = aggregate(b, fine_counts)
        auto = np.isin(b.chrom, AUTOSOMES) & b.usable
        reads = density[auto] * b.n_fine[auto]
        if np.nanmedian(reads) >= min_reads:
            return size
    return max(bins_by_size)


# ------------------------------------------------------------------ GC correction
def gc_correct(bins: Bins, density: np.ndarray, n_classes: int = 30) -> Tuple[np.ndarray, dict]:
    """
    Read density divided by the median density at the bin's GC content, scaled so that the median autosomal
    bin is 1. Returns (ratio, information about the fit).
    """
    auto = np.isin(bins.chrom, AUTOSOMES) & bins.usable & np.isfinite(density) & (density > 0) \
        & np.isfinite(bins.gc)
    if auto.sum() < 50:
        raise ValueError("fewer than 50 usable autosomal bins with reads: the sample is too shallow")
    gc, y = bins.gc[auto], np.log2(density[auto])
    edges = np.unique(np.quantile(gc, np.linspace(0, 1, n_classes + 1)))
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (gc >= lo) & (gc <= hi)
        if inside.sum() >= 5:
            centres.append(float(np.median(gc[inside])))
            medians.append(float(np.median(y[inside])))
    centres, medians = np.array(centres), np.array(medians)
    # a cubic through the class medians: smooth, and flexible enough for the curved GC response of
    # nanopore libraries; with few classes, straight lines between them
    if len(centres) >= 8:
        coef = np.polyfit(centres, medians, 3)
        curve = lambda g: np.polyval(coef, g)                      # noqa: E731
        smooth = curve(centres)
    else:
        curve = lambda g: np.interp(g, centres, medians)           # noqa: E731
        smooth = medians
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.where(np.isfinite(bins.gc), 2.0 ** curve(np.nan_to_num(bins.gc, nan=0.4)), np.nan)
        ratio = density / expected
    ratio[~bins.usable] = np.nan
    # at the extremes of GC the correction rests on few bins or would be extrapolated: leave those bins out
    lo_gc, hi_gc = np.quantile(gc, [0.005, 0.995])
    ratio[(bins.gc < lo_gc) | (bins.gc > hi_gc)] = np.nan
    scale = np.nanmedian(ratio[np.isin(bins.chrom, AUTOSOMES)])
    ratio = ratio / scale
    info = {"gc_classes": len(centres), "gc_effect_log2": float(smooth.max() - smooth.min()) if len(smooth) else None}
    return ratio, info


# ------------------------------------------------------------------ sex
def sex_measures(sample: Sample, table: pd.DataFrame) -> Tuple[float, float]:
    """Read density on usable bins of Y and of X, each relative to the autosomes (raw counts, 100 kb bins)."""
    usable = ~table["excluded"].to_numpy()
    chrom = table["chrom"].to_numpy()
    auto = usable & np.isin(chrom, AUTOSOMES)
    per_bin = sample.counts[auto].sum() / max(1, auto.sum())
    if per_bin <= 0:
        return float("nan"), float("nan")
    x, y = usable & (chrom == "chrX"), usable & (chrom == "chrY")
    return (float(sample.counts[y].sum() / max(1, y.sum()) / per_bin),
            float(sample.counts[x].sum() / max(1, x.sum()) / per_bin))


def cohort_y_threshold(values: List[float], min_samples: int = 10, min_separation: float = 0.7) -> Optional[float]:
    """
    Threshold on the Y density from the cohort: the midpoint, on a log scale, between the centres of two groups
    found by two-means clustering, provided each group has at least two samples and the centres are at least
    `min_separation` apart in log10 (five-fold). None when the cohort does not show two such groups.
    """
    v = np.log10(np.asarray([x for x in values if np.isfinite(x)]) + 1e-4)
    if len(v) < min_samples:
        return None
    v = np.sort(v)
    best, split = np.inf, None
    for k in range(2, len(v) - 1):                    # every split into a low and a high group of two or more
        lo, hi = v[:k], v[k:]
        cost = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        if cost < best:
            best, split = cost, k
    if split is None:
        return None
    lo, hi = v[:split].mean(), v[split:].mean()
    if hi - lo < min_separation:
        return None
    return float(10 ** ((lo + hi) / 2.0))


def y_thresholds(values: List[float]) -> dict:
    """Thresholds on the Y density: from the samples' own two groups when they show them, else the fixed ones."""
    threshold = cohort_y_threshold(values)
    male, female = (threshold * Y_MARGIN, threshold / Y_MARGIN) if threshold else (Y_MALE, Y_FEMALE)
    return {"female_below": round(female, 4), "male_above": round(male, 4),
            "from": "cohort" if threshold else "fixed values"}


def assign_sex(samples: List[Sample], table: pd.DataFrame, thresholds: Optional[dict] = None) -> dict:
    """
    Set the sex of every sample the sample sheet has not given, from its Y and X densities. The thresholds on the Y
    density come from the samples whose sex is not given, unless `thresholds` (female_below, male_above, from)
    gives them, as a stored reference panel does. Returns the thresholds used.
    """
    for s in samples:
        s.meta["y_density"], s.meta["x_density"] = sex_measures(s, table)
    if thresholds is None:
        thresholds = y_thresholds([s.meta["y_density"] for s in samples if not s.sex])
    male, female = thresholds["male_above"], thresholds["female_below"]
    for s in samples:
        s.meta["y_density_thresholds"] = dict(thresholds)
        if s.sex:
            continue
        y, x = s.meta["y_density"], s.meta["x_density"]
        if not np.isfinite(y):
            s.sex, s.sex_source = "XX", "not inferable (no reads); taken as female"
            s.warnings.append("The sex could not be inferred; give it in the sample sheet.")
        elif y >= male:
            s.sex, s.sex_source = "XY", "inferred from reads on chromosome Y"
        elif y <= female:
            s.sex, s.sex_source = "XX", "inferred from the absence of reads on chromosome Y"
            if np.isfinite(x) and x < X_MALE:
                s.warnings.append(
                    "Few reads on chromosome Y and a single-copy level of chromosome X: either a woman whose "
                    "leukaemic cells have lost an X, or a man whose leukaemic cells have lost the Y. Taken as "
                    "female; give the sex in the sample sheet to settle it.")
        elif np.isfinite(x) and x < X_MALE:
            s.sex, s.sex_source = "XY", "inferred from chromosome X (few reads on Y)"
            s.warnings.append("Few reads on chromosome Y for a male sample: the Y chromosome may be lost in most "
                              "cells. The sex was taken from chromosome X; give it in the sample sheet to settle it.")
        else:
            s.sex, s.sex_source = "XX", "inferred from chromosome X (some reads on Y)"
            s.warnings.append("Some reads on chromosome Y but two X chromosomes: the sex was read as female; give "
                              "it in the sample sheet if this is a man with an extra X.")
    return thresholds


def normal_copies(chrom: str, sex: str) -> int:
    """Constitutional copies of a chromosome in normal cells."""
    if chrom == "chrX":
        return 2 if sex == "XX" else 1
    if chrom == "chrY":
        return 0 if sex == "XX" else 1
    return 2


def per_two_copies(bins: Bins, ratio: np.ndarray, sex: str) -> np.ndarray:
    """Ratio rescaled so that a normal chromosome of either sex reads 1 (for building the reference)."""
    out = ratio.copy()
    for chrom in ("chrX", "chrY"):
        on = bins.chrom == chrom
        n = normal_copies(chrom, sex)
        out[on] = ratio[on] * 2.0 / n if n else np.nan
    return out


# ------------------------------------------------------------------ cohort reference
@dataclass
class Reference:
    size: int
    samples: List[str]
    matrix: np.ndarray                  # samples x bins, per-two-copies ratios (NaN where not usable)

    def for_sample(self, name: str, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, int]:
        """Median and spread of the reference bin by bin, leaving the sample itself out."""
        keep = np.array([s != name for s in self.samples])
        m = self.matrix[keep]
        n = np.sum(np.isfinite(m), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
            median = np.nanmedian(m, axis=0)
            spread = np.nanmedian(np.abs(np.log2(m / median)), axis=0) * 1.4826
        median[n < min_samples] = np.nan
        return median, spread, int(keep.sum())


class PanelReference:
    """
    A stored reference (karyotype/panel.py): the median and spread of a cohort's reference profiles bin by bin,
    made once and used for samples outside that cohort, so that a single new sample can be called. The sample
    being called is not left out of it, so it should not be one of the samples the panel was made from.
    """

    def __init__(self, size: int, median: np.ndarray, spread: np.ndarray, n_samples: int, source: str = ""):
        self.size, self.median, self.spread, self.n_samples, self.source = size, median, spread, n_samples, source
        self.samples: List[str] = []

    def for_sample(self, name: str, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, int]:
        return self.median.copy(), self.spread.copy(), self.n_samples


class FlatReference:
    """
    No reference of other samples: every bin is expected at the sample's own GC-corrected median. For a first look
    at one sample before a cohort is counted; mappability and library artefacts are not removed.
    """

    def __init__(self, size: int, bins: Bins):
        self.size, self.samples, self.n_bins = size, [], len(bins.chrom)

    def for_sample(self, name: str, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray, int]:
        return np.ones(self.n_bins), np.zeros(self.n_bins), 0


def build_reference(size: int, samples: List[Sample], bins: Bins) -> Reference:
    rows, names = [], []
    for s in samples:
        density = aggregate(bins, s.counts)
        ratio, _ = gc_correct(bins, density)
        rows.append(per_two_copies(bins, ratio, s.sex))
        names.append(s.name)
    return Reference(size, names, np.vstack(rows))


def usable_against_reference(bins: Bins, median: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """Bins with a sound reference: not too low, and not much more variable across the cohort than usual."""
    ok = bins.usable & np.isfinite(median) & (median > 0.3) & np.isfinite(spread)
    auto = ok & np.isin(bins.chrom, AUTOSOMES)
    if auto.sum():
        typical = np.nanmedian(spread[auto])
        ok &= spread <= max(3.0 * typical, typical + 0.15)
    return ok


def sample_log_ratio(sample: Sample, bins: Bins, reference: Reference) -> Tuple[np.ndarray, dict]:
    """
    log2 of the sample's GC-corrected ratio over its leave-one-out reference, per analysis bin (NaN where not
    usable). On chromosomes X and Y of a man the values sit near -1 for a normal single copy.
    """
    density = aggregate(bins, sample.counts)
    ratio, gc_info = gc_correct(bins, density)
    median, spread, n_ref = reference.for_sample(sample.name)
    ok = usable_against_reference(bins, median, spread)
    if sample.sex == "XX" or isinstance(reference, FlatReference):
        ok &= bins.chrom != "chrY"                      # without a reference, Y's low mappability looks like a loss
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ratio = np.log2(ratio / median)
    log_ratio[~ok | ~np.isfinite(log_ratio)] = np.nan
    auto = np.isin(bins.chrom, AUTOSOMES) & np.isfinite(log_ratio)
    log_ratio -= np.nanmedian(log_ratio[auto])          # the median autosomal bin is the baseline
    reads = aggregate(bins, sample.counts) * bins.n_fine
    info = {**gc_info, "reference_samples": n_ref, "bins_used": int(np.isfinite(log_ratio).sum()),
            "median_reads_per_bin": float(np.nanmedian(reads[auto])) if auto.any() else None}
    return log_ratio, info


def robust_noise(values: np.ndarray) -> float:
    """Standard deviation of bin-to-bin noise from first differences (insensitive to segment changes)."""
    v = values[np.isfinite(values)]
    if len(v) < 10:
        return float("nan")
    d = np.diff(v)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2.0))


def load_cohort(counts_dir: Path, names: Optional[List[str]], build: str, sheet: Optional[pd.DataFrame] = None,
                sex_thresholds: Optional[dict] = None) -> Tuple[pd.DataFrame, List[Sample]]:
    table = load_bin_table(build)
    names = names or list_counted(counts_dir)
    samples = []
    for name in names:
        s = load_sample(counts_dir, name, table)
        if s.meta.get("build") and s.meta["build"] != build:
            raise ValueError(f"{name} was counted on {s.meta['build']}, not {build}")
        if sheet is not None:
            row = sheet.loc[sheet["key"] == safe_name(s.name).upper()]
            if len(row) and isinstance(row.iloc[0].get("sex"), str) and row.iloc[0]["sex"].strip():
                sex = row.iloc[0]["sex"].strip().upper()
                s.sex = {"F": "XX", "FEMALE": "XX", "XX": "XX", "M": "XY", "MALE": "XY", "XY": "XY"}.get(sex)
                if s.sex is None:
                    raise ValueError(f"sample sheet: sex {row.iloc[0]['sex']!r} for {s.name} is not F or M")
                s.sex_source = "sample sheet"
        samples.append(s)
    assign_sex(samples, table, sex_thresholds)
    return table, samples
