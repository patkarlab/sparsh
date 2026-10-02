"""
Tumour fraction, baseline ploidy and integer copy numbers from segment means.

After normalisation the median autosomal bin sits near a log ratio of 0. If a
fraction p of the cells are leukaemic, the median bin has m copies in those
cells (m is 2 for most leukaemias, 1 for a near-haploid clone, 3 or 4 for
near-triploid or near-tetraploid ones) and the normal cells carry their
constitutional copies n (2 for autosomes; for the sex chromosomes it depends
on sex), a segment with c copies in every leukaemic cell is expected at

    log2( (p*c + (1 - p)*n) / (p*m + 2*(1 - p)) ) + d

where d is a small offset of the whole profile: when a large part of the
genome is gained, the median bin is no longer exactly at the level of m copies.

Every combination of p (0.10 to 1.00) and m (1 to 4) is tried, with d fitted
for each. Each segment is explained either by an integer copy number in all
leukaemic cells, or, at a fixed cost, by a one-copy change in part of the
cells (a subclone, or an artefact of a short segment). The cost of a
combination is the sum over segments of the squared error in standard errors,
plus a fixed cost for every segment two or more copies away from the baseline
of its chromosome, the constitutional number scaled to the ploidy of the clone
(the same for a high-level amplification as for two extra copies), plus the
cost of the subclonal explanations, plus a preference for a near-diploid
baseline. Segments of fewer than five bins do not take part in
the fit; they are assigned afterwards. Only segments of five bins or more can
be explained as a change in part of the cells, and a segment whose mean lies
within four standard errors of its normal level is always called normal.

Each segment takes the integer level nearest its mean; the cost for extra
copies then makes the fit prefer, among fractions that explain the profile
equally well, the one that needs the smallest changes, and among solutions
whose costs differ by less than 2 the one whose segments lie closest to the
baseline is taken. No single segment adds more than 25 to the cost (five
standard errors), so one segment that fits no solution, often an artefact,
cannot decide the fit on its own. The autosomes decide the fit whenever they carry a change;
the sex chromosomes, more prone to artefacts, are then assigned with it. Read depth alone cannot
always separate solutions: a near-haploid clone and the same clone doubled
give the same profile, and a small leukaemic fraction with large changes can
mimic a large one with small changes. The best cost for every baseline is
therefore kept and reported, and a blast percentage from the sample sheet
limits the fraction when it is known.

A baseline other than two copies is taken only when at least one autosome, over 90% of its fitted length, lies
at another copy number in that reading. Read depth places a genome on another baseline only through whole
chromosomes that differ from it: a genome multiplied as a whole looks like a diploid one, and a triploid
baseline with a fraction p puts every level where a diploid one with the fraction 2p / (p + 2) does. Without
the rule, a profile with shorter changes only, present in fewer cells than the blast percentage allows, would
be moved to a triploid or tetraploid baseline merely to reach an allowed fraction, although such changes are
often present in part of the leukaemic cells. The best reading turned down this way is kept when it costs less
than the diploid one taken, so that it can be named.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .normalise import normal_copies

PENALTY = {1: 12.0, 2: 0.0, 3: 12.0, 4: 24.0}   # preference for a near-diploid baseline
EXTRA_COPY = 10.0                                # per segment two or more copies away from the baseline
SUBCLONAL = 6.0                                  # per segment explained as a change in part of the cells
MODEL_ERROR = 0.05                               # log2 allowance per segment beyond counting noise
MAX_OFFSET = 0.25
MIN_FIT_BINS = 5
MAX_SEGMENT_COST = 25.0                          # a single segment adds at most this much (five standard errors)
TIE = 2.0                                        # costs within this of the best count as equally good
MIN_PARTIAL_BINS = 5                             # a change in part of the cells needs a longer segment
MIN_CHANGE_Z = 4.0                               # standard errors from the normal level to count as a change
ALTERNATIVE = 40.0                               # other baselines within this of the best cost are considered:
CLOSE = 6.0                                      # noted when within this, or when hypo- against hyperdiploid
WHOLE_SHARE = 0.9                                # share of an autosome's fitted bins that makes it whole


def expected_log2(c, n, p, m):
    num = p * np.asarray(c, dtype=float) + (1.0 - p) * n
    den = p * m + 2.0 * (1.0 - p)
    with np.errstate(divide="ignore"):
        return np.log2(np.maximum(num, 1e-6) / den)


@dataclass
class Fit:
    purity: Optional[float]
    baseline: int
    score: float
    offset: float
    copies: np.ndarray                 # integer copy number per segment
    cell_fraction: np.ndarray          # fraction of all cells carrying the change (NaN when none)
    subclonal: np.ndarray              # explained as a change in part of the cells
    alternatives: List[Dict] = field(default_factory=list)
    estimable: bool = True
    unsupported: List[Dict] = field(default_factory=list)   # better readings on a baseline no whole autosome leaves


def diploid_fraction(p: float, m: int) -> float:
    """The leukaemic fraction at which a diploid baseline puts every level where baseline m with fraction p does."""
    return 2.0 * p / (p * m + 2.0 * (1.0 - p))


def _whole_autosome_away(chroms: np.ndarray, copies: np.ndarray, nbins: np.ndarray, m: int) -> bool:
    """Whether any autosome lies, over WHOLE_SHARE of its fitted bins, at a copy number other than m."""
    for c in np.unique(chroms):
        on = chroms == c
        total = nbins[on].sum()
        for v in np.unique(copies[on]):
            if v != m and nbins[on & (copies == v)].sum() >= WHOLE_SHARE * total:
                return True
    return False


def _explain(means, se2, normals, nbins, p, m, d, max_cn):
    """Best explanation of every segment at (p, m, d): (cost, copies, subclonal, fitted log2)."""
    cands = np.arange(0, max_cn + 1)
    grid = np.stack([expected_log2(c, normals, p, m) for c in cands], axis=1) + d
    z2 = (grid - means[:, None]) ** 2 / se2[:, None]
    best = np.argmin(z2, axis=1)                    # the level nearest the segment mean
    rows = np.arange(len(means))
    # a segment not clearly away from its normal level stays normal
    normal_level = expected_log2(normals, normals, p, m) + d
    z_normal = np.abs(means - normal_level) / np.sqrt(se2)
    quiet = z_normal < MIN_CHANGE_Z
    best = np.where(quiet, np.searchsorted(cands, normals.astype(int)), best)
    copies = cands[best]
    # a segment two or more copies away from the baseline costs EXTRA_COPY once, however high the level (so an
    # amplification does not push the fit towards a halved baseline). The baseline of a chromosome is its
    # constitutional number scaled to the ploidy of the clone, so that in a near-triploid or high hyperdiploid
    # reading a tetrasomy is one step from the baseline, as a trisomy is in a diploid one.
    extra = (np.abs(copies - normals * m / 2.0) >= 2) * EXTRA_COPY
    clonal_cost = z2[rows, best] + extra
    # a one-copy change in part of the cells: per-cell change strictly between 0 and one full clonal copy
    per_cell = 2.0 ** (means - d) * (p * m + 2.0 * (1.0 - p))
    change = per_cell - normals
    partial = (np.abs(change) > 0.08) & (np.abs(change) < p - 0.03) & (nbins >= MIN_PARTIAL_BINS) & ~quiet
    use_partial = partial & (SUBCLONAL < clonal_cost)
    total = np.where(use_partial, SUBCLONAL, clonal_cost)
    fitted = grid[rows, best]
    return total, copies, use_partial, fitted, change


def fit_segments(seg: pd.DataFrame, sex: str, sigma: float, max_cn: int = 8,
                 purity_range: Optional[tuple] = None) -> Fit:
    """
    seg: segments with chrom, n_bins, mean and (optionally) sigma, the per-bin noise of their chromosome;
    sigma: per-bin noise used where the segments do not carry their own.
    purity_range: (low, high) limits on the leukaemic fraction, for example from the blast percentage.
    """
    means = seg["mean"].to_numpy(dtype=float)
    nbins = seg["n_bins"].to_numpy(dtype=float)
    chroms = seg["chrom"].to_numpy()
    normals = np.array([normal_copies(c, sex) for c in seg["chrom"]], dtype=float)
    auto = ~seg["chrom"].isin(["chrX", "chrY"]).to_numpy()
    seg_sigma = seg["sigma"].to_numpy(dtype=float) if "sigma" in seg else np.full(len(seg), sigma)
    se2 = seg_sigma ** 2 / nbins + MODEL_ERROR ** 2
    use = nbins >= MIN_FIT_BINS
    lo, hi = purity_range or (0.10, 1.00)
    purities = np.round(np.arange(max(0.05, lo), min(1.0, hi) + 1e-9, 0.01), 2)

    base = expected_log2(normals, normals, 1.0, 2)
    z = np.abs(means - base) / np.sqrt(se2)
    changed = use & (z > 4.0)
    # the autosomes decide the clone; the sex chromosomes, more prone to artefacts, only when they alone change
    if (changed & auto).any():
        use = use & auto
    if not changed.any():
        return Fit(None, 2, 0.0, 0.0, normals.astype(int), np.full(len(seg), np.nan), np.zeros(len(seg), bool),
                   [], estimable=False)

    results, unsupported = [], []
    for m in (1, 2, 3, 4):
        for p in purities:
            d = 0.0
            for _ in range(3):                 # alternate: explain segments, then refit the offset
                cost, copies, partial, fitted, _ = _explain(means[use], se2[use], normals[use], nbins[use], p, m, d,
                                                            max_cn)
                clonal = ~partial
                if not clonal.any():
                    break
                w = 1.0 / se2[use][clonal]
                resid = means[use][clonal] - (fitted[clonal] - d)
                d = float(np.clip(np.sum(w * resid) / np.sum(w), -MAX_OFFSET, MAX_OFFSET))
            cost, copies, partial, fitted, _ = _explain(means[use], se2[use], normals[use], nbins[use], p, m, d,
                                                        max_cn)
            au = auto[use]
            if au.any():
                cn_for_median = np.where(partial, normals[use], copies)[au]
                order = np.argsort(cn_for_median, kind="stable")
                cum = np.cumsum(nbins[use][au][order])
                median_cn = cn_for_median[order][np.searchsorted(cum, cum[-1] / 2.0)]
                if median_cn != m:
                    continue
            score = float(np.minimum(cost, MAX_SEGMENT_COST).sum()) + PENALTY[m]
            clonal_cn = np.where(partial, m, copies)[au] if au.any() else np.array([m])
            # another baseline needs a whole autosome away from it (see the notes at the top)
            if m != 2 and not (au.any() and _whole_autosome_away(chroms[use][au], clonal_cn, nbins[use][au], m)):
                unsupported.append((score, m, float(p), d))
                continue
            distance = float(np.sum(np.abs(clonal_cn - m) * nbins[use][au]) / max(1.0, nbins[use][au].sum())) \
                if au.any() else 0.0
            results.append((score, m, float(p), d, distance))
    if not results:
        return Fit(None, 2, np.inf, 0.0, normals.astype(int), np.full(len(seg), np.nan), np.zeros(len(seg), bool),
                   [], estimable=False)

    best = min(r[0] for r in results)
    near = [r for r in results if r[0] <= best + TIE]
    score, m, p, d, _ = min(near, key=lambda r: (round(r[4], 3), r[0]))
    alternatives = []
    for mm in (1, 2, 3, 4):
        rs = [r for r in results if r[1] == mm]
        if rs:
            r = min(rs, key=lambda r: r[0])
            alternatives.append({"baseline": mm, "purity": r[2], "offset": round(r[3], 4), "score": round(r[0], 2),
                                 "delta": round(r[0] - score, 2)})
    fit = explain_all(seg, sex, sigma, p, int(m), d, max_cn, score, alternatives)
    better = [r for r in unsupported if r[0] < score]
    if better and m == 2:                        # the best reading turned down, when it costs less than the diploid one
        r = min(better, key=lambda r: r[0])
        fit.unsupported = [{"baseline": r[1], "purity": r[2], "offset": round(r[3], 4), "score": round(r[0], 2),
                            "delta": round(r[0] - score, 2),
                            "diploid_fraction": round(diploid_fraction(r[2], r[1]), 2)}]
    return fit


def explain_all(seg: pd.DataFrame, sex: str, sigma: float, p: float, m: int, d: float, max_cn: int = 8,
                score: float = float("nan"), alternatives: Optional[List[Dict]] = None) -> Fit:
    """Copy number of every segment, short ones included, at the leukaemic fraction p, baseline m and offset d."""
    means = seg["mean"].to_numpy(dtype=float)
    nbins = seg["n_bins"].to_numpy(dtype=float)
    normals = np.array([normal_copies(c, sex) for c in seg["chrom"]], dtype=float)
    seg_sigma = seg["sigma"].to_numpy(dtype=float) if "sigma" in seg else np.full(len(seg), sigma)
    se2 = seg_sigma ** 2 / nbins + MODEL_ERROR ** 2
    _, copies, partial, _, change = _explain(means, se2, normals, nbins, p, m, d, max_cn)
    copies = copies.astype(int)
    fraction = np.full(len(seg), np.nan)
    for s in range(len(seg)):
        if partial[s]:
            copies[s] = int(normals[s] + np.sign(change[s]))
            fraction[s] = float(np.clip(abs(change[s]), 0.0, 1.0))
        elif copies[s] != normals[s]:
            fraction[s] = float(np.clip(change[s] / (copies[s] - normals[s]), 0.0, 1.0))
    return Fit(p, int(m), score, d, copies, fraction, partial.astype(bool), list(alternatives or []))
