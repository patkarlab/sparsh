"""
Segmentation of log ratios along each chromosome by circular binary segmentation.

This follows the method of Olshen and colleagues (Biostatistics 2004), as in
the DNAcopy package: within a stretch of bins, find the arc (a run of bins,
possibly touching either end) whose mean differs most from the rest, in units
of its standard error; if that difference is significant, split there and
repeat on each part. Significance is judged against the largest of all the
arc statistics that pure noise would give, by a Bonferroni bound over the
number of arcs at a level of 0.001, so long chromosomes are not split more
easily than short ones and a genome of 24 chromosomes rarely gains a false
breakpoint. Single outlying bins are first pulled toward their neighbors, and
neighboring segments that turn out not to differ are merged again.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .genome import CHROMOSOMES


def smooth_outliers(x: np.ndarray, sigma: float, width: int = 2, limit: float = 4.0) -> np.ndarray:
    """Pull bins lying more than `limit` noise units from the median of their neighbors back to that limit."""
    if len(x) < 2 * width + 1 or not np.isfinite(sigma) or sigma <= 0:
        return x.copy()
    out = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - width), min(len(x), i + width + 1)
        neighbours = np.concatenate([x[lo:i], x[i + 1:hi]])
        med = np.median(neighbours)
        if abs(x[i] - med) > limit * sigma:
            out[i] = med + np.sign(x[i] - med) * limit * sigma
    return out


def critical_z(n: int, alpha: float = 0.001) -> float:
    arcs = max(1, n * (n - 1) // 2)
    return float(np.clip(norm.isf(alpha / arcs), 3.0, 7.0))


def best_arc(x: np.ndarray, sigma: float, min_size: int) -> Tuple[int, int, float]:
    """(i, j, z) of the arc x[i:j] whose mean differs most from the rest of x."""
    n = len(x)
    s = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    k = (j - i).astype(float)
    valid = (k >= min_size) & ((n - k) >= min_size)
    inside = s[None, :] - s[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = inside / k - (s[-1] - inside) / (n - k)
        z = diff / (sigma * np.sqrt(1.0 / k + 1.0 / (n - k)))
    z = np.where(valid, z, 0.0)
    flat = int(np.argmax(np.abs(z)))
    a, b = divmod(flat, n + 1)
    return a, b, float(z[a, b])


def cbs(x: np.ndarray, sigma: float, alpha: float = 0.001, min_size: int = 2) -> List[Tuple[int, int]]:
    """Segment boundaries [(start, end), ...] over the index range of x."""
    n = len(x)
    if n == 0:
        return []
    done, todo = [], [(0, n)]
    while todo:
        a, b = todo.pop()
        if b - a < 2 * min_size:
            done.append((a, b))
            continue
        i, j, z = best_arc(x[a:b], sigma, min_size)
        if abs(z) < critical_z(b - a, alpha):
            done.append((a, b))
            continue
        cuts = sorted({a, a + i, a + j, b})
        todo.extend((u, v) for u, v in zip(cuts[:-1], cuts[1:]) if v > u)
    return sorted(done)


def merge_similar(x: np.ndarray, bounds: List[Tuple[int, int]], sigma: float,
                  z_merge: float = 3.0, min_shift: float = 0.08) -> List[Tuple[int, int]]:
    """Merge neighboring segments whose means do not differ significantly or by a meaningful amount."""
    bounds = list(bounds)
    while len(bounds) > 1:
        scores = []
        for (a1, b1), (a2, b2) in zip(bounds[:-1], bounds[1:]):
            m1, m2 = x[a1:b1].mean(), x[a2:b2].mean()
            se = sigma * np.sqrt(1.0 / (b1 - a1) + 1.0 / (b2 - a2))
            z = abs(m1 - m2) / se if se > 0 else np.inf
            scores.append(z if abs(m1 - m2) >= min_shift else min(z, z_merge - 1e-9))
        k = int(np.argmin(scores))
        if scores[k] >= z_merge:
            break
        bounds[k] = (bounds[k][0], bounds[k + 1][1])
        del bounds[k + 1]
    return bounds


def segment_genome(chrom: np.ndarray, start: np.ndarray, end: np.ndarray, log_ratio: np.ndarray,
                   sigma, alpha: float = 0.001, min_size: int = 2) -> pd.DataFrame:
    """
    Segments per chromosome from bins with a finite log ratio. sigma is the per-bin noise, one value or a
    dictionary by chromosome (a single X or Y chromosome carries half the reads, so its bins are noisier).
    Columns: chrom, start, end (bases, 0-based, end exclusive, spanning the first to the last bin of the
    segment), n_bins, mean, sigma, and the bin indices (into the full bin arrays) of the first and last bin.
    """
    rows = []
    for c in CHROMOSOMES:
        idx = np.flatnonzero((chrom == c) & np.isfinite(log_ratio))
        if len(idx) == 0:
            continue
        sd = sigma.get(c, np.nan) if isinstance(sigma, dict) else sigma
        x = smooth_outliers(log_ratio[idx], sd)
        bounds = merge_similar(x, cbs(x, sd, alpha, min_size), sd)
        for a, b in bounds:
            seg = idx[a:b]
            rows.append({"chrom": c, "start": int(start[seg[0]]), "end": int(end[seg[-1]]), "n_bins": int(b - a),
                         "mean": float(np.mean(log_ratio[seg])), "sigma": float(sd), "first_bin": int(seg[0]),
                         "last_bin": int(seg[-1])})
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "n_bins", "mean", "sigma", "first_bin",
                                       "last_bin"])
