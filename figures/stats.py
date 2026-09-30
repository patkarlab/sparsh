"""
Each class against all other arrays, CpG by CpG.

For every class and CpG: the mean beta value in the class and in all other arrays, and their difference; and
Welch's t statistic comparing the M-values (log2 of beta over one minus beta, the scale on which array
methylation is closer to normally distributed; betas are clipped to 0.001 to 0.999 first) of the class with those
of all other arrays, with its two-sided p-value and the Benjamini-Hochberg false discovery rate over the CpGs.
Missing values are left out CpG by CpG. The sums are taken over blocks of CpGs, so the M-values of all arrays
are never held in memory at once.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    if not len(q):
        return out
    order = np.argsort(q)
    ranked = q[order] * len(q) / np.arange(1, len(q) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    fdr = np.empty(len(q))
    fdr[order] = np.minimum(ranked, 1.0)
    out[ok] = fdr
    return out


def one_vs_rest(betas: np.ndarray, classes: np.ndarray, which: Sequence[str], block: int = 10000,
                log=print) -> Dict[str, pd.DataFrame]:
    """{class: DataFrame over all CpGs with mean_in, mean_out, delta, t, df, p, fdr, n_in, n_out}."""
    classes = np.asarray(classes, dtype=object)
    rows = {c: np.flatnonzero(classes == c) for c in which}
    n_cpg = betas.shape[1]
    fields = ("mean_in", "mean_out", "t", "df", "n_in", "n_out")
    out = {c: {f: np.full(n_cpg, np.nan) for f in fields} for c in which}
    for a in range(0, n_cpg, block):
        b = np.asarray(betas[:, a:a + block], dtype=np.float64)
        valid = np.isfinite(b)
        clipped = np.clip(np.where(valid, b, 0.5), 0.001, 0.999)
        m = np.where(valid, np.log2(clipped / (1.0 - clipped)), 0.0)
        b0 = np.where(valid, b, 0.0)
        n_all, s_all, q_all, b_all = valid.sum(0), m.sum(0), (m * m).sum(0), b0.sum(0)
        for c, idx in rows.items():
            v = valid[idx]
            n1, s1, q1, b1 = v.sum(0), m[idx].sum(0), (m[idx] ** 2).sum(0), b0[idx].sum(0)
            n2, s2, q2, b2 = n_all - n1, s_all - s1, q_all - q1, b_all - b1
            with np.errstate(invalid="ignore", divide="ignore"):
                mean1, mean2 = s1 / n1, s2 / n2
                var1 = np.maximum(q1 - n1 * mean1 ** 2, 0.0) / (n1 - 1)
                var2 = np.maximum(q2 - n2 * mean2 ** 2, 0.0) / (n2 - 1)
                se2 = var1 / n1 + var2 / n2
                t = (mean1 - mean2) / np.sqrt(se2)
                df = se2 ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
            o = out[c]
            o["mean_in"][a:a + block], o["mean_out"][a:a + block] = b1 / np.maximum(n1, 1), b2 / np.maximum(n2, 1)
            o["t"][a:a + block], o["df"][a:a + block] = t, df
            o["n_in"][a:a + block], o["n_out"][a:a + block] = n1, n2
        done = min(a + block, n_cpg)
        if done == n_cpg or (a // block) % 10 == 9:
            log(f"  {done} of {n_cpg} CpGs")
    result = {}
    for c in which:
        o = out[c]
        small = (o["n_in"] < 2) | (o["n_out"] < 2)
        for f in ("t", "df"):
            o[f][small] = np.nan
        p = 2.0 * stats.t.sf(np.abs(o["t"]), o["df"])
        frame = pd.DataFrame({"mean_in": o["mean_in"], "mean_out": o["mean_out"],
                              "delta": o["mean_in"] - o["mean_out"], "t": o["t"], "df": o["df"], "p": p,
                              "fdr": benjamini_hochberg(p), "n_in": o["n_in"].astype(int),
                              "n_out": o["n_out"].astype(int)})
        result[c] = frame
    return result
