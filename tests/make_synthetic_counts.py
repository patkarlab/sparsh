#!/usr/bin/env python3
"""
Synthetic read counts for the copy-number tests: a cohort in the format of `scripts/karyotype.py count`.

Every sample gets its own GC bias, every 100 kb bin a mappability factor shared by all samples (as the
library chemistry and aligner would give), and Poisson counting noise. A background cohort of leukaemia-like
samples with a few random changes provides the reference; the named cases below have known answers.

    python tests/make_synthetic_counts.py --out_dir /tmp/counts
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karyotype.counts import BinCounter, write_counts  # noqa: E402
from karyotype.genome import AUTOSOMES, CHROMOSOMES, LENGTHS  # noqa: E402
from karyotype.reference import load_bin_table  # noqa: E402

# name: (sex, purity, reads per Mb, events as (chromosome, start, end or None, copies))
CASES = {
    "CASE_NORMAL_F": ("XX", 0.9, 110, []),
    "CASE_NORMAL_M": ("XY", 0.9, 110, []),
    "CASE_T21_M": ("XY", 0.9, 110, [("chr21", 0, None, 3)]),
    "CASE_HYPER_F": ("XX", 0.85, 110, [("chrX", 0, None, 3), ("chr4", 0, None, 3), ("chr6", 0, None, 3),
                                       ("chr10", 0, None, 3), ("chr14", 0, None, 3), ("chr17", 0, None, 3),
                                       ("chr18", 0, None, 3), ("chr21", 0, None, 4)]),
    "CASE_AML_F": ("XX", 0.7, 110, [("chr7", 0, None, 1), ("chr5", 85_000_000, 150_000_000, 1),
                                    ("chr8", 0, None, 3)]),
    "CASE_MASKED_F": ("XX", 0.9, 110, [("chrX", 0, None, 4), ("chr14", 0, None, 4), ("chr18", 0, None, 4),
                                       ("chr21", 0, None, 4)]),
    "CASE_IAMP21_M": ("XY", 0.9, 110, [("chr21", 15_000_000, 25_000_000, 3), ("chr21", 33_000_000, 40_000_000, 6),
                                       ("chr21", 40_000_000, None, 1)]),
    "CASE_SUB17P_F": ("XX", 0.9, 110, [("chr17", 0, 22_000_000, "f0.4"), ("chr8", 0, None, 3)]),
    "CASE_LOY_M": ("XY", 0.9, 110, [("chrY", 0, None, 0)]),
    "CASE_LOWDEPTH_F": ("XX", 0.9, 12, [("chr21", 0, None, 3)]),
}


def simulate(name, sex, purity, reads_per_mb, events, table, mappability, rng, out_dir):
    chrom = table["chrom"].to_numpy()
    start = table["start"].to_numpy()
    end = table["end"].to_numpy()
    gc = table["gc"].fillna(0.4).to_numpy()
    normal = np.where(chrom == "chrX", 2 if sex == "XX" else 1, np.where(chrom == "chrY", 0 if sex == "XX" else 1, 2))
    tumour = normal.astype(float).copy()
    subclonal = np.zeros(len(chrom))
    for c, a, b, copies in events:
        on = (chrom == c) & (end > a) & (start < (b if b is not None else LENGTHS["hg19"][c]))
        if isinstance(copies, str):                          # "f0.4": one-copy loss in that fraction of all cells
            subclonal[on] = float(copies[1:])
        else:
            tumour[on] = copies
    mean_copies = purity * tumour + (1 - purity) * normal - subclonal
    a, b = rng.normal(0, 1.2), rng.normal(0, 8)
    gc_factor = 2.0 ** (a * (gc - 0.42) + b * (gc - 0.42) ** 2)
    lam = reads_per_mb * 0.1 * (mean_copies / 2.0) * gc_factor * mappability
    lam = np.where(table["excluded"].to_numpy(), lam * 0.05, lam)
    if sex == "XX":
        lam = np.where(chrom == "chrY", reads_per_mb * 0.1 * 0.004, lam)
    counts = rng.poisson(np.maximum(lam, 0))
    counter = BinCounter("hg19", 100_000, 20)
    for c in CHROMOSOMES:
        counter.counts[c] = counts[chrom == c].astype(np.int64)
    counter.stats["counted"] = int(counts.sum())
    counter.read_lengths.extend([2500] * 10)
    write_counts(counter, name, [f"synthetic/{name}.bam"], out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_background", type=int, default=36)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    table = load_bin_table("hg19")
    mappability = rng.lognormal(0.0, 0.12, size=len(table))
    low = rng.random(len(table)) < 0.01
    mappability[low] *= 0.2
    for i in range(args.n_background):
        sex = "XX" if i % 2 else "XY"
        events = []
        for _ in range(rng.integers(0, 3)):
            c = str(rng.choice(AUTOSOMES))
            events.append((c, 0, None, int(rng.choice([1, 3]))))
        simulate(f"BG{i:02d}", sex, float(rng.uniform(0.5, 0.95)), float(rng.uniform(60, 160)), events, table,
                 mappability, rng, out)
    for name, (sex, purity, depth, events) in CASES.items():
        simulate(name, sex, purity, depth, events, table, mappability, rng, out)
    print(f"{args.n_background + len(CASES)} samples written to {out}")


if __name__ == "__main__":
    main()
