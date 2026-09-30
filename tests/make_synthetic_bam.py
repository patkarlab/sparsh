#!/usr/bin/env python3
"""
Synthetic alignments for the copy-number tests: SAM text for one sample with a chosen copy-number profile.

Reads are placed uniformly at random along each chromosome, in proportion to its length times its copy number,
with a lognormal length distribution like that of a rapid (transposase) nanopore library. A share of the reads
are secondary, supplementary or of low mapping quality, so the filters are exercised. A mixture with normal
cells is simulated by --purity: a changed segment then has purity * copies + (1 - purity) * 2 copies on average.

    python tests/make_synthetic_bam.py --profile "chr21:x3,chr7:100000000-159138663:x1,chrX:x3" \\
        --reads 30000 --purity 0.9 --sex XX | samtools view -b -o sample.bam -
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karyotype.genome import CHROMOSOMES, LENGTHS  # noqa: E402


def parse_profile(text):
    """'chr21:x3,chr7:100000000-159138663:x1' -> list of (chrom, start, end, copies)."""
    events = []
    for item in filter(None, (s.strip() for s in text.split(","))):
        parts = item.split(":")
        chrom, copies = parts[0], int(parts[-1].lstrip("x"))
        if len(parts) == 3:
            start, end = (int(v) for v in parts[1].split("-"))
        else:
            start, end = 0, None
        events.append((chrom, start, end, copies))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="hg19")
    ap.add_argument("--profile", default="", help="comma-separated chrom[:start-end]:xN events")
    ap.add_argument("--sex", choices=["XX", "XY"], default="XX")
    ap.add_argument("--purity", type=float, default=1.0)
    ap.add_argument("--reads", type=int, default=20000, help="primary reads for a normal diploid genome")
    ap.add_argument("--median_length", type=float, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no_header", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    lengths = LENGTHS[args.build]
    out = sys.stdout
    if not args.no_header:
        out.write("@HD\tVN:1.6\tSO:coordinate\n")
        for chrom in CHROMOSOMES:
            out.write(f"@SQ\tSN:{chrom}\tLN:{lengths[chrom]}\n")
    base = {c: 2 for c in CHROMOSOMES}
    base["chrY"] = 1 if args.sex == "XY" else 0
    base["chrX"] = 1 if args.sex == "XY" else 2
    events = parse_profile(args.profile)
    genome = sum(lengths[c] for c in CHROMOSOMES[:22])
    per_base = args.reads / (2 * genome)                      # reads per base per copy
    records = []
    for chrom in CHROMOSOMES:
        length = lengths[chrom]
        # piecewise copy number along the chromosome
        cuts = sorted({0, length} | {s for c, s, e, _ in events if c == chrom}
                      | {min(e or length, length) for c, s, e, _ in events if c == chrom})
        for a, b in zip(cuts[:-1], cuts[1:]):
            copies = base[chrom]
            for c, s, e, n in events:
                if c == chrom and s <= a and (e or length) >= b:
                    copies = n
            mean_copies = args.purity * copies + (1 - args.purity) * base[chrom]
            n_reads = rng.poisson(per_base * mean_copies * (b - a))
            starts = rng.integers(a, b, size=n_reads)
            lens = np.maximum(200, rng.lognormal(np.log(args.median_length), 0.8, size=n_reads)).astype(int)
            for st, ln in zip(starts, lens):
                st = int(min(st, length - ln - 1)) if length > ln + 1 else 0
                records.append((chrom, st + 1, int(ln)))
    records.sort(key=lambda r: (CHROMOSOMES.index(r[0]), r[1]))
    for i, (chrom, pos, ln) in enumerate(records):
        flag = 16 if rng.random() < 0.5 else 0
        mapq = 60 if rng.random() > 0.05 else 5              # 5% low mapping quality, filtered
        seq = "*"
        out.write(f"read{i}\t{flag}\t{chrom}\t{pos}\t{mapq}\t{ln}M\t*\t0\t0\t{seq}\t*\n")
        if rng.random() < 0.02:                              # extra secondary and supplementary records
            out.write(f"read{i}\t{flag | 256}\t{chrom}\t{pos}\t0\t{ln}M\t*\t0\t0\t*\t*\n")
            out.write(f"read{i}\t{flag | 2048}\t{chrom}\t{pos}\t60\t{ln}M\t*\t0\t0\t*\t*\n")
    out.write("readU\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n")


if __name__ == "__main__":
    main()
