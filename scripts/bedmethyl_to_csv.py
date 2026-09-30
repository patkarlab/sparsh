#!/usr/bin/env python3
r"""
A nanopore sample in the form SPARSH reads: one CSV row with a value for every array CpG that its reads cover.

    python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE.bed.gz --annotation cpg_annotation_hg19.tsv.gz \
        --sample SAMPLE --out SAMPLE.csv

The input is the bedMethyl file that `modkit pileup` writes for 5-methylcytosine at CpG sites, from reads aligned
to the genome build of the annotation (GRCh37, hg19, for the annotation made by scripts/cpg_annotation.py). Only
rows of the modification code given by --code (default m, 5-methylcytosine) are used, so rows of
5-hydroxymethylcytosine (h) are left out. The reads of the two strands of a CpG are added together, so the file may
or may not have been made with --combine-strands. An array CpG is found by the position of its C on the forward
strand, which is the annotation's pos (MAPINFO in Illumina's manifest).

Values written (--values):
  binary    1 when most reads at the CpG are methylated and 0 when most are not (the default); a CpG with as many
            methylated as unmethylated reads is left out. The released model was trained for such calls.
  fraction  the methylated fraction of the reads.
CpGs with fewer than --min_reads reads that were called methylated or unmethylated (default 1) are left out.
A summary line gives the CpGs matched and the reads per matched CpG.
"""

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def open_text(path: str):
    """A plain or gzip-compressed (also bgzip) text file, recognised by its first two bytes, not its name."""
    with open(path, "rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if compressed else open(path)


def read_bedmethyl(path: str, positions: dict, code: str = "m"):
    """{probe: [methylated reads, unmethylated reads]} for the array CpGs in positions {(chrom, pos): probe}."""
    counts = defaultdict(lambda: [0, 0])
    lines = used = 0
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            lines += 1
            f = line.split()                # modkit separates the last columns by spaces in some versions
            if len(f) < 13 or f[3] != code:
                continue
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            start, strand = int(f[1]), f[5]
            pos = start if strand == "-" else start + 1     # the forward-strand C of the CpG, 1-based
            probe = positions.get((chrom, pos))
            if probe is None:
                continue
            used += 1
            c = counts[probe]
            c[0] += int(f[11])              # Nmod
            c[1] += int(f[12])              # Ncanonical
    return counts, lines, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bedmethyl", required=True, help="modkit pileup output, plain or compressed")
    ap.add_argument("--annotation", required=True, help="CpG annotation from scripts/cpg_annotation.py")
    ap.add_argument("--sample", required=True, help="sample name, written as the row name")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--values", choices=["binary", "fraction"], default="binary")
    ap.add_argument("--min_reads", type=int, default=1)
    ap.add_argument("--code", default="m", help="modification code of the rows to use (default m)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; choose a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(args.annotation, sep="\t", usecols=["probe", "chrom", "pos"], dtype={"chrom": str})
    ann = ann.dropna(subset=["pos"])
    positions = {(c, int(p)): probe for probe, c, p in zip(ann["probe"], ann["chrom"], ann["pos"])}
    counts, lines, used = read_bedmethyl(args.bedmethyl, positions, args.code)

    values = {}
    ties = few = 0
    for probe in ann["probe"]:
        if probe not in counts:
            continue
        m, u = counts[probe]
        if m + u < args.min_reads or m + u == 0:
            few += 1
            continue
        if args.values == "fraction":
            values[probe] = m / (m + u)
        elif m == u:
            ties += 1
        else:
            values[probe] = 1 if m > u else 0
    if not values:
        sys.exit(f"No array CpG with reads in {args.bedmethyl}: check the genome build and the modification code")
    depth = sum(sum(counts[p]) for p in values) / len(values)
    row = pd.DataFrame([values], index=pd.Index([args.sample], name="sample"))
    row.to_csv(out, float_format="%.4g")
    print(f"{args.sample}: {len(values)} of {len(ann)} array CpGs with a value ({len(values) / len(ann):.1%}), "
          f"{depth:.2f} reads per CpG; {lines} bedMethyl rows read, {used} at array CpGs"
          + (f"; left out: {ties} with as many methylated as unmethylated reads" if ties else "")
          + (f", {few} with fewer than {args.min_reads} reads" if few else "") + f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
