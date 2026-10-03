"""
Counting reads per genomic bin from aligned nanopore reads.

This is the first step of the copy-number analysis and the only one that
touches the alignments. Reads arrive as text lines from `samtools view`,
either whole SAM records or only the five columns FLAG, RNAME, POS, MAPQ and
CIGAR (`cut -f 2-6`), which keeps the stream small when the BAMs carry
methylation tags. Several BAMs of one sample can be streamed one after the
other into the same count.

A read is counted once, in the bin that holds the midpoint of its alignment
on the reference, when it is a primary alignment (not secondary,
supplementary, unmapped, duplicate or failing quality checks) with a mapping
quality of at least `min_mapq`. Counting reads rather than aligned bases is
deliberate: for a fixed depth, the counting noise of a bin depends on the
number of reads it holds, and adding up bases makes it worse, because long and
short reads then carry very different weights.

Counts are written at a fine resolution (100 kb by default) so that the
calling step can add neighboring bins together to whatever size the depth of
a sample supports, without reading the alignments again.
"""

import gzip
import json
import os
import re
import time
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .genome import CHROMOSOMES, LENGTHS, canonical_build, detect_build, normalise_chrom

SKIP_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
CIGAR_OP = re.compile(r"(\d+)([MIDNSHP=X])")
REF_OPS = set("MDN=X")
QUERY_OPS = set("MIS=X")
FORMAT_VERSION = 1


def cigar_lengths(cigar: str):
    """Reference span and read length of an alignment from its CIGAR string."""
    ref = query = 0
    for n, op in CIGAR_OP.findall(cigar):
        n = int(n)
        if op in REF_OPS:
            ref += n
        if op in QUERY_OPS:
            query += n
    return ref, query


class BinCounter:
    """Accumulates read counts per bin for the 24 chromosomes of one genome build."""

    def __init__(self, build: str, bin_size: int = 100_000, min_mapq: int = 20):
        self.build = canonical_build(build)
        if bin_size < 1000:
            raise ValueError("bin_size below 1 kb is not useful for low-pass data")
        self.bin_size = int(bin_size)
        self.min_mapq = int(min_mapq)
        self.lengths = LENGTHS[self.build]
        self.counts = {c: np.zeros(self.n_bins(c), dtype=np.int64) for c in CHROMOSOMES}
        self.read_lengths = array("I")
        self.stats = {"lines": 0, "header_lines": 0, "counted": 0, "skipped_flag": 0, "low_mapq": 0,
                      "other_contigs": 0, "outside_chromosome": 0, "malformed": 0}
        self.header_lengths: Dict[str, int] = {}

    def n_bins(self, chrom: str) -> int:
        return (self.lengths[chrom] + self.bin_size - 1) // self.bin_size

    def add_header_line(self, line: str) -> None:
        if not line.startswith("@SQ"):
            return
        fields = dict(f.split(":", 1) for f in line.rstrip("\n").split("\t")[1:] if ":" in f)
        if "SN" in fields and "LN" in fields:
            self.header_lengths[fields["SN"]] = int(fields["LN"])

    def add_line(self, line: str) -> None:
        """One alignment line: a SAM record or the columns FLAG RNAME POS MAPQ CIGAR."""
        self.stats["lines"] += 1
        if line.startswith("@"):
            self.stats["header_lines"] += 1
            self.add_header_line(line)
            return
        fields = line.rstrip("\n").split("\t", 11)
        if len(fields) >= 11:                                 # a full SAM record (11 or more columns): drop QNAME
            fields = fields[1:6]
        if len(fields) < 5:
            self.stats["malformed"] += 1
            return
        try:
            flag, rname, pos, mapq, cigar = int(fields[0]), fields[1], int(fields[2]), int(fields[3]), fields[4]
        except ValueError:
            self.stats["malformed"] += 1
            return
        if flag & SKIP_FLAGS:
            self.stats["skipped_flag"] += 1
            return
        if mapq < self.min_mapq:
            self.stats["low_mapq"] += 1
            return
        chrom = normalise_chrom(rname)
        if chrom is None:
            self.stats["other_contigs"] += 1
            return
        span, query = cigar_lengths(cigar) if cigar != "*" else (0, 0)
        middle = pos - 1 + span // 2
        if middle < 0 or middle >= self.lengths[chrom]:
            self.stats["outside_chromosome"] += 1
            return
        self.counts[chrom][middle // self.bin_size] += 1
        self.stats["counted"] += 1
        if query:
            self.read_lengths.append(min(query, 4_294_967_295))

    def add_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            if line.strip():
                self.add_line(line)

    # ---------------------------------------------------------------- summaries
    def read_length_summary(self) -> Dict[str, Optional[float]]:
        if not self.read_lengths:
            return {"n": 0, "median": None, "mean": None, "n50": None}
        lengths = np.frombuffer(self.read_lengths, dtype=np.uint32).astype(np.int64)
        ordered = np.sort(lengths)[::-1]
        cumulative = np.cumsum(ordered)
        n50 = int(ordered[np.searchsorted(cumulative, cumulative[-1] / 2.0)])
        return {"n": int(len(lengths)), "median": float(np.median(lengths)), "mean": float(lengths.mean()),
                "n50": n50}

    def summary(self, sample: str, sources: List[str]) -> dict:
        per_chrom = {c: int(self.counts[c].sum()) for c in CHROMOSOMES}
        total = sum(per_chrom.values())
        autosomal = sum(per_chrom[c] for c in CHROMOSOMES[:22])
        genome = sum(self.lengths[c] for c in CHROMOSOMES[:22])
        header_build = detect_build(self.header_lengths) if self.header_lengths else None
        return {
            "format_version": FORMAT_VERSION,
            "sample": sample,
            "sources": sources,
            "build": self.build,
            "build_from_header": header_build,
            "bin_size": self.bin_size,
            "min_mapq": self.min_mapq,
            "reads_counted": total,
            "reads_per_chromosome": per_chrom,
            "fraction_chrX": per_chrom["chrX"] / total if total else None,
            "fraction_chrY": per_chrom["chrY"] / total if total else None,
            "autosomal_reads_per_mb": autosomal / (genome / 1e6) if genome else None,
            "read_length": self.read_length_summary(),
            "lines": dict(self.stats),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


# -------------------------------------------------------------------- files
def safe_name(sample: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample.strip())


def count_paths(out_dir: Path, sample: str):
    stem = safe_name(sample)
    return out_dir / f"{stem}.counts.tsv.gz", out_dir / f"{stem}.counts.json"


def write_counts(counter: BinCounter, sample: str, sources: List[str], out_dir: Path) -> dict:
    """Write <sample>.counts.tsv.gz and <sample>.counts.json; the JSON is written last and marks completion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv, js = count_paths(out_dir, sample)
    summary = counter.summary(sample, sources)
    if counter.header_lengths and summary["build_from_header"] not in (None, counter.build):
        raise ValueError(f"The alignments are on {summary['build_from_header']}, not {counter.build}; "
                         "pass the right --build")
    tmp_tsv = tsv.with_name(tsv.name + ".part")
    with gzip.open(tmp_tsv, "wt") as fh:
        fh.write(f"# sample\t{sample}\n# build\t{counter.build}\n# bin_size\t{counter.bin_size}\n"
                 f"# min_mapq\t{counter.min_mapq}\n")
        fh.write("chrom\tstart\tend\treads\n")
        for chrom in CHROMOSOMES:
            length = counter.lengths[chrom]
            for i, n in enumerate(counter.counts[chrom]):
                start = i * counter.bin_size
                fh.write(f"{chrom}\t{start}\t{min(start + counter.bin_size, length)}\t{int(n)}\n")
    os.replace(tmp_tsv, tsv)
    tmp_js = js.with_name(js.name + ".part")
    with open(tmp_js, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp_js, js)
    return summary


def read_counts(path: Path):
    """Read a counts file back: (metadata dict, pandas DataFrame with chrom, start, end, reads)."""
    import pandas as pd
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].strip().partition("\t")
            meta[key.strip()] = value.strip()
    df = pd.read_csv(path, sep="\t", comment="#", dtype={"chrom": str, "start": np.int64, "end": np.int64,
                                                          "reads": np.int64})
    return meta, df


def write_wig(counts_tsv: Path, out: Path, bin_size: int = 1_000_000) -> Path:
    """
    The 100 kb counts summed into bins of bin_size and written as readCounter writes them for ichorCNA: one
    fixedStep block per chromosome, bins from position 1, the last bin of a chromosome shorter than the rest.
    """
    header, df = read_counts(counts_tsv)
    fine = int(header.get("bin_size", 100_000))
    if bin_size % fine:
        raise ValueError(f"bin size {bin_size} is not a multiple of the counted bins ({fine})")
    with open(out, "w") as fh:
        for chrom in CHROMOSOMES:
            rows = df[df["chrom"] == chrom]
            if rows.empty:
                continue
            groups = (rows["start"].to_numpy() // bin_size)
            sums = np.bincount(groups, weights=rows["reads"].to_numpy(dtype=float))
            fh.write(f"fixedStep chrom={chrom} start=1 step={bin_size} span={bin_size}\n")
            fh.write("".join(f"{int(v)}\n" for v in sums))
    return out

