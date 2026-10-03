"""
Fixed properties of the genome bins: GC content, unknown bases, centromeres, and genes of interest.

The GC fraction and the fraction of unknown bases (N) of every 100 kb bin come
from the reference sequence and are shipped in karyotype/data, one table per
build, so that calling needs no reference sequence. They were computed with
`python scripts/karyotype.py reference`, which can rebuild them from the UCSC
FASTA files (hg19.fa.gz, hg38.fa.gz).

Bins are excluded from calling when more than a tenth of their sequence is
unknown, or when they overlap a centromere, the heterochromatin of 1q12, 9q12
and 16q11.2 ('gvar' bands), the short arm of an acrocentric chromosome, the
pseudoautosomal regions of X and Y (whose reads are shared between the two
chromosomes and fail the mapping-quality filter), or the X-transposed region,
about 4 Mb of Yp11.2 that is 99% identical to Xq21.3, where reads from one
chromosome can be placed on the other. Its limits were found by matching exact
32-base sequences between chrY and chrX of each build, with a margin of
100 kb. Further unreliable bins are found from the cohort itself (see
normalise.py).
"""

import gzip
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .genome import CHROMOSOMES, DATA_DIR, LENGTHS, canonical_build, centromere, is_acrocentric, load_bands, \
    normalise_chrom

FINE_BIN = 100_000
PSEUDOAUTOSOMAL = {
    "hg19": [("chrX", 60_000, 2_699_520), ("chrX", 154_931_043, 155_260_560),
             ("chrY", 10_000, 2_649_520), ("chrY", 59_034_049, 59_363_566)],
    "hg38": [("chrX", 10_000, 2_781_479), ("chrX", 155_701_382, 156_030_895),
             ("chrY", 10_000, 2_781_479), ("chrY", 56_887_902, 57_217_415)],
}


X_TRANSPOSED = {
    "hg19": [("chrX", 88_300_000, 92_500_000), ("chrY", 2_800_000, 6_700_000)],
    "hg38": [("chrX", 89_100_000, 93_200_000), ("chrY", 2_900_000, 6_900_000)],
}


def bin_table_path(build: str) -> Path:
    return DATA_DIR / f"bins_{canonical_build(build)}_100kb.tsv.gz"


def compute_bin_table(fasta: str, build: str, bin_size: int = FINE_BIN) -> pd.DataFrame:
    """GC and N fraction per bin from a (gzipped) FASTA of the build; only the 24 chromosomes are read."""
    build = canonical_build(build)
    lengths = LENGTHS[build]
    opener = gzip.open if str(fasta).endswith(".gz") else open
    rows = []

    def flush(chrom, chunks):
        if chrom is None:
            return
        seq = np.frombuffer(b"".join(chunks).upper(), dtype=np.uint8)
        if len(seq) != lengths[chrom]:
            raise ValueError(f"{chrom} in {fasta} has {len(seq)} bases, not {lengths[chrom]}: is this {build}?")
        gc = (seq == ord("G")) | (seq == ord("C"))
        n = (seq != ord("A")) & (seq != ord("C")) & (seq != ord("G")) & (seq != ord("T"))
        for start in range(0, len(seq), bin_size):
            end = min(start + bin_size, len(seq))
            n_n = int(n[start:end].sum())
            acgt = (end - start) - n_n
            rows.append((chrom, start, end, (int(gc[start:end].sum()) / acgt) if acgt else np.nan,
                         n_n / (end - start)))

    chrom, chunks = None, []
    with opener(fasta, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                flush(chrom, chunks)
                chrom, chunks = normalise_chrom(line[1:].split()[0].decode()), []
                continue
            if chrom is not None:
                chunks.append(line.rstrip())
    flush(chrom, chunks)
    table = pd.DataFrame(rows, columns=["chrom", "start", "end", "gc", "n_fraction"])
    missing = set(CHROMOSOMES) - set(table["chrom"])
    if missing:
        raise ValueError(f"{fasta} lacks {sorted(missing)}")
    table["order"] = table["chrom"].map(CHROMOSOMES.index)
    return table.sort_values(["order", "start"]).drop(columns="order").reset_index(drop=True)


def write_bin_table(table: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt") as fh:
        table.to_csv(fh, sep="\t", index=False, float_format="%.5f")


def load_bin_table(build: str) -> pd.DataFrame:
    """The shipped 100 kb table of the build with a column 'excluded' for bins never used in calling."""
    build = canonical_build(build)
    table = pd.read_csv(bin_table_path(build), sep="\t", dtype={"chrom": str})
    bands = load_bands(build)
    excluded = (table["n_fraction"] > 0.1) | table["gc"].isna()
    for chrom in CHROMOSOMES:
        on = table["chrom"] == chrom
        cen_start, cen_end = centromere(bands, chrom)
        blocked = [(cen_start, cen_end)]
        blocked += [(b.start, b.end) for b in bands[chrom] if b.stain in ("gvar", "stalk")]
        if is_acrocentric(chrom):
            blocked.append((0, cen_end))
        blocked += [(a, b) for c, a, b in PSEUDOAUTOSOMAL[build] + X_TRANSPOSED[build] if c == chrom]
        for a, b in blocked:
            excluded |= on & (table["end"] > a) & (table["start"] < b)
    table["excluded"] = excluded
    return table


GENES_OF_INTEREST = [
    # B-ALL and T-ALL deletions, and genes named in acute leukemia cytogenetics
    "CDKN2A", "CDKN2B", "MTAP", "IKZF1", "PAX5", "ETV6", "RB1", "EBF1", "BTG1", "ERG", "RUNX1", "KMT2A",
    "TP53", "NF1", "PTEN", "LEF1", "CREBBP", "NR3C1", "TBL1XR1", "VPREB1", "CRLF2", "P2RY8", "SH2B3",
    "IKZF2", "IKZF3", "ZNF384", "MYC", "MYB", "NOTCH1", "FBXW7", "LMO2", "TAL1", "STIL", "BCL11B",
    "EZH2", "CUX1", "EGR1", "CSNK1A1", "RPS14", "CDKN1B", "KDM6A", "ASXL1", "DNMT3A", "MECOM", "NUP98",
    "NPM1", "WT1", "PHF6",
]


def gene_table(refgene: str, build: str, genes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Span of each gene of interest (all RefSeq transcripts together) from a UCSC refGene table."""
    wanted = set(genes or GENES_OF_INTEREST)
    cols = ["bin", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount",
            "exonStarts", "exonEnds", "score", "gene", "cdsStartStat", "cdsEndStat", "exonFrames"]
    ref = pd.read_csv(refgene, sep="\t", header=None, names=cols, usecols=["chrom", "txStart", "txEnd", "gene"])
    ref["chrom"] = ref["chrom"].map(normalise_chrom)
    ref = ref[ref["gene"].isin(wanted) & ref["chrom"].notna()]
    spans = []
    for gene, rows in ref.groupby("gene"):
        chrom = rows["chrom"].mode().iloc[0]           # genes in the pseudoautosomal regions are listed on X and Y
        rows = rows[rows["chrom"] == chrom]
        spans.append((gene, chrom, int(rows["txStart"].min()), int(rows["txEnd"].max())))
    out = pd.DataFrame(spans, columns=["gene", "chrom", "start", "end"])
    out["order"] = out["chrom"].map(CHROMOSOMES.index)
    return out.sort_values(["order", "start"]).drop(columns="order").reset_index(drop=True)


def genes_path(build: str) -> Path:
    return DATA_DIR / f"genes_{canonical_build(build)}.tsv"


def load_genes(build: str) -> pd.DataFrame:
    path = genes_path(build)
    if not path.exists():
        return pd.DataFrame(columns=["gene", "chrom", "start", "end"])
    return pd.read_csv(path, sep="\t", dtype={"chrom": str})


def genes_in(genes: pd.DataFrame, chrom: str, start: int, end: int) -> list:
    hit = genes[(genes["chrom"] == chrom) & (genes["end"] > start) & (genes["start"] < end)]
    return hit["gene"].tolist()


def describe_exclusions(table: pd.DataFrame) -> Dict[str, int]:
    return {"bins": int(len(table)), "excluded": int(table["excluded"].sum())}
