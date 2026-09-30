"""
Genome builds, chromosome names and cytogenetic bands.

The copy-number module works on the 24 nuclear chromosomes of either GRCh37
(hg19) or GRCh38 (hg38). Alignments name chromosomes either "chr1" or "1";
both are accepted and stored as "chr1". Band names come from the UCSC
cytoBand tables shipped in karyotype/data, which are the ideogram bands at
roughly 850-band resolution that cytogenetic nomenclature uses.
"""

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"

CHROMOSOMES: List[str] = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
AUTOSOMES: List[str] = CHROMOSOMES[:22]

LENGTHS: Dict[str, Dict[str, int]] = {
    "hg19": {
        "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276, "chr5": 180915260,
        "chr6": 171115067, "chr7": 159138663, "chr8": 146364022, "chr9": 141213431, "chr10": 135534747,
        "chr11": 135006516, "chr12": 133851895, "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
        "chr16": 90354753, "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    },
    "hg38": {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
        "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
        "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
        "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    },
}

# Name used in cytogenetic nomenclature for each build, as in "sseq[GRCh37]".
ASSEMBLY_NAME = {"hg19": "GRCh37", "hg38": "GRCh38"}
BUILD_ALIASES = {"hg19": "hg19", "grch37": "hg19", "b37": "hg19", "hg38": "hg38", "grch38": "hg38"}


def canonical_build(build: str) -> str:
    key = str(build).strip().lower()
    if key not in BUILD_ALIASES:
        raise ValueError(f"Unknown genome build {build!r}; use hg19 (GRCh37) or hg38 (GRCh38)")
    return BUILD_ALIASES[key]


def normalise_chrom(name: str) -> Optional[str]:
    """'1', 'chr1', 'X', 'chrX' -> 'chr1' / 'chrX'; None for anything that is not one of the 24 chromosomes."""
    s = str(name).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    if s in ("x", "y"):
        s = s.upper()
    chrom = f"chr{s}"
    return chrom if chrom in CHROMOSOMES else None


def chrom_label(chrom: str) -> str:
    """'chr7' -> '7', 'chrX' -> 'X'."""
    return chrom[3:] if chrom.startswith("chr") else chrom


def chrom_order(chrom: str) -> int:
    return CHROMOSOMES.index(chrom) if chrom in CHROMOSOMES else len(CHROMOSOMES)


def detect_build(lengths: Dict[str, int]) -> Optional[str]:
    """Build whose chromosome 1 length matches, from a sequence dictionary (for example a BAM header)."""
    chr1 = None
    for name, length in lengths.items():
        if normalise_chrom(name) == "chr1":
            chr1 = int(length)
    for build, table in LENGTHS.items():
        if chr1 == table["chr1"]:
            return build
    return None


@dataclass(frozen=True)
class Band:
    chrom: str
    start: int      # 0-based, inclusive
    end: int        # exclusive
    name: str       # e.g. "q14.3"
    stain: str      # gneg, gpos25, ..., acen, gvar, stalk


def load_bands(build: str) -> Dict[str, List[Band]]:
    """UCSC cytoBand table of the build, by chromosome, in order along the chromosome."""
    build = canonical_build(build)
    path = DATA_DIR / f"cytoBand_{build}.txt.gz"
    bands: Dict[str, List[Band]] = {c: [] for c in CHROMOSOMES}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom = normalise_chrom(parts[0])
            if chrom is None:
                continue
            bands[chrom].append(Band(chrom, int(parts[1]), int(parts[2]), parts[3], parts[4]))
    for chrom in bands:
        bands[chrom].sort(key=lambda b: b.start)
    return bands


def centromere(bands: Dict[str, List[Band]], chrom: str) -> Tuple[int, int]:
    """Start and end of the centromere (the 'acen' bands) of a chromosome."""
    acen = [b for b in bands[chrom] if b.stain == "acen"]
    if not acen:
        raise ValueError(f"No centromere band for {chrom}")
    return min(b.start for b in acen), max(b.end for b in acen)


def band_at(bands: Dict[str, List[Band]], chrom: str, position: int) -> str:
    """Band name ('p36.33') containing a 0-based position; the nearest band at the ends."""
    table = bands[chrom]
    for b in table:
        if b.start <= position < b.end:
            return b.name
    return table[0].name if position < table[0].start else table[-1].name


def is_acrocentric(chrom: str) -> bool:
    """Chromosomes whose short arm carries no unique sequence (13, 14, 15, 21, 22)."""
    return chrom in ("chr13", "chr14", "chr15", "chr21", "chr22")
