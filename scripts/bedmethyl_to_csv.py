#!/usr/bin/env python3
r"""
A nanopore sample in the form SPARSH reads: one CSV row with a methylation call, 1 or 0, for every array probe whose
CpG the reads cover.

    python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE_pileup.bed --manifest EPIC-8v2-0_A2.csv \
        --sample SAMPLE --out SAMPLE.csv

This is the matrix step of the laboratory's MethPipe pipeline (intersect_epic.py), written again without its
dependence on bedtools; on test data the two give the same file, byte for byte (tests/test_bedmethyl.sh checks the
rules). Its input is the bedMethyl file of

    modkit pileup --ref hg19.fasta --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed

for reads aligned to GRCh37 (hg19): one row for each strand of every CpG with reads, 5-methylcytosine and
5-hydroxymethylcytosine counted together as modified, and the strands kept apart.

The probes come from Illumina's manifest of the MethylationEPIC v2.0 array (EPIC-8v2-0_A2.csv; its column names are
on line 8): the probe identifier from the first column whose name contains "ilmn" (IlmnID), the chromosome from the
column CHR, and the position from the first column whose name contains both "mapinfo" and "37", the GRCh37 position
MAPINFO_37 (without such a column, the first whose name contains "mapinfo"). Chromosome names are compared without
"chr", other punctuation and case, and only chromosomes 1 to 22, X and Y are used. Each probe stands for the single
base at its position, the C of its CpG on the forward strand, so it takes the pileup row that starts there; the row
of the reverse strand starts one base later, on the G, and is not used. The value is 1 when the row's percentage of
modified calls is 50 or more and 0 otherwise (the mean percentage is taken if more than one row overlaps a probe).

Columns are named by the probe identifier up to its first underscore (cg25324105 for cg25324105_BC11), in the order
of the full identifiers, so the replicate probes of one CpG give columns with the same name; scripts/predict.py
combines such columns (by default their mean). The row is named after --sample. A summary line gives the pileup rows
read, the probes with a call and the mean and median methylated fraction before the calls are made.

The manifest is checked first: a file with fewer probe rows than its 'Loci Count' line declares, such as a copy cut
short in transfer, is refused, since it would otherwise give matrices with few probes and no error.

--strands both is a variant under evaluation, not MethPipe's method: the reads of both strands of the probe's
CpG are added (modified calls over valid calls of the two rows), the call is 1 above one half and 0 below, and a CpG
whose reads are split exactly in half is left out. The pileup may be gzip-compressed.
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VALID_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
BEDMETHYL_COLUMNS = ["chrom", "start", "end", "base", "score", "strand", "thickStart", "thickEnd", "itemRgb",
                     "n_validcov", "percent", "n_mod", "n_canon", "n_other", "n_del", "n_fall", "n_diff", "n_nocal"]
CHUNK_ROWS = 2_000_000


def clean_chrom(values: pd.Series) -> pd.Series:
    """Chromosome names as compared: 'chr' removed, then every character other than a letter, digit or underscore,
    in upper case."""
    return (values.astype(str).str.strip()
            .str.replace("chr", "", regex=False)
            .str.replace(r"[^\w]", "", regex=True)
            .str.upper())


def manifest_rows(path: Path):
    """The number of probes the manifest declares on its 'Loci Count' line (None without such a line) and the number
    of rows in its probe section: the lines after the column names, up to the next section line ([Controls]) or the
    end of the file. Lines are counted as pandas counts them for header=7, blank lines left out."""
    declared, rows, seen = None, 0, 0
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            if seen <= 7:
                text = line.decode(errors="replace")
                if text.lower().startswith("loci count"):
                    numbers = [f.strip() for f in text.split(",")[1:] if f.strip().isdigit()]
                    declared = int(numbers[0]) if numbers else None
                continue
            if seen == 8:
                continue                                   # the column names
            if line.startswith(b"["):
                break
            rows += 1
    return declared, rows


def check_manifest(path: Path) -> None:
    """Refuse a manifest that holds fewer probe rows than it declares."""
    declared, rows = manifest_rows(path)
    if declared is not None and rows < declared:
        raise SystemExit(f"{path}: {rows:,} probe rows, but the manifest declares {declared:,} on its Loci Count line; "
                         "the file is incomplete, for example a copy that was cut short. Copy it again and compare "
                         "its checksum with the original's.")


def manifest_columns(path: Path):
    """The probe, chromosome and position columns of the manifest, chosen as MethPipe chooses them."""
    columns = [c.strip() for c in pd.read_csv(path, header=7, nrows=0).columns]
    probe = [c for c in columns if "ilmn" in c.lower()]
    if not probe:
        raise SystemExit(f"{path}: no probe identifier column (a name containing 'ilmn') on line 8")
    chrom = [c for c in columns if c.lower() == "chr"]
    if not chrom:
        raise SystemExit(f"{path}: no column 'CHR' on line 8")
    position = [c for c in columns if "mapinfo" in c.lower() and "37" in c.lower()] or \
               [c for c in columns if "mapinfo" in c.lower()]
    if not position:
        raise SystemExit(f"{path}: no column whose name contains 'MAPINFO' on line 8")
    return probe[0], chrom[0], position[0]


def read_probes(path: Path) -> pd.DataFrame:
    """Probes as one-base intervals: chrom, start (0-based), end, probe_id (the full identifier)."""
    check_manifest(path)
    probe_col, chrom_col, pos_col = manifest_columns(path)
    manifest = pd.read_csv(path, header=7, low_memory=False)
    manifest.columns = [c.strip() for c in manifest.columns]
    probes = manifest[[probe_col, chrom_col, pos_col]].copy()
    probes.columns = ["probe_id", "chrom", "MAPINFO"]
    probes["chrom"] = clean_chrom(probes["chrom"])
    probes["MAPINFO"] = pd.to_numeric(probes["MAPINFO"], errors="coerce").fillna(0).astype(int)
    probes["start"] = probes["MAPINFO"] - 1
    probes["end"] = probes["MAPINFO"]
    keep = probes["chrom"].isin(VALID_CHROMS) & (probes["start"] < probes["end"]) & (probes["start"] >= 0)
    probes = probes[keep]
    print(f"Manifest: {len(probes):,} probes placed (identifier {probe_col}, chromosome {chrom_col}, "
          f"position {pos_col})", flush=True)
    return probes[["chrom", "start", "end", "probe_id"]].reset_index(drop=True)


def first_line(path: Path) -> str:
    with open(path, "rb") as fh:
        compressed = fh.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rt") if compressed else open(path)) as fh:
        return next((line for line in fh if line.strip()), "")


def read_pileup(path: Path, probes: pd.DataFrame):
    """
    The pileup rows at a probe or at the other strand of its CpG, joined to the probe: chrom, start, end, strand,
    meth_5mc (the percentage of modified calls, cleaned as MethPipe cleans it), n_valid, n_mod, probe_id and role
    ('own' for the row at the probe's base, 'other' for the other strand of its CpG); and the number of rows on
    chromosomes 1 to 22, X and Y. The other strand is the minus-strand row one base after a probe on the C, or the
    plus-strand row one base before a probe on the G; a CpG cannot have both.
    """
    first = first_line(path)
    if not first:
        raise SystemExit(f"{path} is empty")
    fields = first.rstrip("\n").split("\t")
    if len(fields) < 12 or " " in fields[9]:
        raise SystemExit(f"{path}: expected 18 columns separated by tabs, found {len(fields)}"
                         + ("; the columns after the ninth are separated by spaces, as modkit writes them with "
                            "--header, so run modkit pileup without it" if len(fields) > 9 and " " in fields[9]
                            else ""))
    keys = probes[["chrom", "start", "probe_id"]].rename(columns={"start": "base"})
    targets = [("own", None, keys),
               ("other", "-", keys.assign(base=keys["base"] + 1)),
               ("other", "+", keys.assign(base=keys["base"] - 1))]
    kept, total = [], 0
    reader = pd.read_csv(path, sep="\t", header=None, names=BEDMETHYL_COLUMNS, usecols=[0, 1, 2, 5, 9, 10, 11],
                         dtype=str, chunksize=CHUNK_ROWS, compression="infer")
    for chunk in reader:
        rows = pd.DataFrame({
            "chrom": clean_chrom(chunk["chrom"]),
            "start": pd.to_numeric(chunk["start"], errors="coerce").fillna(0).astype(int),
            "end": pd.to_numeric(chunk["end"], errors="coerce").fillna(0).astype(int),
            "strand": chunk["strand"].astype(str).str.strip(),
            "meth_5mc": pd.to_numeric(chunk["percent"], errors="coerce").fillna(0).astype(float),
            "n_valid": pd.to_numeric(chunk["n_validcov"], errors="coerce").fillna(0).astype(int),
            "n_mod": pd.to_numeric(chunk["n_mod"], errors="coerce").fillna(0).astype(int),
        })
        rows = rows[rows["chrom"].isin(VALID_CHROMS) & (rows["start"] < rows["end"]) & (rows["start"] >= 0)]
        total += len(rows)
        length = (rows["end"] - rows["start"]).to_numpy()
        if (length > 1).any():                     # a row longer than one base overlaps every probe inside it
            rows = rows.loc[rows.index.repeat(length)].copy()
            rows["base"] = rows["start"] + np.concatenate([np.arange(n) for n in length])
        else:
            rows = rows.assign(base=rows["start"])
        for role, strand, table in targets:
            part = rows if strand is None else rows[rows["strand"] == strand]
            kept.append(part.merge(table, on=["chrom", "base"]).drop(columns="base").assign(role=role))
    overlap = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["chrom", "start", "end", "strand", "meth_5mc", "n_valid", "n_mod", "probe_id", "role"])
    return overlap, total


def forward_calls(overlap: pd.DataFrame) -> pd.Series:
    """MethPipe's calls: the mean percentage of the rows at the probe's base, 1 from 50 up, by full identifier."""
    own = overlap[overlap["role"] == "own"]
    return own.groupby("probe_id")["meth_5mc"].mean()


def both_strand_calls(overlap: pd.DataFrame):
    """The two-strand variant: modified over valid calls of both rows of the CpG; exact halves left out."""
    sums = overlap.groupby("probe_id")[["n_mod", "n_valid"]].sum()
    sums = sums[sums["n_valid"] > 0]
    fraction = sums["n_mod"] / sums["n_valid"]
    ties = int((sums["n_mod"] * 2 == sums["n_valid"]).sum())
    fraction = fraction[sums["n_mod"] * 2 != sums["n_valid"]]
    per_strand = overlap[overlap["n_valid"] > 0].assign(
        call=lambda d: np.sign(2 * d["n_mod"] - d["n_valid"])).pivot_table(
        index="probe_id", columns="role", values="call", aggfunc="first")
    both = per_strand.dropna() if {"own", "other"} <= set(per_strand.columns) else per_strand.iloc[0:0]
    decided = both[(both["own"] != 0) & (both["other"] != 0)] if len(both) else both
    stats = {"ties": ties, "both_strands": int(len(both)),
             "strands_agree": int((decided["own"] == decided["other"]).sum()) if len(decided) else 0,
             "strands_decided": int(len(decided))}
    return fraction * 100, stats


def write_row(per_probe: pd.Series, sample: str, out: Path, threshold_rule: str) -> pd.DataFrame:
    matched = per_probe.reset_index()
    matched.columns = ["probe_id", "meth_5mc"]
    matched["probe_id"] = matched["probe_id"].astype(str).str.split("_").str[0]
    if threshold_rule == "at_least_half":
        matched["meth_5mc"] = (matched["meth_5mc"] / 100 >= 0.5).astype(int)
    else:
        matched["meth_5mc"] = (matched["meth_5mc"] / 100 > 0.5).astype(int)
    row = matched.set_index("probe_id").T
    row.index = [sample]
    out.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(out, index=True)
    return matched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bedmethyl", required=True, help="output of modkit pileup (see above)")
    ap.add_argument("--manifest", required=True, help="Illumina's MethylationEPIC v2.0 manifest (CSV)")
    ap.add_argument("--sample", required=True, help="sample name, written as the row name")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--strands", choices=["forward", "both"], default="forward",
                    help="forward: MethPipe's method (default); both: the variant under evaluation")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; choose a new --out")
    probes = read_probes(Path(args.manifest))
    overlap, total = read_pileup(Path(args.bedmethyl), probes)
    if total == 0:
        sys.exit(f"{args.bedmethyl}: no pileup rows on chromosomes 1 to 22, X or Y")
    own = overlap[overlap["role"] == "own"]
    if own.empty and (args.strands == "forward" or overlap.empty):
        sys.exit(f"{args.bedmethyl}: no pileup row at a probe position; check that the reads are aligned to GRCh37 "
                 "and that the manifest is the MethylationEPIC v2.0 one")
    if args.strands == "forward":
        per_probe = forward_calls(overlap)
        matched = write_row(per_probe, args.sample, out, "at_least_half")
        print(f"{args.sample}: {total:,} pileup rows, {len(own):,} at probe positions, {len(matched):,} probes with "
              f"a call ({int(matched['meth_5mc'].sum()):,} methylated); methylated fraction mean "
              f"{per_probe.mean() / 100:.3f}, median {per_probe.median() / 100:.3f} -> {out}")
    else:
        per_probe, stats = both_strand_calls(overlap)
        matched = write_row(per_probe, args.sample, out, "more_than_half")
        print(f"{args.sample}: {total:,} pileup rows, {len(overlap):,} at the CpGs of probes, {len(matched):,} probes "
              f"with a call, the reads of both strands added ({int(matched['meth_5mc'].sum()):,} methylated), "
              f"{stats['ties']:,} left out as exact halves; both strands called at {stats['strands_decided']:,} "
              f"probes, the same call at {stats['strands_agree']:,} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
