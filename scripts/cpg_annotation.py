#!/usr/bin/env python3
"""
CpG annotation for the figures, from Illumina's manifest of the MethylationEPIC array.

    python scripts/cpg_annotation.py --manifest MethylationEPIC_v-1-0_B5.csv --out cpg_annotation_hg19.tsv.gz

The manifest is the CSV Illumina publishes for the array (infinium-methylationepic-v-1-0-b5-manifest-file.csv).
Written, one row per CpG probe (cg...): probe, chrom and pos in GRCh37 (hg19, as the manifest gives them), the
genes of the probe (UCSC RefGene names, each once), their gene regions (TSS1500, TSS200, 5'UTR, 1stExon, Body,
3'UTR) and the relation to CpG islands (Island, N_Shore, S_Shore, N_Shelf, S_Shelf or OpenSea).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def unique_items(value) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return ";".join(dict.fromkeys(v for v in value.split(";") if v))


def read_manifest(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("[Assay]")), None)
    if start is None:
        sys.exit(f"{path}: no [Assay] section; is this an Illumina manifest?")
    end = next((i for i, line in enumerate(lines[start + 1:], start + 1) if line.startswith("[Controls]")),
               len(lines))
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines[start + 1:end])), dtype=str, low_memory=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = read_manifest(args.manifest)
    needed = {"IlmnID", "CHR", "MAPINFO", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island"}
    if needed - set(m.columns):
        sys.exit(f"{args.manifest} lacks the columns {sorted(needed - set(m.columns))}")
    m = m[m["IlmnID"].str.startswith("cg", na=False)]
    out = pd.DataFrame({
        "probe": m["IlmnID"].str.strip(),
        "chrom": "chr" + m["CHR"].str.strip(),
        "pos": pd.to_numeric(m["MAPINFO"], errors="coerce").astype("Int64"),
        "genes": m["UCSC_RefGene_Name"].map(unique_items),
        "gene_regions": m["UCSC_RefGene_Group"].map(unique_items),
        "island": m["Relation_to_UCSC_CpG_Island"].fillna("").replace("", "OpenSea"),
    })
    out = out.dropna(subset=["pos"]).drop_duplicates("probe")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"{len(out)} CpGs written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
