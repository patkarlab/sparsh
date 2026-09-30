#!/usr/bin/env python3
"""
SPARSH copy-number karyotype from the low-pass nanopore runs used for methylation.

The same run that gives the methylation calls also gives the read depth along
the genome, and at the depth of a low-pass run that is enough to see gains and
losses of whole chromosomes and chromosome arms, and large segmental changes.
The work is split into steps so that the alignments are read only once:

  manifest  match sample names to the BAM files of their runs (object-store
            listing or file paths) and write a manifest to check by eye;
  count     count reads per bin (100 kb by default) from alignment lines
            streamed on standard input by samtools, and write
            <sample>.counts.tsv.gz with <sample>.counts.json (the JSON is
            written last, so its presence marks a finished sample);
  call      call copy number for counted samples against a reference made
            from the cohort (each sample left out of its own), and write per
            sample a JSON with the ISCN description, karyotype line, ploidy
            class and flags, the segments, an arm table, the bins, and
            figures of the genome and of each chromosome (see
            karyotype/call.py); --no_reference gives a first look at samples
            before a cohort is counted, with GC correction only;
  wig       write the counts as readCounter WIG files (1 Mb bins by default),
            to run ichorCNA on the same reads;
  reference rebuild the GC table of the 100 kb bins from a reference FASTA
            (the tables for hg19 and hg38 are shipped in karyotype/data).

Counting one sample, streaming its BAM from the archive:

    aws s3 cp s3://bucket/run/SAMPLE.bam - | samtools view -F 0x904 -q 20 - | cut -f 2-6 \\
        | python scripts/karyotype.py count --sample SAMPLE --build hg19 --out_dir counts \\
              --source s3://bucket/run/SAMPLE.bam

jobs/karyotype_count.sh does this for every sample of a manifest. Then, once all samples are counted:

    python scripts/karyotype.py call --counts_dir counts --out_dir copy_number --sample_sheet samples.csv

The sample sheet is optional: a CSV with a column 'sample' and, when known, 'sex' (F or M) and
'blast_percent' (0 to 100), which limits the leukaemic fraction to within 20 points of it.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from karyotype.counts import BinCounter, count_paths, safe_name, write_counts  # noqa: E402
from karyotype.manifest import match_samples, read_listing, read_names  # noqa: E402


def cmd_manifest(args) -> int:
    names = read_names(args.names)
    listing = read_listing(args.listing)
    if not names:
        sys.exit(f"No sample names in {args.names}")
    if not listing:
        sys.exit(f"No BAM files in {args.listing}")
    matched, review, unmatched = match_samples(names, listing)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("sample\tbam\tsize_gb\tmatch\n")
        for name in names:
            for uri, size, how in matched.get(name, []):
                fh.write(f"{name}\t{uri}\t{size / 1e9:.2f}\t{how}\n")
    review_path = out.with_name(out.stem + "_review.tsv")
    with open(review_path, "w") as fh:
        fh.write("sample\tcandidate_bam\tsize_gb\treason\n")
        for name, items in review.items():
            for uri, size, how in items:
                fh.write(f"{name}\t{uri}\t{size / 1e9:.2f}\t{how}\n")
    unmatched_path = out.with_name(out.stem + "_unmatched.txt")
    unmatched_path.write_text("".join(f"{n}\n" for n in unmatched))
    n_chunked = sum(1 for items in matched.values() if len(items) > 1)
    print(f"{len(names)} samples, {len(listing)} BAM files listed")
    print(f"  matched: {len(matched)} ({n_chunked} with more than one BAM) -> {out}")
    print(f"  to review (name found only inside a longer path, or a BAM claimed by two samples): {len(review)}"
          f" -> {review_path}")
    print(f"  no BAM found: {len(unmatched)} -> {unmatched_path}")
    print("Rows of the review file can be copied into the manifest once checked.")
    return 0


def cmd_count(args) -> int:
    out_dir = Path(args.out_dir)
    tsv, js = count_paths(out_dir, args.sample)
    if js.exists() and not args.overwrite:
        print(f"{args.sample}: already counted ({js}); pass --overwrite to count again")
        return 0
    counter = BinCounter(args.build, args.bin_size, args.min_mapq)
    stream = sys.stdin if args.input == "-" else open(args.input)
    try:
        counter.add_lines(stream)
    finally:
        if stream is not sys.stdin:
            stream.close()
    if counter.stats["counted"] == 0:
        sys.exit(f"{args.sample}: no reads counted from {counter.stats['lines']} lines "
                 f"({counter.stats}); is the input an aligned stream?")
    summary = write_counts(counter, args.sample, args.source, out_dir)
    rl = summary["read_length"]
    print(f"{args.sample}: {summary['reads_counted']:,} reads counted "
          f"({summary['autosomal_reads_per_mb']:.0f} per Mb of autosome); read length median "
          f"{rl['median']:.0f}, N50 {rl['n50']}; chrX {summary['fraction_chrX']:.4f}, "
          f"chrY {summary['fraction_chrY']:.5f} of reads -> {tsv}")
    return 0


def read_sheet(path):
    import pandas as pd
    if not path:
        return None
    sheet = pd.read_csv(path, dtype=str)
    if "sample" not in sheet.columns:
        sys.exit(f"{path}: the sample sheet needs a column 'sample'")
    sheet["key"] = [safe_name(str(s)).upper() for s in sheet["sample"]]
    if sheet["key"].duplicated().any():
        sys.exit(f"{path}: samples listed twice: {sorted(set(sheet.loc[sheet['key'].duplicated(), 'sample']))}")
    return sheet


def cmd_call(args) -> int:
    from karyotype.call import call_cohort
    from karyotype.genome import canonical_build
    from karyotype.normalise import list_counted, load_cohort

    build = canonical_build(args.build)
    counts_dir, out_dir = Path(args.counts_dir), Path(args.out_dir)
    if not args.no_plots and importlib.util.find_spec("matplotlib") is None:
        sys.exit("The figures need matplotlib: install it (pip install matplotlib), or pass --no_plots")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"{out_dir} is not empty; choose a new --out_dir or pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    counted = list_counted(counts_dir)
    if not counted:
        sys.exit(f"No counted samples in {counts_dir}")
    if args.no_reference and args.reference_samples:
        sys.exit("--no_reference and --reference_samples exclude each other")
    if args.no_reference:
        ref_names = []
    elif args.reference_samples:
        ref_names = read_names(args.reference_samples)
    else:
        ref_names = counted
    to_call = args.samples or counted
    by_key = {safe_name(n).upper(): n for n in counted}
    missing = [n for n in set(ref_names) | set(to_call) if safe_name(n).upper() not in by_key]
    if missing:
        sys.exit(f"Not counted in {counts_dir}: {sorted(missing)[:10]}")
    sheet = read_sheet(args.sample_sheet)
    names = list(dict.fromkeys(by_key[safe_name(n).upper()] for n in list(ref_names) + list(to_call)))
    table, samples = load_cohort(counts_dir, names, build, sheet)
    by_name = {s.name: s for s in samples}
    reference = [by_name[by_key[safe_name(n).upper()]] for n in ref_names]
    calling = [by_name[by_key[safe_name(n).upper()]] for n in to_call]
    if reference and len(reference) < 10:
        print(f"Warning: a reference of {len(reference)} samples is small; 30 or more is better")
    purity = {}
    if sheet is not None and "blast_percent" in sheet.columns:
        for s in calling:
            row = sheet.loc[sheet["key"] == safe_name(s.name).upper()]
            if len(row) and str(row.iloc[0]["blast_percent"]).strip() not in ("", "nan"):
                b = float(row.iloc[0]["blast_percent"]) / 100.0
                if not 0.0 < b <= 1.0:
                    sys.exit(f"blast_percent for {s.name} must be between 0 and 100")
                purity[s.name] = (max(0.05, b - 0.2), min(1.0, b + 0.2))
    bin_size = None if args.bin_size == "auto" else int(args.bin_size)
    limits = samples[0].meta.get("y_density_thresholds", {})
    if limits:
        print(f"Sex from chromosome Y (read density relative to the autosomes, thresholds from the "
              f"{limits['from']}): female below {limits['female_below']}, male above {limits['male_above']}, "
              "chromosome X in between; samples in the sample sheet keep the sex given there")
    print(f"Calling {len(calling)} samples ({build}); "
          f"{'no reference' if not reference else f'reference of {len(reference)} samples'}; bin size "
          f"{'chosen per sample' if bin_size is None else bin_size}")
    summary = call_cohort(calling, reference, table, build, out_dir, bin_size, args.min_reads, args.max_cn,
                          purity, plot=not args.no_plots)
    print(f"Written to {out_dir} (karyotype_summary.csv and one set of files per sample); "
          f"{int((summary.get('qc_status') == 'failed').sum()) if 'qc_status' in summary else 0} failed")
    return 0


def cmd_wig(args) -> int:
    from karyotype.counts import write_wig
    from karyotype.normalise import list_counted

    counts_dir, out_dir = Path(args.counts_dir), Path(args.out_dir)
    names = args.samples or list_counted(counts_dir)
    if not names:
        sys.exit(f"No counted samples in {counts_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        tsv, _ = count_paths(counts_dir, name)
        if not tsv.exists():
            sys.exit(f"{tsv} not found: {name} has not been counted")
        path = write_wig(tsv, out_dir / f"{safe_name(name)}.wig", args.bin_size)
        print(f"{name} -> {path}")
    return 0


def cmd_reference(args) -> int:
    from karyotype.reference import bin_table_path, compute_bin_table, write_bin_table
    out = Path(args.out) if args.out else bin_table_path(args.build)
    table = compute_bin_table(args.fasta, args.build)
    write_bin_table(table, out)
    print(f"{len(table)} bins written to {out}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="SPARSH copy-number karyotype from low-pass nanopore runs",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="match sample names to BAM files")
    m.add_argument("--names", required=True, help="one sample name per line (a .csv ending is dropped)")
    m.add_argument("--listing", required=True,
                   help="BAM files: 'size<TAB>uri' lines, bare URIs or paths, or an 's3 ls' listing")
    m.add_argument("--out", required=True, help="manifest to write (sample, bam, size_gb, match)")
    m.set_defaults(func=cmd_manifest)

    c = sub.add_parser("count", help="count reads per bin from alignment lines on standard input")
    c.add_argument("--sample", required=True)
    c.add_argument("--build", required=True, help="hg19 (GRCh37) or hg38 (GRCh38), as the BAMs were aligned")
    c.add_argument("--out_dir", required=True)
    c.add_argument("--bin_size", type=int, default=100_000)
    c.add_argument("--min_mapq", type=int, default=20)
    c.add_argument("--input", default="-", help="file of alignment lines; '-' reads standard input")
    c.add_argument("--source", nargs="*", default=[], help="BAM files or URIs the lines came from (recorded)")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_count)

    k = sub.add_parser("call", help="call copy number for counted samples")
    k.add_argument("--counts_dir", required=True)
    k.add_argument("--out_dir", required=True)
    k.add_argument("--build", default="hg19")
    k.add_argument("--samples", nargs="*", default=None, help="samples to call (default: every counted sample)")
    k.add_argument("--reference_samples", default=None,
                   help="file of sample names for the reference (default: every counted sample)")
    k.add_argument("--sample_sheet", default=None, help="CSV: sample, and optionally sex (F/M), blast_percent")
    k.add_argument("--bin_size", default="auto", choices=["auto", "500000", "1000000", "2000000", "5000000"])
    k.add_argument("--min_reads", type=float, default=60,
                   help="with --bin_size auto: reads per bin the median autosomal bin must reach")
    k.add_argument("--max_cn", type=int, default=8)
    k.add_argument("--no_plots", action="store_true")
    k.add_argument("--no_reference", action="store_true",
                   help="no reference of other samples: GC correction only, for a first look at a few samples")
    k.add_argument("--overwrite", action="store_true")
    k.set_defaults(func=cmd_call)

    w = sub.add_parser("wig", help="write counts as readCounter WIG files, for ichorCNA")
    w.add_argument("--counts_dir", required=True)
    w.add_argument("--out_dir", required=True)
    w.add_argument("--samples", nargs="*", default=None, help="default: every counted sample")
    w.add_argument("--bin_size", type=int, default=1_000_000, help="a multiple of 100000 (ichorCNA uses 1000000)")
    w.set_defaults(func=cmd_wig)

    r = sub.add_parser("reference", help="rebuild the GC table of the 100 kb bins from a FASTA")
    r.add_argument("--fasta", required=True, help="reference genome FASTA, optionally gzipped")
    r.add_argument("--build", required=True)
    r.add_argument("--out", default=None, help="default: the shipped table in karyotype/data")
    r.set_defaults(func=cmd_reference)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
