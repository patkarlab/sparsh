#!/usr/bin/env python3
"""
Synthetic input for the pipeline test (tests/test_pipeline.sh): a small reference genome, a manifest in the layout of
Illumina's MethylationEPIC v2.0 file with a probe at every planted CpG, and aligned reads on both strands carrying
modified-base calls (MM and ML tags for 5-methylcytosine and 5-hydroxymethylcytosine) as SAM text.

    python tests/make_synthetic_modbam.py --out_dir /tmp/pipeline

Writes ref.fa, manifest.csv (probes named cg00000000, cg00000001, ... as the CpGs of tests/make_synthetic.py) and
reads.sam, sorted by position. No other packages than the standard library are needed.
"""

import argparse
import random
from pathlib import Path

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reads", type=int, default=800)
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    chroms = {"chr1": 30000, "chr2": 24000}
    ref, cpgs = {}, []
    with open(out / "ref.fa", "w") as fh:
        for c, n in chroms.items():
            s = [rng.choice("ACGT") for _ in range(n)]
            for i in range(1, n - 1):                       # no CpGs but the planted ones
                if s[i - 1] == "C" and s[i] == "G":
                    s[i] = "A"
            for i in range(40, n - 40, 37):
                s[i], s[i + 1] = "C", "G"
                cpgs.append((c, i + 1))                     # 1-based position of the C
            ref[c] = "".join(s)
            fh.write(f">{c}\n" + "\n".join(ref[c][i:i + 60] for i in range(0, n, 60)) + "\n")

    names = [f"Col{i}" for i in range(1, 52)]
    names[0], names[1], names[15], names[16], names[18], names[49], names[50] = \
        "IlmnID", "Name", "CHR", "MAPINFO", "Genome_Build", "CHR_37", "MAPINFO_37"
    lines = ["Illumina, Inc.,,", "[Heading],,", "Descriptor File Name,EPIC-8v2-0_A2.bpm,", "Assay Format,Infinium,",
             "Date Manufactured,1/1/2023,", f"Loci Count ,{len(cpgs)},", "[Assay],,", ",".join(names)]
    for k, (c, pos) in enumerate(cpgs):
        row = [""] * 51
        row[0], row[1], row[15], row[16] = f"cg{k:08d}_BC11", f"cg{k:08d}", c[3:], str(pos + 1000)
        row[18], row[49], row[50] = "GRCh38", c[3:], str(pos)
        lines.append(",".join(row))
    (out / "manifest.csv").write_bytes(("\r\n".join(lines) + "\r\n").encode())

    # a methylation level per CpG, shared by the reads, so that the strands mostly agree
    level = {key: rng.choice([0.05, 0.1, 0.9, 0.95, 0.5]) for key in cpgs}
    reads = []
    for k in range(args.reads):
        c = rng.choice(list(chroms))
        length = rng.randint(600, 2500)
        start = rng.randint(0, chroms[c] - length)
        fwd = ref[c][start:start + length]
        reverse = rng.random() < 0.5
        original = revcomp(fwd) if reverse else fwd         # the read as sequenced
        cs = [i for i, b in enumerate(original) if b == "C"]
        skips, probs_m, probs_h, last = [], [], [], -1
        for j, i in enumerate(cs):
            if i + 1 < len(original) and original[i + 1] == "G":
                ref_c = start + (len(original) - 1 - i - 1 if reverse else i) + 1   # 1-based forward C of the CpG
                p = level.get((c, ref_c), 0.5)
                methylated = rng.random() < p
                pm = rng.randint(200, 250) if methylated else rng.randint(0, 30)
                ph = rng.randint(0, 255 - pm) if methylated and rng.random() < 0.1 else 0
                skips.append(j - last - 1)
                last = j
                probs_m.append(pm)
                probs_h.append(ph)
        sk = ",".join(map(str, skips))
        tags = f"MM:Z:C+h?,{sk};C+m?,{sk};\tML:B:C,{','.join(map(str, probs_h + probs_m))}" if skips else \
            "MM:Z:C+h?;C+m?;"
        flag = 16 if reverse else 0
        reads.append((c, start, f"r{k}\t{flag}\t{c}\t{start + 1}\t60\t{length}M\t*\t0\t0\t{fwd}\t*\t{tags}"))
    reads.sort(key=lambda r: (list(chroms).index(r[0]), r[1]))
    with open(out / "reads.sam", "w") as fh:
        fh.write("@HD\tVN:1.6\tSO:coordinate\n")
        for c, n in chroms.items():
            fh.write(f"@SQ\tSN:{c}\tLN:{n}\n")
        for _, _, line in reads:
            fh.write(line + "\n")
    print(f"{len(cpgs)} CpGs with probes, {len(reads)} reads -> {out}")


if __name__ == "__main__":
    main()
