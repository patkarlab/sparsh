#!/bin/bash
# From a modkit bedMethyl file to the one-row CSV that scripts/predict.py reads (scripts/bedmethyl_to_csv.py), on a
# small manifest in the layout of Illumina's MethylationEPIC v2.0 file: GRCh37 positions taken from MAPINFO_37, the
# two strands of each CpG added by default (calls of 1 above one half, exact halves left out), also on a compressed
# pileup; with --strands forward, MethPipe's rules (the forward-strand row alone, 1 from 50% modified up); replicate
# probes giving columns of one name, chromosome names with and without chr, and the CSV read back with the loader of
# predict.py.
# A manifest with fewer probe rows than it declares is refused. CPU only, a few seconds. Run from the repository
# folder:
#   bash tests/test_bedmethyl.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PYTHON=${PYTHON:-python}

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

# Seven lines before the column names, Windows line ends and a controls section, as in Illumina's file. MAPINFO is
# the GRCh38 position and must not be used; MAPINFO_37 is the GRCh37 one.
{
    printf 'Illumina, Inc.,,\r\n[Heading],,\r\nDescriptor File Name,EPIC-8v2-0_A2.bpm,\r\nAssay Format,Infinium,\r\n'
    printf 'Date Manufactured,1/1/2023,\r\nLoci Count ,6,\r\n[Assay],,\r\n'
    printf 'IlmnID,Name,CHR,MAPINFO,Genome_Build,CHR_37,MAPINFO_37\r\n'
    printf 'cg01_BC11,cg01,1,99001,GRCh38,1,1001\r\n'          # forward C at 1001: plus-strand row at 1000
    printf 'cg02_TC21,cg02,1,99002,GRCh38,1,2001\r\n'          # a tie: 50%% gives 1
    printf 'cg03_BC11,cg03,2,99003,GRCh38,2,501\r\n'           # unmethylated
    printf 'cg04_BC11,cg04,2,99004,GRCh38,2,901\r\n'           # two replicate probes of one CpG
    printf 'cg04_BC21,cg04,2,99004,GRCh38,2,901\r\n'
    printf 'cg05_BC11,cg05,X,99005,GRCh38,X,\r\n'              # no GRCh37 position: left out
    printf 'cg06_BC11,cg06,1,1001,GRCh38,1,3001\r\n'           # its GRCh38 position matches a row: must not be used
    printf 'cg07_BC11,cg07,1,99007,GRCh38,1,4002\r\n'          # a probe placed on the G of its CpG
    printf '[Controls],,,\r\n21630339,STAINING,Red,DNP (High)\r\n'
} > "$TMP/manifest.csv"
{
    printf 'chr1\t1000\t1001\tC\t3\t+\t1000\t1001\t255,0,0\t3\t66.67\t2\t1\t0\t0\t0\t0\t0\n'     # cg01: 1
    printf 'chr1\t1001\t1002\tC\t2\t-\t1001\t1002\t255,0,0\t2\t0.00\t0\t2\t0\t0\t0\t0\t0\n'      # reverse strand
    printf 'chr1\t2000\t2001\tC\t2\t+\t2000\t2001\t255,0,0\t2\t50.00\t1\t1\t0\t0\t0\t0\t0\n'     # cg02: 1
    printf '2\t500\t501\tC\t1\t+\t500\t501\t255,0,0\t1\t0.00\t0\t1\t0\t0\t0\t0\t0\n'            # cg03: 0
    printf '2\t700\t701\tC\t4\t+\t700\t701\t255,0,0\t4\t100.00\t4\t0\t0\t0\t0\t0\t0\n'          # not a probe
    printf '2\t900\t901\tC\t5\t+\t900\t901\t255,0,0\t5\t80.00\t4\t1\t0\t0\t0\t0\t0\n'           # cg04: 1, twice
    printf 'chr1\t4000\t4001\tC\t2\t+\t4000\t4001\t255,0,0\t2\t100.00\t2\t0\t0\t0\t0\t0\t0\n'   # C of cg07's CpG
    printf 'chr1\t4001\t4002\tC\t1\t-\t4001\t4002\t255,0,0\t1\t0.00\t0\t1\t0\t0\t0\t0\t0\n'     # cg07: 0
} > "$TMP/S1_pileup.bed"
printf ',cg01,cg02,cg03,cg04,cg04,cg07\nS1,1,1,0,1,1,0\n' > "$TMP/expected.csv"
# both strands added: cg01 2 of 5 gives 0, cg02 1 of 2 is left out, cg07 2 of 3 gives 1
printf ',cg01,cg03,cg04,cg04,cg07\nS1,0,0,1,1,1\n' > "$TMP/expected_both.csv"

step convert  "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/S1_pileup.bed" --manifest "$TMP/manifest.csv" \
                  --sample S1 --out "$TMP/out/S1.csv" --strands forward
step exact    cmp "$TMP/out/S1.csv" "$TMP/expected.csv"
gzip -c "$TMP/S1_pileup.bed" > "$TMP/S1_pileup.bed.gz"
step both     "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/S1_pileup.bed.gz" --manifest "$TMP/manifest.csv" \
                  --sample S1 --out "$TMP/out/S1_both.csv" --strands both
step exact_both cmp "$TMP/out/S1_both.csv" "$TMP/expected_both.csv"
step default_both "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/S1_pileup.bed" \
                  --manifest "$TMP/manifest.csv" --sample S1 --out "$TMP/out/S1_default.csv"
step exact_default cmp "$TMP/out/S1_default.csv" "$TMP/expected_both.csv"
step refuse_existing bash -c "! '$PYTHON' scripts/bedmethyl_to_csv.py --bedmethyl '$TMP/S1_pileup.bed' \
                  --manifest '$TMP/manifest.csv' --sample S1 --out '$TMP/out/S1.csv' --strands forward > /dev/null 2>&1"
# a manifest cut short in transfer, with fewer probe rows than its Loci Count line declares, is refused
head -n 12 "$TMP/manifest.csv" > "$TMP/manifest_cut.csv"
step refuse_incomplete_manifest bash -c "! '$PYTHON' scripts/bedmethyl_to_csv.py --bedmethyl '$TMP/S1_pileup.bed' \
                  --manifest '$TMP/manifest_cut.csv' --sample S1 --out '$TMP/out/cut.csv' > '$TMP/cut.log' 2>&1 \
                  && grep -q 'the file is incomplete' '$TMP/cut.log' && [ ! -e '$TMP/out/cut.csv' ]"
# columns 10-18 separated by spaces (modkit pileup --header) are refused rather than read as zeros
awk -F'\t' 'BEGIN {OFS = "\t"} {s = $10; for (i = 11; i <= NF; i++) s = s " " $i; print $1, $2, $3, $4, $5, $6, $7, $8, $9, s}' \
    "$TMP/S1_pileup.bed" > "$TMP/spaces.bed"
step refuse_spaces bash -c "! '$PYTHON' scripts/bedmethyl_to_csv.py --bedmethyl '$TMP/spaces.bed' \
                  --manifest '$TMP/manifest.csv' --sample S1 --out '$TMP/out/spaces.csv' > '$TMP/spaces.log' 2>&1 \
                  && grep -q 'separated by spaces, as modkit writes them' '$TMP/spaces.log'"
step loader "$PYTHON" - "$TMP" <<'PY'
import sys
import numpy as np
from data.ont import read_ont_csv
x, info = read_ont_csv(f"{sys.argv[1]}/out/S1.csv", ["cg01", "cg02", "cg03", "cg04", "cg05", "cg06", "cg07"])
assert np.array_equal(x[[0, 1, 2, 3, 6]], [1.0, 1.0, 0.0, 1.0, 0.0]) and np.isnan(x[4]) and np.isnan(x[5]), x
assert info["n_observed"] == 5 and info["n_repeated_probes"] == 1, info
print("loader ok")
PY
echo "BEDMETHYL TEST PASSED"
