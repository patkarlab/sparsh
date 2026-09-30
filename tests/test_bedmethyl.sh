#!/bin/bash
# From a modkit bedMethyl file to the one-row CSV that scripts/predict.py reads (scripts/bedmethyl_to_csv.py):
# strands added together, other modification codes left out, majority calls with ties left out, and the CSV read
# back with the loader of predict.py. CPU only, a few seconds. Run from the repository folder:
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

printf 'probe\tchrom\tpos\tgenes\tgene_regions\tisland\n' > "$TMP/ann.tsv"
printf 'cg01\tchr1\t1001\tA\tBody\tIsland\ncg02\tchr1\t2001\t\t\tOpenSea\ncg03\tchr2\t501\t\t\tOpenSea\n' >> "$TMP/ann.tsv"
printf 'cg04\tchr2\t901\t\t\tOpenSea\ncg05\tchrX\t77\t\t\tOpenSea\n' >> "$TMP/ann.tsv"
# columns 10-18 separated by spaces, as some modkit versions write them
{
    printf 'chr1\t1000\t1001\tm\t3\t+\t1000\t1001\t255,0,0\t3 66.67 2 1 0 0 0 0 0\n'     # cg01 plus strand: 2 of 3
    printf 'chr1\t1001\t1002\tm\t2\t-\t1001\t1002\t255,0,0\t2 50.00 1 1 0 0 0 0 0\n'     # cg01 minus strand: 1 of 2
    printf 'chr1\t1000\t1001\th\t3\t+\t1000\t1001\t255,0,0\t3 33.33 1 2 0 0 0 0 0\n'     # 5hmC row: not used
    printf 'chr1\t2000\t2001\tm\t2\t.\t2000\t2001\t255,0,0\t2 50.00 1 1 0 0 0 0 0\n'     # cg02: a tie
    printf 'chr2\t500\t501\tm\t1\t.\t500\t501\t255,0,0\t1 0.00 0 1 0 0 0 0 0\n'          # cg03: unmethylated
    printf 'chr2\t700\t701\tm\t4\t.\t700\t701\t255,0,0\t4 100.00 4 0 0 0 0 0 0\n'        # not an array CpG
    printf '2\t900\t901\tm\t5\t.\t900\t901\t255,0,0\t5 100.00 5 0 0 0 0 0 0\n'           # cg04, no chr prefix
} | gzip > "$TMP/sample.bed.gz"

step binary   "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/sample.bed.gz" --annotation "$TMP/ann.tsv" \
                  --sample S1 --out "$TMP/S1.csv"
step fraction "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/sample.bed.gz" --annotation "$TMP/ann.tsv" \
                  --sample S1 --out "$TMP/S1_fraction.csv" --values fraction
# modkit writes plain text unless asked to compress, whatever the file is called; the reader goes by the content
zcat "$TMP/sample.bed.gz" > "$TMP/plain.bed.gz"
step plain    "$PYTHON" scripts/bedmethyl_to_csv.py --bedmethyl "$TMP/plain.bed.gz" --annotation "$TMP/ann.tsv" \
                  --sample S1 --out "$TMP/new_folder/S1_plain.csv"
step refuse_existing bash -c "! '$PYTHON' scripts/bedmethyl_to_csv.py --bedmethyl '$TMP/sample.bed.gz' \
                  --annotation '$TMP/ann.tsv' --sample S1 --out '$TMP/S1.csv' > /dev/null 2>&1"
step contents "$PYTHON" - "$TMP" <<'PY'
import sys
import numpy as np
from data.ont import read_ont_csv
tmp = sys.argv[1]
cpgs = ["cg01", "cg02", "cg03", "cg04", "cg05"]
x, info = read_ont_csv(f"{tmp}/S1.csv", cpgs)
assert np.array_equal(x[[0, 2, 3]], [1.0, 0.0, 1.0]) and np.isnan(x[1]) and np.isnan(x[4]), x   # 3 of 5 reads at cg01
assert info["n_observed"] == 3, info
p, _ = read_ont_csv(f"{tmp}/new_folder/S1_plain.csv", cpgs)
assert np.array_equal(p, x, equal_nan=True), p
f, _ = read_ont_csv(f"{tmp}/S1_fraction.csv", cpgs)
assert np.allclose(f[:4], [0.6, 0.5, 0.0, 1.0]) and np.isnan(f[4]), f
print("contents ok")
PY
echo "BEDMETHYL TEST PASSED"
