#!/bin/bash
# Copy-number counting on synthetic data: manifest matching and read counting through a stand-in object store. A
# file named like a piece of a sample but stored with another run (here S_NORMAL_10, as a dilution would be named) is
# listed for review, not pooled with the sample.
# Needs samtools (set SAMTOOLS if it is not on PATH); skipped without it. CPU only, under a minute. Run from the
# repository folder:
#   bash tests/test_karyotype_count.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SAMTOOLS=${SAMTOOLS:-$(command -v samtools || echo samtools)}
PYTHON=${PYTHON:-python}
[ -x "$SAMTOOLS" ] || { echo "SKIPPED: samtools not found (set SAMTOOLS=/path/to/samtools)"; exit 0; }

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

mkdir -p "$TMP/store/run1" "$TMP/store/run2" "$TMP/bin"
bam() {  # bam <path> <arguments for make_synthetic_bam.py>
    local path=$1; shift
    "$PYTHON" tests/make_synthetic_bam.py "$@" | "$SAMTOOLS" view -b -o "$path" -
}
step bams bash -c "$(declare -f bam); PYTHON=$PYTHON SAMTOOLS=$SAMTOOLS
    bam '$TMP/store/run1/S_NORMAL.bam' --reads 20000 --sex XX --seed 1 &&
    bam '$TMP/store/run1/S_ABN.bam' --reads 20000 --sex XY --profile chr21:x3 --seed 2 &&
    bam '$TMP/store/run2/S_CHUNK_0.bam' --reads 10000 --sex XX --seed 3 &&
    bam '$TMP/store/run2/S_CHUNK_1.bam' --reads 10000 --sex XX --seed 4 &&
    bam '$TMP/store/run2/S_NORMAL_10.bam' --reads 5000 --sex XX --seed 5"
cat > "$TMP/bin/fakes3" <<EOF
#!/bin/bash
[ "\$1" = get ] || exit 2
cat "$TMP/store/\${2#s3://test/}"
EOF
chmod +x "$TMP/bin/fakes3"
( cd "$TMP/store" && for f in */*.bam; do printf '%s\ts3://test/%s\n' "$(stat -c %s "$f")" "$f"; done ) > "$TMP/listing.tsv"
printf 'S_NORMAL.csv\nS_ABN\nS_CHUNK\nS_MISSING\nNORMAL\n' > "$TMP/names.txt"

step manifest "$PYTHON" scripts/karyotype.py manifest --names "$TMP/names.txt" --listing "$TMP/listing.tsv" \
                  --out "$TMP/manifest.tsv"
step count env PATH="$TMP/bin:$PATH" FETCH="fakes3 get" SAMTOOLS="$SAMTOOLS" PYTHON="$PYTHON" \
                  bash jobs/karyotype_count.sh "$TMP/manifest.tsv" "$TMP/counts" 2
step rerun_skips bash -c "env PATH='$TMP/bin:$PATH' FETCH='fakes3 get' SAMTOOLS='$SAMTOOLS' PYTHON='$PYTHON' \
                  bash jobs/karyotype_count.sh '$TMP/manifest.tsv' '$TMP/counts' 2 | grep -c '^skip' | grep -q 3"
printf 'sample\tbam\tsize_gb\tmatch\nS_BROKEN\ts3://test/run1/NOT_THERE.bam\t0\texact\n' > "$TMP/broken.tsv"
step failure_cleans bash -c "env PATH='$TMP/bin:$PATH' FETCH='fakes3 get' SAMTOOLS='$SAMTOOLS' PYTHON='$PYTHON' \
                  bash jobs/karyotype_count.sh '$TMP/broken.tsv' '$TMP/counts_broken' 1 | grep -q '^FAIL' &&
                  [ ! -f '$TMP/counts_broken/S_BROKEN.counts.json' ]"
step contents "$PYTHON" - "$TMP" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
tmp = Path(sys.argv[1])
man = pd.read_csv(tmp / "manifest.tsv", sep="\t")
assert sorted(man["sample"].unique()) == ["S_ABN", "S_CHUNK", "S_NORMAL"], man
assert (man["sample"] == "S_CHUNK").sum() == 2
assert (man["sample"] == "S_NORMAL").sum() == 1 and not man["bam"].str.contains("S_NORMAL_10").any(), man
review = pd.read_csv(tmp / "manifest_review.tsv", sep="\t")
assert set(review["sample"]) == {"NORMAL", "S_NORMAL"}, review
other_run = review[review["sample"] == "S_NORMAL"]
assert len(other_run) == 1 and other_run["candidate_bam"].str.endswith("run2/S_NORMAL_10.bam").all(), other_run
assert (other_run["reason"] == "piece in another folder").all(), other_run
assert (tmp / "manifest_unmatched.txt").read_text().split() == ["S_MISSING"]
assert len(json.load(open(tmp / "counts" / "S_NORMAL.counts.json"))["sources"]) == 1
for s in ("S_NORMAL", "S_ABN", "S_CHUNK"):
    d = json.load(open(tmp / "counts" / f"{s}.counts.json"))
    assert d["reads_counted"] > 15000, (s, d["reads_counted"])
    assert d["bin_size"] == 100000 and d["build"] == "hg19"
    counts = pd.read_csv(tmp / "counts" / f"{s}.counts.tsv.gz", sep="\t", comment="#")
    assert counts["reads"].sum() == d["reads_counted"]
    assert len(counts) == sum((L + 99999) // 100000 for L in [249250621, 243199373, 198022430, 191154276,
        180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878,
        107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566,
        155270560, 59373566])
abn = json.load(open(tmp / "counts" / "S_ABN.counts.json"))["reads_per_chromosome"]
norm = json.load(open(tmp / "counts" / "S_NORMAL.counts.json"))["reads_per_chromosome"]
rest = [f"chr{i}" for i in range(1, 21)] + ["chr22"]
def share21(d):
    return d["chr21"] / sum(d[c] for c in rest)
ratio = share21(abn) / share21(norm)
assert 1.3 < ratio < 1.7, ratio              # trisomy 21: 1.5
assert abn["chrY"] > 50 and norm["chrY"] == 0
chunk = json.load(open(tmp / "counts" / "S_CHUNK.counts.json"))
assert len(chunk["sources"]) == 2
print("contents ok")
PY
echo "KARYOTYPE COUNT TEST PASSED"
