#!/bin/bash
# Count reads per bin for every sample of a manifest, streaming each sample's BAM files once.
#
#   bash jobs/karyotype_count.sh MANIFEST OUT_DIR [PARALLEL]
#
# MANIFEST is the file written by "python scripts/karyotype.py manifest" (columns sample, bam, ...; a sample
# with several BAMs has several rows). OUT_DIR receives <sample>.counts.tsv.gz and <sample>.counts.json per
# sample and a log per sample in OUT_DIR/logs. A sample whose .counts.json exists is skipped, so the script
# can be stopped and started again. PARALLEL (default 2) is the number of samples streamed at the same time.
#
# Settings, from the environment:
#   FETCH     command that writes an object to standard output when given "<uri> -"  (default: aws s3 cp)
#   SAMTOOLS  samtools binary (default: samtools on PATH)
#   BUILD     genome build of the alignments, hg19 or hg38 (default: hg19)
#   MIN_MAPQ  minimum mapping quality (default: 20)
#   BIN_SIZE  bin size in bases (default: 100000)
#   PYTHON    python with numpy and pandas (default: python)
# Paths that do not start with s3:// are read from disk.

set -uo pipefail
MANIFEST=${1:?usage: bash jobs/karyotype_count.sh MANIFEST OUT_DIR [PARALLEL]}
OUT_DIR=${2:?usage: bash jobs/karyotype_count.sh MANIFEST OUT_DIR [PARALLEL]}
PARALLEL=${3:-2}
REPO=$(cd "$(dirname "$0")/.." && pwd)

export FETCH=${FETCH:-aws s3 cp}
export BUILD=${BUILD:-hg19}
export MIN_MAPQ=${MIN_MAPQ:-20}
export BIN_SIZE=${BIN_SIZE:-100000}
export PYTHON=${PYTHON:-python}
if [ -z "${SAMTOOLS:-}" ]; then
    SAMTOOLS=$(command -v samtools || echo samtools)
fi
export SAMTOOLS REPO OUT_DIR MANIFEST

[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST"; exit 1; }
[ -x "$SAMTOOLS" ] || { echo "ERROR: samtools not found at $SAMTOOLS; set SAMTOOLS=/path/to/samtools"; exit 1; }
if grep -q $'\ts3://' "$MANIFEST" && ! type "${FETCH%% *}" >/dev/null 2>&1; then
    echo "ERROR: the manifest has s3:// objects but '${FETCH%% *}' is not a command in this shell."
    echo "If it is an alias or a function of your login shell, set FETCH to what it stands for."
    exit 1
fi
"$PYTHON" -c "import numpy, pandas" 2>/dev/null || { echo "ERROR: $PYTHON lacks numpy or pandas"; exit 1; }
mkdir -p "$OUT_DIR/logs"

count_one() {
    # this function runs in child shells started by xargs, which do not inherit the options set above
    set -o pipefail
    local sample=$1 safe log status
    safe=$(printf '%s' "$sample" | sed -E 's/[^A-Za-z0-9._-]+/_/g')
    [ -f "$OUT_DIR/$safe.counts.json" ] && { echo "skip  $sample (already counted)"; return 0; }
    log="$OUT_DIR/logs/$safe.log"
    mapfile -t bams < <(awk -F'\t' -v s="$sample" 'NR > 1 && $1 == s {print $2}' "$MANIFEST")
    [ "${#bams[@]}" -gt 0 ] || { echo "FAIL  $sample (no BAM in the manifest)"; return 1; }
    {
        echo "sample $sample: ${#bams[@]} BAM file(s); $(date)"
        (
            status=0
            for bam in "${bams[@]}"; do
                if [[ $bam == s3://* ]]; then
                    $FETCH "$bam" - | "$SAMTOOLS" view -@ 2 -F 0x904 -q "$MIN_MAPQ" - || status=1
                else
                    "$SAMTOOLS" view -@ 2 -F 0x904 -q "$MIN_MAPQ" "$bam" || status=1
                fi
            done
            exit $status
        ) | cut -f 2-6 | "$PYTHON" "$REPO/scripts/karyotype.py" count --sample "$sample" --build "$BUILD" \
                --bin_size "$BIN_SIZE" --min_mapq "$MIN_MAPQ" --out_dir "$OUT_DIR" --source "${bams[@]}"
    } > "$log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        rm -f "$OUT_DIR/$safe.counts.json" "$OUT_DIR/$safe.counts.tsv.gz"
        echo "FAIL  $sample (see $log)"
        return 1
    fi
    echo "done  $sample: $(tail -1 "$log" | cut -c1-150)"
}
export -f count_one

awk -F'\t' 'NR > 1 && $1 != "" {print $1}' "$MANIFEST" | awk '!seen[$0]++' > "$OUT_DIR/.samples"
n=$(wc -l < "$OUT_DIR/.samples")
echo "$n samples in $MANIFEST; build $BUILD, bins of $BIN_SIZE bases, mapping quality >= $MIN_MAPQ; $PARALLEL at a time"
tr '\n' '\0' < "$OUT_DIR/.samples" | xargs -0 -n 1 -P "$PARALLEL" bash -c 'count_one "$1"' _
n_done=$(ls "$OUT_DIR"/*.counts.json 2>/dev/null | wc -l)
echo "Finished $(date): $n_done of $n samples counted in $OUT_DIR"
