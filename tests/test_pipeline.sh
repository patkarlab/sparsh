#!/bin/bash
# The single-sample pipeline (scripts/run_sample.py) on synthetic reads with modified-base calls: one sorted file,
# the same reads split into two unsorted files as the hourly files of a run, and the reads unaligned, aligned again
# with minimap2 and with dorado when they are installed; each must give the same matrix, and a finished run must not
# be redone. Needs samtools and modkit (set SAMTOOLS, MODKIT, MINIMAP2 and DORADO if they are not on PATH); skipped
# without them. Trains a tiny model first. CPU only, about two minutes. Run from the repository folder:
#   bash tests/test_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PYTHON=$(command -v "${PYTHON:-python}")             # a full path: one step runs with a reduced PATH
SAMTOOLS=${SAMTOOLS:-$(command -v samtools || echo samtools)}
MODKIT=${MODKIT:-$(command -v modkit || echo modkit)}
MINIMAP2=${MINIMAP2:-$(command -v minimap2 || true)}
DORADO=${DORADO:-$(command -v dorado || true)}
[ -x "$SAMTOOLS" ] || { echo "SKIPPED: samtools not found (set SAMTOOLS=/path/to/samtools)"; exit 0; }
[ -x "$MODKIT" ] || { echo "SKIPPED: modkit not found (set MODKIT=/path/to/modkit)"; exit 0; }

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_data "$PYTHON" tests/make_synthetic.py --output_dir "$TMP/data"
step train     "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/model" \
                   --exclude_prefixes MPAL --n_folds 2 --epochs 2 --hidden_dims 32 16 --device cpu \
                   --train_sim binary --val_sim binary --eval_sims binary --eval_coverages 0.1 0.3
step make_reads "$PYTHON" tests/make_synthetic_modbam.py --out_dir "$TMP/in"
step bam bash -c "'$SAMTOOLS' faidx '$TMP/in/ref.fa' && '$SAMTOOLS' view -b -o '$TMP/in/reads.bam' '$TMP/in/reads.sam' \
    && '$SAMTOOLS' index '$TMP/in/reads.bam' && mkdir -p '$TMP/in/hourly' \
    && '$SAMTOOLS' view -h '$TMP/in/reads.sam' | awk '/^@/ || NR % 2' | sed 's/SO:coordinate/SO:unknown/' \
       | '$SAMTOOLS' view -b -o '$TMP/in/hourly/part_0.bam' - \
    && '$SAMTOOLS' view -h '$TMP/in/reads.sam' | awk '/^@/ || !(NR % 2)' | sed 's/SO:coordinate/SO:unknown/' \
       | '$SAMTOOLS' view -b -o '$TMP/in/hourly/part_1.bam' - \
    && '$SAMTOOLS' reset -o '$TMP/in/unaligned.bam' '$TMP/in/reads.bam' \
    && cp '$TMP/in/hourly/part_0.bam' '$TMP/in/hourly/run_fail_barcode01_0.bam'"

run() {  # run <name> <aligner> <bam> [more options]
    local name=$1 aligner=$2 bam=$3; shift 3
    "$PYTHON" scripts/run_sample.py --sample "$name" --bam "$bam" --out_dir "$TMP/out/$name" \
        --reference "$TMP/in/ref.fa" --manifest "$TMP/in/manifest.csv" --model_dir "$TMP/model" --no_panel \
        --threads 2 --samtools "$SAMTOOLS" --modkit "$MODKIT" --aligner "$aligner" "$@"
}
step sorted run S_SORTED dorado "$TMP/in/reads.bam" --sex F --blast_percent 70
step hourly run S_HOURLY dorado "$TMP/in/hourly"
if [ -n "$MINIMAP2" ]; then
    step minimap2 run S_MINIMAP2 minimap2 "$TMP/in/unaligned.bam" --minimap2 "$MINIMAP2"
else
    echo "skip  minimap2 (not found; set MINIMAP2)"
fi
if [ -n "$DORADO" ]; then
    step dorado run S_DORADO dorado "$TMP/in/unaligned.bam" --dorado "$DORADO"
else
    echo "skip  dorado (not found; set DORADO)"
fi
step refuse_unaligned_without_aligner bash -c "! env -u DORADO PATH=/usr/bin:/bin '$PYTHON' scripts/run_sample.py \
    --sample S_NONE --bam '$TMP/in/unaligned.bam' --out_dir '$TMP/out/S_NONE' --reference '$TMP/in/ref.fa' \
    --manifest '$TMP/in/manifest.csv' --model_dir '$TMP/model' --no_panel --samtools '$SAMTOOLS' --modkit '$MODKIT' \
    > '$TMP/none.log' 2>&1 && grep -q 'dorado was not found' '$TMP/none.log'"
run S_SORTED dorado "$TMP/in/reads.bam" --sex F --blast_percent 70 > "$TMP/again.log" 2>&1
step not_redone test "$(grep -c 'done before' "$TMP/again.log")" -eq 5
run S_SORTED dorado "$TMP/in/reads.bam" --sex F --blast_percent 70 --redo report > "$TMP/redo.log" 2>&1
step redo_report test "$(grep -c 'done before' "$TMP/redo.log")" -eq 4
step contents "$PYTHON" - "$TMP" <<'PY'
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1]) / "out"

def row(name):
    lines = (tmp / name / "methylation" / f"{name}.csv").read_text().splitlines()
    return lines[0], lines[1].split(",", 1)[1]

header, values = row("S_SORTED")
assert header.startswith(",cg") and len(header.split(",")) > 500, header[:80]
assert set(values.split(",")) == {"0", "1"}
assert row("S_HOURLY") == (header, values), "the hourly files must give the matrix of the merged file"
for other in ("S_MINIMAP2", "S_DORADO"):
    if (tmp / other).exists():
        assert row(other) == (header, values), f"{other}: realigned reads must give the same matrix"
info = json.loads((tmp / "S_SORTED" / "run_info.json").read_text())
assert all(info["steps"][s]["status"] == "done" for s in ("bam", "methylation", "prediction", "copy_number", "report"))
assert info["steps"]["bam"]["action"] == "linked the sorted input", info["steps"]["bam"]
hourly = json.loads((tmp / "S_HOURLY" / "run_info.json").read_text())
assert hourly["steps"]["bam"]["action"] == "merged and sorted" and hourly["steps"]["bam"]["inputs"] == 2, hourly
assert not any("_fail_" in f for f in hourly["inputs"]), "files of failed reads must be left out"
assert (tmp / "S_SORTED" / "S_SORTED.report.html").exists()
assert (tmp / "S_SORTED" / "methylation" / "S_SORTED_pileup.bed.gz").exists()
pred = (tmp / "S_SORTED" / "prediction" / "predictions.csv").read_text().splitlines()
assert len(pred) == 2 and pred[1].startswith("S_SORTED,"), pred[:2]
assert "sex" in (tmp / "S_SORTED" / "sample_sheet.csv").read_text()
print("contents ok")
PY
echo "PIPELINE TEST PASSED"
