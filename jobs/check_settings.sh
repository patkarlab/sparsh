#!/bin/bash
# Check jobs/settings.sh on the login node before submitting jobs. From the repository folder:
#   bash jobs/check_settings.sh            everything
#   bash jobs/check_settings.sh training   only what the training jobs need
#   bash jobs/check_settings.sh pipeline   only what jobs/run_sample.pbs and jobs/predict.pbs need
# Only reads files, apart from creating RUNS_DIR if it does not exist yet.
cd "$(dirname "$0")/.." || exit 1
part=${1:-all}
case $part in
    all | training | pipeline) ;;
    *) echo "usage: bash jobs/check_settings.sh [training | pipeline]"; exit 2 ;;
esac
source "${SETTINGS_FILE:-jobs/settings.sh}"
problems=0
pipeline_required=no
[ "$part" = pipeline ] && pipeline_required=yes

report_file() {  # usage: report_file NAME PATH REQUIRED(yes|no)
    if [ -z "$2" ]; then
        if [ "$3" = yes ]; then
            echo "PROBLEM  $1 is empty"; problems=$((problems + 1))
        else
            echo "ok       $1 not set (optional)"
        fi
    elif [ -f "$2" ]; then
        echo "ok       $1 = $2 ($(du -h "$2" | cut -f1))"
    else
        echo "PROBLEM  $1 = $2 does not exist"; problems=$((problems + 1))
    fi
}

if [ "$part" != pipeline ]; then
    report_file DATA_PATH "$DATA_PATH" yes
    report_file EXCLUDE_IDS "$EXCLUDE_IDS" no
    if [ -n "$EXCLUDE_IDS" ] && [ -f "$EXCLUDE_IDS" ]; then
        echo "         $(grep -cv '^[[:space:]]*$' "$EXCLUDE_IDS") non-empty lines; the first three:"
        head -n 3 "$EXCLUDE_IDS" | sed 's/^/           /'
    fi
    report_file GROUPS_FILE "$GROUPS_FILE" no
fi

if [ -d "$CONDA_BASE/envs/$CONDA_ENV" ]; then
    echo "ok       conda environment $CONDA_ENV"
else
    echo "PROBLEM  conda environment $CONDA_ENV not found in $CONDA_BASE/envs (see README.md)"
    problems=$((problems + 1))
fi

if mkdir -p "$RUNS_DIR" 2>/dev/null && [ -w "$RUNS_DIR" ]; then
    echo "ok       RUNS_DIR = $RUNS_DIR ($(df -Ph "$RUNS_DIR" | awk 'NR == 2 {print $4}') free on that file system)"
    if command -v lfs >/dev/null 2>&1; then
        mount_point=$(df -P "$RUNS_DIR" | awk 'NR == 2 {print $6}')
        echo "         Your quota there (limit 0 or - means no limit):"
        lfs quota -h -u "${USER:-$(id -un)}" "$mount_point" 2>/dev/null | sed 's/^/           /'
    fi
else
    echo "PROBLEM  RUNS_DIR = $RUNS_DIR is not writable"; problems=$((problems + 1))
fi
[ "$part" = pipeline ] || echo "         Classes left out by scripts/check_data.py: ${EXCLUDE_PREFIXES:-none}"

# the single-sample pipeline (jobs/run_sample.pbs); not needed for training
if [ "$part" != training ]; then
    echo "Single-sample pipeline:"
    report_file REFERENCE_FASTA "$REFERENCE_FASTA" "$pipeline_required"
    [ -z "$REFERENCE_FASTA" ] || [ -f "$REFERENCE_FASTA.fai" ] || {
        echo "PROBLEM  $REFERENCE_FASTA.fai does not exist: index the reference with samtools faidx"
        problems=$((problems + 1)); }
    report_file EPIC_MANIFEST "$EPIC_MANIFEST" "$pipeline_required"
    report_file "PIPELINE_MODEL (config.json)" "${PIPELINE_MODEL:+$PIPELINE_MODEL/config.json}" "$pipeline_required"
    report_file CN_PANEL "$CN_PANEL" no
    for t in SAMTOOLS MODKIT DORADO MINIMAP2; do
        value=${!t:-$(echo "$t" | tr '[:upper:]' '[:lower:]')}
        path=$(command -v "$value" 2>/dev/null)
        if [ -n "$path" ]; then
            echo "ok       $t = $path ($("$path" --version 2>&1 | head -1))"
        else
            case $t in
                DORADO) hint="; needed only for reads not yet aligned" ;;
                MINIMAP2) hint="; needed only with ALIGNER=minimap2" ;;
                *) hint="" ;;
            esac
            echo "note     $t = $value not found$hint"
        fi
    done
fi

if [ "$problems" -eq 0 ]; then
    echo "SETTINGS OK"
else
    echo "$problems problem(s): fix them in ${SETTINGS_FILE:-jobs/settings.sh} and run this check again"
    exit 1
fi
