#!/bin/bash
# Runs every test one after the other and prints a summary. Each test works on synthetic data in a temporary
# folder, on the CPU. The counting test is skipped without samtools, the pipeline test without samtools and modkit
# (and its alignment steps without minimap2 or dorado), the tuning test without Optuna, and the region maps of the
# figures test without R and DMRcate (RSCRIPT, see envs/figures_r.yml). Run from anywhere:
#   bash tests/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

tests=(test_bedmethyl test_training test_options test_data_preparation test_tuning test_report
       test_karyotype_count test_karyotype_call test_pipeline test_figures)
passed=() partly=() skipped=() failed=()
start=$SECONDS
for t in "${tests[@]}"; do
    echo "=== $t"
    log=$(mktemp)
    bash "tests/$t.sh" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        failed+=("$t")
    elif grep -q "^SKIPPED" "$log"; then
        skipped+=("$t")
    elif grep -q "^skip " "$log"; then
        partly+=("$t")                                  # passed, with some steps skipped
    else
        passed+=("$t")
    fi
    rm -f "$log"
done

echo
echo "Passed: $(( ${#passed[@]} + ${#partly[@]} ))  Skipped: ${#skipped[@]}  Failed: ${#failed[@]}  ($(( (SECONDS - start) / 60 )) min)"
[ ${#partly[@]} -eq 0 ] || echo "Passed with some steps skipped (see their output): ${partly[*]}"
[ ${#skipped[@]} -eq 0 ] || echo "Skipped: ${skipped[*]}"
[ ${#failed[@]} -eq 0 ] || { echo "Failed: ${failed[*]}"; exit 1; }
echo "ALL TESTS PASSED"
