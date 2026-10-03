#!/bin/bash
# Training and prediction end to end on synthetic data: data checks, training with the recipe file and with
# other options, prediction with shown class names, coverage and run comparison. CPU only, about two minutes,
# writes only to a temporary folder that is deleted afterwards. Run from the repository folder:
#   bash tests/test_training.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
PYTHON=${PYTHON:-python}
trap 'rm -rf "$TMP"' EXIT

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_data    "$PYTHON" tests/make_synthetic.py --output_dir "$TMP/data"
step check_data   "$PYTHON" scripts/check_data.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/checks" \
                      --exclude_prefixes MPAL
printf '# samples to leave out\nGSM100001\n\nGSM100002,replicate\n' > "$TMP/exclude.txt"
step exclusion    "$PYTHON" scripts/check_data.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/checks_excl" \
                      --exclude_prefixes MPAL --exclude_ids "$TMP/exclude.txt"
grep -q "Excluding 2 listed samples" "$TMP/exclusion.log" || { echo "FAIL  exclusion list not applied"; exit 1; }
step train_new    "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_new" \
                      --exclude_prefixes MPAL --groups_file "$TMP/checks/groups.csv" \
                      --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu --final_model
step train_legacy "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_legacy" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --imbalance legacy_upsample --train_sim mask --val_sim mask --coverage_mode schedule
step train_wide   "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_wide" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9
step train_wide2  "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_wide2" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --seed 43
# the recipe of the released model, read from its file, with the settings of this small test after it
step train_recipe "$PYTHON" scripts/train.py @configs/recipe_v0.4.0.args --data_path "$TMP/data/train.pkl" \
                      --output_dir "$TMP/run_recipe" --exclude_classes --normal_class T-ALL_TAL1-like --n_folds 2 --epochs 2 \
                      --hidden_dims 16 --device cpu --val_coverages 0.1 0.5 --eval_coverages 0.1 0.5 \
                      --eval_blasts 1 0.5
step predict      "$PYTHON" scripts/predict.py --model_dir "$TMP/run_new" --ont_dir "$TMP/data/ont" \
                      --ground_truth "$TMP/data/ground_truth.csv" --output_dir "$TMP/pred" --device cpu
# names shown for classes (evaluation/display.py); everything is computed with the internal names
printf '%s\n' '{"display": {"AML_KMT2A-r": "AML_KMT2A_shown", "NOT_A_CLASS": "ignored",' \
    '             "B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like": "B-ALL_PH_shown"}}' > "$TMP/names.json"
sed 's/,AML_KMT2A-r$/,AML_KMT2A_shown/' "$TMP/data/ground_truth.csv" > "$TMP/ground_truth_shown.csv"
step predict_names "$PYTHON" scripts/predict.py --model_dir "$TMP/run_new" --ont_dir "$TMP/data/ont" \
                      --ground_truth "$TMP/ground_truth_shown.csv" --output_dir "$TMP/pred_names" --device cpu \
                      --display_names "$TMP/names.json"
printf '%s\n' '{"display": {"AML_KMT2A-r": "T-ALL_TAL1-like"}}' > "$TMP/names_clash.json"
step names_refused bash -c "'$PYTHON' scripts/predict.py --model_dir '$TMP/run_new' --ont_dir '$TMP/data/ont' \
                      --output_dir '$TMP/pred_clash' --device cpu --display_names '$TMP/names_clash.json' 2>&1 \
                      | grep -q 'is the internal name of another'"
step ont_coverage "$PYTHON" scripts/ont_coverage.py --run "$TMP/run_new" "$TMP/data/ont" --output "$TMP/ont_coverage.csv"
step compare      "$PYTHON" scripts/compare_runs.py "$TMP/run_legacy" "$TMP/run_new" "$TMP/run_wide" \
                      --ont_coverage "$TMP/ont_coverage.csv" --output "$TMP/comparison.csv"
step simulation   "$PYTHON" - <<'PY'
import numpy as np
import torch
from models.corruption import corrupt_numpy, corrupt_torch
beta = np.random.default_rng(0).random(200000).astype(np.float32)
for sim in ("binary", "oneread"):
    for f in (0.05, 0.5, 0.9):
        out = corrupt_numpy(beta, f, sim, np.random.default_rng(1))
        obs = np.isfinite(out)
        assert abs(obs.mean() - f) < 0.01, (sim, f, obs.mean())
        assert set(np.unique(out[obs])) <= {0.0, 1.0}, (sim, f)
        t = corrupt_torch(torch.from_numpy(beta[None, :]), torch.tensor([[f]]), sim)
        tobs = ~torch.isnan(t)
        assert abs(float(tobs.float().mean()) - f) < 0.01, (sim, f)
        assert set(torch.unique(t[tobs]).tolist()) <= {0.0, 1.0}, (sim, f)
# a majority call is right more often than a single read where beta is far from 0.5
far = (beta < 0.2) | (beta > 0.8)
agree = {sim: np.nanmean(np.where(far, corrupt_numpy(beta, 0.9, sim, np.random.default_rng(2)) == (beta > 0.5), np.nan))
         for sim in ("binary", "oneread")}
assert agree["binary"] > agree["oneread"], agree
print("simulation ok", agree)
PY
step duplicates   "$PYTHON" - "$TMP" <<'PY'
import sys
import numpy as np
from data.ont import read_ont_csv
f = f"{sys.argv[1]}/repeated_probes.csv"
open(f, "w").write("sample,cg1,cg2,cg1,cg3,cg2\nS1,1,,0,0.5,1\n")
cpgs = ["cg1", "cg2", "cg3", "cg4"]
x, info = read_ont_csv(f, cpgs)                        # default: mean of the copies with a value
assert np.allclose(x[:3], [0.5, 1.0, 0.5]) and np.isnan(x[3]), x
assert (info["n_repeated_probes"], info["n_repeated_multi_observed"], info["n_repeated_disagree"]) == (2, 1, 1), info
assert (info["n_observed"], info["n_observed_first"], info["max_copies"]) == (3, 2, 2), info
x, _ = read_ont_csv(f, cpgs, duplicates="first")
assert x[0] == 1.0 and np.isnan(x[1]), x
try:
    read_ont_csv(f, cpgs, duplicates="error")
    raise SystemExit("the error rule did not refuse repeated probes")
except ValueError:
    pass
print("repeated probes ok")
PY

for f in run_new/cv_metrics_by_condition.csv run_new/fold_models/fold1.pt run_new/model.pt \
         run_new/config.json pred/predictions.csv pred/summary_metrics.json checks/groups.csv; do
    [ -s "$TMP/$f" ] || { echo "FAIL  expected output missing: $f"; exit 1; }
done
step contents "$PYTHON" - "$TMP" <<'PY'
import json, sys
import pandas as pd
tmp = sys.argv[1]
s = json.load(open(f"{tmp}/pred/summary_metrics.json"))
assert s["n_with_truth"] == 8, s["n_with_truth"]
assert s["n_out_of_scheme"] == 1, "the out-of-scheme sample must be reported, not dropped"
g = pd.read_csv(f"{tmp}/checks/groups.csv")
assert len(g) == 2, "the planted duplicate pair must be found"
m = pd.read_csv(f"{tmp}/run_new/cv_metrics_by_condition.csv")
assert {"dense", "reads_0.20", "mask_0.20"} <= set(m["condition"])
c = pd.read_csv(f"{tmp}/comparison.csv", index_col=0)
assert "callable_share_at_accuracy_0.98" in set(c["metric"]), "calibration-free table missing"
assert not any(str(i).startswith(("dense", "mask_")) for i in c.index), "only nanopore rows by default"
for row in ("mean over reads_*", "ONT samples, binary_*", "mean over oneread_*", "binary_0.90"):
    assert row in c.index, f"row {row} missing from the comparison"
w = pd.read_csv(f"{tmp}/run_wide/cv_metrics_by_condition.csv")
assert set(w["condition"]) == {"dense", "binary_0.10", "binary_0.50", "binary_0.90",
                               "oneread_0.10", "oneread_0.50", "oneread_0.90"}, set(w["condition"])
o = pd.read_csv(f"{tmp}/ont_coverage.csv")
assert len(o) == 8 and o["error"].isna().all(), o
assert o["coverage_pct"].between(10, 40).all() and (o["share_0_or_1"] > 0.8).all(), o
# shown names change only the names in the outputs
import os
names = json.load(open(f"{tmp}/names.json"))["display"]
default = json.load(open("configs/display_names.json"))["display"]      # the names shown without --display_names
back = {v: default.get(k, k) for k, v in names.items()}
a = pd.read_csv(f"{tmp}/pred/predictions.csv")
b = pd.read_csv(f"{tmp}/pred_names/predictions.csv")
assert "prob_AML_KMT2A_shown" in b.columns and "prob_AML_KMT2A-r" not in b.columns, list(b.columns)
b.columns = [f"prob_{back.get(c[5:], c[5:])}" if c.startswith("prob_") else c for c in b.columns]
for col in ("prediction", "top2_class", "top3_class", "reported_call"):
    assert not set(b[col].dropna()) & set(names), (col, set(b[col]))
    b[col] = b[col].map(lambda v: back.get(v, v))
pd.testing.assert_frame_equal(a, b)
sa = json.load(open(f"{tmp}/pred/summary_metrics.json"))
sb = json.load(open(f"{tmp}/pred_names/summary_metrics.json"))
assert sb["display_names"]["shown"] == {k: v for k, v in names.items() if k != "NOT_A_CLASS"}, sb["display_names"]
for key in ("n_with_truth", "n_out_of_scheme", "all_with_truth", "in_scheme", "reported_call"):
    assert sa[key] == sb[key], (key, sa[key], sb[key])   # a shown name in the truth file is read as its class
e = pd.read_csv(f"{tmp}/pred_names/evaluation_per_sample.csv")
assert "AML_KMT2A_shown" in set(e["true_label"]) and not set(e["true_label"]) & set(names), set(e["true_label"])
cm = pd.read_csv(f"{tmp}/pred_names/confusion_matrix.csv", index_col=0)
assert list(cm.index) == list(cm.columns) and not set(cm.index) & set(names), list(cm.index)
assert os.path.exists(f"{tmp}/pred_names/display_names_used.json") and os.path.exists(f"{tmp}/pred/display_names_used.json")
rc = json.load(open(f"{tmp}/run_recipe/config.json"))
r = rc["training"]
assert (r["train_sim"], r["dilution_prob"], r["cov_max"], r["seed"]) == ("binary", 0.5, 0.95, 42), r
assert rc["model"]["input_encoding"] == "scaled" and rc["data"]["exclude_prefixes"] == ["MPAL", "AML_NOS", "B-ALL_NOS"]
print("contents ok")
PY
echo "TRAINING TEST PASSED"
