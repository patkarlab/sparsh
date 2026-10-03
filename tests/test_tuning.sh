#!/bin/bash
# Hyperparameter study (scripts/tune.py) on synthetic data: CPU only, a few minutes, writes only to a temporary
# folder that is deleted afterwards. Needs Optuna (pip install optuna); skipped without it. Run from the
# repository folder:
#   bash tests/test_tuning.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PYTHON=${PYTHON:-python}
"$PYTHON" -c "import optuna" 2>/dev/null || { echo "SKIPPED: Optuna is not installed (pip install optuna)"; exit 0; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_data "$PYTHON" tests/make_synthetic.py --output_dir "$TMP/data"
# Base run: the recipe in miniature, scored at the objective's conditions. Its settings lie inside the
# search space, so the study's first trial repeats it. T-ALL_TAL1-like stands in for normal marrow as the dilution
# partner.
step base_run  "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/base" \
                   --exclude_prefixes MPAL --n_folds 2 --epochs 5 --hidden_dims 512 256 --device cpu \
                   --train_sim binary --val_sim binary --eval_sims binary --cov_min 0.02 --cov_max 0.95 \
                   --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                   --eval_coverages 0.1 0.3 0.5 --dilution_prob 0.5 --blast_min 0.2 --val_dilution \
                   --eval_blasts 1 0.5 0.3 --normal_class T-ALL_TAL1-like --no_fold_models
step study_a   "$PYTHON" scripts/tune.py --base_run "$TMP/base" --study_dir "$TMP/study" --max_trials 3 --device cpu \
                   --startup_trials 1 --warmup_epochs 1 --sampler_seed 1
step study_b   "$PYTHON" scripts/tune.py --base_run "$TMP/base" --study_dir "$TMP/study" --max_trials 4 --device cpu \
                   --startup_trials 1 --warmup_epochs 1 --sampler_seed 2
step study_full "$PYTHON" scripts/tune.py --base_run "$TMP/base" --study_dir "$TMP/study" --max_trials 4 --device cpu
grep -q "already has 4 finished or pruned trials" "$TMP/study_full.log" || { echo "FAIL  a full study started a trial"; exit 1; }
cp -r "$TMP/base" "$TMP/base_copy"
step other_base bash -c "'$PYTHON' scripts/tune.py --base_run '$TMP/base_copy' --study_dir '$TMP/study' --device cpu \
                   2>&1 | grep -q 'does not match this job'"
cp -r "$TMP/base" "$TMP/base_changed"
"$PYTHON" - "$TMP/base_changed/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
k = sorted(c["data"]["class_counts"])[0]
c["data"]["class_counts"][k] += 1
json.dump(c, open(sys.argv[1], "w"))
PY
step changed_data bash -c "'$PYTHON' scripts/tune.py --base_run '$TMP/base_changed' --study_dir '$TMP/study2' --device cpu \
                   2>&1 | grep -q 'no longer match'"
step report    "$PYTHON" scripts/tune.py --study_dir "$TMP/study" --report
mkdir -p "$TMP/runs"
cp -r "$TMP/base" "$TMP/runs/grp_s1"; cp -r "$TMP/base" "$TMP/runs/grp_s2"
step score     "$PYTHON" scripts/tune.py --score_runs "$TMP/base" "$TMP/runs/grp_s1" "$TMP/runs/grp_s2"
grep -q "Mean over the runs of each group" "$TMP/score.log" || { echo "FAIL  no group mean"; exit 1; }
step contents  "$PYTHON" - "$TMP" <<'PY'
import json
import sys
from pathlib import Path
import optuna
import pandas as pd
sys.path.insert(0, "scripts")
from tune import STUDY_NAME, journal_storage, score_run
tmp = Path(sys.argv[1])
study = optuna.load_study(study_name=STUDY_NAME, storage=journal_storage(optuna, tmp / "study" / "study.journal"))
trials = study.get_trials(deepcopy=False)
states = [t.state.name for t in trials]
assert len(trials) == 4 and all(s in ("COMPLETE", "PRUNED") for s in states), states
first = trials[0]
assert first.user_attrs.get("baseline") and first.state.name == "COMPLETE", first
assert first.params["hidden_dims"] == "512-256" and abs(first.params["learning_rate"] - 1e-4) < 1e-12, first.params
settings = json.load(open(tmp / "study" / "trials" / "trial_000" / "settings.json"))
assert settings["baseline"] and settings["model"]["hidden_dims"] == [512, 256], settings
assert settings["training"]["eval_blasts"] == [1.0, 0.5, 0.3] and settings["training"]["dilution_prob"] == 0.5
# the baseline trial repeats the base run exactly: same objective from the trial's logits and the run's files
base = score_run(tmp / "base")
assert abs(base["objective"] - first.value) < 1e-9, (base["objective"], first.value)
m = pd.read_csv(tmp / "study" / "trials" / "trial_000" / "metrics.csv")
assert {"binary_0.30", "binary-blast30_0.30", "binary-blast50_0.10"} <= set(m["condition"]), m["condition"].tolist()
t = pd.read_csv(tmp / "study" / "trials.csv")
assert len(t) == 4 and t["value"].notna().any()
space = json.load(open(tmp / "study" / "space.json"))
assert space["objective"]["coverages"] == [0.1, 0.3, 0.5] and "hidden_dims" in space["space"]
print("tune contents ok")
PY
step no_callback_change "$PYTHON" - <<'PY'
# The epoch callback observes training without changing it: two identical short runs, one with a callback.
import numpy as np
from training.trainer import TrainConfig, train_one_model
rng = np.random.default_rng(0)
X = rng.random((60, 50)).astype(np.float32); y = np.repeat(np.arange(3), 20)
inner = [rng.random((12, 50)).astype(np.float32)]; y_inner = np.repeat(np.arange(3), 4)
cfg = TrainConfig(epochs=3, device="cpu", train_sim="binary", val_sim="binary", cov_max=0.9, calibrate=False)
mc = {"input_dim": 50, "hidden_dims": [16], "n_classes": 3, "dropout": 0.1, "activation": "gelu",
      "input_encoding": "scaled"}
seen = []
a = train_one_model(X, y, inner, y_inner, 3, mc, cfg, seed=5, tag="fold1")
b = train_one_model(X, y, inner, y_inner, 3, mc, cfg, seed=5, tag="fold1",
                    epoch_callback=lambda tag, e, nll, acc: seen.append((tag, e)))
assert seen == [("fold1", 1), ("fold1", 2), ("fold1", 3)], seen
for k, v in a["model"].state_dict().items():
    assert np.array_equal(v.numpy(), b["model"].state_dict()[k].numpy()), k
print("callback ok")
PY
echo "TUNING TEST PASSED"
