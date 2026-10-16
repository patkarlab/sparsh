#!/bin/bash
# Training options and reporting on synthetic data: per-read call errors, a background class reported as
# "<lineage>, no specific subtype", the family and lineage fallback, classes left out of training, and dilution
# by normal marrow. CPU only, about a minute, writes only to a temporary folder that is deleted afterwards.
# Run from the repository folder:
#   bash tests/test_options.sh
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
step train_wide   "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_wide" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9
step ont_coverage "$PYTHON" scripts/ont_coverage.py --run "$TMP/run_wide" "$TMP/data/ont" --output "$TMP/ont_coverage.csv"
# per-read call errors, a background class (AML_other), the family fallback and classes left out
cat > "$TMP/label_map_other.json" <<'EOF'
{"merge": {"B-ALL_PAX5 P80R": "B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like", "AML_KMT2A-r": "AML_other"}}
EOF
cat > "$TMP/hierarchy.json" <<'EOF'
{"lineage_by_prefix": [["AML", "AML"], ["B-ALL", "B-ALL"], ["T-ALL", "T-ALL"]],
 "families": {"B-ALL, test family": ["B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like", "B-ALL_ETV6-RUNX1"],
              "family absent from this model": ["X1", "X2"]},
 "no_subtype_classes": ["AML_other"]}
EOF
step train_err    "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_err" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --call_error_max 0.2 --val_call_error 0.1 --eval_call_errors 0 0.1 \
                      --label_map "$TMP/label_map_other.json" --exclude_classes B-ALL_TCF3-PBX1
step compare_err  "$PYTHON" scripts/compare_runs.py "$TMP/run_wide" "$TMP/run_err" \
                      --ont_coverage "$TMP/ont_coverage.csv" --output "$TMP/comparison_err.csv"
step hierarchy    "$PYTHON" scripts/hierarchy_report.py "$TMP/run_err" "$TMP/run_wide" --hierarchy "$TMP/hierarchy.json" \
                      --ont_coverage "$TMP/ont_coverage.csv"
step excluded     "$PYTHON" scripts/score_excluded.py --run "$TMP/run_err" --hierarchy "$TMP/hierarchy.json" \
                      --coverages 0.1 0.5
step predict_h    "$PYTHON" scripts/predict.py --model_dir "$TMP/run_err" --ont_dir "$TMP/data/ont" \
                      --ground_truth "$TMP/data/ground_truth.csv" --output_dir "$TMP/pred_err" --device cpu \
                      --hierarchy "$TMP/hierarchy.json"
step call_errors  "$PYTHON" - <<'PY'
import numpy as np
import torch
from models.corruption import corrupt_numpy, corrupt_rows, corrupt_torch
X = np.random.default_rng(0).random((3, 20000)).astype(np.float32)
a = corrupt_rows(X, ["a", "b", "c"], 0.3, "binary", 12345, salt=2)
b = corrupt_rows(X, ["a", "b", "c"], 0.3, "binary", 12345, salt=2, call_error=0.0)
assert np.array_equal(a, b, equal_nan=True), "a zero call error must leave the inputs without errors unchanged"
ones, zeros = np.ones(300000, np.float32), np.zeros(300000, np.float32)
for sim in ("reads", "binary", "oneread"):
    hi = np.nanmean(corrupt_numpy(ones, 0.3, sim, np.random.default_rng(1), 0.1))
    lo = np.nanmean(corrupt_numpy(zeros, 0.3, sim, np.random.default_rng(2), 0.1))
    assert 0.88 < hi < 0.92 and 0.08 < lo < 0.12, (sim, hi, lo)
t = corrupt_torch(torch.ones(2, 300000), torch.tensor([[0.5], [0.5]]), "oneread", torch.tensor([[0.0], [0.2]]))
m = [float(torch.nanmean(r)) for r in t]
assert m[0] == 1.0 and 0.78 < m[1] < 0.82, m
try:
    corrupt_numpy(ones, 0.3, "mask", np.random.default_rng(1), 0.1)
    raise SystemExit("call errors on the mask simulation were not refused")
except ValueError:
    pass
print("call errors ok")
PY
step hierarchy_none "$PYTHON" scripts/hierarchy_report.py "$TMP/run_wide" --hierarchy none
step hierarchy_unit "$PYTHON" - <<'PY'
import json
import tempfile
import numpy as np
from evaluation.hierarchy import hierarchical_calls, load_hierarchy, score_calls
cfg = {"lineage_by_prefix": [["T-ALL", "T-ALL"], ["AML", "AML"]],
       "families": {"T-ALL, not TAL1": ["T-ALL_TLX1", "T-ALL_TLX3", "T-ALL_NKX2"]},
       "no_subtype_classes": ["AML_other"]}
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, f)
f.close()
classes = ["AML_KMT2A-r", "AML_other", "T-ALL_TAL1", "T-ALL_TLX1", "T-ALL_TLX3"]   # no T-ALL_NKX2 in this model
h = load_hierarchy(f.name, classes)
probs = np.array([[0.00, 0.00, 0.05, 0.50, 0.45],     # family call
                  [0.02, 0.95, 0.01, 0.01, 0.01],     # background class: AML, no specific subtype
                  [0.50, 0.45, 0.02, 0.02, 0.01],     # lineage call: AML
                  [0.30, 0.10, 0.30, 0.20, 0.10],     # nothing reaches 0.90
                  [0.96, 0.01, 0.01, 0.01, 0.01]])    # subtype
calls = hierarchical_calls(probs, classes, h, 0.90)
assert list(calls["reported_level"]) == ["family", "no_subtype", "lineage", "none", "subtype"], calls
truth = ["T-ALL_NKX2", "AML_other", "AML NOS", "T-ALL_TAL1", "AML_KMT2A-r"]
pred = [classes[i] for i in probs.argmax(1)]
s = score_calls(truth, pred, calls, h)
assert list(s["reported_correct"].fillna(-1)) == [1.0, 1.0, 1.0, -1.0, 1.0], s
# provisional subtypes: a top class from the caution threshold up to the threshold, when no family reaches the
# threshold; a family above the threshold comes first, a provisional subtype before a lineage
from evaluation.hierarchy import check_thresholds, resolve_caution
probs2 = np.array([[0.00, 0.00, 0.05, 0.85, 0.10],     # family 0.95 reaches 0.90: family, not provisional
                   [0.85, 0.10, 0.02, 0.02, 0.01],     # provisional AML_KMT2A-r, although AML sums to 0.95
                   [0.02, 0.85, 0.05, 0.04, 0.04],     # background class: never provisional; lineage AML is 0.87
                   [0.05, 0.02, 0.82, 0.06, 0.05],     # provisional T-ALL_TAL1 (no family reaches 0.90)
                   [0.79, 0.12, 0.05, 0.02, 0.02],     # below 0.80: lineage AML (0.91)
                   [0.96, 0.01, 0.01, 0.01, 0.01]])    # subtype
on = hierarchical_calls(probs2, classes, h, 0.90, caution_threshold=0.80)
assert list(on["reported_level"]) == ["family", "provisional", "none", "provisional", "lineage", "subtype"], on
assert list(on["reported_call"])[1] == "AML_KMT2A-r" and list(on["reported_call"])[3] == "T-ALL_TAL1", on
off = hierarchical_calls(probs2, classes, h, 0.90, caution_threshold=0)
assert list(off["reported_level"]) == ["family", "lineage", "none", "lineage", "lineage", "subtype"], off
assert off.equals(hierarchical_calls(probs2, classes, h, 0.90)), "caution 0 must equal no caution threshold"
truth2 = ["T-ALL_TLX1", "AML_KMT2A-r", "AML_other", "T-ALL_TLX3", "AML_KMT2A-r", "AML_KMT2A-r"]
s2 = score_calls(truth2, [classes[i] for i in probs2.argmax(1)], on, h)
assert list(s2["reported_correct"].fillna(-1)) == [1.0, 1.0, -1.0, 0.0, 1.0, 1.0], s2
assert list(s2["lineage_correct"].fillna(-1))[3] == 1.0, s2
assert resolve_caution(0.90, None) == 0.80 and resolve_caution(0.80, None) is None and resolve_caution(0.90, 0) is None
for bad in (0.90, 0.95, -0.1):
    try:
        check_thresholds(0.90, bad)
        raise SystemExit(f"caution threshold {bad} was not refused")
    except ValueError:
        pass
print("hierarchy ok")
PY
step contents_err "$PYTHON" - "$TMP" <<'PY'
import json
import os
import sys
import numpy as np
import pandas as pd
tmp = sys.argv[1]
m = pd.read_csv(f"{tmp}/run_err/cv_metrics_by_condition.csv")
want = {"dense"} | {f"{s}{e}_{c}" for s in ("binary", "oneread") for e in ("", "-err10") for c in ("0.10", "0.50", "0.90")}
assert set(m["condition"]) == want, set(m["condition"]) ^ want
classes = set(json.load(open(f"{tmp}/run_err/class_mapping.json")).values())
assert "AML_other" in classes and "B-ALL_TCF3-PBX1" not in classes and "AML_KMT2A-r" not in classes, classes
t = json.load(open(f"{tmp}/run_err/config.json"))["training"]
assert t["call_error_max"] == 0.2 and t["val_call_error"] == 0.1 and t["eval_call_errors"] == [0.0, 0.1], t
c = pd.read_csv(f"{tmp}/comparison_err.csv", index_col=0)
for row in ("ONT samples, binary-err10_*", "mean over oneread-err10_*", "binary-err10_0.50", "binary_0.50"):
    assert row in c.index, f"row {row} missing from the comparison"
h = pd.read_csv(f"{tmp}/run_err/hierarchy_report/hierarchy_by_condition.csv", index_col=0)
levels = ["share_subtype", "share_no_subtype", "share_family", "share_provisional", "share_lineage", "share_none"]
assert np.allclose(h[levels].sum(axis=1), 1.0), h[levels]
assert (h["share_reported_any_level"] >= h["share_subtype"] - 1e-9).all()
assert all(str(i).startswith(("binary", "oneread")) for i in h.index), list(h.index)
bc = sorted(f for f in os.listdir(f"{tmp}/run_err/hierarchy_report") if f.startswith("by_class_"))
assert bc == ["by_class_binary_0.10.csv"], bc
e = pd.read_csv(f"{tmp}/run_err/excluded_check/summary.csv")
assert list(e["class"]) == ["B-ALL_TCF3-PBX1"] * 2 and (e["n"] == 7).all(), e
assert len(pd.read_csv(f"{tmp}/run_err/excluded_check/per_sample.csv")) == 14
p = pd.read_csv(f"{tmp}/pred_err/predictions.csv")
assert {"reported_call", "reported_level", "family", "lineage", "lineage_prob"} <= set(p.columns)
assert set(p["reported_level"]) <= {"subtype", "no_subtype", "family", "provisional", "lineage", "none"}
s = json.load(open(f"{tmp}/pred_err/summary_metrics.json"))
assert s["reported_call"]["all_with_truth"]["n"] == 8, s["reported_call"]
print("call-error and hierarchy contents ok")
PY
# dilution by normal marrow; the synthetic data have no normal-marrow class, so T-ALL_TAL1-like stands in as the
# partner
step train_dil    "$PYTHON" scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_dil" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --dilution_prob 0.5 --blast_min 0.2 --val_dilution \
                      --eval_blasts 1 0.5 --normal_class T-ALL_TAL1-like --final_model
step dil_refused  bash -c "'$PYTHON' scripts/train.py --data_path '$TMP/data/train.pkl' --output_dir '$TMP/run_refused' \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 1 --hidden_dims 8 --device cpu \
                      --eval_blasts 0.5 --normal_class NO_SUCH_CLASS 2>&1 | grep -q 'which this training set lacks'"
step compare_dil  "$PYTHON" scripts/compare_runs.py "$TMP/run_wide" "$TMP/run_dil" \
                      --ont_coverage "$TMP/ont_coverage.csv" --output "$TMP/comparison_dil.csv"
step hierarchy_dil "$PYTHON" scripts/hierarchy_report.py "$TMP/run_dil" --hierarchy none
step dilution_unit "$PYTHON" - <<'PY'
import numpy as np
import torch
from models.dilution import dilute_batch, dilute_fixed, dilute_random, mix, pick_partner
x = np.array([[0.0, 1.0, 0.5, np.nan], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
partners = np.array([[1.0, 0.0, np.nan, 0.2]], dtype=np.float32)
out = dilute_fixed(x, ["a", "b"], np.array([False, True]), partners, ["n1"], 0.3, 1)
assert np.allclose(out[0, :3], [0.7, 0.3, 0.5]) and np.isnan(out[0, 3]), out   # missing partner keeps x
assert np.array_equal(out[1], x[1]), "normal marrows must not be diluted"
assert np.array_equal(dilute_fixed(x, ["a", "b"], np.array([False, True]), partners, ["n1"], 0.3, 1), out,
                      equal_nan=True)
assert dilute_fixed(x, ["a", "b"], np.array([False, False]), partners, ["n1"], 1.0, 1) is x
# the partner depends only on the sample and on which normal marrows are in the pool
ids = [f"N{i}" for i in range(30)]
full = [pick_partner(f"S{k}", ids, "eval0.3", 12345) for k in range(50)]
half = ids[::2]
for k, j in enumerate(full):
    if ids[j] in half:
        assert half[pick_partner(f"S{k}", half, "eval0.3", 12345)] == ids[j]
v = dilute_random(np.ones((200, 5), np.float32), [str(i) for i in range(200)], np.zeros(200, bool),
                  np.zeros((3, 5), np.float32), ["n1", "n2", "n3"], 0.5, 0.2, 7)
diluted = v[:, 0] < 1.0
assert 0.35 < diluted.mean() < 0.65 and (v[diluted, 0] >= 0.2 - 1e-6).all(), diluted.mean()
state = torch.get_rng_state()
xb = torch.ones(4, 5)
assert dilute_batch(xb, torch.zeros(4, dtype=torch.long), torch.zeros(3, 5), torch.arange(3), 1, 0.0, 0.2) is xb
assert torch.equal(state, torch.get_rng_state()), "dilution switched off must not draw random numbers"
yb = torch.tensor([0, 0, 1, 1])
xs = torch.ones(4, 5)
torch.manual_seed(0)
o = dilute_batch(xs, yb, torch.zeros(3, 5), torch.arange(3), 1, 1.0, 0.2)
assert torch.equal(o[2:], xs[2:]) and (o[:2] < 1).all() and (o[:2] >= 0.2 - 1e-6).all(), o
assert np.isclose(float(mix(torch.tensor(1.0), torch.tensor(0.0), 0.25)), 0.25)
print("dilution ok")
PY
step contents_dil "$PYTHON" - "$TMP" <<'PY'
import json
import os
import sys
import pandas as pd
tmp = sys.argv[1]
m = pd.read_csv(f"{tmp}/run_dil/cv_metrics_by_condition.csv")
want = {"dense"} | {f"binary{b}_{c}" for b in ("", "-blast50") for c in ("0.10", "0.50", "0.90")}
assert set(m["condition"]) == want, set(m["condition"]) ^ want
assert set(m.loc[m["condition"] == "binary-blast50_0.50", "blast"]) == {0.5}
t = json.load(open(f"{tmp}/run_dil/config.json"))["training"]
assert t["dilution_prob"] == 0.5 and t["val_dilution"] and t["eval_blasts"] == [1.0, 0.5], t
assert os.path.exists(f"{tmp}/run_dil/model.pt")
c = pd.read_csv(f"{tmp}/comparison_dil.csv", index_col=0)
for row in ("ONT samples, binary-blast50_*", "binary-blast50_0.50", "binary_0.50"):
    assert row in c.index, f"row {row} missing from the comparison"
print("dilution contents ok")
PY
echo "OPTIONS TEST PASSED"
