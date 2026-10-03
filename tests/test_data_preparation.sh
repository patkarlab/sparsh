#!/bin/bash
# Preparing training data on synthetic arrays: a relabeled pickle with a class renamed
# (scripts/make_training_pickle.py) and its refusals, merged cross-validation groups (scripts/merge_groups.py),
# training on the new pickle, and the label audit (scripts/label_audit.py) with and without a run. CPU only, about
# a minute, writes only to a temporary folder that is deleted afterwards. Run from the repository folder:
#   bash tests/test_data_preparation.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
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

step make_data    python tests/make_synthetic.py --output_dir "$TMP/data"
step lists        python - "$TMP" <<'PY'
import sys
import pandas as pd
tmp = sys.argv[1]
df = pd.read_pickle(f"{tmp}/data/train.pkl")
ids = (df["Sample_ID"] if "Sample_ID" in df.columns else pd.Series(df.index)).astype(str).to_numpy()
lab = df["ANNOTATION"].astype(str).to_numpy()
kmt = ids[lab == "AML_KMT2A-r"][:5]
pd.DataFrame({"Sample_ID": kmt, "new_label": "AML_KMT2A_test", "reason": "test"}).to_csv(f"{tmp}/relabel.csv", index=False)
pd.DataFrame({"Sample_ID": [kmt[0]], "new_label": "OTHER"}).to_csv(f"{tmp}/relabel_clash.csv", index=False)
pd.DataFrame({"Sample_ID": ["X_missing"], "new_label": "Y"}).to_csv(f"{tmp}/relabel_unknown.csv", index=False)
pd.DataFrame({"Sample_ID": ids[:4], "group": ["d1", "d1", "d2", "d2"]}).to_csv(f"{tmp}/groups.csv", index=False)
pd.DataFrame({"Sample_ID": [ids[1], ids[2], ids[10], ids[11]], "patient": ["p1", "p1", "p2", "p2"]}).to_csv(
    f"{tmp}/patients.csv", index=False)
print("lists written")
PY
step pickle       python scripts/make_training_pickle.py --data_path "$TMP/data/train.pkl" --output "$TMP/new.pkl" \
                      --relabel "$TMP/relabel.csv" --rename "T-ALL=T-ALL_renamed"
step rename_bad   bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/z.pkl' \
                      --rename NO_SUCH_CLASS=X 2>&1 | grep -q 'no sample has the label'"
step pickle_again bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/new.pkl' \
                      2>&1 | grep -q 'exists'"
step clash        bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/x.pkl' \
                      --relabel '$TMP/relabel.csv' '$TMP/relabel_clash.csv' 2>&1 | grep -q 'different new labels'"
step unknown_id   bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/y.pkl' \
                      --relabel '$TMP/relabel_unknown.csv' 2>&1 | grep -q 'not in the pickle'"
step groups       python scripts/merge_groups.py --groups "$TMP/groups.csv" --patients "$TMP/patients.csv" \
                      --output "$TMP/groups_merged.csv"
step train        python scripts/train.py --data_path "$TMP/new.pkl" --output_dir "$TMP/run" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --groups_file "$TMP/groups_merged.csv" --min_samples 2
step audit        python scripts/label_audit.py --data_path "$TMP/new.pkl" --classes AML B-ALL_High \
                      --run "$TMP/run" --n_cpgs 2000 --n_pcs 10 --k 5 --out_dir "$TMP/audit"
step audit_norun  python scripts/label_audit.py --data_path "$TMP/data/train.pkl" --n_cpgs 1000 --n_pcs 5 --k 5 \
                      --out_dir "$TMP/audit_norun"
step contents     python - "$TMP" <<'PY'
import json
import os
import sys
import pandas as pd
tmp = sys.argv[1]
old = pd.read_pickle(f"{tmp}/data/train.pkl")
new = pd.read_pickle(f"{tmp}/new.pkl")
assert new.shape[0] == old.shape[0] and "ANNOTATION_ORIGINAL" in new.columns
assert (new["ANNOTATION_ORIGINAL"].astype(str).to_numpy() == old["ANNOTATION"].astype(str).to_numpy()).all()
changes = pd.read_csv(f"{tmp}/new.pkl.changes.csv", dtype=str)
relabel = pd.read_csv(f"{tmp}/relabel.csv", dtype=str)
assert set(changes.loc[changes["file"] != "--rename", "Sample_ID"]) == set(relabel["Sample_ID"])
assert (changes.loc[changes["file"] == "--rename", "new_label"] == "T-ALL_renamed").all()
ids = (new["Sample_ID"] if "Sample_ID" in new.columns else pd.Series(new.index, index=new.index)).astype(str)
label = dict(zip(ids, new["ANNOTATION"].astype(str)))
assert all(label[s] == "AML_KMT2A_test" for s in relabel["Sample_ID"])
labels = new["ANNOTATION"].astype(str)
assert (labels == "T-ALL_renamed").sum() > 0 and not (labels == "T-ALL").any()
groups = pd.read_csv(f"{tmp}/groups_merged.csv", dtype=str)
assert groups["group"].nunique() == 2 and len(groups) == 6, groups       # {0,1,2,3} joined through p1, and {10,11}
config = json.load(open(f"{tmp}/run/config.json"))
counts = config["data"]["class_counts"]
assert counts.get("AML_KMT2A_test") == 5 and "T-ALL_renamed" in counts, counts
assert config["data"]["groups_file"].endswith("groups_merged.csv")
audit = pd.read_csv(f"{tmp}/audit/audit.csv")
assert {"same_label_of_5", "neighbour_majority", "cv_own_prob_mean", "cv_own_prob_binary_0.90", "flag_lamprey",
        "flag_review"} <= set(audit.columns), audit.columns
assert audit["cv_own_prob_mean"].notna().sum() > 0 and audit["label"].str.startswith(("AML", "B-ALL_High")).all()
assert audit["same_label_of_5"].between(0, 5).all() and audit["same_label_fraction"].between(0, 1).all()
assert os.path.exists(f"{tmp}/audit_norun/flagged.csv")
print("contents ok")
PY
echo "DATA PREPARATION TEST PASSED"
