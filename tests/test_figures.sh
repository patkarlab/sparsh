#!/bin/bash
# Figures on synthetic arrays with known class signatures: t-SNE maps (with a class outside the model added to the
# supplementary map), class statistics, the signature heatmap, global methylation, and, when R with DMRcate is
# available (RSCRIPT, default Rscript), the regions and their maps. CPU only, about a minute. Run from the
# repository folder:
#   bash tests/test_figures.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PYTHON=${PYTHON:-python}
RSCRIPT=${RSCRIPT:-Rscript}

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_arrays "$PYTHON" tests/make_synthetic_arrays.py --out_dir "$TMP/data"
ARGS=(--model_dir "$TMP/data/run" --out_dir "$TMP/fig" --annotation "$TMP/data/annotation.tsv.gz")
step tsne    "$PYTHON" scripts/figures.py tsne "${ARGS[@]}" --extra_classes AML-MR --n_cpgs 500 --iterations 500
step markers "$PYTHON" scripts/figures.py markers "${ARGS[@]}" --extra_classes AML-MR --per_class 10
grep -q "Arrays from the cache" "$TMP/markers.log" || { echo "FAIL  cache: markers read the pickle again"; exit 1; }
DMR=no
if "$RSCRIPT" -e 'suppressPackageStartupMessages(library(DMRcate))' > /dev/null 2>&1; then
    step dmr     "$RSCRIPT" figures/dmr.R --input "$TMP/fig/regions_input" --out "$TMP/fig/regions"
    step regions "$PYTHON" scripts/figures.py regions "${ARGS[@]}" --extra_classes AML-MR
    DMR=yes
else
    echo "skip  dmr and regions (no R with DMRcate at '$RSCRIPT')"
fi
step contents "$PYTHON" - "$TMP" "$DMR" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

tmp, dmr = Path(sys.argv[1]), sys.argv[2] == "yes"
fig = tmp / "fig"
truth = pd.read_csv(tmp / "data" / "truth.csv")
ann = pd.read_csv(tmp / "data" / "annotation.tsv.gz", sep="\t").set_index("probe")

for stem in ("tsne", "tsne_with_excluded_classes"):
    for kind in ("classes", "classes_labelled", "lineages"):
        for ext in ("png", "pdf", "svg"):
            assert (fig / f"{stem}_{kind}.{ext}").exists(), (stem, kind, ext)
main = pd.read_csv(fig / "tsne_coordinates.csv")
supp = pd.read_csv(fig / "tsne_with_excluded_classes_coordinates.csv")
folds = pd.read_csv(tmp / "data" / "run" / "folds.csv")
assert sorted(main["sample_id"]) == sorted(folds["sample_id"]) and main["in_model"].all()
assert set(supp["class"]) - set(main["class"]) == {"AML-MR"} and (supp["class"] == "AML-MR").sum() == 11, \
    "the excluded AML-MR array must stay out"
assert "B-ALL_High hyperdiploid" not in set(supp["class"]), "the label map was not applied"
used = (fig / "tsne_cpgs.txt").read_text().split()
assert len(used) == 500 and not ann.loc[used, "chrom"].isin(["chrX", "chrY"]).any(), "sex chromosome CpGs used"
settings = json.loads((fig / "tsne_settings.json").read_text())
assert settings["arrays"] == len(folds) and settings["cpgs_used"] == 500

sig = pd.read_csv(fig / "signature_cpgs.csv")
assert set(sig["class"]) == set(folds["label"]), set(sig["class"])
for cls, rows in sig.groupby("class"):
    marked = truth[(truth["class"] == cls)]
    clusters = {int(p[2:]) // 5 for p in rows["cpg"]}
    hits = clusters & set(marked["cluster"])
    assert len(hits) >= 0.8 * len(clusters), (cls, len(hits), len(clusters))
    for row in rows.itertuples():
        want = marked.loc[marked["cluster"] == int(row.cpg[2:]) // 5, "direction"]
        assert want.empty or want.iloc[0] == row.direction, (cls, row.cpg)
for name in ("signature_heatmap", "global_methylation"):
    for ext in ("png", "pdf", "svg"):
        assert (fig / f"{name}.{ext}").exists(), name
listing = pd.read_csv(fig / "regions_input" / "classes.tsv", sep="\t")
assert len(listing) == folds["label"].nunique()
one = pd.read_csv(fig / "regions_input" / listing["file"].iloc[0], sep="\t")
assert one["chrom"].str.match(r"^chr\d+$").all(), "only autosomes go to the region analysis"

if dmr:
    regions = pd.read_csv(fig / "dmrs_all.tsv", sep="\t")
    kept = regions[regions["kept"]]
    for cls in folds["label"].unique():
        mine = kept[kept["class"] == cls]
        marked = truth[truth["class"] == cls]
        found = 0
        for t in marked.itertuples():
            hit = mine[(mine["chrom"] == t.chrom) & (mine["start"] <= t.end) & (mine["end"] >= t.start)]
            if len(hit):
                found += 1
                assert (hit["direction"] == t.direction).all(), (cls, t.cluster)
        assert found >= 0.8 * len(marked), (cls, found, len(marked))
    for name in ("dmr_counts", "dmr_map", "dmr_loci"):
        assert (fig / f"{name}.png").exists(), name
    assert (fig / "regions" / "sessionInfo.txt").exists()
print("contents ok")
PY
echo "FIGURES TEST PASSED"
