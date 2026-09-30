#!/bin/bash
# Per-case reports (scripts/report.py, package report/) on synthetic data: trains a tiny model, predicts the
# synthetic nanopore samples, writes the reports with a sample sheet and copy-number results for two samples,
# and checks the pages (self-contained, complete, valid HTML), report_summary.csv, prediction_info.json
# and the refusals of bad input. CPU only, under two minutes, writes only to a temporary folder that is deleted
# afterwards. Run from the repository folder:
#   bash tests/test_report.sh
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

# a command that must fail with a message containing the given text
refused() {
    local name=$1 text=$2; shift 2
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "FAIL  $name: the command succeeded but should have been refused"; exit 1
    elif grep -q -- "$text" "$TMP/$name.log"; then
        echo "ok    $name"
    else
        echo "FAIL  $name: refused without the expected message ($text)"; tail -10 "$TMP/$name.log"; exit 1
    fi
}

step make_data python tests/make_synthetic.py --output_dir "$TMP/data"
# binary reads, as the deployed recipe uses: the report's cross-validation section reads the binary_* conditions
step train     python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run" \
                   --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                   --train_sim binary --val_sim binary --eval_sims binary --eval_coverages 0.05 0.1 0.2 0.3 0.5
step predict   python scripts/predict.py --model_dir "$TMP/run" --ont_dir "$TMP/data/ont" \
                   --output_dir "$TMP/pred" --device cpu

# Sample sheet (no names of patients) and a copy-number result for one sample, written by hand. The sheet uses
# another case and spacing for one name, as a sheet typed by hand would.
cat > "$TMP/sheet.csv" <<'EOF'
sample,centre,sample_type,blast_percent,collection_date
ont01 ,Centre A,bone marrow,82,2026-09-01
ONT02,Centre B,bone marrow,35,2026-09-03
ONT05,Centre A,bone marrow,,2026-09-04
EOF
mkdir -p "$TMP/karyotypes"
cat > "$TMP/karyotypes/ONT01.karyotype.json" <<'EOF'
{"sample": "ONT01", "iscn": "54,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21[cp]",
 "karyotype": "High hyperdiploid, 54 chromosomes", "modal_number": 54, "ploidy_class": "high hyperdiploid",
 "sex": "female",
 "events": [{"type": "gain", "chrom": "21", "start": 1, "end": 46709983, "bands": "q11.2-q22.3", "copy_number": 4,
             "cell_fraction": 0.85, "whole_chromosome": true},
            {"type": "gain", "chrom": "X", "start": 1, "end": 156040895, "bands": "p22.33-q28", "copy_number": 3,
             "cell_fraction": null, "whole_chromosome": true}],
 "flags": ["high hyperdiploid pattern"], "warnings": ["Test file written by hand"],
 "qc": {"bin_size": 500000, "median_reads_per_bin": 150.5, "detection_limit": "about 20% of cells for 5 Mb events"}}
EOF
# ONT02: a result in the form written by scripts/karyotype.py call
cat > "$TMP/karyotypes/ONT02.karyotype.json" <<'EOF'
{"sample": "ONT02", "iscn": "sseq[GRCh37] (7)x1,9p21.3(20000001_24000000)x0", "karyotype": "45,XY,-7",
 "modal_number": 45, "ploidy_class": "near-diploid", "sex": "XY", "sex_source": "inferred from reads on chromosome Y",
 "leukaemic_fraction": 0.62, "baseline_copies": 2,
 "events": [{"level": "whole", "type": "loss", "chrom": "7", "arm": null, "start": 0, "end": 159138663,
             "size_mb": 159.1, "bands": "7p22.3q36.3", "copy_number": 1, "normal_copies": 2, "cell_fraction": 0.62,
             "subclonal": false, "whole_chromosome": true, "genes": ["IKZF1", "CUX1", "EZH2"]},
            {"level": "segment", "type": "loss", "chrom": "9", "arm": "p", "start": 20000000, "end": 24000000,
             "size_mb": 4.0, "bands": "9p21.3", "copy_number": 0, "normal_copies": 2, "cell_fraction": 0.6,
             "subclonal": false, "whole_chromosome": false,
             "genes": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]}],
 "flags": [{"code": "loss_7", "text": "Monosomy 7 or loss of 7q."},
           {"code": "cdkn2a_loss", "text": "Loss of CDKN2A (9p21.3), 0 copies."}],
 "warnings": [], "fit": {"score": 3.1, "estimable": true, "alternatives": [
     {"baseline": 2, "purity": 0.62, "offset": 0.0, "score": 3.1, "delta": 0.0, "noted": false},
     {"baseline": 3, "purity": 0.9, "offset": 0.01, "score": 20.0, "delta": 16.9, "noted": true, "modal_number": 68,
      "karyotype": "68<3n>,XXY,-7", "ploidy_class": "near-triploid"},
     {"baseline": 1, "purity": 0.4, "offset": 0.0, "score": 20.1, "delta": 17.0, "noted": false}]},
 "qc": {"status": "ok", "bin_size": 1000000, "median_reads_per_bin": 110.0, "noise_log2": 0.14,
        "detection_limit": {"5 Mb": 0.43, "20 Mb": 0.22}}, "method": "Read counts in bins.",
 "sequencing": {"reads": 1204311, "read_length_median": 2310, "read_length_mean": 4998, "read_length_n50": 7905,
                "bases_gb": 6.02, "mean_depth": 1.943, "reads_per_mb_autosomes": 394.8}}
EOF
printf '%s\n' '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="40" viewBox="0 0 240 40">' \
    '<rect x="4" y="12" width="232" height="16" fill="#e0e0e0"/><rect x="170" y="12" width="30" height="16" fill="#2166ac"/>' \
    '</svg>' > "$TMP/karyotypes/ONT01.karyotype.svg"

# the shipped interpretation text covers the classes of the real model; the synthetic classes get short entries
step interpretation python - "$TMP/interpretation.json" <<'INTERP'
import json, sys
text = json.load(open("configs/interpretation.json"))
for name in ("AML_KMT2A-r", "AML_mutated_NPM1_Nup98_DEKnup214", "B-ALL_ETV6-RUNX1", "B-ALL_High hyperdiploid",
             "B-ALL_PAX5 P80R", "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like", "B-ALL_TCF3-PBX1", "T-ALL"):
    text["classes"].setdefault(name, {"name": f"Synthetic class {name}",
                                      "summary": f"The methylation profile matches the synthetic class {name}."})
json.dump(text, open(sys.argv[1], "w"), indent=1)
INTERP
step report    python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report" \
                   --sample_sheet "$TMP/sheet.csv" --karyotype_dir "$TMP/karyotypes" --title "Synthetic test report" \
                   --interpretation "$TMP/interpretation.json"
refused refuse_nonempty "is not empty" \
    python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report"
step overwrite python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report" --overwrite \
                   --sample_sheet "$TMP/sheet.csv" --karyotype_dir "$TMP/karyotypes" --title "Synthetic test report" \
                   --interpretation "$TMP/interpretation.json"
refused refuse_no_predictions "No predictions.csv" \
    python scripts/report.py --predictions_dir "$TMP/data" --output_dir "$TMP/report_none"
printf 'sample,centre\nONT01,Centre A\nNOT_A_SAMPLE,Centre B\n' > "$TMP/sheet_bad.csv"
refused refuse_unknown_sample "NOT_A_SAMPLE" \
    python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report_bad" --sample_sheet "$TMP/sheet_bad.csv"
refused refuse_unknown_selection "not in" \
    python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report_sel" --samples ONT01 NOPE

# the same predictions without the model folder (no cross-validation section), for two samples only, at another
# threshold, after the model folder has been moved away
mv "$TMP/run" "$TMP/run_moved"
step report_without_model python scripts/report.py --predictions_dir "$TMP/pred" --output_dir "$TMP/report_nomodel" \
                   --samples ont02 ONT03 --threshold 0.15
mv "$TMP/run_moved" "$TMP/run"

step contents python - "$TMP" <<'PY'
import json, re, sys
from html.parser import HTMLParser
from pathlib import Path
import pandas as pd
tmp = Path(sys.argv[1])
rep = tmp / "report"

info = json.loads((tmp / "pred" / "prediction_info.json").read_text())
expected = {"model_dir", "model_name", "models_used", "n_networks", "threshold", "min_coverage", "n_classes", "n_cpgs",
            "sparsh_version", "created", "hierarchy_file", "display_names_file"}
assert expected <= set(info), sorted(expected - set(info))
assert info["model_name"] == "run" and Path(info["model_dir"]).is_absolute(), info["model_dir"]
assert info["n_networks"] == 2 and info["models_used"] == "ensemble" and info["threshold"] == 0.9, info
assert info["sparsh_version"] == (Path.cwd() / "VERSION").read_text().strip(), info["sparsh_version"]

pred = pd.read_csv(tmp / "pred" / "predictions.csv")
samples = sorted(pred["sample"])
assert len(samples) == 8, samples
assert (rep / "index.html").exists()
pages = {s: rep / "cases" / f"{s}.html" for s in samples}
missing = [s for s, p in pages.items() if not p.exists()]
assert not missing, missing
assert len(list((rep / "cases").glob("*.html"))) == len(samples)


class Checker(HTMLParser):
    """Parses a page and records what a self-contained page must not have."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.problems, self.text = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and "src" in a:
            self.problems.append("script with src")
        if tag == "link":
            self.problems.append("link element")
        for key in ("src", "href", "xlink:href", "srcset", "poster", "data", "action", "formaction"):
            value = (a.get(key) or "").strip().lower()
            if value.startswith(("http:", "https:", "//", "ftp:")):
                self.problems.append(f"{tag} {key}={value[:40]}")
        if "style" in a and "url(" in (a["style"] or ""):
            self.problems.append("url() in a style attribute")
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self.stack and self.stack[-1] in ("style", "script"):
            return
        self.text.append(data)


def check_page(path):
    raw = path.read_text(encoding="utf-8")
    checker = Checker()
    checker.feed(raw)
    checker.close()
    problems = list(checker.problems)
    if checker.stack:
        problems.append(f"unclosed elements {checker.stack}")
    for style in re.findall(r"<style[^>]*>(.*?)</style>", raw, flags=re.S):
        if re.search(r"@import|url\(", style):
            problems.append("@import or url() in a style sheet")
    for script in re.findall(r"<script[^>]*>(.*?)</script>", raw, flags=re.S):
        if re.search(r"https?:|fetch\(|XMLHttpRequest|import\(", script):
            problems.append("network access in a script")
    if re.search(r"<script[^>]+src\s*=|<link[^>]+href\s*=", raw, flags=re.I):
        problems.append("<script src= or <link href=")
    assert raw.lstrip().lower().startswith("<!doctype html>"), f"{path.name}: no doctype"
    assert not problems, f"{path.name}: {problems}"
    return " ".join(" ".join(checker.text).split())


ruo = "For research use only. Not for use in diagnostic procedures."
summary = pd.read_csv(rep / "report_summary.csv")
assert len(summary) == len(samples) and sorted(summary["sample"]) == samples, summary["sample"].tolist()
assert {"sample", "lineage", "result", "confidence", "reported_level", "interpretation_headline"} <= set(summary.columns)
by_sample = summary.set_index("sample")
for s, page in pages.items():
    text = check_page(page)
    row = pred.set_index("sample").loc[s]
    result = by_sample.at[s, "result"]
    for needed in (s, ruo, "Synthetic test report", str(result), "Class probabilities", "Interpretation",
                   "Evidence from cross-validation", "Input and quality", "About this result"):
        assert needed in text, f"{s}: {needed!r} missing"
    if row["reported_level"] == "subtype":
        assert result == row["reported_call"], (s, result, row["reported_call"])
    elif row["reported_level"] == "none":
        assert result == "Not callable", (s, result)
    else:
        assert result == row["reported_call"] or result.startswith(row["lineage"]), (s, result)
    assert "<svg" in page.read_text() and "Threshold 0.90" in page.read_text(), f"{s}: no chart"
    order = [text.index(h) for h in ("Class probabilities", "Interpretation", "Evidence from cross-validation",
                                     "Input and quality", "About this result")]
    assert order == sorted(order), f"{s}: sections out of order"
    has_kt = "Copy-number karyotype" in text
    assert has_kt == (s in ("ONT01", "ONT02")), f"{s}: copy-number section {'present' if has_kt else 'missing'}"

one = check_page(pages["ONT01"])
cv = pd.read_csv(tmp / "run" / "cv_metrics_by_condition.csv").set_index("condition")
cov = float(pred.set_index("sample").at["ONT01", "coverage_pct"]) / 100
binary = cv[cv.index.str.match(r"^binary_[0-9.]+$")]
nearest = (binary["observed_fraction"] - cov).abs().idxmin()
assert f"condition {nearest} in cv_metrics_by_condition.csv" in one, f"ONT01: not the nearest condition {nearest}"
assert "not from nanopore samples" in one
for needed in ("54,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21[cp]", "High hyperdiploid, 54 chromosomes",
               "Test file written by hand", "Centre A", "Blast percent", "data:image/svg+xml;base64,"):
    assert needed in one or needed in pages["ONT01"].read_text().replace("<wbr>", ""), f"ONT01: {needed!r} missing"
two = check_page(pages["ONT02"])
assert "Centre B" in two
for needed in ("sseq[GRCh37] (7)x1,9p21.3(20000001_24000000)x0", "45,XY,-7", "Leukaemic fraction (from copy number)",
               "0.62", "Extent", "whole chromosome", "G1, G2, G3, G4, G5, G6 and 2 more", "Monosomy 7 or loss of 7q.",
               "one of 20 Mb when at least 22% do", "Whole-genome reads used", "1,204,311",
               "median 2,310 bases, N50 7,905", "1.94x (6.02 Gb of reads over the genome)", "Other ploidies that fit",
               "68 chromosomes (near-triploid), baseline 3 copies, leukaemic fraction 0.90 (cost higher by 16.9)"):
    assert needed in two or needed in pages["ONT02"].read_text().replace("<wbr>", ""), f"ONT02: {needed!r} missing"
events_head = two[two.index("Extent"):two.index("Genes of interest")]
assert "Start" not in events_head and "IKZF1" not in two, "ONT02: module events should show neither start nor " \
    "the genes of whole-chromosome events"
assert "Note" not in two[two.index("Input and quality"):two.index("About this result")], "ONT02: empty note shown"
assert "fraction 0.40" not in two, "ONT02: an alternative the copy-number step did not describe is shown"
assert "The sample sheet has no row for this sample" in check_page(pages["ONT03"])

index = check_page(rep / "index.html")
for s in samples:
    assert f'href="cases/{s}.html"' in (rep / "index.html").read_text(), f"index: no link to {s}"
assert ruo in index and "Filter" in index

# without the model folder: the cross-validation section is left out and the calls follow the new threshold
other = tmp / "report_nomodel"
assert sorted(p.name for p in (other / "cases").glob("*.html")) == ["ONT02.html", "ONT03.html"]
for p in (other / "cases").glob("*.html"):
    text = check_page(p)
    assert "Evidence from cross-validation" not in text and "reporting threshold 0.15" in text, p.name
    assert "Class probabilities" in text and ruo in text, p.name
check_page(other / "index.html")
s2 = pd.read_csv(other / "report_summary.csv")
assert len(s2) == 2 and (s2["threshold"] == 0.15).all(), s2
assert (s2["reported_level"] != "none").all(), "at 0.15 the synthetic samples should be reported at some level"
# every class of the synthetic model has interpretation text: no warning about missing entries
for log in ("report.log", "overwrite.log"):
    assert "has no entry for" not in (tmp / log).read_text(), f"{log}: a synthetic class lacks interpretation text"
# the shipped text covers every class, family and lineage of the released model
colours = json.load(open("configs/class_colours.json"))["classes"]
released = [c for c in colours if c not in ("AML-MR", "AML_MECOM-r")]          # the two are not model classes
shipped = json.load(open("configs/interpretation.json"))
lacking = [c for c in released if not (shipped["classes"].get(c, {}).get("name")
                                       and shipped["classes"].get(c, {}).get("summary"))]
assert len(released) == 32 and not lacking, lacking
hierarchy = json.load(open("configs/class_hierarchy.json"))
assert not [f for f in hierarchy["families"] if f not in shipped["families"]], "a family lacks text"
assert not [lin for _, lin in hierarchy["lineage_by_prefix"] if lin not in shipped["lineages"]], "a lineage lacks text"
assert set(shipped["not_in_model"]) <= {lin for _, lin in hierarchy["lineage_by_prefix"]}
print(f"contents ok: {len(samples)} pages, index, summary, prediction_info.json")
PY
echo "REPORT TEST PASSED"
