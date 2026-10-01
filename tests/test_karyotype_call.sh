#!/bin/bash
# Copy-number calling on a synthetic cohort with known answers: normal female and male, trisomy 21, high
# hyperdiploidy, an AML-like profile, a near-haploid pattern, iAMP21, a subclonal 17p loss, loss of Y and a very
# shallow sample. CPU only, about a minute. Run from the repository folder:
#   bash tests/test_karyotype_call.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PYTHON=${PYTHON:-python}

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_counts "$PYTHON" tests/make_synthetic_counts.py --out_dir "$TMP/counts" --seed "${SEED:-11}"
printf 'sample,sex,blast_percent\nCASE_LOY_M,M,90\nCASE_T21_M,M,\n' > "$TMP/sheet.csv"
step call "$PYTHON" scripts/karyotype.py call --counts_dir "$TMP/counts" --out_dir "$TMP/calls" \
              --sample_sheet "$TMP/sheet.csv"
step refuse_nonempty bash -c "! '$PYTHON' scripts/karyotype.py call --counts_dir '$TMP/counts' \
              --out_dir '$TMP/calls' 2>&1 | grep -q 'Calling'"
step fixed_bins "$PYTHON" scripts/karyotype.py call --counts_dir "$TMP/counts" --out_dir "$TMP/calls_2mb" \
              --bin_size 2000000 --samples CASE_AML_F CASE_T21_M --no_plots
mkdir -p "$TMP/one" && cp "$TMP"/counts/CASE_AML_F.counts.* "$TMP/one/"
step one_sample "$PYTHON" scripts/karyotype.py call --counts_dir "$TMP/one" --out_dir "$TMP/calls_one" --no_reference
step wig "$PYTHON" scripts/karyotype.py wig --counts_dir "$TMP/counts" --out_dir "$TMP/wig" --samples CASE_T21_M
# a stored panel from the background samples only, and the named cases, which are not in it, called against it
ls "$TMP/counts" | sed -n 's/^\(BG[0-9]*\)\.counts\.json$/\1/p' > "$TMP/background.txt"
mapfile -t BACKGROUND < "$TMP/background.txt"
step panel "$PYTHON" scripts/karyotype.py panel --counts_dir "$TMP/counts" --out "$TMP/panel.tsv.gz" \
              --samples "${BACKGROUND[@]}" --sample_sheet "$TMP/sheet.csv"
step call_panel "$PYTHON" scripts/karyotype.py call --counts_dir "$TMP/counts" --out_dir "$TMP/calls_panel" \
              --panel "$TMP/panel.tsv.gz" --samples CASE_T21_M CASE_HYPER_F CASE_AML_F CASE_LOY_M CASE_NORMAL_F \
              CASE_NORMAL_M --sample_sheet "$TMP/sheet.csv" --no_plots
step refuse_two_references bash -c "! '$PYTHON' scripts/karyotype.py call --counts_dir '$TMP/counts' \
              --out_dir '$TMP/calls_bad' --panel '$TMP/panel.tsv.gz' --no_reference > /dev/null 2>&1"
step contents "$PYTHON" - "$TMP" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

tmp = Path(sys.argv[1])
calls = tmp / "calls"

def load(name):
    return json.load(open(calls / f"{name}.karyotype.json"))

def codes(d):
    return {f["code"] for f in d["flags"]}

expect = {
    "CASE_NORMAL_F": ("sseq[GRCh37] (1-22,X)x2", "46,XX"),
    "CASE_NORMAL_M": ("sseq[GRCh37] (1-22)x2,(X,Y)x1", "46,XY"),
    "CASE_T21_M": ("sseq[GRCh37] (21)x3", "47,XY,+21"),
    "CASE_HYPER_F": ("sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4", "55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21"),
    "CASE_LOWDEPTH_F": ("sseq[GRCh37] (21)x3", "47,XX,+21"),
}
for name, (iscn, karyotype) in expect.items():
    d = load(name)
    assert d["iscn"] == iscn, (name, d["iscn"])
    assert d["karyotype"] == karyotype, (name, d["karyotype"])

hyper = load("CASE_HYPER_F")
assert hyper["ploidy_class"] == "high hyperdiploid" and "high_hyperdiploid" in codes(hyper)
assert abs(hyper["leukaemic_fraction"] - 0.85) <= 0.1, hyper["leukaemic_fraction"]

aml = load("CASE_AML_F")
assert aml["iscn"].startswith("sseq[GRCh37] (7)x1,(8)x3,5q"), aml["iscn"]
assert {"loss_5q", "loss_7", "complex", "monosomal"} <= codes(aml), codes(aml)
assert abs(aml["leukaemic_fraction"] - 0.7) <= 0.1

masked = load("CASE_MASKED_F")
assert masked["ploidy_class"] in ("near-haploid", "high hyperdiploid"), masked["ploidy_class"]
assert codes(masked) & {"hypodiploid", "hypodiploid_pattern"}, codes(masked)
assert not codes(masked) & {"loss_5q", "loss_7", "complex", "monosomal"}, codes(masked)

iamp = load("CASE_IAMP21_M")
assert "iamp21_pattern" in codes(iamp), codes(iamp)
assert iamp["gene_copies"]["RUNX1"] >= 5 and iamp["karyotype"] == "46,XY", iamp["karyotype"]

sub = load("CASE_SUB17P_F")
assert "(8)x3" in sub["iscn"] and "17p13.3" in sub["iscn"] and "x1[" in sub["iscn"], sub["iscn"]
e17 = [e for e in sub["events"] if e["chrom"] == "17"][0]
# the loss is in 40% of cells, so its end on 17p is less sharp than that of a clonal change
assert e17["start"] == 0 and 12e6 <= e17["end"] <= 25e6, e17
assert e17["subclonal"] and 0.25 <= e17["cell_fraction"] <= 0.55, e17
assert "loss_17p" in codes(sub)

loy = load("CASE_LOY_M")
assert loy["sex_source"] == "sample sheet" and "(Y)x0" in loy["iscn"] and "loss_of_y" in codes(loy)
assert 0.7 <= loy["leukaemic_fraction"] <= 1.0

seq = load("CASE_T21_M")["sequencing"]
assert seq["reads"] > 0 and seq["read_length_median"] == 2500 and 0 < seq["mean_depth"] < 1, seq

low = load("CASE_LOWDEPTH_F")
assert low["qc"]["bin_size"] == 5000000, low["qc"]
assert load("CASE_T21_M")["qc"]["bin_size"] == 1000000
for name in expect:
    for suffix in (".segments.tsv", ".bins.tsv.gz", ".karyotype.svg", ".karyotype.png"):
        assert (calls / f"{name}{suffix}").exists(), (name, suffix)
summary = pd.read_csv(calls / "karyotype_summary.csv")
assert len(summary) == 46 and summary["qc_status"].ne("failed").all()
assert (calls / "reference_1000000.tsv.gz").exists()
fixed = json.load(open(tmp / "calls_2mb" / "CASE_AML_F.karyotype.json"))
assert fixed["qc"]["bin_size"] == 2000000 and "(7)x1" in fixed["iscn"]
assert not (tmp / "calls_2mb" / "CASE_AML_F.karyotype.png").exists()
arms = pd.read_csv(calls / "CASE_AML_F.arms.tsv", sep="\t").set_index("arm")
assert arms.loc[["7p", "7q"], "copy_number"].eq(1).all() and arms.loc[["8p", "8q"], "copy_number"].eq(3).all()
assert arms.loc["1q", "call"] == "normal" and "13p" not in arms.index, arms
assert (calls / "CASE_AML_F.chromosomes.png").exists() and (calls / "CASE_AML_F.chromosomes.svg").exists()
one = json.load(open(tmp / "calls_one" / "CASE_AML_F.karyotype.json"))
assert one["qc"]["status"] == "no reference" and one["qc"]["reference_samples"] == 0, one["qc"]
assert "(7)x1" in one["iscn"] and "(8)x3" in one["iscn"], one["iscn"]
assert not list((tmp / "calls_one").glob("reference_*")), "no reference file without a reference"
wig = (tmp / "wig" / "CASE_T21_M.wig").read_text().splitlines()
assert sum(line.startswith("fixedStep") for line in wig) == 24 and wig[0].startswith("fixedStep chrom=chr1 start=1")
counts = pd.read_csv(tmp / "counts" / "CASE_T21_M.counts.tsv.gz", sep="\t", comment="#")
chr1 = [int(v) for v in wig[1:1 + 250]]
assert len(chr1) == 250 and sum(chr1) == counts.loc[counts.chrom == "chr1", "reads"].sum()

# other baselines that fit are described with their chromosome count; implausible counts are not described
for f in calls.glob("*.karyotype.json"):
    d = json.load(open(f))
    for a in d["fit"]["alternatives"]:
        if "modal_number" in a:
            assert 24 <= a["modal_number"] <= 103 and a["delta"] <= 40 and a["karyotype"] and a["ploidy_class"], a

# the karyotype line is written against the ploidy level of its ploidy class
from karyotype.genome import CHROMOSOMES
from karyotype.nomenclature import karyotype_line
def line(changes, sex="XX"):
    c = {ch: 2 for ch in CHROMOSOMES}
    c["chrX"], c["chrY"] = (2, 0) if sex == "XX" else (1, 1)
    c.update(changes)
    return karyotype_line(c, sex)
heh = line({**{f"chr{k}": 3 for k in (2, 4, 5, 6, 8, 10, 11, 14, 16, 17, 18, 22)}, "chrX": 3, "chr21": 4})
assert heh["karyotype"] == "61,XX,+X,+2,+4,+5,+6,+8,+10,+11,+14,+16,+17,+18,+21,+21,+22", heh
assert heh["ploidy_class"] == "high hyperdiploid"
hap = line({**{f"chr{k}": 1 for k in range(1, 23) if k not in (10, 14, 18, 21)}, "chrX": 1})
assert hap["karyotype"] == "27<1n>,X,+10,+14,+18,+21" and hap["ploidy_class"] == "near-haploid", hap
tri = line({**{f"chr{k}": 3 for k in range(1, 23)}, "chrX": 3, "chr7": 2})
assert tri["karyotype"] == "68<3n>,XXX,-7" and tri["ploidy_class"] == "near-triploid", tri

# called against the stored panel of the 36 background samples, the cases give the cohort's answers
panel_calls = tmp / "calls_panel"
for name in ("CASE_T21_M", "CASE_HYPER_F", "CASE_NORMAL_F", "CASE_NORMAL_M"):
    d = json.load(open(panel_calls / f"{name}.karyotype.json"))
    assert (d["iscn"], d["karyotype"]) == expect[name], (name, d["iscn"], d["karyotype"])
    assert d["qc"]["reference"] == "stored panel panel.tsv.gz" and d["qc"]["reference_samples"] == 36, d["qc"]
    assert "stored reference panel of 36 samples" in d["method"], d["method"]
aml_p = json.load(open(panel_calls / "CASE_AML_F.karyotype.json"))
assert "(7)x1" in aml_p["iscn"] and "(8)x3" in aml_p["iscn"], aml_p["iscn"]
loy_p = json.load(open(panel_calls / "CASE_LOY_M.karyotype.json"))
assert "(Y)x0" in loy_p["iscn"] and loy_p["sex_source"] == "sample sheet", loy_p["iscn"]
normal_m = json.load(open(panel_calls / "CASE_NORMAL_M.karyotype.json"))
assert normal_m["sex"] == "XY" and normal_m["qc"]["y_density_thresholds"]["from"].startswith("reference panel")
import gzip
text = gzip.open(tmp / "panel.tsv.gz", "rt").read()
assert "BG0" not in text and "CASE" not in text, "the panel must hold no sample names"
print("contents ok")
PY
echo "KARYOTYPE CALL TEST PASSED"
