"""
Flags for copy-number findings that matter in acute leukemia, each with a short explanation.

These restate what the profile shows in the terms a hematopathologist looks
for; they are not a classification. Structural changes that leave copy number
unchanged (balanced translocations, inversions) are invisible to this method,
and small deletions such as those of IKZF1 are usually below its resolution.

Gains and losses are read against the ploidy level of the karyotype line (the events against the ploidy level
of karyotype/nomenclature.py), so that in a near-triploid genome a chromosome at two copies counts as a loss
and the chromosomes at three copies count as no change. The definitions taken from karyotyping, a complex
karyotype and a monosomal one, count only changes a karyotype can show: whole chromosomes, arms, and segments
of 5 Mb or more.
"""

from typing import Dict, List

AML_MR_REGIONS = [
    # (code, chromosome, arm, text); whole-chromosome losses count for the arm
    ("loss_5q", "5", "q", "Loss of 5q (or monosomy 5)"),
    ("loss_7", "7", "q", "Monosomy 7 or loss of 7q"),
    ("loss_11q", "11", "q", "Loss of 11q"),
    ("loss_12p", "12", "p", "Loss of 12p"),
    ("loss_13q", "13", "q", "Monosomy 13 or loss of 13q"),
    ("loss_17p", "17", "p", "Loss of 17p (TP53 region) or monosomy 17"),
]
MR_NOTE = ("When found by karyotype, this is among the cytogenetic criteria for acute myeloid leukemia, "
           "myelodysplasia-related.")
KARYOTYPE_MB = 5.0                     # smallest segment counted as a change a karyotype can show
LEVEL_NAME = {1: "haploid", 3: "triploid", 4: "tetraploid"}


def _loses(events, chrom, arm, min_mb=KARYOTYPE_MB):
    for e in events:
        if e["chrom"] != chrom or e["type"] != "loss":
            continue
        if e["level"] == "whole" or (e["arm"] == arm and (e["level"] == "arm" or e["size_mb"] >= min_mb)):
            return True
    return False


def _gains(events, chrom, arm, min_mb=KARYOTYPE_MB):
    return any(e["chrom"] == chrom and e["type"] == "gain" and
               (e["level"] == "whole" or (e["arm"] == arm and (e["level"] == "arm" or e["size_mb"] >= min_mb)))
               for e in events)


def lesion_flags(result: Dict, sex: str, fit) -> List[Dict[str, str]]:
    events = result.get("events_against_ploidy", result["events"])
    level = result.get("ploidy_level", 2)
    against = f" against the {LEVEL_NAME[level]} level" if level in LEVEL_NAME else ""
    n = result["modal_number"]
    cls = result["ploidy_class"]
    out = []

    def add(code, text):
        out.append({"code": code, "text": text})

    if not events:
        add("no_change", "No copy-number change detected at the resolution of this run.")
        return out

    whole = [e for e in events if e["level"] == "whole"]
    gains = sorted([e for e in whole if e["type"] == "gain"], key=lambda e: e["chrom"] not in ("X", "Y"))
    losses = [e for e in whole if e["type"] == "loss"]
    gain_list = ", ".join("+" + e["chrom"] + (f" ({e['copy_number']} copies)"
                                              if e["copy_number"] - e["normal_copies"] > 1 else "") for e in gains)
    alt = {a["baseline"]: a for a in fit.alternatives}

    # hypodiploid genomes: the losses are the ploidy, not separate lesions, so the flags that count losses
    # (the myelodysplasia-related regions, the complex and the monosomal pattern) are left out for them
    losses_are_ploidy = cls in ("near-haploid", "low hypodiploid")
    if losses_are_ploidy:
        copies = result["chromosome_copies"]
        once = sum(1 for chrom, c in copies.items() if c == 1 and chrom not in ("X", "Y"))
        two = [chrom for chrom, c in copies.items() if c >= 2]
        doubled = alt.get(2)
        add("hypodiploid",
            f"{cls[0].upper() + cls[1:]}: {n} chromosomes, {once} autosomes present once"
            + (f", with two or more copies of {', '.join(two)}" if two else "") + ". The same profile is given "
            "by this clone after doubling (twice as many chromosomes, masked hypodiploidy), which read depth "
            "cannot tell apart" + (" (the doubled fit is almost as good)" if doubled and doubled["delta"] < 15 else "")
            + ". Confirm with a DNA index by flow cytometry and a single-nucleotide polymorphism array or a "
              "karyotype.")
    elif cls == "high hypodiploid":
        add("hypodiploid", f"High hypodiploidy: {n} chromosomes. Confirm with a DNA index by flow cytometry and a "
                           "karyotype.")
    elif cls == "high hyperdiploid":
        add("high_hyperdiploid", f"High hyperdiploidy: {n} chromosomes, with gains of {gain_list}.")
    elif cls in ("near-triploid", "near-tetraploid"):
        add(cls.replace("-", "_"), f"{cls[0].upper() + cls[1:]}: {n} chromosomes.")

    # a doubled near-haploid clone has four copies of its disomic autosomes; the sex chromosomes do not count,
    # since a man's single X is at two copies after doubling
    autosomal_gains = [e for e in gains if e["chrom"] not in ("X", "Y")]
    if result.get("ploidy_level") == 2 and len(autosomal_gains) >= 2 and \
            all(e["copy_number"] - e["normal_copies"] >= 2 for e in autosomal_gains):
        near_haploid = alt.get(1)
        extra = ""
        if near_haploid is not None and near_haploid["delta"] < 15:
            extra = " A fit with a near-haploid clone is almost as good as the one reported."
        add("hypodiploid_pattern",
            "Every whole-chromosome gain is to four copies, with no three-copy gains. This pattern also fits a "
            "near-haploid or low-hypodiploid clone that has doubled (masked hypodiploidy); confirm with a DNA "
            "index by flow cytometry and a single-nucleotide polymorphism array or a karyotype." + extra)

    levels = result.get("chr21q_levels", [])
    runx1 = result.get("gene_copies", {}).get("RUNX1")
    if runx1 is not None and runx1 >= 5 and len(levels) >= 3:
        add("iamp21_pattern", f"Chromosome 21 shows {len(levels)} copy-number levels, with {runx1} copies at "
                              "RUNX1: the pattern of intrachromosomal amplification of chromosome 21 (iAMP21). "
                              "Confirm by fluorescence in situ hybridization with a RUNX1 probe.")
    elif runx1 is not None and runx1 >= 5:
        add("runx1_amplified", f"{runx1} copies of the RUNX1 region on 21q22.")

    for code, chrom, arm, text in AML_MR_REGIONS:
        if not losses_are_ploidy and _loses(events, chrom, arm):
            add(code, f"{text}. {MR_NOTE}")
    if not losses_are_ploidy and _loses(events, "17", "p") and _gains(events, "17", "q") and not any(
            e["chrom"] == "17" and e["level"] == "whole" for e in events):
        add("i17q_pattern", "Loss of 17p with gain of 17q, the pattern of an isochromosome 17q. " + MR_NOTE)

    visible = [e for e in events if e["level"] != "segment" or e["size_mb"] >= KARYOTYPE_MB]
    changed = sorted({e["chrom"] for e in visible}, key=lambda c: (c in ("X", "Y"), c.zfill(2)))
    # gains alone, as in high hyperdiploidy, do not make a complex karyotype
    only_gains = all(e["type"] == "gain" for e in visible)
    if len(changed) >= 3 and not only_gains and not losses_are_ploidy:
        add("complex", f"Copy-number changes a karyotype can show (whole chromosomes, arms, or 5 Mb or more) on "
                       f"{len(changed)} chromosomes{against} ({', '.join(changed)}). In acute myeloid leukemia a "
                       "complex karyotype means three or more unrelated abnormalities; balanced changes seen only by "
                       "karyotype also count, and hyperdiploid karyotypes with only trisomies do not.")
    auto_losses = [e for e in losses if e["chrom"] not in ("X", "Y")]
    structural = [e for e in visible if e["level"] != "whole"]
    if not losses_are_ploidy and (len(auto_losses) >= 2 or (len(auto_losses) == 1 and structural)):
        add("monosomal", "Monosomal pattern: " + ("two or more autosomal monosomies" if len(auto_losses) >= 2 else
                         "an autosomal monosomy with a structural change") + ".")

    cdkn2a = [e for e in events if e["type"] == "loss" and "CDKN2A" in e["genes"]]
    if cdkn2a:
        both = any(e["copy_number"] == 0 for e in cdkn2a)
        add("cdkn2a_loss", "Loss at 9p21.3 including CDKN2A and CDKN2B"
            + ((", with both copies lost" if level == 2 else ", with every copy lost") if both else "")
            + ". Small deletions of this region are often below the resolution of this method, so its absence "
              "does not exclude them.")
    if sex == "XY" and any(e["chrom"] == "Y" and e["type"] == "loss" for e in events):
        add("loss_of_y", "Loss of the Y chromosome, which also occurs in normal blood cells with age.")
    if fit.purity is not None and fit.purity < 0.3:
        add("low_fraction", f"The leukemic fraction is estimated at {fit.purity:.0%}: changes present in "
                            "part of the leukemic cells may be missed.")
    return out
