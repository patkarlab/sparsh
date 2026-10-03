"""
Copy-number calls in the vocabulary of cytogenetics: events, an ISCN description, a karyotype line, ploidy
class and flags for lesions that matter in acute leukemia.

Events
    A chromosome whose copy number is the same over at least 90% of its analyzed length is called whole
    (on the acrocentric chromosomes 13, 14, 15, 21 and 22 only the long arm is analyzed). Otherwise each arm
    that is uniform over 90% of its length is called at arm level, and any other stretch that differs from
    its chromosome or arm is a segmental event.

ISCN
    ISCN 2024 introduced "sseq" for copy-number results of shallow (low-pass) genome sequencing, written as
    for microarrays: the build in brackets, only abnormal copy numbers listed, sex chromosomes first, whole
    chromosomes with the same copy number grouped, "x" and the copy number after each item, and the
    proportion of cells in brackets for a change present in part of them, for example

        sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4
        sseq[GRCh37] (7)x1,5q14.3q33.1(83000001_150000000)x1,(8)x3
        sseq[GRCh37] 17p13.3p11.2(1_22200000)x1[0.4]

    Segments give the bands at their two ends and their coordinates, 1-based, without separators. A
    normal result lists the full complement: (1-22,X)x2 for a woman, (1-22)x2,(X,Y)x1 for a man.

Karyotype line
    The number of chromosomes and the whole-chromosome gains and losses, as a karyotype would give them,
    for example 55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21. The line is written against the ploidy level of
    the ploidy class, as acute leukemias are usually described: near-haploid genomes (up to 30 chromosomes)
    against the haploid level, with <1n>, near-triploid genomes (68 to 79) against the triploid level, with
    <3n>, near-tetraploid genomes (80 or more) against the tetraploid level, with <4n>, and everything from
    31 to 67 chromosomes, hypodiploid and high hyperdiploid genomes included, against the diploid level.
    It rests on read depth only: structural changes, balanced rearrangements and the arrangement of the
    extra material are not seen. A chromosome changed in part keeps its count: two copies on a diploid
    baseline, and on another baseline, where the copies of the rest of such a chromosome are not known
    beforehand, the copy number covering most of it.

Events against the ploidy level
    The events and the ISCN description give copy numbers against the constitutional two copies, as ISCN
    does. The flags are read against the ploidy level of the karyotype line instead, as a cytogeneticist reads
    a near-triploid or near-tetraploid karyotype: a chromosome at two copies in a near-triploid genome is a
    loss, and one at three copies there is not a change. For a genome written against the diploid level the
    two lists are the same.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .genome import ASSEMBLY_NAME, AUTOSOMES, CHROMOSOMES, band_at, centromere, chrom_label, is_acrocentric
from .normalise import normal_copies
from .reference import genes_in

WHOLE = 0.9                  # share of a chromosome or arm that one copy number must cover
MIN_EVENT_BP = 1_000_000     # smallest segmental event reported, in bases
MIN_EVENT_BINS = 3           # and in analysis bins


def ploidy_class(n: int) -> str:
    if n <= 23:
        return "below near-haploid"
    if n <= 30:
        return "near-haploid"
    if n <= 39:
        return "low hypodiploid"
    if n <= 43:
        return "high hypodiploid"
    if n <= 45:
        return "near-diploid"
    if n == 46:
        return "diploid"
    if n <= 50:
        return "low hyperdiploid"
    if n <= 67:
        return "high hyperdiploid"
    if n <= 79:
        return "near-triploid"
    if n <= 103:
        return "near-tetraploid"
    return "above near-tetraploid"


def ploidy_level(n: int) -> int:
    """
    Ploidy level a karyotype of n chromosomes is written against: that of its ploidy class (see ploidy_class), so
    that a high hyperdiploid leukemia of up to 67 chromosomes is written with gains against the diploid level.
    ISCN's own ranges (up to 34, 35 to 57, 58 to 80, 81 or more) would write 58 to 67 chromosomes against the
    triploid level, as losses, which is not how high hyperdiploid leukemia is read.
    """
    if n <= 30:
        return 1
    if n <= 67:
        return 2
    if n <= 79:
        return 3
    return 4


def _bins_by_copy(seg: pd.DataFrame, copies: np.ndarray, bins_chrom, bins_start, bins_end, finite):
    """Copy number per analysis bin (-1 where the bin was not used)."""
    cn = np.full(len(bins_chrom), -1, dtype=int)
    for s, row in enumerate(seg.itertuples()):
        idx = np.arange(row.first_bin, row.last_bin + 1)
        idx = idx[finite[idx]]
        cn[idx] = copies[s]
    return cn


def _majority(cn: np.ndarray, length: np.ndarray, mask: np.ndarray) -> Tuple[Optional[int], float]:
    use = mask & (cn >= 0)
    if not use.any():
        return None, 0.0
    totals: Dict[int, float] = {}
    for c, l in zip(cn[use], length[use]):
        totals[int(c)] = totals.get(int(c), 0.0) + float(l)
    best = max(totals, key=totals.get)
    return best, totals[best] / sum(totals.values())


def _range_text(numbers: List[int]) -> str:
    """[1, 2, 3, 5, 22] -> '1-3,5,22'."""
    numbers = sorted(numbers)
    parts, run = [], [numbers[0]]
    for v in numbers[1:]:
        if v == run[-1] + 1:
            run.append(v)
        else:
            parts.append(run)
            run = [v]
    parts.append(run)
    return ",".join(f"{r[0]}-{r[-1]}" if len(r) > 2 else ",".join(str(x) for x in r) for r in parts)


def _chrom_list(labels: List[str]) -> str:
    sex = [l for l in labels if l in ("X", "Y")]
    auto = sorted(int(l) for l in labels if l not in ("X", "Y"))
    items = []
    if auto:
        items.append(_range_text(auto))
    items += sex
    return ",".join(items)


def band_range(bands, chrom: str, start: int, end: int) -> str:
    b1, b2 = band_at(bands, chrom, start), band_at(bands, chrom, max(start, end - 1))
    label = chrom_label(chrom)
    return f"{label}{b1}" if b1 == b2 else f"{label}{b1}{b2}"


def sex_copies(level: int, sex: str) -> Tuple[int, int]:
    """Copies of X and Y at a ploidy level: X, XX, XXX, XXXX for a woman; X, XY, XXY, XXYY for a man."""
    if sex == "XX":
        return level, 0
    return {1: (1, 0), 2: (1, 1), 3: (2, 1), 4: (2, 2)}[level]


def level_copies(chrom: str, sex: str, level: int) -> int:
    """Copies of a chromosome at a ploidy level; at level 2 the constitutional number."""
    if chrom in ("chrX", "chrY"):
        return sex_copies(level, sex)[0 if chrom == "chrX" else 1]
    return level


def _analysed(chrom: str, bins, cn_bin: np.ndarray, bands) -> np.ndarray:
    """Bins of a chromosome with a copy number; on the acrocentric chromosomes only those of the long arm."""
    analysed = (bins.chrom == chrom) & (cn_bin >= 0)
    if is_acrocentric(chrom):
        analysed &= bins.start >= centromere(bands, chrom)[1]
    return analysed


def _events(cn_bin: np.ndarray, length: np.ndarray, bins, bands, genes, frac_bin, sub_bin, sex: str,
            reference: Dict[str, int]) -> List[dict]:
    """Whole-chromosome, arm and segmental events against the copies in reference (by chromosome)."""
    events = []
    for chrom in CHROMOSOMES:
        if chrom == "chrY" and sex == "XX":
            continue
        normal = reference[chrom]
        cen_start, cen_end = centromere(bands, chrom)
        analysed = _analysed(chrom, bins, cn_bin, bands)
        if not analysed.any():
            continue
        whole_cn, whole_share = _majority(cn_bin, length, analysed)
        regions = []                                    # (arm, mask, copy number of the region)
        if whole_share >= WHOLE:
            if whole_cn != normal:
                idx = np.flatnonzero(analysed & (cn_bin == whole_cn))
                events.append(_event("whole", chrom, None, bins, idx, whole_cn, normal, frac_bin, sub_bin, bands,
                                     genes))
            regions.append((None, analysed, whole_cn))
        else:
            for arm, mask in (("p", analysed & (bins.end <= cen_start + 1)), ("q", analysed & (bins.start >= cen_end - 1))):
                if not mask.any():
                    continue
                arm_cn, arm_share = _majority(cn_bin, length, mask)
                if arm_share >= WHOLE and arm_cn != normal:
                    idx = np.flatnonzero(mask & (cn_bin == arm_cn))
                    events.append(_event("arm", chrom, arm, bins, idx, arm_cn, normal, frac_bin, sub_bin, bands,
                                         genes))
                regions.append((arm, mask, arm_cn if arm_share >= WHOLE else normal))
        for arm, mask, region_cn in regions:
            idx_all = np.flatnonzero(mask)
            run = []
            for i in list(idx_all) + [None]:
                if i is not None and cn_bin[i] != region_cn and (not run or cn_bin[i] == cn_bin[run[-1]]):
                    run.append(i)
                    continue
                if run:
                    span = bins.end[run[-1]] - bins.start[run[0]]
                    # a stretch at the reference number inside a gained or lost chromosome or arm is not
                    # an abnormality of its own: the gain or loss around it already ends there
                    if span >= MIN_EVENT_BP and len(run) >= MIN_EVENT_BINS and int(cn_bin[run[0]]) != normal:
                        events.append(_event("segment", chrom, arm, bins, np.array(run), int(cn_bin[run[0]]),
                                             normal, frac_bin, sub_bin, bands, genes))
                    run = []
                if i is not None and cn_bin[i] != region_cn:
                    run = [i]
    return events


def describe(seg: pd.DataFrame, fit, sex: str, bins, log_ratio: np.ndarray, bands, genes: pd.DataFrame,
             build: str) -> dict:
    """Events, ISCN, karyotype line, ploidy class and flags for one sample."""
    finite = np.isfinite(log_ratio)
    length = (bins.end - bins.start).astype(float)
    cn_bin = _bins_by_copy(seg, fit.copies, bins.chrom, bins.start, bins.end, finite)
    frac_bin = np.full(len(bins.chrom), np.nan)
    sub_bin = np.zeros(len(bins.chrom), bool)
    for s, row in enumerate(seg.itertuples()):
        idx = np.arange(row.first_bin, row.last_bin + 1)
        frac_bin[idx] = fit.cell_fraction[s]
        sub_bin[idx] = fit.subclonal[s]

    counts = {}
    for chrom in CHROMOSOMES:
        if chrom == "chrY" and sex == "XX":
            continue
        normal = normal_copies(chrom, sex)
        analysed = _analysed(chrom, bins, cn_bin, bands)
        if not analysed.any():
            counts[chrom] = normal
            continue
        whole_cn, whole_share = _majority(cn_bin, length, analysed)
        # the karyotype line counts whole chromosomes; a chromosome changed in part keeps its count, two on a
        # diploid baseline and otherwise the copy number covering most of it
        counts[chrom] = whole_cn if whole_share >= WHOLE or fit.baseline != 2 else normal
    line = karyotype_line(counts, sex)
    constitutional = {c: normal_copies(c, sex) for c in CHROMOSOMES}
    events = _events(cn_bin, length, bins, bands, genes, frac_bin, sub_bin, sex, constitutional)
    level = line["ploidy_level"]
    at_level = {c: level_copies(c, sex, level) for c in CHROMOSOMES}
    against_ploidy = events if level == 2 else _events(cn_bin, length, bins, bands, genes, frac_bin, sub_bin, sex,
                                                       at_level)

    assembly = ASSEMBLY_NAME[build]
    gene_copies = {}
    for g in genes.itertuples():
        mid = (g.start + g.end) // 2
        at = np.flatnonzero((bins.chrom == g.chrom) & (bins.start <= mid) & (bins.end > mid) & (cn_bin >= 0))
        if len(at):
            gene_copies[g.gene] = int(cn_bin[at[0]])
    cen21 = centromere(bands, "chr21")[1]
    q21 = (bins.chrom == "chr21") & (bins.start >= cen21) & (cn_bin >= 0)
    out = {
        "events": events,
        "gene_copies": gene_copies,
        "chr21q_levels": sorted({int(v) for v in cn_bin[q21]}),
        "iscn": iscn(events, sex, assembly),
        "chromosome_copies": {chrom_label(c): int(v) for c, v in counts.items()},
        **line,
    }
    if level != 2:
        out["events_against_ploidy"] = against_ploidy
    return out


def arm_table(seg: pd.DataFrame, copies: np.ndarray, sex: str, bins, log_ratio: np.ndarray, bands) -> List[dict]:
    """
    Every chromosome arm, as arm-level copy-number reports give it: the copy number covering most of the
    arm's analyzed length, the share of that length it covers (so a change of part of an arm shows as partial),
    the mean log2 ratio of those bins, and whether that copy number is a gain or a loss against the normal
    copies of the chromosome. Arms without analyzed bins (the short arms of the acrocentric chromosomes, and Y in
    a woman) are left out.
    """
    finite = np.isfinite(log_ratio)
    length = (bins.end - bins.start).astype(float)
    cn_bin = _bins_by_copy(seg, copies, bins.chrom, bins.start, bins.end, finite)
    rows = []
    for chrom in CHROMOSOMES:
        if chrom == "chrY" and sex == "XX":
            continue
        normal = normal_copies(chrom, sex)
        cen_start, cen_end = centromere(bands, chrom)
        on = (bins.chrom == chrom) & (cn_bin >= 0)
        for arm, mask in (("p", on & (bins.end <= cen_start + 1)), ("q", on & (bins.start >= cen_end - 1))):
            if not mask.any():
                continue
            cn, share = _majority(cn_bin, length, mask)
            chosen = mask & (cn_bin == cn)
            rows.append({"arm": f"{chrom_label(chrom)}{arm}", "copy_number": int(cn), "normal_copies": int(normal),
                         "share_of_arm": round(share, 3),
                         "log2_ratio": round(float(np.mean(log_ratio[chosen])), 3),
                         "analysed_mb": round(float(length[mask].sum()) / 1e6, 1),
                         "call": "gain" if cn > normal else "loss" if cn < normal else "normal"})
    return rows


def _event(level, chrom, arm, bins, idx, cn, normal, frac_bin, sub_bin, bands, genes) -> dict:
    start, end = int(bins.start[idx[0]]), int(bins.end[idx[-1]])
    if arm is None and level != "whole":
        cen_start, cen_end = centromere(bands, chrom)
        arm = "p" if (start + end) / 2 < (cen_start + cen_end) / 2 else "q"
    fraction = np.nanmedian(frac_bin[idx]) if np.isfinite(frac_bin[idx]).any() else np.nan
    return {
        "level": level, "type": "gain" if cn > normal else "loss", "chrom": chrom_label(chrom), "arm": arm,
        "start": start, "end": end, "size_mb": round((end - start) / 1e6, 1),
        "bands": band_range(bands, chrom, start, end), "copy_number": int(cn), "normal_copies": int(normal),
        "cell_fraction": None if not np.isfinite(fraction) else round(float(fraction), 2),
        "subclonal": bool(sub_bin[idx].any()), "whole_chromosome": level == "whole",
        "genes": genes_in(genes, chrom, start, end),
    }


def iscn(events: List[dict], sex: str, assembly: str) -> str:
    prefix = f"sseq[{assembly}]"
    if not events:
        return f"{prefix} (1-22,X)x2" if sex == "XX" else f"{prefix} (1-22)x2,(X,Y)x1"
    items = []
    whole = [e for e in events if e["level"] == "whole"]
    for group in (["X", "Y"], None):
        members = [e for e in whole if (e["chrom"] in ("X", "Y")) == (group is not None)]
        by_cn: Dict[int, List[str]] = {}
        for e in members:
            by_cn.setdefault(e["copy_number"], []).append(e["chrom"])
        order = sorted(by_cn, key=lambda c: min(CHROMOSOMES.index("chr" + l) for l in by_cn[c]))
        for c in order:
            items.append(f"({_chrom_list(by_cn[c])})x{c}{_proportion(members, c)}")
    parts = [e for e in events if e["level"] != "whole"]
    parts.sort(key=lambda e: (0 if e["chrom"] in ("X", "Y") else 1, CHROMOSOMES.index("chr" + e["chrom"]), e["start"]))
    for e in parts:
        prop = f"[{e['cell_fraction']:.1f}]" if e["subclonal"] and e["cell_fraction"] is not None else ""
        items.append(f"{e['bands']}({e['start'] + 1}_{e['end']})x{e['copy_number']}{prop}")
    return f"{prefix} " + ",".join(items)


def _proportion(members: List[dict], cn: int) -> str:
    fr = [e["cell_fraction"] for e in members if e["copy_number"] == cn and e["subclonal"]
          and e["cell_fraction"] is not None]
    return f"[{np.median(fr):.1f}]" if fr and len(fr) == len([e for e in members if e["copy_number"] == cn]) else ""


def karyotype_line(counts: Dict[str, int], sex: str) -> dict:
    """Number of chromosomes and whole-chromosome changes, written as a karyotype would give them."""
    total = int(sum(counts.get(c, 0) for c in CHROMOSOMES))
    level = ploidy_level(total)
    x, y = counts.get("chrX", 0), counts.get("chrY", 0)
    # sex chromosomes expected at the ploidy level: X, XX, XXX, XXXX for a woman; X, XY, XXY, XXYY for a man
    const_x, const_y = sex_copies(level, sex)
    sex_text = "X" * min(x, const_x) + "Y" * min(y, const_y) or "-"
    changes = ["-X"] * max(0, const_x - x) + ["-Y"] * max(0, const_y - y)
    changes += ["+X"] * max(0, x - const_x) + ["+Y"] * max(0, y - const_y)
    for chrom in AUTOSOMES:
        delta = counts.get(chrom, 2) - level
        label = chrom_label(chrom)
        changes += [f"+{label}"] * max(0, delta) + [f"-{label}"] * max(0, -delta)
    marker = {1: "<1n>", 3: "<3n>", 4: "<4n>"}.get(level, "")
    line = f"{total}{marker},{sex_text}" + ("," + ",".join(changes) if changes else "")
    return {"karyotype": line, "modal_number": total, "ploidy_class": ploidy_class(total), "ploidy_level": level}
