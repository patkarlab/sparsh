"""
Calling copy number for a cohort of counted samples and writing the results.

For every sample: choose the analysis bin size from its depth, divide its GC-corrected profile by the cohort
reference (itself left out), segment, fit the leukaemic fraction and integer copy numbers, and describe the
result (events, ISCN, karyotype line, ploidy class, flags). Written per sample, under the upper-case sample
name that scripts/predict.py uses:

  <SAMPLE>.karyotype.json   everything above, with quality measures and warnings
  <SAMPLE>.segments.tsv     segments with mean log ratio, copy number and fraction of cells
  <SAMPLE>.bins.tsv.gz      analysis bins with their log ratio
  <SAMPLE>.arms.tsv         every chromosome arm: its majority copy number and the share of the arm it covers
  <SAMPLE>.karyotype.svg / .png   the genome-wide figure
  <SAMPLE>.chromosomes.svg / .png one panel per chromosome, with its bands, for reading the calls by eye

and for the cohort karyotype_summary.csv and the references used (reference_<size>.tsv.gz).
"""

import gzip
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .counts import safe_name
from .fit import ALTERNATIVE, CLOSE, MIN_CHANGE_Z, MODEL_ERROR, explain_all, fit_segments
from .genome import ASSEMBLY_NAME, AUTOSOMES, LENGTHS, load_bands
from .lesions import lesion_flags
from .nomenclature import arm_table, describe
from .normalise import (BIN_SIZES, Bins, FlatReference, PanelReference, Reference, Sample, build_reference,
                        choose_bin_size, make_bins, normal_copies, robust_noise, sample_log_ratio)
from .plot import plot_chromosomes, plot_genome
from .reference import load_genes
from .segment import segment_genome

METHOD = ("Reads with a mapping quality of 20 or more were counted in genomic bins; each profile was corrected "
          "for GC content and divided by the median profile of the other samples of the cohort, segmented by "
          "circular binary segmentation, and fitted with integer copy numbers, a leukaemic fraction and a "
          "baseline ploidy. Balanced rearrangements and changes smaller than a few bins are not seen.")
METHOD_PANEL = METHOD.replace("the median profile of the other samples of the cohort",
                              "the median profile of a stored reference panel of {n} samples sequenced the same way")


HYPODIPLOID = {"near-haploid", "low hypodiploid", "high hypodiploid"}
HYPERDIPLOID = {"high hyperdiploid", "near-triploid"}

CONFIRM = ("The ploidy class, the baseline and the leukaemic fraction are computed from read depth alone; confirm "
           "them from the plots.")


def unsupported_note(alt: dict, purity_range) -> str:
    """
    Warning for a reading on another baseline that fits the depths better than the diploid one reported but has no
    whole autosome away from its baseline (karyotype/fit.py): read depth cannot place a genome on that baseline.
    """
    name = {1: "haploid", 3: "triploid", 4: "tetraploid"}[alt["baseline"]]
    copies = {1: "one copy", 3: "three copies", 4: "four copies"}[alt["baseline"]]
    text = (f"A {name} baseline with a leukaemic fraction of {alt['purity']:.2f} fits the depths better than the "
            "reading reported")
    q = alt.get("diploid_fraction")
    if purity_range is not None and q is not None and q < purity_range[0] - 0.005:
        text += (f": it puts every level where a diploid baseline with a fraction of {q:.2f} would, below the range "
                 "the blast percentage allows")
    return (text + f". In that reading no autosome as a whole lies away from {copies}, and read depth cannot tell a "
            "genome multiplied as a whole from a diploid one, so the diploid reading is reported; changes shorter "
            "than a chromosome are often present in only part of the leukaemic cells. A DNA index by flow cytometry "
            f"would show a {name} clone.")


def detection_limit(sigma: float, bin_size: int, span_bp: int) -> Optional[float]:
    """
    Smallest fraction of cells in which a one-copy gain of span_bp is called: its segment must lie MIN_CHANGE_Z
    standard errors from normal, the error being the counting noise of its bins and the fixed allowance of the fit.
    A one-copy loss is seen at a slightly smaller fraction.
    """
    n = span_bp / bin_size
    if not np.isfinite(sigma) or n < 1:
        return None
    shift = MIN_CHANGE_Z * np.sqrt(sigma ** 2 / n + MODEL_ERROR ** 2)
    return float(min(1.0, 2.0 * (2.0 ** shift - 1.0)))


def chromosome_noise(bins: Bins, log_ratio: np.ndarray, chromosomes) -> float:
    """Per-bin noise from first differences within the given chromosomes (robust to segment boundaries)."""
    diffs = []
    for c in chromosomes:
        v = log_ratio[(bins.chrom == c) & np.isfinite(log_ratio)]
        if len(v) > 5:
            diffs.append(np.diff(v))
    d = np.concatenate(diffs) if diffs else np.array([])
    if len(d) <= 10:
        return robust_noise(log_ratio) if chromosomes == AUTOSOMES else float("nan")
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2.0))


def _rounded(value, digits: int):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def sequencing_summary(meta: dict, build: str) -> dict:
    """
    Depth of the whole-genome reads that were counted (primary alignments with mapping quality 20 or more on the
    24 chromosomes): their number, read lengths, bases, and mean depth over the genome. Bases are the reads' own
    lengths, soft-clipped ends included, so the depth is slightly overstated.
    """
    reads = meta.get("reads_counted")
    lengths = meta.get("read_length") or {}
    mean = lengths.get("mean")
    genome = float(sum(LENGTHS[build].values()))
    bases = reads * mean if isinstance(reads, (int, float)) and isinstance(mean, (int, float)) else None
    return {"reads": reads, "read_length_median": lengths.get("median"), "read_length_mean": _rounded(mean, 0),
            "read_length_n50": lengths.get("n50"), "bases_gb": _rounded(None if bases is None else bases / 1e9, 2),
            "mean_depth": _rounded(None if bases is None else bases / genome, 3),
            "reads_per_mb_autosomes": _rounded(meta.get("autosomal_reads_per_mb"), 1)}


def output_name(sample: str) -> str:
    return safe_name(sample).upper()


def call_sample(sample: Sample, bins_by_size: Dict[int, Bins], references: Dict[int, Reference], build: str,
                bin_size: Optional[int], min_reads: float, max_cn: int, purity_range, bands, genes) -> dict:
    size = bin_size or choose_bin_size(bins_by_size, sample.counts, min_reads)
    bins = bins_by_size[size]
    log_ratio, info = sample_log_ratio(sample, bins, references[size])
    sigma = chromosome_noise(bins, log_ratio, AUTOSOMES)
    sigmas = {c: sigma for c in AUTOSOMES}
    for c in ("chrX", "chrY"):
        # one copy carries half the reads: its bins are noisier by about the square root of two
        single = normal_copies(c, sample.sex) == 1
        own = chromosome_noise(bins, log_ratio, [c])
        expected = sigma * (np.sqrt(2.0) if single else 1.0)
        sigmas[c] = float(max(expected, own)) if np.isfinite(own) and (bins.chrom == c).sum() >= 30 else expected
    seg = segment_genome(bins.chrom, bins.start, bins.end, log_ratio, sigmas)
    fit = fit_segments(seg, sample.sex, sigma, max_cn, purity_range)
    normals = np.array([normal_copies(c, sample.sex) for c in seg["chrom"]])
    result = describe(seg, fit, sample.sex, bins, log_ratio, bands, genes, build)
    close = []
    for alt in fit.alternatives:
        # Read depth gives copy numbers only relative to each other: another baseline with another leukaemic
        # fraction can explain the same profile, and only the preferences for small changes separate them.
        # Noted, with a description, is another ploidy class that comes close, and at any cost up to
        # ALTERNATIVE one on the other side of diploidy (hypodiploid against hyperdiploid), which read depth
        # alone often cannot separate. A near-haploid call already carries a flag about its doubled form.
        alt["noted"] = False
        if alt["baseline"] == fit.baseline or alt["delta"] > ALTERNATIVE or \
                (fit.baseline == 1 and alt["baseline"] == 2):
            continue
        other = explain_all(seg, sample.sex, sigma, alt["purity"], alt["baseline"], alt["offset"], max_cn)
        desc = describe(seg, other, sample.sex, bins, log_ratio, bands, genes, build)
        if not 24 <= desc["modal_number"] <= 103:      # fewer chromosomes than near-haploidy allows, or more
            continue                                    # than near-tetraploidy: not a ploidy to consider
        classes = {desc["ploidy_class"], result["ploidy_class"]}
        crossing = bool(classes & HYPODIPLOID) and bool(classes & HYPERDIPLOID)
        if crossing or (alt["delta"] <= CLOSE and len(classes) == 2):
            alt.update({"noted": True, "modal_number": desc["modal_number"], "karyotype": desc["karyotype"],
                        "ploidy_class": desc["ploidy_class"]})
            close.append(alt)
    result["flags"] = lesion_flags(result, sample.sex, fit)
    result["arms"] = arm_table(seg, fit.copies, sample.sex, bins, log_ratio, bands)

    auto_seg = ~seg["chrom"].isin(["chrX", "chrY"]).to_numpy()
    lengths = (seg["end"] - seg["start"]).to_numpy(dtype=float)
    ploidy = float(np.sum(fit.copies[auto_seg] * lengths[auto_seg]) / lengths[auto_seg].sum()) if auto_seg.any() \
        else None
    notes = list(sample.warnings)
    for alt in close:
        decide = ("A blast percentage in the sample sheet, which limits the leukaemic fraction, would decide between "
                  "them, as would a DNA index by flow cytometry or a karyotype." if purity_range is None else
                  "Both agree with the blast percentage given; a DNA index by flow cytometry or a karyotype decides.")
        notes.append(f"Read depth also fits {alt['modal_number']} chromosomes ({alt['ploidy_class']}: "
                     f"{alt['karyotype']}) with a leukaemic fraction of {alt['purity']:.2f}, against "
                     f"{result['modal_number']} with {fit.purity:.2f} as reported, which needs smaller changes. "
                     + decide)
    for alt in fit.unsupported:
        notes.append(unsupported_note(alt, purity_range))
    median_reads = info.get("median_reads_per_bin")
    status = "ok"
    if median_reads is not None and median_reads < 30:
        status = "low depth"
        notes.append(f"Only {median_reads:.0f} reads per bin at the largest bin size: only large changes are "
                     "reliable.")
    if sigma > 0.35:
        status = "noisy" if status == "ok" else status
        notes.append(f"The profile is noisy (bin-to-bin standard deviation {sigma:.2f} of log2 ratio).")
    if not fit.estimable:
        notes.append("No change away from the baseline was found, so the leukaemic fraction cannot be "
                     "estimated from copy number; a leukaemia without copy-number changes and a sample with few "
                     "leukaemic cells look the same.")
    reference = references[size]
    if isinstance(reference, FlatReference):
        status = "no reference"
        notes.append("Called without a reference of other samples: only GC content was corrected, so artefacts of "
                     "mappability and of the library may remain, and chromosome Y was not assessed. For a first "
                     "look only.")
    result.update({
        "sample": output_name(sample.name),
        "sample_as_counted": sample.name,
        "build": build,
        "assembly": ASSEMBLY_NAME[build],
        "sex": sample.sex,
        "sex_source": sample.sex_source,
        "leukaemic_fraction": fit.purity,
        "baseline_copies": fit.baseline,
        "mean_copies_autosomes": None if ploidy is None else round(ploidy, 2),
        "fit": {"score": None if not np.isfinite(fit.score) else round(fit.score, 2),
                "alternatives": fit.alternatives, "estimable": fit.estimable, "unsupported": fit.unsupported},
        "confirmation": CONFIRM,
        "sequencing": sequencing_summary(sample.meta, build),
        "warnings": notes,
        "qc": {
            "status": status,
            "bin_size": int(size),
            "median_reads_per_bin": None if median_reads is None else round(float(median_reads), 1),
            "noise_log2": round(sigma, 3),
            "segments": int(len(seg)),
            "bins_used": info.get("bins_used"),
            "reference_samples": info.get("reference_samples"),
            "reference": ("stored panel " + reference.source) if isinstance(reference, PanelReference) else
                         ("none" if isinstance(reference, FlatReference) else "the other samples of the cohort"),
            "reads_counted": sample.meta.get("reads_counted"),
            "read_length_median": (sample.meta.get("read_length") or {}).get("median"),
            "detection_limit": {
                "5 Mb": detection_limit(sigma, size, 5_000_000),
                "20 Mb": detection_limit(sigma, size, 20_000_000),
            },
            "gc_effect_log2": info.get("gc_effect_log2"),
            "y_density": _rounded(sample.meta.get("y_density"), 4),
            "x_density": _rounded(sample.meta.get("x_density"), 3),
            "y_density_thresholds": sample.meta.get("y_density_thresholds"),
        },
        "method": METHOD_PANEL.format(n=reference.n_samples) if isinstance(reference, PanelReference) else METHOD,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    for key in ("5 Mb", "20 Mb"):
        v = result["qc"]["detection_limit"][key]
        result["qc"]["detection_limit"][key] = None if v is None else round(v, 2)
    seg_out = seg.copy()
    seg_out["copies"] = fit.copies
    seg_out["normal_copies"] = normals
    seg_out["cell_fraction"] = np.round(fit.cell_fraction, 2)
    seg_out["subclonal"] = fit.subclonal
    return {"result": result, "segments": seg_out, "bins": bins, "log_ratio": log_ratio, "fit": fit,
            "normals": normals}


def write_sample(out_dir: Path, called: dict, build: str, plot: bool = True) -> None:
    result = called["result"]
    stem = out_dir / result["sample"]
    with open(str(stem) + ".karyotype.json", "w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    seg = called["segments"].drop(columns=["first_bin", "last_bin", "sigma"])
    seg.to_csv(str(stem) + ".segments.tsv", sep="\t", index=False, float_format="%.4f")
    bins = called["bins"]
    with gzip.open(str(stem) + ".bins.tsv.gz", "wt") as fh:
        pd.DataFrame({"chrom": bins.chrom, "start": bins.start, "end": bins.end,
                      "log2_ratio": called["log_ratio"]}).to_csv(fh, sep="\t", index=False, float_format="%.4f")
    pd.DataFrame(result["arms"]).to_csv(str(stem) + ".arms.tsv", sep="\t", index=False)
    if plot:
        plot_genome(Path(str(stem) + ".karyotype"), result["sample"], bins, called["log_ratio"],
                    called["segments"], called["fit"].copies, called["normals"], called["fit"], result["sex"],
                    LENGTHS[build])
        plot_chromosomes(Path(str(stem) + ".chromosomes"), result["sample"], bins, called["log_ratio"],
                         called["segments"], called["fit"].copies, called["normals"], called["fit"],
                         result["sex"], LENGTHS[build], load_bands(build), result["arms"])


def summary_row(result: dict) -> dict:
    return {
        "sample": result["sample"], "sex": result["sex"], "iscn": result["iscn"], "karyotype": result["karyotype"],
        "modal_number": result["modal_number"], "ploidy_class": result["ploidy_class"],
        "leukaemic_fraction": result["leukaemic_fraction"], "baseline_copies": result["baseline_copies"],
        "flags": "; ".join(f["code"] for f in result["flags"]), "qc_status": result["qc"]["status"],
        "bin_size": result["qc"]["bin_size"], "median_reads_per_bin": result["qc"]["median_reads_per_bin"],
        "noise_log2": result["qc"]["noise_log2"], "warnings": " | ".join(result["warnings"]),
    }


def write_reference(out_dir: Path, size: int, reference: Reference, bins: Bins) -> None:
    with warnings.catch_warnings():                   # bins unusable in every reference sample stay empty
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(reference.matrix, axis=0)
    with gzip.open(out_dir / f"reference_{size}.tsv.gz", "wt") as fh:
        pd.DataFrame({"chrom": bins.chrom, "start": bins.start, "end": bins.end, "gc": bins.gc,
                      "usable": bins.usable, "median_ratio": median}).to_csv(fh, sep="\t", index=False,
                                                                           float_format="%.4f")


def call_cohort(samples: List[Sample], reference_samples: List[Sample], table: pd.DataFrame, build: str,
                out_dir: Path, bin_size: Optional[int] = None, min_reads: float = 60, max_cn: int = 8,
                purity: Optional[Dict[str, tuple]] = None, plot: bool = True, log=print,
                panel: Optional[Dict[int, PanelReference]] = None) -> pd.DataFrame:
    """
    Call every sample of `samples` against the leave-one-out reference of `reference_samples`, or against the
    stored references of `panel` (karyotype/panel.py) when it is given, or with no reference at all.
    """
    bands = load_bands(build)
    genes = load_genes(build)
    sizes = [bin_size] if bin_size else list(BIN_SIZES)
    bins_by_size = {s: make_bins(table, s) for s in sizes}
    references = {}
    for s in sizes:
        if panel is not None:
            references[s] = panel[s]
        elif reference_samples:
            references[s] = build_reference(s, reference_samples, bins_by_size[s])
            write_reference(out_dir, s, references[s], bins_by_size[s])
        else:
            references[s] = FlatReference(s, bins_by_size[s])
    if panel is not None:
        first = panel[sizes[0]]
        log(f"Reference: the stored panel {first.source}, made from {first.n_samples} samples")
    elif reference_samples:
        log(f"Reference: {len(reference_samples)} samples at bin sizes "
            f"{', '.join(f'{s // 1000} kb' for s in sizes)}")
    else:
        log("No reference of other samples: GC correction only, for a first look")
    rows = []
    for sample in samples:
        try:
            called = call_sample(sample, bins_by_size, references, build, bin_size, min_reads, max_cn,
                                 (purity or {}).get(sample.name), bands, genes)
        except ValueError as exc:
            log(f"  {sample.name}: not called ({exc})")
            rows.append({"sample": output_name(sample.name), "qc_status": "failed", "warnings": str(exc)})
            continue
        write_sample(out_dir, called, build, plot)
        r = called["result"]
        rows.append(summary_row(r))
        frac = "not estimable" if r["leukaemic_fraction"] is None else f"{r['leukaemic_fraction']:.2f}"
        log(f"  {r['sample']}: {r['karyotype']} | fraction {frac} | {r['iscn']}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "karyotype_summary.csv", index=False)
    return summary
