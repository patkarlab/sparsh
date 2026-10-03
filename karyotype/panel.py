"""
A stored copy-number reference, so that one new sample can be called on its own.

Called as a cohort, every sample is divided by the median profile of the other samples (karyotype/normalise.py).
A single new sample has no cohort, so the reference of a cohort is stored once instead: for every analysis bin size,
the median and the spread of the cohort's GC-corrected profiles bin by bin, the number of samples with a value in each
bin, and the thresholds on the read density of chromosome Y that tell men from women in that cohort. The panel holds
no profile of any single sample and no sample names.

File: gzip-compressed text, '# key<TAB>value' lines first (format, build, samples, the Y thresholds, the
SPARSH version, creation time), then a table with the columns size, chrom, start, end, median, spread and n.
"""

import gzip
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .normalise import BIN_SIZES, PanelReference, Sample, build_reference, make_bins, y_thresholds

FORMAT = "sparsh copy-number reference panel 1"


def make_panel(samples: List[Sample], table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """The panel table for all bin sizes, from samples whose sex has been set (load_cohort)."""
    frames = []
    for size in BIN_SIZES:
        bins = make_bins(table, size)
        reference = build_reference(size, samples, bins)
        median, spread, _ = reference.for_sample("")        # no sample is left out
        n = np.sum(np.isfinite(reference.matrix), axis=0)
        frames.append(pd.DataFrame({"size": size, "chrom": bins.chrom, "start": bins.start, "end": bins.end,
                                    "median": median, "spread": spread, "n": n}))
    return pd.concat(frames, ignore_index=True)


def write_panel(path: Path, frame: pd.DataFrame, samples: List[Sample], build: str, version: str) -> dict:
    thresholds = y_thresholds([s.meta.get("y_density", float("nan")) for s in samples],
                              [s.meta.get("x_density", float("nan")) for s in samples])
    meta = {"format": FORMAT, "build": build, "samples": str(len(samples)),
            "y_female_below": str(thresholds["female_below"]), "y_male_above": str(thresholds["male_above"]),
            "y_thresholds_from": "the panel's samples" if thresholds["from"] == "cohort" else "fixed values",
            "sparsh_version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}\t{value}\n")
        frame.to_csv(fh, sep="\t", index=False, float_format="%.6g")
    return meta


def read_panel(path: Path, table: pd.DataFrame, build: str) -> Tuple[Dict[int, PanelReference], dict]:
    """The stored references by bin size, checked against this build's bins, and the panel's sex thresholds."""
    path = Path(path)
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].strip().partition("\t")
            meta[key.strip()] = value.strip()
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a SPARSH copy-number reference panel")
    if meta.get("build") != build:
        raise ValueError(f"{path} was made on {meta.get('build')}, not {build}")
    frame = pd.read_csv(path, sep="\t", comment="#", dtype={"chrom": str})
    n_samples = int(meta["samples"])
    references = {}
    for size in BIN_SIZES:
        part = frame[frame["size"] == size]
        bins = make_bins(table, size)
        if len(part) != len(bins.chrom) or not (part["chrom"].to_numpy() == bins.chrom).all() \
                or not (part["start"].to_numpy() == bins.start).all():
            raise ValueError(f"{path}: the bins of {size} bases do not match this version's bin table; make the "
                             "panel again with this version")
        references[size] = PanelReference(size, part["median"].to_numpy(dtype=float),
                                          part["spread"].to_numpy(dtype=float), n_samples, path.name)
    thresholds = {"female_below": float(meta["y_female_below"]), "male_above": float(meta["y_male_above"]),
                  "from": f"reference panel ({meta.get('y_thresholds_from', 'unknown')})"}
    return references, {**meta, "thresholds": thresholds}
