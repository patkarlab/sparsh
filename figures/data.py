"""
The arrays behind the figures.

A training run (scripts/train.py) lists in folds.csv every array it was trained on, with the class it was given
after the label map and all exclusions, and names in config.json the pickle the values came from. The figures use
exactly those arrays and classes. Arrays of classes that were left out of the model, such as acute myeloid
leukaemia with myelodysplasia-related changes, can be added from the same pickle for a supplementary figure; they
are chosen with the run's own label map and exclusion list, so an array left out of training on purpose stays out.

Reading a pickle of a few thousand arrays takes minutes, so the arrays are kept in a cache folder (a float32
matrix and two small tables) that the later steps read in seconds.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from data.dataset import LABEL_COLUMN, SAMPLE_COLUMN, apply_label_map, load_ids_to_exclude, load_label_map


@dataclass
class Arrays:
    betas: np.ndarray                   # arrays x CpGs, float32, NaN where missing
    samples: np.ndarray
    classes: np.ndarray                 # class of every array (internal names)
    in_model: np.ndarray                # False for arrays added from classes outside the model
    cpgs: List[str]

    def subset(self, keep: np.ndarray) -> "Arrays":
        return Arrays(self.betas[keep], self.samples[keep], self.classes[keep], self.in_model[keep], self.cpgs)


def _recorded(config: dict, key: str, given: Optional[str]) -> Optional[str]:
    return given if given else config.get("data", {}).get(key)


def read_run(model_dir: Path) -> tuple:
    model_dir = Path(model_dir)
    for name in ("folds.csv", "config.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"{model_dir / name} not found: is {model_dir} a training run folder?")
    folds = pd.read_csv(model_dir / "folds.csv", dtype=str)
    if not {"sample_id", "label"} <= set(folds.columns):
        raise ValueError(f"{model_dir / 'folds.csv'} lacks the columns sample_id and label")
    config = json.loads((model_dir / "config.json").read_text())
    return folds, config


def load_arrays(model_dir, data_path: Optional[str] = None, label_map: Optional[str] = None,
                exclude_ids: Optional[str] = None, extra_classes: Sequence[str] = (), log=print) -> Arrays:
    folds, config = read_run(Path(model_dir))
    path = _recorded(config, "data_path", data_path)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"training pickle {path!r} not found; give it with --data_path")
    log(f"Reading {path}")
    df = pd.read_pickle(path)
    ids = (df[SAMPLE_COLUMN] if SAMPLE_COLUMN in df.columns else pd.Series(df.index, index=df.index))
    ids = ids.astype(str).str.strip().to_numpy()
    row_of = {s: i for i, s in enumerate(ids)}
    wanted = folds["sample_id"].str.strip().tolist()
    missing = [s for s in wanted if s not in row_of]
    if missing:
        raise ValueError(f"{len(missing)} arrays of folds.csv are not in {path}, for example {missing[:5]}")
    rows = [row_of[s] for s in wanted]
    classes = folds["label"].str.strip().tolist()
    in_model = [True] * len(rows)

    if extra_classes:
        mapping = load_label_map(_recorded(config, "label_map", label_map))
        mapped = apply_label_map(df[LABEL_COLUMN].astype(str).to_numpy(), mapping, log=False)
        excluded = load_ids_to_exclude(_recorded(config, "exclude_ids", exclude_ids))
        taken = set(wanted)
        for cls in extra_classes:
            add = [i for i, (s, c) in enumerate(zip(ids, mapped)) if c == cls and s not in excluded
                   and s not in taken]
            if not add:
                raise ValueError(f"no array of class {cls!r} in {path} (after the label map and exclusions)")
            log(f"  added {len(add)} arrays of {cls}, a class outside the model")
            rows += add
            classes += [cls] * len(add)
            in_model += [False] * len(add)

    cpg_positions = [j for j, c in enumerate(df.columns) if str(c).startswith("cg")]
    cpgs = [str(df.columns[j]) for j in cpg_positions]
    betas = df.iloc[rows, cpg_positions].to_numpy(dtype=np.float32)
    del df
    log(f"  {betas.shape[0]} arrays x {betas.shape[1]} CpGs; {len(set(classes))} classes")
    return Arrays(betas, np.array([ids[i] for i in rows], dtype=object), np.array(classes, dtype=object),
                  np.array(in_model), cpgs)


def save_cache(arrays: Arrays, folder: Path, key: dict) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "betas.npy", arrays.betas)
    pd.DataFrame({"sample_id": arrays.samples, "class": arrays.classes, "in_model": arrays.in_model}).to_csv(
        folder / "arrays.csv", index=False)
    (folder / "cpgs.txt").write_text("\n".join(arrays.cpgs) + "\n")
    (folder / "cache.json").write_text(json.dumps(key, indent=1))


def load_cache(folder: Path, key: dict) -> Optional[Arrays]:
    folder = Path(folder)
    try:
        if json.loads((folder / "cache.json").read_text()) != key:
            return None
        table = pd.read_csv(folder / "arrays.csv", dtype={"sample_id": str, "class": str})
        betas = np.load(folder / "betas.npy", mmap_mode="r")
        cpgs = (folder / "cpgs.txt").read_text().split()
    except (OSError, ValueError):
        return None
    if betas.shape != (len(table), len(cpgs)):
        return None
    return Arrays(np.asarray(betas), table["sample_id"].to_numpy(dtype=object),
                  table["class"].to_numpy(dtype=object), table["in_model"].to_numpy(dtype=bool), cpgs)


def arrays_for(args, log=print) -> Arrays:
    """The arrays of a run, from the cache folder when it holds the same selection, else from the pickle."""
    key = {"model_dir": str(Path(args.model_dir).resolve()), "data_path": args.data_path,
           "extra_classes": sorted(getattr(args, "extra_classes", None) or [])}
    cache = Path(args.cache_dir) if getattr(args, "cache_dir", None) else None
    if cache is not None:
        arrays = load_cache(cache, key)
        if arrays is not None:
            log(f"Arrays from the cache in {cache}: {arrays.betas.shape[0]} arrays x {arrays.betas.shape[1]} CpGs")
            return arrays
    arrays = load_arrays(args.model_dir, args.data_path, getattr(args, "label_map", None),
                         getattr(args, "exclude_ids", None), key["extra_classes"], log)
    if cache is not None:
        save_cache(arrays, cache, key)
        log(f"  kept in the cache {cache}")
    return arrays


def read_annotation(path: Optional[str], cpgs: Sequence[str], log=print) -> pd.DataFrame:
    """
    The CpG annotation (scripts/cpg_annotation.py) for the given CpGs, in their order: chrom, pos, genes,
    gene_regions, island. CpGs the annotation lacks get empty fields and are reported.
    """
    frame = pd.DataFrame(index=pd.Index(list(cpgs), name="probe"))
    if not path:
        log("No CpG annotation given: CpGs on the sex chromosomes cannot be recognised and regions cannot be placed")
        return frame.assign(chrom="", pos=np.nan, genes="", gene_regions="", island="")
    ann = pd.read_csv(path, sep="\t", dtype=str).set_index("probe")
    ann = ann.reindex(frame.index)
    out = pd.DataFrame({"chrom": ann["chrom"].fillna(""), "pos": pd.to_numeric(ann["pos"], errors="coerce"),
                        "genes": ann.get("genes", pd.Series(index=ann.index, dtype=str)).fillna(""),
                        "gene_regions": ann.get("gene_regions", pd.Series(index=ann.index, dtype=str)).fillna(""),
                        "island": ann.get("island", pd.Series(index=ann.index, dtype=str)).fillna("")},
                       index=frame.index)
    missing = int((out["chrom"] == "").sum())
    if missing:
        log(f"  {missing} of {len(out)} CpGs are not in the annotation {path}")
    return out
