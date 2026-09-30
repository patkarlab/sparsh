#!/usr/bin/env python3
"""
Synthetic arrays for the figure tests: a training pickle, a run folder that lists its arrays (folds.csv and
config.json, as scripts/train.py writes them), a CpG annotation, a label map and an exclusion list.

CpGs sit in clusters of five, 100 bases apart, spread over the autosomes, with a few on X and Y whose values follow
sex. Each class is hyper- or hypomethylated over its own clusters, so it has both signature CpGs and regions.
One class outside the model (AML-MR) is in the pickle but not in folds.csv, and one array of it is excluded.

    python tests/make_synthetic_arrays.py --out_dir /tmp/figtest
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {"B-ALL_ETV6-RUNX1": 40, "B-ALL_Aneuploidy": 45, "T-ALL_TLX3": 25, "AML_KMT2A-r": 35, "AML_IDH": 20,
           "Normal_Control_BM": 15}
RAW = {"B-ALL_Aneuploidy": "B-ALL_High hyperdiploid"}       # a raw label the label map turns into the class
EXTRA = {"AML-MR": 12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "run").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_clusters = 400
    chroms = rng.choice([f"chr{i}" for i in range(1, 23)], size=n_clusters)
    starts = rng.integers(1_000_000, 40_000_000, size=n_clusters)
    probes, chrom, pos = [], [], []
    for k in range(n_clusters):
        for j in range(5):
            probes.append(f"cg{len(probes):08d}")
            chrom.append(chroms[k])
            pos.append(int(starts[k] + 100 * j))
    for sex_chrom in ("chrX", "chrY"):
        for j in range(40):
            probes.append(f"cg{len(probes):08d}")
            chrom.append(sex_chrom)
            pos.append(5_000_000 + 1000 * j)
    n_cpg = len(probes)
    base = rng.choice([0.1, 0.5, 0.85], size=n_cpg, p=[0.4, 0.2, 0.4])

    names = list(CLASSES) + list(EXTRA)
    marks = {}
    free = list(rng.permutation(n_clusters))
    for c in names:
        chosen = [free.pop() for _ in range(12)]
        marks[c] = [(k, 1 if i < 6 else -1) for i, k in enumerate(chosen)]
        for k, sign in marks[c]:                    # the other classes sit at the opposite level
            base[k * 5:(k + 1) * 5] = 0.1 if sign > 0 else 0.9

    rows, labels, ids = [], [], []
    for c in names:
        n = {**CLASSES, **EXTRA}[c]
        for i in range(n):
            x = base.copy()
            for k, sign in marks[c]:
                x[k * 5:(k + 1) * 5] = 0.9 if sign > 0 else 0.05
            female = rng.random() < 0.5
            sex_rows = np.arange(n_clusters * 5, n_cpg)
            x[sex_rows[:40]] = 0.5 if female else 0.1            # X: half methylated in women
            x[sex_rows[40:]] = np.nan if female else 0.6         # Y: no values in women
            x = np.clip(x + rng.normal(0, 0.06, n_cpg), 0, 1)
            x[rng.random(n_cpg) < 0.005] = np.nan
            rows.append(x)
            labels.append(RAW.get(c, c))
            ids.append(f"GSM{len(ids) + 7000000}")
    df = pd.DataFrame(np.array(rows, dtype=np.float32), columns=probes)
    df.insert(0, "Sample_ID", ids)
    df.insert(1, "ANNOTATION", labels)
    df.to_pickle(out / "train.pkl")

    label_map = {"merge": {raw: cls for cls, raw in RAW.items()}}
    (out / "label_map.json").write_text(json.dumps(label_map))
    mapped = [next((cls for cls, raw in RAW.items() if raw == lab), lab) for lab in labels]
    extra_ids = [s for s, c in zip(ids, mapped) if c in EXTRA]
    (out / "exclude.txt").write_text(extra_ids[0] + "\n")
    folds = pd.DataFrame({"sample_id": [s for s, c in zip(ids, mapped) if c in CLASSES],
                          "label": [c for c in mapped if c in CLASSES]})
    folds["group"] = folds["sample_id"]
    folds["outer_fold"] = np.arange(len(folds)) % 5 + 1
    folds.to_csv(out / "run" / "folds.csv", index=False)
    config = {"data": {"data_path": str(out / "train.pkl"), "label_map": str(out / "label_map.json"),
                       "exclude_ids": str(out / "exclude.txt"), "exclude_classes": list(EXTRA)}}
    (out / "run" / "config.json").write_text(json.dumps(config, indent=1))
    pd.DataFrame({"probe": probes, "chrom": chrom, "pos": pos, "genes": [f"GENE{k // 5}" for k in range(n_cpg)],
                  "gene_regions": "Body", "island": "OpenSea"}).to_csv(out / "annotation.tsv.gz", sep="\t",
                                                                         index=False)
    truth = [{"class": c, "cluster": k, "direction": "hyper" if s > 0 else "hypo", "chrom": chroms[k],
              "start": int(starts[k]), "end": int(starts[k] + 400)} for c in names for k, s in marks[c]]
    pd.DataFrame(truth).to_csv(out / "truth.csv", index=False)
    print(f"{len(ids)} arrays x {n_cpg} CpGs written to {out}")


if __name__ == "__main__":
    main()
