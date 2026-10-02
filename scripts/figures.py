#!/usr/bin/env python3
"""
Figures of a model for the manuscript: from its training arrays a t-SNE map, signature heatmaps with global
methylation by class, and maps of differentially methylated regions; from its cross-validation the F1 score of
every class; and a schematic of its training and network.

    python scripts/figures.py tsne        --model_dir RUN --out_dir FIGURES --annotation cpg_annotation_hg19.tsv.gz
    python scripts/figures.py markers     --model_dir RUN --out_dir FIGURES --annotation cpg_annotation_hg19.tsv.gz
    python scripts/figures.py performance --model_dir RUN --out_dir FIGURES
    python scripts/figures.py schematic   --model_dir RUN --out_dir FIGURES
    Rscript figures/dmr.R --input FIGURES/regions_input --out FIGURES/regions
    python scripts/figures.py regions     --model_dir RUN --out_dir FIGURES --annotation cpg_annotation_hg19.tsv.gz

RUN is the training run folder of the model (its folds.csv lists the arrays and their classes, its config.json
the training pickle, which --data_path can replace, and the settings the schematic shows; its cv_predictions
files hold the cross-validation). The annotation comes from scripts/cpg_annotation.py. The arrays are read from
the pickle once and kept in FIGURES/cache for the later steps. --extra_classes adds arrays of classes left out of
the model (for example AML-MR AML_MECOM-r) to a second, supplementary t-SNE map; the other figures use the
model's arrays only. Names come from configs/figure_names.json. jobs/figures.pbs runs all the steps.
docs/FIGURES.md describes the methods.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from figures.data import arrays_for, read_annotation  # noqa: E402
from figures.style import load_style  # noqa: E402


def common(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.cache_dir:
        args.cache_dir = str(out / "cache")
    arrays = arrays_for(args)
    annotation = read_annotation(args.annotation, arrays.cpgs)
    style = load_style(sorted(set(arrays.classes)))
    return out, arrays, annotation, style


def remove_stale(out: Path, patterns) -> None:
    """Removes figures an earlier version of these steps wrote under names no longer used."""
    stale = [f for pattern in patterns for f in sorted(out.glob(pattern))]
    for f in stale:
        f.unlink()
    if stale:
        print(f"Removed {len(stale)} files of an earlier layout: {', '.join(sorted({f.name for f in stale}))}")


def heatmap_groups(style, present):
    """[(name, classes)] for the signature heatmaps: the groups of lineages of figure_names.json, then any rest."""
    if not style.heatmap_groups:
        return [("all", list(present))]
    groups, used = [], set()
    for name, lineages in style.heatmap_groups.items():
        members = [c for c in present if style.lineage[c] in lineages]
        if members:
            groups.append((name, members))
            used.update(members)
    rest = [c for c in present if c not in used]
    if rest:
        groups.append(("other", rest))
    return groups


def cmd_tsne(args) -> int:
    from figures.tsne import correlation_distance, plot_map, run_tsne, variable_cpgs

    out, arrays, annotation, style = common(args)
    if arrays.in_model.all():
        remove_stale(out, ["tsne_with_excluded_classes_*"])
    settings = {"n_cpgs": args.n_cpgs, "max_missing": 0.05, "distance": "1 - Pearson correlation",
                "perplexity": args.perplexity, "seed": args.seed, "iterations": args.iterations}
    runs = [("tsne", arrays.in_model)]
    if (~arrays.in_model).any():
        runs.append(("tsne_with_excluded_classes", np.ones(len(arrays.samples), bool)))
    for stem, keep in runs:
        part = arrays.subset(keep)
        print(f"{stem}: {len(part.samples)} arrays")
        cols = variable_cpgs(part.betas, annotation, args.n_cpgs)
        coords = run_tsne(correlation_distance(part.betas[:, cols]), args.perplexity, args.seed, args.iterations)
        pd.DataFrame({"sample_id": part.samples, "class": part.classes,
                      "shown": [style.name(c) for c in part.classes],
                      "lineage": [style.lineage[c] for c in part.classes], "in_model": part.in_model,
                      "tsne_1": coords[:, 0].round(4), "tsne_2": coords[:, 1].round(4)}).to_csv(
            out / f"{stem}_coordinates.csv", index=False)
        (out / f"{stem}_cpgs.txt").write_text("\n".join(np.array(part.cpgs)[cols]) + "\n")
        (out / f"{stem}_settings.json").write_text(json.dumps({**settings, "arrays": int(len(part.samples)),
                                                                 "cpgs_used": int(len(cols))}, indent=1))
        highlight = ~part.in_model if stem != "tsne" else None
        plot_map(coords, part.classes, style, out / f"{stem}_classes", highlight=highlight)
        plot_map(coords, part.classes, style, out / f"{stem}_classes_labelled", labelled=True, highlight=highlight)
        plot_map(coords, part.classes, style, out / f"{stem}_lineages", colour_by="lineage", highlight=highlight)
    print(f"Written to {out}")
    return 0


def cmd_markers(args) -> int:
    from figures.heatmap import (column_order, global_methylation, plot_global, plot_heatmap, signature_cpgs,
                                 summary_table)
    from figures.regions import write_input
    from figures.stats import one_vs_rest

    out, arrays, annotation, style = common(args)
    remove_stale(out, ["signature_heatmap.png", "signature_heatmap.pdf", "signature_heatmap.svg",
                       "signature_heatmap_columns.csv", "signature_cpgs.csv"])
    part = arrays.subset(arrays.in_model)
    order = [c for c in style.order if c in set(part.classes)]
    print(f"Each of {len(order)} classes against all other arrays, over {part.betas.shape[1]} CpGs")
    statistics = one_vs_rest(part.betas, part.classes, order)
    summary_table(statistics, order, annotation, args.min_delta).to_csv(out / "class_markers_summary.csv",
                                                                         index=False)
    listing = write_input(statistics, annotation, order, style, out / "regions_input")
    print(f"Region input for {len(listing)} classes in {out / 'regions_input'}")
    del statistics
    total = 0
    for name, members in heatmap_groups(style, order):
        if len(members) < 2:
            print(f"Heatmap {name}: only {len(members)} class, not drawn")
            continue
        sub = part.subset(np.isin(part.classes, members))
        print(f"Heatmap {name}: {len(members)} classes, {len(sub.samples)} arrays, each class against the other "
              f"arrays of the group")
        group_stats = one_vs_rest(sub.betas, sub.classes, members)
        signatures = signature_cpgs(group_stats, members, annotation, args.per_class, args.min_delta)
        signatures.drop(columns="index").to_csv(out / f"signature_cpgs_{name}.csv", index=False)
        columns = column_order(sub.betas, sub.classes, members, signatures)
        plot_heatmap(sub.betas, sub.classes, signatures, columns, style, out / f"signature_heatmap_{name}")
        pd.DataFrame({"sample_id": sub.samples[columns], "class": sub.classes[columns]}).to_csv(
            out / f"signature_heatmap_{name}_columns.csv", index=False)
        total += len(signatures)
        del sub, group_stats
    values = global_methylation(part.betas)
    pd.DataFrame({"sample_id": part.samples, "class": part.classes, "mean_beta": values.round(4)}).to_csv(
        out / "global_methylation.csv", index=False)
    plot_global(values, part.classes, style, out / "global_methylation")
    print(f"Signature CpGs in the heatmaps: {total}")
    print(f"Next: Rscript figures/dmr.R --input {out / 'regions_input'} --out {out / 'regions'}")
    return 0


def cmd_performance(args) -> int:
    from figures.data import read_run
    from figures.performance import condition_label, f1_table, plot_f1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds, _ = read_run(Path(args.model_dir))
    style = load_style(sorted(set(folds["label"])))
    classes = [c for c in style.order if c in set(folds["label"])]
    table = f1_table(Path(args.model_dir), args.conditions, classes, style)
    table.to_csv(out / "f1_by_class.csv", index=False)
    plot_f1(table, args.conditions, style, out / "f1_by_class")
    for cond in args.conditions:
        print(f"{condition_label(cond)}: mean F1 over {len(classes)} classes "
              f"{np.nanmean(table[f'f1_{cond}']):.3f}, lowest {np.nanmin(table[f'f1_{cond}']):.3f}")
    print(f"Written to {out}")
    return 0


def cmd_schematic(args) -> int:
    from figures.schematic import plot_schematic, read_settings

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = read_settings(Path(args.model_dir))
    plot_schematic(settings, out / "training_schematic")
    print(f"{settings['arrays']:,} arrays, {settings['classes']} classes, {settings['cpgs']:,} CpGs; layers "
          f"{settings['hidden']}; written to {out / 'training_schematic.png'}")
    return 0


def cmd_regions(args) -> int:
    from figures.regions import plot_counts, plot_genome_map, plot_loci, read_regions, top_regions

    out, arrays, annotation, style = common(args)
    part = arrays.subset(arrays.in_model)
    order = [c for c in style.order if c in set(part.classes)]
    listing = pd.read_csv(out / "regions_input" / "classes.tsv", sep="\t")
    regions = read_regions(Path(args.dmr_dir) if args.dmr_dir else out / "regions", listing, annotation,
                           args.min_cpgs, args.min_delta)
    regions.to_csv(out / "dmrs_all.tsv", sep="\t", index=False)
    kept = regions[regions["kept"]]
    print(f"{len(regions)} regions; {len(kept)} with at least {args.min_cpgs} CpGs and a mean difference of at "
          f"least {args.min_delta}")
    plot_counts(regions, style, order, out / "dmr_counts")
    plot_genome_map(regions, style, order, out / "dmr_map")
    picks = top_regions(regions, order, args.loci_per_class)
    picks.to_csv(out / "dmr_loci.tsv", sep="\t", index=False)
    if len(picks):
        plot_loci(picks, part.betas, part.classes, annotation, style, order, out / "dmr_loci")
    print(f"Written to {out}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def shared(s):
        s.add_argument("--model_dir", required=True, help="training run folder (folds.csv, config.json)")
        s.add_argument("--out_dir", required=True)
        s.add_argument("--annotation", default=None, help="CpG annotation from scripts/cpg_annotation.py")
        s.add_argument("--data_path", default=None, help="training pickle (default: as recorded in config.json)")
        s.add_argument("--cache_dir", default=None, help="default: OUT_DIR/cache")
        s.add_argument("--extra_classes", nargs="*", default=[],
                       help="classes outside the model to add to a supplementary t-SNE map")
        s.add_argument("--label_map", default=None, help="for --extra_classes (default: as recorded)")
        s.add_argument("--exclude_ids", default=None, help="for --extra_classes (default: as recorded)")

    t = sub.add_parser("tsne", help="t-SNE maps of the arrays")
    shared(t)
    t.add_argument("--n_cpgs", type=int, default=5000)
    t.add_argument("--perplexity", type=float, default=30.0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--iterations", type=int, default=2000)
    t.set_defaults(func=cmd_tsne)

    m = sub.add_parser("markers", help="class statistics, signature heatmaps, global methylation, region input")
    shared(m)
    m.add_argument("--per_class", type=int, default=25, help="signature CpGs per class and direction")
    m.add_argument("--min_delta", type=float, default=0.2, help="smallest difference in mean beta")
    m.set_defaults(func=cmd_markers)

    f = sub.add_parser("performance", help="F1 score of every class in cross-validation")
    f.add_argument("--model_dir", required=True, help="training run folder (folds.csv, cv_predictions_*.csv)")
    f.add_argument("--out_dir", required=True)
    f.add_argument("--conditions", nargs="+", default=["dense", "binary_0.30"],
                   help="cross-validation conditions, lightest bar first (default: dense binary_0.30)")
    f.set_defaults(func=cmd_performance)

    s = sub.add_parser("schematic", help="schematic of the training and the network, from config.json")
    s.add_argument("--model_dir", required=True, help="training run folder (config.json)")
    s.add_argument("--out_dir", required=True)
    s.set_defaults(func=cmd_schematic)

    r = sub.add_parser("regions", help="maps and locus plots of the regions from figures/dmr.R")
    shared(r)
    r.add_argument("--dmr_dir", default=None, help="output folder of dmr.R (default: OUT_DIR/regions)")
    r.add_argument("--min_cpgs", type=int, default=3)
    r.add_argument("--min_delta", type=float, default=0.2, help="smallest mean difference in beta of a region")
    r.add_argument("--loci_per_class", type=int, default=1)
    r.set_defaults(func=cmd_regions)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    missing = [name for name in ("matplotlib", "sklearn") if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(f"The figures need matplotlib and scikit-learn; not installed: {', '.join(missing)}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
