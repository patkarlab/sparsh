"""
Figures of a model for the manuscript.

  data.py     the arrays a model was trained on, with their classes, read from the model's run folder and the
              training pickle (and kept in a cache for the later steps);
  style.py    names (configs/figure_names.json), colours and order of classes and lineages;
  tsne.py     t-SNE map of the arrays on their most variable CpGs;
  stats.py    each class against other arrays, CpG by CpG;
  heatmap.py  signature heatmaps of the CpGs that mark each class, and global methylation by class;
  regions.py  input for the region analysis in figures/dmr.R, and the maps and locus plots of its regions;
  performance.py  the F1 score of every class in the cross-validation of the training run;
  schematic.py    a schematic of the training and the network, from the run's config.json.

scripts/figures.py is the command-line entry point and docs/FIGURES.md describes the methods.
"""
