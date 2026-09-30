# Figures of the training arrays

The figures describe the arrays a model was trained on: a t-SNE map, a heatmap of the CpGs that mark each class, the global methylation of each class, and maps of the regions that are differentially methylated in each class. They are made in four steps, tsne, markers, the R script figures/dmr.R and regions, which one batch job runs in order. The figures need a training run together with the training arrays it lists, so they can be made only by those who hold the arrays:

```bash
python scripts/cpg_annotation.py --manifest infinium-methylationepic-v-1-0-b5-manifest-file.csv \
    --out cpg_annotation_hg19.tsv.gz
conda env create -f envs/figures_r.yml
qsub -v RUN_NAME=v04_labels,ANNOTATION=/path/to/cpg_annotation_hg19.tsv.gz,EXTRA_CLASSES=AML-MR:AML_MECOM-r \
    jobs/figures.pbs
```

The steps can also be run one by one; `python scripts/figures.py <step> --help` lists the options of a step.

## Arrays and annotation

The arrays are exactly those the model was trained on. A training run lists them in folds.csv with the class each was given after the label map and all exclusions, and names in config.json the pickle the values came from. The arrays are read from that pickle once and kept in a cache folder next to the figures. Arrays of classes left out of the model, such as acute myeloid leukaemia with myelodysplasia-related changes and with MECOM rearrangement, can be added to a second, supplementary t-SNE map. They are chosen from the same pickle with the run's own label map and exclusion list, so an array left out of training on purpose stays out. All other figures use the model's arrays only.

CpGs are placed on the genome with Illumina's manifest of the MethylationEPIC array (version 1.0 B5), in GRCh37 (hg19) coordinates, with the genes and the relation to CpG islands the manifest gives. scripts/cpg_annotation.py turns the manifest into the table the figures read. Class names, colours and their order come from configs/display_names.json, configs/class_colours.json and configs/class_hierarchy.json, so the figures use the names and colours of the reports.

## t-SNE map

The map uses the 5,000 CpGs with the largest standard deviation of beta values across the arrays. CpGs on the X and Y chromosomes, whose methylation follows sex, and CpGs missing in more than 5% of the arrays are left out; the few remaining missing values of a CpG are filled with its mean. The distance between two arrays is one minus the Pearson correlation of their values over these CpGs. The map is computed from the distance matrix with t-SNE (van der Maaten and Hinton, 2008) as implemented in scikit-learn, with a perplexity of 30, 2,000 iterations, random initialisation and a fixed seed of 42. The step writes the map coloured by class with a legend, the same with class names at the median of their arrays, the map coloured by lineage, the coordinates of every array, the list of CpGs used and the settings. With classes outside the model added, the same set is written again with the prefix tsne_with_excluded_classes, and the added arrays are outlined in black.

## Class statistics and signature heatmap

Each class is compared with all other arrays of the model, CpG by CpG. The difference is taken in mean beta value, and the test is Welch's t test on M-values (the log2 ratio of beta to one minus beta, after clipping beta to 0.001 to 0.999). The false discovery rate over all CpGs follows Benjamini and Hochberg (1995). Missing values are left out CpG by CpG.

The signature CpGs of a class are, among CpGs off the sex chromosomes whose mean beta differs by at least 0.2 between the class and the rest, the 25 with the largest positive t statistic and the 25 with the largest negative one. A CpG chosen for more than one class is shown once, in the block of the first class in the legend order. The heatmap shows their beta values in every array, with rows in blocks by the class they mark and columns grouped by class. Within a class the arrays are ordered by their signature score, the mean beta of the class's hypermethylated signature CpGs minus that of its hypomethylated ones. The CpGs chosen, with their statistics and genes, are written to signature_cpgs.csv. class_markers_summary.csv counts, for every class, the CpGs whose mean beta differs by at least 0.2 at a false discovery rate below 0.05.

Global methylation is the mean beta over all CpGs of an array, drawn per class as a box plot with the arrays as points.

## Differentially methylated regions

Regions are found with DMRcate (Peters and colleagues, Epigenetics and Chromatin, 2015) from the class statistics above. For every class, dmr.R places the autosomal CpGs on the genome and smooths their t statistics with a Gaussian kernel, using the DMRcate defaults for arrays (lambda 1,000 bases, C 2). It then joins neighbouring significant CpGs into regions. A CpG counts as individually significant when its false discovery rate is below 0.05 and its mean beta differs by at least 0.1. The second condition matters with thousands of arrays, where very small differences reach significance, because DMRcate sets its cut-off on the smoothed statistics from the number of individually significant CpGs. For each region DMRcate gives the number of CpGs, the smallest smoothed false discovery rate, three combined measures of significance and the largest and mean differences in beta.

The maps keep regions of at least three CpGs whose mean difference in beta is at least 0.2. dmr_map shows, class by class, where along the genome the class is hyper- or hypomethylated against all other arrays, and dmr_counts how many regions of each kind every class has. dmr_loci shows the strongest region of each class, the one with the smallest smoothed false discovery rate and then the largest difference, with the mean beta of every class at each CpG of the region and 2 kb around it. All regions, with the genes of their CpGs and whether they pass the filters, are in dmrs_all.tsv, and the versions of R and its packages in regions/sessionInfo.txt.

## Checking

tests/test_figures.sh runs every step on synthetic arrays whose class signatures are known. It checks that the maps use the model's arrays, add only the requested classes and leave out the sex chromosomes. It checks that the signature CpGs and the regions recover the planted signatures in the right direction. The region steps run when R with DMRcate is available (RSCRIPT=/path/to/Rscript).
