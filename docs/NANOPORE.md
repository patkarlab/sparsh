# Classifying nanopore samples

This document describes how a nanopore run becomes the input of SPARSH, how scripts/predict.py classifies it, what it writes, and how predictions are scored against known diagnoses.

## From reads to the input file

The model reads one CSV file per sample, in wide format. The first column holds a row name, and every other column is an array CpG probe ID (cg...) with the methylation value of that CpG, between 0 and 1. SPARSH names each sample after its file. CpGs without reads may be absent, or left empty, NA or NaN. There is exactly one data row. The loader (data/ont.py) refuses anything that would otherwise give a silently wrong prediction: several rows, a file whose first column is already a CpG, values that are not numbers, and values above 1, which are usually percentages. A probe ID that occurs in more than one column is combined by the mean of the copies that have a value, or by the first column only (--duplicate_probes), and the number of such probes is reported.

The released model was trained for one call per covered CpG, 1 when most reads at the CpG are methylated and 0 when most are not (docs/TRAINING.md). To make such a file from a nanopore run, basecall with modified-base calling for 5-methylcytosine, align the reads to GRCh37 (hg19), and summarise the methylation calls per CpG with modkit:

```bash
modkit pileup SAMPLE.bam SAMPLE.bed --cpg --ref hg19.fa --combine-strands
python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE.bed --annotation refs/cpg_annotation_hg19.tsv.gz \
    --sample SAMPLE --out samples/SAMPLE.csv
```

scripts/bedmethyl_to_csv.py keeps the rows of 5-methylcytosine, adds the reads of the two strands of a CpG together, and finds the array CpGs by the position of their C on the forward strand, taken from the CpG annotation that scripts/cpg_annotation.py builds from Illumina's manifest. It writes 1 or 0 by majority and leaves out a CpG with as many methylated as unmethylated reads; `--values fraction` writes the methylated fraction instead. A compressed bedMethyl file is read as well.

Before predicting, scripts/ont_coverage.py reads the files the way prediction will and reports, for each folder, the share of the model's CpGs each sample covers next to the range the model was trained on. It also reports the share of observed values that are exactly 0 or 1, which should be close to 100% for such files, and any repeated probe IDs.

## Prediction

```bash
python scripts/predict.py --model_dir /path/to/v04_labels --ont_dir samples --output_dir predictions/batch1
```

The model folder supplies everything else: the CpGs in the order of the network's inputs, the classes, the label map, and the fold networks with their temperatures. The sample name is the file name without its extension, in capitals. The probabilities are the average of the five fold networks' calibrated probabilities (`--use final` takes instead the one network trained on all arrays but an inner split of about one seventh, when the run has it). Prediction runs on a GPU when one is available and otherwise on the CPU, where a batch of samples takes seconds.

predictions.csv has one row per sample. It gives the input file, the number of CpG columns in it, how many of them are model CpGs and how many of those have a value, the coverage as a percentage of the model's CpGs, and the number of repeated probe IDs. Then come the most probable class with its probability, the second and third classes with theirs, and whether the most probable class reaches the threshold (callable). Then come the family of the most probable class, or the family the call falls back to, with its summed probability, the most probable lineage with its summed probability, and the reported call with its level. The last columns give the probability of every class, prob_<class>. A sample whose coverage is below --min_coverage (1% of the model's CpGs by default) is marked in the note column and gets probabilities but no reported call.

The reported call is the most specific statement that reaches the threshold, 0.90 unless --threshold says otherwise (evaluation/hierarchy.py). It is a subtype when the most probable class reaches it. When no class does, it is a family of related subtypes whose summed probability reaches it, for example the HOX-related classes of acute myeloid leukaemia, and when no family does, a lineage such as "B-ALL, subtype undetermined". Otherwise the sample is not callable. configs/class_hierarchy.json defines the families and lineages, and a copy of the file used is written next to the predictions.

Two classes are shown under clearer names than their internal ones: T-ALL_HOXA9_ETP as T-ALL_HOXA_ETP, and B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like as B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like_Other (configs/display_names.json). Everything is computed with the internal names; only the names in the outputs change, and `--display_names none` shows the internal names.

prediction_info.json records the model folder, the networks and threshold used, the minimum coverage, the numbers of classes and CpGs, the SPARSH version and the time of the run; scripts/report.py reads it. Files that cannot be read are listed with the reason in input_errors.csv, and the other samples are still predicted.

## Scoring against known diagnoses

With `--ground_truth truth.csv`, a CSV with the columns sample and true_label, the predictions are also scored. The labels may use the internal or the shown class names, and they pass through the label map of the model folder first. A sample whose true class is not a class of the model is reported and counted as an error in the overall accuracy, never dropped. Its reported call can still be right at the lineage level, which the scoring records.

The scoring writes evaluation_per_sample.csv, evaluation_by_coverage.csv (in coverage bands of 0-5, 5-10, 10-20, 20-30 and above 30%), evaluation_by_class.csv, confusion_matrix.csv and summary_metrics.json. The summary gives the accuracy, balanced accuracy, top-2 accuracy, the share called at the threshold with the accuracy of those calls, and the same for the reported calls at every level.

## Batch jobs

jobs/predict.pbs runs the prediction on a PBS cluster for a run folder in RUNS_DIR (jobs/settings.sh), for example `qsub -v RUN_NAME=v04_labels,ONT_DIR=/path/to/samples jobs/predict.pbs`.
