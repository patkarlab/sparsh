# Classifying nanopore samples

This document describes how a nanopore run becomes the input of SPARSH, how scripts/predict.py classifies it, what it writes, and how predictions are scored against known diagnoses.

## From reads to the input file

The model reads one CSV file per sample, in wide format. The first column holds a row name, and every other column is an array CpG probe ID (cg...) with the methylation value of that CpG, between 0 and 1. SPARSH names each sample after its file. CpGs without reads may be absent, or left empty, NA or NaN. There is exactly one data row. The loader (data/ont.py) refuses anything that would otherwise give a silently wrong prediction: several rows, a file whose first column is already a CpG, values that are not numbers, and values above 1, which are usually percentages. A probe ID that occurs in more than one column is combined by the mean of the copies that have a value, by their majority call (missing when they are split evenly), by the first column only, or refused (--duplicate_probes mean, majority, first or error), and the number of such probes is reported. The samples of the article were read with the mean; with calls of 0 and 1, two replicate probes that disagree then give 0.5, which under the model's input encoding carries no signal but counts toward the coverage, and scripts/ont_coverage.py reports the share of such values. The majority rule keeps the input to calls.

The model was trained for one call per covered CpG, 1 when most reads at the CpG are methylated and 0 when most are not (docs/TRAINING.md). scripts/run_sample.py makes such a file from a sample's BAM files as one step of the whole pipeline (docs/PIPELINE.md). On its own, from reads aligned to GRCh37 with their modified-base calls:

```bash
modkit pileup --ref hg19.fasta --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed
python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE_pileup.bed --manifest EPIC-8v2-0_A2.csv \
    --sample SAMPLE --out samples/SAMPLE.csv
```

modkit counts 5-methylcytosine and 5-hydroxymethylcytosine together as modified and reports the two strands of a CpG apart. scripts/bedmethyl_to_csv.py places every probe of Illumina's MethylationEPIC v2.0 manifest at its GRCh37 position (the column MAPINFO_37, with the chromosome of the column CHR, which is the GRCh37 one for all but 55 probes; see docs/PIPELINE.md), adds the calls of the two strands of the probe's CpG, and writes 1 when more than half of them are modified and 0 when fewer than half are; a probe whose calls are split exactly in half is left out. It names the columns by the probe identifier up to its first underscore, so the replicate probes of a CpG give columns of the same name, which prediction combines. A compressed pileup is read as well. `--strands forward` uses the row at the probe's own base alone, for most probes the forward strand (docs/PIPELINE.md).

Before predicting, scripts/ont_coverage.py reads the files the way prediction will and reports, for each folder, what share of the model's CpGs its samples cover (the median, the range and the number of samples per band) next to the range the model was trained on; --output writes one line per sample. It also reports the share of observed values that are exactly 0 or 1, which should be close to 100% for such files, and any repeated probe identifiers.

## Prediction

```bash
python scripts/predict.py --model_dir /path/to/sparsh_model_0.4.0 --ont_dir samples --output_dir predictions/batch1
```

The model folder supplies everything else: the CpGs in the order of the network's inputs, the classes, the label map, and the fold networks with their temperatures. The sample name is the file name without its extension, in capitals. The probabilities are the average of the five fold networks' calibrated probabilities (`--use final` takes instead the one network trained on all arrays but an inner split of about one seventh, when the run has it). Prediction runs on a graphics processing unit when one is available and otherwise on the processor. Most of its time, from one to a few minutes, goes into loading the five networks, which need about 8 GB of memory; the samples of a batch then take seconds.

predictions.csv has one row per sample. It gives the input file, the number of CpG columns in it, how many of them are model CpGs and how many of those have a value, the coverage as a percentage of the model's CpGs, and the number of repeated probe IDs. Then come the most probable class with its probability, the second and third classes with theirs, and whether the most probable class reaches the threshold (callable). Then come the family of the most probable class, or the family the call falls back to, with its summed probability, the most probable lineage with its summed probability, and the reported call with its level. The last columns give the probability of every class, prob_<class>. A sample whose coverage is below --min_coverage, a fraction of the model's CpGs (0.01, that is 1%, by default), is marked in the note column and gets probabilities but no reported call.

The reported call is the most specific statement that reaches the threshold, the one stored with the model (0.90) unless --threshold says otherwise (evaluation/hierarchy.py). It is a subtype when the most probable class reaches it. When no class does, it is a family of related subtypes whose summed probability reaches it, for example the HOX-related classes of acute myeloid leukemia, and when no family does, a lineage such as "B-ALL, subtype undetermined". Otherwise the sample is not callable. configs/class_hierarchy.json defines the families and lineages, and a copy of the file used is written next to the predictions (class_hierarchy_used.json).

Two classes are shown under clearer names than their internal ones: T-ALL_HOXA9_ETP as T-ALL_HOXA_ETP, and B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like as B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like_Other (configs/display_names.json). Everything is computed with the internal names; only the names in the outputs change, and `--display_names none` shows the internal names. A copy of the names used is written next to the predictions (display_names_used.json).

prediction_info.json records the model folder, the networks and threshold used, the minimum coverage, the numbers of classes and CpGs, the SPARSH version and the time of the run; scripts/report.py reads it. Files that cannot be read are listed with the reason in input_errors.csv, and the other samples are still predicted.

## Scoring against known diagnoses

With `--ground_truth truth.csv`, a CSV with the columns sample and true_label, the predictions are also scored. The labels may use the internal or the shown class names, and they pass through the label map of the model folder first. A sample whose true class is not a class of the model is reported and counted as an error in the overall accuracy, never dropped. Its reported call can still be right at the lineage level, which the scoring records.

The scoring writes evaluation_per_sample.csv, evaluation_by_coverage.csv (in coverage bands of 0-5, 5-10, 10-20, 20-30 and above 30%), evaluation_by_class.csv, confusion_matrix.csv with a drawing of it in confusion_matrix.png, summary_metrics.json, and out_of_scheme_samples.csv, which lists the samples whose true class is not a class of the model. The summary gives the accuracy, balanced accuracy, top-2 accuracy, the share called at the threshold with the accuracy of those calls, and the same for the reported calls at every level.

## Batch jobs

jobs/predict.pbs runs the prediction on a PBS cluster for a model folder in RUNS_DIR (jobs/settings.sh), writing to RUNS_DIR/predictions/<model folder>/ont_eval, for example `qsub -v RUN_NAME=sparsh_model_0.4.0,ONT_DIR=/path/to/samples jobs/predict.pbs`. EVAL_NAME replaces ont_eval, so that each batch gets its own folder, and OUT_DIR sets the whole path; prediction stops at a folder that is not empty unless it is given --overwrite.
