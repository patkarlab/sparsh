# SPARSH

SPARSH classifies acute leukemia from sparse DNA methylation data. A neural network trained on Illumina methylation arrays reads the CpG methylation calls of a low-pass nanopore genome run, in which only a small and random part of the array CpGs is covered, and reports the lineage and genetic subtype of the leukemia with a probability calibrated on simulated nanopore samples. From the same nanopore run, SPARSH also estimates copy number along the genome and writes it as an ISCN karyotype. It then puts both into a report for each case.

SPARSH stands for Subtype Prediction via Adaptive Recognition from Sparse Hematological data.

SPARSH is research software. It is not a medical device, has not been cleared or approved by any regulatory authority, and must not be used for clinical decisions. Every result needs confirmation by established diagnostic methods.

## What is in this repository

The classifier is in data/, models/, training/ and evaluation/, with scripts/train.py to train it and scripts/predict.py to apply it to nanopore samples. The model described in the article, version 0.4.0, has 32 classes of acute myeloid leukemia, B- and T-lymphoblastic leukemia, acute leukemia of ambiguous lineage, blastic plasmacytoid dendritic cell neoplasm and normal bone marrow; docs/MODEL_CARD.md describes it, and it is archived apart from the code (see Getting the model). The copy-number karyotype is in karyotype/ with scripts/karyotype.py, and the per-case report in report/ with scripts/report.py. scripts/run_sample.py runs all of it for one sample, from its BAM files to its report. The documents in docs/ describe each part, and jobs/ has batch jobs for a PBS cluster.

## System requirements

SPARSH runs on 64-bit Linux. The Python parts need Python 3.10 to 3.12 with PyTorch, NumPy, pandas, SciPy, scikit-learn and matplotlib, and scripts/tune.py also needs Optuna. The tests pass on Ubuntu 24.04 (x86-64) with the versions that envs/sparsh.yml installed on October 2, 2026: Python 3.12, PyTorch 2.13 (built for the processor), NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9, matplotlib 3.11, Optuna 5.0, samtools 1.24, modkit 0.6.1 and minimap2 2.31. The model was trained, and the samples of the article were classified, on a cluster running Red Hat Enterprise Linux 8.10 (x86-64); its environment at release had Python 3.11.14, PyTorch 2.5.1 (built for CUDA 12.1), NumPy 2.4.0, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.8.0, matplotlib 3.11.2, Optuna 4.6.0, samtools 1.24, minimap2 2.31 and modkit 0.6.1. Working from a sample's reads also needs samtools 1.17 or later (for samtools reset), modkit and, for reads that are not yet aligned, minimap2 or dorado; SPARSH was tested with modkit 0.6.1 and dorado 1.3.0. The batch jobs and the tests are bash scripts that need bash 4 or later, which macOS does not have by default.

Prediction needs no graphics processing unit, and scripts/predict.py uses one when it is present. The model's five networks take about 8 GB of memory together, and the batch jobs ask for more to be safe. Training the full model needs a graphics processing unit: the model was trained on one NVIDIA H200, in about an hour. The toy model of the demo trains on the processor.

## Installation

```bash
git clone https://github.com/patkarlab/sparsh.git
cd sparsh
conda env create -f envs/sparsh.yml
conda activate sparsh
bash tests/run_all.sh
```

envs/sparsh.yml installs Python with the packages above, samtools, modkit 0.6.1, minimap2 and Optuna from conda-forge and bioconda (mamba and micromamba read the same file). Creating it takes a few minutes; with micromamba on a fast connection it took about a minute, and the environment takes about 3 GB. With pip, `pip install -r requirements.txt` installs the Python packages, and samtools, modkit and minimap2 are installed separately. dorado is not packaged for conda; it is installed from Oxford Nanopore's releases when it is wanted as the aligner, and without it scripts/run_sample.py aligns reads with --aligner minimap2.

tests/run_all.sh runs every part on synthetic data, without a graphics processing unit, in about seven minutes on two processor cores. Each test prints a line ending in PASSED, or SKIPPED when a tool it needs (samtools, modkit, Optuna) is missing, and the run ends with ALL TESTS PASSED. Without dorado, the summary also notes that the pipeline test passed with some steps skipped.

## Demo

The commands below train a toy model on synthetic arrays and classify eight synthetic nanopore samples, in under a minute on two processor cores. The toy model has eight classes and 3,000 CpGs and is not the model described in the article.

```bash
python tests/make_synthetic.py --output_dir demo/data
python scripts/train.py --data_path demo/data/train.pkl --output_dir demo/model --exclude_prefixes MPAL \
    --n_folds 3 --epochs 60 --hidden_dims 128 64 --device cpu \
    --train_sim binary --val_sim binary --eval_sims binary --eval_coverages 0.05 0.1 0.2 0.3 0.5
python scripts/predict.py --model_dir demo/model --ont_dir demo/data/ont --output_dir demo/predictions \
    --ground_truth demo/data/ground_truth.csv --device cpu
python scripts/report.py --predictions_dir demo/predictions --output_dir demo/reports
```

demo/model holds the trained networks and the cross-validation tables (docs/TRAINING.md). demo/predictions/predictions.csv gives the probabilities and calls of the eight samples, and the files beside it score them against demo/data/ground_truth.csv (docs/NANOPORE.md). On two processor cores the four commands took 36 seconds. predict.py printed "Reported at: subtype 0, no_subtype 0, family 0, lineage 4, none 4": the toy model's calls stay at the level of the lineage or below the threshold. demo/predictions/summary_metrics.json gave an accuracy of 1.0 for the seven samples whose true class is a class of the toy model, and 0.875 over all eight, the eighth counting as an error because its true class is not in the toy model. Other hardware or library versions can give slightly different numbers. demo/reports/index.html links the eight report pages (docs/REPORTS.md); the report step warns that the interpretation text has no entry for five of the toy model's eight classes, whose pages then use the class name and a neutral sentence.

tests/test_pipeline.sh runs scripts/run_sample.py on synthetic reads with modified-base calls, from BAM files to the report, and tests/test_karyotype_call.sh calls copy number on synthetic cohorts; their scripts show those parts on data as well.

## Getting the model

The model described in the article, version 0.4.0, is archived on Zenodo under the digital object identifier 10.5281/zenodo.MODEL_RECORD (https://doi.org/10.5281/zenodo.MODEL_RECORD), with a checksum for every file. Until the article is published, access is restricted to its editors and reviewers, who receive a link with the submission. The record holds the model folder, sparsh_model_0.4.0, of several gigabytes, mostly the five networks, under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/). After unpacking, `sha256sum -c SHA256SUMS` in the model folder checks every file. docs/MODEL_CARD.md describes the model, its training data, its performance in cross-validation and its limits.

## Classifying nanopore samples

SPARSH reads nanopore runs whose reads carry modified-base calls for 5-methylcytosine and 5-hydroxymethylcytosine at CpGs (the MM and ML tags of the BAM files), aligned to GRCh37 or not yet aligned. It needs a GRCh37 reference FASTA with chromosomes 1 to 22, X and Y, indexed with `samtools faidx`, and Illumina's manifest of the Infinium MethylationEPIC v2.0 array, EPIC-8v2-0_A2.csv, from Illumina's support website. When the reads are already aligned, the reference should be the FASTA they were aligned to, with the same sequence names and lengths; otherwise every file is aligned again.

scripts/run_sample.py takes one sample from its BAM files to its report page: it aligns the reads to GRCh37 when needed, summarizes their methylation calls per CpG with modkit, makes the model's input, predicts, calls copy number from the same reads, and writes the report. A stopped run goes on where it stopped. docs/PIPELINE.md describes it.

```bash
python scripts/run_sample.py --sample S01 --bam /data/run1/bam_pass/barcode01 --out_dir results/S01 \
    --reference hg19.fasta --manifest EPIC-8v2-0_A2.csv --model_dir /path/to/sparsh_model_0.4.0 \
    --panel cn_panel.tsv.gz --sex F --blast_percent 80
```

The run writes the report page to <out_dir>/<sample>.report.html, the predictions to <out_dir>/prediction/predictions.csv and the copy-number call to <out_dir>/copy_number/calls/. With reads that are not yet aligned and no dorado, give --aligner minimap2. --panel is a copy-number reference that a laboratory makes once from at least ten of its own samples, sequenced the way its new samples are: the read counts of each sample run this way are in <out_dir>/copy_number/counts/, and gathered into one folder they are the input of `scripts/karyotype.py panel --counts_dir` (docs/PIPELINE.md). Without a panel, --no_panel calls copy number with GC correction only, as a first look.

The steps can also be run on their own, for many samples at once. The model reads one CSV row per sample with a methylation call, 1 or 0, for every array CpG that the reads cover. scripts/bedmethyl_to_csv.py makes it from the output of `modkit pileup --modified-bases 5mC 5hmC --combine-mods --cpg`, with the probe positions of the manifest, adding the calls of the two strands of each CpG:

```bash
modkit pileup --ref hg19.fasta --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed
python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE_pileup.bed --manifest EPIC-8v2-0_A2.csv \
    --sample SAMPLE --out samples/SAMPLE.csv
python scripts/predict.py --model_dir /path/to/sparsh_model_0.4.0 --ont_dir samples --output_dir predictions/batch1
python scripts/report.py --predictions_dir predictions/batch1 --model_dir /path/to/sparsh_model_0.4.0 \
    --output_dir reports/batch1
```

predictions.csv gives, for every sample, the probability of each class, the most probable classes and the reported call. A subtype is reported when its probability reaches 0.90, the threshold stored with the model. When no subtype does, the call falls back to a family of related subtypes and then to a lineage, if their summed probability reaches the threshold. Samples with fewer than 1% of the model's CpGs covered get probabilities but no reported call. The reports are self-contained HTML pages, one per sample, with an index page. docs/NANOPORE.md and docs/REPORTS.md give the details.

## Copy-number karyotype

The reads of the same run give read depth along the genome. scripts/karyotype.py counts reads in 100 kb bins, streaming each BAM file once. It then corrects each sample for GC content and divides it by the median profile of the other samples of the cohort, or, for a single new sample, by a stored reference made once from such a cohort. Finally it segments the genome, fits integer copy numbers with the leukemic fraction and the baseline ploidy, and writes the result in ISCN 2024 notation for shallow sequencing, for example `sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4`. It also writes a karyotype line (`55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21`), flags for lesions that matter in acute leukemia (high hyperdiploidy, a doubled hypodiploid clone, intrachromosomal amplification of chromosome 21, the losses that count toward acute myeloid leukemia with myelodysplasia-related changes, loss of CDKN2A), and figures of the genome and of each chromosome for review by eye. With `--karyotype_dir`, the report shows all of this next to the classification. docs/COPY_NUMBER.md describes the method and its limits.

## Training a model

The training data are a pandas DataFrame of array beta values, one row per array, with an ANNOTATION column for the class. configs/recipe_v0.4.0.args holds the recipe of the model described in the article, and scripts/train.py reads it with a leading @. The recipe leaves out two classes of that model's training data by name and dilutes with a class named Normal_Control_BM, so for other data give --exclude_classes (with the classes to leave out, or none) and --normal_class after it:

```bash
python scripts/check_data.py --data_path arrays.pkl --output_dir runs/data_checks --exclude_prefixes MPAL AML_NOS B-ALL_NOS
python scripts/train.py @configs/recipe_v0.4.0.args --data_path arrays.pkl --output_dir runs/my_model \
    --groups_file runs/data_checks/groups.csv --exclude_classes --normal_class NORMAL_MARROW_CLASS
```

Training runs a nested five-fold cross-validation. Each training array is presented as a simulated nanopore sample, in which a random fraction of its CpGs is reduced to single methylation calls, and a leukemia array is first diluted with normal marrow half of the times it is drawn. The five fold networks, each calibrated by temperature scaling on its own inner validation split, form the deployed ensemble. Every outer fold is scored once, under the same simulated conditions in every run, so runs can be compared directly. docs/TRAINING.md describes the recipe, the outputs of a run and the analyses that go with it. The training arrays and their labels are not part of this repository; the article lists the public datasets they come from.

## Repository layout

```
configs/     class names, colors, label merges, reporting hierarchy, interpretation text, training recipe
data/        reading the training arrays and the nanopore samples
models/      the network, the simulation of nanopore sparsity and of dilution, loading a trained run
training/    nested cross-validation and reproducibility
evaluation/  metrics, evaluation conditions, the reporting hierarchy, shown class names
karyotype/   read counting, normalization, segmentation, copy-number fit and ISCN nomenclature
report/      the per-case HTML reports
scripts/     command-line tools, among them run_sample.py, the whole pipeline for one sample
jobs/        batch jobs for a PBS cluster; jobs/settings.sh holds the paths
envs/        the conda environment
tests/       synthetic data and a test for every part
docs/        documentation
```

## Citation

If you use SPARSH, please cite the article that describes it. Until it is published, please cite this software; CITATION.cff has the details.

## License

The code is under the MIT License; see LICENSE. The model files on Zenodo are under the Creative Commons Attribution 4.0 International License (see Getting the model).
