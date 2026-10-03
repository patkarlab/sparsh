# SPARSH

Subtype Prediction via Adaptive Recognition from Sparse Hematological data

SPARSH (स्पर्श, Sanskrit for "touch") classifies acute leukemia from the DNA methylation calls of a low-pass nanopore genome run. A neural network trained on Illumina methylation arrays reads whichever CpGs the reads happen to cover and reports the lineage and genetic subtype of the leukemia with a probability calibrated on simulated nanopore samples. From the same reads, SPARSH estimates copy number along the genome and writes it as a karyotype, and it puts both into one report page for each case.

SPARSH is research software. It is not a medical device, has not been cleared or approved by any regulatory authority, and must not be used for clinical decisions. Every result needs confirmation by established diagnostic methods.

## Contents

1. [Overview](#overview)
2. [How SPARSH works](#how-sparsh-works)
3. [Performance in cross-validation](#performance-in-cross-validation)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [Predicting with our model](#predicting-with-our-model)
7. [Training on your own data](#training-on-your-own-data)
8. [Repository layout](#repository-layout)
9. [Documentation](#documentation)
10. [Citation](#citation)
11. [License](#license)

## Overview

SPARSH 1.0.0 runs the model described in the article, version 0.4.0. The model:

- reads one methylation call, 1 or 0, for each of its 354,551 CpGs (CpGs of Illumina's MethylationEPIC v2.0 array) that the reads cover, made with modkit from reads aligned to GRCh37, with 5-methylcytosine and 5-hydroxymethylcytosine counted together;
- knows 32 classes: 13 of acute myeloid leukemia, 10 of B-lymphoblastic leukemia, 5 of T-lymphoblastic leukemia, 2 of acute leukemia of ambiguous lineage, blastic plasmacytoid dendritic cell neoplasm and normal bone marrow (docs/MODEL_CARD.md lists them);
- was trained on 2,845 arrays, each presented as a simulated nanopore sample covering between 2% and 95% of its CpGs;
- averages five networks, each calibrated, and reports a subtype when its probability reaches 0.90, falling back to a family of related subtypes and then to a lineage when no subtype does;
- classifies on an ordinary processor; a graphics processing unit is needed only to train a full-size model.

The same reads give a copy-number karyotype: gains and losses of whole chromosomes and arms, and large segmental changes, in the notation of the International System for Human Cytogenomic Nomenclature (ISCN 2024), with flags for lesions that matter in acute leukemia. scripts/run_sample.py runs all of it for one sample in a single command, from its BAM files to its report page.

## How SPARSH works

### From arrays to sparse reads

The model learns from arrays, which measure every CpG, and is used on low-pass nanopore runs, which give a call only where reads fall:

```
TRAINING: a methylation array                 USE: a low-pass nanopore run
a beta value at every CpG                     a call where reads fall, nothing elsewhere

CpG 1     0.82  ████████████████              CpG 1     1
CpG 2     0.07  █                             CpG 2     .
CpG 3     0.91  ██████████████████            CpG 3     .
CpG 4     0.44  █████████                     CpG 4     0
CpG 5     0.63  ████████████                  CpG 5     .
...                                           ...
354,551 CpGs, all measured                    the CpGs that reads cover, one call each
```

Training bridges the two by turning each array into a simulated nanopore sample every time it is drawn:

```
an array of the training set: a beta value at every CpG
        │
        │   half of the times it is drawn, a leukemia array is first mixed with
        │   a normal marrow array at a blast fraction between 20% and 100%
        ▼
a coverage between 2% and 95% is chosen; each CpG receives a Poisson number of
reads, and each read is methylated with the probability given by the beta value
        │
        ▼
each covered CpG becomes one call: 1 when most of its reads are methylated,
0 when most are not, and 0 or 1 at random when they are split evenly;
every other CpG is missing
        │
        ▼
the same kind of input as a nanopore sample
```

A nanopore sample is turned into the same kind of input. modkit reports the calls of both strands of every CpG with reads, and scripts/bedmethyl_to_csv.py adds them for each probe of the array: 1 when more than half of them are modified, 0 when fewer than half are, and no value when they are split exactly in half.

### The network

```
354,551 inputs: 1, 0 or missing
        │
        ▼
encoding: each call minus 0.5 (1 → 0.5, 0 → -0.5), missing → 0,
all divided by the share of CpGs observed
        │
        ▼
Linear(354,551 → 1,024) → BatchNorm1d → GELU → Dropout(0.45)
        │
        ▼
Linear(1,024 → 512) → BatchNorm1d → GELU → Dropout(0.45)
        │
        ▼
Linear(512 → 256) → BatchNorm1d → GELU → Dropout(0.45)
        │
        ▼
Linear(256 → 32), divided by the network's temperature, softmax
        │
        ▼
average of the five networks' probabilities
        │
        ▼
reported call: a class at 0.90, else a family, else a lineage
```

The division by the share observed keeps the input to the first layer at the same scale at every coverage. Each of the five networks comes from one fold of a nested cross-validation, and its temperature was fitted on that fold's inner validation split. Almost all of the model's size, about 1.45 gigabytes per network, is the first layer, which connects every CpG to each of 1,024 units.

### One sample, from reads to report

```
BAM files of one sample, aligned or not
        │
        ▼
bam/           align to GRCh37 when needed; merge, sort and index
        │
        ▼
methylation/   modkit pileup at CpGs; one call, 1 or 0, for each array CpG with reads
        │
        ▼
prediction/    the five networks; the class probabilities and the reported call
        │
        ▼
copy_number/   reads of the same BAM file in bins of 100,000 bases; GC correction;
        │      comparison with the stored reference; segments; copy-number fit
        ▼
report/        one HTML page for the case, with the call and the karyotype
```

## Performance in cross-validation

Every training array was scored once, by the network of the fold that had held it out, on simulated nanopore reads (docs/MODEL_CARD.md):

- At 30% coverage, undiluted, the accuracy was 0.935 and the balanced accuracy (the mean recall over classes) 0.929. The most probable class reached 0.90 for 85.1% of the arrays, and 98.0% of those calls were correct.
- Averaged over coverages from 5% to 90%, the balanced accuracy was 0.918.
- With the leukemia arrays diluted to 30% blasts, at 30% coverage, the balanced accuracy was 0.804, and 38.9% of the arrays reached the threshold.

These figures describe single networks on simulated reads. The performance on nanopore samples is reported in the article.

## Dependencies

### Software

- 64-bit Linux, with bash 4 or later for the batch jobs and the tests (macOS does not have it by default).
- Python 3.10 to 3.12 with PyTorch 2.0 or later, NumPy 1.24 or later, pandas 2.0 or later, SciPy 1.10 or later, scikit-learn 1.3 or later and matplotlib 3.7 or later. Optuna 3.6 or later is needed only by scripts/tune.py.
- samtools 1.17 or later (https://www.htslib.org), to merge, sort and index reads, to return aligned reads to unaligned ones (samtools reset), and to count reads for the copy number.
- modkit (https://github.com/nanoporetech/modkit), for the methylation calls. The article's samples were processed with modkit 0.6.1, and envs/sparsh.yml installs that version.
- For reads that are not yet aligned to the reference: dorado (https://github.com/nanoporetech/dorado), tested in version 1.3.0, or minimap2 2.24 or later (https://github.com/lh3/minimap2).
- Optionally a PBS batch system, for the jobs in jobs/.

envs/sparsh.yml installs all of these except dorado; see [Installation](#installation). Versions that SPARSH has run with:

- The tests pass on Ubuntu 24.04 (x86-64) with the versions that envs/sparsh.yml installed on October 2, 2026: Python 3.12, PyTorch 2.13 (built for the processor), NumPy 2.5, pandas 3.0, SciPy 1.18, scikit-learn 1.9, matplotlib 3.11, Optuna 5.0, samtools 1.24, modkit 0.6.1 and minimap2 2.31.
- The model was trained on a cluster running Red Hat Enterprise Linux 8.10 (x86-64), on one NVIDIA H200, with Python 3.11.14, PyTorch 2.5.1 built for CUDA 12.1, NumPy 2.4.0, pandas 2.3.3 and scikit-learn 1.8.0, as environment.json in the model folder records. The samples of the article were classified on the same cluster, which also had SciPy 1.16.3, matplotlib 3.11.2 and Optuna 4.6.0, and their methylation calls were made with modkit 0.6.1 and samtools 1.24.

### Reference files

These are not part of the repository; download them once.

- The human reference genome GRCh37 (hg19) as one FASTA file with its samtools index. The samples of the article were aligned to the 24 main chromosomes of UCSC's hg19, chr1 to chr22, chrX and chrY, without the mitochondrial genome, the alternative haplotypes and the unplaced sequences. These commands make that file, hg19_main.fa, byte for byte:

  ```bash
  wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
  gunzip hg19.fa.gz
  samtools faidx hg19.fa
  samtools faidx -n 50 hg19.fa chr{1..22} chrX chrY > hg19_main.fa
  md5sum hg19_main.fa                                            # 1831aa6238818b0686686125b51b91e5
  samtools faidx hg19_main.fa
  minimap2 -x map-ont -d hg19_main.fa.map-ont.mmi hg19_main.fa   # only for --aligner minimap2
  rm hg19.fa hg19.fa.fai
  ```

  Another GRCh37 FASTA with chromosomes 1 to 22, X and Y, named with or without "chr", works as well. Reads that are already aligned are used as they are only when they were aligned to the same FASTA, with the same sequence names and lengths; otherwise they are aligned again. Use one reference for all samples and for the copy-number panel. The minimap2 index next to the FASTA saves building it again for every sample.
- Illumina's manifest of the Infinium MethylationEPIC v2.0 array, EPIC-8v2-0_A2.csv, from the Infinium MethylationEPIC v2.0 Product Files (https://support.illumina.com/downloads/infinium-methylationepic-v2-0-product-files.html). SPARSH takes the probes' GRCh37 positions from it.
- The model folder, sparsh_model_0.4.0, from Zenodo (see [Getting the model](#getting-the-model)).
- For the copy-number karyotype of single samples, a copy-number panel, which a laboratory makes once from its own samples (see [The copy-number karyotype](#the-copy-number-karyotype)).

The repository itself carries the GC content of the bins of 100,000 bases, the chromosome bands and the gene positions of GRCh37 and GRCh38 (karyotype/data), and the class hierarchy, the shown class names, the interpretation text, the label merges and the training recipe (configs/).

### Input

Nanopore reads as BAM files with modified-base calls for 5-methylcytosine and 5-hydroxymethylcytosine at CpGs (the MM and ML tags that dorado writes when basecalling with a modified-base model), aligned to the reference or not yet aligned, such as the barcode folder of a run's bam_pass.

### Hardware

- Classifying: the five networks take about 8 gigabytes of memory. A graphics processing unit is optional; with one, scripts/predict.py loads the networks onto it, which with a batch of samples needs about 9 gigabytes of its memory, so give --device cpu when the card has less than about 10 gigabytes. Loading the networks takes from one to a few minutes, and the samples of a batch then take seconds.
- The whole pipeline for one sample: on 16 processor cores, a sample of 54 hourly BAM files already aligned to the reference took about 15 minutes.
- Disk: about 7 gigabytes for the model folder (as much again for its zip), about 3 gigabytes for the reference, and room for each sample's merged BAM file.
- Training the full model: one graphics processing unit; the released recipe took about an hour on one NVIDIA H200.

## Installation

With conda (mamba and micromamba read the same file):

```bash
git clone https://github.com/patkarlab/sparsh.git
cd sparsh
conda env create -f envs/sparsh.yml
conda activate sparsh
```

envs/sparsh.yml installs Python with PyTorch, NumPy, pandas, SciPy, scikit-learn, matplotlib and Optuna from conda-forge, and samtools, modkit 0.6.1 and minimap2 from bioconda. conda-forge installs a PyTorch build with CUDA when the machine has an NVIDIA driver, and a build for the processor otherwise. With micromamba on a fast connection it took about a minute, and with the processor build of PyTorch the environment takes about 3 gigabytes; the build with CUDA is larger.

With pip, install the PyTorch build that matches your hardware (https://pytorch.org) and then the other Python packages; samtools, modkit and minimap2 are installed separately:

```bash
pip install -r requirements.txt
```

dorado is not packaged for conda. Install it from Oxford Nanopore's releases (https://github.com/nanoporetech/dorado) when you want it as the aligner. Without it, give scripts/run_sample.py --aligner minimap2 for reads that need aligning.

To check the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
samtools --version | head -1
modkit --version
bash tests/run_all.sh
```

tests/run_all.sh runs every part of SPARSH on synthetic data, in about seven minutes on two processor cores, and ends with:

```
Passed: 9  Skipped: 0  Failed: 0  (7 min)
Passed with some steps skipped (see their output): test_pipeline
ALL TESTS PASSED
```

The middle line appears when dorado or minimap2 is not installed, since the pipeline test then skips its step with that aligner. A test prints a line starting with SKIPPED when a tool it needs (samtools, modkit, Optuna) is missing. Tools that are not on the PATH can be given to the tests as variables:

```bash
SAMTOOLS=/path/to/samtools MODKIT=/path/to/modkit MINIMAP2=/path/to/minimap2 DORADO=/path/to/dorado \
    bash tests/run_all.sh
```

## Predicting with our model

### Getting the model

The model described in the article, version 0.4.0, is archived on Zenodo under the digital object identifier 10.5281/zenodo.23118416 (https://doi.org/10.5281/zenodo.23118416). Until the article is published, access to its files is restricted to the editors and reviewers, who receive a link with the submission. The record holds the model folder as one zip file of about 7 gigabytes, sparsh_model_0.4.0.zip, stored in seven parts of at most 1 gibibyte, sparsh_model_0.4.0.zip.part00 to sparsh_model_0.4.0.zip.part06, under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/). Download the seven parts into one folder, join them into the zip and unpack it:

```bash
cat sparsh_model_0.4.0.zip.part* > sparsh_model_0.4.0.zip     # joins the parts in the order of their names
unzip sparsh_model_0.4.0.zip        # or: python -m zipfile -e sparsh_model_0.4.0.zip .
cd sparsh_model_0.4.0
sha256sum -c SHA256SUMS             # every line should end in OK
cd ..
```

The folder holds:

```
sparsh_model_0.4.0/
├── config.json                   settings of the model, the training and the data;
│                                 the five networks with their temperatures; the threshold 0.90
├── class_mapping.json            the 32 classes, in the order of the network's outputs
├── label_map.json                label merges applied before training
├── selected_cpgs.json            the 354,551 CpGs, in the order of the network's inputs
├── environment.json              software versions used in training
├── fold_models/                  the five networks, fold1.pt to fold5.pt, about 1.45 gigabytes each
├── cv_metrics_by_condition.csv   cross-validation figures for every tested condition
├── cv_recall_by_class.csv        the recall of every class in cross-validation
└── SHA256SUMS                    the checksum of every file
```

Tables that list the training arrays one by one are not part of the release. The two cross-validation tables give the evidence that each report page quotes. docs/MODEL_CARD.md describes the model, its training data, its performance in cross-validation and its limitations.

### One sample, from BAM files to report

With the reference files of [Dependencies](#dependencies) in place:

```bash
python scripts/run_sample.py --sample S01 --bam /data/run1/bam_pass/barcode01 --out_dir results/S01 \
    --reference hg19_main.fa --manifest EPIC-8v2-0_A2.csv --model_dir sparsh_model_0.4.0 \
    --panel cn_panel.tsv.gz --sex F --blast_percent 80
```

- --sample: the name of the sample, starting with a letter or digit, with letters, digits, dots, hyphens and underscores.
- --bam: one or more BAM files, or folders of them, such as the barcode folder of a run's bam_pass with its hourly files. In a folder, files of failed reads, with _fail_ in their names, are left out.
- --reference, --manifest and --model_dir: the reference genome, the manifest and the model folder.
- --panel or --no_panel, one of the two: the copy-number panel, or a call with GC correction only, as a first look until you have a panel.
- --sex (F or M) and --blast_percent (above 0, at most 100): optional. The copy-number call uses them, and the report shows them.
- --aligner minimap2 when dorado is not installed; --threads (8 by default); --device (cpu by default).
- --redo STEP runs one step and the ones after it again: bam, methylation, prediction, copy_number or report.

On the synthetic reads of the test suite (tests/make_synthetic_modbam.py, with a toy genome and a toy model), run with --no_panel, the command prints:

```
S01: 2 BAM file(s) -> /path/to/results/S01
  bam          2 file(s): merged and sorted (0.0 min)
  methylation  1,422 probes with a call (0.0 min)
  prediction   coverage of the model's CpGs 47.4% (0.0 min)
  copy_number  not called (see copy_number/calls/karyotype_summary.csv) (0.0 min)
  report       S01.report.html (0.0 min)
Report: /path/to/results/S01/S01.report.html
```

The toy genome is too small for a copy-number call. On a real run most of the time goes into alignment, when it is needed, and into modkit: on 16 processor cores, a sample of 54 hourly BAM files already aligned to the reference took about 15 minutes, 2 to merge and sort, 9 for modkit and the model's input, 1 to predict and 2 for the copy number. The run writes, among other files:

```
results/S01/
├── S01.report.html       the report page
├── bam/                  S01.bam with its index: the reads merged and sorted, or a link to
│                         the input when it was a single file already sorted and aligned
├── methylation/          S01.csv, the model's input, and S01_pileup.bed.gz, modkit's pileup
├── prediction/           predictions.csv, prediction_info.json and the class hierarchy and
│                         display names used
├── copy_number/          counts/, the reads in bins, and calls/, the karyotype with its figures
├── report/               index.html, cases/S01.html and report_summary.csv
├── sample_sheet.csv      the sex and the blast percentage, when given
├── run.log               every command and its output
└── run_info.json         the inputs, settings, tool versions and the time of each step
```

A stopped run goes on where it stopped when the same command is given again. docs/PIPELINE.md describes every step.

### Many samples, step by step

The steps can also be run on their own, for many samples at once. modkit needs each sample's reads as one BAM file aligned to GRCh37, sorted and indexed, such as bam/S01.bam of scripts/run_sample.py:

```bash
modkit pileup --ref hg19_main.fa --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed
python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE_pileup.bed --manifest EPIC-8v2-0_A2.csv \
    --sample SAMPLE --out samples/SAMPLE.csv
python scripts/ont_coverage.py --run sparsh_model_0.4.0 samples
python scripts/predict.py --model_dir sparsh_model_0.4.0 --ont_dir samples --output_dir predictions/batch1
python scripts/report.py --predictions_dir predictions/batch1 --output_dir reports/batch1
```

The model reads one CSV file per sample: a header of probe identifiers and one row, named after the sample, with a call of 1 or 0 for every array CpG that the reads cover. CpGs without reads have no column. For the synthetic sample above, methylation/S01.csv begins:

```
,cg00000002,cg00000003,cg00000004,cg00000005,cg00000006,cg00000007,cg00000008,cg00000009,...
S01,0,0,1,0,1,1,1,1,...
```

Before anything is predicted, scripts/ont_coverage.py reports for each folder what share of the model's CpGs its samples cover (the median, the range and the number of samples per band) next to the range the model was trained on; --output writes one line per sample. The sample's name in the predictions is the file name without .csv, in capitals. docs/NANOPORE.md gives the details of each step.

On a cluster with the PBS batch system, jobs/run_sample.pbs and jobs/predict.pbs run these commands as batch jobs; docs/PIPELINE.md and docs/NANOPORE.md describe their settings.

### Reading the results

For every sample the model gives a probability for each of its 32 classes, and the call is made at the threshold stored with the model, 0.90:

1. When the most probable class reaches 0.90, that subtype is reported.
2. Otherwise, when the summed probability of a family of related subtypes reaches it, the family is reported, for example "AML, HOX-related (NPM1, NUP98, DEK::NUP214, UBTF or KMT2A-r)".
3. Otherwise, when the summed probability of a lineage reaches it, the lineage is reported: "B-ALL, subtype undetermined".
4. Otherwise the sample is "not callable".

A sample whose reads cover fewer than 1% of the model's CpGs gets probabilities but no call ("not callable (coverage)"), and the report flags samples below 5%. configs/class_hierarchy.json defines the families and lineages. It can also name background classes, reported as "<lineage>, no specific subtype" at the level no_subtype; model 0.4.0 has none.

predictions.csv has one row per sample, with these columns:

- sample, file, n_columns, n_matched, n_observed, coverage_pct and n_repeated_probes: the input file and the share of the model's CpGs it covers;
- prediction, confidence, top2_class, top2_prob, top3_class and top3_prob: the three most probable classes;
- callable and note: whether the most probable class reaches the threshold, and a note on samples whose coverage is below the minimum;
- family, family_prob, lineage and lineage_prob: the family and the lineage with their summed probabilities;
- reported_call and reported_level: the call and its level (subtype, family, lineage or none);
- prob_ followed by a class name: the probability of every class.

Selected columns of the predictions of the synthetic demo (see [Training on your own data](#training-on-your-own-data)), aligned for reading, show two of the levels:

```
sample  coverage_pct  prediction               confidence  lineage  lineage_prob  reported_call                reported_level
ONT05   26.66667      B-ALL_High hyperdiploid  0.66494     B-ALL    0.98295       B-ALL, subtype undetermined  lineage
ONT01   27.50000      B-ALL_ETV6-RUNX1         0.69561     B-ALL    0.82888       not callable                 none
```

In ONT05 no class reaches 0.90, but the B-lymphoblastic classes together reach 0.98, so the lineage is reported. In ONT01 the lineage reaches only 0.83, and nothing is reported.

scripts/report.py writes one self-contained HTML page per sample, written for a hematologist to read and print, and an index page that lists every sample. A page gives the call with its probability, the class probabilities, the interpretation of the class with the tests that confirm it, the copy-number karyotype when one was called, the evidence from cross-validation at a coverage like the sample's, and the input and its quality. The interpretation text comes from configs/interpretation.json and is marked as a draft until a hematopathologist has reviewed it. docs/REPORTS.md describes the pages.

### The copy-number karyotype

The reads of the same run give read depth along the genome. scripts/karyotype.py counts reads in bins of 100,000 bases, streaming each BAM file once. It then corrects each sample for GC content and divides it by the median profile of the other samples of the cohort, or, for a single new sample, by a stored reference made once from such a cohort. Finally it segments the genome, fits integer copy numbers with the leukemic fraction and the baseline ploidy, and writes the result in ISCN 2024 notation for shallow sequencing. It also writes a karyotype line, flags for lesions that matter in acute leukemia (high hyperdiploidy, a doubled hypodiploid clone, intrachromosomal amplification of chromosome 21, the losses that count toward acute myeloid leukemia with myelodysplasia-related changes, loss of CDKN2A), and figures of the genome and of each chromosome for review by eye.

On the synthetic cohort of the test suite (tests/make_synthetic_counts.py), three samples called against a panel of the cohort's 36 background samples give:

```
CASE_HYPER_F: 55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21 | fraction 0.83 | sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4
CASE_T21_M: 47,XY,+21 | fraction 0.92 | sseq[GRCh37] (21)x3
CASE_AML_F: 46,XX,-7,+8 | fraction 0.69 | sseq[GRCh37] (7)x1,(8)x3,5q14.3q33.1(85000001_150000000)x1
```

Each line gives the karyotype line, the fitted leukemic fraction and the ISCN description. karyotype_summary.csv adds the ploidy class and the flags: high_hyperdiploid for the first sample, and loss_5q, loss_7, complex and monosomal for the third.

A single new sample is called against a panel, which a laboratory makes once from at least ten samples prepared and sequenced the way its new samples will be; thirty or more give a steadier reference. To start, run the panel's samples through scripts/run_sample.py with --no_panel, gather their read counts, make the panel, and call those samples as a cohort, each against the others, rather than against the panel they belong to:

```bash
mkdir -p cn_counts
cp results/*/copy_number/counts/* cn_counts/
python scripts/karyotype.py panel --counts_dir cn_counts --out cn_panel.tsv.gz --sample_sheet samples.csv
python scripts/karyotype.py call --counts_dir cn_counts --out_dir cn_calls --sample_sheet samples.csv
```

Their report pages can then be made again with the cohort's calls, by scripts/report.py with --karyotype_dir cn_calls. New samples get --panel cn_panel.tsv.gz. The sample sheet is a CSV with a column sample and, when known, sex (F or M) and blast_percent. docs/COPY_NUMBER.md describes the method and its limits.

## Training on your own data

### A first try on synthetic data

The repository can make synthetic arrays and nanopore samples, so the whole cycle of training, predicting and reporting can be tried without data of your own, for example to check an installation. These commands train a toy model of eight classes and 3,000 CpGs, which demonstrates the commands and is not the model described in the article, and classify eight synthetic nanopore samples, in under a minute on two processor cores:

```bash
python tests/make_synthetic.py --output_dir demo/data
python scripts/train.py --data_path demo/data/train.pkl --output_dir demo/model --exclude_prefixes MPAL \
    --n_folds 3 --epochs 60 --hidden_dims 128 64 --device cpu \
    --train_sim binary --val_sim binary --eval_sims binary --eval_coverages 0.05 0.1 0.2 0.3 0.5
python scripts/predict.py --model_dir demo/model --ont_dir demo/data/ont --output_dir demo/predictions \
    --ground_truth demo/data/ground_truth.csv --device cpu
python scripts/report.py --predictions_dir demo/predictions --output_dir demo/reports
```

Training ends with the outer folds scored under each simulated condition (binary_0.05 is 5% of the CpGs covered). Accuracy rises with coverage, and so does the share of samples whose most probable class reaches 0.90:

```
  condition  accuracy  balanced_accuracy  top2_accuracy  callable_share_0.90  accuracy_callable_0.90   ece
      dense     1.000              1.000          1.000                1.000                   1.000 0.000
binary_0.05     0.410              0.444          0.635                0.004                   1.000 0.027
binary_0.10     0.624              0.637          0.817                0.070                   0.970 0.083
binary_0.20     0.841              0.880          0.947                0.473                   0.973 0.055
binary_0.30     0.930              0.929          0.985                0.745                   0.991 0.025
binary_0.50     0.992              0.994          1.000                0.953                   0.998 0.014
```

Prediction and the reports then print, among other lines:

```
Model: demo/model (ensemble, 3 network(s), 3000 CpGs, 8 classes)
Predicted 8 samples (0 unreadable); coverage median 23.4% (range 20.1-27.6%); callable at >= 0.90: 0
Reported at: subtype 0, no_subtype 0, family 0, lineage 4, none 4
...
Reports for 8 samples: subtype 0, no_subtype 0, family 0, lineage 4, none 4
Written to demo/reports: index.html, cases/ (8 pages), report_summary.csv
```

The toy model trains only briefly, so its probabilities stay low. Its most probable class was right for all seven samples whose true class is one of its classes, but none reached 0.90, and its calls stay at the level of the lineage or below the threshold. demo/predictions/summary_metrics.json gives these scores against demo/data/ground_truth.csv: an accuracy of the most probable class of 1.0 for those seven, and 0.875 over all eight, the eighth counting as an error because its true class is not in the toy model. Other hardware or library versions can give slightly different numbers. demo/reports/index.html links the eight report pages; the report step warns that the interpretation text has no entry for five of the toy model's classes, whose pages then use the class name and a neutral sentence.

### Training data

The training data are one pickle file holding a pandas DataFrame with one row per array: beta values between 0 and 1 in columns named by their probe identifiers (cg...), the class in the column ANNOTATION, the array's name in Sample_ID (otherwise the index names it), and optionally its study in Source_Dataset. Missing values are allowed. The synthetic data of the first try look like this:

```
     Sample_ID  cg00000000  cg00000001  cg00000002  ...               ANNOTATION Source_Dataset
0    GSM100000       0.786       0.353       0.805  ...  B-ALL_High hyperdiploid          GSE_B
270  GSM100270       0.823       0.552       0.710  ...              AML_NUP98-r          GSE_B
405  GSM100405       0.977       0.629       0.779  ...           B-ALL_BCR-ABL1          GSE_B
```

configs/label_map.json merges raw labels into model classes before training. The training arrays and labels of the released model are not part of this repository; they come from published studies and from the developers' own patients, and the article lists the sources.

### Training

```bash
python scripts/check_data.py --data_path arrays.pkl --output_dir runs/data_checks \
    --exclude_prefixes MPAL AML_NOS B-ALL_NOS
python scripts/train.py @configs/recipe_v0.4.0.args --data_path arrays.pkl --output_dir runs/my_model \
    --groups_file runs/data_checks/groups.csv --exclude_classes --normal_class NORMAL_MARROW_CLASS
python scripts/predict.py --model_dir runs/my_model --ont_dir samples --output_dir predictions/my_model
```

scripts/check_data.py counts the classes, sets class against source study, and finds near-identical arrays, which it writes to groups.csv so that they stay on the same side of every cross-validation split. configs/recipe_v0.4.0.args holds the recipe of the released model, read with a leading @. The recipe leaves out two classes of that model's training data by name and dilutes with a class named Normal_Control_BM, so for other data give --exclude_classes (with the classes to leave out, or none) and --normal_class after it. jobs/train.pbs runs the same as a batch job on a PBS cluster (docs/TRAINING.md); the released recipe took about an hour on one NVIDIA H200 for 2,845 arrays.

Training runs a nested five-fold cross-validation. Within each outer fold, an inner validation split of about one seventh sets early stopping and the temperature of the fold's network, and the outer fold is scored once, under the same simulated conditions in every run, so runs can be compared directly. The five fold networks form the deployed model, which scripts/predict.py and scripts/run_sample.py take as --model_dir like the released one. A run writes, among other files:

```
runs/my_model/
├── config.json, class_mapping.json, label_map.json, selected_cpgs.json, environment.json
├── fold_models/                     the five networks
├── cv_metrics_by_condition.csv      out-of-fold figures for every condition, and the same
│                                    before calibration in cv_metrics_by_condition_uncalibrated.csv
├── cv_recall_by_class.csv           the recall of every class under every condition
├── cv_predictions_<condition>.csv   every array's out-of-fold probabilities
├── cv_confusion_dense.csv           confusion matrices of the arrays as measured and, in
│                                    cv_confusion_binary_0.20.csv, at 20% coverage
├── folds.csv, folds_summary.csv     every array's fold; each fold's best epoch and temperature
├── training_history.csv, training_curves.png
└── training_log.txt
```

Other tools support training: scripts/compare_runs.py sets finished runs side by side, condition by condition; scripts/tune.py runs a bounded hyperparameter study with Optuna; scripts/label_audit.py lists training arrays whose methylation disagrees with their label; scripts/hierarchy_report.py shows what the family and lineage fallback adds; and scripts/score_excluded.py shows how the classes left out of training would be reported. docs/TRAINING.md describes the recipe, the outputs of a run and these analyses.

## Repository layout

```
configs/     class names, colors, label merges, reporting hierarchy, interpretation text, training recipe
data/        reading the training arrays and the nanopore samples
models/      the network, the simulation of nanopore reads and of dilution, loading a trained model
training/    nested cross-validation, calibration and reproducibility
evaluation/  metrics, evaluation conditions, the reporting hierarchy, shown class names
karyotype/   read counting, normalization, segmentation, copy-number fit and ISCN nomenclature
report/      the per-case HTML reports
scripts/     command-line tools; run_sample.py runs the whole pipeline for one sample
jobs/        batch jobs for a PBS cluster; jobs/settings.sh holds the paths
envs/        the conda environment
tests/       synthetic data and a test for every part
docs/        documentation
```

## Documentation

- docs/PIPELINE.md: scripts/run_sample.py, step by step.
- docs/NANOPORE.md: from reads to the model's input, prediction, and scoring against known diagnoses.
- docs/REPORTS.md: the report pages.
- docs/COPY_NUMBER.md: the copy-number method and its limits.
- docs/TRAINING.md: the training recipe, the outputs of a run and the analyses of runs.
- docs/MODEL_CARD.md: the released model, its training data, its performance in cross-validation and its limitations.
- CHANGELOG.md: the changes of each version.

## Citation

If you use SPARSH, please cite the article that describes it. Until it is published, please cite this software; CITATION.cff has the details. If you use the model, please cite its Zenodo record as well, 10.5281/zenodo.23118416. Questions and problems can be raised as issues at https://github.com/patkarlab/sparsh/issues.

## License

The code is under the MIT License; see LICENSE. The model files on Zenodo are under the Creative Commons Attribution 4.0 International License (see [Getting the model](#getting-the-model)).
