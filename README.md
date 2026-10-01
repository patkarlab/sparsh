# SPARSH

SPARSH classifies acute leukaemia from sparse DNA methylation data. A neural network trained on Illumina methylation arrays reads the CpG methylation calls of a low-pass nanopore genome run, in which only a small and random part of the array CpGs is covered, and reports the lineage and genetic subtype of the leukaemia with a calibrated probability. From the same nanopore run, SPARSH also estimates copy number along the genome and writes it as an ISCN karyotype. It then puts both into a report for each case.

SPARSH stands for Subtype Prediction via Adaptive Recognition from Sparse Haematological data.

SPARSH is research software. It is not a medical device, has not been cleared or approved by any regulatory authority, and must not be used for clinical decisions. Every result needs confirmation by established diagnostic methods.

## What is in this repository

The classifier is in data/, models/, training/ and evaluation/, with scripts/train.py to train it and scripts/predict.py to apply it to nanopore samples. The released model, version 0.4.0, has 32 classes of acute myeloid leukaemia, B- and T-lymphoblastic leukaemia, acute leukaemia of ambiguous lineage, blastic plasmacytoid dendritic cell neoplasm and normal bone marrow; docs/MODEL_CARD.md describes it. The copy-number karyotype is in karyotype/ with scripts/karyotype.py, the per-case report in report/ with scripts/report.py, and the figures of the training arrays in figures/ with scripts/figures.py. The documents in docs/ describe each part, and jobs/ has batch jobs for a PBS cluster.

## Installation

SPARSH needs Python 3.10 to 3.12 with PyTorch, NumPy, pandas, SciPy, scikit-learn and matplotlib. Working from a sample's reads also needs samtools and modkit, and dorado or minimap2 for reads that are not yet aligned; the copy-number step needs samtools; and the region analysis of the figures needs R with DMRcate (envs/figures_r.yml).

```bash
git clone https://github.com/patkarlab/sparsh.git
cd sparsh
conda env create -f envs/sparsh.yml
conda activate sparsh
bash tests/run_all.sh
```

The tests run every part on synthetic data, on the CPU, in about ten minutes, and each prints a line ending in PASSED. A GPU is only needed for training.

## Classifying nanopore samples

scripts/run_sample.py takes one sample from its BAM files, aligned or not, to its report page: it aligns the reads to GRCh37 when needed, summarises their methylation calls per CpG with modkit, makes the model's input, predicts, calls copy number from the same reads, and writes the report. A stopped run goes on where it stopped. docs/PIPELINE.md describes it.

```bash
python scripts/run_sample.py --sample S01 --bam /data/run1/bam_pass/barcode01 --out_dir results/S01 \
    --reference hg19_all.fasta --manifest EPIC-8v2-0_A2.csv --model_dir /path/to/v04_labels \
    --panel cn_panel.tsv.gz --sex F --blast_percent 80
```

The steps can also be run on their own, for many samples at once. The model reads one CSV row per sample with a methylation call, 1 or 0, for every array CpG that the reads cover. scripts/bedmethyl_to_csv.py makes it from the output of `modkit pileup --modified-bases 5mC 5hmC --combine-mods --cpg`, with the probe positions of Illumina's manifest of the MethylationEPIC v2.0 array, as the laboratory's MethPipe pipeline makes it:

```bash
modkit pileup --ref hg19_all.fasta --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed
python scripts/bedmethyl_to_csv.py --bedmethyl SAMPLE_pileup.bed --manifest EPIC-8v2-0_A2.csv \
    --sample SAMPLE --out samples/SAMPLE.csv
python scripts/predict.py --model_dir /path/to/v04_labels --ont_dir samples --output_dir predictions/batch1
python scripts/report.py --predictions_dir predictions/batch1 --model_dir /path/to/v04_labels \
    --output_dir reports/batch1
```

Here /path/to/v04_labels is the folder of the released model, which is not public yet; docs/MODEL_CARD.md describes it and how it will be released. predictions.csv gives, for every sample, the probability of each class, the most probable classes and the reported call. A subtype is reported when its probability reaches 0.90. When no subtype does, the call falls back to a family of related subtypes and then to a lineage, if their summed probability reaches the threshold. Samples with fewer than 1% of the model's CpGs covered get probabilities but no reported call. The reports are self-contained HTML pages, one per sample, with an index page. docs/NANOPORE.md and docs/REPORTS.md give the details.

## Copy-number karyotype

The reads of the same run give read depth along the genome. scripts/karyotype.py counts reads in 100 kb bins, streaming each BAM file once. It then corrects each sample for GC content and divides it by the median profile of the other samples of the cohort, or, for a single new sample, by a stored reference made once from such a cohort. Finally it segments the genome, fits integer copy numbers with the leukaemic fraction and the baseline ploidy, and writes the result in ISCN 2024 notation for shallow sequencing, for example `sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4`. It also writes a karyotype line (`55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21`), flags for lesions that matter in acute leukaemia (high hyperdiploidy, a doubled hypodiploid clone, intrachromosomal amplification of chromosome 21, the losses that count towards acute myeloid leukaemia with myelodysplasia-related changes, loss of CDKN2A), and figures of the genome and of each chromosome for review by eye. With `--karyotype_dir`, the report shows all of this next to the classification. docs/COPY_NUMBER.md describes the method and its limits.

## Training a model

The training data are a pandas DataFrame of array beta values, one row per array, with an ANNOTATION column for the class. configs/recipe_v0.4.0.args holds the recipe of the released model, and scripts/train.py reads it with a leading @. The recipe names classes of the released model's training data, so for other data give --exclude_classes and --normal_class after it:

```bash
python scripts/check_data.py --data_path arrays.pkl --output_dir runs/data_checks --exclude_prefixes MPAL AML_NOS B-ALL_NOS
python scripts/train.py @configs/recipe_v0.4.0.args --data_path arrays.pkl --output_dir runs/my_model \
    --groups_file runs/data_checks/groups.csv
```

Training runs a nested five-fold cross-validation. Each training array is presented as a simulated nanopore sample, in which a random fraction of its CpGs is reduced to single methylation calls, and a leukaemia array is first diluted with normal marrow half of the times it is drawn. The five fold networks, each calibrated by temperature scaling on its own inner validation split, form the deployed ensemble. Every outer fold is scored once, under the same simulated conditions in every run, so runs can be compared directly. docs/TRAINING.md describes the recipe, the outputs of a run and the analyses that go with it.

## Figures

scripts/figures.py draws a t-SNE map of the training arrays, a heatmap of the CpGs that mark each class, the global methylation of each class, and maps of the regions that DMRcate finds differentially methylated in each class (figures/dmr.R). docs/FIGURES.md gives the methods.

## Repository layout

```
configs/     class names, colours, label merges, reporting hierarchy, interpretation text, training recipe
data/        reading the training arrays and the nanopore samples
models/      the network, the simulation of nanopore sparsity and of dilution, loading a trained run
training/    nested cross-validation and reproducibility
evaluation/  metrics, evaluation conditions, the reporting hierarchy, shown class names
karyotype/   read counting, normalisation, segmentation, copy-number fit and ISCN nomenclature
report/      the per-case HTML reports
figures/     the figures of the training arrays, with the R script for the region analysis
scripts/     command-line tools, among them run_sample.py, the whole pipeline for one sample
jobs/        batch jobs for a PBS cluster; jobs/settings.sh holds the paths
tests/       synthetic data and a test for every part
docs/        documentation
```

## Citation

If you use SPARSH, please cite the article that describes it. Until it is published, please cite this repository; CITATION.cff has the details.

## Licence

MIT; see LICENSE.
