# Training

This document describes how a SPARSH model is trained from methylation arrays, the recipe of the released model, the files a training run writes and the analyses that go with them.

## Training data

The training data are one pickle file holding a pandas DataFrame with one row per array. The CpG columns are named by their probe IDs (cg...) and hold beta values between 0 and 1, and missing values are allowed and kept as missing. The class of each array is in the column ANNOTATION. A column Sample_ID names the arrays (otherwise the index does), and an optional Source_Dataset column records the study each array comes from.

Before training, configs/label_map.json merges raw labels into model classes; a run saves the copy it used, and the same map is applied to ground-truth files when nanopore samples are scored. Arrays can be left out by listing their Sample_IDs in a text file (--exclude_ids), and whole classes by name (--exclude_classes) or by the start of their name (--exclude_prefixes). Classes with fewer than five arrays are dropped. Related arrays, such as two arrays of one patient or a repeat of the same sample, should be kept on the same side of every cross-validation split. A CSV with the columns Sample_ID and group (--groups_file) does this.

## Preparing the data

scripts/check_data.py looks at the data before any training. It writes the class counts after the label map, a table of class by source dataset (a class that comes from a single study can be confounded with it), the missing values of every source, and each array's most correlated other array. Pairs above a correlation threshold are likely the same sample or patient; they are written as groups for cross-validation.

scripts/make_training_pickle.py writes a new pickle with per-sample label changes, and optionally with whole classes renamed, leaving the input untouched and recording every change. scripts/merge_groups.py joins groupings, for example the duplicate clusters of check_data.py with a list of arrays per patient. scripts/label_audit.py looks for arrays whose methylation disagrees with their label. It uses two independent signals, the labels of an array's nearest neighbors in a principal component space and, when a training run is given, the out-of-fold probability of its own class, and writes a table of arrays to review. The training labels of the released model were reviewed with this audit together with independent genetic data (RNA sequencing, fusion calls, karyotypes and mutations, where available).

## The recipe of the released model

configs/recipe_v0.4.0.args lists the settings of the released model that differ from the defaults of scripts/train.py. The file is read with a leading @, and any option given after it overrides it. The recipe leaves out two classes and dilutes with a class named Normal_Control_BM, so for training data without these classes give --exclude_classes (with the classes to leave out, or none) and --normal_class after it:

```bash
python scripts/train.py @configs/recipe_v0.4.0.args --data_path arrays.pkl --output_dir runs/my_model \
    --exclude_ids exclude.txt --groups_file groups.csv --exclude_classes --normal_class NORMAL_MARROW_CLASS
```

The last two options are for data other than those of the released model: --exclude_classes without a class empties the recipe's list of classes left out, and --normal_class names the class of normal marrow used for dilution.

A nanopore run at low depth covers a random part of the array CpGs, and at each covered CpG it gives the calls of one or a few reads. Training therefore presents every array as such a sample. For every array, each time it is drawn, a coverage is chosen uniformly between 2% and 95%. Every CpG then receives a Poisson number of reads, with a mean chosen so that this share of the CpGs has at least one read, and each read is methylated with the probability given by the array's beta value. A covered CpG is reported as one call: 1 when most of its reads are methylated, 0 when most are not, and 0 or 1 at random when they are split evenly (models/corruption.py). The other CpGs are missing. The network receives each call minus 0.5, so that 1 becomes 0.5 and 0 becomes -0.5, with missing CpGs set to zero, and divided by the fraction of CpGs observed. This keeps the expected input to its first layer the same at every coverage.

A diagnostic array usually comes from a sample with many blasts, while a nanopore sample may have far fewer. Each leukemia array is therefore, half of the times it is drawn, first mixed with a normal bone marrow array at a blast fraction drawn between 20% and 100%, before the reads are drawn (models/dilution.py). The label stays that of the leukemia, and normal marrows are never mixed.

The network is a feed-forward classifier with three hidden layers of 1,024, 512 and 256 units, each with batch normalization, Gaussian error linear unit (GELU) activation and dropout of 0.45 (models/sparse_nn.py). It is trained with a focal loss (gamma 2) with label smoothing of 0.05 and a sampler that puts five arrays of every class into each batch. The optimizer is AdamW with a learning rate of 1e-4 and a weight decay of 1e-4, for up to 300 epochs. Classes whose names start with MPAL, AML_NOS or B-ALL_NOS are left out, as are acute myeloid leukemia with myelodysplasia-related changes and acute myeloid leukemia with MECOM rearrangement (AML-MR and AML_MECOM-r); configs/interpretation.json lists what the model does not contain.

## Cross-validation and the deployed model

Training is a nested five-fold cross-validation (training/trainer.py). The outer folds are stratified by class, and grouped when groups are given. Within the training part of each outer fold, about one seventh of the arrays (--inner_val_frac 0.15) is held out as an inner validation split, presented as simulated nanopore samples at several coverages and, in the released recipe, diluted in the same way as training. The inner split alone sets the learning-rate schedule, early stopping (after 30 epochs without improvement), the checkpoint that is kept and the temperature that calibrates the network's probabilities. The outer fold is never used for any of these choices and is scored once.

The five fold networks are the deployed model. scripts/predict.py averages their calibrated probabilities. `--final_model` trains one more network with the same recipe on all arrays but an inner split of about one seventh, which it keeps for early stopping and calibration.

The outer folds are scored under fixed conditions. These are the array as measured (dense), and simulated nanopore samples at each coverage given by --eval_coverages, undiluted and diluted to each blast fraction given by --eval_blasts. The corruption of every array depends only on its Sample_ID, the condition and a fixed seed, so every run is scored on identical inputs, and the differences between runs are differences between models. In the diluted conditions the normal marrow mixed into an array comes from the same outer fold, so there the inputs are identical only between runs that share their folds. Condition names give the simulation, the dilution and the coverage, for example binary_0.30 or binary-blast30_0.30 (evaluation/conditions.py).

## What a run writes

Everything a run needs later goes into its output folder, which must be new or empty.

config.json holds the settings of the model, the training and the data, the inference settings (the fold networks and their temperatures, the threshold) and a record of every fold. class_mapping.json, label_map.json and selected_cpgs.json give the classes in the order of the network's outputs, the label merges used and the CpGs in the order of its inputs. The fold networks are in fold_models/, and the extra network, when trained, in model.pt. folds.csv lists every array with its class and outer fold, and folds_summary.csv each fold's best epoch, temperature and running time. training_history.csv and training_curves.png follow the losses epoch by epoch, and environment.json and training_log.txt record the software versions and the log.

cv_metrics_by_condition.csv gives, for every condition, the accuracy and balanced accuracy (the mean recall over classes), the macro F1 score, the share of arrays whose class is among the two most probable, the share called at the threshold and the accuracy of those calls, the expected calibration error and the negative log-likelihood, from the calibrated out-of-fold probabilities. cv_metrics_by_condition_uncalibrated.csv gives the same before calibration. cv_recall_by_class.csv gives the recall of every class, cv_predictions_<condition>.csv every array's out-of-fold probabilities, and cv_confusion_dense.csv the confusion matrix of the arrays as measured, with a second one for the primary condition (--primary_condition, by default the undiluted condition of the first --eval_sims nearest 20% coverage, which is binary_0.20 with the recipe).

## Analyses of runs

scripts/compare_runs.py sets finished runs side by side, condition by condition. With the coverage of a set of nanopore samples, from the CSV file that scripts/ont_coverage.py writes with --output, it interpolates each metric to the coverage those samples actually have; only their coverage is used, never their labels. It also gives the largest share of arrays that can be called, most confident first, while at least 98% of the calls remain correct. That share does not depend on calibration, so it compares how well each model knows when it is right. Differences of one or two points between single runs are within the spread between seeds, and decisions that depend on them need runs on several seeds.

scripts/tune.py runs a bounded hyperparameter study with Optuna over six settings: learning rate, weight decay, dropout, label smoothing, focal gamma and the hidden layer sizes. Every trial is a full nested cross-validation of a base run's recipe, scored on its out-of-fold predictions at 10, 30 and 50% coverage, undiluted and at 50% and 30% blasts. The first trial is the base recipe itself. The study behind the released model found no setting that raised this objective by one point, the margin set before the study, when repeated on three further seeds.

scripts/hierarchy_report.py shows what the family and lineage fallback adds, from a run's out-of-fold predictions. It also lists the class pairs the model confuses most, the evidence for grouping classes into families (configs/class_hierarchy.json). scripts/score_excluded.py presents the arrays of the classes left out of training to a run's networks as simulated nanopore samples. The result shows how a leukemia the model does not contain would be reported.

## Reproducibility

Every run sets the seeds of Python, NumPy and PyTorch and asks PyTorch for deterministic kernels (training/reproducibility.py). Runs on the same hardware and software agree closely, but PyTorch does not guarantee bitwise identity for every operation. The batch jobs set PYTHONHASHSEED before Python starts. jobs/train.pbs trains with the recipe file on a PBS cluster, reading the paths from jobs/settings.sh; a run of the released recipe on 2,845 arrays took about an hour on one NVIDIA H200 graphics processing unit.
