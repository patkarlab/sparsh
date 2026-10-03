# Model card: SPARSH model 0.4.0

## The model

The model described in the article, version 0.4.0, classifies acute leukemia from the sparse CpG methylation calls of a low-pass nanopore genome run. It was trained with configs/recipe_v0.4.0.args on 2,845 methylation arrays in 32 classes, with the seed 42, and SPARSH 1.0.0 runs it. It consists of five neural networks, one from each fold of a nested cross-validation, each calibrated by temperature scaling, and its probabilities are the average of the five. The model and its recipe were fixed, and a checksum of every file was recorded, before the model was applied to any nanopore sample.

## Intended use

The model is meant for research on the diagnosis of acute leukemia by nanopore sequencing: a diagnostic sample of a patient with known or suspected acute leukemia, sequenced at low depth with modified-base calling. It gives a methylation class with a probability calibrated on simulated nanopore samples, to be set beside the established tests that make the diagnosis. It is not a medical device, has not been cleared by any regulatory authority, and must not be used on its own for decisions about patients.

It is not meant for methylation arrays, for which other classifiers exist. Nor is it meant for samples taken after treatment or at relapse, or for other hematological neoplasms such as lymphoma, myeloma or myelodysplastic neoplasms, on which it has not been tested.

## Input and output

The input is one CSV file per sample with a methylation call, 1 or 0, for every array CpG that the reads cover, made from a modkit bedMethyl file of reads aligned to GRCh37 (docs/NANOPORE.md). The model reads the CpGs listed in its selected_cpgs.json, and CpGs without reads are treated as missing. It was trained on samples covering 2% to 95% of its CpGs and tested in cross-validation from 5% to 90%. Samples covering less than 1% get probabilities but no reported call, and the report flags samples below 5%.

The output is a probability for each of the 32 classes and a reported call at a threshold of 0.90. When no class reaches the threshold, the call falls back to a family of related subtypes and then to a lineage, using their summed probabilities (configs/class_hierarchy.json).

## Classes

Acute myeloid leukemia has thirteen classes: APL_PML-RARA, AML_RUNX1-RUNX1T1, AML_CBFB-MYH11, AML_mutated CEBPA, AML_HOXdys_NPM1_NUP98_DEK-NUP214_UBTF, AML_NPM1_IDH, AML_IDH, AML_KMT2A-r, AML_NUP98-NSD1, AML_ETV6-r, AML_FET-ETS, AML_CBFA2T3-GLIS2 and AMKL_AMKL-Like. B-lymphoblastic leukemia has ten: B-ALL_ETV6-RUNX1, B-ALL_Aneuploidy, B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like_Other, B-ALL_DUX4-r, B-ALL_KMT2A-r, B-ALL_TCF3-PBX1, B-ALL_TCF3-HLF, B-ALL_ZNF384-r, B-ALL_MEF2D-r and B-ALL_iAMP21. T-lymphoblastic leukemia has five: T-ALL_TAL1-like, T-ALL_TLX1, T-ALL_TLX3, T-ALL_NKX2 and T-ALL_HOXA_ETP. Acute leukemia of ambiguous lineage has two, AL_BCL11B-r and AL_MLLT10-r, and the remaining classes are blastic plasmacytoid dendritic cell neoplasm (BPDCN) and normal bone marrow (Normal_Control_BM). Two classes carry other names inside the model: T-ALL_HOXA_ETP is T-ALL_HOXA9_ETP, and B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like_Other is B-ALL_PAX5_BCR-ABL1_BCR-ABL1-Like (configs/display_names.json). configs/interpretation.json describes each class.

## Training data

The model was trained on 2,845 Illumina methylation arrays of acute leukemia and normal bone marrow, from published studies and from the developers' own patients; the article lists the sources. The label of each array is the subtype found by genetic testing (RNA sequencing, fusion calls, karyotype or mutations) where this was available; T-lymphoblastic arrays from studies without genetic data for each sample were labeled by clustering of their methylation. The labels were reviewed with a label audit (scripts/label_audit.py) together with the genetic data. A label was kept only when the methylation agreed with it; an array was relabeled only when its genetic data and its methylation agreed on another class, and was left out otherwise. Arrays of one patient, and near-identical arrays found by scripts/check_data.py, were kept on the same side of every cross-validation split.

Arrays of mixed-phenotype acute leukemia, of acute myeloid leukemia and B-lymphoblastic leukemia without further classification, of acute myeloid leukemia, myelodysplasia-related, and of acute myeloid leukemia with MECOM rearrangement were not used for training.

## Training

Every array is presented as a simulated nanopore sample: a random share of its CpGs, between 2% and 95%, is covered by reads drawn from its beta values, and each covered CpG is reduced to a single call. Each leukemia array, half of the times it is drawn, is first mixed with normal marrow at a blast fraction between 20% and 100%. The network is a feed-forward classifier with three hidden layers, trained with a focal loss and a sampler that balances the classes. docs/TRAINING.md describes the recipe in full.

## Performance in cross-validation

Every array was scored once, by the network of the fold that had held it out, on simulated nanopore reads. The file cv_metrics_by_condition.csv of the model gives every condition. At 30% coverage, without dilution, the accuracy was 0.935 and the balanced accuracy (the mean recall over classes) 0.929. The most probable class reached 0.90 for 85.1% of the arrays, and 98.0% of those calls were correct. Averaged over coverages from 5% to 90%, the balanced accuracy was 0.918. With the leukemia arrays diluted to 30% blasts, at 30% coverage, the balanced accuracy fell to 0.804, and 38.9% of the arrays reached the threshold.

These figures describe single networks on simulated reads. The deployed model averages five networks, which makes its probabilities somewhat more conservative, and real nanopore calls carry errors and coverage patterns that the simulation does not fully reproduce. The performance on nanopore samples is reported in the article.

## Limitations

The model knows only its 32 classes and has no answer for "none of these". A leukemia of an entity outside the model is reported as the most similar class, family or lineage, sometimes with a probability above the threshold. configs/interpretation.json lists the entities known to be outside the model, and the report names them for every acute myeloid leukemia result and every result reported at the level of a lineage.

Hypodiploid B-lymphoblastic leukemia is not a class of its own. It is merged with high hyperdiploid leukemia in B-ALL_Aneuploidy, and the few hypodiploid arrays in training fall partly into other classes. The chromosome count, from a karyotype, the DNA index or the copy-number section of the report, is needed to tell them apart.

Some classes overlap in methylation. The HOX-related class of acute myeloid leukemia (NPM1, NUP98, DEK::NUP214, UBTF) overlaps with AML_KMT2A-r, and part of the confident errors in cross-validation fall between the two. The T-lymphoblastic classes other than TAL1-like overlap with each other, and so do the two classes with IDH1 or IDH2 mutations. The families of configs/class_hierarchy.json group these classes, so that a sample split between them is still reported at the level of the family.

Performance falls as the blast percentage falls. Training simulated dilution down to 20% blasts, and cross-validation tested the model down to 30%; below that its performance is unknown. Some classes have few training arrays, so their recall is known only roughly. The training arrays come from a limited number of centers and studies, and use in other populations and laboratories needs its own validation.

## Files

The model folder holds config.json (the settings of the model, the training and the data, and the five networks with their temperatures), class_mapping.json, label_map.json, selected_cpgs.json, environment.json (the software versions used in training), the five networks in fold_models/, and the tables of cross-validation that the report uses (cv_metrics_by_condition.csv and cv_recall_by_class.csv). Tables that list the training arrays one by one are not part of the release. The folder is archived on Zenodo under the digital object identifier 10.5281/zenodo.23118416, as one zip file, sparsh_model_0.4.0.zip, stored in seven parts that are joined again with cat (README.md, Getting the model), with a checksum of every file of the folder in SHA256SUMS; until the article is published, access is restricted to its editors and reviewers. The files of the folder are licensed under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0/).
