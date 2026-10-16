# Changelog

## 1.1.0

A provisional level for subtypes between 0.80 and 0.90. The model, version 0.4.0, its files and its stored threshold of 0.90 are unchanged.

- When no subtype reaches the threshold of 0.90 and no family of related subtypes does, a most probable class that reaches the caution threshold, 0.80 by default, is reported as a provisional subtype (reported_level provisional), to be confirmed by other tests. A family above 0.90 is reported before a provisional subtype, and a provisional subtype before a lineage (evaluation/hierarchy.py). The level of 0.80 follows the threshold of MARLIN.
- scripts/predict.py, scripts/report.py and scripts/hierarchy_report.py take --caution_threshold; 0 leaves the provisional level out and gives the calls of 1.0.0. prediction_info.json records the caution threshold. With known diagnoses, summary_metrics.json and evaluation_by_coverage.csv also give the share and accuracy of the most probable class between the two levels and at or above 0.80.
- The report labels a provisional subtype as such on its page, in the index and in report_summary.csv, draws the caution threshold in the chart, adds a caveat, and quotes how often calls between the two levels were correct in cross-validation. Pages made from predictions of 1.0.0 have no provisional level unless --caution_threshold asks for one.
- New scripts/threshold_analysis.py: from the out-of-fold predictions of a training run, the share and accuracy of the most probable class at both levels and between them, with a receiver operating characteristic analysis and the Youden index; --write_bands records the figures between the levels in configs/cv_caution_bands.csv for the report.
- Samples already run with scripts/run_sample.py take up the change with --redo prediction, which runs the prediction, copy-number and report steps again and keeps the merged reads and the methylation input.

## 1.0.0

First public release, which accompanies the article describing SPARSH. It contains the classifier and the recipe of the model described in the article, version 0.4.0; prediction for nanopore samples; the copy-number karyotype from the same runs; the per-case reports; and scripts/run_sample.py, which runs all of it for one sample, from its BAM files to its report. The model's input is made from the modkit pileup with the calls of both strands of each CpG added, and the strand of the probe's own base alone remains an option. Copy number for a single new sample is called against a stored reference made from a cohort sequenced the same way. The model itself is archived on Zenodo (docs/MODEL_CARD.md).
