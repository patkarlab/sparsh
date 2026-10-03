# Changelog

## 1.0.0

First public release, which accompanies the article describing SPARSH. It contains the classifier and the recipe of the model described in the article, version 0.4.0; prediction for nanopore samples; the copy-number karyotype from the same runs; the per-case reports; and scripts/run_sample.py, which runs all of it for one sample, from its BAM files to its report. The model's input is made from the modkit pileup with the calls of both strands of each CpG added, and the strand of the probe's own base alone remains an option. Copy number for a single new sample is called against a stored reference made from a cohort sequenced the same way. The model itself is archived on Zenodo (docs/MODEL_CARD.md).
