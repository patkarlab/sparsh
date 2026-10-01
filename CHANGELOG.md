# Changelog

## 1.0.0

First public release. It contains the classifier and the recipe of the released model, version 0.4.0, prediction for nanopore samples, the copy-number karyotype from the same runs, the per-case reports, and the figures of the training arrays. scripts/run_sample.py runs all of it for one sample, from its BAM files to its report; the model's input is made from the modkit pileup with the calls of both strands of each CpG added (the forward strand alone, as the laboratory's MethPipe pipeline uses it, is an option), and copy number for a single sample is called against a stored reference made from a cohort.
