# Per-case reports

scripts/report.py turns the output folder of scripts/predict.py into one HTML page per sample, written for a haematologist to read and print, and an index page that lists every sample. When a copy-number result exists for a sample, its page shows that as well. The pages are for research use only and say so at the top and at the foot of every page.

```bash
python scripts/report.py --predictions_dir predictions/batch1 --output_dir reports/batch1 \
    --karyotype_dir copy_number --sample_sheet samples.csv
```

## Inputs

The prediction folder is the only input that is required. From it the report reads predictions.csv and prediction_info.json, which names the model folder and the threshold of the prediction run. --model_dir names the model folder directly instead. Its config.json, cv_metrics_by_condition.csv and cv_recall_by_class.csv supply the details of the model and the evidence from cross-validation. Without the first two that section is left out, and without the third only the recall of the class is missing. Classes are shown under the names used in predictions.csv (docs/NANOPORE.md), in the colours of configs/class_colours.json.

--karyotype_dir is a folder of <SAMPLE>.karyotype.json files from scripts/karyotype.py (docs/COPY_NUMBER.md), with their figures. Sample names are matched ignoring case, the way predict.py names samples. A karyotype of a sample that is not in predictions.csv is ignored with a warning. A file that cannot be read gives a note on the page instead of stopping the report.

--sample_sheet is a CSV file with a column sample and any further columns, for example the centre, the sample type, the blast percentage or the date of collection, which the page shows as sample information. Do not put names of patients in it or anywhere else in the inputs, since sample names and the entries of the sheet are printed on the pages.

--samples limits the report to some of the samples, and --title sets the title of the pages. --threshold sets another reporting threshold. At the threshold of the prediction run, the calls of predictions.csv are used as they are. At any other threshold they are worked out again from the class probabilities, as predict.py would do, so that every page agrees with the threshold it states. The output folder must be new or empty, unless --overwrite is given.

## The case page

A case page has a header, the result, the class probabilities, the interpretation, the evidence from cross-validation, the copy-number karyotype, the input and its quality, and a note on the method, in that order.

The header names the sample, the time of the report, the SPARSH version and the model. The result panel gives the lineage, the reported result, its probability and the level at which it is reported: a subtype, a family of related subtypes, a lineage with the subtype undetermined, or nothing when the sample is not callable. For a family or a lineage the probability is the sum over its classes. Probabilities are shown with two decimals but are never rounded up to the threshold, so 0.8996 stays 0.8996 when the threshold is 0.90. Values that would round to 1.00 or 0.00 are shown as > 0.99 and < 0.01, because a rounded 1.00 reads as certainty. The chart below the panel shows the five most probable classes, with the threshold as a dashed line. It is drawn as SVG inside the page, so it stays sharp on paper.

The interpretation starts with the text written for the reported entity: its full name, what the methylation profile means, the tests that confirm it and a point to keep in mind. Then come the caveats that follow from the data of the sample. These cover a coverage below 5% of the model's CpGs, a most probable class below the threshold (indicative only, not a result), a subtype that could not be resolved within its family or lineage, and a normal marrow profile in a sample sent for leukaemia. Every acute myeloid leukaemia result, and a result of another lineage reported at lineage level, also lists the entities that the model does not contain, because a leukaemia of such an entity is reported as the most similar class, family or lineage, or not at all.

The evidence from cross-validation states how the model did on its training arrays with nanopore reads simulated at the tested coverage nearest to that of the sample. It gives the share of arrays whose most probable class reached the threshold, the share of those calls that were correct, and, for a subtype, the recall of that class. The section says plainly that these figures come from simulated reads and not from nanopore samples.

The copy-number section appears when a karyotype is given. It shows the ISCN description and the karyotype line, the modal number and ploidy class, the sex and how it was determined, and the leukaemic fraction, together with the other ploidies that fit the read depth with another leukaemic fraction. It lists the arms that changed, the flags and warnings of the copy-number step, and a table of events with their bands, size, copy number, fraction of cells and genes of interest. Below come the genome-wide figure, the chromosome panels and the quality of the profile: the bin size, the reads per bin, the noise, and the smallest fraction of cells in which a one-copy gain of 5 Mb and of 20 Mb is detected. For a result at subtype level the section also gives the copy-number note of that class from the interpretation text.

The input and quality section lists the input file, its CpG columns, how many of them are model CpGs and how many of those have a value, repeated probe IDs, and the note of predict.py when there is one. With a copy-number result it adds the number of whole-genome reads used, their length and the depth over the genome. The sample information from the sheet follows. The last section explains in a few sentences how the classifier works, and names the interpretation text, with its version, and the class hierarchy used.

## The index page and the summary table

index.html lists every sample with its coverage, reported result, level, confidence, lineage and flags (low coverage, below threshold, not callable), each linked to its page. The table sorts by any column and filters as one types, through a short script inside the page; the case pages have no script at all. report_summary.csv has one row per sample with the fields of the result panel and the first sentence of the interpretation.

## The interpretation text

configs/interpretation.json holds every sentence of the report that depends on the class. Its keys are the internal class names of the model (not the shown names), the family names of configs/class_hierarchy.json and its lineage names. A class needs a name and a summary, and may have confirm, note and copy_number_note. A family may have its own name, summary and confirmation, and a lineage has the text for a result at lineage level. levels holds the text for a sample that is not classified, and not_in_model lists, lineage by lineage, the entities the model does not contain. A class without an entry is reported under its own name with a neutral sentence. Another file can be given with --interpretation.

The file carries a version. The text shipped here was written for model version 0.4.0 and is marked as a draft. Until a haematopathologist has reviewed it and its version no longer starts with "draft", every page says that the text has not been reviewed.

## Pages that stand alone

Each page is a single HTML file with its styles, chart and figures inside it. It loads no fonts, scripts or images from elsewhere, so it can be archived, sent or opened without a network. Every value taken from an input file is escaped before it is written into a page, so a sample name or an entry of the sheet cannot change the page. The pages print on A4 with margins of 15 mm, and browsers that support page-margin boxes print the sample name, the research-use statement and the page number at the foot of every page.

tests/test_report.sh trains a tiny model on synthetic data, predicts synthetic samples, writes the reports with a sample sheet and copy-number results for two samples, and checks the pages, the summary table and the refusal of bad input.
