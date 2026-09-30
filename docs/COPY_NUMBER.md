# Copy-number karyotype

A low-pass nanopore run sequenced for methylation also measures read depth along the whole genome. At the depth of such a run, from well under one to a few times the length of the genome, that is enough to see gains and losses of whole chromosomes and chromosome arms and large segmental changes. scripts/karyotype.py turns the aligned reads into a copy-number profile, fits integer copy numbers, and writes the result as ISCN 2024 notation for shallow sequencing, as a karyotype line, and as flags for the copy-number lesions that matter in acute leukaemia, with figures to check the calls by eye. The report of scripts/report.py shows all of this next to the methylation class (docs/REPORTS.md).

The copy-number result rests on read depth alone. It does not see balanced translocations or inversions, copy-neutral loss of heterozygosity, or deletions smaller than a few bins, such as most deletions of IKZF1. Its ploidy and leukaemic fraction are estimates that need confirmation. Like the rest of SPARSH it is research software, and a karyotype, fluorescence in situ hybridisation or a chromosomal microarray remains the test.

## Steps

The work is split into steps so that the alignments are read only once.

```bash
# match sample names to their BAM files, from an object-store listing or from file paths
python scripts/karyotype.py manifest --names names.txt --listing bams.tsv --out manifest.tsv

# count reads per 100 kb bin, streaming each sample's BAM files once
bash jobs/karyotype_count.sh manifest.tsv counts 3

# call copy number for all counted samples
python scripts/karyotype.py call --counts_dir counts --out_dir copy_number --sample_sheet samples.csv
```

The sample sheet is optional. It is a CSV file with a column sample and, where known, sex (F or M) and blast_percent (0 to 100); for a sample the sheet does not list, the sex is inferred from the reads and the leukaemic fraction is left free.

The manifest step accepts a sample's BAM file when its name, without .bam, is the sample name, or the sample name followed by a separator and a suffix, as for a run written in several parts (SAMPLE_0.bam, SAMPLE_1.bam). Looser matches are listed for review rather than accepted, since short names are often contained in longer ones. Check the manifest before counting.

jobs/karyotype_count.sh streams the BAM files of each sample through samtools into the counting step, several samples at a time. Files in an object store are fetched with the command in FETCH (by default `aws s3 cp`), so the BAM files never need to be stored locally. A sample is finished when its <sample>.counts.json exists, and finished samples are skipped, so the script can be stopped and started again. Counting takes a few minutes per sample, most of it spent on the transfer and decompression of the BAM files. jobs/karyotype_count.pbs and jobs/karyotype_call.pbs run the two steps as batch jobs.

`scripts/karyotype.py call --no_reference` gives a first look at one or a few samples before a cohort is counted. It corrects for GC content only, so artefacts of mappability and of the library remain, and chromosome Y is not assessed.

## Counting reads

A read is counted once, in the 100 kb bin that holds the midpoint of its alignment, when it is a primary alignment with a mapping quality of at least 20 and not a duplicate or a read that failed quality checks. Reads are counted rather than aligned bases. For a given depth, the noise of a bin depends on how many reads it holds, and weighting reads by their length would let a few long reads dominate. This matters for libraries made with transposase-based fragmentation, whose reads are shorter and more variable in length than those of ligation libraries. The count file also records the number of reads and their lengths, from which the depth over the genome is worked out for the report.

The 100 kb bins are fixed for each genome build (GRCh37 and GRCh38) and shipped in karyotype/data with their GC content. A bin is excluded when more than a tenth of its sequence is unknown, or when it overlaps a centromere, the heterochromatin of 1q12, 9q12 and 16q11.2, the short arm of an acrocentric chromosome, the pseudoautosomal regions, or the X-transposed region. The X-transposed region is about 4 Mb of Yp11.2 that is 99% identical to Xq21.3, where reads from one chromosome can be placed on the other.

## From counts to a profile

The counts of each sample are added up into analysis bins of 0.5, 1, 2 or 5 Mb, the smallest size at which the median autosomal bin holds at least 60 reads (--min_reads). The counting noise of a bin falls with the square root of its reads, so deeper samples get finer bins. Each sample is then corrected for GC content: the median read density of the autosomal bins is taken within narrow classes of GC content, a cubic curve is laid through those medians, and every bin is divided by the value of the curve at its GC content.

What GC content does not explain, such as mappability, repeats and the preferences of the library chemistry and the aligner, is shared by every sample sequenced the same way. It is removed by dividing each sample by a reference made from the cohort: the median, bin by bin, of the GC-corrected profiles of the other samples, always leaving out the sample being called. No panel of healthy donors is needed, and the reference carries the biases of the same library chemistry and sequencer, which a panel sequenced another way would not. The median is robust to the aberrations of the cohort as long as no bin is changed in most samples, which holds for a mixed leukaemia cohort. Bins with a very low reference, or whose spread across the cohort is much larger than usual, are not used. A reference needs at least five other samples, and a few dozen samples prepared and sequenced the same way give a much steadier one. --reference_samples restricts the reference to a list of samples, for example normal marrows when there are enough of them.

The sex chromosomes need the constitutional sex of each sample. It is taken from the sample sheet when given. Otherwise it is inferred from the read density on the usable part of chromosome Y relative to the autosomes, about one half in men and near zero in women. Where the two groups lie depends on the aligner and the library, so the threshold is set from the cohort itself, midway between its two groups on a logarithmic scale, and from fixed values when the cohort shows no two clear groups. A sample near the threshold is decided by chromosome X and gets a warning, because loss of the Y chromosome in the leukaemic cells, or loss of an X, makes such a sample hard to read.

## Segments and copy numbers

Each chromosome is segmented by circular binary segmentation (Olshen and colleagues, Biostatistics 2004). A split is kept when the difference in means is significant at a level of 0.001, judged against the largest statistic that noise alone would give over all possible splits, so that long chromosomes are not split more easily than short ones. Single outlying bins are first pulled towards their neighbours, and neighbouring segments that turn out not to differ are merged again.

The segment means are then explained by integer copy numbers. If a fraction p of the cells is leukaemic, the median bin has m copies in those cells, and the normal cells carry their constitutional n copies, a segment with c copies in every leukaemic cell is expected at a log2 ratio of log2((p c + (1 - p) n) / (p m + 2 (1 - p))), plus a small offset of the whole profile. The baseline m is 2 for most leukaemias, 1 for a near-haploid clone, and 3 or 4 for near-triploid and near-tetraploid ones. Unlike a fit that pins the ploidy at two, every combination of a baseline of 1 to 4 copies and a leukaemic fraction of 0.10 to 1.00 is tried.

The cost of a combination is the sum over segments of the squared distance from the nearest expected level, in standard errors, with no single segment adding more than 25, so that one artefact cannot decide the fit. To this are added a fixed cost for every segment two or more copies away from the baseline of its chromosome (its constitutional number scaled to the ploidy of the clone, so that a tetrasomy in a near-triploid reading is one step, as a trisomy is in a diploid one), a cost for explaining a segment as a one-copy change in part of the cells (a subclone), and a preference for a near-diploid baseline. Among fits that explain the profile about equally well, the one that needs the smallest changes is taken. A segment counts as changed only when its mean lies at least four standard errors from its normal level, the error being the counting noise of its bins plus an allowance of 0.05 in log2 ratio for systematic noise. A blast percentage in the sample sheet keeps the leukaemic fraction within 20 points of it.

Read depth gives copy numbers only relative to each other, so it cannot always tell solutions apart. A near-haploid clone and the same clone after doubling give the same profile. More generally, another baseline with another leukaemic fraction can explain the same segments, and only the preference for small changes separates the two. When more than half of the genome is gained, for example, a high hyperdiploid genome of about 60 chromosomes with 88% leukaemic cells gives the same profile as a high hypodiploid one of about 40 chromosomes with 60%. Without a blast percentage the preference for a diploid baseline chooses the second; with a blast percentage near 88% the first is chosen, because the second then needs a leukaemic fraction that fits the profile worse. The best fit at every other baseline within a cost of 40 is therefore considered. A reading on the other side of diploidy (hypodiploid against hyperdiploid), or of another ploidy class within a cost of 6, is described with its chromosome count, karyotype line and leukaemic fraction, named in a warning and shown in the report under the other ploidies that fit. A blast percentage in the sample sheet usually decides between such readings, and a DNA index by flow cytometry or a karyotype decides in any case. The ploidy class, the baseline and the leukaemic fraction are marked as computed from read depth, to be confirmed from the figures.

## Nomenclature

A chromosome whose copy number is the same over at least 90% of its analysed length is called as a whole. Otherwise each arm that is uniform over 90% of its length is called at arm level, and any other stretch of at least three analysis bins and 1 Mb that differs from its chromosome or arm is a segmental event. On the acrocentric chromosomes 13, 14, 15, 21 and 22 only the long arm is analysed. The arm table gives, for every arm, its majority copy number and the share of the arm that copy number covers.

The ISCN description follows ISCN 2024 for shallow genome sequencing ("sseq"), written as for microarrays: the build in brackets, only abnormal copy numbers, the sex chromosomes first, whole chromosomes with the same copy number grouped, and the proportion of cells in brackets for a change present in part of them. Segments give the bands at their ends and their coordinates. For example:

```
sseq[GRCh37] (X)x3,(4,6,10,14,17,18)x3,(21)x4
sseq[GRCh37] (7)x1,5q14.3q33.1(83000001_150000000)x1,(8)x3
sseq[GRCh37] 17p13.3p11.2(1_22200000)x1[0.4]
```

A normal result lists the full complement, (1-22,X)x2 for a woman and (1-22)x2,(X,Y)x1 for a man. The karyotype line gives the number of chromosomes and the whole-chromosome gains and losses as a karyotype would, for example 55,XX,+X,+4,+6,+10,+14,+17,+18,+21,+21. It is written against the ploidy level of its ploidy class, as acute leukaemias are usually described: near-haploid genomes against the haploid level (<1n>), near-triploid and near-tetraploid genomes against the triploid (<3n>) and tetraploid (<4n>) levels, and everything from 31 to 67 chromosomes against the diploid level, so that a high hyperdiploid leukaemia of 61 chromosomes is written with its gains, 61,XX,+X,+2,+4,... ISCN's own ranges would write 58 to 67 chromosomes against the triploid level, as losses. The line cannot show structural changes, balanced rearrangements or how the extra material is arranged.

The ploidy class follows the modal number: near-haploid (24 to 30 chromosomes), low hypodiploid (31 to 39), high hypodiploid (40 to 43), near-diploid (44 and 45), diploid (46), low hyperdiploid (47 to 50), high hyperdiploid (51 to 67), near-triploid (68 to 79) and near-tetraploid (80 to 103).

The flags restate the profile in the terms used for acute leukaemia. They cover high hyperdiploidy with its gained chromosomes, near-haploid and low hypodiploid genomes, and the pattern of a doubled hypodiploid clone, in which every gained chromosome has four copies and none has three. They also cover the pattern of intrachromosomal amplification of chromosome 21 (three or more levels on chromosome 21 with five or more copies of RUNX1) and the losses of 5q, 7 or 7q, 11q, 12p, 13q and 17p that count towards acute myeloid leukaemia with myelodysplasia-related changes when found by karyotype. Other flags cover high hypodiploid, near-triploid and near-tetraploid genomes, amplification of RUNX1 without the pattern of chromosome 21, an isochromosome 17q pattern, changes on three or more chromosomes, a monosomal pattern, loss at 9p21.3 including CDKN2A, loss of the Y chromosome, a leukaemic fraction below 0.30, at which changes in part of the leukaemic cells may be missed, and a profile without any change. Each flag is written out as a sentence with its caveats and, where one applies, the test that confirms it.

## Outputs

Every sample's files carry the upper-case sample name that scripts/predict.py uses. <SAMPLE>.karyotype.json holds the ISCN description, the karyotype line, the ploidy class, the fit with the other ploidies that fit, the events, arms, flags and warnings, the quality measures and the sequencing summary. <SAMPLE>.segments.tsv lists the segments with their mean log2 ratio, copy number and fraction of cells, <SAMPLE>.arms.tsv every arm with its majority copy number and the share of the arm it covers, and <SAMPLE>.bins.tsv.gz the analysis bins with their log2 ratio. <SAMPLE>.karyotype.svg and .png hold the genome-wide figure, and <SAMPLE>.chromosomes.svg and .png one panel per chromosome with its bands. For the cohort, karyotype_summary.csv has one row per sample, and reference_<size>.tsv.gz files hold the references used.

In the figures, grey points are bins and lines are segments, red for gains and blue for losses. The log2 ratios are drawn with zero at two copies in the leukaemic cells, at the fitted fraction and baseline, so gains lie above the line and losses below it whatever the ploidy; the files keep the ratios as computed, centred on the median autosomal bin. The right-hand axis of the genome-wide figure marks the levels expected for 0 to 8 copies, so that a segment can be read against it. The chromosome panels show the same data chromosome by chromosome, with the bands beneath, the centromere as a dashed line, the other levels of 1 to 4 copies as dotted lines, and the changed arms named in each panel. The ISCN description is not printed on the figures; it is in the JSON file and the report.

The quality measures are the bin size, the median reads per bin, the bin-to-bin noise and, from these, the smallest fraction of cells in which a one-copy gain of 5 Mb and of 20 Mb would be called. A sample with fewer than 30 reads per bin at the largest bin size is marked low depth, and one with a noise above 0.35 in log2 ratio is marked noisy.

## Comparing with ichorCNA

`scripts/karyotype.py wig --counts_dir counts --out_dir wig` writes the same counts as WIG files in the format of readCounter, at 1 Mb by default. ichorCNA (Adalsteinsson and colleagues, Nature Communications 2017) can then be run on the reads SPARSH used, with the GC and mappability files of the build, to compare the two calls.

## Testing

tests/test_karyotype_count.sh builds small BAM files and checks the manifest matching and the counting through a stand-in object store. tests/test_karyotype_call.sh calls a synthetic cohort with known answers: normal female and male samples, trisomy 21, high hyperdiploidy, a profile with myelodysplasia-related losses, a near-haploid clone, intrachromosomal amplification of chromosome 21, a subclonal loss of 17p, loss of Y and a very shallow sample.
