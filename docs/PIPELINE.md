# From reads to report for one sample

scripts/run_sample.py runs every step of SPARSH for one sample: it takes the sample's nanopore reads as BAM files, aligned or not, and ends with the report page. It is meant for the routine case, a run or a barcode of a run, and it can be started again where it stopped.

It needs the GRCh37 reference genome the laboratory aligns to, with its .fai index; Illumina's manifest of the MethylationEPIC v2.0 array (EPIC-8v2-0_A2.csv); the folder of the model; and a stored copy-number reference made from a cohort sequenced the same way (see below). The tools are samtools, modkit, and, for reads that are not yet aligned, dorado or minimap2. The laboratory's MethPipe pipeline runs modkit 0.6.1 and dorado 1.3.0; the versions used are recorded, and others are pointed out.

## Running it

```bash
python scripts/run_sample.py --sample S01 --bam /data/run1/bam_pass/barcode01 --out_dir results/S01 \
    --reference hg19_all.fasta --manifest EPIC-8v2-0_A2.csv --model_dir /models/v04_labels \
    --panel cn_panel.tsv.gz --sex F --blast_percent 80
```

--bam takes one or more BAM files, or folders of them. A folder is read for the files ending in .bam directly inside it, leaving out files of failed reads, which MinKNOW marks with _fail_ in their names, so the barcode folder of a run's bam_pass, with its hourly files, can be given as it is. The sex (F or M) and the blast percentage are optional. When given, they go into the sample's sample_sheet.csv, which the copy-number call uses to set the sex chromosomes and to keep the leukaemic fraction within 20 points of the blast percentage, and which the report shows. --threads sets the processors for alignment, sorting and modkit (8 by default). The sample name may hold letters, digits, dots, hyphens and underscores, and is shown in capitals in the outputs, as everywhere in SPARSH.

## The steps

Each step writes into its own folder of --out_dir, and a step that has finished is not run again. A run that stopped, for example at the walltime of a batch job, therefore goes on from the step it was in when it is started again with the same command. --redo with the name of a step runs that step and the ones after it again.

The first step, bam, makes one sorted and indexed file of the reads, <sample>.bam. Files already aligned to the reference, which means their header lists the same sequences with the same lengths as the reference's .fai, are merged and sorted; a single such file that is already sorted is linked rather than copied. Any other file is first returned to unaligned reads in their original orientation (samtools reset, primary records only), and all the reads are then aligned again with dorado aligner, as the laboratory's pipeline does, keeping the tags that carry the modified-base calls. --aligner minimap2 aligns with samtools fastq and minimap2 -y instead, using the reference's .map-ont.mmi index when it exists. The number of records in the result is checked against the input before the step counts as finished.

The second step, methylation, runs modkit pileup with the command of the laboratory's MethPipe pipeline,

```bash
modkit pileup --ref hg19_all.fasta --modified-bases 5mC 5hmC --combine-mods --cpg SAMPLE.bam SAMPLE_pileup.bed
```

and turns the pileup into the model's input with scripts/bedmethyl_to_csv.py (next section). The pileup is kept, compressed.

The third step, prediction, runs scripts/predict.py on the sample's CSV file (docs/NANOPORE.md). It runs on the CPU unless --device says otherwise; one sample takes seconds.

The fourth step, copy_number, counts the reads in 100 kb bins, primary alignments with a mapping quality of 20 or more, and calls copy number against the stored reference given by --panel (docs/COPY_NUMBER.md). --no_panel instead calls with GC correction only, which is a first look and is marked as such. A sample too shallow to call gets no karyotype, and the report then has no copy-number section.

The last step, report, writes the report page with scripts/report.py (docs/REPORTS.md) and copies it to <out_dir>/<sample>.report.html.

## The methylation matrix

The model reads one row per sample with a call of 1 or 0 for every array probe whose CpG the reads cover (docs/NANOPORE.md). scripts/bedmethyl_to_csv.py makes that row from the pileup in the way the laboratory's MethPipe pipeline does, and on test data its files are identical to MethPipe's, byte for byte.

modkit reports every CpG with reads, once for each strand, with 5-methylcytosine and 5-hydroxymethylcytosine counted together as modified. The probes come from the manifest: the probe identifier from the column IlmnID, the chromosome from the column CHR, and the position from MAPINFO_37, the GRCh37 position (the column MAPINFO of the v2.0 manifest is the GRCh38 one). Each probe stands for the single base at that position, the C of its CpG on the forward strand, and takes the pileup row that starts there. A probe gets 1 when 50% or more of the calls in that row are modified, and 0 otherwise. The reverse-strand row of the same CpG starts one base later, on the G, and is not used.

The columns are named by the probe identifier up to its first underscore, so the replicate probes of one CpG (cg25324105_BC11 and cg25324105_TC21, for example) give two columns with the same name and the same value, and scripts/predict.py combines them.

`--strands both` adds the calls of the two strands of each CpG instead, and calls 1 above one half, 0 below, and leaves out a CpG whose calls are split exactly in half. It roughly doubles the reads behind each call, but MethPipe uses the forward strand alone, so it is not the default; it is there to be evaluated.

## A stored copy-number reference

A cohort is called against the median profile of its other samples. A single new sample has no cohort, so the reference of a cohort is stored once and used for every new sample:

```bash
python scripts/karyotype.py panel --counts_dir counts --sample_sheet samples.csv --out cn_panel.tsv.gz
```

The panel holds, for every bin size, the median and spread of the cohort's GC-corrected profiles bin by bin, the number of samples behind each bin, and the cohort's thresholds for reading the sex from chromosome Y. It holds no profile of a single sample and no sample names. It must come from samples prepared and sequenced the way the new samples are, since it is there to remove the biases of the library chemistry, the sequencer and the aligner; a few dozen samples or more give a steady one, and the command refuses fewer than ten. Samples of the panel should be called with the cohort, not against the panel, because a sample in the panel is part of its own reference.

## The run record

run.log holds every command with its output, in order, and a note of each step. run_info.json holds the inputs, the settings, the paths and versions of the tools, the version of SPARSH, and for each step its status, the time it took and what it found, such as the number of probes with a call or the bin size of the copy-number call. The folder of a finished sample is self-contained apart from the BAM file of the bam step, which is a link when the input was used as it was.

## On a cluster

jobs/run_sample.pbs runs the command as a batch job. The paths of the reference, the manifest, the model, the panel and the tools are set once in jobs/settings.sh, and `bash jobs/check_settings.sh` checks them. Each sample is one job:

```bash
qsub -v SAMPLE=S01,BAM=/data/run1/bam_pass/barcode01,OUT_DIR=/results/S01,SEX=F,BLAST=80 jobs/run_sample.pbs
```

Several BAM files or folders are separated by colons in BAM. A job stopped by its walltime goes on from where it stopped when it is submitted again with the same values. Most of the time goes into alignment, when it is needed, and into modkit.
