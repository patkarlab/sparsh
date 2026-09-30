#!/usr/bin/env Rscript
# Differentially methylated regions of each class against all other arrays, with DMRcate.
#
#   Rscript figures/dmr.R --input FIGURES/regions_input --out FIGURES/regions
#
# Options (defaults in brackets): --lambda kernel bandwidth in bases [1000], --C kernel scaling [2], --fdr false
# discovery rate below which a CpG is individually significant [0.05], --min_cpg_delta smallest difference in mean
# beta for a significant CpG [0.1], --min_cpgs smallest region [2], --classes a comma-separated subset of classes.
#
# The input folder is written by `python scripts/figures.py markers`: classes.tsv, and one table per class with
# probe, chrom, pos (hg19), t, p, fdr and delta for the autosomal CpGs, from Welch's test of the class against all
# other arrays on M-values. For every class, DMRcate (Peters and colleagues, Epigenetics and Chromatin 2015)
# smooths the t statistics along the genome with a Gaussian kernel and joins neighbouring significant CpGs into
# regions. A CpG counts as individually significant when its false discovery rate is below --fdr and its mean
# beta differs by at least --min_cpg_delta; with thousands of arrays very small differences reach significance,
# and DMRcate sets its smoothed cut-off from the number of individually significant CpGs.
#
# Written into --out, per class: <class>.dmrs.tsv with chrom, start, end, no.cpgs, min_smoothed_fdr, Stouffer,
# HMFDR, Fisher, maxdiff and meandiff (DMRcate's measures; maxdiff and meandiff are differences in beta), and
# sessionInfo.txt with the versions used. Genes and figures come afterwards from
# `python scripts/figures.py regions`.

suppressPackageStartupMessages({
  library(DMRcate)
  library(GenomicRanges)
})

opt <- list(input = NA, out = NA, lambda = "1000", C = "2", fdr = "0.05", min_cpg_delta = "0.1", min_cpgs = "2",
            classes = "")
args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("--help", "-h")) || length(args) == 0) {
  lines <- readLines(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
  cat(sub("^# ?", "", lines[grepl("^#", lines)][-1]), sep = "\n")
  quit(status = 0)
}
i <- 1
while (i <= length(args)) {
  key <- sub("^--", "", args[i])
  if (!key %in% names(opt) || i == length(args)) stop("unknown option or missing value: ", args[i])
  opt[[key]] <- args[i + 1]
  i <- i + 2
}
if (is.na(opt$input) || is.na(opt$out)) stop("--input and --out are required")
lambda <- as.numeric(opt$lambda)
kernel_c <- as.numeric(opt$C)
fdr <- as.numeric(opt$fdr)
min_delta <- as.numeric(opt$min_cpg_delta)
min_cpgs <- as.integer(opt$min_cpgs)

classes <- read.delim(file.path(opt$input, "classes.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
if (nzchar(opt$classes)) {
  wanted <- strsplit(opt$classes, ",")[[1]]
  unknown <- setdiff(wanted, classes$class)
  if (length(unknown)) stop("not in classes.tsv: ", paste(unknown, collapse = ", "))
  classes <- classes[classes$class %in% wanted, ]
}
dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)
columns <- c("chrom", "start", "end", "no.cpgs", "min_smoothed_fdr", "Stouffer", "HMFDR", "Fisher", "maxdiff",
             "meandiff")

for (k in seq_len(nrow(classes))) {
  cls <- classes$class[k]
  table <- read.delim(gzfile(file.path(opt$input, classes$file[k])), stringsAsFactors = FALSE)
  significant <- table$fdr < fdr & abs(table$delta) >= min_delta
  out_file <- file.path(opt$out, sub("\\.tsv\\.gz$", ".dmrs.tsv", classes$file[k]))
  if (sum(significant) < 2) {
    message(sprintf("%s: %d individually significant CpGs, no regions", cls, sum(significant)))
    write.table(setNames(data.frame(matrix(nrow = 0, ncol = length(columns))), columns), out_file, sep = "\t",
                quote = FALSE, row.names = FALSE)
    next
  }
  ranges <- GRanges(table$chrom, IRanges(table$pos, table$pos), stat = table$t, diff = table$delta,
                    rawpval = table$p, ind.fdr = table$fdr, is.sig = significant)
  names(ranges) <- table$probe
  ranges <- sort(ranges)
  annotated <- new("CpGannotated", ranges = ranges, betas = matrix(numeric(0), 0, 0))
  regions <- suppressMessages(dmrcate(annotated, lambda = lambda, C = kernel_c, min.cpgs = min_cpgs))
  coords <- extractCoords(regions@coord)
  result <- data.frame(chrom = coords$chrom, start = as.integer(as.character(coords$chromStart)),
                       end = as.integer(as.character(coords$chromEnd)), no.cpgs = regions@no.cpgs,
                       min_smoothed_fdr = regions@min_smoothed_fdr, Stouffer = regions@Stouffer,
                       HMFDR = regions@HMFDR, Fisher = regions@Fisher, maxdiff = regions@maxdiff,
                       meandiff = regions@meandiff, check.names = FALSE)
  write.table(result, out_file, sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("%s: %d significant CpGs, %d regions -> %s", cls, sum(significant), nrow(result), out_file))
}
writeLines(capture.output(sessionInfo()), file.path(opt$out, "sessionInfo.txt"))
