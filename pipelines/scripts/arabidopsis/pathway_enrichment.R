# MS peaks to pathways (mummichog, MetaboAnalystR) of one ranked peak list, as the MetaboAnalyst
# web server's Functional Analysis runs it on an uploaded peak list (the code behind the server).
#
# The list ("m.z" "rt", from enrichment_inputs.py) has no p-values, so MetaboAnalyst keeps its
# order and takes the first min(n / 10, 500) peaks as the significant peaks; all peaks are the
# reference. Version 2 (empirical compounds), 100 permutations (seeded), pathways of at least 3
# compounds.
#
#   Rscript pathway_enrichment.R peaks.txt lib.qs out.csv mode ppm primary_ion rt
#
# lib.qs: a mummichog library (e.g. ath_kegg.qs); mode: positive, negative or mixed; ppm: mass
# tolerance; primary_ion: yes or no; rt: seconds, minutes or no (no: ignore the retention times).
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 7)
peaks <- normalizePath(args[1])
lib <- normalizePath(args[2])
out <- file.path(normalizePath(dirname(args[3])), basename(args[3]))
mode <- args[4]
ppm <- as.numeric(args[5])
primary_ion <- args[6]
rt <- args[7]

suppressMessages(library(MetaboAnalystR))
# MetaboAnalystR writes its files to the working directory and reads the library <name>.qs there
work <- tempfile("mummichog_")
dir.create(work)
setwd(work)
lib_name <- tools::file_path_sans_ext(basename(lib))
file.symlink(lib, paste0(lib_name, ".qs"))

check <- function(m) {
  if (!is.list(m)) stop(paste(GetErrMsg(), collapse = "\n"))
  m
}
mSet <- InitDataObjects("mass_all", "mummichog", FALSE)
mSet <- SetPeakFormat(mSet, "mpt")
mSet <- UpdateInstrumentParameters(mSet, ppm, mode, primary_ion, 0.02)
mSet <- check(Read.PeakListData(mSet, peaks))
mSet <- SetRTincluded(mSet, rt)
mSet <- check(SanityCheckMummichogData(mSet))
mSet <- SetPeakEnrichMethod(mSet, "mum", "v2")
mSet <- check(SetMummichogPval(mSet, 0))
mSet <- check(PerformPSEA(mSet, lib_name, "current", 3, 100))
cat(mSet$msgSet$check.msg, sep = "\n")
file.copy("mummichog_pathway_enrichment_mummichog.csv", out, overwrite = TRUE)
