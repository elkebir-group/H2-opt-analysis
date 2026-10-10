# lme4 heritability of each feature: feature ~ 1 + (1 | group) + (1 | each other design column),
# h2 = group variance / (sum of all variance components, residual included). This matches the
# "group variance over total variance" form of h2opt's ANOVA and Henderson3 estimators.
#
# The input .csv has the design columns (read as factors) and the features (all other columns).
#
#     Rscript lme4_heritability.R input.csv group "env1,env2" out.csv 8
#
# Arguments: input, group column, other design columns (comma separated, "" for none), output,
# cores (parallel::mclapply).

suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
input <- args[1]
group <- args[2]
others <- if (args[3] == "") character(0) else strsplit(args[3], ",")[[1]]
output <- args[4]
cores <- as.integer(args[5])

data <- read.csv(input, check.names = FALSE)
design <- c(group, others)
for (column in design) data[[column]] <- factor(data[[column]])
features <- setdiff(names(data), design)
terms <- paste0("(1 | `", design, "`)", collapse = " + ")

h2 <- parallel::mclapply(features, function(feature) {
  model <- lmer(as.formula(paste0("`", feature, "` ~ 1 + ", terms)), data = data,
                control = lmerControl(calc.derivs = FALSE,
                                      check.conv.singular = .makeCC("ignore", tol = 1e-4)))
  components <- as.data.frame(VarCorr(model))
  components$vcov[components$grp == group] / sum(components$vcov)
}, mc.cores = cores)
write.csv(data.frame(feature = features, h2 = unlist(h2)), output, row.names = FALSE)
cat(length(features), "features,", sum(is.na(unlist(h2))), "failed\n")
