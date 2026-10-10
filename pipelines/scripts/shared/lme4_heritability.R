# lme4 heritability of each feature, in the model of h2opt's estimators: environment terms fixed,
# the group and its subgroups random, h2 = group variance / variance of the feature (the
# "group variance over total variance" form of h2opt's ANOVA and Henderson3 estimators):
#     feature ~ 1 + fixed terms + (1 | group) + (1 | each random term)
# Missing values (NA) are left out per feature.
#
# The input .csv has the design columns (read as factors) and the features (all other columns).
#
#     Rscript lme4_heritability.R input.csv group "plot" "rep,column" out.csv 8
#
# Arguments: input, group column, other random columns and fixed columns (each comma separated,
# "" for none), output, cores (parallel::mclapply).

suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
split_terms <- function(text) if (text == "") character(0) else strsplit(text, ",")[[1]]
input <- args[1]
group <- args[2]
random <- split_terms(args[3])
fixed <- split_terms(args[4])
output <- args[5]
cores <- as.integer(args[6])

data <- read.csv(input, check.names = FALSE)
design <- c(group, random, fixed)
for (column in design) data[[column]] <- factor(data[[column]])
features <- setdiff(names(data), design)
terms <- paste(c("1", paste0("`", fixed, "`"), paste0("(1 | `", c(group, random), "`)")),
               collapse = " + ")

h2 <- parallel::mclapply(features, function(feature) {
  rows <- data[!is.na(data[[feature]]), ]
  model <- lmer(as.formula(paste0("`", feature, "` ~ ", terms)), data = rows,
                control = lmerControl(calc.derivs = FALSE,
                                      check.conv.singular = .makeCC("ignore", tol = 1e-4)))
  components <- as.data.frame(VarCorr(model))
  components$vcov[components$grp == group] / var(rows[[feature]])
}, mc.cores = cores)
write.csv(data.frame(feature = features, h2 = unlist(h2)), output, row.names = FALSE)
cat(length(features), "features,", sum(is.na(unlist(h2))), "failed\n")
