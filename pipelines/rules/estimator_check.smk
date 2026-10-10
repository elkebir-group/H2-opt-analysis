# The lme4 check of the heritability estimators (paper Fig. ANOVA): for the raw features of each
# dataset (and for the Miscanthus H2-opt traits), the pipeline's estimator (ANOVA or Henderson3) against an lme4 mixed model with the
# same design (environment fixed; group and plots random). R and lme4 come from pipelines/envs/lme4.yaml.

CHECK_OUT = OUT_ROOT / "estimator_check"
SHARED_SCRIPTS = PIPELINE_DIR / "scripts" / "shared"
CHECK_DATASETS = ["sorghum", "miscanthus-central", "miscanthus-south", "arabidopsis-0d",
                  "arabidopsis-6d", "miscanthus-traits-central", "miscanthus-traits-south"]


# lme4 model terms per dataset kind: group, other random terms, fixed (environment) terms; the
# design columns that estimator_inputs.py writes.
CHECK_TERMS = {"sorghum": ("genotype", "", "set"),
               "miscanthus": ("family", "plot", "rep,column"),
               "arabidopsis": ("accession", "", "year")}


def check_terms(wildcards):
    return CHECK_TERMS[wildcards.dataset.split("-")[0]]


def check_inputs(wildcards):
    if wildcards.dataset.startswith("miscanthus-traits"):
        population = wildcards.dataset.split("-")[2]
        return [str(MISC_OUT / population / "h2opt_hybrid" / f"date{d}.npz") for d in MISC_DATES]
    if wildcards.dataset.startswith("miscanthus"):
        return [str(OUT_ROOT / "miscanthus" / wildcards.dataset.split("-")[1] / "images.npy")]
    if wildcards.dataset.startswith("arabidopsis"):
        return [ARABI_FEATURES]
    return []


def check_input_args(wildcards, input):
    if wildcards.dataset.startswith("miscanthus-traits"):
        return "--traits " + " ".join(input)
    if wildcards.dataset.startswith("miscanthus"):
        return f"--images {input[0]} --date {MISC['selection_date']}"
    if wildcards.dataset.startswith("arabidopsis"):
        return f"--features {input[0]}"
    return ""


rule estimator_check_all:
    input:
        str(CHECK_OUT / "figures" / "ANOVA.pdf"),


rule check_estimator_inputs:
    input:
        check_inputs,
    output:
        csv=str(CHECK_OUT / "{dataset}.csv"),
        npz=str(CHECK_OUT / "{dataset}.npz"),
    log:
        str(CHECK_OUT / "{dataset}.log"),
    wildcard_constraints:
        dataset="|".join(CHECK_DATASETS),
    params:
        extra=check_input_args,
    threads: 4
    shell:
        "python -u {SHARED_SCRIPTS}/estimator_inputs.py --dataset {wildcards.dataset}"
        " {params.extra} --out-csv {output.csv} --out {output.npz} > {log} 2>&1"


rule check_lme4_heritability:
    input:
        csv=str(CHECK_OUT / "{dataset}.csv"),
        npz=str(CHECK_OUT / "{dataset}.npz"),
    output:
        str(CHECK_OUT / "{dataset}_lme4.csv"),
    log:
        str(CHECK_OUT / "{dataset}_lme4.log"),
    wildcard_constraints:
        dataset="|".join(CHECK_DATASETS),
    params:
        group=lambda wc: check_terms(wc)[0],
        random=lambda wc: check_terms(wc)[1],
        fixed=lambda wc: check_terms(wc)[2],
    threads: 8
    conda:
        "../envs/lme4.yaml"
    shell:
        "Rscript {SHARED_SCRIPTS}/lme4_heritability.R {input.csv} {params.group}"
        " '{params.random}' '{params.fixed}' {output} {threads} > {log} 2>&1"


rule check_plot_estimators:
    input:
        npz=expand(str(CHECK_OUT / "{dataset}.npz"), dataset=CHECK_DATASETS),
        lme4=expand(str(CHECK_OUT / "{dataset}_lme4.csv"), dataset=CHECK_DATASETS),
    output:
        pdf=str(CHECK_OUT / "figures" / "ANOVA.pdf"),
        png=str(CHECK_OUT / "figures" / "ANOVA.png"),
    log:
        str(CHECK_OUT / "figures" / "ANOVA.log"),
    params:
        datasets=" ".join(f"--dataset {d} {CHECK_OUT / d}.npz {CHECK_OUT / d}_lme4.csv"
                          for d in CHECK_DATASETS),
    threads: 1
    shell:
        "python -u {SHARED_SCRIPTS}/plot_estimator_check.py {params.datasets}"
        " --out {output.pdf} > {log} 2>&1"
