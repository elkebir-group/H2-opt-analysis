# The lme4 check of the heritability estimators (paper Fig. ANOVA): for the raw features of each
# dataset, the pipeline's estimator (ANOVA or Henderson3) against an lme4 mixed model with the
# same design (all design terms random). R and lme4 come from pipelines/envs/lme4.yaml.

import numpy as np

CHECK_OUT = OUT_ROOT / "estimator_check"
SHARED_SCRIPTS = PIPELINE_DIR / "scripts" / "shared"
CHECK_DATASETS = ["sorghum", "miscanthus-central", "miscanthus-south", "arabidopsis-0d",
                  "arabidopsis-6d"]


def check_inputs(wildcards):
    if wildcards.dataset.startswith("miscanthus"):
        return [str(OUT_ROOT / "miscanthus" / wildcards.dataset.split("-")[1] / "images.npy")]
    if wildcards.dataset.startswith("arabidopsis"):
        return [ARABI_FEATURES]
    return []


def check_input_args(wildcards, input):
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
        group=lambda wc, input: str(np.load(input.npz)["group"]),
        others=lambda wc, input: str(np.load(input.npz)["others"]),
    threads: 8
    conda:
        "../envs/lme4.yaml"
    shell:
        "Rscript {SHARED_SCRIPTS}/lme4_heritability.R {input.csv} {params.group}"
        " '{params.others}' {output} {threads} > {log} 2>&1"


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
