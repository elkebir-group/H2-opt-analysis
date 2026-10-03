# Sorghum hyperspectral analyses: H2-opt per genotype fold, heritability of
# H2-opt and the baselines, and GWAS of the per-line trait means.
#
# A trait source is one way to get the H2-opt traits of a fold:
#   paper   - the predictions behind the paper (raw data, not built here)
#   rmsprop - retrained with the paper's settings (train_h2opt.py)
#   lbfgs   - linear H2-opt trained to convergence with L-BFGS

from h2opt_analysis import sorghum

SORGHUM = config["sorghum"]
SORGHUM_OUT = OUT_ROOT / "sorghum"
SORGHUM_SCRIPTS = PIPELINE_DIR / "scripts" / "sorghum"
SORGHUM_FOLDS = SORGHUM["folds"]
SORGHUM_SOURCES = ["paper", "rmsprop", "lbfgs"]

wildcard_constraints:
    fold=r"\d+",
    source="|".join(SORGHUM_SOURCES),


def sorghum_traits(source, fold):
    """Traits file of one fold for one trait source."""
    if source == "paper":
        return str(sorghum.paper_traits(fold))
    return str(SORGHUM_OUT / f"h2opt_{source}" / f"fold{fold}.npz")


rule sorghum_all:
    input:
        expand(str(SORGHUM_OUT / "figures" / "top_trait_heritability_{source}.pdf"),
               source=SORGHUM_SOURCES),
        expand(str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.npz"),
               source=["paper", "lbfgs"], fold=SORGHUM_FOLDS),


rule sorghum_train_rmsprop:
    output:
        traits=str(SORGHUM_OUT / "h2opt_rmsprop" / "fold{fold}.npz"),
        model=str(SORGHUM_OUT / "h2opt_rmsprop" / "fold{fold}.pt"),
    log:
        str(SORGHUM_OUT / "h2opt_rmsprop" / "fold{fold}.log"),
    params:
        traits=SORGHUM["rmsprop"]["traits"],
        steps=SORGHUM["rmsprop"]["steps"],
        device=SORGHUM["rmsprop"]["device"],
    threads: 1
    resources:
        # Small GPU jobs are limited by Python overhead, so several share one GPU.
        gpu=lambda wc: 1 if SORGHUM["rmsprop"]["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/train_h2opt.py --fold {wildcards.fold}"
        " --out-traits {output.traits} --out-model {output.model}"
        " --traits {params.traits} --steps {params.steps} --device {params.device}"
        " > {log} 2>&1"


rule sorghum_train_lbfgs:
    output:
        str(SORGHUM_OUT / "h2opt_lbfgs" / "fold{fold}.npz"),
    log:
        str(SORGHUM_OUT / "h2opt_lbfgs" / "fold{fold}.log"),
    params:
        traits=SORGHUM["lbfgs"]["traits"],
        max_steps=SORGHUM["lbfgs"]["max_steps"],
    threads: 2
    shell:
        "python -u {SORGHUM_SCRIPTS}/train_h2opt_lbfgs.py --fold {wildcards.fold} --out {output}"
        " --traits {params.traits} --steps {params.max_steps}"
        " > {log} 2>&1"


rule sorghum_heritability:
    input:
        lambda wc: [sorghum_traits(wc.source, fold) for fold in SORGHUM_FOLDS],
    output:
        str(SORGHUM_OUT / "heritability_{source}.npz"),
    log:
        str(SORGHUM_OUT / "heritability_{source}.log"),
    params:
        traits=lambda wc: SORGHUM["evaluate_traits"][wc.source],
    threads: 4
    shell:
        "python -u {SORGHUM_SCRIPTS}/evaluate_heritability.py --source {wildcards.source}"
        " --h2opt-traits {input} --out {output} --traits {params.traits}"
        " > {log} 2>&1"


rule sorghum_plot_heritability:
    input:
        str(SORGHUM_OUT / "heritability_{source}.npz"),
    output:
        pdf=str(SORGHUM_OUT / "figures" / "top_trait_heritability_{source}.pdf"),
        png=str(SORGHUM_OUT / "figures" / "top_trait_heritability_{source}.png"),
    log:
        str(SORGHUM_OUT / "figures" / "top_trait_heritability_{source}.log"),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_heritability.py --heritability {input}"
        " --out {output.pdf} > {log} 2>&1"


rule sorghum_gwas:
    input:
        traits=lambda wc: sorghum_traits(wc.source, wc.fold),
    output:
        str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.log"),
    params:
        work=lambda wc: str(SORGHUM_OUT / "gwas" / "work" / f"{wc.source}_fold{wc.fold}"),
        traits=SORGHUM["gwas"]["traits"],
        pcs=SORGHUM["gwas"]["pcs"],
        jobs=SORGHUM["gwas"]["jobs"],
        threads=SORGHUM["gwas"]["threads"],
    threads: SORGHUM["gwas"]["jobs"] * SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/gwas.py --h2opt-traits {input.traits} --out {output}"
        " --work {params.work} --traits {params.traits} --pcs {params.pcs}"
        " --jobs {params.jobs} --threads {params.threads}"
        " > {log} 2>&1"
