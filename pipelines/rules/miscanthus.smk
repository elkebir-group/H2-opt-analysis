# Miscanthus multispectral drone images, one date at a time: the baselines per date, and the
# heritability of their traits and of the hybrid H2-opt traits by Henderson's Method III.

from h2opt_analysis import miscanthus

MISC = config["miscanthus"]
MISC_OUT = OUT_ROOT / "miscanthus"
MISC_SCRIPTS = PIPELINE_DIR / "scripts" / "miscanthus"

wildcard_constraints:
    population="|".join(miscanthus.POPULATIONS),


rule misc_prepare_images:
    output:
        str(MISC_OUT / "{population}" / "images.npy"),
    log:
        str(MISC_OUT / "{population}" / "images.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/images.py --population {wildcards.population} --out {output}"
        " > {log} 2>&1"


MISC_POPULATIONS = MISC["populations"]
MISC_DATES = range(len(miscanthus.DATES))
MISC_FOLDS = range(miscanthus.N_FOLDS)
MISC_BASELINES = MISC["baselines"]


def misc_images(wildcards):
    return str(MISC_OUT / wildcards.population / "images.npy")


rule miscanthus_all:
    input:
        expand(str(MISC_OUT / "{population}" / "figures" / "heritability.pdf"),
               population=MISC_POPULATIONS),


rule misc_train_baselines:
    input:
        misc_images,
    output:
        str(MISC_OUT / "{population}" / "{baseline}" / "fold{fold}.npz"),
    log:
        str(MISC_OUT / "{population}" / "{baseline}" / "fold{fold}.log"),
    wildcard_constraints:
        baseline="|".join(MISC_BASELINES),
        fold=r"\d+",
    threads: 4
    shell:
        "python -u {MISC_SCRIPTS}/baseline.py --population {wildcards.population}"
        " --images {input} --method {wildcards.baseline} --fold {wildcards.fold}"
        " --traits {MISC[traits]} --out {output} > {log} 2>&1"


def misc_hybrid_traits(population):
    """Traits of the hybrid H2-opt, one file per date. Imported from a scratch run of the earlier
    hybrid design (h2opt_hybrid/PROVENANCE.md) until a rule trains them."""
    return [str(MISC_OUT / population / "h2opt_hybrid" / f"date{d}.npz") for d in MISC_DATES]


def misc_method_args(wildcards):
    """'--by-date h2opt_hybrid date files --by-fold baseline fold files ...' for one population."""
    out = MISC_OUT / wildcards.population
    args = "--by-date h2opt_hybrid " + " ".join(misc_hybrid_traits(wildcards.population))
    for baseline in MISC_BASELINES:
        args += f" --by-fold {baseline} " + " ".join(
            str(out / baseline / f"fold{f}.npz") for f in MISC_FOLDS)
    return args


rule misc_score_heritability:
    input:
        hybrid=lambda wc: misc_hybrid_traits(wc.population),
        baselines=lambda wc: [str(MISC_OUT / wc.population / b / f"fold{f}.npz")
                              for b in MISC_BASELINES for f in MISC_FOLDS],
    output:
        str(MISC_OUT / "{population}" / "heritability.npz"),
    log:
        str(MISC_OUT / "{population}" / "heritability.log"),
    params:
        methods=misc_method_args,
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/evaluate_heritability.py --population {wildcards.population}"
        " {params.methods} --traits {MISC[traits]} --out {output} > {log} 2>&1"


rule misc_plot_heritability:
    input:
        str(MISC_OUT / "{population}" / "heritability.npz"),
    output:
        pdf=str(MISC_OUT / "{population}" / "figures" / "heritability.pdf"),
        png=str(MISC_OUT / "{population}" / "figures" / "heritability.png"),
    log:
        str(MISC_OUT / "{population}" / "figures" / "heritability.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_heritability.py --heritability {input}"
        " --out {output.pdf} > {log} 2>&1"
