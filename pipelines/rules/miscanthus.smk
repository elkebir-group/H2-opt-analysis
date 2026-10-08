# Miscanthus multispectral drone images, one date at a time: the H2-opt noise level per plant
# fold (one level for all dates), the H2-opt CNN per date, the baselines per date, and the
# heritability of their traits by Henderson's Method III.

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
H2OPT = MISC["h2opt"]


def misc_images(wildcards):
    return str(MISC_OUT / wildcards.population / "images.npy")


rule miscanthus_all:
    input:
        expand(str(MISC_OUT / "{population}" / "figures" / "heritability.pdf"),
               population=MISC_POPULATIONS),
        expand(str(MISC_OUT / "{population}" / "figures" / "noise.pdf"),
               population=MISC_POPULATIONS),


def misc_noise_scores(wildcards):
    return [str(MISC_OUT / wildcards.population / "h2opt" / "noise" / f"level{level}.npz")
            for level in H2OPT["levels"]]


# The CNN jobs take the whole GPU (gpu: the total of run.sh --gpu-jobs): a few models at once
# already fill its memory. One job per noise level: a level added to the config trains only that
# level.
rule misc_score_cnn_h2opt_noise:
    input:
        misc_images,
    output:
        str(MISC_OUT / "{population}" / "h2opt" / "noise" / "level{level}.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt" / "noise" / "level{level}.log"),
    wildcard_constraints:
        level=r"[0-9.e+-]+",
    threads: 2
    resources:
        gpu=H2OPT["gpu"],
    shell:
        "python -u {MISC_SCRIPTS}/score_noise.py --population {wildcards.population}"
        " --images {input} --level {wildcards.level} --out {output} --steps {H2OPT[steps]}"
        " --splits {H2OPT[splits]} --max-copies {H2OPT[max_copies]} --device {H2OPT[device]}"
        " > {log} 2>&1"


rule misc_choose_cnn_h2opt_noise:
    input:
        misc_noise_scores,
    output:
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
        pdf=str(MISC_OUT / "{population}" / "figures" / "noise.pdf"),
        png=str(MISC_OUT / "{population}" / "figures" / "noise.png"),
    log:
        str(MISC_OUT / "{population}" / "h2opt" / "noise.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/choose_noise.py --scores {input} --out {output.noise}"
        " --figure {output.pdf} > {log} 2>&1"


rule misc_train_cnn_h2opt:
    input:
        images=misc_images,
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
    output:
        traits=str(MISC_OUT / "{population}" / "h2opt" / "date{date}.npz"),
        models=str(MISC_OUT / "{population}" / "h2opt" / "date{date}.pt"),
    log:
        str(MISC_OUT / "{population}" / "h2opt" / "date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    threads: 2
    resources:
        gpu=H2OPT["gpu"],
    shell:
        "python -u {MISC_SCRIPTS}/train_h2opt.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --date {wildcards.date}"
        " --out-traits {output.traits} --out-models {output.models} --steps {H2OPT[steps]}"
        " --seed 0 --max-copies {H2OPT[max_copies]} --device {H2OPT[device]} > {log} 2>&1"


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


def misc_method_args(wildcards):
    """'--by-date h2opt date files --by-fold baseline fold files ...' for one population."""
    out = MISC_OUT / wildcards.population
    args = "--by-date h2opt " + " ".join(str(out / "h2opt" / f"date{d}.npz") for d in MISC_DATES)
    for baseline in MISC_BASELINES:
        args += f" --by-fold {baseline} " + " ".join(
            str(out / baseline / f"fold{f}.npz") for f in MISC_FOLDS)
    return args


rule misc_score_heritability:
    input:
        h2opt=lambda wc: [str(MISC_OUT / wc.population / "h2opt" / f"date{d}.npz")
                          for d in MISC_DATES],
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
