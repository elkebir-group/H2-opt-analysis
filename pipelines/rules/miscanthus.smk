# Miscanthus multispectral drone images, one date at a time: linear and hybrid H2-opt and the
# baselines per date, and the heritability of their traits by Henderson's Method III.

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


# Linear and hybrid H2-opt: one linear noise level (choose_noise.py) and one CNN noise level
# (choose_cnn_level.py, on one fold), each for all folds and dates, chosen on one date; then one
# job per date with all 5 folds.
MISC_H2OPT_METHODS = ["h2opt", "h2opt_hybrid"]


def misc_date_files(population, method):
    return [str(MISC_OUT / population / method / f"date{d}.npz") for d in MISC_DATES]


rule misc_choose_linear_h2opt_noise:
    input:
        misc_images,
    output:
        str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt" / "noise.log"),
    params:
        steps=MISC["h2opt"]["steps"],
        max_copies=MISC["h2opt"]["max_copies"],
        device=MISC["h2opt"]["device"],
    threads: 4
    resources:
        gpu=lambda wc: 4 if MISC["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/choose_noise.py --population {wildcards.population}"
        " --images {input} --date {MISC[selection_date]} --out {output} --steps {params.steps}"
        " --max-copies {params.max_copies} --device {params.device} > {log} 2>&1"


rule misc_train_linear_h2opt:
    input:
        images=misc_images,
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
    output:
        str(MISC_OUT / "{population}" / "h2opt" / "date{date}.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt" / "date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    params:
        steps=MISC["h2opt"]["steps"],
        device=MISC["h2opt"]["device"],
    threads: 2
    resources:
        gpu=lambda wc: 2 if MISC["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/train_h2opt.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --date {wildcards.date}"
        " --traits {MISC[traits]} --steps {params.steps} --device {params.device}"
        " --out {output} > {log} 2>&1"


rule misc_choose_hybrid_cnn_level:
    input:
        images=misc_images,
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
    output:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "cnn_level.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "cnn_level.log"),
    params:
        linear_steps=MISC["hybrid"]["linear_steps"],
        steps=MISC["hybrid"]["steps"],
        max_copies=MISC["hybrid"]["max_copies"],
        device=MISC["hybrid"]["device"],
    threads: 2
    # The CNN on images needs most of the GPU memory: one job at a time.
    resources:
        gpu=lambda wc: 8 if MISC["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/choose_cnn_level.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --out {output}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --max-copies {params.max_copies} --seed 0 --device {params.device} > {log} 2>&1"


rule misc_train_hybrid_h2opt:
    input:
        images=misc_images,
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
        cnn_level=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "cnn_level.npz"),
    output:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    params:
        linear_steps=MISC["hybrid"]["linear_steps"],
        steps=MISC["hybrid"]["steps"],
        max_copies=MISC["hybrid"]["max_copies"],
        device=MISC["hybrid"]["device"],
    threads: 2
    resources:
        gpu=lambda wc: 4 if MISC["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/train_hybrid.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --cnn-level {input.cnn_level}"
        " --date {wildcards.date} --traits {MISC[traits]}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --max-copies {params.max_copies} --device {params.device} --out {output} > {log} 2>&1"


def misc_method_args(wildcards):
    """'--by-date method date files ... --by-fold baseline fold files ...' for one population."""
    out = MISC_OUT / wildcards.population
    args = " ".join(f"--by-date {m} " + " ".join(misc_date_files(wildcards.population, m))
                    for m in MISC_H2OPT_METHODS)
    for baseline in MISC_BASELINES:
        args += f" --by-fold {baseline} " + " ".join(
            str(out / baseline / f"fold{f}.npz") for f in MISC_FOLDS)
    return args


rule misc_score_heritability:
    input:
        h2opt=lambda wc: [f for m in MISC_H2OPT_METHODS for f in misc_date_files(wc.population, m)],
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
