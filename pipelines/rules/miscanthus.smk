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


# GWAS of trait 1 of hybrid H2-opt, every date and fold. Run only on request.
rule miscanthus_gwas_all:
    input:
        expand(str(MISC_OUT / "{population}" / "gwas" / "date{date}.npz"),
               population=MISC_POPULATIONS, date=MISC_DATES),


# The permutation null of the whole hybrid H2-opt -> GWAS pipeline on selection_date, and the GWAS
# of the cross-fitted trait 1 (each plant's phenotype from the fold model that held it out) on
# every date. Run only on request.
rule miscanthus_null_gwas_all:
    input:
        expand(str(MISC_OUT / "{population}" / "null" / "perm{k}.npz"),
               population=MISC_POPULATIONS, k=range(MISC["gwas"]["permutations"])),
        expand(str(MISC_OUT / "{population}" / "gwas" / "crossfit_date{date}.npz"),
               population=MISC_POPULATIONS, date=MISC_DATES),


rule misc_prepare_genotypes:
    output:
        multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"), ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    log:
        str(MISC_OUT / "{population}" / "gwas" / "genotypes.log"),
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/genotypes.py --population {wildcards.population}"
        " --out {params.prefix} --out-pcs {output.pcs} --pcs {MISC[gwas][pcs]} > {log} 2>&1"


rule misc_run_gwas:
    input:
        traits=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.npz"),
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(MISC_OUT / "{population}" / "gwas" / "date{date}.npz"),
    log:
        str(MISC_OUT / "{population}" / "gwas" / "date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
        work=lambda wc: str(MISC_OUT / wc.population / "gwas" / "work" / f"date{wc.date}"),
    threads: MISC["gwas"]["jobs"] * MISC["gwas"]["threads"]
    shell:
        "python -u {MISC_SCRIPTS}/gwas.py --population {wildcards.population}"
        " --h2opt-traits {input.traits} --genotypes {params.prefix} --pcs {input.pcs}"
        " --out {output} --work {params.work} --jobs {MISC[gwas][jobs]}"
        " --threads {MISC[gwas][threads]} > {log} 2>&1"


rule misc_choose_pch_ridge:
    input:
        misc_images,
    output:
        str(MISC_OUT / "{population}" / "pch" / "ridge.npz"),
    log:
        str(MISC_OUT / "{population}" / "pch" / "ridge.log"),
    threads: 8
    shell:
        "python -u {MISC_SCRIPTS}/choose_pch_ridge.py --population {wildcards.population}"
        " --images {input} --date {MISC[selection_date]} --out {output} > {log} 2>&1"


def misc_baseline_inputs(wildcards):
    inputs = {"images": misc_images(wildcards)}
    if wildcards.baseline == "pch":
        inputs["ridge"] = str(MISC_OUT / wildcards.population / "pch" / "ridge.npz")
    return inputs


def misc_ridge_arg(wildcards, input):
    return f"--ridge {input.ridge}" if wildcards.baseline == "pch" else ""


rule misc_train_baselines:
    input:
        unpack(misc_baseline_inputs),
    output:
        str(MISC_OUT / "{population}" / "{baseline}" / "fold{fold}.npz"),
    log:
        str(MISC_OUT / "{population}" / "{baseline}" / "fold{fold}.log"),
    wildcard_constraints:
        baseline="|".join(MISC_BASELINES),
        fold=r"\d+",
    params:
        ridge=misc_ridge_arg,
    threads: 4
    shell:
        "python -u {MISC_SCRIPTS}/baseline.py --population {wildcards.population}"
        " --images {input.images} --method {wildcards.baseline} {params.ridge}"
        " --fold {wildcards.fold}"
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
        traits=str(MISC_OUT / "{population}" / "h2opt" / "date{date}.npz"),
        models=str(MISC_OUT / "{population}" / "h2opt" / "date{date}.pt"),
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
        " --out {output.traits} --out-models {output.models} > {log} 2>&1"


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
        traits=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.npz"),
        models=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.pt"),
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
        " --max-copies {params.max_copies} --device {params.device} --out {output.traits}"
        " --out-models {output.models} > {log} 2>&1"


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


# The Miscanthus figures of the paper, in figures/paper/ (file names as in the paper). miscFig and
# miscGWAS need the GWAS (miscanthus_gwas_all). miscSecondTrait is not made: it needs trait 2.
MISC_PAPER = MISC_OUT / "figures" / "paper"
MISC_PAPER_FIGURES = ["miscFig", "NDVI_canopy", "miscVegIndex", "trainHeritMisc",
                      "miscCorTimeSimple", "miscCorTime", "traitCorrelation_allTraits",
                      "traitCorrelation_simplePearson", "miscGWAS"]


def misc_by_population(name):
    return [str(MISC_OUT / p / name) for p in miscanthus.POPULATIONS]


def misc_gwas_files(dates):
    return [str(MISC_OUT / p / "gwas" / f"date{d}.npz") for p in miscanthus.POPULATIONS
            for d in dates]


rule miscanthus_paper_figures:
    input:
        expand(str(MISC_PAPER / "{stem}.pdf"), stem=MISC_PAPER_FIGURES),


rule misc_score_ndvi:
    input:
        misc_images,
    output:
        str(MISC_OUT / "{population}" / "ndvi.npz"),
    log:
        str(MISC_OUT / "{population}" / "ndvi.log"),
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/ndvi.py --population {wildcards.population} --images {input}"
        " --out {output} > {log} 2>&1"


rule misc_date_correlations:
    input:
        lambda wc: misc_date_files(wc.population, "h2opt_hybrid"),
    output:
        str(MISC_OUT / "{population}" / "date_correlations.npz"),
    log:
        str(MISC_OUT / "{population}" / "date_correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/date_correlations.py --population {wildcards.population}"
        " --traits {input} --out {output} > {log} 2>&1"


rule misc_measured_correlations:
    input:
        misc_date_files("central", "h2opt_hybrid"),
    output:
        str(MISC_OUT / "central" / "measured_correlations.npz"),
    log:
        str(MISC_OUT / "central" / "measured_correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/measured_correlations.py --traits {input} --out {output}"
        " > {log} 2>&1"


rule misc_plot_paper_heritability:
    input:
        heritability=misc_by_population("heritability.npz"),
        ndvi=misc_by_population("ndvi.npz"),
    output:
        train=str(MISC_PAPER / "trainHeritMisc.pdf"),
        ndvi=str(MISC_PAPER / "miscVegIndex.pdf"),
        canopy=str(MISC_PAPER / "NDVI_canopy.pdf"),
    log:
        str(MISC_PAPER / "heritability.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_heritability.py --heritability {input.heritability}"
        " --ndvi {input.ndvi} --out-train {output.train} --out-ndvi-heritability {output.ndvi}"
        " --out-canopy {output.canopy} > {log} 2>&1"


rule misc_plot_paper_correlations:
    input:
        dates=misc_by_population("date_correlations.npz"),
        measured=str(MISC_OUT / "central" / "measured_correlations.npz"),
    output:
        dates_plant=str(MISC_PAPER / "miscCorTimeSimple.pdf"),
        dates_family=str(MISC_PAPER / "miscCorTime.pdf"),
        measured_family=str(MISC_PAPER / "traitCorrelation_allTraits.pdf"),
        measured_plant=str(MISC_PAPER / "traitCorrelation_simplePearson.pdf"),
    log:
        str(MISC_PAPER / "correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_correlations.py --dates {input.dates}"
        " --measured {input.measured} --out-dates-plant {output.dates_plant}"
        " --out-dates-family {output.dates_family}"
        " --out-measured-family {output.measured_family}"
        " --out-measured-plant {output.measured_plant} > {log} 2>&1"


rule misc_plot_paper_gwas:
    input:
        misc_gwas_files(MISC_DATES),
    output:
        str(MISC_PAPER / "miscGWAS.pdf"),
    log:
        str(MISC_PAPER / "miscGWAS.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_gwas.py --gwas {input} --out {output} > {log} 2>&1"


rule misc_plot_paper_main:
    input:
        images=misc_by_population("images.npy"),
        heritability=misc_by_population("heritability.npz"),
        dates=misc_by_population("date_correlations.npz"),
        measured=str(MISC_OUT / "central" / "measured_correlations.npz"),
        gwas=misc_gwas_files(range(5)),
    output:
        str(MISC_PAPER / "miscFig.pdf"),
    log:
        str(MISC_PAPER / "miscFig.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_main.py --images {input.images}"
        " --heritability {input.heritability} --dates {input.dates}"
        " --measured {input.measured} --gwas {input.gwas} --out {output} > {log} 2>&1"


rule misc_null_gwas:
    input:
        images=misc_images,
        noise=str(MISC_OUT / "{population}" / "h2opt" / "noise.npz"),
        cnn_level=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "cnn_level.npz"),
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(MISC_OUT / "{population}" / "null" / "perm{k}.npz"),
    log:
        str(MISC_OUT / "{population}" / "null" / "perm{k}.log"),
    wildcard_constraints:
        k=r"\d+",
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
        work=lambda wc: str(MISC_OUT / wc.population / "null" / "work" / f"perm{wc.k}"),
    threads: MISC["gwas"]["threads"]
    resources:
        gpu=lambda wc: 4 if MISC["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/null_gwas.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --cnn-level {input.cnn_level}"
        " --date {MISC[selection_date]} --permutation {wildcards.k}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --linear-steps {MISC[hybrid][linear_steps]} --steps {MISC[hybrid][steps]}"
        " --max-copies {MISC[hybrid][max_copies]} --device {MISC[hybrid][device]}"
        " --threads {MISC[gwas][threads]} > {log} 2>&1"


rule misc_cross_fit_traits:
    input:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "date{date}.npz"),
    output:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "crossfit_date{date}.npz"),
    log:
        str(MISC_OUT / "{population}" / "h2opt_hybrid" / "crossfit_date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    threads: 1
    shell:
        "python -u {PIPELINE_DIR}/scripts/shared/cross_fit_traits.py --dataset miscanthus"
        " --population {wildcards.population} --trait 1 --traits {input} --out {output}"
        " > {log} 2>&1"


rule misc_crossfit_gwas:
    input:
        traits=str(MISC_OUT / "{population}" / "h2opt_hybrid" / "crossfit_date{date}.npz"),
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(MISC_OUT / "{population}" / "gwas" / "crossfit_date{date}.npz"),
    log:
        str(MISC_OUT / "{population}" / "gwas" / "crossfit_date{date}.log"),
    wildcard_constraints:
        date=r"\d+",
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
        work=lambda wc: str(MISC_OUT / wc.population / "gwas" / "work" / f"crossfit_date{wc.date}"),
    threads: MISC["gwas"]["threads"]
    shell:
        "python -u {MISC_SCRIPTS}/gwas.py --population {wildcards.population}"
        " --h2opt-traits {input.traits} --genotypes {params.prefix} --pcs {input.pcs}"
        " --out {output} --work {params.work} --jobs 1 --threads {MISC[gwas][threads]}"
        " > {log} 2>&1"
