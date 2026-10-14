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
# The run of the paper (rules misc_adjusted_*): H2-opt trained on the neighbor-adjusted heritability.
MISC_ADJ = OUT_ROOT / "miscanthus_adjusted"
MISC_ADJ_CONFIG = MISC["adjusted"]
MISC_ADJ_DATES = MISC_ADJ_CONFIG["dates"]
MISC_FOLDS = range(miscanthus.N_FOLDS)
MISC_BASELINES = MISC["baselines"]


def misc_images(wildcards):
    return str(MISC_OUT / wildcards.population / "images.npy")


rule miscanthus_all:
    input:
        expand(str(MISC_OUT / "{population}" / "figures" / "heritability.pdf"),
               population=MISC_POPULATIONS),
        expand(str(MISC_OUT / "{population}" / "heritability_spatial.npz"),
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


# The test heritability after a nearest-neighbor field adjustment (Papadakis 1937), with the
# measured traits of Central as a control.
rule misc_score_heritability_spatial:
    input:
        h2opt=lambda wc: [f for m in MISC_H2OPT_METHODS for f in misc_date_files(wc.population, m)],
        baselines=lambda wc: [str(MISC_OUT / wc.population / b / f"fold{f}.npz")
                              for b in MISC_BASELINES for f in MISC_FOLDS],
    output:
        str(MISC_OUT / "{population}" / "heritability_spatial.npz"),
    log:
        str(MISC_OUT / "{population}" / "heritability_spatial.log"),
    params:
        methods=misc_method_args,
        dates=" ".join(str(d) for d in MISC_DATES),
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/evaluate_heritability_spatial.py"
        " --population {wildcards.population} {params.methods}"
        " --dates {params.dates} --width {MISC[spatial_width]} --out {output} > {log} 2>&1"


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
MISC_PAPER_FIGURES = ["miscFig", "miscVegIndex", "trainHeritMisc",
                      "miscCorTime", "traitCorrelation_allTraits",
                      "traitCorrelation_simplePearson", "miscGWAS_central", "miscGWAS_south",
                      "miscNull"]


def misc_by_population(name):
    return [str(MISC_OUT / p / name) for p in miscanthus.POPULATIONS]


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
        " --width {MISC[spatial_width]} --out {output} > {log} 2>&1"


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
        lambda wc: misc_date_files(wc.population, "h2opt_hybrid"),
    output:
        str(MISC_OUT / "{population}" / "measured_correlations.npz"),
    log:
        str(MISC_OUT / "{population}" / "measured_correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/measured_correlations.py --population {wildcards.population}"
        " --traits {input} --out {output} > {log} 2>&1"


rule misc_plot_paper_heritability:
    input:
        adjusted=[str(MISC_ADJ / p / "heritability.npz") for p in miscanthus.POPULATIONS],
        ndvi=misc_by_population("ndvi.npz"),
    output:
        train=str(MISC_PAPER / "trainHeritMisc.pdf"),
        ndvi=str(MISC_PAPER / "miscVegIndex.pdf"),
    log:
        str(MISC_PAPER / "heritability.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_heritability.py"
        " --heritability-adjusted {input.adjusted} --ndvi {input.ndvi}"
        " --out-train {output.train} --out-ndvi-heritability {output.ndvi} > {log} 2>&1"


rule misc_plot_paper_correlations:
    input:
        dates=[str(MISC_ADJ / p / "date_correlations.npz") for p in miscanthus.POPULATIONS],
        measured=str(MISC_ADJ / "central" / "measured_correlations.npz"),
    output:
        dates=str(MISC_PAPER / "miscCorTime.pdf"),
        measured_family=str(MISC_PAPER / "traitCorrelation_allTraits.pdf"),
        measured_plant=str(MISC_PAPER / "traitCorrelation_simplePearson.pdf"),
    log:
        str(MISC_PAPER / "correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_correlations.py --dates {input.dates}"
        " --measured {input.measured} --out-dates {output.dates}"
        " --out-measured-family {output.measured_family}"
        " --out-measured-plant {output.measured_plant} > {log} 2>&1"


# miscGWAS_<population>: the adjusted run's GWAS of H2-opt (CNN) trait 1 on each date with a
# significant SNP, per fold and cross-fitted, in the style of sorGWAS.
rule misc_plot_paper_gwas:
    input:
        gwas=lambda wc: [str(MISC_ADJ / wc.population / "gwas" / f"date{d}.npz")
                         for d in MISC_DATES],
        crossfit=lambda wc: [str(MISC_ADJ / wc.population / "gwas" / f"crossfit_date{d}.npz")
                             for d in MISC_DATES],
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
    output:
        pdf=str(MISC_PAPER / "miscGWAS_{population}.pdf"),
        png=str(MISC_PAPER / "miscGWAS_{population}.png"),
    log:
        str(MISC_PAPER / "miscGWAS_{population}.log"),
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_gwas.py --population {wildcards.population}"
        " --gwas {input.gwas} --crossfit {input.crossfit} --genotypes {params.prefix}"
        " --r2 {config[gwas_loci][r2]} --kb {config[gwas_loci][kb]} --out {output.pdf}"
        " > {log} 2>&1"


# miscNull: the adjusted run's GWAS of H2-opt (CNN) trait 1 of all folds and dates against its
# cross-fit and the permutation null (largest -log10 p, lambda GC, QQ plot), in the style of sorNull.
rule misc_plot_paper_null:
    input:
        **{f"{kind}{p}": [str(MISC_ADJ / p / "gwas" / f"{prefix}date{d}.npz") for d in MISC_DATES]
           for p in miscanthus.POPULATIONS
           for kind, prefix in (("gwas_", ""), ("crossfit_", "crossfit_"))},
        **{f"null_{p}": [str(MISC_ADJ / p / "null" / f"perm{k}.npz")
                         for k in range(MISC["gwas"]["permutations"])]
           for p in miscanthus.POPULATIONS},
    output:
        str(MISC_PAPER / "miscNull.pdf"),
    log:
        str(MISC_PAPER / "miscNull.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_null.py"
        " --gwas-central {input.gwas_central} --crossfit-central {input.crossfit_central}"
        " --null-central {input.null_central} --gwas-south {input.gwas_south}"
        " --crossfit-south {input.crossfit_south} --null-south {input.null_south}"
        " --null-date {MISC[selection_date]} --out {output} > {log} 2>&1"


rule misc_plot_paper_main:
    input:
        images=misc_by_population("images.npy"),
        heritability=[str(MISC_ADJ / p / "heritability.npz") for p in miscanthus.POPULATIONS],
        dates=[str(MISC_ADJ / p / "date_correlations.npz") for p in miscanthus.POPULATIONS],
        measured=[str(MISC_ADJ / p / "measured_correlations.npz") for p in miscanthus.POPULATIONS],
    output:
        str(MISC_PAPER / "miscFig.pdf"),
    log:
        str(MISC_PAPER / "miscFig.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/plot_paper_main.py --images {input.images}"
        " --heritability {input.heritability} --dates {input.dates}"
        " --measured {input.measured} --out {output} > {log} 2>&1"


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


# The run of the paper: H2-opt trained on the heritability after the nearest-neighbor field
# adjustment (config miscanthus.adjusted). Outputs under miscanthus_adjusted/ (MISC_ADJ); the
# baselines and images come from the main run.


rule miscanthus_adjusted_all:
    input:
        expand(str(MISC_ADJ / "{population}" / "heritability.npz"),
               population=MISC_POPULATIONS),


rule misc_adjusted_choose_linear_h2opt_noise:
    input:
        misc_images,
    output:
        str(MISC_ADJ / "{population}" / "h2opt" / "noise.npz"),
    log:
        str(MISC_ADJ / "{population}" / "h2opt" / "noise.log"),
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
        " --max-copies {params.max_copies} --device {params.device}"
        " --neighbor-width {MISC[spatial_width]} > {log} 2>&1"


rule misc_adjusted_train_linear_h2opt:
    input:
        images=misc_images,
        noise=str(MISC_ADJ / "{population}" / "h2opt" / "noise.npz"),
    output:
        traits=str(MISC_ADJ / "{population}" / "h2opt" / "date{date}.npz"),
        models=str(MISC_ADJ / "{population}" / "h2opt" / "date{date}.pt"),
    log:
        str(MISC_ADJ / "{population}" / "h2opt" / "date{date}.log"),
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
        " --neighbor-width {MISC[spatial_width]}"
        " --out {output.traits} --out-models {output.models} > {log} 2>&1"


def misc_adjusted_cnn_level(wildcards):
    """The main run's CNN level file of the population that cnn_level_population names for this
    population."""
    source = MISC_ADJ_CONFIG["cnn_level_population"][wildcards.population]
    return str(MISC_OUT / source / "h2opt_hybrid" / "cnn_level.npz")


rule misc_adjusted_train_hybrid_h2opt:
    input:
        images=misc_images,
        noise=str(MISC_ADJ / "{population}" / "h2opt" / "noise.npz"),
        cnn_level=misc_adjusted_cnn_level,
    output:
        traits=str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "date{date}.npz"),
        models=str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "date{date}.pt"),
    log:
        str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "date{date}.log"),
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
        " --max-copies {params.max_copies} --device {params.device}"
        " --neighbor-width {MISC[spatial_width]} --out {output.traits}"
        " --out-models {output.models} > {log} 2>&1"


# Traits 2, 3, ... of the adjusted run's hybrid H2-opt: trait k trained on top of each fold's
# traits 1 to k - 1 (train_hybrid.py --earlier), and the adjusted test heritability of each.
MISC_ADJ_LATER = range(2, MISC_ADJ_CONFIG["traits"] + 1)


def misc_adjusted_earlier(wildcards):
    hybrid = MISC_ADJ / wildcards.population / "h2opt_hybrid"
    return ([str(hybrid / f"date{wildcards.date}.npz")]
            + [str(hybrid / f"trait{j}_date{wildcards.date}.npz")
               for j in range(2, int(wildcards.k))])


rule miscanthus_adjusted_later_traits_all:
    input:
        expand(str(MISC_ADJ / "{population}" / "heritability_trait{k}.npz"),
               population=MISC_POPULATIONS, k=MISC_ADJ_LATER),


rule misc_adjusted_train_hybrid_later:
    input:
        images=misc_images,
        noise=str(MISC_ADJ / "{population}" / "h2opt" / "noise.npz"),
        cnn_level=misc_adjusted_cnn_level,
        earlier=misc_adjusted_earlier,
    output:
        traits=str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "trait{k}_date{date}.npz"),
        models=str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "trait{k}_date{date}.pt"),
    log:
        str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "trait{k}_date{date}.log"),
    wildcard_constraints:
        k=r"\d+",
        date=r"\d+",
    threads: 2
    resources:
        gpu=lambda wc: 4 if MISC["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {MISC_SCRIPTS}/train_hybrid.py --population {wildcards.population}"
        " --images {input.images} --noise {input.noise} --cnn-level {input.cnn_level}"
        " --date {wildcards.date} --traits 1 --earlier {input.earlier}"
        " --linear-steps {MISC[hybrid][linear_steps]} --steps {MISC[hybrid][steps]}"
        " --max-copies {MISC[hybrid][max_copies]} --device {MISC[hybrid][device]}"
        " --neighbor-width {MISC[spatial_width]} --out {output.traits}"
        " --out-models {output.models} > {log} 2>&1"


rule misc_adjusted_score_later:
    input:
        lambda wc: [str(MISC_ADJ / wc.population / "h2opt_hybrid" / f"trait{wc.k}_date{d}.npz")
                    for d in MISC_ADJ_DATES],
    output:
        str(MISC_ADJ / "{population}" / "heritability_trait{k}.npz"),
    log:
        str(MISC_ADJ / "{population}" / "heritability_trait{k}.log"),
    wildcard_constraints:
        k=r"\d+",
    params:
        dates=" ".join(str(d) for d in MISC_ADJ_DATES),
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/evaluate_heritability_spatial.py"
        " --population {wildcards.population} --by-date h2opt_hybrid {input}"
        " --dates {params.dates} --width {MISC[spatial_width]} --out {output} > {log} 2>&1"


# GWAS of traits 1 and 2 of the adjusted run's hybrid H2-opt per date (all folds) and of their
# cross-fitted traits, with the neighbor mean as a covariate (gwas.py --neighbor-width); genotypes
# from the main run. {trait} is "" for trait 1 (date{date}.npz) and "trait2_" for trait 2.
MISC_ADJ_GWAS_TRAITS = ["", "trait2_"]


rule miscanthus_adjusted_gwas_all:
    input:
        expand(str(MISC_ADJ / "{population}" / "gwas" / "{kind}{trait}date{date}.npz"),
               population=MISC_POPULATIONS, kind=["", "crossfit_"], trait=MISC_ADJ_GWAS_TRAITS,
               date=MISC_ADJ_DATES),


rule misc_adjusted_crossfit_traits:
    input:
        str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "{trait}date{date}.npz"),
    output:
        str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "crossfit_{trait}date{date}.npz"),
    log:
        str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "crossfit_{trait}date{date}.log"),
    wildcard_constraints:
        trait="|trait2_",
        date=r"\d+",
    threads: 1
    shell:
        "python -u {PIPELINE_DIR}/scripts/shared/cross_fit_traits.py --dataset miscanthus"
        " --population {wildcards.population} --trait 1 --traits {input} --out {output}"
        " > {log} 2>&1"


rule misc_adjusted_run_gwas:
    input:
        traits=str(MISC_ADJ / "{population}" / "h2opt_hybrid" / "{kind}{trait}date{date}.npz"),
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(MISC_ADJ / "{population}" / "gwas" / "{kind}{trait}date{date}.npz"),
    log:
        str(MISC_ADJ / "{population}" / "gwas" / "{kind}{trait}date{date}.log"),
    wildcard_constraints:
        kind="|crossfit_",
        trait="|trait2_",
        date=r"\d+",
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
        work=lambda wc: str(MISC_ADJ / wc.population / "gwas" / "work"
                            / f"{wc.kind}{wc.trait}date{wc.date}"),
    threads: MISC["gwas"]["jobs"] * MISC["gwas"]["threads"]
    shell:
        "python -u {MISC_SCRIPTS}/gwas.py --population {wildcards.population}"
        " --h2opt-traits {input.traits} --genotypes {params.prefix} --pcs {input.pcs}"
        " --neighbor-width {MISC[spatial_width]} --out {output} --work {params.work}"
        " --jobs {MISC[gwas][jobs]} --threads {MISC[gwas][threads]} > {log} 2>&1"


rule misc_adjusted_date_correlations:
    input:
        lambda wc: [str(MISC_ADJ / wc.population / "h2opt_hybrid" / f"date{d}.npz")
                    for d in MISC_DATES],
    output:
        str(MISC_ADJ / "{population}" / "date_correlations.npz"),
    log:
        str(MISC_ADJ / "{population}" / "date_correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/date_correlations.py --population {wildcards.population}"
        " --traits {input} --out {output} > {log} 2>&1"


rule misc_adjusted_measured_correlations:
    input:
        lambda wc: [str(MISC_ADJ / wc.population / "h2opt_hybrid" / f"date{d}.npz")
                    for d in MISC_DATES],
    output:
        str(MISC_ADJ / "{population}" / "measured_correlations.npz"),
    log:
        str(MISC_ADJ / "{population}" / "measured_correlations.log"),
    threads: 1
    shell:
        "python -u {MISC_SCRIPTS}/measured_correlations.py --population {wildcards.population}"
        " --traits {input} --out {output} > {log} 2>&1"


# The permutation null of the adjusted run (null_gwas.py --neighbor-width) on selection_date.
rule miscanthus_adjusted_null_all:
    input:
        expand(str(MISC_ADJ / "{population}" / "null" / "perm{k}.npz"),
               population=MISC_POPULATIONS, k=range(MISC["gwas"]["permutations"])),


rule misc_adjusted_null_gwas:
    input:
        images=misc_images,
        noise=str(MISC_ADJ / "{population}" / "h2opt" / "noise.npz"),
        cnn_level=misc_adjusted_cnn_level,
        genotypes=multiext(str(MISC_OUT / "{population}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(MISC_OUT / "{population}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(MISC_ADJ / "{population}" / "null" / "perm{k}.npz"),
    log:
        str(MISC_ADJ / "{population}" / "null" / "perm{k}.log"),
    wildcard_constraints:
        k=r"\d+",
    params:
        prefix=lambda wc: str(MISC_OUT / wc.population / "gwas" / "genotypes"),
        work=lambda wc: str(MISC_ADJ / wc.population / "null" / "work" / f"perm{wc.k}"),
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
        " --threads {MISC[gwas][threads]} --neighbor-width {MISC[spatial_width]} > {log} 2>&1"


def misc_adjusted_method_args(wildcards):
    """--by-date and --by-fold arguments of the adjusted run: its H2-opt traits, the main baselines."""
    args = " ".join(
        f"--by-date {m} " + " ".join(str(MISC_ADJ / wildcards.population / m / f"date{d}.npz")
                                     for d in MISC_ADJ_DATES)
        for m in MISC_H2OPT_METHODS)
    for baseline in MISC_ADJ_CONFIG["baselines"]:
        args += f" --by-fold {baseline} " + " ".join(
            str(MISC_OUT / wildcards.population / baseline / f"fold{f}.npz") for f in MISC_FOLDS)
    return args


rule misc_adjusted_score_heritability:
    input:
        h2opt=lambda wc: [str(MISC_ADJ / wc.population / m / f"date{d}.npz")
                          for m in MISC_H2OPT_METHODS for d in MISC_ADJ_DATES],
        baselines=lambda wc: [str(MISC_OUT / wc.population / b / f"fold{f}.npz")
                              for b in MISC_ADJ_CONFIG["baselines"] for f in MISC_FOLDS],
    output:
        str(MISC_ADJ / "{population}" / "heritability.npz"),
    log:
        str(MISC_ADJ / "{population}" / "heritability.log"),
    params:
        methods=misc_adjusted_method_args,
        dates=" ".join(str(d) for d in MISC_ADJ_DATES),
    threads: 2
    shell:
        "python -u {MISC_SCRIPTS}/evaluate_heritability_spatial.py"
        " --population {wildcards.population} {params.methods}"
        " --dates {params.dates} --width {MISC[spatial_width]} --out {output} > {log} 2>&1"
