# Arabidopsis LC-MS metabolomics, each treatment (0d, 6d) on its own: the H2-opt noise level per
# accession fold (one for all traits), H2-opt per fold, the baselines per fold, and the ANOVA
# heritability of their traits among accessions, with the year as environment.

ARABI = config["arabidopsis"]
ARABI_OUT = OUT_ROOT / "arabidopsis"
ARABI_SCRIPTS = PIPELINE_DIR / "scripts" / "arabidopsis"
ARABI_FOLDS = ARABI["folds"]
ARABI_FEATURES = str(ARABI_OUT / "features.npy")

wildcard_constraints:
    treatment="|".join(ARABI["treatments"]),


def arabi_traits(treatment, method, fold):
    """Traits file of one fold for H2-opt or a baseline."""
    return str(ARABI_OUT / treatment / method / f"fold{fold}.npz")


def arabi_method_args(treatment):
    """'--method slug file0 ... file4' for H2-opt and every baseline."""
    return " ".join(f"--method {m} " + " ".join(arabi_traits(treatment, m, f) for f in ARABI_FOLDS)
                    for m in ["h2opt"] + ARABI["baselines"])


rule arabidopsis_all:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "figures" / "noise.pdf"),
               treatment=ARABI["treatments"]),


# GWAS of the H2-opt traits, every treatment and fold. Run only on request.
rule arabidopsis_gwas_all:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "gwas" / "fold{fold}.npz"),
               treatment=ARABI["treatments"], fold=ARABI_FOLDS),


# The permutation null of the whole H2-opt -> GWAS pipeline, and the GWAS of the cross-fitted
# trait 1 (each accession's phenotype from the fold model that held it out). Run only on request.
rule arabidopsis_null_gwas_all:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "null" / "perm{k}.npz"),
               treatment=ARABI["treatments"], k=range(ARABI["gwas"]["permutations"])),
        expand(str(ARABI_OUT / "{treatment}" / "gwas" / "crossfit.npz"),
               treatment=ARABI["treatments"]),


rule arabi_prepare_genotypes:
    output:
        multiext(str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes"), ".bed", ".bim", ".fam"),
        pcs=str(ARABI_OUT / "{treatment}" / "gwas" / "genotype_pcs.npy"),
    log:
        str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes.log"),
    params:
        prefix=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "genotypes"),
    threads: 2
    shell:
        "python -u {ARABI_SCRIPTS}/genotypes.py --treatment {wildcards.treatment}"
        " --out {params.prefix} --out-pcs {output.pcs} --pcs {ARABI[gwas][pcs]} > {log} 2>&1"


rule arabi_run_gwas:
    input:
        traits=lambda wc: arabi_traits(wc.treatment, "h2opt", wc.fold),
        genotypes=multiext(str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(ARABI_OUT / "{treatment}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(ARABI_OUT / "{treatment}" / "gwas" / "fold{fold}.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "gwas" / "fold{fold}.log"),
    wildcard_constraints:
        fold=r"\d+",
    params:
        prefix=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "genotypes"),
        work=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "work" / f"fold{wc.fold}"),
    threads: ARABI["gwas"]["jobs"] * ARABI["gwas"]["threads"]
    shell:
        "python -u {ARABI_SCRIPTS}/gwas.py --treatment {wildcards.treatment}"
        " --h2opt-traits {input.traits} --genotypes {params.prefix} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits {ARABI[gwas][traits]}"
        " --jobs {ARABI[gwas][jobs]} --threads {ARABI[gwas][threads]} > {log} 2>&1"


rule arabi_prepare_features:
    output:
        ARABI_FEATURES,
    log:
        str(ARABI_OUT / "features.log"),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/features.py --out {output} > {log} 2>&1"


rule arabi_choose_linear_h2opt_noise:
    input:
        ARABI_FEATURES,
    output:
        noise=str(ARABI_OUT / "{treatment}" / "h2opt" / "noise.npz"),
        traits=str(ARABI_OUT / "{treatment}" / "h2opt" / "noise_traits.npz"),
        pdf=str(ARABI_OUT / "{treatment}" / "figures" / "noise.pdf"),
        png=str(ARABI_OUT / "{treatment}" / "figures" / "noise.png"),
    log:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "noise.log"),
    params:
        folds=" ".join(str(f) for f in ARABI_FOLDS),
    threads: 2
    resources:
        gpu=lambda wc: 1 if ARABI["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {ARABI_SCRIPTS}/choose_noise.py --features {input}"
        " --treatment {wildcards.treatment} --folds {params.folds} --out {output.noise}"
        " --out-traits {output.traits} --figure {output.pdf}"
        " --steps {ARABI[h2opt][steps]} --device {ARABI[h2opt][device]} > {log} 2>&1"


# All folds train in one job (h2opt.train_models).
rule arabi_train_linear_h2opt:
    input:
        features=ARABI_FEATURES,
        noise=str(ARABI_OUT / "{treatment}" / "h2opt" / "noise.npz"),
    output:
        traits=[str(ARABI_OUT / "{treatment}" / "h2opt" / f"fold{f}.npz") for f in ARABI_FOLDS],
        models=[str(ARABI_OUT / "{treatment}" / "h2opt" / f"fold{f}.pt") for f in ARABI_FOLDS],
    log:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "train.log"),
    params:
        folds=" ".join(str(f) for f in ARABI_FOLDS),
        penalty=" ".join(str(v) for v in ARABI["h2opt"]["penalty"]),
    threads: 2
    resources:
        gpu=lambda wc: 1 if ARABI["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {ARABI_SCRIPTS}/train_h2opt.py --features {input.features}"
        " --treatment {wildcards.treatment} --folds {params.folds} --noise {input.noise}"
        " --out-traits {output.traits} --out-models {output.models}"
        " --traits {ARABI[h2opt][traits]} --steps {ARABI[h2opt][steps]} --seed 0"
        " --penalty {params.penalty} --device {ARABI[h2opt][device]} > {log} 2>&1"


rule arabi_train_baselines:
    input:
        ARABI_FEATURES,
    output:
        str(ARABI_OUT / "{treatment}" / "{baseline}" / "fold{fold}.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "{baseline}" / "fold{fold}.log"),
    wildcard_constraints:
        baseline="|".join(ARABI["baselines"]),
        fold=r"\d+",
    threads: 2
    shell:
        "python -u {ARABI_SCRIPTS}/baseline.py --features {input}"
        " --treatment {wildcards.treatment} --fold {wildcards.fold}"
        " --method {wildcards.baseline} --out {output} --traits {ARABI[traits]} > {log} 2>&1"


rule arabi_score_heritability:
    input:
        lambda wc: [arabi_traits(wc.treatment, m, f)
                    for m in ["h2opt"] + ARABI["baselines"] for f in ARABI_FOLDS],
    output:
        str(ARABI_OUT / "{treatment}" / "heritability.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "heritability.log"),
    params:
        folds=" ".join(str(f) for f in ARABI_FOLDS),
        methods=lambda wc: arabi_method_args(wc.treatment),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/evaluate_heritability.py --treatment {wildcards.treatment}"
        " --folds {params.folds} {params.methods} --out {output} --traits {ARABI[traits]}"
        " > {log} 2>&1"


# Figures of the paper (file names as in the paper). The coefficients are from fold ARABI_PAPER_FOLD
# ("the first cross-validation fold"); the GWAS figures (metab0d_GWAS, metab6d_GWAS) use all
# folds. Panel e of metabFig uses the tables of arabi_pathway_enrichment. Run only on request, as
# the GWAS.
ARABI_PAPER = ARABI_OUT / "figures" / "paper"
ARABI_PAPER_FOLD = 0


def arabi_paper_coefficients(treatment):
    return str(ARABI_OUT / treatment / "h2opt" / f"coefficients_fold{ARABI_PAPER_FOLD}.npz")


rule arabidopsis_paper_figures:
    input:
        expand(str(ARABI_PAPER / "{stem}.pdf"),
               stem=["metabFig", "ArabiHeritTrain", "arabiNull"]
                    + ["arabiCoef", "arabiSparsity"]
                    + [f"metab{t}_GWAS" for t in ARABI["treatments"]]),


# arabiNull: the GWAS of the H2-opt traits of all folds against the cross-fit of trait 1 and the
# permutation null (traits 1 to null_traits), in the style of sorNull; one row per treatment.
rule arabi_paper_null:
    input:
        **{f"gwas_{t}": [str(ARABI_OUT / t / "gwas" / f"fold{f}.npz") for f in ARABI_FOLDS]
           for t in ARABI["treatments"]},
        **{f"heritability_{t}": str(ARABI_OUT / t / "heritability.npz")
           for t in ARABI["treatments"]},
        **{f"crossfit_{t}": str(ARABI_OUT / t / "gwas" / "crossfit.npz")
           for t in ARABI["treatments"]},
        **{f"null_{t}": [str(ARABI_OUT / t / "null" / f"perm{k}.npz")
                         for k in range(ARABI["gwas"]["permutations"])]
           for t in ARABI["treatments"]},
    output:
        pdf=str(ARABI_PAPER / "arabiNull.pdf"),
        png=str(ARABI_PAPER / "arabiNull.png"),
    log:
        str(ARABI_PAPER / "arabiNull.log"),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_null.py"
        " --gwas-0d {input.gwas_0d} --heritability-0d {input.heritability_0d}"
        " --crossfit-0d {input.crossfit_0d} --null-0d {input.null_0d}"
        " --gwas-6d {input.gwas_6d} --heritability-6d {input.heritability_6d}"
        " --crossfit-6d {input.crossfit_6d} --null-6d {input.null_6d}"
        " --out {output.pdf} > {log} 2>&1"


rule arabi_trait_coefficients:
    input:
        features=ARABI_FEATURES,
        model=str(ARABI_OUT / "{treatment}" / "h2opt" / "fold{fold}.pt"),
        traits=str(ARABI_OUT / "{treatment}" / "h2opt" / "fold{fold}.npz"),
    output:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "coefficients_fold{fold}.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "coefficients_fold{fold}.log"),
    wildcard_constraints:
        fold=r"\d+",
    threads: 2
    shell:
        "python -u {ARABI_SCRIPTS}/coefficients.py --features {input.features}"
        " --treatment {wildcards.treatment} --fold {wildcards.fold} --model {input.model}"
        " --traits {input.traits} --out {output} > {log} 2>&1"


rule arabi_paper_main_figure:
    input:
        features=ARABI_FEATURES,
        heritability=expand(str(ARABI_OUT / "{treatment}" / "heritability.npz"),
                            treatment=ARABI["treatments"]),
        coefficients=[arabi_paper_coefficients(t) for t in ARABI["treatments"]],
        gwas=[str(ARABI_OUT / t / "gwas" / f"fold{f}.npz") for t in ARABI["treatments"]
              for f in ARABI_FOLDS],
        genotypes=[str(ARABI_OUT / t / "gwas" / f"genotypes{e}") for t in ARABI["treatments"]
                   for e in (".bed", ".bim", ".fam")],
        pathways=expand(str(ARABI_OUT / "{treatment}" / "enrichment" / "trait{k}_pathways.csv"),
                        treatment=ARABI["treatments"],
                        k=range(1, ARABI["h2opt"]["traits"] + 1)),
    output:
        pdf=str(ARABI_PAPER / "metabFig.pdf"),
        png=str(ARABI_PAPER / "metabFig.png"),
    log:
        str(ARABI_PAPER / "metabFig.log"),
    params:
        treatments=" ".join(ARABI["treatments"]),
        genotypes=" ".join(str(ARABI_OUT / t / "gwas" / "genotypes") for t in ARABI["treatments"]),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_main.py --features {input.features}"
        " --treatments {params.treatments} --heritability {input.heritability}"
        " --coefficients {input.coefficients} --gwas {input.gwas}"
        " --genotypes {params.genotypes} --r2 {config[gwas_loci][r2]}"
        " --kb {config[gwas_loci][kb]} --pathways {input.pathways} --out {output.pdf}"
        " > {log} 2>&1"


rule arabi_paper_heritability:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "heritability.npz"),
               treatment=ARABI["treatments"]),
    output:
        pdf=str(ARABI_PAPER / "ArabiHeritTrain.pdf"),
        png=str(ARABI_PAPER / "ArabiHeritTrain.png"),
    log:
        str(ARABI_PAPER / "ArabiHeritTrain.log"),
    params:
        treatments=" ".join(ARABI["treatments"]),
        traits=ARABI["h2opt"]["traits"],
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_heritability.py --treatments {params.treatments}"
        " --heritability {input} --out {output.pdf} --traits {params.traits} > {log} 2>&1"


# arabiCoef: the coefficients of the H2-opt traits of both treatments in the first fold.
rule arabi_paper_coefficients:
    input:
        [arabi_paper_coefficients(t) for t in ARABI["treatments"]],
    output:
        pdf=str(ARABI_PAPER / "arabiCoef.pdf"),
        png=str(ARABI_PAPER / "arabiCoef.png"),
    log:
        str(ARABI_PAPER / "arabiCoef.log"),
    params:
        treatments=" ".join(ARABI["treatments"]),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_coefficients.py --treatments {params.treatments}"
        " --coefficients {input} --out {output.pdf} > {log} 2>&1"


# PCH coefficients per fold (PCH refitted at its saved ridge), for arabiSparsity. All folds of a
# treatment in one job.
rule arabi_pch_coefficients:
    input:
        features=ARABI_FEATURES,
        pch=lambda wc: [arabi_traits(wc.treatment, "pch", f) for f in ARABI_FOLDS],
    output:
        [str(ARABI_OUT / "{treatment}" / "pch_coefficients" / f"fold{f}.npz") for f in ARABI_FOLDS],
    log:
        str(ARABI_OUT / "{treatment}" / "pch_coefficients" / "coefficients.log"),
    params:
        folds=" ".join(str(f) for f in ARABI_FOLDS),
    threads: 4
    shell:
        "python -u {ARABI_SCRIPTS}/pch_coefficients.py --features {input.features}"
        " --treatment {wildcards.treatment} --folds {params.folds} --pch {input.pch}"
        " --out {output} > {log} 2>&1"


def arabi_sparsity_files(treatment):
    """Per fold: H2-opt and PCH coefficients, H2-opt and PCH traits."""
    out = ARABI_OUT / treatment
    return {"h2opt_coefficients": [str(out / "h2opt" / f"coefficients_fold{f}.npz")
                                   for f in ARABI_FOLDS],
            "pch_coefficients": [str(out / "pch_coefficients" / f"fold{f}.npz")
                                 for f in ARABI_FOLDS],
            "h2opt_traits": [arabi_traits(treatment, "h2opt", f) for f in ARABI_FOLDS],
            "pch_traits": [arabi_traits(treatment, "pch", f) for f in ARABI_FOLDS]}


# arabiSparsity: H2-opt (L1 penalty) against PCH, the sparsity of the coefficients, in the style
# of sorCoefficients.
rule arabi_paper_sparsity:
    input:
        [path for t in ARABI["treatments"] for paths in arabi_sparsity_files(t).values()
         for path in paths],
    output:
        pdf=str(ARABI_PAPER / "arabiSparsity.pdf"),
        png=str(ARABI_PAPER / "arabiSparsity.png"),
    log:
        str(ARABI_PAPER / "arabiSparsity.log"),
    params:
        files=" ".join(f"--treatment {t} " + " ".join(
            f"--{kind.replace('_', '-')} " + " ".join(paths)
            for kind, paths in arabi_sparsity_files(t).items()) for t in ARABI["treatments"]),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_sparsity.py {params.files} --out {output.pdf}"
        " > {log} 2>&1"


rule arabi_paper_gwas:
    input:
        gwas=lambda wc: [str(ARABI_OUT / wc.treatment / "gwas" / f"fold{f}.npz")
                         for f in ARABI_FOLDS],
        genotypes=multiext(str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
    output:
        pdf=str(ARABI_PAPER / "metab{treatment}_GWAS.pdf"),
        png=str(ARABI_PAPER / "metab{treatment}_GWAS.png"),
    log:
        str(ARABI_PAPER / "metab{treatment}_GWAS.log"),
    params:
        genotypes=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "genotypes"),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_gwas.py --gwas {input.gwas}"
        " --genotypes {params.genotypes} --r2 {config[gwas_loci][r2]}"
        " --kb {config[gwas_loci][kb]} --out {output.pdf} > {log} 2>&1"


# Peak lists for pathway enrichment (MetaboAnalyst functional analysis): the input of
# arabidopsis_pathway_enrichment, or upload each trait<k>.txt to the web tool. Run only on request.
rule arabidopsis_enrichment_inputs:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "enrichment" / "trait1.txt"),
               treatment=ARABI["treatments"]),


rule arabi_prepare_precise_masses:
    output:
        str(ARABI_OUT / "{treatment}" / "precise_masses.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "precise_masses.log"),
    threads: 2
    resources:
        mem_gb=22,
    shell:
        "python -u {ARABI_SCRIPTS}/precise_masses.py --treatment {wildcards.treatment}"
        " --out {output} > {log} 2>&1"


rule arabi_enrichment_inputs:
    input:
        precise=str(ARABI_OUT / "{treatment}" / "precise_masses.npz"),
        coefficients=str(ARABI_OUT / "{treatment}" / "h2opt"
                         / f"coefficients_fold{ARABI_PAPER_FOLD}.npz"),
    output:
        expand(str(ARABI_OUT / "{{treatment}}" / "enrichment" / "trait{k}.txt"),
               k=range(1, ARABI["h2opt"]["traits"] + 1)),
    log:
        str(ARABI_OUT / "{treatment}" / "enrichment" / "inputs.log"),
    params:
        out_dir=lambda wc: str(ARABI_OUT / wc.treatment / "enrichment"),
    threads: 2
    shell:
        "python -u {ARABI_SCRIPTS}/enrichment_inputs.py --precise {input.precise}"
        " --coefficients {input.coefficients} --peaks {ARABI[enrichment_peaks]}"
        " --out-dir {params.out_dir} > {log} 2>&1"


# MS peaks to pathways of each peak list, offline with MetaboAnalystR, with the settings of
# config arabidopsis.pathways. Not part of arabidopsis_all: run only on request.
ARABI_PATHWAYS = ARABI["pathways"]
ARABI_MUMMICHOG_LIBRARY = str(ARABI_OUT / "pathway_library" / f"{ARABI_PATHWAYS['library']}.qs")


rule arabidopsis_pathway_enrichment:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "enrichment" / "trait{k}_pathways.csv"),
               treatment=ARABI["treatments"], k=range(1, ARABI["h2opt"]["traits"] + 1)),


# The library changes on the server over time: the log records the download date.
rule arabi_download_pathway_library:
    output:
        ARABI_MUMMICHOG_LIBRARY,
    log:
        str(ARABI_OUT / "pathway_library" / "download.log"),
    params:
        url=f"https://www.metaboanalyst.ca/resources/libs/mummichog/{ARABI_PATHWAYS['library']}.qs",
    shell:
        "(date -u && curl -sSfL {params.url} -o {output} && sha256sum {output}) > {log} 2>&1"


rule arabi_pathway_enrichment:
    input:
        peaks=str(ARABI_OUT / "{treatment}" / "enrichment" / "trait{k}.txt"),
        library=ARABI_MUMMICHOG_LIBRARY,
    output:
        str(ARABI_OUT / "{treatment}" / "enrichment" / "trait{k}_pathways.csv"),
    log:
        str(ARABI_OUT / "{treatment}" / "enrichment" / "trait{k}_pathways.log"),
    params:
        settings=" ".join(str(ARABI_PATHWAYS[key])
                          for key in ("mode", "ppm", "primary_ion", "retention_time")),
    threads: 1
    conda:
        "../envs/metaboanalyst.yaml"
    shell:
        "Rscript {ARABI_SCRIPTS}/pathway_enrichment.R {input.peaks} {input.library} {output}"
        " {params.settings} > {log} 2>&1"


rule arabi_null_gwas:
    input:
        features=ARABI_FEATURES,
        noise=str(ARABI_OUT / "{treatment}" / "h2opt" / "noise.npz"),
        genotypes=multiext(str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(ARABI_OUT / "{treatment}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(ARABI_OUT / "{treatment}" / "null" / "perm{k}.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "null" / "perm{k}.log"),
    wildcard_constraints:
        k=r"\d+",
    params:
        prefix=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "genotypes"),
        work=lambda wc: str(ARABI_OUT / wc.treatment / "null" / "work" / f"perm{wc.k}"),
        penalty=" ".join(str(v) for v in ARABI["h2opt"]["penalty"]),
    threads: ARABI["gwas"]["jobs"] * ARABI["gwas"]["threads"]
    resources:
        gpu=lambda wc: 1 if ARABI["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {ARABI_SCRIPTS}/null_gwas.py --features {input.features}"
        " --treatment {wildcards.treatment} --permutation {wildcards.k} --noise {input.noise}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --traits {ARABI[gwas][null_traits]} --steps {ARABI[h2opt][steps]}"
        " --penalty {params.penalty} --device {ARABI[h2opt][device]}"
        " --jobs {ARABI[gwas][jobs]} --threads {ARABI[gwas][threads]} > {log} 2>&1"


rule arabi_cross_fit_traits:
    input:
        [str(ARABI_OUT / "{treatment}" / "h2opt" / f"fold{f}.npz") for f in ARABI_FOLDS],
    output:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "crossfit.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "h2opt" / "crossfit.log"),
    threads: 1
    shell:
        "python -u {PIPELINE_DIR}/scripts/shared/cross_fit_traits.py --dataset arabidopsis"
        " --treatment {wildcards.treatment} --trait 1 --traits {input} --out {output} > {log} 2>&1"


rule arabi_crossfit_gwas:
    input:
        traits=str(ARABI_OUT / "{treatment}" / "h2opt" / "crossfit.npz"),
        genotypes=multiext(str(ARABI_OUT / "{treatment}" / "gwas" / "genotypes"),
                           ".bed", ".bim", ".fam"),
        pcs=str(ARABI_OUT / "{treatment}" / "gwas" / "genotype_pcs.npy"),
    output:
        str(ARABI_OUT / "{treatment}" / "gwas" / "crossfit.npz"),
    log:
        str(ARABI_OUT / "{treatment}" / "gwas" / "crossfit.log"),
    params:
        prefix=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "genotypes"),
        work=lambda wc: str(ARABI_OUT / wc.treatment / "gwas" / "work" / "crossfit"),
    threads: ARABI["gwas"]["threads"]
    shell:
        "python -u {ARABI_SCRIPTS}/gwas.py --treatment {wildcards.treatment}"
        " --h2opt-traits {input.traits} --genotypes {params.prefix} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits 1 --jobs 1"
        " --threads {ARABI[gwas][threads]} > {log} 2>&1"
