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
        expand(str(ARABI_OUT / "{treatment}" / "figures" / "heritability_test.pdf"),
               treatment=ARABI["treatments"]),
        expand(str(ARABI_OUT / "{treatment}" / "figures" / "noise.pdf"),
               treatment=ARABI["treatments"]),


# GWAS of the H2-opt traits, every treatment and fold. Run only on request.
rule arabidopsis_gwas_all:
    input:
        expand(str(ARABI_OUT / "{treatment}" / "gwas" / "fold{fold}.npz"),
               treatment=ARABI["treatments"], fold=ARABI_FOLDS),


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
        penalty=" ".join(str(v) for v in ARABI["h2opt"]["penalty"]),
    threads: 2
    resources:
        gpu=lambda wc: 1 if ARABI["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {ARABI_SCRIPTS}/choose_noise.py --features {input}"
        " --treatment {wildcards.treatment} --folds {params.folds} --out {output.noise}"
        " --out-traits {output.traits} --figure {output.pdf}"
        " --steps {ARABI[h2opt][steps]} --penalty {params.penalty}"
        " --device {ARABI[h2opt][device]} > {log} 2>&1"


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


rule arabi_plot_heritability:
    input:
        str(ARABI_OUT / "{treatment}" / "heritability.npz"),
    output:
        test=str(ARABI_OUT / "{treatment}" / "figures" / "heritability_test.pdf"),
        train=str(ARABI_OUT / "{treatment}" / "figures" / "heritability_train.pdf"),
    log:
        str(ARABI_OUT / "{treatment}" / "figures" / "heritability.log"),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_heritability.py --heritability {input}"
        " --out-test {output.test} --out-train {output.train} > {log} 2>&1"


# Figures of the paper (main.tex image names), from fold ARABI_PAPER_FOLD ("the first
# cross-validation split") and the GWAS of that fold. Panel e of metabFig (pathway enrichment) is
# not made: its input (the mummichog files) is missing. Run only on request, as the GWAS.
ARABI_PAPER = ARABI_OUT / "figures" / "paper"
ARABI_PAPER_FOLD = 0


def arabi_paper_coefficients(treatment):
    return str(ARABI_OUT / treatment / "h2opt" / f"coefficients_fold{ARABI_PAPER_FOLD}.npz")


def arabi_paper_gwas(treatment):
    return str(ARABI_OUT / treatment / "gwas" / f"fold{ARABI_PAPER_FOLD}.npz")


rule arabidopsis_paper_figures:
    input:
        expand(str(ARABI_PAPER / "{stem}.pdf"),
               stem=["metabFig", "ArabiHeritTrain"]
                    + [f"coef_{t.removesuffix('d')}day" for t in ARABI["treatments"]]
                    + [f"metab{t}_GWAS" for t in ARABI["treatments"]]),


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
        coefficients=arabi_paper_coefficients(ARABI["treatments"][0]),
        gwas=[arabi_paper_gwas(t) for t in ARABI["treatments"]],
    output:
        pdf=str(ARABI_PAPER / "metabFig.pdf"),
        png=str(ARABI_PAPER / "metabFig.png"),
    log:
        str(ARABI_PAPER / "metabFig.log"),
    params:
        treatments=" ".join(ARABI["treatments"]),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_main.py --features {input.features}"
        " --treatments {params.treatments} --heritability {input.heritability}"
        " --coefficients {input.coefficients} --gwas {input.gwas} --out {output.pdf}"
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
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_heritability.py --treatments {params.treatments}"
        " --heritability {input} --out {output.pdf} > {log} 2>&1"


# coef_0day for treatment 0d, coef_6day for 6d.
rule arabi_paper_coefficients:
    input:
        lambda wc: arabi_paper_coefficients(f"{wc.days}d"),
    output:
        pdf=str(ARABI_PAPER / "coef_{days}day.pdf"),
        png=str(ARABI_PAPER / "coef_{days}day.png"),
    log:
        str(ARABI_PAPER / "coef_{days}day.log"),
    wildcard_constraints:
        days="|".join(t.removesuffix("d") for t in ARABI["treatments"]),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_coefficients.py --coefficients {input}"
        " --title '{wildcards.days} day, fold {ARABI_PAPER_FOLD}' --out {output.pdf} > {log} 2>&1"


rule arabi_paper_gwas:
    input:
        lambda wc: arabi_paper_gwas(wc.treatment),
    output:
        pdf=str(ARABI_PAPER / "metab{treatment}_GWAS.pdf"),
        png=str(ARABI_PAPER / "metab{treatment}_GWAS.png"),
    log:
        str(ARABI_PAPER / "metab{treatment}_GWAS.log"),
    threads: 1
    shell:
        "python -u {ARABI_SCRIPTS}/plot_paper_gwas.py --gwas {input}"
        " --title '{wildcards.treatment}, fold {ARABI_PAPER_FOLD}' --out {output.pdf} > {log} 2>&1"


# Peak lists for pathway enrichment (MetaboAnalyst functional analysis, web tool, default
# settings): upload each trait<k>.txt by hand. Run only on request.
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

