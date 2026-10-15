# Sorghum hyperspectral analyses: the H2-opt noise level per genotype fold and
# trait, H2-opt per fold, heritability of H2-opt and the baselines, and GWAS of the per-line
# trait means.

from h2opt_analysis import sorghum
from h2opt_analysis.methods import BASELINES

SORGHUM = config["sorghum"]
SORGHUM_OUT = OUT_ROOT / "sorghum"
SORGHUM_SCRIPTS = PIPELINE_DIR / "scripts" / "sorghum"
SORGHUM_FOLDS = SORGHUM["folds"]
SORGHUM_PAPER_FIGURES = SORGHUM_OUT / "figures" / "paper"
# Genotype means of the measured traits (Ferguson et al. 2021, Supplemental Table S12).
SORGHUM_MEASURED_TRAITS = sorghum.MEASURED_TRAITS
# Reference traits of the GWAS: NDVI and every measured trait.
SORGHUM_REFERENCE_TRAITS = ["ndvi"] + [sorghum.trait_slug(n) for n in sorghum.measured_traits()[0]]
assert SORGHUM["h2opt"]["traits"] <= SORGHUM["baselines"]["traits"], \
    "sorghum.baselines.traits must cover the traits of H2-opt"

wildcard_constraints:
    fold=r"\d+",


def sorghum_traits(fold):
    """H2-opt traits file of one fold."""
    return str(SORGHUM_OUT / "h2opt" / f"fold{fold}.npz")


def sorghum_baseline(baseline, fold):
    """Traits file of one fold for one baseline."""
    return str(SORGHUM_OUT / baseline / f"fold{fold}.npz")


def sorghum_method_args():
    """'--method slug file0 ... file9' for H2-opt and every baseline."""
    files = {"h2opt": [sorghum_traits(f) for f in SORGHUM_FOLDS]}
    files.update({b: [sorghum_baseline(b, f) for f in SORGHUM_FOLDS] for b in BASELINES})
    return " ".join(f"--method {m} " + " ".join(fs) for m, fs in files.items())


# Traits and their heritability only. The GWAS is a separate target, sorghum_gwas_all, run only
# on request.
rule sorghum_all:
    input:
        str(SORGHUM_OUT / "figures" / "noise.pdf"),
        str(SORGHUM_OUT / "ndvi.npz"),
        expand(str(SORGHUM_OUT / "pch_coefficients" / "fold{fold}.npz"), fold=SORGHUM_FOLDS),


rule sorghum_gwas_all:
    input:
        expand(str(SORGHUM_OUT / "gwas" / "h2opt_fold{fold}.npz"), fold=SORGHUM_FOLDS),


# The GWAS of the reference traits (NDVI and the measured traits) at the loci of the H2-opt traits.
# Run only on request.
rule sorghum_reference_gwas_all:
    input:
        str(SORGHUM_OUT / "gwas" / "reference" / "summary.tsv"),


# The permutation null of the whole H2-opt -> GWAS pipeline, and the GWAS of the cross-fitted
# traits 1 to crossfit_traits (each line's phenotype from the fold model that held it out). Run
# only on request.
rule sorghum_null_gwas_all:
    input:
        expand(str(SORGHUM_OUT / "null" / "perm{k}.npz"),
               k=range(SORGHUM["gwas"]["permutations"])),
        expand(str(SORGHUM_OUT / "gwas" / "h2opt_crossfit_trait{trait}.npz"),
               trait=range(1, SORGHUM["gwas"]["crossfit_traits"] + 1)),


rule sorghum_choose_linear_h2opt_noise:
    output:
        noise=str(SORGHUM_OUT / "h2opt" / "noise.npz"),
        traits=str(SORGHUM_OUT / "h2opt" / "noise_traits.npz"),
        pdf=str(SORGHUM_OUT / "figures" / "noise.pdf"),
        png=str(SORGHUM_OUT / "figures" / "noise.png"),
    log:
        str(SORGHUM_OUT / "h2opt" / "noise.log"),
    params:
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
        steps=SORGHUM["h2opt"]["steps"],
        device=SORGHUM["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if SORGHUM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/choose_noise.py --folds {params.folds} --out {output.noise}"
        " --out-traits {output.traits} --figure {output.pdf} --steps {params.steps}"
        " --device {params.device} > {log} 2>&1"


# All folds train in one job (h2opt.train_models), which is much faster than one job per fold.
rule sorghum_train_linear_h2opt:
    input:
        noise=str(SORGHUM_OUT / "h2opt" / "noise.npz"),
    output:
        traits=[str(SORGHUM_OUT / "h2opt" / f"fold{f}.npz") for f in SORGHUM_FOLDS],
        models=[str(SORGHUM_OUT / "h2opt" / f"fold{f}.pt") for f in SORGHUM_FOLDS],
    log:
        str(SORGHUM_OUT / "h2opt" / "train.log"),
    params:
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
        traits=SORGHUM["h2opt"]["traits"],
        steps=SORGHUM["h2opt"]["steps"],
        device=SORGHUM["h2opt"]["device"],
        penalty=" ".join(str(v) for v in SORGHUM["h2opt"]["penalty"]),
    threads: 1
    resources:
        gpu=lambda wc: 1 if SORGHUM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/train_h2opt.py --folds {params.folds} --noise {input.noise}"
        " --out-traits {output.traits} --out-models {output.models}"
        " --traits {params.traits} --steps {params.steps} --seed 0"
        " --penalty {params.penalty} --device {params.device} > {log} 2>&1"


rule sorghum_train_baselines:
    output:
        str(SORGHUM_OUT / "{baseline}" / "fold{fold}.npz"),
    log:
        str(SORGHUM_OUT / "{baseline}" / "fold{fold}.log"),
    wildcard_constraints:
        baseline="|".join(BASELINES),
    params:
        traits=SORGHUM["baselines"]["traits"],
    threads: 2
    shell:
        "python -u {SORGHUM_SCRIPTS}/baseline.py --fold {wildcards.fold}"
        " --method {wildcards.baseline} --out {output} --traits {params.traits} > {log} 2>&1"


# Wavelength coefficients of PCH (refitted at the saved ridge) and of linear H2-opt per fold, and
# their smoothness (total variation over the sum of magnitudes). All folds in one job.
rule sorghum_pch_coefficients:
    input:
        pch=[sorghum_baseline("pch", f) for f in SORGHUM_FOLDS],
        models=[str(SORGHUM_OUT / "h2opt" / f"fold{f}.pt") for f in SORGHUM_FOLDS],
        traits=[sorghum_traits(f) for f in SORGHUM_FOLDS],
    output:
        [str(SORGHUM_OUT / "pch_coefficients" / f"fold{f}.npz") for f in SORGHUM_FOLDS],
    log:
        str(SORGHUM_OUT / "pch_coefficients" / "coefficients.log"),
    params:
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
    threads: 2
    shell:
        "python -u {SORGHUM_SCRIPTS}/pch_coefficients.py --folds {params.folds} --pch {input.pch}"
        " --h2opt {input.models} --h2opt-traits {input.traits} --out {output} > {log} 2>&1"


rule sorghum_score_heritability:
    input:
        h2opt=[sorghum_traits(fold) for fold in SORGHUM_FOLDS],
        baselines=[sorghum_baseline(b, f) for b in BASELINES for f in SORGHUM_FOLDS],
    output:
        str(SORGHUM_OUT / "heritability_h2opt.npz"),
    log:
        str(SORGHUM_OUT / "heritability_h2opt.log"),
    params:
        traits=SORGHUM["h2opt"]["traits"],
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
        methods=sorghum_method_args(),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/evaluate_heritability.py --folds {params.folds}"
        " {params.methods} --out {output} --traits {params.traits} > {log} 2>&1"


# NDVI is a fixed index: its heritability on the training and the test plants of each fold.
rule sorghum_score_ndvi:
    output:
        str(SORGHUM_OUT / "ndvi.npz"),
    log:
        str(SORGHUM_OUT / "ndvi.log"),
    params:
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/ndvi.py --folds {params.folds} --out {output} > {log} 2>&1"


rule sorghum_prepare_genotype_pcs:
    input:
        multiext(str(sorghum.GENOTYPES), ".bed", ".bim", ".fam"),
    output:
        str(SORGHUM_OUT / "genotype_pcs.npy"),
    log:
        str(SORGHUM_OUT / "genotype_pcs.log"),
    params:
        pcs=SORGHUM["gwas"]["pcs"],
    threads: 4
    shell:
        "python -u {SORGHUM_SCRIPTS}/genotype_pcs.py --out {output} --pcs {params.pcs}"
        " > {log} 2>&1"


rule sorghum_run_gwas:
    input:
        traits=lambda wc: sorghum_traits(wc.fold),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "gwas" / "h2opt_fold{fold}.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "h2opt_fold{fold}.log"),
    params:
        work=lambda wc: str(SORGHUM_OUT / "gwas" / "work" / f"h2opt_fold{wc.fold}"),
        traits=SORGHUM["gwas"]["traits"],
        jobs=SORGHUM["gwas"]["jobs"],
        threads=SORGHUM["gwas"]["threads"],
    threads: SORGHUM["gwas"]["jobs"] * SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/gwas.py --h2opt-traits {input.traits} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits {params.traits}"
        " --jobs {params.jobs} --threads {params.threads}"
        " > {log} 2>&1"


rule sorghum_reference_gwas:
    input:
        ndvi=str(SORGHUM_OUT / "ndvi.npz"),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "gwas" / "reference" / "{trait}.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "reference" / "{trait}.log"),
    wildcard_constraints:
        trait="|".join(SORGHUM_REFERENCE_TRAITS),
    params:
        work=lambda wc: str(SORGHUM_OUT / "gwas" / "work" / f"reference_{wc.trait}"),
    threads: SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/reference_gwas.py --trait {wildcards.trait}"
        " --ndvi {input.ndvi} --pcs {input.pcs} --out {output} --work {params.work}"
        " --threads {SORGHUM[gwas][threads]} > {log} 2>&1"


rule sorghum_reference_gwas_summary:
    input:
        h2opt=[str(SORGHUM_OUT / "gwas" / f"h2opt_fold{f}.npz") for f in SORGHUM_FOLDS],
        reference=expand(str(SORGHUM_OUT / "gwas" / "reference" / "{trait}.npz"),
                         trait=SORGHUM_REFERENCE_TRAITS),
        genotypes=multiext(str(sorghum.GENOTYPES), ".bed", ".bim", ".fam"),
    output:
        str(SORGHUM_OUT / "gwas" / "reference" / "summary.tsv"),
    log:
        str(SORGHUM_OUT / "gwas" / "reference" / "summary.log"),
    params:
        window=SORGHUM["gwas"]["locus_window"],
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/reference_gwas_summary.py --h2opt {input.h2opt}"
        " --genotypes {sorghum.GENOTYPES} --r2 {config[gwas_loci][r2]}"
        " --kb {config[gwas_loci][kb]} --reference {input.reference}"
        " --window {params.window} --out {output} > {log} 2>&1"


# The figures of the paper, made from linear H2-opt. They need
# the GWAS, so they are not part of sorghum_all.
rule sorghum_paper_figures:
    input:
        expand(str(SORGHUM_PAPER_FIGURES / "{stem}.pdf"),
               stem=["sorFig", "multiTraitHerit_all", "sorCoefficients", "sorGWAS",
                     "sorNull"]),


rule sorghum_plot_paper_main:
    input:
        heritability=str(SORGHUM_OUT / "heritability_h2opt.npz"),
        coefficients=str(SORGHUM_OUT / "pch_coefficients" / f"fold{SORGHUM_FOLDS[0]}.npz"),
        traits=[sorghum_traits(f) for f in SORGHUM_FOLDS],
        measured=str(SORGHUM_MEASURED_TRAITS),
        ndvi=str(SORGHUM_OUT / "ndvi.npz"),
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "sorFig.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "sorFig.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "sorFig.log"),
    params:
        fold_traits=sorghum_traits(SORGHUM_FOLDS[0]),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_main.py --heritability {input.heritability}"
        " --coefficients {input.coefficients} --fold-traits {params.fold_traits}"
        " --traits-files {input.traits} --measured-traits {input.measured}"
        " --ndvi {input.ndvi} --out {output.pdf} --traits 4 > {log} 2>&1"


rule sorghum_plot_paper_heritability:
    input:
        str(SORGHUM_OUT / "heritability_h2opt.npz"),
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "multiTraitHerit_all.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "multiTraitHerit_all.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "multiTraitHerit_all.log"),
    params:
        traits=SORGHUM["h2opt"]["traits"],
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_heritability.py --heritability {input}"
        " --out {output.pdf} --traits {params.traits} > {log} 2>&1"


# sorCoefficients: wavelength coefficients of H2-opt and of the paired PCH traits in the first fold,
# a zoom on trait 1 (zoom, nm) and the total variation of the coefficients over the folds.
rule sorghum_plot_paper_coefficients:
    input:
        [str(SORGHUM_OUT / "pch_coefficients" / f"fold{f}.npz") for f in SORGHUM_FOLDS],
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "sorCoefficients.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "sorCoefficients.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "sorCoefficients.log"),
    params:
        fold=SORGHUM_FOLDS[0],
        zoom="400 800",
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_coefficients.py --fold {params.fold}"
        " --coefficients {input} --zoom {params.zoom} --out {output.pdf} > {log} 2>&1"

rule sorghum_plot_paper_gwas:
    input:
        gwas=[str(SORGHUM_OUT / "gwas" / f"h2opt_fold{f}.npz") for f in SORGHUM_FOLDS],
        crossfit=expand(str(SORGHUM_OUT / "gwas" / "h2opt_crossfit_trait{trait}.npz"),
                        trait=range(1, SORGHUM["gwas"]["crossfit_traits"] + 1)),
        genotypes=multiext(str(sorghum.GENOTYPES), ".bed", ".bim", ".fam"),
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "sorGWAS.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "sorGWAS.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "sorGWAS.log"),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_gwas.py --gwas {input.gwas}"
        " --crossfit {input.crossfit} --genotypes {sorghum.GENOTYPES}"
        " --r2 {config[gwas_loci][r2]} --kb {config[gwas_loci][kb]} --out {output.pdf}"
        " > {log} 2>&1"


# sorNull: the GWAS of the H2-opt traits of all folds against their cross-fit and the permutation
# null (test heritability, largest -log10 p, QQ plot, lambda GC), in the style of simNull.
rule sorghum_plot_paper_null:
    input:
        gwas=[str(SORGHUM_OUT / "gwas" / f"h2opt_fold{f}.npz") for f in SORGHUM_FOLDS],
        heritability=str(SORGHUM_OUT / "heritability_h2opt.npz"),
        nulls=expand(str(SORGHUM_OUT / "null" / "perm{k}.npz"),
                     k=range(SORGHUM["gwas"]["permutations"])),
        crossfit=expand(str(SORGHUM_OUT / "gwas" / "h2opt_crossfit_trait{trait}.npz"),
                        trait=range(1, SORGHUM["gwas"]["crossfit_traits"] + 1)),
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "sorNull.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "sorNull.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "sorNull.log"),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_null.py --gwas {input.gwas}"
        " --heritability {input.heritability} --null {input.nulls} --crossfit {input.crossfit}"
        " --out {output.pdf} > {log} 2>&1"


rule sorghum_null_gwas:
    input:
        noise=str(SORGHUM_OUT / "h2opt" / "noise.npz"),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "null" / "perm{k}.npz"),
    log:
        str(SORGHUM_OUT / "null" / "perm{k}.log"),
    wildcard_constraints:
        k=r"\d+",
    params:
        work=lambda wc: str(SORGHUM_OUT / "null" / "work" / f"perm{wc.k}"),
        penalty=" ".join(str(v) for v in SORGHUM["h2opt"]["penalty"]),
    threads: SORGHUM["gwas"]["jobs"] * SORGHUM["gwas"]["threads"]
    resources:
        gpu=lambda wc: 1 if SORGHUM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/null_gwas.py --permutation {wildcards.k}"
        " --noise {input.noise} --pcs {input.pcs} --out {output} --work {params.work}"
        " --traits {SORGHUM[gwas][traits]} --steps {SORGHUM[h2opt][steps]}"
        " --penalty {params.penalty} --device {SORGHUM[h2opt][device]}"
        " --jobs {SORGHUM[gwas][jobs]} --threads {SORGHUM[gwas][threads]} > {log} 2>&1"


rule sorghum_cross_fit_traits:
    input:
        [str(SORGHUM_OUT / "h2opt" / f"fold{f}.npz") for f in SORGHUM_FOLDS],
    output:
        str(SORGHUM_OUT / "h2opt" / "crossfit_trait{trait}.npz"),
    log:
        str(SORGHUM_OUT / "h2opt" / "crossfit_trait{trait}.log"),
    wildcard_constraints:
        trait=r"\d+",
    threads: 1
    shell:
        "python -u {PIPELINE_DIR}/scripts/shared/cross_fit_traits.py --dataset sorghum"
        " --trait {wildcards.trait} --traits {input} --out {output} > {log} 2>&1"


rule sorghum_crossfit_gwas:
    input:
        traits=str(SORGHUM_OUT / "h2opt" / "crossfit_trait{trait}.npz"),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "gwas" / "h2opt_crossfit_trait{trait}.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "h2opt_crossfit_trait{trait}.log"),
    wildcard_constraints:
        trait=r"\d+",
    params:
        work=lambda wc: str(SORGHUM_OUT / "gwas" / "work" / f"h2opt_crossfit_trait{wc.trait}"),
    threads: SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/gwas.py --h2opt-traits {input.traits} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits 1 --jobs 1"
        " --threads {SORGHUM[gwas][threads]} > {log} 2>&1"
