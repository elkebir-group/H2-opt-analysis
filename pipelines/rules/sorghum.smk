# Sorghum hyperspectral analyses: the H2-opt noise level per genotype fold and
# trait, H2-opt per fold, heritability of H2-opt and the baselines, and GWAS of the per-line
# trait means.
#
# A trait source is one way to get the H2-opt traits of a fold:
#   paper - the predictions behind the paper (raw data, not built here)
#   h2opt - retrained here (train_h2opt.py)

from h2opt_analysis import sorghum
from h2opt_analysis.methods import BASELINES
import h2opt_analysis.paths

SORGHUM = config["sorghum"]
SORGHUM_OUT = OUT_ROOT / "sorghum"
SORGHUM_SCRIPTS = PIPELINE_DIR / "scripts" / "sorghum"
SORGHUM_FOLDS = SORGHUM["folds"]
SORGHUM_PAPER_FIGURES = SORGHUM_OUT / "figures" / "paper"
# Genotype means of the measured traits (Ferguson et al. 2021, Supplemental Table S12).
SORGHUM_MEASURED_TRAITS = h2opt_analysis.paths.data(
    "plant", "processed", "traits", "SorgumTraits.tsv")
# Method slug (h2opt_analysis.methods.LABELS) of each trait source.
SORGHUM_SOURCES = {"paper": "h2opt_paper", "h2opt": "h2opt"}
assert all(SORGHUM[s]["traits"] <= SORGHUM["baselines"]["traits"] for s in SORGHUM_SOURCES), \
    "sorghum.baselines.traits must cover the traits of every source"

wildcard_constraints:
    fold=r"\d+",
    source="|".join(SORGHUM_SOURCES),


def sorghum_traits(source, fold):
    """Traits file of one fold for one trait source."""
    if source == "paper":
        return str(sorghum.paper_traits(fold))
    return str(SORGHUM_OUT / "h2opt" / f"fold{fold}.npz")


def sorghum_baseline(baseline, fold):
    """Traits file of one fold for one baseline."""
    return str(SORGHUM_OUT / baseline / f"fold{fold}.npz")


def sorghum_method_args(source):
    """'--method slug file0 ... file9' for the H2-opt source and every baseline."""
    files = {SORGHUM_SOURCES[source]: [sorghum_traits(source, f) for f in SORGHUM_FOLDS]}
    files.update({b: [sorghum_baseline(b, f) for f in SORGHUM_FOLDS] for b in BASELINES})
    return " ".join(f"--method {m} " + " ".join(fs) for m, fs in files.items())


# Traits and their heritability only. The GWAS is a separate target, sorghum_gwas_all, run only
# on request once the methods and their tuning are settled.
rule sorghum_all:
    input:
        expand(str(SORGHUM_OUT / "figures" / "top_trait_heritability_{source}.pdf"),
               source=SORGHUM_SOURCES),
        str(SORGHUM_OUT / "figures" / "noise.pdf"),
        str(SORGHUM_OUT / "ndvi.npz"),


rule sorghum_gwas_all:
    input:
        expand(str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.npz"),
               source=["paper", "h2opt"], fold=SORGHUM_FOLDS),


# The permutation null of the whole H2-opt -> GWAS pipeline, and the GWAS of the cross-fitted
# trait 1 (each line's phenotype from the fold model that held it out). Run only on request.
rule sorghum_null_gwas_all:
    input:
        expand(str(SORGHUM_OUT / "null" / "perm{k}.npz"),
               k=range(SORGHUM["gwas"]["permutations"])),
        str(SORGHUM_OUT / "gwas" / "h2opt_crossfit.npz"),


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


rule sorghum_score_heritability:
    input:
        h2opt=lambda wc: [sorghum_traits(wc.source, fold) for fold in SORGHUM_FOLDS],
        baselines=[sorghum_baseline(b, f) for b in BASELINES for f in SORGHUM_FOLDS],
    output:
        str(SORGHUM_OUT / "heritability_{source}.npz"),
    log:
        str(SORGHUM_OUT / "heritability_{source}.log"),
    params:
        traits=lambda wc: SORGHUM[wc.source]["traits"],
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
        methods=lambda wc: sorghum_method_args(wc.source),
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
        traits=lambda wc: sorghum_traits(wc.source, wc.fold),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.log"),
    params:
        work=lambda wc: str(SORGHUM_OUT / "gwas" / "work" / f"{wc.source}_fold{wc.fold}"),
        traits=SORGHUM["gwas"]["traits"],
        jobs=SORGHUM["gwas"]["jobs"],
        threads=SORGHUM["gwas"]["threads"],
    threads: SORGHUM["gwas"]["jobs"] * SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/gwas.py --h2opt-traits {input.traits} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits {params.traits}"
        " --jobs {params.jobs} --threads {params.threads}"
        " > {log} 2>&1"


# The figures of the paper, made from the h2opt source (linear H2-opt retrained here). They need
# the GWAS, so they are not part of sorghum_all.
rule sorghum_paper_figures:
    input:
        expand(str(SORGHUM_PAPER_FIGURES / "{stem}.pdf"),
               stem=["sorFig", "multiTraitHerit_all", "crossGWAS", "sorGWAS"]),


rule sorghum_plot_paper_main:
    input:
        heritability=str(SORGHUM_OUT / "heritability_h2opt.npz"),
        model=str(SORGHUM_OUT / "h2opt" / f"fold{SORGHUM_FOLDS[0]}.pt"),
        traits=[sorghum_traits("h2opt", f) for f in SORGHUM_FOLDS],
        measured=str(SORGHUM_MEASURED_TRAITS),
        gwas=str(SORGHUM_OUT / "gwas" / f"h2opt_fold{SORGHUM_FOLDS[0]}.npz"),
    output:
        pdf=str(SORGHUM_PAPER_FIGURES / "sorFig.pdf"),
        png=str(SORGHUM_PAPER_FIGURES / "sorFig.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "sorFig.log"),
    params:
        fold=SORGHUM_FOLDS[0],
        fold_traits=sorghum_traits("h2opt", SORGHUM_FOLDS[0]),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_main.py --heritability {input.heritability}"
        " --model {input.model} --fold-traits {params.fold_traits} --fold {params.fold}"
        " --traits-files {input.traits} --measured-traits {input.measured} --gwas {input.gwas}"
        " --out {output.pdf} --traits 4 > {log} 2>&1"


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


rule sorghum_plot_paper_gwas:
    input:
        [str(SORGHUM_OUT / "gwas" / f"h2opt_fold{f}.npz") for f in SORGHUM_FOLDS],
    output:
        folds=str(SORGHUM_PAPER_FIGURES / "crossGWAS.pdf"),
        folds_png=str(SORGHUM_PAPER_FIGURES / "crossGWAS.png"),
        manhattan=str(SORGHUM_PAPER_FIGURES / "sorGWAS.pdf"),
        manhattan_png=str(SORGHUM_PAPER_FIGURES / "sorGWAS.png"),
    log:
        str(SORGHUM_PAPER_FIGURES / "gwas.log"),
    params:
        folds=" ".join(str(f) for f in SORGHUM_FOLDS),
        example=SORGHUM_FOLDS[0],
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/plot_paper_gwas.py --folds {params.folds} --gwas {input}"
        " --trait 1 --example-fold {params.example} --out-folds {output.folds}"
        " --out-manhattan {output.manhattan} > {log} 2>&1"


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
        str(SORGHUM_OUT / "h2opt" / "crossfit.npz"),
    log:
        str(SORGHUM_OUT / "h2opt" / "crossfit.log"),
    threads: 1
    shell:
        "python -u {PIPELINE_DIR}/scripts/shared/cross_fit_traits.py --dataset sorghum"
        " --traits {input} --out {output} > {log} 2>&1"


rule sorghum_crossfit_gwas:
    input:
        traits=str(SORGHUM_OUT / "h2opt" / "crossfit.npz"),
        pcs=str(SORGHUM_OUT / "genotype_pcs.npy"),
    output:
        str(SORGHUM_OUT / "gwas" / "h2opt_crossfit.npz"),
    log:
        str(SORGHUM_OUT / "gwas" / "h2opt_crossfit.log"),
    params:
        work=str(SORGHUM_OUT / "gwas" / "work" / "h2opt_crossfit"),
    threads: SORGHUM["gwas"]["threads"]
    shell:
        "python -u {SORGHUM_SCRIPTS}/gwas.py --h2opt-traits {input.traits} --pcs {input.pcs}"
        " --out {output} --work {params.work} --traits 1 --jobs 1"
        " --threads {SORGHUM[gwas][threads]} > {log} 2>&1"
