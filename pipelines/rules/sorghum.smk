# Sorghum hyperspectral analyses: H2-opt per genotype fold, heritability of
# H2-opt and the baselines, a scan of the H2-opt noise level on held-out
# genotypes, and GWAS of the per-line trait means.
#
# A trait source is one way to get the H2-opt traits of a fold:
#   paper - the predictions behind the paper (raw data, not built here)
#   h2opt - retrained here (train_h2opt.py)

from h2opt_analysis import sorghum
from h2opt_analysis.methods import BASELINES

SORGHUM = config["sorghum"]
SORGHUM_OUT = OUT_ROOT / "sorghum"
SORGHUM_SCRIPTS = PIPELINE_DIR / "scripts" / "sorghum"
SORGHUM_FOLDS = SORGHUM["folds"]
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
        expand(str(SORGHUM_OUT / "noise_scan" / "{variant}" / "noise_scan.npz"),
               variant=SORGHUM["noise_scan"]["variants"]),


rule sorghum_gwas_all:
    input:
        expand(str(SORGHUM_OUT / "gwas" / "{source}_fold{fold}.npz"),
               source=["paper", "h2opt"], fold=SORGHUM_FOLDS),


rule sorghum_train_h2opt:
    output:
        traits=str(SORGHUM_OUT / "h2opt" / "fold{fold}.npz"),
        model=str(SORGHUM_OUT / "h2opt" / "fold{fold}.pt"),
    log:
        str(SORGHUM_OUT / "h2opt" / "fold{fold}.log"),
    params:
        traits=SORGHUM["h2opt"]["traits"],
        steps=SORGHUM["h2opt"]["steps"],
        noise=SORGHUM["h2opt"]["noise"],
        noise_level=SORGHUM["h2opt"]["noise_level"],
        penalty=SORGHUM["h2opt"]["penalty"],
        device=SORGHUM["h2opt"]["device"],
    threads: 1
    resources:
        # Small GPU jobs are limited by Python overhead, so several share one GPU.
        gpu=lambda wc: 1 if SORGHUM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/train_h2opt.py --test-folds {wildcards.fold}"
        " --out-traits {output.traits} --out-model {output.model}"
        " --traits {params.traits} --steps {params.steps}"
        " --noise {params.noise} --noise-level {params.noise_level}"
        " --penalty {params.penalty} --seed {wildcards.fold}"
        " --device {params.device} > {log} 2>&1"


NOISE_SCAN = SORGHUM["noise_scan"]
NOISE_SCAN_OUT = SORGHUM_OUT / "noise_scan"


rule sorghum_noise_scan_all:
    input:
        expand(str(NOISE_SCAN_OUT / "{variant}" / "noise_scan.npz"),
               variant=NOISE_SCAN["variants"]),


rule sorghum_noise_scan_train:
    output:
        traits=str(NOISE_SCAN_OUT / "{variant}" / "sd{level}.npz"),
        model=str(NOISE_SCAN_OUT / "{variant}" / "sd{level}.pt"),
    log:
        str(NOISE_SCAN_OUT / "{variant}" / "sd{level}.log"),
    wildcard_constraints:
        variant="|".join(NOISE_SCAN["variants"]),
        level=r"[0-9.e-]+",
    params:
        test_folds=" ".join(str(f) for f in NOISE_SCAN["test_folds"]),
        traits=NOISE_SCAN["traits"],
        steps=lambda wc: NOISE_SCAN["variants"][wc.variant]["steps"],
        noise_level=lambda wc: float(wc.level) * sorghum.feature_sd(),
        penalty=lambda wc: NOISE_SCAN["variants"][wc.variant]["penalty"],
        device=NOISE_SCAN["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if NOISE_SCAN["device"] == "cuda" else 0,
    shell:
        "python -u {SORGHUM_SCRIPTS}/train_h2opt.py --test-folds {params.test_folds}"
        " --out-traits {output.traits} --out-model {output.model}"
        " --traits {params.traits} --steps {params.steps}"
        " --noise normal --noise-level {params.noise_level} --penalty {params.penalty}"
        " --seed 0 --device {params.device} > {log} 2>&1"


rule sorghum_noise_scan:
    input:
        lambda wc: [str(NOISE_SCAN_OUT / wc.variant / f"sd{level}.npz")
                    for level in NOISE_SCAN["levels"]],
    output:
        npz=str(NOISE_SCAN_OUT / "{variant}" / "noise_scan.npz"),
        pdf=str(SORGHUM_OUT / "figures" / "noise_scan_{variant}.pdf"),
        png=str(SORGHUM_OUT / "figures" / "noise_scan_{variant}.png"),
    log:
        str(NOISE_SCAN_OUT / "{variant}" / "noise_scan.log"),
    wildcard_constraints:
        variant="|".join(NOISE_SCAN["variants"]),
    params:
        test_folds=" ".join(str(f) for f in NOISE_SCAN["test_folds"]),
        levels=lambda wc: " ".join(f"--level {level} {NOISE_SCAN_OUT}/{wc.variant}/sd{level}.npz"
                                   for level in NOISE_SCAN["levels"]),
    threads: 1
    shell:
        "python -u {SORGHUM_SCRIPTS}/noise_scan.py --test-folds {params.test_folds}"
        " {params.levels} --out {output.npz} --figure {output.pdf} > {log} 2>&1"


rule sorghum_baseline:
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


rule sorghum_heritability:
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


rule sorghum_genotype_pcs:
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


rule sorghum_gwas:
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
