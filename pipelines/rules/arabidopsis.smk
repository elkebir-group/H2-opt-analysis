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
    threads: 2
    resources:
        gpu=lambda wc: 1 if ARABI["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {ARABI_SCRIPTS}/train_h2opt.py --features {input.features}"
        " --treatment {wildcards.treatment} --folds {params.folds} --noise {input.noise}"
        " --out-traits {output.traits} --out-models {output.models}"
        " --traits {ARABI[h2opt][traits]} --steps {ARABI[h2opt][steps]} --seed 0"
        " --device {ARABI[h2opt][device]} > {log} 2>&1"


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
