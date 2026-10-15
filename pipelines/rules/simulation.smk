# The paper's simulation (h2opt_analysis.simulation): three heritable traits encoded into
# simulated sorghum spectra, 10 instances. H2-opt and the baselines are trained on the training
# lines of each instance and compared with the true traits by correlation, heritability and the
# detection of the causal SNPs by GWAS.
#
# Per instance, under simulation/instance{i}/: spectra.npz, truth.npz, one traits file per
# method ({method}.npz, array 'traits'), the GWAS genotypes, and gwas/{method}.npz.

from h2opt_analysis import simulation, sorghum
from h2opt_analysis.methods import BASELINES, LABELS

SIM = config["simulation"]
SIM_OUT = OUT_ROOT / "simulation"
SIM_SCRIPTS = PIPELINE_DIR / "scripts" / "simulation"
SIM_INSTANCES = SIM["instances"]
SIM_GWAS_METHODS = ["truth"] + SIM["methods"]

wildcard_constraints:
    instance=r"\d+",
    method="|".join(LABELS),
    baseline="|".join(BASELINES),


def sim_file(instance, name):
    return str(SIM_OUT / f"instance{instance}" / name)


def sim_files(name, instances=SIM_INSTANCES):
    return [sim_file(i, name) for i in instances]


def sim_method_args(name_pattern):
    """'--method slug file0 ... file9' for every method, files from name_pattern.format(slug)."""
    return " ".join(
        f"--method {m} " + " ".join(sim_files(name_pattern.format(m)))
        for m in SIM["methods"])


# Traits and their heritability only. The GWAS (causal-SNP detection) is a separate target,
# simulation_gwas_all, run only on request once the methods and their tuning are settled.
rule simulation_all:
    input:
        str(SIM_OUT / "evaluation.npz"),


rule simulation_gwas_all:
    input:
        str(SIM_OUT / "roc.npz"),


rule sim_simulate_spectra:
    input:
        autoencoder=str(simulation.PAPER_AUTOENCODER),
        reference=str(sorghum.SPECTRA),
        truth=[str(simulation.truth_file(i)) for i in range(simulation.N_INSTANCES)],
    output:
        spectra=sim_files("spectra.npz", range(simulation.N_INSTANCES)),
        truth=sim_files("truth.npz", range(simulation.N_INSTANCES)),
    log:
        str(SIM_OUT / "encode.log"),
    params:
        out_dirs=[str(SIM_OUT / f"instance{i}") for i in range(simulation.N_INSTANCES)],
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/encode.py --autoencoder {input.autoencoder}"
        " --reference {input.reference} --out-dirs {params.out_dirs} > {log} 2>&1"


rule sim_train_linear_h2opt:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        noise=sim_file("{instance}", "noise.npz"),
    output:
        traits=sim_file("{instance}", "h2opt.npz"),
        model=sim_file("{instance}", "h2opt.pt"),
    log:
        sim_file("{instance}", "h2opt.log"),
    params:
        traits=SIM["traits"],
        steps=SIM["h2opt"]["steps"],
        device=SIM["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if SIM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/train_h2opt.py --measurements {input.spectra} --noise {input.noise}"
        " --out-traits {output.traits} --out-model {output.model} --traits {params.traits}"
        " --steps {params.steps} --seed {wildcards.instance} --instance {wildcards.instance}"
        " --device {params.device} > {log} 2>&1"


rule sim_choose_linear_h2opt_noise:
    input:
        sim_file("{instance}", "spectra.npz"),
    output:
        sim_file("{instance}", "noise.npz"),
    log:
        sim_file("{instance}", "noise.log"),
    params:
        steps=SIM["h2opt"]["steps"],
        device=SIM["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if SIM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/choose_noise.py --measurements {input} --out {output}"
        " --steps {params.steps} --seed {wildcards.instance} --device {params.device}"
        " > {log} 2>&1"


rule sim_choose_hybrid_cnn_level:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        noise=sim_file("{instance}", "noise.npz"),
    output:
        sim_file("{instance}", "cnn_level.npz"),
    log:
        sim_file("{instance}", "cnn_level.log"),
    params:
        linear_steps=SIM["hybrid"]["linear_steps"],
        steps=SIM["hybrid"]["steps"],
        device=SIM["hybrid"]["device"],
    threads: 1
    # All levels and splits train at once (105 models, about 4 GB of GPU memory): half of
    # run.sh's default --gpu-jobs, so at most two jobs share the GPU.
    resources:
        gpu=lambda wc: 4 if SIM["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/choose_cnn_level.py --measurements {input.spectra}"
        " --noise {input.noise} --out {output}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --seed {wildcards.instance} --device {params.device} > {log} 2>&1"


rule sim_train_hybrid_h2opt:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        noise=sim_file("{instance}", "noise.npz"),
        cnn_level=sim_file("{instance}", "cnn_level.npz"),
    output:
        traits=sim_file("{instance}", "h2opt_hybrid.npz"),
        model=sim_file("{instance}", "h2opt_hybrid.pt"),
    log:
        sim_file("{instance}", "h2opt_hybrid.log"),
    params:
        traits=SIM["traits"],
        linear_steps=SIM["hybrid"]["linear_steps"],
        steps=SIM["hybrid"]["steps"],
        device=SIM["hybrid"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if SIM["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/train_hybrid.py --measurements {input.spectra} --noise {input.noise}"
        " --cnn-level {input.cnn_level}"
        " --out-traits {output.traits} --out-model {output.model} --traits {params.traits}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --seed {wildcards.instance} --instance {wildcards.instance}"
        " --device {params.device} > {log} 2>&1"


rule sim_train_baselines:
    input:
        sim_file("{instance}", "spectra.npz"),
    output:
        sim_file("{instance}", "{baseline}.npz"),
    log:
        sim_file("{instance}", "{baseline}.log"),
    params:
        traits=SIM["traits"],
    threads: 2
    shell:
        "python -u {SIM_SCRIPTS}/baseline.py --measurements {input}"
        " --method {wildcards.baseline} --out {output} --traits {params.traits} > {log} 2>&1"


rule sim_prepare_gwas_genotypes:
    input:
        multiext(str(sorghum.GENOTYPES), ".bed", ".bim", ".fam"),
    output:
        multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    log:
        sim_file("{instance}", "genotypes.log"),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        pcs=SIM["gwas"]["pcs"],
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/genotypes.py --instance {wildcards.instance}"
        " --out-prefix {params.prefix} --out-pcs {output.pcs} --pcs {params.pcs} > {log} 2>&1"


rule sim_run_gwas:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        traits=sim_file("{instance}", "{method}.npz"),
        genotypes=multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    output:
        sim_file("{instance}", "gwas/{method}.npz"),
    log:
        sim_file("{instance}", "gwas/{method}.log"),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        work=lambda wc: sim_file(wc.instance, f"gwas/work/{wc.method}"),
        jobs=SIM["gwas"]["jobs"],
        threads=SIM["gwas"]["threads"],
    threads: SIM["gwas"]["jobs"] * SIM["gwas"]["threads"]
    shell:
        "python -u {SIM_SCRIPTS}/gwas.py --spectra {input.spectra} --traits {input.traits}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --jobs {params.jobs} --threads {params.threads} > {log} 2>&1"


rule sim_score_traits:
    input:
        spectra=sim_files("spectra.npz"),
        truth=sim_files("truth.npz"),
        traits=[sim_files(f"{m}.npz") for m in SIM["methods"]],
    output:
        str(SIM_OUT / "evaluation.npz"),
    log:
        str(SIM_OUT / "evaluation.log"),
    params:
        methods=sim_method_args("{}.npz"),
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/evaluate.py --measurements {input.spectra} --truth {input.truth}"
        " {params.methods} --out {output} > {log} 2>&1"


rule sim_score_gwas:
    input:
        [sim_files(f"gwas/{m}.npz") for m in SIM_GWAS_METHODS],
    output:
        str(SIM_OUT / "roc.npz"),
    log:
        str(SIM_OUT / "roc.log"),
    params:
        instances=" ".join(str(i) for i in SIM_INSTANCES),
        methods=" ".join(
            f"--method {m} " + " ".join(sim_files(f"gwas/{m}.npz")) for m in SIM_GWAS_METHODS),
    threads: 2
    shell:
        "python -u {SIM_SCRIPTS}/roc.py --instances {params.instances} {params.methods}"
        " --out {output} > {log} 2>&1"


# The simulation figures of the paper (file stems as in the paper's images/), from the saved
# results. They need the GWAS outputs, so they are not part of simulation_all. simFig has one row
# per simulation of simulation.paper_figures.rows, each with its H2-opt variant
# (simulation.paper_figures.h2opt); simCompareSNP has the same rows; simGWASexample shows the
# spectral simulation.
SIM_PAPER = SIM_OUT / "figures" / "paper"
SIM_PAPER_ROWS = SIM["paper_figures"]["rows"]
SIM_PAPER_H2OPT = SIM["paper_figures"]["h2opt"]


rule simulation_paper_figures:
    input:
        [str(SIM_PAPER / f"{stem}.{ext}")
         for stem in ["simFig", "simCompareSNP", "simGWASexample", "simNull", "simLinear"]
         for ext in ["pdf", "png"]],


rule sim_plot_paper_figures:
    input:
        evaluations=[str(OUT_ROOT / s / "evaluation.npz") for s in SIM_PAPER_ROWS],
        rocs=[str(OUT_ROOT / s / "roc.npz") for s in SIM_PAPER_ROWS],
        example=sim_file(SIM["example"]["instance"],
                         f"gwas/{SIM_PAPER_H2OPT['simulation']}.npz"),
    output:
        figure=str(SIM_PAPER / "simFig.pdf"),
        figure_png=str(SIM_PAPER / "simFig.png"),
        compare=str(SIM_PAPER / "simCompareSNP.pdf"),
        compare_png=str(SIM_PAPER / "simCompareSNP.png"),
        example=str(SIM_PAPER / "simGWASexample.pdf"),
        example_png=str(SIM_PAPER / "simGWASexample.png"),
    log:
        str(SIM_PAPER / "plot_paper.log"),
    params:
        baselines=" ".join(SIM["paper_figures"]["baselines"]),
        rows=lambda wc, input: " ".join(
            f"--row '{label}' {ev} {roc} {SIM_PAPER_H2OPT[s]}" for (s, label), ev, roc in
            zip(SIM_PAPER_ROWS.items(), input.evaluations, input.rocs, strict=True)),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_paper.py {params.rows}"
        " --baselines {params.baselines} --gwas-example {input.example}"
        " --out-figure {output.figure} --out-compare {output.compare}"
        " --out-example {output.example} > {log} 2>&1"


# Selection bias of single trait (response letter): training vs test heritability of every
# measurement of one instance of each simulation, with the single trait picks marked.
rule sim_plot_selection_bias:
    input:
        spectra=sim_file(SIM["selection_bias"]["instance"], "spectra.npz"),
        images=str(OUT_ROOT / "helios_simulation" / f"instance{SIM['selection_bias']['instance']}"
                   / "images.npz"),
    output:
        str(SIM_PAPER / "selectionBias.pdf"),
    log:
        str(SIM_PAPER / "selectionBias.log"),
    params:
        traits=SIM["selection_bias"]["traits"],
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/plot_selection_bias.py --row Hyperspectral {input.spectra}"
        " --row Images {input.images} --traits {params.traits} --out {output} > {log} 2>&1"


# simNull: per simulation of simulation.paper_figures.rows, the real H2-opt traits of the null
# instance (simulation.paper_figures.h2opt) against the permutation null (sim_null_gwas,
# img_null_gwas): test heritability, largest -log10 p, QQ plot of trait 1 and lambda GC of traits
# 1-3.


def sim_null_files(sim, name):
    return str(OUT_ROOT / sim / f"instance{SIM['gwas']['null_instance']}" / name)


def sim_null_perms(sim):
    return [sim_null_files(sim, f"null/perm{k}.npz") for k in range(SIM["gwas"]["permutations"])]


rule sim_plot_paper_null:
    input:
        evaluations=[str(OUT_ROOT / s / "evaluation.npz") for s in SIM_PAPER_ROWS],
        gwas=[sim_null_files(s, f"gwas/{SIM_PAPER_H2OPT[s]}.npz") for s in SIM_PAPER_ROWS],
        nulls=[f for s in SIM_PAPER_ROWS for f in sim_null_perms(s)],
    output:
        figure=str(SIM_PAPER / "simNull.pdf"),
        figure_png=str(SIM_PAPER / "simNull.png"),
    log:
        str(SIM_PAPER / "plot_paper_null.log"),
    params:
        rows=" ".join(
            f"--label '{label}' --slug {SIM_PAPER_H2OPT[s]}"
            f" --evaluation {OUT_ROOT / s / 'evaluation.npz'}"
            f" --gwas {sim_null_files(s, f'gwas/{SIM_PAPER_H2OPT[s]}.npz')}"
            f" --null {' '.join(sim_null_perms(s))}"
            for s, label in SIM_PAPER_ROWS.items()),
        instance=SIM["gwas"]["null_instance"],
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_paper_null.py {params.rows} --instance {params.instance}"
        " --out {output.figure} > {log} 2>&1"


# simLinear: per simulation of simulation.paper_figures.rows, the hybrid and the linear H2-opt
# against PCH and the ground truth: test heritability, |r| with the true traits and AUC.
rule sim_plot_paper_linear:
    input:
        evaluations=[str(OUT_ROOT / s / "evaluation.npz") for s in SIM_PAPER_ROWS],
        rocs=[str(OUT_ROOT / s / "roc.npz") for s in SIM_PAPER_ROWS],
    output:
        figure=str(SIM_PAPER / "simLinear.pdf"),
        figure_png=str(SIM_PAPER / "simLinear.png"),
    log:
        str(SIM_PAPER / "plot_paper_linear.log"),
    params:
        rows=lambda wc, input: " ".join(
            f"--row '{label}' {ev} {roc}" for label, ev, roc in
            zip(SIM_PAPER_ROWS.values(), input.evaluations, input.rocs, strict=True)),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_paper_linear.py {params.rows}"
        " --out {output.figure} > {log} 2>&1"


# The permutation null of the GWAS (sim_null_gwas). Run only on request.
SIM_NULL = SIM["gwas"]


rule simulation_null_gwas_all:
    input:
        [sim_file(SIM_NULL["null_instance"], f"null/perm{k}.npz")
         for k in range(SIM_NULL["permutations"])],


rule sim_null_gwas:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        noise=sim_file("{instance}", "noise.npz"),
        genotypes=multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    output:
        sim_file("{instance}", "null/perm{k}.npz"),
    log:
        sim_file("{instance}", "null/perm{k}.log"),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        work=lambda wc: sim_file(wc.instance, f"null/work/perm{wc.k}"),
        steps=SIM["h2opt"]["steps"],
        device=SIM["h2opt"]["device"],
        jobs=SIM["gwas"]["jobs"],
        threads=SIM["gwas"]["threads"],
    threads: SIM["gwas"]["jobs"] * SIM["gwas"]["threads"]
    resources:
        gpu=lambda wc: 1 if SIM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/null_gwas.py --measurements {input.spectra}"
        " --lines {input.spectra} --noise {input.noise} --instance {wildcards.instance}"
        " --permutation {wildcards.k} --genotypes {params.prefix} --pcs {input.pcs}"
        " --out {output} --work {params.work} --steps {params.steps} --device {params.device}"
        " --jobs {params.jobs} --threads {params.threads} > {log} 2>&1"
