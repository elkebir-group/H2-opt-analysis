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
        str(SIM_OUT / "figures" / "heritability.pdf"),


rule simulation_gwas_all:
    input:
        str(SIM_OUT / "roc.npz"),
        str(SIM_OUT / "figures" / "auc.pdf"),


rule sim_encode:
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


rule sim_choose_noise:
    input:
        sim_file("{instance}", "spectra.npz"),
    output:
        sim_file("{instance}", "h2opt_noise.npz"),
    log:
        sim_file("{instance}", "h2opt_noise.log"),
    params:
        steps=SIM["h2opt"]["steps"],
        device=SIM["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if SIM["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/choose_noise.py --spectra {input} --out {output}"
        " --steps {params.steps} --seed {wildcards.instance} --device {params.device}"
        " > {log} 2>&1"


rule sim_train_h2opt:
    input:
        spectra=sim_file("{instance}", "spectra.npz"),
        noise=sim_file("{instance}", "h2opt_noise.npz"),
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
        "python -u {SIM_SCRIPTS}/train_h2opt.py --spectra {input.spectra} --noise {input.noise}"
        " --out-traits {output.traits} --out-model {output.model} --traits {params.traits}"
        " --steps {params.steps} --seed {wildcards.instance} --device {params.device}"
        " > {log} 2>&1"


rule sim_baseline:
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
        "python -u {SIM_SCRIPTS}/baseline.py --spectra {input} --method {wildcards.baseline}"
        " --out {output} --traits {params.traits} > {log} 2>&1"


rule sim_genotypes:
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


rule sim_gwas:
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


rule sim_evaluate:
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
        "python -u {SIM_SCRIPTS}/evaluate.py --spectra {input.spectra} --truth {input.truth}"
        " {params.methods} --out {output} > {log} 2>&1"


rule sim_roc:
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


rule sim_plot_traits:
    input:
        str(SIM_OUT / "evaluation.npz"),
    output:
        str(SIM_OUT / "figures" / "heritability.pdf"),
        str(SIM_OUT / "figures" / "correlation.pdf"),
    log:
        str(SIM_OUT / "figures" / "plot_traits.log"),
    params:
        out_dir=str(SIM_OUT / "figures"),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_traits.py --evaluation {input}"
        " --out-dir {params.out_dir} > {log} 2>&1"


rule sim_plot_gwas:
    input:
        roc=str(SIM_OUT / "roc.npz"),
        example=sim_file(SIM["example"]["instance"], f"gwas/{SIM['example']['method']}.npz"),
    output:
        [str(SIM_OUT / "figures" / name) for name in
         ["auc.pdf", "roc_trait1.pdf", "roc_trait2.pdf", "roc_trait3.pdf",
          f"gwas_{SIM['example']['method']}_instance{SIM['example']['instance']}.png"]],
    log:
        str(SIM_OUT / "figures" / "plot_gwas.log"),
    params:
        out_dir=str(SIM_OUT / "figures"),
        name=f"{SIM['example']['method']}_instance{SIM['example']['instance']}",
        label=(f"{LABELS[SIM['example']['method']]}, "
               f"instance {SIM['example']['instance']}"),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_gwas.py --roc {input.roc}"
        " --gwas-example {input.example} --example-name {params.name}"
        " --example-label '{params.label}' --out-dir {params.out_dir} > {log} 2>&1"
