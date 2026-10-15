# Image simulation: the paper's three true traits drive leaf greenness, canopy size and tiller
# number of simulated plot images, 10 instances. The images are rendered with the Helios 3D plant
# and radiation simulator (h2opt_analysis.helios_simulation, pipelines/helios/field_render). The
# methods are those of the spectral simulation, with the scripts of scripts/simulation: linear
# models take the flattened pixels, and the CNN of the hybrid is h2opt.ImageConvModel.
#
# Per instance, under helios_simulation/instance{i}/: images.npz, truth.npz and one traits file
# per method ({method}.npz, array 'traits').

IMG = config["image_simulation"]
IMG_SCRIPTS = PIPELINE_DIR / "scripts" / "image_simulation"
IMG_INSTANCES = IMG["instances"]
IMG_ROOT = OUT_ROOT / "helios_simulation"
HELIOS_OUT = OUT_ROOT / "helios"


def img_file(instance, name):
    return str(IMG_ROOT / f"instance{instance}" / name)


def img_files(name, instances=IMG_INSTANCES):
    return [img_file(i, name) for i in instances]


rule image_simulation_all:
    input:
        str(IMG_ROOT / "evaluation.npz"),


rule img_build_helios:
    input:
        PIPELINE_DIR / "helios" / "field_render" / "main.cpp",
        PIPELINE_DIR / "helios" / "field_render" / "CMakeLists.txt",
    output:
        str(HELIOS_OUT / "build" / "field_render"),
    log:
        str(HELIOS_OUT / "build.log"),
    params:
        repository=IMG["helios"]["repository"],
        commit=IMG["helios"]["commit"],
        source=PIPELINE_DIR / "helios" / "field_render",
    threads: 8
    shell:
        "bash {IMG_SCRIPTS}/build_helios.sh {params.repository} {params.commit} {params.source}"
        " {HELIOS_OUT} {threads} > {log} 2>&1"


rule img_render_helios:
    input:
        str(HELIOS_OUT / "build" / "field_render"),
    output:
        images=img_file("{instance}", "images.npz"),
        truth=img_file("{instance}", "truth.npz"),
    log:
        img_file("{instance}", "images.log"),
    params:
        workdir=lambda wc: str(HELIOS_OUT / "tiles" / f"instance{wc.instance}"),
    threads: 2
    # A tile of 196 plots holds about 1 GB of GPU memory while it renders.
    resources:
        gpu=2,
    shell:
        "python -u {IMG_SCRIPTS}/render_helios.py --instance {wildcards.instance}"
        " --binary {input} --workdir {params.workdir} --out-images {output.images}"
        " --out-truth {output.truth} > {log} 2>&1"


rule img_train_linear_h2opt:
    input:
        measurements=img_file("{instance}", "images.npz"),
        noise=img_file("{instance}", "noise.npz"),
    output:
        traits=img_file("{instance}", "h2opt.npz"),
        model=img_file("{instance}", "h2opt.pt"),
    log:
        img_file("{instance}", "h2opt.log"),
    params:
        traits=IMG["traits"],
        steps=IMG["h2opt"]["steps"],
        device=IMG["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if IMG["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/train_h2opt.py --measurements {input.measurements}"
        " --noise {input.noise} --out-traits {output.traits} --out-model {output.model}"
        " --traits {params.traits} --steps {params.steps} --seed {wildcards.instance}"
        " --instance {wildcards.instance} --device {params.device} > {log} 2>&1"


# One fixed linear level and one fixed CNN level for every instance, not selected (config
# image_simulation.h2opt.noise_level and image_simulation.hybrid.cnn_level).
rule img_fixed_levels:
    input:
        img_file("{instance}", "images.npz"),
    output:
        noise=img_file("{instance}", "noise.npz"),
        cnn_level=img_file("{instance}", "cnn_level.npz"),
    log:
        img_file("{instance}", "levels.log"),
    threads: 1
    shell:
        "python -u {IMG_SCRIPTS}/fixed_levels.py --measurements {input}"
        " --linear-level {IMG[h2opt][noise_level]} --cnn-level {IMG[hybrid][cnn_level]}"
        " --out-noise {output.noise} --out-cnn-level {output.cnn_level} > {log} 2>&1"


rule img_train_hybrid_h2opt:
    input:
        measurements=img_file("{instance}", "images.npz"),
        noise=img_file("{instance}", "noise.npz"),
        cnn_level=img_file("{instance}", "cnn_level.npz"),
    output:
        traits=img_file("{instance}", "h2opt_hybrid.npz"),
        model=img_file("{instance}", "h2opt_hybrid.pt"),
    log:
        img_file("{instance}", "h2opt_hybrid.log"),
    params:
        traits=IMG["traits"],
        linear_steps=IMG["hybrid"]["linear_steps"],
        steps=IMG["hybrid"]["steps"],
        max_copies=IMG["hybrid"]["max_copies"],
        device=IMG["hybrid"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 4 if IMG["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/train_hybrid.py --measurements {input.measurements}"
        " --noise {input.noise} --cnn-level {input.cnn_level}"
        " --out-traits {output.traits} --out-model {output.model} --traits {params.traits}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --max-copies {params.max_copies} --seed {wildcards.instance}"
        " --instance {wildcards.instance} --device {params.device} > {log} 2>&1"


rule img_train_baselines:
    input:
        img_file("{instance}", "images.npz"),
    output:
        img_file("{instance}", "{baseline}.npz"),
    log:
        img_file("{instance}", "{baseline}.log"),
    params:
        traits=IMG["traits"],
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/baseline.py --measurements {input} --method {wildcards.baseline}"
        " --traits {params.traits} --out {output} > {log} 2>&1"


rule img_score_traits:
    input:
        measurements=img_files("images.npz"),
        truth=img_files("truth.npz"),
        traits=[f for m in IMG["methods"] for f in img_files(f"{m}.npz")],
    output:
        str(IMG_ROOT / "evaluation.npz"),
    log:
        str(IMG_ROOT / "evaluation.log"),
    params:
        methods=" ".join(f"--method {m} " + " ".join(img_files(f"{m}.npz"))
                         for m in IMG["methods"]),
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/evaluate.py --measurements {input.measurements}"
        " --truth {input.truth} {params.methods} --out {output} > {log} 2>&1"


# GWAS of the image simulations, as the spectral simulation's (simulation.smk): the instances have
# the plants, lines and true traits of the spectral instances, so the genotypes, the plant lines
# (spectra.npz 'lines') and the GWAS of the true traits come from there. Run only on request.
IMG_GWAS_METHODS = IMG["methods"]


rule image_simulation_gwas_all:
    input:
        str(IMG_ROOT / "roc.npz"),


rule img_run_gwas:
    input:
        lines=sim_file("{instance}", "spectra.npz"),
        traits=img_file("{instance}", "{method}.npz"),
        genotypes=multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    output:
        img_file("{instance}", "gwas/{method}.npz"),
    log:
        img_file("{instance}", "gwas/{method}.log"),
    wildcard_constraints:
        method="|".join(IMG_GWAS_METHODS),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        work=lambda wc: img_file(wc.instance, f"gwas/work/{wc.method}"),
        jobs=SIM["gwas"]["jobs"],
        threads=SIM["gwas"]["threads"],
    threads: SIM["gwas"]["jobs"] * SIM["gwas"]["threads"]
    shell:
        "python -u {SIM_SCRIPTS}/gwas.py --spectra {input.lines} --traits {input.traits}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --jobs {params.jobs} --threads {params.threads} > {log} 2>&1"


def img_gwas_files(method):
    if method == "truth":
        return sim_files("gwas/truth.npz", IMG_INSTANCES)
    return img_files(f"gwas/{method}.npz")


rule img_score_gwas:
    input:
        [f for m in ["truth"] + IMG_GWAS_METHODS for f in img_gwas_files(m)],
    output:
        str(IMG_ROOT / "roc.npz"),
    log:
        str(IMG_ROOT / "roc.log"),
    params:
        instances=" ".join(str(i) for i in IMG_INSTANCES),
        methods=" ".join(f"--method {m} " + " ".join(img_gwas_files(m))
                         for m in ["truth"] + IMG_GWAS_METHODS),
    threads: 2
    shell:
        "python -u {SIM_SCRIPTS}/roc.py --instances {params.instances} {params.methods}"
        " --out {output} > {log} 2>&1"


# The permutation null of the hybrid, as for the spectral simulation (simulation_null_gwas_all).
# Run only on request.
rule image_simulation_null_gwas_all:
    input:
        [img_file(SIM_NULL["null_instance"], f"null/perm{k}.npz")
         for k in range(SIM_NULL["permutations"])],


rule img_null_gwas:
    input:
        measurements=img_file("{instance}", "images.npz"),
        lines=sim_file("{instance}", "spectra.npz"),
        noise=img_file("{instance}", "noise.npz"),
        cnn_level=img_file("{instance}", "cnn_level.npz"),
        genotypes=multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    output:
        img_file("{instance}", "null/perm{k}.npz"),
    log:
        img_file("{instance}", "null/perm{k}.log"),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        work=lambda wc: img_file(wc.instance, f"null/work/perm{wc.k}"),
        linear_steps=IMG["hybrid"]["linear_steps"],
        steps=IMG["hybrid"]["steps"],
        max_copies=IMG["hybrid"]["max_copies"],
        device=IMG["hybrid"]["device"],
        jobs=SIM["gwas"]["jobs"],
        threads=SIM["gwas"]["threads"],
    threads: SIM["gwas"]["jobs"] * SIM["gwas"]["threads"]
    resources:
        gpu=lambda wc: 4 if IMG["hybrid"]["device"] == "cuda" else 0,
    shell:
        "python -u {IMG_SCRIPTS}/null_gwas.py --measurements {input.measurements}"
        " --lines {input.lines} --noise {input.noise} --cnn-level {input.cnn_level}"
        " --instance {wildcards.instance} --permutation {wildcards.k}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --max-copies {params.max_copies} --device {params.device}"
        " --jobs {params.jobs} --threads {params.threads} > {log} 2>&1"
