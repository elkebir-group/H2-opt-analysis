# Image simulation: the paper's three true traits drive leaf greenness, canopy size and tiller
# number of simulated plot images, 10 instances, from two image sources:
# - image_simulation: hand-made images (h2opt_analysis.image_simulation);
# - helios_simulation: images rendered with the Helios 3D plant and radiation simulator
#   (h2opt_analysis.helios_simulation, pipelines/helios/field_render).
# The methods are those of the spectral simulation, with the scripts of scripts/simulation: linear
# models take the flattened pixels, and the CNN of the hybrid is h2opt.ImageConvModel.
#
# Per instance, under {imgsim}/instance{i}/: images.npz, truth.npz and one traits file per method
# ({method}.npz, array 'traits').

IMG = config["image_simulation"]
IMG_SCRIPTS = PIPELINE_DIR / "scripts" / "image_simulation"
IMG_INSTANCES = IMG["instances"]
IMG_SOURCES = ["image_simulation", "helios_simulation"]
HELIOS_OUT = OUT_ROOT / "helios"

wildcard_constraints:
    imgsim="|".join(IMG_SOURCES),


def img_file(imgsim, instance, name):
    return str(OUT_ROOT / imgsim / f"instance{instance}" / name)


def img_files(imgsim, name, instances=IMG_INSTANCES):
    return [img_file(imgsim, i, name) for i in instances]


rule image_simulation_all:
    input:
        [str(OUT_ROOT / s / "evaluation.npz") for s in IMG_SOURCES],
        [str(OUT_ROOT / s / "figures" / "heritability.pdf") for s in IMG_SOURCES],


rule img_simulate_images:
    output:
        images=img_file("image_simulation", "{instance}", "images.npz"),
        truth=img_file("image_simulation", "{instance}", "truth.npz"),
    log:
        img_file("image_simulation", "{instance}", "images.log"),
    threads: 1
    shell:
        "python -u {IMG_SCRIPTS}/simulate.py --instance {wildcards.instance}"
        " --out-images {output.images} --out-truth {output.truth} > {log} 2>&1"


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
        images=img_file("helios_simulation", "{instance}", "images.npz"),
        truth=img_file("helios_simulation", "{instance}", "truth.npz"),
    log:
        img_file("helios_simulation", "{instance}", "images.log"),
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
        measurements=img_file("{imgsim}", "{instance}", "images.npz"),
        noise=img_file("{imgsim}", "{instance}", "noise.npz"),
    output:
        traits=img_file("{imgsim}", "{instance}", "h2opt.npz"),
        model=img_file("{imgsim}", "{instance}", "h2opt.pt"),
    log:
        img_file("{imgsim}", "{instance}", "h2opt.log"),
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
        " --device {params.device} > {log} 2>&1"


# One fixed linear level and one fixed CNN level for every instance of both image sources (PI: no
# selection here; config image_simulation.h2opt.noise_level and image_simulation.hybrid.cnn_level).
rule img_fixed_levels:
    input:
        img_file("{imgsim}", "{instance}", "images.npz"),
    output:
        noise=img_file("{imgsim}", "{instance}", "noise.npz"),
        cnn_level=img_file("{imgsim}", "{instance}", "cnn_level.npz"),
    log:
        img_file("{imgsim}", "{instance}", "levels.log"),
    threads: 1
    shell:
        "python -u {IMG_SCRIPTS}/fixed_levels.py --measurements {input}"
        " --linear-level {IMG[h2opt][noise_level]} --cnn-level {IMG[hybrid][cnn_level]}"
        " --out-noise {output.noise} --out-cnn-level {output.cnn_level} > {log} 2>&1"


rule img_train_hybrid_h2opt:
    input:
        measurements=img_file("{imgsim}", "{instance}", "images.npz"),
        noise=img_file("{imgsim}", "{instance}", "noise.npz"),
        cnn_level=img_file("{imgsim}", "{instance}", "cnn_level.npz"),
    output:
        traits=img_file("{imgsim}", "{instance}", "h2opt_hybrid.npz"),
        model=img_file("{imgsim}", "{instance}", "h2opt_hybrid.pt"),
    log:
        img_file("{imgsim}", "{instance}", "h2opt_hybrid.log"),
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
        " --device {params.device} > {log} 2>&1"


rule img_train_baselines:
    input:
        img_file("{imgsim}", "{instance}", "images.npz"),
    output:
        img_file("{imgsim}", "{instance}", "{baseline}.npz"),
    log:
        img_file("{imgsim}", "{instance}", "{baseline}.log"),
    params:
        traits=IMG["traits"],
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/baseline.py --measurements {input} --method {wildcards.baseline}"
        " --traits {params.traits} --out {output} > {log} 2>&1"


rule img_score_traits:
    input:
        measurements=lambda wc: img_files(wc.imgsim, "images.npz"),
        truth=lambda wc: img_files(wc.imgsim, "truth.npz"),
        traits=lambda wc: [f for m in IMG["methods"] for f in img_files(wc.imgsim, f"{m}.npz")],
    output:
        str(OUT_ROOT / "{imgsim}" / "evaluation.npz"),
    log:
        str(OUT_ROOT / "{imgsim}" / "evaluation.log"),
    params:
        methods=lambda wc: " ".join(f"--method {m} " + " ".join(img_files(wc.imgsim, f"{m}.npz"))
                                    for m in IMG["methods"]),
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/evaluate.py --measurements {input.measurements}"
        " --truth {input.truth} {params.methods} --out {output} > {log} 2>&1"


rule img_plot_traits:
    input:
        str(OUT_ROOT / "{imgsim}" / "evaluation.npz"),
    output:
        str(OUT_ROOT / "{imgsim}" / "figures" / "heritability.pdf"),
        str(OUT_ROOT / "{imgsim}" / "figures" / "correlation.pdf"),
    log:
        str(OUT_ROOT / "{imgsim}" / "figures" / "plot_traits.log"),
    params:
        out_dir=lambda wc: str(OUT_ROOT / wc.imgsim / "figures"),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_traits.py --evaluation {input}"
        " --out-dir {params.out_dir} > {log} 2>&1"


# GWAS of the image simulations, as the spectral simulation's (simulation.smk): the instances have
# the plants, lines and true traits of the spectral instances, so the genotypes, the plant lines
# (spectra.npz 'lines') and the GWAS of the true traits come from there. Run only on request.
IMG_GWAS_METHODS = IMG["methods"]


rule image_simulation_gwas_all:
    input:
        str(OUT_ROOT / "helios_simulation" / "roc.npz"),
        str(OUT_ROOT / "helios_simulation" / "figures" / "auc.pdf"),


rule img_run_gwas:
    input:
        lines=sim_file("{instance}", "spectra.npz"),
        traits=img_file("{imgsim}", "{instance}", "{method}.npz"),
        genotypes=multiext(sim_file("{instance}", "genotypes"), ".bed", ".bim", ".fam"),
        pcs=sim_file("{instance}", "genotype_pcs.npy"),
    output:
        img_file("{imgsim}", "{instance}", "gwas/{method}.npz"),
    log:
        img_file("{imgsim}", "{instance}", "gwas/{method}.log"),
    wildcard_constraints:
        method="|".join(IMG_GWAS_METHODS),
    params:
        prefix=lambda wc: sim_file(wc.instance, "genotypes"),
        work=lambda wc: img_file(wc.imgsim, wc.instance, f"gwas/work/{wc.method}"),
        jobs=SIM["gwas"]["jobs"],
        threads=SIM["gwas"]["threads"],
    threads: SIM["gwas"]["jobs"] * SIM["gwas"]["threads"]
    shell:
        "python -u {SIM_SCRIPTS}/gwas.py --spectra {input.lines} --traits {input.traits}"
        " --genotypes {params.prefix} --pcs {input.pcs} --out {output} --work {params.work}"
        " --jobs {params.jobs} --threads {params.threads} > {log} 2>&1"


def img_gwas_files(imgsim, method):
    if method == "truth":
        return sim_files("gwas/truth.npz", IMG_INSTANCES)
    return img_files(imgsim, f"gwas/{method}.npz")


rule img_score_gwas:
    input:
        lambda wc: [f for m in ["truth"] + IMG_GWAS_METHODS for f in img_gwas_files(wc.imgsim, m)],
    output:
        str(OUT_ROOT / "{imgsim}" / "roc.npz"),
    log:
        str(OUT_ROOT / "{imgsim}" / "roc.log"),
    params:
        instances=" ".join(str(i) for i in IMG_INSTANCES),
        methods=lambda wc: " ".join(f"--method {m} " + " ".join(img_gwas_files(wc.imgsim, m))
                                    for m in ["truth"] + IMG_GWAS_METHODS),
    threads: 2
    shell:
        "python -u {SIM_SCRIPTS}/roc.py --instances {params.instances} {params.methods}"
        " --out {output} > {log} 2>&1"


rule img_plot_gwas:
    input:
        roc=str(OUT_ROOT / "{imgsim}" / "roc.npz"),
        example=img_file("{imgsim}", IMG["example"]["instance"],
                         f"gwas/{IMG['example']['method']}.npz"),
    output:
        [str(OUT_ROOT / "{imgsim}" / "figures" / name) for name in
         ["auc.pdf", "roc_trait1.pdf", "roc_trait2.pdf", "roc_trait3.pdf",
          f"gwas_{IMG['example']['method']}_instance{IMG['example']['instance']}.png"]],
    log:
        str(OUT_ROOT / "{imgsim}" / "figures" / "plot_gwas.log"),
    params:
        out_dir=lambda wc: str(OUT_ROOT / wc.imgsim / "figures"),
        name=f"{IMG['example']['method']}_instance{IMG['example']['instance']}",
        label=(f"{LABELS[IMG['example']['method']]}, "
               f"instance {IMG['example']['instance']}"),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_gwas.py --roc {input.roc}"
        " --gwas-example {input.example} --example-name {params.name}"
        " --example-label '{params.label}' --out-dir {params.out_dir} > {log} 2>&1"
