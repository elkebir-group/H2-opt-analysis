# Image simulation (h2opt_analysis.image_simulation): the paper's three true traits drive leaf
# greenness, canopy size and tiller number of simulated plot images, 10 instances. The methods
# are those of the spectral simulation, with the scripts of scripts/simulation: linear models
# take the flattened pixels, and the CNN of the hybrid is h2opt.ImageConvModel.
#
# Per instance, under image_simulation/instance{i}/: images.npz, truth.npz and one traits file
# per method ({method}.npz, array 'traits').

IMG = config["image_simulation"]
IMG_OUT = OUT_ROOT / "image_simulation"
IMG_SCRIPTS = PIPELINE_DIR / "scripts" / "image_simulation"
IMG_INSTANCES = IMG["instances"]


def img_file(instance, name):
    return str(IMG_OUT / f"instance{instance}" / name)


def img_files(name, instances=IMG_INSTANCES):
    return [img_file(i, name) for i in instances]


rule image_simulation_all:
    input:
        str(IMG_OUT / "evaluation.npz"),
        str(IMG_OUT / "figures" / "heritability.pdf"),


rule img_simulate_images:
    output:
        images=img_file("{instance}", "images.npz"),
        truth=img_file("{instance}", "truth.npz"),
    log:
        img_file("{instance}", "images.log"),
    threads: 1
    shell:
        "python -u {IMG_SCRIPTS}/simulate.py --instance {wildcards.instance}"
        " --out-images {output.images} --out-truth {output.truth} > {log} 2>&1"


rule img_choose_linear_h2opt_noise:
    input:
        img_file("{instance}", "images.npz"),
    output:
        img_file("{instance}", "noise.npz"),
    log:
        img_file("{instance}", "noise.log"),
    params:
        steps=IMG["h2opt"]["steps"],
        device=IMG["h2opt"]["device"],
    threads: 1
    resources:
        gpu=lambda wc: 1 if IMG["h2opt"]["device"] == "cuda" else 0,
    shell:
        "python -u {SIM_SCRIPTS}/choose_noise.py --measurements {input} --out {output}"
        " --steps {params.steps} --seed {wildcards.instance} --device {params.device}"
        " > {log} 2>&1"


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
        " --device {params.device} > {log} 2>&1"


rule img_train_hybrid_h2opt:
    input:
        measurements=img_file("{instance}", "images.npz"),
        noise=img_file("{instance}", "noise.npz"),
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
        " --noise {input.noise}"
        " --out-traits {output.traits} --out-model {output.model} --traits {params.traits}"
        " --linear-steps {params.linear_steps} --steps {params.steps}"
        " --max-copies {params.max_copies} --seed {wildcards.instance}"
        " --device {params.device} > {log} 2>&1"


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
        traits=[img_files(f"{m}.npz") for m in IMG["methods"]],
    output:
        str(IMG_OUT / "evaluation.npz"),
    log:
        str(IMG_OUT / "evaluation.log"),
    params:
        methods=" ".join(f"--method {m} " + " ".join(img_files(f"{m}.npz"))
                         for m in IMG["methods"]),
    threads: 4
    shell:
        "python -u {SIM_SCRIPTS}/evaluate.py --measurements {input.measurements}"
        " --truth {input.truth} {params.methods} --out {output} > {log} 2>&1"


rule img_plot_traits:
    input:
        str(IMG_OUT / "evaluation.npz"),
    output:
        str(IMG_OUT / "figures" / "heritability.pdf"),
        str(IMG_OUT / "figures" / "correlation.pdf"),
    log:
        str(IMG_OUT / "figures" / "plot_traits.log"),
    params:
        out_dir=str(IMG_OUT / "figures"),
    threads: 1
    shell:
        "python -u {SIM_SCRIPTS}/plot_traits.py --evaluation {input}"
        " --out-dir {params.out_dir} > {log} 2>&1"
