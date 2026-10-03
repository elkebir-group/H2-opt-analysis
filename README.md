# H2-opt-analysis

This repository holds the code that reproduces the analyses and figures of the
[H2-opt](https://github.com/elkebir-group/H2-opt) paper: sorghum, Miscanthus,
metabolomics and simulations. There are three parts:

- **`h2opt_analysis/`**: the dataset-specific library. Data loading and folds
  (`sorghum.py`, `simulation.py`), the methods and their baseline fits
  (`methods.py`), GEMMA GWAS and per-line phenotypes (`gwas.py`), permuted-genotype
  nulls (`nulls.py`) and shared figure helpers (`figures.py`).
- **`pipelines/`**: a Snakemake pipeline that runs every analysis from the raw
  data to the figures. See "Running the pipeline" below.
- **`scripts/`**: setup scripts (`install_gemma.sh`).

## Install

Needs [uv](https://docs.astral.sh/uv/) and an `H2-opt` checkout at `../H2-opt`,
which is installed as an editable package.

```bash
git clone git@github.com:elkebir-group/H2-opt.git ../H2-opt
uv sync --all-extras        # creates .venv; the pipeline extra holds Snakemake
uv run pre-commit install   # one-time, after cloning: runs ruff and codespell on commit
scripts/install_gemma.sh    # GEMMA 0.97, into bin/ (Linux or macOS)
```

Always pass `--all-extras`. A bare `uv sync` removes Snakemake and the dev tools.
To use an `H2-opt` checkout elsewhere, run `uv pip install -e /path/to/H2-opt`.
GWAS use GEMMA 0.97, the version used for the paper. Set `GEMMA=/path/to/gemma`
to use another binary.

## Data

Nothing under `data/` is committed.

| Path | What | Source |
|---|---|---|
| `data/raw/` | the H2-opt datasets (`plant/`, `miscPlant/`, `metab/`, `software/`, ...) | a symlink to the lab's H2-opt data directory |
| `data/derived/` | everything the pipeline writes | `pipelines/run.sh` |

```bash
mkdir -p data && ln -s /path/to/h2opt-data data/raw
```

Alternatively set `raw_root` in `pipelines/config/config.yaml` (or `$H2OPT_DATA`
when you run a script by hand).

## Running the pipeline

`pipelines/run.sh` runs `pipelines/Snakefile` with the config in
`pipelines/config/config.yaml`. A target is an output path relative to the
repository root, or a rule name.

```bash
pipelines/run.sh --dry-run                                        # what would run
pipelines/run.sh -- data/derived/sorghum/gwas/paper_fold0.npz     # one output
pipelines/run.sh -- sorghum_all                                   # all sorghum analyses
pipelines/run.sh                                                  # everything
```

`run.sh` always passes `--rerun-triggers mtime`, so a code or config change never
reruns existing outputs by itself. To rerun one output, delete it or add
`--forcerun <rule>` with that one target. Each job's BLAS threads are capped at
its rule's `threads`, and `--cores` (default: all cores but two) caps their sum.
GPU training jobs share the GPU, 8 at a time (`--gpu-jobs N`).

### Sorghum

All outputs are under `data/derived/sorghum/`. A trait source is one way to get
the H2-opt traits of a genotype fold: `paper` (the predictions behind the paper),
`rmsprop` (retrained with the paper's settings) or `lbfgs` (linear H2-opt
trained to convergence with L-BFGS, with the ridge and the number of steps tuned
on inner validation genotypes).

| Rule | Output | Content |
|---|---|---|
| `sorghum_train_rmsprop` | `h2opt_rmsprop/fold{k}.{npz,pt}` | H2-opt with the paper's settings, fold k |
| `sorghum_train_lbfgs` | `h2opt_lbfgs/fold{k}.npz` | L-BFGS H2-opt, fold k |
| `sorghum_heritability` | `heritability_{source}.npz` | train and test heritability of H2-opt, PCA, genetic PCA, LDA, PCH and the most heritable features |
| `sorghum_plot_heritability` | `figures/top_trait_heritability_{source}.{pdf,png}` | boxplots of the top trait over the folds |
| `sorghum_gwas` | `gwas/{source}_fold{k}.npz` | GEMMA LMM of the per-line trait means (intercept + 5 genotype PCs) |

### Simulation

The paper's simulation (`h2opt_analysis/simulation.py`): in each of 10 instances, three traits
with heritability 0.6, 0.4 and 0.2 (100 additive QTNs each, simulated with simplePHENOTYPES on the
836 sequenced lines, 2 plants per line) are encoded into simulated spectra with the sorghum
autoencoder. The simplePHENOTYPES outputs are raw data; the encoding, the test lines (a third of
the lines) and everything after are rebuilt here. All outputs are under `data/derived/simulation/`.

| Rule | Output | Content |
|---|---|---|
| `sim_encode` | `instance{i}/{spectra,truth}.npz` | simulated spectra, test plants, line of each plant, true traits |
| `sim_train_rmsprop` | `instance{i}/h2opt_rmsprop.{npz,pt}` | H2-opt with the paper's simulation settings (GPU) |
| `sim_train_lbfgs` | `instance{i}/h2opt_lbfgs.npz` | linear H2-opt with L-BFGS |
| `sim_baseline` | `instance{i}/{pca,genetic_pca,lda,pch,max_features}.npz` | baselines fitted on the training lines |
| `sim_genotypes` | `instance{i}/genotypes.{bed,bim,fam}`, `genotype_pcs.npy` | genotypes without the instance's QTNs, and their PCs |
| `sim_gwas` | `instance{i}/gwas/{method}.npz` | GEMMA LMM of the line means of traits 1-3 (intercept + 5 PCs) |
| `sim_evaluate` | `evaluation.npz` | correlation with the true traits, train/test heritability |
| `sim_roc` | `roc.npz` | AUC and ROC curves of causal 100 kb bins |
| `sim_plot` | `figures/` | Fig. 3 and Fig. S2 panels |

Each script under `pipelines/scripts/` also runs by hand; its docstring gives
the arguments.

## Permutation nulls

`h2opt_analysis.nulls.permute_genotypes` permutes the genotype labels once,
within strata. Use the permuted labels everywhere a genotype label is used:
training, folds, heritability evaluation and GWAS.

## Tests

```bash
uv run --all-extras pytest
```

The GEMMA test is skipped when GEMMA is not installed.
