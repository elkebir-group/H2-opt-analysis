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
the H2-opt traits of a genotype fold: `paper` (the predictions behind the paper)
or `h2opt` (retrained here with Adam, with the noise level of each fold, one for all traits, chosen by held-out heritability).

| Rule | Output | Content |
|---|---|---|
| `sorghum_choose_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the H2-opt noise level of each fold (one for all traits), chosen on the fold's training plants by the best held-out heritability averaged over the traits and 5 group folds (`h2opt.selection.select_noise_level`, all folds in one batch); `noise_traits.npz` keeps the trained validation traits, to examine other selection rules without training again |
| `sorghum_train_h2opt` | `h2opt/fold{k}.{npz,pt}` | H2-opt with the chosen noise, all folds in one job |
| `sorghum_baseline` | `{pca,genetic_pca,lda,pch,max_features}/fold{k}.npz` | baselines fitted on the training genotypes of fold k, once for all sources |
| `sorghum_heritability` | `heritability_{source}.npz` | train and test heritability of the source's H2-opt traits and the baselines' |
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
| `sim_train_h2opt` | `instance{i}/h2opt.{npz,pt}` | H2-opt with the fixed noise SD of the config, the paper's 0.02 (GPU) |
| `sim_noise_sweep` | `instance{i}/h2opt_noise_sweep.npz` | H2-opt at every level of `h2opt.selection.NOISE_LEVELS`, all levels in one job |
| `sim_plot_noise_sweep` | `noise_sweep.npz`, `figures/noise_sweep.{pdf,png}` | \|r\| of traits 1-3 with the true traits against the noise level |
| `sim_baseline` | `instance{i}/{pca,genetic_pca,lda,pch,max_features}.npz` | baselines fitted on the training lines (PCH's ridge chosen by `PCH.tune`) |
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
