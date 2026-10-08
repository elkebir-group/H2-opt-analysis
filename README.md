# H2-opt-analysis

This repository holds the code that reproduces the analyses and figures of the
[H2-opt](https://github.com/elkebir-group/H2-opt) paper: sorghum, Miscanthus, Arabidopsis
metabolomics and simulations (the Arabidopsis pathway enrichment and GWAS are not ported yet).
There are three parts:

- **`h2opt_analysis/`**: the dataset-specific library. Data loading and folds
  (`sorghum.py`, `miscanthus.py`, `arabidopsis.py`, `simulation.py`), the methods and their baseline fits
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
| `sorghum_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the H2-opt noise level of each fold (one for all traits), chosen on the fold's training plants by the best held-out heritability averaged over the traits and 5 group folds (`h2opt.selection.select_noise_level`, all folds in one batch); `noise_traits.npz` keeps the trained validation traits, to examine other selection rules without training again |
| `sorghum_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | H2-opt with the chosen noise, all folds in one job |
| `sorghum_train_baselines` | `{pca,genetic_pca,lda,pch,max_features}/fold{k}.npz` | baselines fitted on the training genotypes of fold k, once for all sources |
| `sorghum_score_heritability` | `heritability_{source}.npz` | train and test heritability of the source's H2-opt traits and the baselines' |
| `sorghum_plot_heritability` | `figures/top_trait_heritability_{source}.{pdf,png}` | boxplots of the top trait over the folds |
| `sorghum_run_gwas` | `gwas/{source}_fold{k}.npz` | GEMMA LMM of the per-line trait means (intercept + 5 genotype PCs) |

### Simulation

The paper's simulation (`h2opt_analysis/simulation.py`): in each of 10 instances, three traits
with heritability 0.6, 0.4 and 0.2 (100 additive QTNs each, simulated with simplePHENOTYPES on the
836 sequenced lines, 2 plants per line) are encoded into simulated spectra with the sorghum
autoencoder. The simplePHENOTYPES outputs are raw data; the encoding, the test lines (a third of
the lines) and everything after are rebuilt here. All outputs are under `data/derived/simulation/`.

| Rule | Output | Content |
|---|---|---|
| `sim_simulate_spectra` | `instance{i}/{spectra,truth}.npz` | simulated spectra, test plants, line of each plant, true traits |
| `sim_train_linear_h2opt` | `instance{i}/h2opt.{npz,pt}` | H2-opt with the fixed noise SD of the config, the paper's 0.02 (device from the config) |
| `sim_train_hybrid_h2opt` | `instance{i}/h2opt_hybrid.{npz,pt}` | H2-opt with the linear + CNN model (`h2opt.LinearConvModel`, 1D `ConvModel`), warm started from linear H2-opt, same noise SD (device from the config) |
| `sim_sweep_linear_h2opt_noise` | `instance{i}/h2opt_noise_sweep.npz` | H2-opt at every level of `h2opt.selection.NOISE_LEVELS`, all levels in one job |
| `sim_plot_linear_h2opt_noise_sweep` | `noise_sweep.npz`, `figures/noise_sweep.{pdf,png}` | \|r\| of traits 1-3 with the true traits against the noise level |
| `sim_train_baselines` | `instance{i}/{pca,genetic_pca,lda,pch,max_features}.npz` | baselines fitted on the training lines (PCH's ridge chosen by `PCH.tune`) |
| `sim_prepare_gwas_genotypes` | `instance{i}/genotypes.{bed,bim,fam}`, `genotype_pcs.npy` | genotypes without the instance's QTNs, and their PCs |
| `sim_run_gwas` | `instance{i}/gwas/{method}.npz` | GEMMA LMM of the line means of traits 1-3 (intercept + 5 PCs) |
| `sim_score_traits` | `evaluation.npz` | correlation with the true traits, train/test heritability |
| `sim_score_gwas` | `roc.npz` | AUC and ROC curves of causal 100 kb bins |
| `sim_plot_traits` | `figures/{heritability,correlation}.pdf` | test heritability and \|r\| with the true traits per method (Fig. 3 and Fig. S2 panels) |
| `sim_plot_gwas` | `figures/{auc,roc_trait1-3}.pdf`, `figures/gwas_{method}_instance{i}.png` | AUC, ROC curves, and Manhattan and QQ plots of one example (Fig. 3 and Fig. S2 panels) |

### Miscanthus

Multispectral drone images of two populations of half-sib families, Central and South Japan
(`h2opt_analysis/miscanthus.py`): 14 dates, 6 channels (crop surface model, blue, green, red, red
edge, near infrared), 63 x 63 pixels per plant. Each date is analyzed alone. The clone checks,
'unknown' seed lots and empty plots are left out; the folds are the paper's 5 plant folds. The
heritability is Henderson's Method III (`h2opt.Henderson3`: families, plots within families, rep
and field column fixed), broad-sense on the plant scale (the paper's narrow-sense value is 4 x it,
capped at 1). The config lists the populations. All outputs are under
`data/derived/miscanthus/{population}/`.

| Rule | Output | Content |
|---|---|---|
| `misc_prepare_images` | `images.npy` | the images as uncompressed float32, read one date at a time |
| `misc_score_cnn_h2opt_noise` | `h2opt/noise/level{s}.npz` | held-out Henderson3 of the H2-opt CNN at one noise level, every fold and date (`h2opt.selection.select_noise_level` on 5 splits of the training plants), and the validation traits; one job per level, so a new level trains only itself (device from the config) |
| `misc_choose_cnn_h2opt_noise` | `h2opt/noise.npz`, `figures/noise.{pdf,png}` | the noise level of each fold, one for all dates: the best mean score over all levels (`h2opt.selection.best_level`) |
| `misc_train_cnn_h2opt` | `h2opt/date{d}.{npz,pt}` | the H2-opt CNN (`h2opt.ImageConvModel`) of one date, all folds in one job (device from the config) |
| `misc_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | baselines of every date in one fold; PCH maximizes Henderson3, one ridge for all dates (`PCH.tune` over `miscanthus.PCH_RIDGES`) |
| `misc_score_heritability` | `heritability.npz` | train and test Henderson3 of every method, date and fold |
| `misc_plot_heritability` | `figures/heritability.{pdf,png}` | test heritability per date, H2-opt against the baselines |

### Arabidopsis

LC-MS metabolomics of 280 accessions, 263 in 0d and 277 in 6d, all genotyped
(`h2opt_analysis/arabidopsis.py`): MS1 and MS/MS binned by integer m/z (0-1499) and 30 s of
retention time (0-20 min), 120,000 features over their largest
value, as in the paper. The two treatments, normal growth (0d) and 6 days of darkness (6d), are
analyzed on their own; each accession has one sample per year (Col-0: nine), and the year is the
environment.
The nine labels 6909-1 to 6909-9 are pooled as one accession (Col-0, 6909 in the PLINK set): their
metabolite profiles show no heritability among them. The folds are 5 random folds of accessions
(seed 0). The heritability is ANOVA among accessions (`h2opt.anova_heritability`). All outputs are
under `data/derived/arabidopsis/{treatment}/`.

| Rule | Output | Content |
|---|---|---|
| `arabi_prepare_features` | `../features.npy` | the features of all samples as float32 |
| `arabi_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the noise level of each fold, one for all traits (`h2opt.selection.select_noise_level` on 5 splits of the training accessions), and the validation traits (device from the config) |
| `arabi_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | linear H2-opt, all folds in one job (device from the config) |
| `arabi_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | the baselines of one fold; PCH's ridge by `PCH.tune` |
| `arabi_score_heritability` | `heritability.npz` | train and test heritability of every method, fold and trait |
| `arabi_plot_heritability` | `figures/heritability_{test,train}.pdf` | heritability per trait, H2-opt against the baselines (Fig. arabiResult b and its train counterpart) |

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
