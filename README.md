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
| `sorghum_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the H2-opt noise level of each fold (one for all traits), chosen on the fold's training plants by the highest level within one SE of the best held-out heritability of the first trait over 20 validation splits of the genotypes (`h2opt.selection.select_noise_level`, all folds in one batch); `noise_traits.npz` keeps the trained validation traits, to examine other selection rules without training again |
| `sorghum_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | H2-opt with the chosen noise, all folds in one job |
| `sorghum_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | baselines fitted on the training genotypes of fold k, once for all sources |
| `sorghum_score_heritability` | `heritability_{source}.npz` | train and test heritability of the source's H2-opt traits and the baselines' |
| `sorghum_plot_heritability` | `figures/top_trait_heritability_{source}.{pdf,png}` | boxplots of the top trait over the folds |
| `sorghum_run_gwas` | `gwas/{source}_fold{k}.npz` | GEMMA LMM of the per-line trait means (intercept + 5 genotype PCs) |
| `sorghum_plot_paper_main` | `figures/paper/sorFig.{pdf,png}` | paper Fig. (sorghum): example spectra; mean test heritability of traits 1-4 of `h2opt` and the baselines; wavelength coefficients of traits 1-4, fold 0; median over folds of \|r\| between genotype means of traits 1-4 and the measured traits (Ferguson et al. 2021, Table S12); Bonferroni-significant SNPs per trait, fold 0 |
| `sorghum_plot_paper_heritability` | `figures/paper/multiTraitHerit_all.{pdf,png}` | mean train and test heritability of traits 1-10 of `h2opt` and the baselines |
| `sorghum_plot_paper_gwas` | `figures/paper/{crossGWAS,sorGWAS}.{pdf,png}` | Pearson r of -log10 p of trait 1 between the folds; Manhattan and QQ plots of the fold 0 traits with significant SNPs |
| `sorghum_paper_figures` | | the four paper figures above (needs the GWAS, so not in `sorghum_all`) |

### Simulation

The paper's simulation (`h2opt_analysis/simulation.py`): in each of 10 instances, three traits
with heritability 0.6, 0.4 and 0.2 (100 additive QTNs each, simulated with simplePHENOTYPES on the
836 sequenced lines, 2 plants per line) are encoded into simulated spectra with the sorghum
autoencoder. The simplePHENOTYPES outputs are raw data; the encoding, the test lines (a third of
the lines) and everything after are rebuilt here. All outputs are under `data/derived/simulation/`.

| Rule | Output | Content |
|---|---|---|
| `sim_simulate_spectra` | `instance{i}/{spectra,truth}.npz` | simulated spectra (linearized decoder plus the residual of a random real spectrum), test plants, line of each plant, true traits |
| `sim_choose_linear_h2opt_noise` | `instance{i}/noise.npz` | the noise level of H2-opt, chosen on the training lines by the held-out heritability of trait 1 (`h2opt.selection.select_noise_level`) |
| `sim_train_linear_h2opt` | `instance{i}/h2opt.{npz,pt}` | H2-opt with the chosen noise (device from the config) |
| `sim_choose_hybrid_cnn_level` | `instance{i}/cnn_level.npz` | the CNN noise level of the hybrid: the highest level within one SE of the best held-out heritability of trait 1 (`h2opt.selection.score_linear_conv_levels`, `highest_level_within_one_se`) |
| `sim_train_hybrid_h2opt` | `instance{i}/h2opt_hybrid.{npz,pt}` | H2-opt with the linear + CNN model (`h2opt.LinearConvModel`, 1D `ConvModel`), warm started from linear H2-opt, at the chosen noise and CNN level (device from the config) |
| `sim_sweep_linear_h2opt_noise` | `instance{i}/h2opt_noise_sweep.npz` | H2-opt at every level of `h2opt.selection.NOISE_LEVELS`, all levels in one job |
| `sim_plot_linear_h2opt_noise_sweep` | `noise_sweep.npz`, `figures/noise_sweep.{pdf,png}` | \|r\| of traits 1-3 with the true traits against the noise level |
| `sim_train_baselines` | `instance{i}/{pca,genetic_pca,pch,max_features}.npz` | baselines fitted on the training lines (PCH's ridge chosen by `PCH.tune`, the procedure and rule of H2-opt's noise level) |
| `sim_prepare_gwas_genotypes` | `instance{i}/genotypes.{bed,bim,fam}`, `genotype_pcs.npy` | genotypes without the instance's QTNs, and their PCs |
| `sim_run_gwas` | `instance{i}/gwas/{method}.npz` | GEMMA LMM of the line means of traits 1-3 (intercept + 5 PCs) |
| `sim_score_traits` | `evaluation.npz` | correlation with the true traits, train/test heritability |
| `sim_score_gwas` | `roc.npz` | AUC and ROC curves of causal 100 kb bins |
| `sim_plot_traits` | `figures/{heritability,correlation}.pdf` | test heritability and \|r\| with the true traits per method (Fig. 3 and Fig. S2 panels) |
| `sim_plot_gwas` | `figures/{auc,roc_trait1-3}.pdf`, `figures/gwas_{method}_instance{i}.png` | AUC, ROC curves, and Manhattan and QQ plots of one example (Fig. 3 and Fig. S2 panels) |
| `sim_plot_paper_figures` (target `simulation_paper_figures`) | `figures/paper/{simFig,simCompareSNP}.{pdf,png}` | the paper's Fig. 3 (a-e: heritability, \|r\|, QQ and Manhattan plots of the example, ROC of trait 1, AUC) and Fig. S2 (ROC of traits 2 and 3), from `evaluation.npz`, `roc.npz` and the example GWAS |

### Image simulation

Simulated top-down plot images (`h2opt_analysis/image_simulation.py`): the true traits of the
spectral simulation drive leaf greenness, canopy size and tiller number of images with the shape
and scaling of the Miscanthus images, with non-heritable plot effects (position offset, soil,
pixel noise, senesced patches). Same 10 instances and test lines. The methods are those of the
spectral simulation (scripts of `scripts/simulation`): linear models take the flattened pixels,
and the CNN of the hybrid is `h2opt.ImageConvModel` (global average pooling). All outputs are
under `data/derived/image_simulation/`.

| Rule | Output | Content |
|---|---|---|
| `img_simulate_images` | `instance{i}/{images,truth}.npz` | simulated images, test plants, line of each plant, true traits |
| `img_choose_linear_h2opt_noise` | `instance{i}/noise.npz` | as `sim_choose_linear_h2opt_noise` |
| `img_train_linear_h2opt` | `instance{i}/h2opt.{npz,pt}` | as `sim_train_linear_h2opt` |
| `img_choose_hybrid_cnn_level` | `instance{i}/cnn_level.npz` | as `sim_choose_hybrid_cnn_level`, with `h2opt.ImageConvModel` |
| `img_train_hybrid_h2opt` | `instance{i}/h2opt_hybrid.{npz,pt}` | as `sim_train_hybrid_h2opt`, with `h2opt.ImageConvModel` |
| `img_train_baselines` | `instance{i}/{pca,genetic_pca,pch,max_features}.npz` | as `sim_train_baselines` |
| `img_score_traits` | `evaluation.npz` | correlation with the true traits, train/test heritability |
| `img_plot_traits` | `figures/{heritability,correlation}.pdf` | test heritability and \|r\| with the true traits per method |

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
| `misc_choose_linear_h2opt_noise` | `h2opt/noise.npz` | the one linear noise level for all folds and dates: `h2opt.selection.select_noise_level` on the selection date (06/08, before canopy closure), the validation splits of all folds together |
| `misc_train_linear_h2opt` | `h2opt/date{d}.npz` | linear H2-opt of one date, all folds, at the linear level |
| `misc_choose_hybrid_cnn_level` | `h2opt_hybrid/cnn_level.npz` | the one CNN level of the hybrid for all folds and dates: `h2opt.selection.score_linear_conv_levels` on the selection date, fold 0 |
| `misc_train_hybrid_h2opt` | `h2opt_hybrid/date{d}.npz` | the hybrid of one date, all folds: `h2opt.LinearConvModel` with `h2opt.PositionImageConvModel`, warm started from linear H2-opt, at the linear and CNN levels |
| `misc_choose_pch_ridge` | `pch/ridge.npz` | the one PCH ridge for all folds and dates: `PCH.tune` over `h2opt.baselines.RIDGES` on the selection date, the validation splits of all folds together (as the linear H2-opt level) |
| `misc_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | baselines of every date in one fold; PCH maximizes Henderson3 at the ridge of `pch/ridge.npz` |
| `misc_score_heritability` | `heritability.npz` | train and test Henderson3 of every method, date and fold |
| `misc_plot_heritability` | `figures/heritability.{pdf,png}` | test heritability per date, hybrid H2-opt against the baselines |

The GWAS is a separate target, `miscanthus_gwas_all`, run only on request. The genotypes are the
PLINK sets `miscPlant/SNP/bed/Msi{C,S}_annotated` with their full sample lists (`_copy.fam`; the
`.fam` beside each `.bed` is a subset written by the original code, and GEMMA reads shifted
genotypes with it). Plants with two conflicting samples are left out.

| Rule | Output | Content |
|---|---|---|
| `misc_prepare_genotypes` | `gwas/genotypes.{bed,bim,fam}`, `gwas/genotype_pcs.npy` | the PLINK set (unplaced scaffolds as chromosome 0) and 5 genotype PCs of the genotyped plants with traits |
| `misc_run_gwas` | `gwas/date{d}.npz` | GEMMA LMM of trait 1 of hybrid H2-opt, one run per fold (intercept + 5 PCs) |

The paper figures are a separate target, `miscanthus_paper_figures`; `miscFig` and `miscGWAS` need
the GWAS. The figures are under `data/derived/miscanthus/figures/paper/`, each a `.pdf` and a
`.png`; the data rules write beside the population outputs. Correlations are the median over the
folds of \|r\|; GWAS counts are Bonferroni-significant SNPs of the first fold. `miscSecondTrait`
is not made: it needs a run with `traits: 2`.

| Rule | Output | Content |
|---|---|---|
| `misc_score_ndvi` | `ndvi.npz` | NDVI (mean red and NIR of the pixels) of every plant and date, and its train and test Henderson3 per date and fold, scored as the methods |
| `misc_date_correlations` | `date_correlations.npz` | Pearson r of trait 1 of hybrid H2-opt between dates, per plant and between half-sib family means, per fold (all kept plants) |
| `misc_measured_correlations` | `central/measured_correlations.npz` | the same with the 16 measured traits of the Central plant data table (Central only) |
| `misc_plot_paper_main` | `miscFig.pdf` | a example images; b test heritability per date; c family-mean \|r\| between the first five dates; d the same with the 6 top measured traits; e significant SNPs per date |
| `misc_plot_paper_heritability` | `NDVI_canopy.pdf`, `miscVegIndex.pdf`, `trainHeritMisc.pdf` | mean NDVI per date; test heritability of NDVI and the methods; train heritability |
| `misc_plot_paper_correlations` | `miscCorTimeSimple.pdf`, `miscCorTime.pdf`, `traitCorrelation_{allTraits,simplePearson}.pdf` | \|r\| between dates and with the measured traits, per plant and between family means |
| `misc_plot_paper_gwas` | `miscGWAS.pdf` | Manhattan and QQ plots of the dates with significant SNPs, first fold (one page per population) |

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
| `arabi_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the noise level of each fold, one for all traits (`h2opt.selection.select_noise_level` on 20 validation splits of the training accessions), and the validation traits (device from the config) |
| `arabi_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | linear H2-opt, all folds in one job (device from the config) |
| `arabi_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | the baselines of one fold; PCH's ridge by `PCH.tune` |
| `arabi_score_heritability` | `heritability.npz` | train and test heritability of every method, fold and trait |
| `arabi_plot_heritability` | `figures/heritability_{test,train}.pdf` | heritability per trait, H2-opt against the baselines (Fig. arabiResult b and its train counterpart) |

The GWAS is a separate target, `arabidopsis_gwas_all`, run only on request. The genotypes are the
250k SNP panel `metab/metabData_Alex/call_method_75/call_method_75_TAIR8` with its full sample list
(`_copy.fam`; the `.fam` beside the `.bed` is a subset, as for Miscanthus).

| Rule | Output | Content |
|---|---|---|
| `arabi_prepare_genotypes` | `gwas/genotypes.{bed,bim,fam}`, `gwas/genotype_pcs.npy` | the PLINK set and 5 genotype PCs of the treatment's accessions |
| `arabi_run_gwas` | `gwas/fold{k}.npz` | GEMMA LMM of the per-accession means of H2-opt traits 1-10 (intercept + 5 PCs) |

The paper figures are a separate target, `arabidopsis_paper_figures` (it needs the GWAS of fold 0).
The figures are under `data/derived/arabidopsis/figures/paper/`, named as the paper's images, each
a `.pdf` and a `.png`. Panel e of `metabFig` (pathway enrichment) is not made: its input (the
mummichog files) is missing.

| Rule | Output | Content |
|---|---|---|
| `arabi_trait_coefficients` | `{treatment}/h2opt/coefficients_fold{k}.npz` | coefficients of the uncorrelated H2-opt traits per feature, over each trait's largest magnitude (checked against the saved traits) |
| `arabi_paper_main_figure` | `metabFig.pdf` | Fig. arabiResult a-d: one input matrix per treatment, test heritability per method, coefficients of 0d traits 1-3 (MS1), significant SNPs per trait |
| `arabi_paper_heritability` | `ArabiHeritTrain.pdf` | train and test heritability per trait, every method, both treatments |
| `arabi_paper_coefficients` | `coef_0day.pdf`, `coef_6day.pdf` | coefficients of traits 1-10 (magnitude at least 2% of the largest; MS1 and MS/MS), fold 0 |
| `arabi_paper_gwas` | `metab0d_GWAS.pdf`, `metab6d_GWAS.pdf` | Manhattan and QQ plots of the traits with significant SNPs, fold 0 |

### lme4 check of the heritability estimators

`estimator_check_all` compares, for the raw features of each dataset, the pipeline's estimator
(ANOVA for sorghum and Arabidopsis, Henderson3 for Miscanthus) with an lme4 mixed model of the same
design: environment fixed, group and plots random (paper Fig. ANOVA). R and lme4 come from `pipelines/envs/lme4.yaml`
(Snakemake builds the environment; `run.sh` passes `--use-conda`). Outputs are under
`data/derived/estimator_check/`.

| Rule | Output | Content |
|---|---|---|
| `check_estimator_inputs` | `{dataset}.csv`, `{dataset}.npz` | the raw features with their design, and the estimator's heritability of each |
| `check_lme4_heritability` | `{dataset}_lme4.csv` | lme4 heritability of each feature: group variance over the feature's variance |
| `check_plot_estimators` | `figures/ANOVA.{pdf,png}` | one panel per dataset with the Pearson r |

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
