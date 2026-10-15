# H2-opt-analysis

This repository holds the code that reproduces the analyses and figures of the
[H2-opt](https://github.com/elkebir-group/H2-opt) paper: sorghum, Miscanthus, Arabidopsis
metabolomics and simulations.
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
git clone https://github.com/elkebir-group/H2-opt.git ../H2-opt
uv sync --all-extras        # creates .venv; the pipeline extra holds Snakemake
uv run pre-commit install   # one-time, after cloning: runs ruff and codespell on commit
scripts/install_gemma.sh    # GEMMA 0.97, into bin/ (Linux or macOS)
```

Always pass `--all-extras`. A bare `uv sync` removes Snakemake and the dev tools.
To use an `H2-opt` checkout elsewhere, run `uv pip install -e /path/to/H2-opt`.
GWAS use GEMMA 0.97. Set `GEMMA=/path/to/gemma`
to use another binary.

## Data

The raw data and the trained H2-opt models of all synthetic traits of the paper are deposited in
the Illinois Data Bank: https://doi.org/10.13012/B2IDB-XXXXXXX_V1. Download the data and link it
as `data/raw`:

```bash
mkdir -p data && ln -s /path/to/h2opt-data data/raw
```

Alternatively set `raw_root` in `pipelines/config/config.yaml` (or `$H2OPT_DATA` when you run a
script by hand). The pipeline reads these files:

```
data/raw/
├── sorghum/
│   ├── spectra.npz              # hyperspectral measurements (plants x 2,151 wavelengths)
│   ├── genotypes.npz            # genotype of each plant
│   ├── field_sets.npz           # field set of each plant
│   ├── measured_traits.tsv      # genotype means of the measured traits
│   ├── genotype_names.tsv       # genotype names of the PLINK samples
│   └── snps.bed, .bim, .fam     # PLINK set of the 836 sequenced lines
├── simulation/
│   ├── autoencoder.pt           # autoencoder trained on the sorghum spectra
│   └── instance0/ … instance9/
│       ├── traits.txt           # simplePHENOTYPES traits of the simulated plants
│       ├── qtns.txt             # causal SNPs of each trait
│       └── random_snps.csv      # genotypes given to simplePHENOTYPES (lines in .fam order)
├── miscanthus/
│   └── central/, south/
│       ├── images.npz           # multispectral images (plants x dates x channels x 63 x 63)
│       ├── plants.tsv           # half-sibling family and field position of each plant
│       ├── measured_traits.tsv  # measured traits of each plant
│       └── snps.bed, .bim, .fam # PLINK set of the genotyped plants
└── arabidopsis/
    ├── spectra.npz              # mass spectrometry, binned by m/z and retention time
    ├── precise_masses_0d.npz, precise_masses_6d.npz   # precise masses (pathway enrichment)
    ├── accessions.npz, environments.npz, years.npz     # labels of each sample
    └── snps.bed, .bim, .fam     # 250k SNP panel (call method 75, TAIR8 positions)
```

The pipeline downloads the MetaboAnalyst
pathway library (`arabi_download_pathway_library`) and clones Helios itself. Everything the
pipeline writes goes to `data/derived/`. Nothing under `data/` is committed.

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

All outputs are under `data/derived/sorghum/`. H2-opt is trained with Adam, with the noise
level of each fold, one for all traits, chosen by held-out heritability.

| Rule | Output | Content |
|---|---|---|
| `sorghum_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the H2-opt noise level of each fold (one for all traits), chosen on the fold's training plants by the highest level within one SE of the best held-out heritability of the first trait over 20 validation splits of the genotypes (`h2opt.selection.select_noise_level`, all folds in one batch); `noise_traits.npz` keeps the trained validation traits, to examine other selection rules without training again |
| `sorghum_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | H2-opt with the chosen noise, all folds in one job |
| `sorghum_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | baselines fitted on the training genotypes of fold k |
| `sorghum_score_heritability` | `heritability_h2opt.npz` | train and test heritability of the H2-opt traits and the baselines' |
| `sorghum_run_gwas` | `gwas/h2opt_fold{k}.npz` | GEMMA LMM of the per-line trait means (intercept + 5 genotype PCs) |
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
| `sim_train_baselines` | `instance{i}/{pca,genetic_pca,pch,max_features}.npz` | baselines fitted on the training lines (PCH's ridge chosen by `PCH.tune`, the procedure and rule of H2-opt's noise level) |
| `sim_prepare_gwas_genotypes` | `instance{i}/genotypes.{bed,bim,fam}`, `genotype_pcs.npy` | genotypes without the instance's QTNs, and their PCs |
| `sim_run_gwas` | `instance{i}/gwas/{method}.npz` | GEMMA LMM of the line means of traits 1-3 (intercept + 5 PCs) |
| `sim_score_traits` | `evaluation.npz` | correlation with the true traits, train/test heritability |
| `sim_score_gwas` | `roc.npz` | AUC and ROC curves of causal 100 kb bins |
| `sim_plot_paper_figures` (target `simulation_paper_figures`) | `figures/paper/{simFig,simCompareSNP}.{pdf,png}` | the paper's Fig. 3 (a-e: heritability, \|r\|, QQ and Manhattan plots of the example, ROC of trait 1, AUC) and Fig. S2 (ROC of traits 2 and 3), from `evaluation.npz`, `roc.npz` and the example GWAS |

### Image simulation

Simulated top-down plot images rendered with the Helios 3D plant and radiation simulator
(`h2opt_analysis/helios_simulation.py`, `pipelines/helios/field_render`): the true traits of the
spectral simulation drive leaf greenness, canopy size and tiller number of sorghum clumps, with
non-heritable plot effects (position offset, soil, senescence, render noise). The images have the
shape and scaling of the Miscanthus images. Same 10 instances and test lines. The methods are
those of the spectral simulation (scripts of `scripts/simulation`): linear models take the
flattened pixels, and the CNN of the hybrid is `h2opt.ImageConvModel`. The noise levels are fixed
for every instance (config `image_simulation`), not selected. All outputs are under
`data/derived/helios_simulation/`.

| Rule | Output | Content |
|---|---|---|
| `img_build_helios` | `data/derived/helios/build/field_render` | the Helios renderer, built at the config commit |
| `img_render_helios` | `instance{i}/{images,truth}.npz` | rendered images, test plants, line of each plant, true traits |
| `img_fixed_levels` | `instance{i}/{noise,cnn_level}.npz` | the fixed linear and CNN levels |
| `img_train_linear_h2opt` | `instance{i}/h2opt.{npz,pt}` | as `sim_train_linear_h2opt` |
| `img_train_hybrid_h2opt` | `instance{i}/h2opt_hybrid.{npz,pt}` | as `sim_train_hybrid_h2opt`, with `h2opt.ImageConvModel` |
| `img_train_baselines` | `instance{i}/{pca,genetic_pca,pch,max_features}.npz` | as `sim_train_baselines` |
| `img_score_traits` | `evaluation.npz` | correlation with the true traits, train/test heritability |

### Miscanthus

Multispectral drone images of two populations of half-sib families, Central and South Japan
(`h2opt_analysis/miscanthus.py`): 14 dates, 6 channels (crop surface model, blue, green, red, red
edge, near infrared), 63 x 63 pixels per plant. Each date is analyzed alone. The clone checks,
'unknown' seed lots and empty plots are left out. The folds are 5 random folds of plants
(np.random.seed(1), randint(5) over all 3,900 plants). The
heritability is Henderson's Method III (`h2opt.Henderson3`: families, plots within families, rep
and field column fixed), broad-sense on the plant scale (the narrow-sense value is 4 x it,
capped at 1), after the nearest-neighbor field adjustment (`h2opt.spatial`, Papadakis 1937: the
other-family plots within `spatial_width` field columns in the same rep). The config lists the
populations.

The images, the baselines, the genotypes and the CNN level are shared; they are under
`data/derived/miscanthus/{population}/`.

| Rule | Output | Content |
|---|---|---|
| `misc_prepare_images` | `images.npy` | the images as uncompressed float32, read one date at a time |
| `misc_choose_linear_h2opt_noise` | `h2opt/noise.npz` | the linear noise level without the adjustment (`h2opt.selection.select_noise_level` on the selection date, 06/08, the validation splits of all folds together); the input of the CNN level |
| `misc_choose_hybrid_cnn_level` | `h2opt_hybrid/cnn_level.npz` | the one CNN level of the hybrid for all folds and dates: `h2opt.selection.score_linear_conv_levels` on the selection date, fold 0 |
| `misc_choose_pch_ridge` | `pch/ridge.npz` | the one PCH ridge for all folds and dates: `PCH.tune` over `h2opt.baselines.RIDGES` on the selection date, the validation splits of all folds together |
| `misc_train_baselines` | `{pca,genetic_pca,pch,max_features,single_band}/fold{k}.npz` | baselines of every date in one fold; PCH maximizes Henderson3 at the ridge of `pch/ridge.npz` |
| `misc_prepare_genotypes` | `gwas/genotypes.{bed,bim,fam}`, `gwas/genotype_pcs.npy` | the PLINK set (unplaced scaffolds as chromosome 0) and 5 genotype PCs of the genotyped plants with traits |
| `misc_score_ndvi` | `ndvi.npz` | NDVI (mean red and NIR of the pixels) of every plant and date, and its Henderson3 per date and fold, before and after the adjustment |

The adjusted run trains H2-opt on the adjusted heritability (config `miscanthus.adjusted`); its
outputs are under `data/derived/miscanthus_adjusted/{population}/` (target
`miscanthus_adjusted_all`).

| Rule | Output | Content |
|---|---|---|
| `misc_adjusted_choose_linear_h2opt_noise` | `h2opt/noise.npz` | the linear noise level, selected again under the adjustment |
| `misc_adjusted_train_linear_h2opt` | `h2opt/date{d}.{npz,pt}` | linear H2-opt of one date, all folds |
| `misc_adjusted_train_hybrid_h2opt` | `h2opt_hybrid/date{d}.{npz,pt}` | the hybrid (H2-opt (CNN)) of one date, all folds: `h2opt.LinearConvModel` with `h2opt.PositionImageConvModel`, at the linear level and the CNN level of `cnn_level_population` |
| `misc_adjusted_score_heritability` | `heritability.npz` | train and test Henderson3 after the adjustment, every method, date and fold |
| `misc_adjusted_date_correlations` | `date_correlations.npz` | Pearson r of trait 1 of the hybrid between dates, per plant and between half-sib family means, per fold |
| `misc_adjusted_measured_correlations` | `measured_correlations.npz` | the same with the measured traits of the plant data tables |

The GWAS and the null are separate targets, `miscanthus_adjusted_gwas_all` and
`miscanthus_adjusted_null_all`, run only on request. The genotypes are the PLINK sets
`miscanthus/{central,south}/snps`.
Plants with two conflicting samples are left out.

| Rule | Output | Content |
|---|---|---|
| `misc_adjusted_crossfit_traits` | `h2opt_hybrid/crossfit_date{d}.npz` | trait 1 of each plant from the fold model that held it out |
| `misc_adjusted_run_gwas` | `gwas/{,crossfit_}date{d}.npz` | GEMMA LMM of trait 1 of the hybrid, one run per fold (intercept, 5 PCs and the neighbor mean), and of the cross-fitted trait |
| `misc_adjusted_null_gwas` | `null/perm{k}.npz` | one permutation of the whole hybrid H2-opt to GWAS pipeline on the selection date |

The paper figures are a separate target, `miscanthus_paper_figures`, under
`data/derived/miscanthus/figures/paper/`, each a `.pdf` and a `.png`; `miscGWAS` and `miscNull`
need the GWAS and the null.

| Rule | Output | Content |
|---|---|---|
| `misc_plot_paper_main` | `miscFig.pdf` | a example images; b test heritability per date; c and d family-mean \|r\| between dates and with the measured traits |
| `misc_plot_paper_heritability` | `miscVegIndex.pdf`, `trainHeritMisc.pdf` | mean NDVI per date and the test heritability of NDVI and H2-opt (CNN); train heritability |
| `misc_plot_paper_correlations` | `miscCorTime.pdf`, `traitCorrelation_{allTraits,simplePearson}.pdf` | \|r\| between dates and with the measured traits, per plant and between family means |
| `misc_plot_paper_gwas` | `miscGWAS_{population}.pdf` | GWAS of trait 1 per fold and cross-fitted, the dates with significant SNPs |
| `misc_plot_paper_null` | `miscNull.pdf` | the GWAS of trait 1 of all folds and dates against its cross-fit and the permutation null |

### Arabidopsis

LC-MS metabolomics of 280 accessions, 263 in 0d and 277 in 6d, all genotyped
(`h2opt_analysis/arabidopsis.py`): MS1 and MS/MS binned by integer m/z (0-1499) and 30 s of
retention time (0-20 min), 120,000 features over their largest
value. The two treatments, normal growth (0d) and 6 days of darkness (6d), are
analyzed on their own; each accession has one sample per year (Col-0: nine), and the year is the
environment.
The nine labels 6909-1 to 6909-9 are one accession (Col-0, 6909 in the PLINK set) and are pooled.
The folds are 5 random folds of accessions
(seed 0). The heritability is ANOVA among accessions (`h2opt.anova_heritability`). All outputs are
under `data/derived/arabidopsis/{treatment}/`.

| Rule | Output | Content |
|---|---|---|
| `arabi_prepare_features` | `../features.npy` | the features of all samples as float32 |
| `arabi_choose_linear_h2opt_noise` | `h2opt/noise.npz`, `h2opt/noise_traits.npz`, `figures/noise.{pdf,png}` | the noise level of each fold, one for all traits (`h2opt.selection.select_noise_level` on 20 validation splits of the training accessions), and the validation traits (device from the config) |
| `arabi_train_linear_h2opt` | `h2opt/fold{k}.{npz,pt}` | linear H2-opt, all folds in one job (device from the config) |
| `arabi_train_baselines` | `{pca,genetic_pca,pch,max_features}/fold{k}.npz` | the baselines of one fold; PCH's ridge by `PCH.tune` |
| `arabi_score_heritability` | `heritability.npz` | train and test heritability of every method, fold and trait |

Pathway enrichment is a separate target, `arabidopsis_pathway_enrichment`, run only on request.
It runs MetaboAnalyst's functional analysis (MS peaks to pathways, mummichog v2) offline with
MetaboAnalystR, in its own conda environment (`pipelines/envs/metaboanalyst.yaml`; the package is
installed from GitHub at a pinned commit by `metaboanalyst.post-deploy.sh`). The settings are in
the config (`arabidopsis.pathways`). The peak lists alone are the target
`arabidopsis_enrichment_inputs`.

| Rule | Output | Content |
|---|---|---|
| `arabi_prepare_precise_masses` | `precise_masses.npz` | precise m/z, mean retention time and intensity SD of each precise mass in the feature bins (streams `massIntensitiesPrecise2_{treatment}.npz`, 36.6 GB; about 22 GB of memory) |
| `arabi_enrichment_inputs` | `enrichment/trait{k}.txt` | the 10,000 precise masses with the largest \|MS1 coefficient x intensity SD\| per H2-opt trait, fold 0, in the paper's format |
| `arabi_download_pathway_library` | `../pathway_library/ath_kegg.qs` | the mummichog library from the MetaboAnalyst server (the log records the date and checksum) |
| `arabi_pathway_enrichment` | `enrichment/trait{k}_pathways.csv` | MetaboAnalyst's pathway table of one peak list: the first 500 peaks are the significant peaks, all peaks the reference |

The GWAS is a separate target, `arabidopsis_gwas_all`, run only on request. The genotypes are the
250k SNP panel `arabidopsis/snps` (call method 75, TAIR8 positions).

| Rule | Output | Content |
|---|---|---|
| `arabi_prepare_genotypes` | `gwas/genotypes.{bed,bim,fam}`, `gwas/genotype_pcs.npy` | the PLINK set and 5 genotype PCs of the treatment's accessions |
| `arabi_run_gwas` | `gwas/fold{k}.npz` | GEMMA LMM of the per-accession means of H2-opt traits 1-10 (intercept + 5 PCs) |

The paper figures are a separate target, `arabidopsis_paper_figures`
(it needs the GWAS, the null and the pathway enrichment).
The figures are under `data/derived/arabidopsis/figures/paper/`, named as the paper's images, each
a `.pdf` and a `.png`. Panel e of `metabFig` (pathway enrichment) uses the tables of
`arabi_pathway_enrichment`.

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

## Related repositories

- [`elkebir-group/H2-opt`](https://github.com/elkebir-group/H2-opt): the H2-opt package
  (heritability estimators, models, training and baselines). This repository installs it
  from a sibling checkout.
