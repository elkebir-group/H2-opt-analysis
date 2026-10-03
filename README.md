# H2-opt-analysis

Analysis code for the H2-opt paper (sorghum, Miscanthus, metabolomics, simulations): data pipelines, baselines,
permuted-genotype nulls, cross-fit GWAS, genomic inflation (λGC), and figures.

The method itself lives in [H2-opt](https://github.com/elkebir-group/H2-opt); this repo depends on it.

## Setup

```bash
uv sync                      # installs h2opt from the refactor branch of H2-opt
ln -s /path/to/h2opt-data data   # or set H2OPT_DATA=/path/to/h2opt-data
```

GWAS use GEMMA 0.97, the version used for the paper; `scripts/install_gemma.sh` installs it into `bin/` (Linux or
macOS). Set `GEMMA=/path/to/gemma` to use another binary.

Permutation nulls use `h2opt_analysis.nulls.permute_genotypes`: genotype labels permuted once, within strata, and
then used for training, folds, evaluation and GWAS alike.

Scripts are run from the repository root and write to `results/`.

To run a script over folds or seeds, several jobs at once (about 8 small GPU jobs fit on one GPU with near-linear speedup):

```bash
uv run python -m h2opt_analysis.launch --jobs 8 --threads 1 --log-dir results/sorghum/logs --grid fold=0-9 \
    -- python scripts/sorghum/train_h2opt.py --fold {fold} --device cuda
```

## Sorghum

| Script | Result |
|---|---|
| `scripts/sorghum/train_h2opt.py --fold k` | linear H2-opt for genotype fold k (paper settings) |
| `scripts/sorghum/train_h2opt_lbfgs.py --fold k` | linear H2-opt trained to convergence with L-BFGS (ridge and stopping tuned on inner validation) |
| `scripts/sorghum/evaluate_heritability.py [--paper \| --lbfgs]` | train/test heritability of H2-opt, PCA, genetic PCA, LDA, PCH and max features |
| `scripts/sorghum/plot_heritability.py [--lbfgs]` | boxplots of train/test heritability of each method's top trait over the folds |
| `scripts/sorghum/gwas.py --paper --fold k` | GEMMA GWAS of the per-line trait means (intercept + 5 genotype PCs) |
