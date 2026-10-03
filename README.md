# H2-opt-analysis

Analysis code for the H2-opt paper (sorghum, Miscanthus, metabolomics, simulations): data pipelines, baselines,
permuted-genotype nulls, cross-fit GWAS, genomic inflation (λGC), and figures.

The method itself lives in [H2-opt](https://github.com/elkebir-group/H2-opt); this repo depends on it.

## Setup

```bash
uv sync                      # installs h2opt from the refactor branch of H2-opt
ln -s /path/to/h2opt-data data   # or set H2OPT_DATA=/path/to/h2opt-data
```

Scripts are run from the repository root and write to `results/`.

## Sorghum

| Script | Result |
|---|---|
| `scripts/sorghum/train_h2opt.py --fold k` | linear H2-opt for genotype fold k (paper settings) |
| `scripts/sorghum/evaluate_heritability.py [--paper]` | train/test heritability of H2-opt, PCA, genetic PCA, LDA, PCH and max features |
