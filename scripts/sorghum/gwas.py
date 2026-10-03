"""GEMMA GWAS of sorghum H2-opt traits on the 836 sequenced lines (per-line mean of each trait over its plants).

Lines without plants are left out; the covariates are an intercept and the top genotype PCs.
Writes results/sorghum/gwas/{paper,lbfgs}_fold{k}.npz (see h2opt_analysis.gwas.run_gemma).

    uv run python scripts/sorghum/gwas.py --paper --fold 0
"""

import argparse
from pathlib import Path

import numpy as np
from h2opt import load_npz

from h2opt_analysis import gwas, sorghum
from h2opt_analysis.paths import data

RESULTS = Path('results/sorghum/gwas')


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--paper', action='store_true', help="the paper's H2-opt predictions")
    source.add_argument('--lbfgs', action='store_true', help='the L-BFGS H2-opt traits (train_h2opt_lbfgs.py)')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--pcs', type=int, default=5, help='genotype PCs as covariates')
    parser.add_argument('--jobs', type=int, default=1, help='traits run at once')
    parser.add_argument('--threads', type=int, default=4, help='OpenBLAS threads per GEMMA process')
    args = parser.parse_args()

    if args.paper:
        Y = load_npz(data('plant', 'syntheticTraits', f'linear_crossVal_reg4_{args.fold}_mod10.npz'))
    else:
        Y = np.load(Path('results/sorghum/h2opt_lbfgs') / f'fold{args.fold}.npz')['traits']
    Y = Y[:, :args.traits]
    _, genotypes, _ = sorghum.load()
    phenotypes = sorghum.line_phenotypes(Y, genotypes)
    covariates = gwas.genotype_pcs(sorghum.GENOTYPES, args.pcs, individuals=~np.isnan(phenotypes[:, 0]))
    name = f"{'paper' if args.paper else 'lbfgs'}_fold{args.fold}"
    result = gwas.run_gemma(sorghum.GENOTYPES, phenotypes, RESULTS / 'work' / name, covariates=covariates,
                            jobs=args.jobs, threads=args.threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / f'{name}.npz', **result)
    for t in range(args.traits):
        p = result['p_wald'][t]
        print(f'trait {t + 1}: {len(p)} SNPs, min p {p.min():.2e}, {np.sum(p < 0.05 / len(p))} Bonferroni hits')


if __name__ == '__main__':
    main()
