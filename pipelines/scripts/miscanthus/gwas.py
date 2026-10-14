"""GEMMA GWAS of trait 1 of one Miscanthus date, one GWAS per fold's model.

The phenotype of a genotyped plant is its standardized trait, rank-normal transformed
(gwas.line_means with one plant per line). Plants without genotypes are left out. The covariates are
an intercept, the genotype PCs of genotypes.py and the neighbor mean (Papadakis 1937 as an
analysis of covariance): the mean of the trait over the plants of the other families' plots within
--neighbor-width field columns in the same rep (miscanthus.neighbor_weights), after the rank-based
inverse normal transform of the trait over all plants; each fold gets its own GEMMA run. Writes an
.npz (gwas.summarize; its rows are the folds); GEMMA's own files go to --work.

    python pipelines/scripts/miscanthus/gwas.py --population central --h2opt-traits date2.npz \\
        --genotypes genotypes --pcs genotype_pcs.npy --out date2.npz --work work/date2 \\
        --jobs 3 --threads 1 --neighbor-width 3
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from h2opt_analysis import gwas, miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--h2opt-traits', required=True, help='traits per fold (.npz)')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix (genotypes.py)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--jobs', type=int, required=True, help='folds run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    parser.add_argument('--neighbor-width', type=int, required=True,
                        help='the neighbor mean of the trait is a covariate (field columns)')
    args = parser.parse_args()

    Y = np.load(args.h2opt_traits)['traits'][:, :, 0].T  # plants x folds, trait 1
    rows = miscanthus.genotype_rows(args.population, f'{args.genotypes}.fam')
    n_lines = len(gwas.read_fam(args.genotypes))
    phenotypes = gwas.line_means(Y, rows, n_lines)
    pcs = np.load(args.pcs)
    weights = miscanthus.neighbor_weights(miscanthus.design(args.population), args.neighbor_width)
    neighbor = miscanthus.neighbor_covariates(Y, weights, rows, n_lines)  # lines x folds

    def run(f):
        return gwas.run_gemma(args.genotypes, phenotypes[:, f], f'{args.work}/fold{f}',
                              covariates=np.column_stack([pcs, neighbor[:, f]]), jobs=1,
                              threads=args.threads)

    with ThreadPoolExecutor(args.jobs) as pool:
        runs = list(pool.map(run, range(Y.shape[1])))
    for r in runs[1:]:
        if not all(np.array_equal(r[k], runs[0][k]) for k in ('chr', 'rs', 'ps')):
            raise ValueError('the GEMMA runs of the folds report different SNPs')
    result = {k: v for k, v in runs[0].items() if k not in gwas.ASSOC_FIELDS}
    result.update({k: np.concatenate([r[k] for r in runs]) for k in gwas.ASSOC_FIELDS})
    summary = gwas.summarize(result)
    np.savez_compressed(args.out, **summary)
    gwas.report(summary, [f'fold {f}' for f in range(len(summary['lambda_gc']))])


if __name__ == '__main__':
    main()
