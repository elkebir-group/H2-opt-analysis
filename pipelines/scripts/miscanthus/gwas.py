"""GEMMA GWAS of trait 1 of one Miscanthus date, one GWAS per fold's model.

The phenotype of a genotyped plant is its standardized trait (gwas.line_means with one plant per
line). Plants without genotypes are left out. The covariates are an intercept and the genotype PCs
of genotypes.py. Writes an .npz (gwas.summarize; its rows are the folds); GEMMA's own files go to
--work.

    python pipelines/scripts/miscanthus/gwas.py --population central --h2opt-traits date2.npz \\
        --genotypes genotypes --pcs genotype_pcs.npy --out date2.npz --work work/date2 \\
        --jobs 3 --threads 1
"""

import argparse

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
    args = parser.parse_args()

    Y = np.load(args.h2opt_traits)['traits'][:, :, 0].T  # plants x folds, trait 1
    rows = miscanthus.genotype_rows(args.population, f'{args.genotypes}.fam')
    phenotypes = gwas.line_means(Y, rows, len(gwas.read_fam(args.genotypes)))
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    summary = gwas.summarize(result)
    np.savez_compressed(args.out, **summary)
    threshold = -np.log10(0.05 / summary['neg_log10_p'].shape[1])
    for fold, score in enumerate(summary['neg_log10_p']):
        print(f'fold {fold}: {len(score)} SNPs, max -log10 p {score.max():.2f}, '
              f'{np.sum(score > threshold)} Bonferroni hits, '
              f'lambda GC {summary["lambda_gc"][fold]:.3f}', flush=True)


if __name__ == '__main__':
    main()
