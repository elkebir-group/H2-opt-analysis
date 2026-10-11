"""GEMMA GWAS of the H2-opt traits of one Arabidopsis treatment and fold.

The phenotype of an accession is the mean of each standardized trait over its samples
(gwas.line_means, rank-normal transformed). Accessions without samples are left out. The covariates
are an intercept and the genotype PCs of genotypes.py. Writes an .npz (gwas.summarize); GEMMA's own
files go to --work.

    python pipelines/scripts/arabidopsis/gwas.py --treatment 0d --h2opt-traits fold0.npz \\
        --genotypes genotypes --pcs genotype_pcs.npy --out fold0.npz --work work/fold0 \\
        --traits 10 --jobs 3 --threads 1
"""

import argparse

import numpy as np

from h2opt_analysis import arabidopsis, gwas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--h2opt-traits', required=True, help='traits of the samples (.npz)')
    parser.add_argument('--genotypes', required=True, help='PLINK prefix (genotypes.py)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--traits', type=int, required=True, help='first traits analyzed')
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    Y = np.load(args.h2opt_traits)['traits'][:, :args.traits]
    _, accessions, _ = arabidopsis.design(args.treatment)
    rows = arabidopsis.genotype_rows(accessions, f'{args.genotypes}.fam')
    phenotypes = gwas.line_means(Y, rows, len(gwas.read_fam(args.genotypes)))
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    summary = gwas.summarize(result)
    np.savez_compressed(args.out, **summary)
    threshold = -np.log10(0.05 / summary['neg_log10_p'].shape[1])
    for t, score in enumerate(summary['neg_log10_p']):
        print(f'trait {t + 1}: {len(score)} SNPs, max -log10 p {score.max():.2f}, '
              f'{np.sum(score > threshold)} Bonferroni hits, '
              f'lambda GC {summary["lambda_gc"][t]:.3f}', flush=True)


if __name__ == '__main__':
    main()
