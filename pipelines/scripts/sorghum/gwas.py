"""GEMMA GWAS of sorghum H2-opt traits on the 836 sequenced lines.

The phenotype of a line is the mean of each standardized trait over the plants of all its
accessions, rank-normal transformed (gwas.line_means). Lines without plants are left out. The
covariates are an intercept and the genotype PCs of genotype_pcs.py. Writes an .npz
(gwas.summarize); GEMMA's own files go to --work.

    python pipelines/scripts/sorghum/gwas.py --h2opt-traits fold0.npz --pcs genotype_pcs.npy \\
        --out paper_fold0.npz --work work/paper_fold0 --traits 5 --jobs 1 --threads 4
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h2opt-traits', required=True, help='traits of all plants (.npz)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    Y = sorghum.load_traits(args.h2opt_traits)[:, :args.traits]
    _, genotypes, _ = sorghum.load()
    n_lines = len(gwas.read_fam(sorghum.GENOTYPES))
    phenotypes = gwas.line_means(Y, sorghum.line_of_plant(genotypes), n_lines)
    result = gwas.run_gemma(sorghum.GENOTYPES, phenotypes, args.work, covariates=np.load(args.pcs),
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
