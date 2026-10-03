"""GEMMA GWAS of sorghum H2-opt traits on the 836 sequenced lines.

The phenotype of a line is the mean of each standardized trait over the plants of all its
accessions. Lines without plants are left out. The covariates are an intercept and the top
genotype PCs, computed over the phenotyped lines. Writes an .npz (see
h2opt_analysis.gwas.run_gemma); GEMMA's own files go to --work.

    python pipelines/scripts/sorghum/gwas.py --h2opt-traits fold0.npz --out paper_fold0.npz \\
        --work work/paper_fold0 [--traits 5] [--pcs 5] [--jobs 1] [--threads 4]
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h2opt-traits', required=True, help='traits of all plants (.npz)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--pcs', type=int, default=5, help='genotype PCs as covariates')
    parser.add_argument('--jobs', type=int, default=1, help='traits run at once')
    parser.add_argument('--threads', type=int, default=4, help='OpenBLAS threads per GEMMA run')
    args = parser.parse_args()

    Y = sorghum.load_traits(args.h2opt_traits)[:, :args.traits]
    _, genotypes, _ = sorghum.load()
    phenotypes = sorghum.line_phenotypes(Y, genotypes)
    phenotyped = ~np.isnan(phenotypes[:, 0])
    covariates = gwas.genotype_pcs(sorghum.GENOTYPES, args.pcs, individuals=phenotyped)
    result = gwas.run_gemma(sorghum.GENOTYPES, phenotypes, args.work, covariates=covariates,
                            jobs=args.jobs, threads=args.threads)
    np.savez_compressed(args.out, **result)
    for t in range(args.traits):
        p = result['p_wald'][t]
        print(f'trait {t + 1}: {len(p)} SNPs, min p {p.min():.2e}, '
              f'{np.sum(p < 0.05 / len(p))} Bonferroni hits')


if __name__ == '__main__':
    main()
