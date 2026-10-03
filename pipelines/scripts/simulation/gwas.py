"""GEMMA GWAS of the first three traits of one method on one simulation instance.

The phenotype of a line is the mean of each standardized trait over its plants (all 836 lines
have plants). The covariates are an intercept and the genotype PCs. Writes an .npz with chr, ps
(per SNP), neg_log10_p (traits x SNPs, float32) and lambda_gc (per trait).

    python pipelines/scripts/simulation/gwas.py --spectra spectra.npz --traits pch.npz \\
        --genotypes genotypes --pcs pcs.npy --out gwas_pch.npz --work work/pch \\
        [--jobs 3] [--threads 2]
"""

import argparse

import numpy as np
from scipy import stats

from h2opt_analysis import gwas, simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--traits', required=True, help=".npz with array 'traits' (all plants)")
    parser.add_argument('--genotypes', required=True, help='PLINK prefix')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--jobs', type=int, default=3, help='traits run at once')
    parser.add_argument('--threads', type=int, default=2, help='OpenBLAS threads per GEMMA run')
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        lines = f['lines']
    with np.load(args.traits) as f:
        Y = f['traits'][:, :simulation.N_TRAITS]
    n_lines = len(gwas.read_fam(args.genotypes))
    phenotypes = simulation.line_means(Y, lines, n_lines)
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)

    p = result['p_wald']
    lambda_gc = np.median(stats.chi2.isf(p, 1), axis=1) / stats.chi2.ppf(0.5, 1)
    np.savez_compressed(args.out, chr=result['chr'].astype(np.int16),
                        ps=result['ps'].astype(np.int64),
                        neg_log10_p=(-np.log10(p)).astype(np.float32), lambda_gc=lambda_gc)
    for t in range(p.shape[0]):
        print(f'trait {t + 1}: {p.shape[1]} SNPs, min p {p[t].min():.2e}, '
              f'lambda GC {lambda_gc[t]:.3f}', flush=True)


if __name__ == '__main__':
    main()
