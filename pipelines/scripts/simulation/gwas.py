"""GEMMA GWAS of the first three traits of one method on one simulation instance.

The phenotype of a line is the mean of each standardized trait over its plants, rank-normal
transformed (gwas.line_means; all 836 lines have plants). The covariates are an intercept and the
genotype PCs. Writes an .npz (gwas.summarize).

    python pipelines/scripts/simulation/gwas.py --spectra spectra.npz --traits pch.npz \\
        --genotypes genotypes --pcs pcs.npy --out gwas_pch.npz --work work/pch \\
        --jobs 3 --threads 2
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--traits', required=True, help=".npz with array 'traits' (all plants)")
    parser.add_argument('--genotypes', required=True, help='PLINK prefix')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--jobs', type=int, required=True, help='traits run at once')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads per run')
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        lines = f['lines']
    with np.load(args.traits) as f:
        Y = f['traits'][:, :simulation.N_TRAITS]
    n_lines = len(gwas.read_fam(args.genotypes))
    phenotypes = gwas.line_means(Y, lines, n_lines)
    result = gwas.run_gemma(args.genotypes, phenotypes, args.work, covariates=np.load(args.pcs),
                            jobs=args.jobs, threads=args.threads)
    summary = gwas.summarize(result)
    np.savez_compressed(args.out, **summary)
    gwas.report(summary)


if __name__ == '__main__':
    main()
