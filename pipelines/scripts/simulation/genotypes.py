"""Genotypes for the GWAS of one simulation instance: the sorghum PLINK set without the
instance's 300 QTNs (the causal SNPs are not tested themselves; they are found through linked SNPs)
and the top genotype PCs of that set over all lines.

    python pipelines/scripts/simulation/genotypes.py --instance 0 --out-prefix genotypes \\
        --out-pcs pcs.npy [--pcs 5]
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, simulation, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--out-prefix', required=True, help='PLINK prefix to write')
    parser.add_argument('--out-pcs', required=True, help='.npy, lines x pcs, .fam order')
    parser.add_argument('--pcs', type=int, default=5)
    args = parser.parse_args()

    bim = np.loadtxt(f'{sorghum.GENOTYPES}.bim', dtype=str, usecols=(0, 3))
    located = bim.astype(np.int64)
    qtns = simulation.qtns(args.instance).reshape(-1, 2)
    is_qtn = (located[:, None, :] == qtns[None, :, :]).all(axis=2).any(axis=1)
    if is_qtn.sum() != len(np.unique(qtns, axis=0)):
        raise ValueError(f'{is_qtn.sum()} QTNs found in the .bim, expected {len(qtns)}')

    gwas.write_snp_subset(sorghum.GENOTYPES, ~is_qtn, args.out_prefix)
    np.save(args.out_pcs, gwas.genotype_pcs(args.out_prefix, args.pcs))
    print(f'instance {args.instance}: {is_qtn.sum()} QTNs removed, {(~is_qtn).sum()} SNPs kept')


if __name__ == '__main__':
    main()
