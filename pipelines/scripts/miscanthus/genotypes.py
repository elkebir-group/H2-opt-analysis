"""PLINK set of one Miscanthus population for GEMMA, and its genotype PCs (the GWAS covariates).

The set is the population's .bed with its full sample list (miscanthus.GENOTYPE_SAMPLES; the .fam
beside the .bed is a subset and does not match it) and unplaced scaffolds as chromosome 0
(gwas.write_plink_set). The PCs are computed over the genotyped plants that have traits (the kept
plants of miscanthus.design); the other samples get NaN. Writes <out>.{bed,bim,fam} and a .npy
(samples x PCs, .fam order).

    python pipelines/scripts/miscanthus/genotypes.py --population central --out genotypes \\
        --out-pcs genotype_pcs.npy --pcs 5
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--out', required=True, help='PLINK prefix of the set')
    parser.add_argument('--out-pcs', required=True)
    parser.add_argument('--pcs', type=int, required=True)
    args = parser.parse_args()

    gwas.write_plink_set(miscanthus.GENOTYPES[args.population],
                         miscanthus.GENOTYPE_SAMPLES[args.population], args.out)
    rows = miscanthus.genotype_rows(args.population, f'{args.out}.fam')
    phenotyped = np.zeros(len(gwas.read_fam(args.out)), bool)
    phenotyped[rows[rows >= 0]] = True
    np.save(args.out_pcs, gwas.genotype_pcs(args.out, args.pcs, individuals=phenotyped))
    print(f'{phenotyped.sum()} of {len(phenotyped)} samples have traits, '
          f'{(rows >= 0).sum()} of {len(rows)} plants genotyped')


if __name__ == '__main__':
    main()
