"""PLINK set of the genotyped Arabidopsis accessions for GEMMA, and the genotype PCs of one
treatment's accessions (the GWAS covariates).

The set is the .bed with its full sample list (arabidopsis.GENOTYPE_SAMPLES; the .fam beside the
.bed is a subset and does not match it), written by gwas.write_plink_set. The PCs are computed over
the accessions of the treatment; the other samples get NaN. Writes <out>.{bed,bim,fam} and a .npy
(samples x PCs, .fam order).

    python pipelines/scripts/arabidopsis/genotypes.py --treatment 0d --out genotypes \\
        --out-pcs genotype_pcs.npy --pcs 5
"""

import argparse

import numpy as np

from h2opt_analysis import arabidopsis, gwas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--out', required=True, help='PLINK prefix of the set')
    parser.add_argument('--out-pcs', required=True)
    parser.add_argument('--pcs', type=int, required=True)
    args = parser.parse_args()

    gwas.write_plink_set(arabidopsis.GENOTYPES, arabidopsis.GENOTYPE_SAMPLES, args.out)
    _, accessions, _ = arabidopsis.design(args.treatment)
    rows = arabidopsis.genotype_rows(accessions, f'{args.out}.fam')
    phenotyped = np.zeros(len(gwas.read_fam(args.out)), bool)
    phenotyped[rows[rows >= 0]] = True
    np.save(args.out_pcs, gwas.genotype_pcs(args.out, args.pcs, individuals=phenotyped))
    print(f'{phenotyped.sum()} of {len(phenotyped)} accessions have samples, '
          f'{(rows >= 0).sum()} of {len(rows)} samples genotyped')


if __name__ == '__main__':
    main()
