"""Top genotype PCs of the sequenced sorghum lines that have plants (the GWAS covariates).

The PCs are computed over the phenotyped lines; the other lines get NaN. Writes a .npy (lines x
PCs, .fam order).

    python pipelines/scripts/sorghum/genotype_pcs.py --out genotype_pcs.npy --pcs 5
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--pcs', type=int, required=True)
    args = parser.parse_args()

    _, genotypes, _ = sorghum.load()
    lines = sorghum.line_of_plant(genotypes)
    phenotyped = np.zeros(len(gwas.read_fam(sorghum.GENOTYPES)), bool)
    phenotyped[lines[lines >= 0]] = True
    np.save(args.out, gwas.genotype_pcs(sorghum.GENOTYPES, args.pcs, individuals=phenotyped))
    print(f'{phenotyped.sum()} of {len(phenotyped)} lines phenotyped, '
          f'{(lines >= 0).sum()} of {len(lines)} plants on a line')


if __name__ == '__main__':
    main()
