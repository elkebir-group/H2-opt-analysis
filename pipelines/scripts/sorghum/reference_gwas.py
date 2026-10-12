"""GEMMA GWAS of one sorghum reference trait: NDVI or one measured trait.

The reference traits show which loci known traits find with the GWAS of gwas.py. NDVI (--ndvi, the
output of ndvi.py) is a value per plant, as the H2-opt traits. A measured trait (Ferguson et al.
2021, Supplemental Table S12; sorghum.measured_traits) is a genotype mean. The phenotype of a line
is the mean over its plants or genotypes, rank-normal transformed (gwas.line_means). Lines without
a value are left out, and so are lines without plants, as in gwas.py. The covariates are an
intercept and the genotype PCs of genotype_pcs.py. Writes an .npz (gwas.summarize); GEMMA's own
files go to --work.

    python pipelines/scripts/sorghum/reference_gwas.py --trait ndvi --ndvi ndvi.npz \\
        --pcs genotype_pcs.npy --out reference/ndvi.npz --work work/reference_ndvi --threads 1
"""

import argparse

import numpy as np

from h2opt_analysis import gwas, sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trait', required=True,
                        help="'ndvi' or a measured trait (sorghum.trait_slug of its name)")
    parser.add_argument('--ndvi', required=True, help='ndvi.py output (.npz)')
    parser.add_argument('--pcs', required=True, help='.npy of genotype PCs, .fam order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', required=True, help='working directory for GEMMA')
    parser.add_argument('--threads', type=int, required=True, help='OpenBLAS threads')
    args = parser.parse_args()

    _, plant_genotypes, _ = sorghum.load()
    if args.trait == 'ndvi':
        genotypes = plant_genotypes
        with np.load(args.ndvi) as f:
            values = f['ndvi']
    else:
        names, genotypes, measured = sorghum.measured_traits()
        values = measured[:, [sorghum.trait_slug(n) for n in names].index(args.trait)]
    line = sorghum.line_of_plant(genotypes)
    line[~np.isin(line, sorghum.line_of_plant(plant_genotypes))] = -1  # lines without plants
    known = ~np.isnan(values)
    n_lines = len(gwas.read_fam(sorghum.GENOTYPES))
    phenotypes = gwas.line_means(values[known], line[known], n_lines)
    print(f'{args.trait}: {np.sum(~np.isnan(phenotypes))} lines with a value', flush=True)
    result = gwas.run_gemma(sorghum.GENOTYPES, phenotypes, args.work, covariates=np.load(args.pcs),
                            threads=args.threads)
    summary = gwas.summarize(result)
    np.savez_compressed(args.out, **summary)
    gwas.report(summary, [args.trait])


if __name__ == '__main__':
    main()
