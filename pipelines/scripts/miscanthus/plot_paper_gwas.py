"""GWAS figure of the paper's supplement (miscGWAS_<population>): the GWAS of trait 1 of H2-opt
(CNN) of the adjusted run on each date with a significant SNP, per fold and cross-fitted, in the
style of sorGWAS (figures.gwas_fold_rows, with the dates in place of the traits).

The loci are the Bonferroni hits (0.05 / SNPs) of all dates and folds of the population, grouped
by LD clumping (gwas.clump: r^2 above --r2 with the lead SNP within --kb kilobases, LD from
--genotypes). Each locus is labeled by chromosome and the position of its lead SNP; a locus on an
unplaced scaffold (chromosome 0, at the right of the x axis as Un) by its scaffold. Saves --out
(.pdf) and a .png next to it, and prints the loci.

    python pipelines/scripts/miscanthus/plot_paper_gwas.py --population central \\
        --gwas central/gwas/date{0..13}.npz --crossfit central/gwas/crossfit_date{0..13}.npz \\
        --genotypes central/gwas/genotypes --r2 0.5 --kb 250 \\
        --out figures/paper/miscGWAS_central.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402

from h2opt_analysis import gwas, miscanthus  # noqa: E402
from h2opt_analysis.figures import gwas_fold_rows  # noqa: E402

UNPLACED = 0  # chromosome of the unplaced scaffolds


def locus_label(chrom, ps, bim, lead):
    """Chromosome and position in Mb of the lead SNP; for an unplaced scaffold, its scaffold
    (from the SNP ID in the .bim, e.g. SSCAFFOLD00943_46847), or Un if the position is not
    unique among the scaffolds."""
    if chrom[lead] != UNPLACED:
        return f'{chrom[lead]}: {ps[lead] / 1e6:.2f} Mb'
    ids = bim[(bim[:, 0] == str(UNPLACED)) & (bim[:, 2] == str(ps[lead])), 1]
    if len(ids) != 1:
        return 'Un'
    return f'scaffold {int(ids[0].split("_")[0].removeprefix("SSCAFFOLD"))}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--gwas', nargs='+', required=True, help='gwas.py output of each date')
    parser.add_argument('--crossfit', nargs='+', required=True,
                        help='cross-fitted GWAS of each date')
    parser.add_argument('--genotypes', required=True, help='PLINK set (prefix) for the LD')
    parser.add_argument('--r2', type=float, required=True)
    parser.add_argument('--kb', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    assert len(args.gwas) == len(args.crossfit) == len(miscanthus.DATES)

    dates = [np.load(path) for path in args.gwas]
    chrom, ps = dates[0]['chr'], dates[0]['ps']
    crossfit = [np.load(path) for path in args.crossfit]
    for z in dates[1:] + crossfit:
        assert np.array_equal(z['chr'], chrom) and np.array_equal(z['ps'], ps), \
            'the GWAS have different SNPs'
    score = np.stack([z['neg_log10_p'] for z in dates], axis=1)  # folds x dates x SNPs
    lambdas = np.stack([z['lambda_gc'] for z in dates], axis=1)
    _, n_dates, n_snps = score.shape
    loci = gwas.clump(args.genotypes, chrom, ps, score.reshape(-1, n_snps), r2=args.r2,
                      kb=args.kb)
    bim = np.loadtxt(f'{args.genotypes}.bim', dtype=str, usecols=(0, 1, 3))
    labels = [locus_label(chrom, ps, bim, locus['lead']) for locus in loci]
    print(f'{args.population}: {len(loci)} loci (LD clumping, r^2 > {args.r2} within '
          f'{args.kb} kb):')
    for locus, label in zip(loci, labels, strict=True):
        rows = [f'fold {r // n_dates + 1} {miscanthus.DATES[r % n_dates]}' for r in locus['rows']]
        print(f'  {label} (lead {ps[locus["lead"]]}, -log10 p '
              f'{score.reshape(-1, n_snps)[:, locus["lead"]].max():.2f}): {", ".join(rows)}')

    crossfit_score = np.stack([z['neg_log10_p'][0] for z in crossfit])
    crossfit_lambdas = np.array([z['lambda_gc'][0] for z in crossfit])
    placed = [c for c in np.unique(chrom) if c != UNPLACED]
    tick_names = {c: str(c) for c in placed}
    if (chrom == UNPLACED).any():
        tick_names[UNPLACED] = '\nUn'  # below the tick of the last chromosome
    ymax = np.ceil(max(score.max(), crossfit_score.max())) + 0.5
    panels = gwas_fold_rows(args.out, chrom, ps, score, lambdas, crossfit_score,
                            crossfit_lambdas, loci, labels, miscanthus.DATES, tick_names, ymax)
    print('panels:', ', '.join(f'{miscanthus.DATES[d]} '
                               + ('cross-fit' if f is None else f'fold {f + 1}')
                               for d, f in panels))


if __name__ == '__main__':
    main()
