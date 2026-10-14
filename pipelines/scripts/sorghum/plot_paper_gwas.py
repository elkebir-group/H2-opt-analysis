"""GWAS figure of the paper (sorGWAS): the GWAS of the sorghum synthetic traits with significant
SNPs, per fold and cross-fitted.

The loci are the Bonferroni hits (0.05 / SNPs) of all traits and folds, grouped by LD clumping
(gwas.clump: r^2 above --r2 with the lead SNP within --kb kilobases, LD from --genotypes). For each
trait with a significant SNP in some fold, one row per such fold and then a row of the trait's
cross-fitted GWAS (bold label). Each row has a QQ plot (left, observed against expected -log10 p,
with lambda GC) and a Manhattan plot (right; dotted line: the Bonferroni threshold). In a fold
row, a circle marks the strongest significant SNP of each locus that the trait hits in that fold;
in the cross-fit row, the SNP of each of the trait's loci with the largest cross-fitted -log10 p.
Each locus is labeled by chromosome and the position of its lead SNP. Folds are numbered from 1,
as in the paper (drawn by figures.gwas_fold_rows). Saves --out (.pdf) and a .png next to it,
and prints the loci.

    python pipelines/scripts/sorghum/plot_paper_gwas.py --gwas h2opt_fold{0..9}.npz \\
        --crossfit h2opt_crossfit_trait{1..10}.npz --genotypes WEST_original --r2 0.5 --kb 250 \\
        --out figures/paper/sorGWAS.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import numpy as np  # noqa: E402

from h2opt_analysis import gwas  # noqa: E402
from h2opt_analysis.figures import gwas_fold_rows  # noqa: E402

YMAX = 9.5  # -log10 p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', nargs='+', required=True, help='gwas.py output of each fold')
    parser.add_argument('--crossfit', nargs='+', required=True,
                        help='cross-fitted GWAS of traits 1, 2, ... (one .npz each)')
    parser.add_argument('--genotypes', required=True, help='PLINK set (prefix) for the LD')
    parser.add_argument('--r2', type=float, required=True)
    parser.add_argument('--kb', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    folds = [np.load(path) for path in args.gwas]
    chrom, ps = folds[0]['chr'], folds[0]['ps']
    for f in folds[1:]:
        assert np.array_equal(f['ps'], ps), 'the folds have different SNPs'
    score = np.stack([f['neg_log10_p'] for f in folds])  # folds x traits x SNPs
    n_folds, n_traits, n_snps = score.shape
    loci = gwas.clump(args.genotypes, chrom, ps, score.reshape(-1, n_snps), r2=args.r2,
                      kb=args.kb)
    labels = [f'{chrom[locus["lead"]]}: {ps[locus["lead"]] / 1e6:.2f} Mb' for locus in loci]
    print(f'{len(loci)} loci (LD clumping, r^2 > {args.r2} within {args.kb} kb):')
    for locus, label in zip(loci, labels, strict=True):
        rows = [f'fold {r // n_traits + 1} trait {r % n_traits + 1}' for r in locus['rows']]
        print(f'  {label} (lead {ps[locus["lead"]]}, -log10 p '
              f'{score.reshape(-1, n_snps)[:, locus["lead"]].max():.2f}): {", ".join(rows)}')

    crossfit = []
    for path in args.crossfit:
        with np.load(path) as z:
            assert np.array_equal(z['ps'], ps), 'the cross-fit has different SNPs'
            crossfit.append((z['neg_log10_p'][0], z['lambda_gc'][0]))
    lambdas = np.stack([f['lambda_gc'] for f in folds])
    panels = gwas_fold_rows(args.out, chrom, ps, score, lambdas, [c[0] for c in crossfit],
                            [c[1] for c in crossfit], loci, labels,
                            [f'trait {t + 1}' for t in range(n_traits)],
                            {c: str(c) for c in np.unique(chrom)}, YMAX)
    print('panels:', ', '.join(f'trait {t + 1} ' + ('cross-fit' if f is None else f'fold {f + 1}')
                               for t, f in panels))


if __name__ == '__main__':
    main()
