"""The GWAS overview of the paper (sorNull): the real H2-opt traits of the 10 folds, their
cross-fitted GWAS and the permutation null, in the style of simNull.

Four panels, one box per trait for the real folds (color of H2-opt, a point per fold) and one for
the null permutations (gray, a point per permutation), the cross-fit as a black diamond (traits
with a cross-fitted GWAS only): a, the test heritability (heritability_h2opt.npz of
evaluate_heritability.py; null_gwas.py h2_test); b, the largest -log10 p (Bonferroni line
0.05 / SNPs); c, the QQ plot of the trait with significant SNPs in the most folds (the first
such trait on a tie; trait 1 if no trait has one): each fold, the cross-fit and each permutation;
d, the genomic-control lambda. The figure is drawn at the text width of the paper (6.5 in) and
saved as .pdf and, beside it, .png; the numbers are printed.

    python pipelines/scripts/sorghum/plot_paper_null.py --gwas gwas/h2opt_fold*.npz \\
        --heritability heritability_h2opt.npz --null null/perm*.npz \\
        --crossfit gwas/h2opt_crossfit_trait1.npz ... --out figures/paper/sorNull.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MARK_COLOR,
    MATH_RC,
    NULL_COLOR,
    NULL_LINE_COLOR,
    SIM_FIG_RC,
    letter,
    qq_real_vs_null,
    real_vs_null,
)

FIG_SIZE = (6.5, 4.4)
LETTER_X = -0.13  # x of the panel letters, in axes coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gwas', nargs='+', required=True,
                        help='gwas.py result of each fold, in fold order')
    parser.add_argument('--heritability', required=True,
                        help='evaluate_heritability.py result (method h2opt)')
    parser.add_argument('--null', nargs='+', required=True, help='null_gwas.py results')
    parser.add_argument('--crossfit', nargs='+', required=True,
                        help='gwas.py result of the cross-fitted traits 1, 2, ... in order')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()

    folds = [np.load(p) for p in args.gwas]
    nulls = [np.load(p) for p in args.null]
    crossfit = [np.load(p) for p in args.crossfit]
    n_traits = folds[0]['neg_log10_p'].shape[0]
    assert all(n['neg_log10_p'].shape[0] >= n_traits for n in nulls), 'null has fewer traits'
    n_snps = folds[0]['neg_log10_p'].shape[1]
    threshold = -np.log10(0.05 / n_snps)
    with np.load(args.heritability) as f:
        h2 = f['test'][list(f['methods']).index('h2opt'), :, :n_traits]  # folds x traits
    real_max = np.array([f['neg_log10_p'].max(axis=1) for f in folds])
    null_max = np.array([n['neg_log10_p'][:n_traits].max(axis=1) for n in nulls])
    real_lambda = np.array([f['lambda_gc'] for f in folds])
    null_lambda = np.array([n['lambda_gc'][:n_traits] for n in nulls])
    null_h2 = np.array([n['h2_test'][:n_traits] for n in nulls])
    mark_max = np.full(n_traits, np.nan)
    mark_lambda = np.full(n_traits, np.nan)
    for t, c in enumerate(crossfit[:n_traits]):
        mark_max[t] = c['neg_log10_p'][0].max()
        mark_lambda[t] = c['lambda_gc'][0]
    hit_folds = (real_max > threshold).sum(axis=0)
    qq_trait = int(np.argmax(hit_folds))

    print(f'{len(folds)} folds, {len(nulls)} permutations, {len(crossfit)} cross-fitted traits, '
          f'Bonferroni -log10 p {threshold:.2f}')
    for t in range(n_traits):
        print(f'trait {t + 1}: test h2 median {np.median(h2[:, t]):.3f} (null '
              f'{np.median(null_h2[:, t]):.3f}); folds with a significant SNP {hit_folds[t]}, '
              f'largest -log10 p median {np.median(real_max[:, t]):.2f} (null max '
              f'{null_max[:, t].max():.2f}, cross-fit {mark_max[t]:.2f}); lambda GC median '
              f'{np.median(real_lambda[:, t]):.3f} (null {np.median(null_lambda[:, t]):.3f})')
    print(f'null traits with a significant SNP: {(null_max > threshold).sum()} of '
          f'{null_max.size}; QQ plot of trait {qq_trait + 1}')

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=FIG_SIZE)
        real_vs_null(a, h2, null_h2)
        a.axhline(0, color='0.3', linewidth=0.8, linestyle=':')
        a.set_ylabel('test heritability')
        real_vs_null(b, real_max, null_max, mark_max)
        b.axhline(threshold, color='black', linewidth=0.8, linestyle=':')
        b.set_ylabel(r'largest $-\log_{10} p$')
        mark = crossfit[qq_trait]['neg_log10_p'][0] if qq_trait < len(crossfit) else None
        qq_real_vs_null(c, [f['neg_log10_p'][qq_trait] for f in folds],
                        [n['neg_log10_p'][qq_trait] for n in nulls], mark)
        c.set_ylabel(rf'observed $-\log_{{10}} p$ (trait {qq_trait + 1})')
        real_vs_null(d, real_lambda, null_lambda, mark_lambda)
        d.axhline(1, color='0.3', linewidth=0.8, linestyle=':')
        d.set_ylabel(r'$\lambda_\mathrm{GC}$')
        for ax, text in zip((a, b, c, d), 'abcd', strict=True):
            letter(ax, text, LETTER_X)
        handles = [Line2D([], [], marker='s', color=COLORS['h2opt'], linestyle='-', markersize=4),
                   Line2D([], [], marker='D', color=MARK_COLOR, markeredgecolor='white',
                          linestyle='-', markersize=4),
                   Line2D([], [], marker='o', color=NULL_LINE_COLOR, markerfacecolor=NULL_COLOR,
                          markeredgewidth=0, linestyle='-', markersize=3)]
        fig.legend(handles, [f'H2-opt, {len(folds)} cross-validation folds', 'H2-opt, cross-fit',
                             'permutation null'], loc='upper center', ncol=3, frameon=False,
                   handlelength=2, handletextpad=0.5, columnspacing=1.4)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(out.with_suffix('.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
