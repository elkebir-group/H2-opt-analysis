"""The Arabidopsis GWAS controls of the paper (arabiNull), in the style of sorNull: the H2-opt
traits of the 5 folds, the cross-fitted GWAS of trait 1 and the permutation null (null_gwas.py:
fold 0, traits 1 to null_traits), the traits that the null has. One row per treatment.

a: the test heritability (heritability.npz of evaluate_heritability.py; null_gwas.py h2_test). b:
the largest -log10 p (Bonferroni line 0.05 / SNPs). c: the QQ plot of the trait with significant
SNPs in the most folds (the first such trait on a tie): each fold, the cross-fit (trait 1 only)
and each permutation. d: the genomic-control lambda. One box per trait for the folds (color of
H2-opt, a point per fold) and one for the permutations (gray), the cross-fit as a black diamond.
Drawn at the text width of the paper (6.5 in), saved as .pdf and, beside it, .png; the numbers
are printed.

    python pipelines/scripts/arabidopsis/plot_paper_null.py \\
        --gwas-0d 0d/gwas/fold0.npz ... --heritability-0d 0d/heritability.npz \\
        --crossfit-0d 0d/gwas/crossfit.npz --null-0d 0d/null/perm*.npz ... (6d the same) \\
        --out figures/paper/arabiNull.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plot_paper_main import ROW_LABEL_SIZE, treatment_label  # noqa: E402

from h2opt_analysis import arabidopsis  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MARK_COLOR,
    MATH_RC,
    NULL_COLOR,
    NULL_LINE_COLOR,
    SIM_FIG_RC,
    TICK_SIZE,
    letter,
    qq_real_vs_null,
    real_vs_null,
)

FIG_SIZE = (6.5, 3.6)


def main():
    parser = argparse.ArgumentParser()
    for treatment in arabidopsis.TREATMENTS:
        parser.add_argument(f'--gwas-{treatment}', nargs='+', required=True,
                            help='gwas.py result of each fold, in fold order')
        parser.add_argument(f'--heritability-{treatment}', required=True,
                            help='evaluate_heritability.py result (method h2opt)')
        parser.add_argument(f'--crossfit-{treatment}', required=True,
                            help='gwas.py result of the cross-fitted trait 1')
        parser.add_argument(f'--null-{treatment}', nargs='+', required=True,
                            help='null_gwas.py results')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(2, 4, figsize=FIG_SIZE)
        for row, treatment in zip(axes, arabidopsis.TREATMENTS, strict=True):
            given = {name: getattr(args, f'{name}_{treatment}')
                     for name in ('gwas', 'heritability', 'crossfit', 'null')}
            folds = [np.load(p) for p in given['gwas']]
            nulls = [np.load(p) for p in given['null']]
            crossfit = np.load(given['crossfit'])
            n_traits = min(n['neg_log10_p'].shape[0] for n in nulls)
            threshold = -np.log10(0.05 / folds[0]['neg_log10_p'].shape[1])
            with np.load(given['heritability']) as f:
                h2 = f['test'][list(f['methods']).index('h2opt'), :, :n_traits]  # folds x traits
            real_max = np.array([f['neg_log10_p'][:n_traits].max(axis=1) for f in folds])
            null_max = np.array([n['neg_log10_p'][:n_traits].max(axis=1) for n in nulls])
            real_lambda = np.array([f['lambda_gc'][:n_traits] for f in folds])
            null_lambda = np.array([n['lambda_gc'][:n_traits] for n in nulls])
            null_h2 = np.array([n['h2_test'][:n_traits] for n in nulls])
            mark_max = np.full(n_traits, np.nan)
            mark_lambda = np.full(n_traits, np.nan)
            mark_max[0] = crossfit['neg_log10_p'][0].max()
            mark_lambda[0] = crossfit['lambda_gc'][0]
            hit_folds = (real_max > threshold).sum(axis=0)
            qq_trait = int(np.argmax(hit_folds))

            print(f'{treatment}: Bonferroni -log10 p {threshold:.2f}; folds with a significant '
                  f'SNP per trait {hit_folds.tolist()}; cross-fit trait 1 largest -log10 p '
                  f'{mark_max[0]:.2f}; null traits with a significant SNP '
                  f'{(null_max > threshold).sum()} of {null_max.size} (largest '
                  f'{null_max.max():.2f}); lambda GC folds {real_lambda.min():.2f}-'
                  f'{real_lambda.max():.2f}, null {null_lambda.min():.2f}-'
                  f'{null_lambda.max():.2f}; test h2 null median {np.median(null_h2):.3f}; QQ '
                  f'trait {qq_trait + 1}')

            a, b, c, d = row
            real_vs_null(a, h2, null_h2)
            a.axhline(0, color='0.3', linewidth=0.8, linestyle=':')
            a.set_ylabel('test heritability')
            real_vs_null(b, real_max, null_max, mark_max)
            b.axhline(threshold, color='black', linewidth=0.8, linestyle=':')
            b.set_ylabel(r'largest $-\log_{10} p$')
            mark = crossfit['neg_log10_p'][0] if qq_trait == 0 else None
            qq_real_vs_null(c, [f['neg_log10_p'][qq_trait] for f in folds],
                            [n['neg_log10_p'][qq_trait] for n in nulls], mark)
            c.set_ylabel(rf'observed $-\log_{{10}} p$ (trait {qq_trait + 1})')
            real_vs_null(d, real_lambda, null_lambda, mark_lambda)
            d.axhline(1, color='0.3', linewidth=0.8, linestyle=':')
            d.set_ylabel(r'$\lambda_\mathrm{GC}$')
            d.text(1.08, 0.5, treatment_label(treatment), transform=d.transAxes, rotation=-90,
                   ha='left', va='center', fontsize=ROW_LABEL_SIZE, fontweight='bold')
        for ax, text in zip(axes[0], 'abcd', strict=True):
            letter(ax, text, -0.4)
        for ax in axes[0]:
            ax.set_xlabel('')
        handles = [Line2D([], [], marker='s', color=COLORS['h2opt'], linestyle='-', markersize=4),
                   Line2D([], [], marker='D', color=MARK_COLOR, markeredgecolor='white',
                          linestyle='-', markersize=4),
                   Line2D([], [], marker='o', color=NULL_LINE_COLOR, markerfacecolor=NULL_COLOR,
                          markeredgewidth=0, linestyle='-', markersize=3)]
        fig.legend(handles, [f'H2-opt, {len(folds)} cross-validation folds',
                             'H2-opt, cross-fit (trait 1)', 'permutation null'],
                   loc='upper center', ncol=3, frameon=False, fontsize=TICK_SIZE,
                   handlelength=2, handletextpad=0.5, columnspacing=1.4)
        fig.tight_layout(rect=(0, 0, 0.97, 0.94))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(out.with_suffix('.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
