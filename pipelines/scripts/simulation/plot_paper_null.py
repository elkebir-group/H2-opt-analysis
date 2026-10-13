"""The permutation-null figure of the paper (simNull), from the saved null and real results.

One row per simulation (--label, with its --slug, --evaluation, --gwas, --crossfit and --null in
the same order), each with four panels in the style of the null figures of the real datasets
(h2opt_analysis.figures.real_vs_null): per synthetic trait 1-3, a box in the color of H2-opt with
a point per instance for the real H2-opt traits of all instances, and a gray box with a point per
permutation for the null of the null instance; the cross-fitted GWAS of the null instance (trait 1
only) is a black diamond. a, the test heritability; b, the largest -log10 p (Bonferroni line
0.05 / SNPs); c, the QQ plot of the GWAS of trait 1 (each instance, the cross-fit and each
permutation, the diagonal and the Bonferroni line); d, the genomic-control lambda. The real test
heritability is the ANOVA heritability on the plants of fold 0 (evaluate.py herit[method,
instance, 1]); the null one is null_gwas.py h2_test. The style is that of simFig (plot_paper.py):
a panel has the same y-axis in all rows, the letters go down the columns and the row labels are at
the right of the rows. The figure is also saved as a .png next to the .pdf.

    python pipelines/scripts/simulation/plot_paper_null.py \\
        --label Hyperspectral --slug h2opt --evaluation simulation/evaluation.npz \\
        --gwas simulation/instance*/gwas/h2opt.npz \\
        --crossfit simulation/instance0/gwas/h2opt_crossfit.npz \\
        --null simulation/instance0/null/perm*.npz \\
        --label Images --slug h2opt_hybrid --evaluation helios_simulation/evaluation.npz \\
        --gwas helios_simulation/instance*/gwas/h2opt_hybrid.npz \\
        --crossfit helios_simulation/instance0/gwas/h2opt_hybrid_crossfit.npz \\
        --null helios_simulation/instance0/null/perm*.npz --out figures/paper/simNull.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from plot_paper import LEGEND_GAP, SIM_FIG_RC, letter, row_label, save, share_y  # noqa: E402

from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MARK_COLOR,
    MATH_RC,
    NULL_COLOR,
    NULL_LINE_COLOR,
    qq_real_vs_null,
    real_vs_null,
)

FIG_SIZE = (6.5, 3.4)
WIDTH_RATIOS = [1, 1, 1.15, 1]
# x of the panel letters, in axes coordinates of each column.
LETTER_X = [-0.3, -0.3, -0.27, -0.3]
TOP_MARGIN = 0.31
BOTTOM_MARGIN = 0.32
N_TRAITS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action='append', required=True, help='row label')
    parser.add_argument('--slug', action='append', required=True, help='real H2-opt method')
    parser.add_argument('--evaluation', action='append', required=True,
                        help='evaluate.py result')
    parser.add_argument('--gwas', nargs='+', action='append', required=True,
                        help='gwas.summarize result of the real traits of each instance')
    parser.add_argument('--crossfit', action='append', required=True,
                        help='gwas.summarize result of the cross-fitted trait 1, null instance')
    parser.add_argument('--null', nargs='+', action='append', required=True,
                        help='null_gwas.py results of the null instance')
    parser.add_argument('--out', required=True, help='simNull .pdf')
    args = parser.parse_args()
    rows = list(zip(args.label, args.slug, args.evaluation, args.gwas, args.crossfit, args.null,
                    strict=True))
    letters = 'abcdefghijklmnop'
    n_rows = len(rows)

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(n_rows, 4, figsize=FIG_SIZE, squeeze=False,
                                 gridspec_kw={'width_ratios': WIDTH_RATIOS})
        for r, (label, slug, evaluation, gwas_files, crossfit_file, null_files) in enumerate(rows):
            with np.load(evaluation) as f:
                methods = list(f['methods'])
                real_h2 = f['herit'][methods.index(slug), :, 1, :N_TRAITS]  # instances x traits
            reals = [np.load(p) for p in gwas_files]
            crossfit = np.load(crossfit_file)
            nulls = [np.load(p) for p in null_files]
            threshold = -np.log10(0.05 / reals[0]['neg_log10_p'].shape[1])
            real_max = np.array([g['neg_log10_p'][:N_TRAITS].max(axis=1) for g in reals])
            real_lambda = np.array([g['lambda_gc'][:N_TRAITS] for g in reals])
            null_h2 = np.array([n['h2_test'][:N_TRAITS] for n in nulls])
            null_max = np.array([n['neg_log10_p'][:N_TRAITS].max(axis=1) for n in nulls])
            null_lambda = np.array([n['lambda_gc'][:N_TRAITS] for n in nulls])
            mark_max = np.full(N_TRAITS, np.nan)
            mark_lambda = np.full(N_TRAITS, np.nan)
            mark_max[0] = crossfit['neg_log10_p'][0].max()
            mark_lambda[0] = crossfit['lambda_gc'][0]
            print(f'{label} ({slug}, {len(reals)} instances, {len(nulls)} permutations)')
            for t in range(N_TRAITS):
                print(f'  trait {t + 1}: test h2 median {np.median(real_h2[:, t]):.3f} (null '
                      f'{np.median(null_h2[:, t]):.3f}); largest -log10 p median '
                      f'{np.median(real_max[:, t]):.2f} (null max {null_max[:, t].max():.2f}, '
                      f'cross-fit {mark_max[t]:.2f}); lambda GC median '
                      f'{np.median(real_lambda[:, t]):.3f} (null '
                      f'{np.median(null_lambda[:, t]):.3f})')
            real_vs_null(axes[r, 0], real_h2, null_h2)
            axes[r, 0].axhline(0, color='0.3', linewidth=0.8, linestyle=':')
            axes[r, 0].set_ylabel('test heritability')
            real_vs_null(axes[r, 1], real_max, null_max, mark_max)
            axes[r, 1].axhline(threshold, color='black', linewidth=0.8, linestyle=':')
            axes[r, 1].set_ylabel(r'largest $-\log_{10} p$')
            qq_real_vs_null(axes[r, 2], [g['neg_log10_p'][0] for g in reals],
                            [n['neg_log10_p'][0] for n in nulls], crossfit['neg_log10_p'][0])
            axes[r, 2].set_ylabel(r'observed $-\log_{10} p$ (trait 1)')
            real_vs_null(axes[r, 3], real_lambda, null_lambda, mark_lambda)
            axes[r, 3].axhline(1, color='0.3', linewidth=0.8, linestyle=':')
            axes[r, 3].set_ylabel(r'$\lambda_\mathrm{GC}$')
        for c in range(4):
            share_y(axes[:, c])
            for ax in axes[:, c]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            for r in range(n_rows):
                letter(axes[r, c], letters[n_rows * c + r], LETTER_X[c])
                if r < n_rows - 1:
                    axes[r, c].set_xlabel('')
        top = 1 - TOP_MARGIN / FIG_SIZE[1]
        fig.subplots_adjust(left=0.065, right=0.97, top=top, bottom=BOTTOM_MARGIN / FIG_SIZE[1],
                            wspace=0.42, hspace=0.27)
        handles = [Line2D([], [], marker='s', color=COLORS['h2opt'], linestyle='-', markersize=4),
                   Line2D([], [], marker='D', color=MARK_COLOR, markeredgecolor='white',
                          linestyle='-', markersize=4),
                   Line2D([], [], marker='o', color=NULL_LINE_COLOR, markerfacecolor=NULL_COLOR,
                          markeredgewidth=0, linestyle='-', markersize=3)]
        labels = [f'H2-opt, {len(rows[0][3])} instances', 'H2-opt, cross-fit (null instance)',
                  'permutation null']
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, top + LEGEND_GAP / FIG_SIZE[1]), borderaxespad=0,
                   ncol=3, frameon=False, handlelength=2, handletextpad=0.5, columnspacing=1.4)
        for r, (label, *_) in enumerate(rows):
            row_label(fig, axes[r, -1], label)
        save(fig, args.out)


if __name__ == '__main__':
    main()
