"""The permutation-null figure of the paper (simNull), from the saved null and real results.

One row per simulation (--row), each with three panels: the test heritability of synthetic traits
1-3 of the real H2-opt traits of the null instance (a colored marker per trait) and of each null
permutation (gray points over a gray box); the QQ plot of the GWAS of trait 1 (real in the color
of H2-opt, each permutation in light gray, the diagonal and the Bonferroni line 0.05 / SNPs); the
genomic-control lambda of traits 1-3, real and null. Both test heritabilities are the ANOVA
heritability on the plants of fold 0 (evaluate.py herit[method, instance, 1]; null_gwas.py
h2_test). The style is that of simFig (plot_paper.py): a panel has the same y-axis in all rows,
the letters go down the columns and the row labels are at the right of the rows. The figure is
also saved as a .png next to the .pdf.

    python pipelines/scripts/simulation/plot_paper_null.py \\
        --row Hyperspectral h2opt simulation/evaluation.npz \\
            simulation/instance0/gwas/h2opt.npz simulation/instance0/null/perm*.npz \\
        --row Images h2opt_hybrid helios_simulation/evaluation.npz \\
            helios_simulation/instance0/gwas/h2opt_hybrid.npz \\
            helios_simulation/instance0/null/perm*.npz \\
        --instance 0 --out figures/paper/simNull.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from plot_paper import LEGEND_GAP, SIM_FIG_RC, letter, row_label, save, share_y  # noqa: E402

from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402

FIG_SIZE = (6.5, 3.4)
WIDTH_RATIOS = [1, 1.15, 1]
# x of the panel letters, in axes coordinates of each column.
LETTER_X = [-0.25, -0.22, -0.25]
TOP_MARGIN = 0.31
BOTTOM_MARGIN = 0.32
N_TRAITS = 3
REAL_COLOR = COLORS['h2opt']
NULL_COLOR = '0.55'
NULL_LINE_COLOR = '0.78'


def real_vs_null(ax, real, null, ylabel):
    """Per trait: the null values (rows of null) as a light box with gray points, the real value as
    a marker in the color of H2-opt."""
    rng = np.random.default_rng(0)
    for t in range(N_TRAITS):
        ax.boxplot([null[:, t]], positions=[t], widths=0.5, showfliers=False,
                   patch_artist=True, boxprops={'facecolor': '0.92', 'edgecolor': NULL_COLOR,
                                                'linewidth': 0.5},
                   medianprops={'color': NULL_COLOR, 'linewidth': 0.8},
                   whiskerprops={'color': NULL_COLOR, 'linewidth': 0.6},
                   capprops={'color': NULL_COLOR, 'linewidth': 0.6})
        ax.scatter(t + rng.uniform(-0.12, 0.12, len(null)), null[:, t], s=5, color=NULL_COLOR,
                   linewidth=0, zorder=3)
        ax.scatter([t], [real[t]], s=22, marker='D', color=REAL_COLOR, edgecolor='white',
                   linewidth=0.4, zorder=4)
    ax.set_xticks(range(N_TRAITS), [str(t + 1) for t in range(N_TRAITS)])
    ax.set_xlim(-0.5, N_TRAITS - 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('trait')
    ax.set_ylabel(ylabel)


def qq_curve(ax, score, color, linewidth, zorder):
    observed = np.sort(score)
    expected = -np.log10(np.arange(len(observed), 0, -1) / (len(observed) + 1))
    ax.plot(expected, observed, color=color, linewidth=linewidth, zorder=zorder, rasterized=True)
    return expected.max()


def qq_panel(ax, real, nulls):
    """QQ plot of trait 1: each null in light gray, the real GWAS in the color of H2-opt."""
    for score in nulls:
        qq_curve(ax, score, NULL_LINE_COLOR, 0.8, 2)
    top = qq_curve(ax, real, REAL_COLOR, 1.0, 3)
    ax.plot([0, top], [0, top], color='black', linewidth=0.6, linestyle='--', zorder=1)
    ax.axhline(-np.log10(0.05 / len(real)), color='black', linestyle=':', linewidth=0.8)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(r'expected $-\log_{10} p$')
    ax.set_ylabel(r'observed $-\log_{10} p$ (trait 1)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', nargs='+', action='append', required=True,
                        metavar='LABEL_SLUG_OR_FILE',
                        help='one row: label, slug of the real H2-opt, evaluate.py result, '
                             'gwas.summarize result of the real traits, null_gwas.py results')
    parser.add_argument('--instance', type=int, required=True,
                        help='instance of the null (index into evaluate.py herit)')
    parser.add_argument('--out', required=True, help='simNull .pdf')
    args = parser.parse_args()
    letters = 'abcdefghijklmnop'
    n_rows = len(args.row)

    # Math labels in the text font of the figure.
    math_rc = {'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
               'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'}
    with plt.rc_context({**SIM_FIG_RC, **math_rc}):
        fig, axes = plt.subplots(n_rows, 3, figsize=FIG_SIZE, squeeze=False,
                                 gridspec_kw={'width_ratios': WIDTH_RATIOS})
        for r, (label, slug, evaluation, real_gwas, *null_files) in enumerate(args.row):
            with np.load(evaluation) as f:
                methods = list(f['methods'])
                real_h2 = f['herit'][methods.index(slug), args.instance, 1, :N_TRAITS]
            with np.load(real_gwas) as f:
                real_score = f['neg_log10_p'][0]
                real_lambda = f['lambda_gc'][:N_TRAITS]
            nulls = [np.load(p) for p in null_files]
            null_h2 = np.array([n['h2_test'][:N_TRAITS] for n in nulls])
            null_lambda = np.array([n['lambda_gc'][:N_TRAITS] for n in nulls])
            print(f'{label} ({slug}, instance {args.instance}, {len(nulls)} permutations)')
            for t in range(N_TRAITS):
                print(f'  trait {t + 1}: test h2 real {real_h2[t]:.3f} null median '
                      f'{np.median(null_h2[:, t]):.3f} [{null_h2[:, t].min():.3f}, '
                      f'{null_h2[:, t].max():.3f}]; lambda GC real {real_lambda[t]:.3f} null '
                      f'median {np.median(null_lambda[:, t]):.3f} '
                      f'[{null_lambda[:, t].min():.3f}, {null_lambda[:, t].max():.3f}]')
            real_vs_null(axes[r, 0], real_h2, null_h2, 'test heritability')
            axes[r, 0].axhline(0, color='0.3', linewidth=0.8, linestyle=':')
            qq_panel(axes[r, 1], real_score, [n['neg_log10_p'][0] for n in nulls])
            real_vs_null(axes[r, 2], real_lambda, null_lambda, r'$\lambda_\mathrm{GC}$')
            axes[r, 2].axhline(1, color='0.3', linewidth=0.8, linestyle=':')
        for c in range(3):
            share_y(axes[:, c])
            for ax in axes[:, c]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            for r in range(n_rows):
                letter(axes[r, c], letters[n_rows * c + r], LETTER_X[c])
                if r < n_rows - 1:
                    axes[r, c].set_xlabel('')
        top = 1 - TOP_MARGIN / FIG_SIZE[1]
        fig.subplots_adjust(left=0.065, right=0.97, top=top, bottom=BOTTOM_MARGIN / FIG_SIZE[1],
                            wspace=0.33, hspace=0.27)
        handles = [Line2D([], [], marker='D', color=REAL_COLOR, markeredgecolor='white',
                          linestyle='-', markersize=4),
                   Line2D([], [], marker='o', color=NULL_LINE_COLOR, markerfacecolor=NULL_COLOR,
                          markeredgewidth=0, linestyle='-', markersize=3)]
        fig.legend(handles, [LABELS['h2opt'], 'permutation null'], loc='lower center',
                   bbox_to_anchor=(0.5, top + LEGEND_GAP / FIG_SIZE[1]), borderaxespad=0,
                   ncol=2, frameon=False, handlelength=2, handletextpad=0.5, columnspacing=1.4)
        for r, row in enumerate(args.row):
            row_label(fig, axes[r, -1], row[0])
        save(fig, args.out)


if __name__ == '__main__':
    main()
