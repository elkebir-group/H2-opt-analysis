"""The linear-vs-hybrid figure of the paper (simLinear), from the saved results.

One row per simulation (--row), each with three panels of boxplots over the instances: the test
heritability of the true traits and of synthetic traits 1-5; |r| of true trait t with synthetic
trait t; the AUC of traits 1-3. The figure shows the ground truth, the hybrid H2-opt
(h2opt_hybrid, labeled 'H2-opt (CNN)' in the color of H2-opt), the linear H2-opt
(h2opt, labeled 'H2-opt (linear)' in a light tint of that color) and PCH. The style is that of
simFig (plot_paper.py): a panel has the same y-axis in all rows, the letters go down the columns
and the row labels are at the right of the rows. The figure is also saved as a .png next to the
.pdf, and the medians over the instances go to stdout.

    python pipelines/scripts/simulation/plot_paper_linear.py \\
        --row Hyperspectral simulation/evaluation.npz simulation/roc.npz \\
        --row Images helios_simulation/evaluation.npz helios_simulation/roc.npz \\
        --out figures/paper/simLinear.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from plot_paper import (  # noqa: E402
    BOTTOM_MARGIN,
    SIM_FIG_RC,
    TOP_MARGIN,
    auc_values,
    boxplot_panel,
    correlation_values,
    heritability_values,
    letter,
    row_label,
    save,
    select_methods,
    share_y,
    top_legend,
)

from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402

SHOWN = ['h2opt_hybrid', 'h2opt', 'pch']
LABELS_LINEAR = {**LABELS, 'h2opt_hybrid': 'H2-opt (CNN)', 'h2opt': 'H2-opt (linear)'}
# The hybrid in the color of H2-opt in simFig (tab:blue), the linear H2-opt in a light tint of it.
COLORS_LINEAR = {**COLORS, 'h2opt_hybrid': COLORS['h2opt'], 'h2opt': '#9ecae1'}
FIG_SIZE = (6.5, 3.5)
# Columns: heritability (traits 1-5), |r| (traits 1-3), AUC (traits 1-3).
WIDTH_RATIOS = [4.6, 3, 3]
# x of the panel letters, in axes coordinates of each column.
LETTER_X = [-0.17, -0.26, -0.26]


def print_medians(label, panel, slugs, values):
    print(f'{label}, {panel} (median over the instances per trait)')
    for slug in slugs:
        medians = [f'{np.median(v):.3f}' for _, v in sorted(values[slug].items()) if len(v)]
        print(f'  {LABELS_LINEAR[slug]}: {" ".join(medians)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'EVALUATION', 'ROC'),
                        help='one row: label, evaluate.py result, roc.py result')
    parser.add_argument('--out', required=True, help='simLinear .pdf')
    args = parser.parse_args()
    roc_axes = {'auc': 1, 'tpr': 1, 'tpr_instances': 1}
    letters = 'abcdefghijklmnop'
    n_rows = len(args.row)

    with plt.rc_context(SIM_FIG_RC):
        fig, axes = plt.subplots(n_rows, 3, figsize=FIG_SIZE, squeeze=False,
                                 gridspec_kw={'width_ratios': WIDTH_RATIOS})
        for r, (label, evaluation, roc) in enumerate(args.row):
            # No H2-opt variant takes the slug 'h2opt': both keep their own slugs.
            ev = select_methods(np.load(evaluation), {'herit': 0, 'cor': 0}, SHOWN, None)
            roc = select_methods(np.load(roc), roc_axes, ['truth'] + SHOWN, None)
            slugs, values, n_traits = heritability_values(ev)
            print_medians(label, 'test heritability', slugs, values)
            boxplot_panel(axes[r, 0], n_traits, slugs, values, 'test heritability',
                          COLORS_LINEAR)
            axes[r, 0].axhline(0, color='0.3', linewidth=0.8, linestyle=':')
            cor_slugs, values = correlation_values(ev)
            print_medians(label, '|r| with true trait', cor_slugs, values)
            boxplot_panel(axes[r, 1], len(next(iter(values.values()))), cor_slugs, values,
                          '|r| with true trait', COLORS_LINEAR)
            auc_slugs, values = auc_values(roc)
            print_medians(label, 'AUC', auc_slugs, values)
            boxplot_panel(axes[r, 2], len(next(iter(values.values()))), auc_slugs, values,
                          'AUC of causal 100 kb bins', COLORS_LINEAR)
        for c in range(3):
            share_y(axes[:, c])
            for ax in axes[:, c]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            for r in range(n_rows):
                letter(axes[r, c], letters[n_rows * c + r], LETTER_X[c])
                if r < n_rows - 1:
                    axes[r, c].set_xlabel('')
        top = 1 - TOP_MARGIN / FIG_SIZE[1]
        fig.subplots_adjust(left=0.065, right=0.97, top=top, bottom=BOTTOM_MARGIN / FIG_SIZE[1],
                            wspace=0.3, hspace=0.27)
        top_legend(fig, slugs, top, LABELS_LINEAR, COLORS_LINEAR)
        for r, row in enumerate(args.row):
            row_label(fig, axes[r, -1], row[0])
        save(fig, args.out)


if __name__ == '__main__':
    main()
