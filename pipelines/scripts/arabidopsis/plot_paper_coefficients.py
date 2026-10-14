"""Coefficients of the H2-opt traits of each Arabidopsis treatment in one fold (images/arabiCoef).

One panel per trait (coefficients.py output, one file per treatment): the coefficients with
magnitude at least 2% of the trait's largest, by retention time and m/z, the marker area
proportional to the magnitude, MS1 and MS/MS in their own colors (arabidopsis.coefficient_map).
Two rows of five traits per treatment, one lettered block per treatment. The figure is drawn at the
text width of the paper (6.5 in), in the style of the other paper figures. Saves --out (.pdf) and
a .png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_coefficients.py --treatments 0d 6d \\
        --coefficients 0d/h2opt/coefficients_fold0.npz 6d/h2opt/coefficients_fold0.npz \\
        --out arabiCoef.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from h2opt_analysis import arabidopsis  # noqa: E402
from h2opt_analysis.figures import LETTER_SIZE, MATH_RC, SIM_FIG_RC  # noqa: E402

COLUMNS = 5
ROW_HEIGHT = 1.3  # inches
HEADER = 0.2  # inches above each treatment's rows, for its letter and name
WIDTH = 6.5  # inches
MZ_SHOWN = (100, 1300)  # m/z range of the panels (all bins below 100 are empty)
MARKER = 8  # marker area of the largest coefficient
RC = {**SIM_FIG_RC, **MATH_RC, 'axes.titlesize': 'medium'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatments', nargs='+', required=True)
    parser.add_argument('--coefficients', nargs='+', required=True, help='one per treatment')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()
    if len(args.treatments) != len(args.coefficients):
        parser.error('give one coefficients file per treatment')

    blocks = [np.load(path)['coefficients'] for path in args.coefficients]  # traits x level x ...
    rows = [-(-len(c) // COLUMNS) for c in blocks]
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(WIDTH, ROW_HEIGHT * sum(rows) + HEADER * len(blocks)),
                     layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.03, wspace=0.02)
    subfigures = fig.subfigures(len(blocks), 1, height_ratios=[r * ROW_HEIGHT + HEADER
                                                               for r in rows])
    for b, (subfigure, treatment, coefficients) in enumerate(
            zip(np.atleast_1d(subfigures), args.treatments, blocks, strict=True)):
        axes = subfigure.subplots(rows[b], COLUMNS, sharex=True, sharey=True, squeeze=False)
        for t, ax in enumerate(axes.flat):
            if t >= len(coefficients):
                ax.set_visible(False)
                continue
            arabidopsis.coefficient_map(ax, coefficients[t], MARKER)
            ax.set_title(f'trait {t + 1}')
            ax.spines[['top', 'right']].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel('$m/z$ (Da)')
        for ax in axes[-1]:
            ax.set_xlabel('retention time (min)')
        axes[0, 0].set_xlim(0, arabidopsis.N_RT * arabidopsis.RT_BIN)
        axes[0, 0].set_ylim(*MZ_SHOWN)
        axes[0, 0].set_xticks([0, 10, 20])
        axes[0, 0].set_yticks([100, 500, 1000])
        subfigure.suptitle(f'$\\bf{{{"ab"[b]}}}$  {treatment.removesuffix("d")} day',
                           x=0.0, ha='left', fontsize=LETTER_SIZE)
    handles = [Line2D([], [], marker='o', ls='', color=c, alpha=0.6)
               for _, c in arabidopsis.LEVEL_COLORS]
    fig.legend(handles, [name for name, _ in arabidopsis.LEVEL_COLORS], loc='upper right',
               frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=300)


if __name__ == '__main__':
    main()
