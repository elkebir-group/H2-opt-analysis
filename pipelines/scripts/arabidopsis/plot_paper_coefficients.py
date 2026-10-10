"""Coefficients of the H2-opt traits of one Arabidopsis treatment (images/coef_0day, coef_6day).

One panel per trait: the coefficients (coefficients.py output) with magnitude at least 2% of the
trait's largest, by retention time and m/z, the marker area proportional to the magnitude. MS1
and MS/MS features have their own color (the paper showed MS1 only). Saves --out (.pdf) and a
.png next to it.

    python pipelines/scripts/arabidopsis/plot_paper_coefficients.py \\
        --coefficients 0d/h2opt/coefficients_fold0.npz --title '0 day, fold 0' \\
        --out coef_0day.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Minutes per retention-time bin.
RT_BIN = 0.5
# Smallest coefficient magnitude shown, as a fraction of the trait's largest.
SHOWN = 0.02
LEVELS = (('MS1', 'tab:blue'), ('MS/MS', 'tab:orange'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficients', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    coefficients = np.load(args.coefficients)['coefficients']  # traits x level x m/z x RT
    n_traits = len(coefficients)
    columns = 5
    rows = -(-n_traits // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 2.6 * rows), sharex=True, sharey=True,
                             squeeze=False)
    for t, ax in enumerate(axes.flat):
        if t >= n_traits:
            ax.set_visible(False)
            continue
        for level, (name, color) in enumerate(LEVELS):
            magnitude = np.abs(coefficients[t, level])
            mz, rt = np.nonzero(magnitude >= SHOWN)
            ax.scatter((rt + 0.5) * RT_BIN, mz, s=30 * magnitude[mz, rt], color=color,
                       alpha=0.5, linewidths=0, label=name)
        ax.set_title(f'trait {t + 1}', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        if t % columns == 0:
            ax.set_ylabel('m/z (Da)')
        if t >= n_traits - columns:
            ax.set_xlabel('retention time (min)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, alpha=0.6) for _, c in LEVELS]
    fig.legend(handles, [name for name, _ in LEVELS], loc='upper right', frameon=False,
               fontsize=8, ncol=2)
    fig.suptitle(f'{args.title}: coefficients with magnitude >= {SHOWN:.0%} of the largest',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out)
    fig.savefig(args.out.removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
