"""Selection bias of the single trait baseline (response letter): the training and the test
heritability of every measurement of one simulation instance, one panel per simulation, with the
measurements that single trait picks marked by their pick order.

Heritability is the ANOVA estimate of each measurement on the training and on the test plants. The
picks are those of h2opt.baselines.MaxHeritabilityFeatures on the training plants (the baseline of
the paper). Drawn at the text width of the paper (6.5 in), saved as .pdf and, beside it, .png; the
heritability of the picks is printed.

    python pipelines/scripts/simulation/plot_selection_bias.py \\
        --row Hyperspectral simulation/instance0/spectra.npz \\
        --row Images helios_simulation/instance0/images.npz --traits 5 --out selectionBias.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from h2opt import anova_heritability  # noqa: E402
from h2opt.baselines import MaxHeritabilityFeatures  # noqa: E402

from h2opt_analysis.figures import COLORS, MATH_RC, SIM_FIG_RC, letter  # noqa: E402

FIG_SIZE = (6.5, 3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', nargs=2, action='append', required=True,
                        metavar=('LABEL', 'MEASUREMENTS'),
                        help='panel title and .npz with X (plants first), names and test')
    parser.add_argument('--traits', type=int, required=True, help='number of picks to mark')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig, axes = plt.subplots(1, len(args.row), figsize=FIG_SIZE)
        for ax, (label, path), text in zip(axes, args.row, 'ab', strict=True):
            with np.load(path) as f:
                X, names, test = f['X'], f['names'], f['test']
            X = X.reshape((len(X), -1)).astype(float)
            h2_train = anova_heritability(X[~test], names[~test])
            h2_test = anova_heritability(X[test], names[test])
            picks = MaxHeritabilityFeatures(args.traits).fit(X[~test], names[~test]).features_
            r = np.corrcoef(h2_train, h2_test)[0, 1]
            print(f'{label}: {X.shape[1]} measurements, r(train, test) {r:.2f}; picks '
                  + ', '.join(f'{k + 1}: train {h2_train[j]:.3f} test {h2_test[j]:.3f}'
                              for k, j in enumerate(picks)))

            ax.scatter(h2_train, h2_test, s=2, color='0.6', alpha=0.3, linewidth=0,
                       rasterized=True, label='all measurements')
            ax.scatter(h2_train[picks], h2_test[picks], s=14, color=COLORS['max_features'],
                       edgecolor='black', linewidth=0.4, zorder=3, label='single trait picks')
            for k, j in enumerate(picks):
                ax.annotate(str(k + 1), (h2_train[j], h2_test[j]), xytext=(3, 2),
                            textcoords='offset points', fontsize=6, zorder=4)
            low = min(h2_train.min(), h2_test.min())
            high = max(h2_train.max(), h2_test.max())
            ax.plot([low, high], [low, high], color='0.3', linewidth=0.8, linestyle=':')
            ax.axhline(0, color='0.85', linewidth=0.6, zorder=0)
            ax.set_xlabel('training set heritability')
            ax.set_ylabel('test set heritability')
            ax.set_title(f'{label} ({X.shape[1]:,} measurements, r = {r:.2f})')
            ax.spines[['top', 'right']].set_visible(False)
            letter(ax, text, -0.18)
        axes[0].legend(loc='upper left', frameon=False, markerscale=1.5)
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix('.png'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
