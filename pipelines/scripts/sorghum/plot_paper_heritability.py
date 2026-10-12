"""Train and test heritability of traits 1 to --traits of each method: median over the folds and
a band from the 25th to the 75th percentile (paper figure multiTraitHerit_all).

Reads the .npz of evaluate_heritability.py; method labels and colors come from the slugs saved
with the values. Left panel: train genotypes; right panel: test genotypes. Writes the figure as
.pdf and, beside it, .png, and prints the median, 25th and 75th percentiles and mean per method.

    python pipelines/scripts/sorghum/plot_paper_heritability.py \\
        --heritability heritability_h2opt.npz --out figures/paper/multiTraitHerit_all.pdf \\
        --traits 10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import median_bands, ordered_legend  # noqa: E402
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()

    results = np.load(args.heritability)
    methods = list(results['methods'])
    order = np.arange(1, args.traits + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, split in zip(axes, ['train', 'test'], strict=True):
        values = results[split][:, :, :args.traits]  # methods x folds x traits
        median_bands(ax, order, values, methods)
        print(f'{split} heritability over the folds, traits 1-{args.traits} '
              '(median, 25th and 75th percentile, mean)')
        for method in ordered(methods):
            v = values[methods.index(method)]
            print(f'  {LABELS[method]:16s} median', np.round(np.median(v, axis=0), 3))
            print(f'  {"":16s} 25th  ', np.round(np.percentile(v, 25, axis=0), 3))
            print(f'  {"":16s} 75th  ', np.round(np.percentile(v, 75, axis=0), 3))
            print(f'  {"":16s} mean  ', np.round(v.mean(axis=0), 3))
        ax.set_xticks(order)
        ax.set_xlabel('synthetic trait')
        ax.set_title(f'{split} genotypes', fontsize=10)
        ax.grid(axis='y', color='0.9')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('heritability (median over folds)')
    axes[0].set_ylim(0, None)
    ordered_legend(axes[1], [LABELS[m] for m in ordered(methods)], frameon=False, fontsize=7)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)


if __name__ == '__main__':
    main()
