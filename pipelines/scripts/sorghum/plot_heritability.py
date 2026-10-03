"""Train and test heritability of each method's top trait over the genotype folds (boxplots).

Reads the .npz of evaluate_heritability.py; method labels and colors come from the slugs saved
with the values. Writes the figure as .pdf and, beside it, .png.

    python pipelines/scripts/sorghum/plot_heritability.py --heritability heritability_lbfgs.npz \\
        --out top_trait_heritability_lbfgs.pdf [--trait 1]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    parser.add_argument('--trait', type=int, default=1)
    args = parser.parse_args()

    results = np.load(args.heritability)
    methods = list(results['methods'])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, split in zip(axes, ['train', 'test'], strict=True):
        values = results[split][:, :, args.trait - 1]  # methods x folds
        boxes = ax.boxplot(values.T, patch_artist=True, widths=0.6,
                           medianprops={'color': 'black'})
        for box, method in zip(boxes['boxes'], methods, strict=True):
            box.set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.8)
        ax.set_xticks(range(1, len(methods) + 1), [LABELS[m] for m in methods], rotation=30,
                      ha='right')
        ax.set_title(f'{split} genotypes', fontsize=10)
        ax.grid(axis='y', color='0.9')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(f'heritability of trait {args.trait}')
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    for m, method in enumerate(methods):
        print(f'{LABELS[method]:16s} train {results["train"][m, :, args.trait - 1].mean():.3f}  '
              f'test {results["test"][m, :, args.trait - 1].mean():.3f}')


if __name__ == '__main__':
    main()
