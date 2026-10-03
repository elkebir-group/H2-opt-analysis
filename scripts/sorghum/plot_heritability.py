"""Train and test heritability of each method's top trait over the 10 genotype folds (boxplots).

Reads results/sorghum/heritability[_lbfgs].npz (evaluate_heritability.py); method labels come from the names
saved with the values. Writes results/sorghum/figures/top_trait_heritability[_lbfgs].pdf.

    uv run python scripts/sorghum/plot_heritability.py [--lbfgs] [--trait 1]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path('results/sorghum')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lbfgs', action='store_true', help='results of evaluate_heritability.py --lbfgs')
    parser.add_argument('--trait', type=int, default=1)
    args = parser.parse_args()

    suffix = '_lbfgs' if args.lbfgs else ''
    results = np.load(RESULTS / f'heritability{suffix}.npz')
    methods = list(results['methods'])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, split in zip(axes, ['train', 'test']):
        values = results[split][:, :, args.trait - 1]  # methods x folds
        boxes = ax.boxplot(values.T, patch_artist=True, widths=0.6, medianprops={'color': 'black'})
        for box, color in zip(boxes['boxes'], COLORS):
            box.set(facecolor=color, edgecolor=color, alpha=0.8)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=30, ha='right')
        ax.set_title(f'{split} genotypes', fontsize=10)
        ax.grid(axis='y', color='0.9')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(f'heritability of trait {args.trait}')
    fig.tight_layout()
    out = RESULTS / 'figures' / f'top_trait_heritability{suffix}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)
    for m, method in enumerate(methods):
        print(f'{method:13s} train {results["train"][m, :, args.trait - 1].mean():.3f}  '
              f'test {results["test"][m, :, args.trait - 1].mean():.3f}')


if __name__ == '__main__':
    main()
