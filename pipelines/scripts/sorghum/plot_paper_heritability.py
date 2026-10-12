"""Train and test heritability of traits 1 to --traits of each method, mean over the folds
(paper figure multiTraitHerit_all).

Reads the .npz of evaluate_heritability.py; method labels and colors come from the slugs saved
with the values. Left panel: train genotypes; right panel: test genotypes. Writes the figure as
.pdf and, beside it, .png, and prints the means.

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

from h2opt_analysis.figures import COLORS  # noqa: E402
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
        mean = results[split][:, :, :args.traits].mean(axis=1)  # methods x traits
        print(f'mean {split} heritability, traits 1-{args.traits}')
        for method in ordered(methods):
            m = methods.index(method)
            on_top = method == 'h2opt'  # PCH is close to H2-opt; draw H2-opt over it
            ax.plot(order, mean[m], 'o-', color=COLORS[method], ms=4 if on_top else 3,
                    lw=2 if on_top else 1, zorder=3 if on_top else 2, label=LABELS[method])
            print(f'  {LABELS[method]:16s}', np.round(mean[m], 3))
        ax.set_xticks(order)
        ax.set_xlabel('synthetic trait')
        ax.set_title(f'{split} genotypes', fontsize=10)
        ax.grid(axis='y', color='0.9')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('heritability (mean over folds)')
    axes[0].set_ylim(0, None)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=150)


if __name__ == '__main__':
    main()
