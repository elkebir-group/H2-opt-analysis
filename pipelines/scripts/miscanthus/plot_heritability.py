"""Figure: narrow-sense test heritability of trait 1 per date, hybrid H2-opt against the baselines.

Left: the mean over folds of each method per date (4 x Henderson3, capped at 1, as in the paper),
with the folds as dots. Right: hybrid H2-opt minus PCH per date and fold. Saves a .pdf and a .png
next to it.

    python pipelines/scripts/miscanthus/plot_heritability.py --heritability heritability.npz \\
        --out heritability.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from h2opt_analysis.figures import COLORS
from h2opt_analysis.methods import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with np.load(args.heritability) as f:
        methods, dates, test = list(f['methods']), list(f['dates']), f['test'][..., 0]
    narrow = np.minimum(4 * test, 1)  # methods x dates x folds
    x = np.arange(len(dates))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [3, 2]})
    width = 0.8 / len(methods)
    for m, method in enumerate(methods):
        offset = (m - (len(methods) - 1) / 2) * width
        left.bar(x + offset, narrow[m].mean(axis=1), width, color=COLORS[method],
                 label=LABELS[method])
        for f in range(narrow.shape[2]):
            left.plot(x + offset, narrow[m, :, f], '.', color='0.2', ms=2)
    left.axhline(0, color='0.5', lw=0.5)
    left.set_xticks(x, dates, rotation=60, fontsize=8)
    left.set_ylabel('test heritability (narrow-sense)')
    left.legend(frameon=False, fontsize=7, ncol=2)
    difference = narrow[methods.index('h2opt_hybrid')] - narrow[methods.index('pch')]
    right.axhline(0, color='0.5', lw=0.5)
    for f in range(difference.shape[1]):
        right.plot(x, difference[:, f], '.', color='0.6', ms=4)
    right.plot(x, difference.mean(axis=1), 'o-', color=COLORS['h2opt_hybrid'], ms=4, label='mean')
    right.set_xticks(x, dates, rotation=60, fontsize=8)
    right.set_ylabel('H2-opt hybrid - PCH, per fold')
    right.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(str(args.out).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
