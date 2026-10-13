"""The Miscanthus figure of the paper (miscFig).

a: example input images (red, green and blue channels) of one plant on three dates, with the
number of kept plants n and of half-sib families c. b: the narrow-sense test heritability of
trait 1 per date, clipped at 0, H2-opt (the hybrid) against the baselines
(plot_paper_heritability.py). c: the median over the folds of |r| between half-sib family means
of H2-opt traits of the first five dates (date_correlations.py). d: the same with the measured
traits of Central Japan, the 6 traits with the largest value on these dates
(measured_correlations.py). e: the number of SNPs that are Bonferroni-significant in the GWAS of
at least one fold, per date, first five dates. Also saved as a .png next to the .pdf.

    python pipelines/scripts/miscanthus/plot_paper_main.py \\
        --images central/images.npy south/images.npy \\
        --heritability central/heritability.npz south/heritability.npz \\
        --dates central/date_correlations.npz south/date_correlations.npz \\
        --measured central/measured_correlations.npz \\
        --gwas central/gwas/date0.npz ... central/gwas/date4.npz \\
        south/gwas/date0.npz ... south/gwas/date4.npz --out figures/paper/miscFig.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_paper_correlations import heatmap, median_abs, sort_traits  # noqa: E402
from plot_paper_heritability import (  # noqa: E402
    FONT_SIZE,
    PAPER_LABELS,
    load_heritability,
    plot_dates,
    save,
)

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import COLORS  # noqa: E402

N_DATES = 5  # panels c-e: the dates before canopy closure
N_MEASURED = 6
EXAMPLE_PLANT = 2  # index among the kept plants
EXAMPLE_DATES = (0, 2, 4)
RGB = (3, 2, 1)


def union_counts(files):
    """Number of SNPs that are Bonferroni-significant in at least one fold, per file (one file
    per date, gwas.py result with one row per fold)."""
    counts = []
    for file in files:
        with np.load(file) as f:
            score = f['neg_log10_p']
        counts.append(int((score > -np.log10(0.05 / score.shape[1])).any(axis=0).sum()))
    return np.array(counts)


def rgb_image(images, row, date):
    """The red, green and blue channels of one plant and date, each stretched to [0, 1] between
    its 2% and 98% quantiles."""
    image = np.asarray(images[row, date][list(RGB)], dtype=np.float64).transpose(1, 2, 0)
    low, high = np.quantile(image, [0.02, 0.98], axis=(0, 1))
    return np.clip((image - low) / (high - low), 0, 1)


def panel_a(fig, cell, images_path, population):
    design = miscanthus.design(population)
    images = np.load(images_path, mmap_mode='r')
    row = np.flatnonzero(design.keep)[EXAMPLE_PLANT]
    left, bottom, width, height = cell.get_position(fig).bounds
    size = 0.55 * height
    for k, date in enumerate(EXAMPLE_DATES):
        ax = fig.add_axes((left + 0.3 * width + 0.1 * size * k,
                           bottom + 0.3 * height - 0.12 * size * k, size * 0.4, size))
        ax.imshow(rgb_image(images, row, date))
        ax.set_xticks([])
        ax.set_yticks([])
        if k == len(EXAMPLE_DATES) - 1:
            ax.set_xlabel(', '.join(miscanthus.DATES[d] for d in EXAMPLE_DATES), fontsize=5)
    n, c = int(design.keep.sum()), len(np.unique(design.families))
    fig.text(left, bottom + 0.55 * height, f'{population.capitalize()}\nn = {n:,}\nc = {c}',
             fontsize=FONT_SIZE - 1, va='center')
    return n, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--heritability', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--dates', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--measured', required=True)
    parser.add_argument('--gwas', nargs=2 * N_DATES, required=True,
                        help=f'the first {N_DATES} dates of Central, then of South')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.size': FONT_SIZE})

    fig = plt.figure(figsize=(16, 4.6))
    # Column 3 is empty: it holds the measured trait names of panel d.
    grid = fig.add_gridspec(2, 6, width_ratios=[1.0, 2.2, 1.5, 0.6, 1.5, 0.8], wspace=0.5,
                            hspace=0.75, left=0.02, right=0.99, top=0.88, bottom=0.14)
    letters = {}
    for p, population in enumerate(miscanthus.POPULATIONS):
        cell = fig.add_subplot(grid[p, 0])
        cell.axis('off')
        if p == 0:
            letters['a'] = cell
        panel_a(fig, cell, args.images[p], population)

        ax = fig.add_subplot(grid[p, 1])
        letters.setdefault('b', ax)
        plot_dates(ax, load_heritability(args.heritability[p], 'test'), COLORS, PAPER_LABELS)
        ax.set_title(population.capitalize(), fontsize=FONT_SIZE + 1)
        ax.set_ylabel('test heritability')
        ax.tick_params(axis='x', labelsize=FONT_SIZE - 1)

        ax = fig.add_subplot(grid[p, 2])
        letters.setdefault('c', ax)
        with np.load(args.dates[p]) as f:
            matrix = median_abs(f['family'])[:N_DATES, :N_DATES]
        dates = miscanthus.DATES[:N_DATES]
        image = heatmap(ax, matrix, dates, dates)
        fig.colorbar(image, ax=ax, fraction=0.06, pad=0.03)
        ax.set_title(population.capitalize(), fontsize=FONT_SIZE + 1)

        ax = fig.add_subplot(grid[p, 5])
        letters.setdefault('e', ax)
        counts = union_counts(args.gwas[p * N_DATES:(p + 1) * N_DATES])
        ax.bar(range(N_DATES), counts, color=COLORS['h2opt_hybrid'])
        ax.set_xticks(range(N_DATES), dates, rotation=90)
        ax.set_ylim(0, max(counts.max(), 1) * 1.15)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel('significant SNPs')
        ax.set_title(population.capitalize(), fontsize=FONT_SIZE + 1)
        ax.spines[['top', 'right']].set_visible(False)
        print(population, 'SNPs significant in at least one fold, dates 1-5:', counts.tolist())
    letters['b'].legend(*letters['b'].get_legend_handles_labels(), loc='lower center',
                        bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False,
                        fontsize=FONT_SIZE - 1, handletextpad=0.2, columnspacing=0.8)

    ax = fig.add_subplot(grid[:, 4])
    letters['d'] = ax
    with np.load(args.measured) as f:
        matrix, names = sort_traits(median_abs(f['family'])[:, :N_DATES], list(f['traits']))
    matrix, names = matrix[:N_MEASURED], names[:N_MEASURED]
    image = heatmap(ax, matrix, names, miscanthus.DATES[:N_DATES], vmax=matrix.max())
    fig.colorbar(image, ax=ax, fraction=0.06, pad=0.03)
    ax.set_title('Central', fontsize=FONT_SIZE + 1)
    for name, value in zip(names, matrix.max(axis=1), strict=True):
        print(f'measured trait {name}: largest median |r| of family means {value:.3f}')

    for letter, ax in letters.items():
        x = ax.get_position().x0 - {'a': 0, 'd': 0.1}.get(letter, 0.035)
        fig.text(x, 0.97, letter, fontsize=14, fontweight='bold', va='top')
    save(fig, args.out)


if __name__ == '__main__':
    main()
