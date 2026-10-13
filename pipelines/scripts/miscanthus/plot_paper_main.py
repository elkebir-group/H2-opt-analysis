"""The Miscanthus figure of the paper (miscFig).

a: example input images (red, green and blue channels) of one plant on three dates, with the
number of kept plants n and of half-sib families c. b: the narrow-sense test heritability of
trait 1 per date, clipped at 0, H2-opt (the hybrid) against the baselines
(plot_paper_heritability.py). c: the median over the folds of |r| between half-sib family means
of H2-opt traits of the first five dates (date_correlations.py). d: the same with the measured
traits of Central Japan, the 6 traits with the largest value on these dates
(measured_correlations.py). e: the number of SNPs that are Bonferroni-significant in the GWAS of
at least one fold, per date, first five dates (Central and South side by side). Also saved as a
.png next to the .pdf.

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
    PAPER_LABELS,
    load_heritability,
    plot_dates,
    save,
)

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import COLORS, MATH_RC, SIM_FIG_RC, TICK_SIZE, letter  # noqa: E402

N_DATES = 5  # panels c-e: the dates before canopy closure
N_MEASURED = 6
EXAMPLE_PLANT = 2  # index among the kept plants
EXAMPLE_DATES = (0, 2, 4)
RGB = (3, 2, 1)
# Drawn at the text width of the paper (6.5 in), in the style of simFig.
FIG_SIZE = (6.5, 4.2)
HEATMAP_SIZE = 5


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


def panel_a(fig, box, images_path, population):
    """Three images of one plant (three dates) as a stack in box (figure coordinates: left,
    bottom, width, height), with the population, n and c to their left."""
    design = miscanthus.design(population)
    images = np.load(images_path, mmap_mode='r')
    row = np.flatnonzero(design.keep)[EXAMPLE_PLANT]
    left, bottom, width, height = box
    size = 0.62 * height  # image height in figure coordinates
    aspect = fig.get_figheight() / fig.get_figwidth()
    for k, date in enumerate(EXAMPLE_DATES):
        ax = fig.add_axes((left + 0.45 * width + 0.16 * size * aspect * k,
                           bottom + 0.3 * height - 0.14 * size * k, size * aspect, size))
        ax.imshow(rgb_image(images, row, date), aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    n, c = int(design.keep.sum()), len(np.unique(design.families))
    n_text = f'{n:,}'.replace(',', '{,}')
    fig.text(left, bottom + 0.55 * height,
             f'{population.capitalize()}\n$n = {n_text}$\n$c = {c}$', va='center')
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
    dates = miscanthus.DATES[:N_DATES]

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig = plt.figure(figsize=FIG_SIZE)
        top = fig.add_gridspec(1, 3, width_ratios=[1.05, 2, 2], wspace=0.3, left=0.01,
                               right=0.99, top=0.85, bottom=0.58)
        grid_c = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.12, left=0.095,
                                  right=0.4, top=0.42, bottom=0.12)
        grid_d = fig.add_gridspec(1, 2, width_ratios=[1, 0.07], wspace=0.08, left=0.6,
                                  right=0.77, top=0.42, bottom=0.12)
        grid_e = fig.add_gridspec(1, 1, left=0.9, right=0.99, top=0.42, bottom=0.12)

        cell = fig.add_subplot(top[0])
        cell.axis('off')
        box = cell.get_position()
        for p, population in enumerate(miscanthus.POPULATIONS):
            panel_a(fig, (box.x0, box.y0 + (1 - p) * box.height / 2, box.width, box.height / 2),
                    args.images[p], population)
        letter(cell, 'a', 0)
        fig.text(box.x0 + box.width, box.y0 - 0.02,
                 ', '.join(miscanthus.DATES[d] for d in EXAMPLE_DATES), ha='right', va='top',
                 fontsize=TICK_SIZE - 1)

        for p, population in enumerate(miscanthus.POPULATIONS):
            ax = fig.add_subplot(top[1 + p])
            plot_dates(ax, load_heritability(args.heritability[p], 'test'), COLORS,
                       PAPER_LABELS)
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
            ax.set_title(population.capitalize())
            ax.set_ylim(0, 1)
            ax.set_ylabel('test heritability' if p == 0 else '')
            if p == 0:
                letter(ax, 'b', -0.22)
                ax.legend(loc='lower left', bbox_to_anchor=(0.3, 1.16), ncol=5, frameon=False,
                          handletextpad=0.2, columnspacing=1.2, borderaxespad=0)

        for p, population in enumerate(miscanthus.POPULATIONS):
            ax = fig.add_subplot(grid_c[p])
            with np.load(args.dates[p]) as f:
                matrix = median_abs(f['family'])[:N_DATES, :N_DATES]
            image = heatmap(ax, matrix, dates, dates, font_size=HEATMAP_SIZE)
            if p == 1:
                ax.set_yticks([])
            ax.set_title(population.capitalize())
            if p == 0:
                letter(ax, 'c', -0.55)
        fig.colorbar(image, cax=fig.add_subplot(grid_c[2]), label='$|r|$')

        ax = fig.add_subplot(grid_d[0])
        with np.load(args.measured) as f:
            matrix, names = sort_traits(median_abs(f['family'])[:, :N_DATES], list(f['traits']))
        matrix, names = matrix[:N_MEASURED], names[:N_MEASURED]
        image = heatmap(ax, matrix, names, dates, vmax=matrix.max(), font_size=HEATMAP_SIZE)
        ax.set_title('Central')
        letter(ax, 'd', -0.95)
        fig.colorbar(image, cax=fig.add_subplot(grid_d[1]), label='$|r|$')
        for name, value in zip(names, matrix.max(axis=1), strict=True):
            print(f'measured trait {name}: largest median |r| of family means {value:.3f}')

        ax = fig.add_subplot(grid_e[0])
        width = 0.4
        for p, population in enumerate(miscanthus.POPULATIONS):
            counts = union_counts(args.gwas[p * N_DATES:(p + 1) * N_DATES])
            ax.bar(np.arange(N_DATES) + (p - 0.5) * width, counts, width,
                   color=COLORS['h2opt_hybrid'], alpha=1 if p == 0 else 0.45,
                   label=population.capitalize())
            print(population, 'SNPs significant in at least one fold, dates 1-5:',
                  counts.tolist())
        ax.set_xticks(range(N_DATES), dates, rotation=90)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel('significant SNPs')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(frameon=False, loc='upper left', handlelength=1, borderaxespad=0)
        ax.spines[['top', 'right']].set_visible(False)
        letter(ax, 'e', -0.9)
        save(fig, args.out)

if __name__ == '__main__':
    main()
