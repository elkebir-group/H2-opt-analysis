"""The Miscanthus figure of the paper (miscFig).

a: example input images (red, green and blue channels) of one plant on three dates, with the
number of kept plants n and of half-sib families c. b: the narrow-sense test heritability of
trait 1 per date after the nearest-neighbor field adjustment (4 x Henderson3, clipped to [0, 1]
per fold; the neighbor-adjusted run, evaluate_heritability_spatial.py): the median over the folds
and a band from the 25th to the 75th percentile, H2-opt with a CNN (the hybrid) and with linear
functions against the baselines; a dotted line before the date of canopy closure. c: the median
over the folds of |r| between half-sib family means of H2-opt traits of the first five dates
(date_correlations.py). d: the same with the measured traits of Central Japan, the 6 traits with
the largest value on these dates (measured_correlations.py). Also saved as a .png next to the
.pdf.

    python pipelines/scripts/miscanthus/plot_paper_main.py \\
        --images central/images.npy south/images.npy \\
        --heritability central/heritability.npz south/heritability.npz \\
        --dates central/date_correlations.npz south/date_correlations.npz \\
        --measured central/measured_correlations.npz --out figures/paper/miscFig.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from plot_paper_correlations import heatmap, median_abs, sort_traits  # noqa: E402
from plot_paper_heritability import MARKERS, PAPER_LABELS, narrow, save  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import COLORS, MATH_RC, SIM_FIG_RC, TICK_SIZE, letter  # noqa: E402
from h2opt_analysis.methods import ordered  # noqa: E402

N_DATES = 5  # panels c and d: the dates before canopy closure
CANOPY_CLOSURE = '07/24'
LABELS_B = {**PAPER_LABELS, 'h2opt_hybrid': 'H2-opt (CNN)', 'h2opt': 'H2-opt (linear)'}
MARKERS_B = {**MARKERS, 'h2opt': 'o'}
N_MEASURED = 6
EXAMPLE_PLANT = 2  # index among the kept plants
EXAMPLE_DATES = (0, 2, 4)
RGB = (3, 2, 1)
# Drawn at the text width of the paper (6.5 in), in the style of simFig.
FIG_SIZE = (6.5, 2.45)
HEATMAP_SIZE = 4.5
# Rows of panels in inches: the bottoms of the Central and the South row, and their height.
ROW_BOTTOM = (1.6, 0.4)
ROW_HEIGHT = 0.64


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
    size = 0.42 * height  # image height in figure coordinates
    aspect = fig.get_figheight() / fig.get_figwidth()
    for k, date in enumerate(EXAMPLE_DATES):
        ax = fig.add_axes((left + 0.52 * width + 0.2 * size * aspect * k,
                           bottom + 0.42 * height - 0.2 * size * k, size * aspect, size))
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


def load_adjusted(path):
    """{method: narrow-sense test heritability of trait 1, dates x folds} of a
    evaluate_heritability_spatial.py output, in figure order."""
    with np.load(path) as f:
        values = dict(zip(f['methods'], narrow(f['test']), strict=True))
    return {m: values[m] for m in ordered(values)}


def plot_bands(ax, curves, dates=miscanthus.DATES):
    """Per method, the median over the folds (line and markers) and a band from the 25th to the
    75th percentile; the first method of the figure order on top."""
    x = np.arange(len(dates))
    for i, (key, values) in enumerate(curves.items()):
        low, median, high = np.percentile(values, [25, 50, 75], axis=1)
        ax.fill_between(x, low, high, color=COLORS[key], alpha=0.2, linewidth=0, zorder=10 - i)
        ax.plot(x, median, '-', color=COLORS[key], alpha=0.6, lw=1, zorder=10 - i)
        ax.plot(x, median, MARKERS_B[key], color=COLORS[key], ms=3, label=LABELS_B[key],
                zorder=10 - i)
    ax.axvline(dates.index(CANOPY_CLOSURE) - 0.5, color='0.5', ls=':', lw=0.8)
    ax.set_xticks(x, dates, rotation=60)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--heritability', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--dates', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--measured', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dates = miscanthus.DATES[:N_DATES]

    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        fig = plt.figure(figsize=FIG_SIZE)

        def axes(x, row, width, rows=1):
            """Axes at x inches from the left, width inches wide, in row 0 (Central), row 1
            (South) or, with rows=2, spanning both."""
            bottom = ROW_BOTTOM[1] if rows == 2 else ROW_BOTTOM[row]
            height = ROW_HEIGHT * rows + (ROW_BOTTOM[0] - ROW_BOTTOM[1] - ROW_HEIGHT) * (rows - 1)
            return fig.add_axes((x / FIG_SIZE[0], bottom / FIG_SIZE[1], width / FIG_SIZE[0],
                                 height / FIG_SIZE[1]))

        for p, population in enumerate(miscanthus.POPULATIONS):
            box = (0, ROW_BOTTOM[p] / FIG_SIZE[1], 0.8 / FIG_SIZE[0], ROW_HEIGHT / FIG_SIZE[1])
            panel_a(fig, box, args.images[p], population)

            ax = axes(1.2, p, 1.65)
            plot_bands(ax, load_adjusted(args.heritability[p]))
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90,
                          fontsize=TICK_SIZE - 1)
            ax.set_title(population.capitalize())
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel('test heritability')
            if p == 0:
                letter(ax, 'a', -0.73)
                letter(ax, 'b', -0.23)
                ax.legend(loc='upper right', ncol=2, frameon=False, fontsize=TICK_SIZE - 1,
                          handletextpad=0.1, columnspacing=0.6, labelspacing=0.2,
                          borderaxespad=0.1, markerscale=0.8)

            ax = axes(3.27, p, 0.85)
            with np.load(args.dates[p]) as f:
                matrix = median_abs(f['family'])[:N_DATES, :N_DATES]
            image = heatmap(ax, matrix, dates, dates, font_size=HEATMAP_SIZE)
            ax.set_title(population.capitalize())
            fig.colorbar(image, cax=axes(4.18, p, 0.05), label='$|r|$', ticks=[0, 0.5, 1])
            if p == 0:
                letter(ax, 'c', -0.4)

        fig.text(0.85 / FIG_SIZE[0], (ROW_BOTTOM[1] - 0.12) / FIG_SIZE[1],
                 ', '.join(miscanthus.DATES[d] for d in EXAMPLE_DATES), ha='right', va='top',
                 fontsize=TICK_SIZE - 1)

        ax = axes(5.25, 0, 0.85, rows=2)
        with np.load(args.measured) as f:
            matrix, names = sort_traits(median_abs(f['family'])[:, :N_DATES], list(f['traits']))
        matrix, names = matrix[:N_MEASURED], names[:N_MEASURED]
        image = heatmap(ax, matrix, [name.replace(' ', '\n', 1) if len(name) > 14 else name
                                     for name in names], dates, vmax=matrix.max(),
                        font_size=HEATMAP_SIZE)
        ax.set_title('Central')
        letter(ax, 'd', -0.95)
        bar = fig.colorbar(image, cax=axes(6.16, 0, 0.05, rows=2))
        bar.locator = MaxNLocator(nbins=3)
        bar.update_ticks()
        for name, value in zip(names, matrix.max(axis=1), strict=True):
            print(f'measured trait {name}: largest median |r| of family means {value:.3f}')
        save(fig, args.out)

if __name__ == '__main__':
    main()
