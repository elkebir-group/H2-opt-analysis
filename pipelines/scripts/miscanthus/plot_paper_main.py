"""The Miscanthus figure of the paper (miscFig).

a: example input images (crop surface model) of one plant on three dates as a stack, with the
number of kept plants n and of half-sib families c. b: the narrow-sense test heritability of
trait 1 per date after the nearest-neighbor field adjustment (4 x Henderson3, clipped to [0, 1]
per fold; the neighbor-adjusted run, evaluate_heritability_spatial.py): the median over the folds
and a band from the 25th to the 75th percentile, H2-opt with a CNN (the hybrid) and with linear
functions against the baselines; a dotted line before the date of canopy closure. c: the median
over the folds of |r| between half-sib family means of H2-opt traits of the first five dates
(date_correlations.py). d: the same with the measured traits of each population, the 6 traits
with the largest value on these dates (measured_correlations.py), on one color scale. Also saved
as a .png next to the .pdf.

    python pipelines/scripts/miscanthus/plot_paper_main.py \\
        --images central/images.npy south/images.npy \\
        --heritability central/heritability.npz south/heritability.npz \\
        --dates central/date_correlations.npz south/date_correlations.npz \\
        --measured central/measured_correlations.npz south/measured_correlations.npz \\
        --out figures/paper/miscFig.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
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
CSM = 0  # image channel of the crop surface model
# Drawn at the text width of the paper (6.5 in), in the style of simFig.
FIG_SIZE = (6.5, 2.45)
HEATMAP_SIZE = 4.5
# Rows of panels in inches: the bottoms of the Central and the South row, and their height.
ROW_BOTTOM = (1.6, 0.4)
ROW_HEIGHT = 0.64
# The population of each row, at the right edge (as in the Arabidopsis figure).
ROW_LABEL_X = 6.33
ROW_LABEL_SIZE = 7.5


def csm_image(images, row, date):
    """The crop surface model (channel 0) of one plant and date, stretched to [0, 1] between its
    2% and 98% quantiles."""
    image = np.asarray(images[row, date][CSM], dtype=np.float64)
    low, high = np.quantile(image, [0.02, 0.98])
    return np.clip((image - low) / (high - low), 0, 1)


def panel_a(fig, box, images_path, population):
    """Three images of one plant (three dates, crop surface model) as a stack of cards in box
    (figure coordinates: left, bottom, width, height): the front card at the bottom left, the
    cards behind it up and to the right, dots for the other images, centered in box with its
    bottom on the bottom of box; n and c below the stack."""
    design = miscanthus.design(population)
    images = np.load(images_path, mmap_mode='r')
    row = np.flatnonzero(design.keep)[EXAMPLE_PLANT]
    left, bottom, width, height = box
    size = 0.72 * height  # card height in figure coordinates
    aspect = fig.get_figheight() / fig.get_figwidth()
    step = 0.13 * size
    stack_width = size * aspect + 2 * step * aspect
    card_left = left + (width - stack_width) / 2
    for k in reversed(range(len(EXAMPLE_DATES))):
        ax = fig.add_axes((card_left + step * aspect * k, bottom + step * k, size * aspect, size))
        ax.imshow(csm_image(images, row, EXAMPLE_DATES[k]), cmap='magma', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_edgecolor('0.6')
    right = card_left + size * aspect + 2 * step * aspect
    for x, y in ((card_left + step * aspect, bottom + size + step),
                 (right - step * aspect, bottom + step)):
        # three dots up and to the right: the other images
        fig.add_artist(Line2D([x + 0.45 * step * aspect * i for i in (0, 1, 2)],
                              [y + 0.45 * step * i + 0.2 * step for i in (0, 1, 2)],
                              transform=fig.transFigure, ls='', marker='o', ms=0.8, color='0.3'))
    n, c = int(design.keep.sum()), len(np.unique(design.families))
    n_text = f'{n:,}'.replace(',', '{,}')
    fig.text(left + width / 2, bottom - 0.06 * height,
             f'$n = {n_text}$\n$c = {c}$ half-siblings', ha='center', va='top',
             fontsize=TICK_SIZE)
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
    parser.add_argument('--measured', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
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
            box = (0, ROW_BOTTOM[p] / FIG_SIZE[1], 0.98 / FIG_SIZE[0], ROW_HEIGHT / FIG_SIZE[1])
            panel_a(fig, box, args.images[p], population)

            ax = axes(1.4, p, 1.42)
            plot_bands(ax, load_adjusted(args.heritability[p]))
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel('test heritability')
            if p == 0:
                letter(ax, 'a', -0.99)
                letter(ax, 'b', -0.23)
                ax.legend(loc='upper right', ncol=2, frameon=False, fontsize=TICK_SIZE - 1,
                          handletextpad=0.1, columnspacing=0.6, labelspacing=0.2,
                          borderaxespad=0.1, markerscale=0.8)

            ax = axes(3.14, p, 0.85)
            with np.load(args.dates[p]) as f:
                matrix = median_abs(f['family'])[:N_DATES, :N_DATES]
            image = heatmap(ax, matrix, dates, dates, font_size=HEATMAP_SIZE)
            bar = fig.colorbar(image, cax=axes(4.05, p, 0.05), ticks=[0, 0.5, 1])
            bar.ax.set_xlabel('$|r|$')
            if p == 0:
                letter(ax, 'c', -0.4)


        measured = []
        for path in args.measured:
            with np.load(path) as f:
                matrix, names = sort_traits(median_abs(f['family'])[:, :N_DATES],
                                            list(f['traits']))
            measured.append((matrix[:N_MEASURED], names[:N_MEASURED]))
        vmax = max(matrix.max() for matrix, _ in measured)
        for p, (population, (matrix, names)) in enumerate(zip(miscanthus.POPULATIONS, measured,
                                                              strict=True)):
            ax = axes(5.12, p, 0.8)
            image = heatmap(ax, matrix, names, dates, vmax=vmax, font_size=HEATMAP_SIZE)
            ax.tick_params(axis='y', labelsize=TICK_SIZE - 1.5)
            if p == 0:
                letter(ax, 'd', -0.85)
            bar = fig.colorbar(image, cax=axes(5.98, p, 0.05))
            bar.ax.set_xlabel('$|r|$')
            bar.locator = MaxNLocator(nbins=3)
            bar.update_ticks()
            fig.text(ROW_LABEL_X / FIG_SIZE[0], (ROW_BOTTOM[p] + ROW_HEIGHT / 2) / FIG_SIZE[1],
                     population.capitalize(), rotation=-90, ha='left', va='center',
                     fontsize=ROW_LABEL_SIZE, fontweight='bold')
            for name, value in zip(names, matrix.max(axis=1), strict=True):
                print(f'{population} measured trait {name}: largest median |r| of family means '
                      f'{value:.3f}')
        save(fig, args.out)

if __name__ == '__main__':
    main()
