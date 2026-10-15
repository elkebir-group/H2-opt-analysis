"""The Miscanthus correlation figures of the supplement.

Every value is the median over the folds of |Pearson r| (the sign of a synthetic trait is
arbitrary) of trait 1 of hybrid H2-opt (date_correlations.py, measured_correlations.py).
--out-dates (miscCorTime): between the dates, both populations (columns); per plant (a, b) and
between half-sib family means (c, d), with one color bar. --out-measured-family
(traitCorrelation_allTraits) and --out-measured-plant (traitCorrelation_simplePearson): with the
measured traits of each population (a: Central, b: South), between family means and per plant;
the traits are sorted by their largest value. The panel function heatmap also makes panels c and
d of miscFig (plot_paper_main.py). Drawn at the text width of the paper (6.5 in) in the style of
simFig. Each figure is also saved as a .png next to its .pdf.

    python pipelines/scripts/miscanthus/plot_paper_correlations.py \\
        --dates central/date_correlations.npz south/date_correlations.npz \\
        --measured central/measured_correlations.npz south/measured_correlations.npz \\
        --out-dates figures/paper/miscCorTime.pdf \\
        --out-measured-family figures/paper/traitCorrelation_allTraits.pdf \\
        --out-measured-plant figures/paper/traitCorrelation_simplePearson.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_paper_heritability import save  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import MATH_RC, SIM_FIG_RC, letter  # noqa: E402

WIDTH = 6.5  # inches
CELL_SIZE = 4  # points, the values in the cells
HEATMAP_FONT_SIZE = 6  # points, the default size of the heatmap labels
COLORBAR_LABEL = r'$|r|$, median over the folds'
# miscCorTime: the rows (key of date_correlations.py, title) and x of the panel letters in axes
# coordinates.
DATE_ROWS = [('plant', 'individual plants'), ('family', 'half-sibling group means')]
DATE_LETTER_X = -0.2
MEASURED_LETTER_X = -0.27


def median_abs(values):
    """Median over the folds (last axis) of |r|."""
    return np.median(np.abs(values), axis=-1)


def heatmap(ax, matrix, rows, columns, vmax=1, font_size=HEATMAP_FONT_SIZE):
    """Annotated heatmap of matrix (rows x columns) from 0 to vmax; returns the image."""
    image = ax.imshow(matrix, cmap='magma', vmin=0, vmax=vmax, aspect='auto')
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, f'{value:.2f}', ha='center', va='center', fontsize=font_size,
                color='black' if value > 0.6 * vmax else 'white')
    ax.set_xticks(range(len(columns)), columns, rotation=90)
    ax.set_yticks(range(len(rows)), rows)
    return image


def sort_traits(matrix, names):
    """Rows of matrix (traits x dates) and names, largest maximum first."""
    order = np.argsort(-matrix.max(axis=1))
    return matrix[order], [names[t] for t in order]


def date_figure(files, path):
    fig, axes = plt.subplots(len(DATE_ROWS), 2, figsize=(WIDTH, 6.6), layout='constrained')
    for r, (key, title) in enumerate(DATE_ROWS):
        for c, (population, file) in enumerate(zip(miscanthus.POPULATIONS, files, strict=True)):
            ax = axes[r, c]
            with np.load(file) as f:
                image = heatmap(ax, median_abs(f[key]), miscanthus.DATES, miscanthus.DATES,
                                font_size=CELL_SIZE)
            ax.set_title(f'{population.capitalize()}, {title}')
            ax.set_xlabel('date')
            ax.set_ylabel('date')
            letter(ax, 'abcd'[2 * r + c], DATE_LETTER_X)
    fig.colorbar(image, ax=axes, shrink=0.5, label=COLORBAR_LABEL)
    save(fig, path)


def measured_figure(files, key, path):
    """One heatmap per population, its height proportional to its number of traits, all on the
    color scale of the largest value."""
    panels = []
    for file in files:
        with np.load(file) as f:
            panels.append(sort_traits(median_abs(f[key]), list(f['traits'])))
    vmax = max(max(matrix.max() for matrix, _ in panels), 0.05)
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, 5.4), layout='constrained',
                             gridspec_kw={'height_ratios': [len(n) + 3 for _, n in panels]})
    for ax, population, (matrix, names), text in zip(axes, miscanthus.POPULATIONS, panels, 'ab',
                                                     strict=True):
        image = heatmap(ax, matrix, names, miscanthus.DATES, vmax=vmax, font_size=CELL_SIZE)
        ax.set_xlabel('date')
        ax.set_title(population.capitalize())
        letter(ax, text, MEASURED_LETTER_X)
    fig.colorbar(image, ax=axes, shrink=0.6, label=COLORBAR_LABEL)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--measured', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--out-dates', required=True)
    parser.add_argument('--out-measured-family', required=True)
    parser.add_argument('--out-measured-plant', required=True)
    args = parser.parse_args()
    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        date_figure(args.dates, args.out_dates)
        measured_figure(args.measured, 'family', args.out_measured_family)
        measured_figure(args.measured, 'plant', args.out_measured_plant)


if __name__ == '__main__':
    main()
