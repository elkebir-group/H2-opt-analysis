"""The Miscanthus correlation figures of the paper's supplement.

Every value is the median over the folds of |Pearson r| (the sign of a synthetic trait is
arbitrary) of trait 1 of hybrid H2-opt (date_correlations.py, measured_correlations.py).
--out-dates-plant (miscCorTimeSimple) and --out-dates-family (miscCorTime): between the dates,
per plant and between half-sib family means, both populations. --out-measured-family
(traitCorrelation_allTraits) and --out-measured-plant (traitCorrelation_simplePearson): with the
measured traits of Central Japan, between family means and per plant; the traits are sorted by
their largest value. The panel function heatmap also makes panels c and d of miscFig
(plot_paper_main.py). Drawn at the text width of the paper (6.5 in) in the style of simFig. Each
figure is also saved as a .png next to its .pdf.

    python pipelines/scripts/miscanthus/plot_paper_correlations.py \\
        --dates central/date_correlations.npz south/date_correlations.npz \\
        --measured central/measured_correlations.npz \\
        --out-dates-plant figures/paper/miscCorTimeSimple.pdf \\
        --out-dates-family figures/paper/miscCorTime.pdf \\
        --out-measured-family figures/paper/traitCorrelation_allTraits.pdf \\
        --out-measured-plant figures/paper/traitCorrelation_simplePearson.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_paper_heritability import FONT_SIZE, save  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import MATH_RC, SIM_FIG_RC  # noqa: E402

WIDTH = 6.5  # inches
CELL_SIZE = 4  # points, the values in the cells
COLORBAR_LABEL = r'$|r|$, median over the folds'


def median_abs(values):
    """Median over the folds (last axis) of |r|."""
    return np.median(np.abs(values), axis=-1)


def heatmap(ax, matrix, rows, columns, vmax=1, font_size=FONT_SIZE - 2):
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


def date_figure(files, key, path):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), layout='constrained')
    for ax, population, file in zip(axes, miscanthus.POPULATIONS, files, strict=True):
        with np.load(file) as f:
            image = heatmap(ax, median_abs(f[key]), miscanthus.DATES, miscanthus.DATES,
                            font_size=CELL_SIZE)
        ax.set_title(population.capitalize())
        ax.set_xlabel('date')
        ax.set_ylabel('date')
    fig.colorbar(image, ax=axes, shrink=0.8, label=COLORBAR_LABEL)
    save(fig, path)


def measured_figure(file, key, path):
    with np.load(file) as f:
        matrix, names = sort_traits(median_abs(f[key]), list(f['traits']))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4), layout='constrained')
    image = heatmap(ax, matrix, names, miscanthus.DATES, vmax=max(matrix.max(), 0.05),
                    font_size=CELL_SIZE)
    ax.set_xlabel('date')
    ax.set_title('Central')
    fig.colorbar(image, ax=ax, label=COLORBAR_LABEL)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--measured', required=True)
    parser.add_argument('--out-dates-plant', required=True)
    parser.add_argument('--out-dates-family', required=True)
    parser.add_argument('--out-measured-family', required=True)
    parser.add_argument('--out-measured-plant', required=True)
    args = parser.parse_args()
    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        date_figure(args.dates, 'plant', args.out_dates_plant)
        date_figure(args.dates, 'family', args.out_dates_family)
        measured_figure(args.measured, 'family', args.out_measured_family)
        measured_figure(args.measured, 'plant', args.out_measured_plant)


if __name__ == '__main__':
    main()
