"""The Miscanthus heritability and NDVI figures of the paper's supplement.

--out-train (trainHeritMisc): the narrow-sense train heritability of trait 1 per date after the
nearest-neighbor field adjustment, H2-opt (the hybrid) and the baselines
(--heritability-adjusted), the median over the folds and a band from the 25th to the 75th
percentile; H2-opt is drawn on top.
--out-ndvi-heritability (miscVegIndex): a, the mean NDVI of the plants per date, both populations;
b, the test heritability per date after the nearest-neighbor field adjustment of H2-opt with a CNN
(--heritability-adjusted, evaluate_heritability_spatial.py) and of NDVI (ndvi.py, test_adjusted),
the median over the folds and a band from the 25th to the 75th percentile; a dotted line before
the date of canopy closure; in the style of miscFig. The heritability is 4 x Henderson3, clipped
to [0, 1] per fold. Each figure is also saved as a .png next to its .pdf.

    python pipelines/scripts/miscanthus/plot_paper_heritability.py \\
        --heritability-adjusted adjusted/central/heritability.npz \\
        adjusted/south/heritability.npz --ndvi central/ndvi.npz south/ndvi.npz \\
        --out-train figures/paper/trainHeritMisc.pdf \\
        --out-ndvi-heritability figures/paper/miscVegIndex.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import COLORS, MATH_RC, SIM_FIG_RC, TICK_SIZE, letter  # noqa: E402
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

FONT_SIZE = 8
# The paper's H2-opt on Miscanthus is the hybrid; linear H2-opt is not shown.
PAPER_LABELS = {**LABELS, 'h2opt_hybrid': 'H2-opt'}
# The first date after canopy closure (from the mean NDVI, miscVegIndex a).
CANOPY_CLOSURE = '07/24'
NDVI_COLOR = 'black'
# Markers tell apart curves that overlap.
MARKERS = {'h2opt_hybrid': 'o', 'pch': 's', 'max_features': '^', 'single_band': 'D',
           'pca': 'v', 'genetic_pca': 'P', 'ndvi': 'X'}


def narrow(values):
    """Narrow-sense heritability of half-sib families: 4 x Henderson3, clipped to [0, 1]."""
    return np.clip(4 * np.asarray(values), 0, 1)


def load_heritability(path, split):
    """{method: narrow-sense heritability of trait 1, dates x folds} of split 'train' or 'test' of
    an evaluate_heritability_spatial.py result, in figure order, without linear H2-opt."""
    with np.load(path) as f:
        values = dict(zip(f['methods'], narrow(f[split]), strict=True))
    return {m: values[m] for m in ordered(values) if m != 'h2opt'}


def plot_dates(ax, curves, colors, labels, dates=miscanthus.DATES):
    """One line with markers per curve: the median over the folds of curves[key] (dates x
    folds), with a band from the 25th to the 75th percentile. The first curve is drawn on top."""
    x = np.arange(len(dates))
    for z, (key, values) in enumerate(curves.items()):
        low, median, high = np.percentile(values, [25, 50, 75], axis=1)
        zorder = len(curves) - z
        ax.fill_between(x, low, high, color=colors[key], alpha=0.15, linewidth=0, zorder=zorder)
        ax.plot(x, median, '-', color=colors[key], alpha=0.6, lw=1, zorder=zorder)
        ax.plot(x, median, MARKERS[key], color=colors[key], ms=3, label=labels[key],
                zorder=zorder)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xticks(x, dates, rotation=60)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


def population_panels(heritability, ylabel):
    """A figure with one plot_dates panel per population."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    for p, (ax, population) in enumerate(zip(axes, miscanthus.POPULATIONS, strict=True)):
        plot_dates(ax, heritability[p], COLORS, PAPER_LABELS)
        ax.set_title(population.capitalize(), fontsize=FONT_SIZE + 1)
        ax.set_xlabel('date')
    axes[0].set_ylabel(ylabel)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False,
               fontsize=FONT_SIZE, handletextpad=0.2, columnspacing=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def ndvi_figure(mean_ndvi, cnn, ndvi):
    """miscVegIndex. a: the mean NDVI of the plants per date (mean_ndvi: one array per
    population). b: per population, the narrow-sense test heritability (dates x folds) of H2-opt
    with a CNN and of NDVI: the median over the folds and a band from the 25th to the 75th
    percentile. A dotted line before the date of canopy closure in every panel. Draw and save it
    under SIM_FIG_RC and MATH_RC."""
    x = np.arange(len(miscanthus.DATES))
    closure = miscanthus.DATES.index(CANOPY_CLOSURE) - 0.5
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), layout='constrained')
    ax = axes[0]
    for population, values, color in zip(miscanthus.POPULATIONS, mean_ndvi,
                                         ('0.2', '0.6'), strict=True):
        ax.plot(x, values, '-', color=color, lw=1)
        ax.plot(x, values, 'o', color=color, ms=3, label=population.capitalize())
    ax.set_ylabel('mean NDVI')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', frameon=False, fontsize=TICK_SIZE - 1)
    letter(ax, 'a', -0.3)
    for ax, population, curves in zip(axes[1:], miscanthus.POPULATIONS,
                                      zip(cnn, ndvi, strict=True), strict=True):
        for values, color, marker, label in zip(
                curves, (COLORS['h2opt_hybrid'], NDVI_COLOR),
                (MARKERS['h2opt_hybrid'], MARKERS['ndvi']), ('H2-opt (CNN)', 'NDVI'),
                strict=True):
            low, median, high = np.percentile(values, [25, 50, 75], axis=1)
            ax.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
            ax.plot(x, median, '-', color=color, alpha=0.6, lw=1)
            ax.plot(x, median, marker, color=color, ms=3, label=label)
        ax.set_ylim(0, 1)
        ax.set_title(population.capitalize())
    axes[1].set_ylabel('test heritability')
    axes[2].legend(loc='upper right', frameon=False, fontsize=TICK_SIZE - 1)
    letter(axes[1], 'b', -0.3)
    for ax in axes:
        ax.axvline(closure, color='0.5', ls=':', lw=0.8)
        ax.set_xticks(x, miscanthus.DATES, rotation=90)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.spines[['top', 'right']].set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability-adjusted', nargs=2, required=True,
                        metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--ndvi', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--out-train', required=True)
    parser.add_argument('--out-ndvi-heritability', required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.size': FONT_SIZE})

    ndvi = [np.load(f) for f in args.ndvi]
    train = [load_heritability(f, 'train') for f in args.heritability_adjusted]
    save(population_panels(train, 'train heritability'), args.out_train)
    cnn = []
    for path in args.heritability_adjusted:
        with np.load(path) as f:
            cnn.append(narrow(f['test'][list(f['methods']).index('h2opt_hybrid')]))
    with plt.rc_context({**SIM_FIG_RC, **MATH_RC}):
        save(ndvi_figure([n['ndvi'].mean(axis=1) for n in ndvi], cnn,
                         [narrow(n['test_adjusted']) for n in ndvi]), args.out_ndvi_heritability)

    print('median narrow-sense test heritability of NDVI per date, after the adjustment')
    for population, values in zip(miscanthus.POPULATIONS, ndvi, strict=True):
        print(f'{population:8s}' + ' '.join(
            f'{v:.3f}' for v in np.median(narrow(values['test_adjusted']), axis=1)))


if __name__ == '__main__':
    main()
