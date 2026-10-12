"""The Miscanthus heritability and NDVI figures of the paper's supplement.

--out-train (trainHeritMisc): the narrow-sense train heritability of trait 1 per date, every
method. --out-ndvi-heritability (miscVegIndex): the test heritability per date, every method and
NDVI. --out-canopy (NDVI_canopy): the mean NDVI of the plants per date, both populations. The
heritability is 4 x Henderson3 (capped at 1), the mean over the folds. The panel function
plot_dates also makes panel b of miscFig (plot_paper_main.py). Each figure is also saved as a
.png next to its .pdf.

    python pipelines/scripts/miscanthus/plot_paper_heritability.py \\
        --heritability central/heritability.npz south/heritability.npz \\
        --ndvi central/ndvi.npz south/ndvi.npz --out-train figures/paper/trainHeritMisc.pdf \\
        --out-ndvi-heritability figures/paper/miscVegIndex.pdf \\
        --out-canopy figures/paper/NDVI_canopy.pdf
"""

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from h2opt_analysis import miscanthus  # noqa: E402
from h2opt_analysis.figures import COLORS  # noqa: E402
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

FONT_SIZE = 8
# The paper's H2-opt on Miscanthus is the hybrid; linear H2-opt is shown as such.
PAPER_LABELS = {**LABELS, 'h2opt_hybrid': 'H2-opt', 'h2opt': 'linear H2-opt'}
NDVI_COLOR = 'black'
# Markers tell apart curves that overlap (linear H2-opt and PCH give near-equal values).
MARKERS = {'h2opt_hybrid': 'o', 'h2opt': 'D', 'pch': 's', 'max_features': '^', 'pca': 'v',
           'genetic_pca': 'P', 'ndvi': 'X'}


def narrow(values):
    """Narrow-sense heritability of half-sib families: 4 x Henderson3, capped at 1."""
    return np.minimum(4 * np.asarray(values), 1)


def load_heritability(path, split):
    """{method: narrow-sense heritability of trait 1, dates x folds} of split 'train' or 'test',
    in figure order."""
    with np.load(path) as f:
        values = dict(zip(f['methods'], narrow(f[split][..., 0]), strict=True))
    return {m: values[m] for m in ordered(values)}


def plot_dates(ax, curves, colors, labels, dates=miscanthus.DATES):
    """One line with markers per curve: the mean over the folds of curves[key] (dates x folds).
    Linear H2-opt is dashed with hollow markers."""
    x = np.arange(len(dates))
    for key, values in curves.items():
        hollow = key == 'h2opt'
        ax.plot(x, values.mean(axis=1), '--' if hollow else '-', color=colors[key], alpha=0.6,
                lw=1)
        ax.plot(x, values.mean(axis=1), MARKERS[key], color=colors[key], ms=5 if hollow else 3,
                mfc='none' if hollow else colors[key], label=labels[key])
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xticks(x, dates, rotation=60)
    ax.spines[['top', 'right']].set_visible(False)


def save(fig, path):
    fig.savefig(path)
    fig.savefig(str(path).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


def population_panels(heritability, ylabel, ndvi=None):
    """A figure with one plot_dates panel per population; ndvi adds the NDVI test curves."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    colors, labels = {**COLORS, 'ndvi': NDVI_COLOR}, {**PAPER_LABELS, 'ndvi': 'NDVI'}
    for p, (ax, population) in enumerate(zip(axes, miscanthus.POPULATIONS, strict=True)):
        curves = dict(heritability[p])
        if ndvi is not None:
            curves['ndvi'] = ndvi[p]
        plot_dates(ax, curves, colors, labels)
        ax.set_title(population.capitalize(), fontsize=FONT_SIZE + 1)
        ax.set_xlabel('date')
    axes[0].set_ylabel(ylabel)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False,
               fontsize=FONT_SIZE, handletextpad=0.2, columnspacing=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--ndvi', nargs=2, required=True, metavar=('CENTRAL', 'SOUTH'))
    parser.add_argument('--out-train', required=True)
    parser.add_argument('--out-ndvi-heritability', required=True)
    parser.add_argument('--out-canopy', required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.size': FONT_SIZE})

    ndvi = [np.load(f) for f in args.ndvi]
    train = [load_heritability(f, 'train') for f in args.heritability]
    save(population_panels(train, 'train heritability'), args.out_train)
    test = [load_heritability(f, 'test') for f in args.heritability]
    save(population_panels(test, 'test heritability', [narrow(n['test']) for n in ndvi]),
         args.out_ndvi_heritability)

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    x = np.arange(len(miscanthus.DATES))
    for population, values, color in zip(miscanthus.POPULATIONS, ndvi, ('tab:blue', 'tab:orange'),
                                         strict=True):
        ax.plot(x, values['ndvi'].mean(axis=1), 'o-', color=color, ms=3, lw=1,
                label=population.capitalize())
    ax.set_xticks(x, miscanthus.DATES, rotation=60)
    ax.set_xlabel('date')
    ax.set_ylabel('mean NDVI')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, args.out_canopy)

    print('mean narrow-sense test heritability of NDVI per date')
    for population, values in zip(miscanthus.POPULATIONS, ndvi, strict=True):
        print(f'{population:8s}' + ' '.join(f'{v:.3f}' for v in narrow(values['test']).mean(1)))


if __name__ == '__main__':
    main()
