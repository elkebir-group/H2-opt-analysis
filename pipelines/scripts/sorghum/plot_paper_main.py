"""Main sorghum figure of the paper (sorFig): input, heritability, coefficients, measured traits.

Panels: a, the input as a stack of cards, one spectrum per plant (the front card shows the
spectrum of the first plant), with the numbers of plants and of genotypes; b, test heritability
of every trait of each method of --heritability (evaluate_heritability.py): the median over the
folds and a band from the 25th to the 75th percentile, and a dotted line at the median over the
folds of the test heritability of NDVI (--ndvi, ndvi.py); c, wavelength coefficients of traits 1
to --traits of linear H2-opt in one fold (--coefficients, pch_coefficients.py), for the traits as
saved (uncorrelated on the fold's training plants; the script checks them against
--fold-traits); d, median over the folds of
|Pearson r| between the genotype means (over all plants) of the H2-opt traits (--traits-files,
one per fold) and the measured genotype means of --measured-traits (Ferguson et al. 2021,
Supplemental Table S12). Writes the figure as .pdf and, beside it, .png, and prints the numbers
of each panel.

    python pipelines/scripts/sorghum/plot_paper_main.py --heritability heritability_h2opt.npz \\
        --coefficients pch_coefficients/fold0.npz --fold-traits h2opt/fold0.npz \\
        --traits-files h2opt/fold0.npz ... h2opt/fold9.npz \\
        --measured-traits measured_traits.tsv --ndvi ndvi.npz \\
        --out figures/paper/sorFig.pdf --traits 4
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from h2opt_analysis import sorghum  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MATH_RC,
    SIM_FIG_RC,
    TICK_SIZE,
    letter,
    median_bands,
    ordered_legend,
)
from h2opt_analysis.methods import LABELS, ordered  # noqa: E402

# The figure is drawn at the text width of the paper (6.5 in), so that LaTeX does not scale it,
# in the style of the simulation figures. Positions are in inches: the bottom and the top of the
# panels, and the left and the width of each panel (a: the frame of the stack; d: the heatmap,
# then its colorbar).
FIG_SIZE = (6.5, 2.4)
BOTTOM = 0.34
TOP = FIG_SIZE[1] - 0.16
PANELS = {'a': (0.0, 1.15), 'b': (1.75, 0.98), 'c': (3.2, 1.05), 'd': (5.0, 1.05),
          'colorbar': (6.1, 0.06)}
# x of the panel letters in inches.
LETTERS = {'a': 0.0, 'b': 1.4, 'c': 2.8, 'd': 4.3}
# Panel a, in inches from the bottom left of its frame: left and bottom of the front card (the
# bottom of the other panels), its width, the offset of each card behind it, the number of near
# cards and the position of the last card, and the room above the last card for its label.
STACK = {'x0': 0.39, 'y0': BOTTOM, 'width': 0.56, 'offset': (0.03, 0.032), 'near': 4, 'far': 6.5,
         'top_room': 0.13}
# Panel c: height of the room above each coefficient axes for its title.
TITLE_ROOM = 0.16
# Panel titles and annotations at the text size of the figure.
RC = {**SIM_FIG_RC, **MATH_RC, 'axes.titlesize': 'medium'}

# Short labels of the measured traits (column names of Supplemental Table S12).
MEASURED_LABELS = {'gs': '$g_s$', 'AN': '$A_N$', 'ci/ca': '$c_i/c_a$'}


def read_measured_traits(path):
    """Genotypes, trait labels and (genotypes, traits) values (NaN where empty) of Table S12."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    header = next(r for r in rows if r and r[0] == 'Genotype')
    n = header.index('', 1)  # the trait columns end at the first empty column
    body = [r for r in rows[rows.index(header) + 1:] if r and r[0]]
    labels = []
    for name in header[1:n]:
        trait, model = name.split(' (', 1)
        labels.append(f"{MEASURED_LABELS.get(trait, trait)} ({model.replace(' model', '')}")
    values = np.array([[float(v) if v else np.nan for v in r[1:n]] for r in body])
    return np.array([r[0] for r in body]), labels, values


def genotype_means(Y, genotypes, names):
    """(names, traits) means of the rows of Y of each genotype in names."""
    return np.array([Y[genotypes == name].mean(axis=0) for name in names])


def axes_at(fig, left, bottom, width, height):
    """Axes at a position given in inches from the bottom left of the figure."""
    fig_width, fig_height = fig.get_size_inches()
    return fig.add_axes([left / fig_width, bottom / fig_height, width / fig_width,
                         height / fig_height])


def spectra_stack(frame, wavelengths, X, n_plants, n_genotypes):
    """Draw the input as a stack of cards in the axes frame: the front card is an axes with the
    spectrum of the first plant, the cards behind it are offset up and to the right, and the last
    card shows part of the spectrum of the second plant, clipped to the card. The depth of the
    stack is labeled with the number of plants, its top with the number of genotypes (clones).

    The data coordinates of frame are inches from its bottom left (STACK gives the geometry)."""
    width, height = (frame.get_position().size * frame.figure.get_size_inches())
    frame.set_axis_off()
    frame.set_xlim(0, width)
    frame.set_ylim(0, height)
    x0, y0, w = STACK['x0'], STACK['y0'], STACK['width']
    dx, dy = STACK['offset']
    near, far = STACK['near'], STACK['far']
    h = height - y0 - far * dy - STACK['top_room']
    card = dict(facecolor='white', edgecolor='black', linewidth=0.6, clip_on=False)
    # the last card, with part of a spectrum, clipped to the card
    bx, by = x0 + far * dx, y0 + far * dy
    back = Rectangle((bx, by), w, h, zorder=1, **card)
    frame.add_patch(back)
    low, high = X[:2].min(), X[:2].max()
    xs = bx + 0.1 * w + 0.85 * w * (wavelengths - wavelengths[0]) / np.ptp(wavelengths)
    ys = by + 0.08 * h + 0.85 * h * (X[1] - low) / (high - low)
    line, = frame.plot(xs, ys, color=COLORS['h2opt'], zorder=1.5)
    line.set_clip_path(back)
    for i in range(near, 0, -1):
        frame.add_patch(Rectangle((x0 + i * dx, y0 + i * dy), w, h, zorder=2, **card))
    # three dots between the near cards and the last card, at the top-left and bottom-right
    # corners
    middle = (near + far) / 2 + np.array([-0.6, 0, 0.6])
    for corner_x, corner_y in [(x0, y0 + h), (x0 + w, y0)]:
        frame.plot(corner_x + middle * dx, corner_y + middle * dy, 'o', color='black', ms=0.8,
                   clip_on=False)
    # the number of plants along the top-left edge of the stack, the number of clones on top
    angle = np.degrees(np.arctan2(dy, dx))
    along = np.array([dx, dy]) / np.hypot(dx, dy)
    start = np.array([x0, y0 + h]) - 0.05 * along + 0.07 * np.array([-along[1], along[0]])
    n_text = f'{n_plants:,}'.replace(',', '{,}')
    frame.text(*start, f'$n = {n_text}$', rotation=angle, rotation_mode='anchor', ha='left',
               va='bottom', fontsize=TICK_SIZE)
    frame.text(bx + 0.12, by + h + 0.03, f'$c = {n_genotypes}$ clones', ha='left', va='bottom',
               fontsize=TICK_SIZE)
    # the front card
    ax = frame.inset_axes([x0, y0, w, h], transform=frame.transData, zorder=3)
    ax.plot(wavelengths, X[0], color=COLORS['h2opt'])
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('intensity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heritability', required=True)
    parser.add_argument('--coefficients', required=True,
                        help='H2-opt coefficients of one fold (pch_coefficients.py .npz)')
    parser.add_argument('--fold-traits', required=True, help='saved H2-opt traits of that fold')
    parser.add_argument('--traits-files', nargs='+', required=True, help='one .npz per fold')
    parser.add_argument('--measured-traits', required=True, help='Supplemental Table S12 (.tsv)')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    parser.add_argument('--ndvi', required=True, help='heritability of NDVI (ndvi.py .npz)')
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()
    k = args.traits

    X, genotypes, _ = sorghum.load()
    coefficients = np.load(args.coefficients)['h2opt_coefficients'].T  # wavelengths x traits
    traits = X @ coefficients
    saved = sorghum.load_traits(args.fold_traits)
    agreement = [np.corrcoef(traits[:, t], saved[:, t])[0, 1] for t in range(saved.shape[1])]
    # float32 outputs (saved) vs float64 (here)
    assert min(agreement) > 1 - 1e-5, f'coefficients and saved traits differ (r {agreement})'

    plt.rcParams.update(RC)
    small = plt.rcParams['xtick.labelsize']  # legends and heatmap values at the tick label size
    fig = plt.figure(figsize=FIG_SIZE)
    panels = {}

    def panel(name):
        left, width = PANELS[name]
        panels[name] = axes_at(fig, left, BOTTOM, width, TOP - BOTTOM)
        return panels[name]

    # a: the input, a stack of cards with one spectrum per plant
    wavelengths = sorghum.FIRST_WAVELENGTH + np.arange(X.shape[1])
    n_plants, n_genotypes = len(X), len(np.unique(genotypes))
    assert (n_plants, n_genotypes) == (1920, 854), (n_plants, n_genotypes)
    left, width = PANELS['a']
    panels['a'] = axes_at(fig, left, 0, width, TOP)
    spectra_stack(panels['a'], wavelengths, X, n_plants, n_genotypes)

    # b: test heritability of every trait: median over the folds and band from the 25th to the
    # 75th percentile; NDVI's median test heritability over the folds as a dotted line
    results = np.load(args.heritability)
    methods = list(results['methods'])
    test = results['test']  # methods x folds x traits
    ndvi = np.median(np.load(args.ndvi)['test'])
    all_traits = np.arange(1, test.shape[2] + 1)
    ax = panel('b')
    median_bands(ax, all_traits, test, methods)
    ax.axhline(ndvi, color='0.3', ls=':', lw=0.8, label='NDVI')
    # the legend at the top right, above the lines of the last traits
    ordered_legend(ax, [LABELS[m] for m in ordered(methods)] + ['NDVI'], frameon=False,
                   loc='upper right', fontsize=small, handlelength=1.2, handletextpad=0.4,
                   borderaxespad=0, labelspacing=0.2)
    ax.set_ylim(0, 1)
    ax.set_xticks(all_traits)
    ax.set_xlabel('synthetic trait')
    ax.set_ylabel('test heritability')
    ax.spines[['top', 'right']].set_visible(False)
    order = np.arange(1, k + 1)

    # c: wavelength coefficients of the traits of one fold
    left, width = PANELS['c']
    step = (TOP - BOTTOM + TITLE_ROOM) / k
    for t in range(k):
        ax = axes_at(fig, left, TOP - (t + 1) * step + TITLE_ROOM, width, step - TITLE_ROOM)
        if t == 0:
            panels['c'] = ax
        top = np.abs(coefficients[:, t]).max()
        ax.plot(wavelengths, coefficients[:, t], lw=0.6, color=COLORS['h2opt'])
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_ylim(-1.1 * top, 1.1 * top)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_xticks(range(500, 2501, 500))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True))
        ax.set_title(f'trait {t + 1}', loc='left', pad=1.5)
        ax.spines[['top', 'right']].set_visible(False)
        if t < k - 1:
            ax.tick_params(axis='x', length=0, labelbottom=False)
    ax.set_xlabel('wavelength (nm)')
    fig.text((left - 0.36) / FIG_SIZE[0], (BOTTOM + TOP) / 2 / FIG_SIZE[1],
             'coefficient', rotation=90, ha='center', va='center')

    # d: |r| of genotype means with the measured traits, median over the folds
    names, labels, measured = read_measured_traits(args.measured_traits)
    phenotyped = np.isin(names, genotypes)  # the table also has genotypes without spectra
    names, measured = names[phenotyped], measured[phenotyped]
    correlation = np.zeros((len(args.traits_files), k, measured.shape[1]))
    for f, path in enumerate(args.traits_files):
        means = genotype_means(sorghum.load_traits(path)[:, :k], genotypes, names)
        for j in range(measured.shape[1]):
            known = ~np.isnan(measured[:, j])
            for t in range(k):
                correlation[f, t, j] = np.corrcoef(means[known, t], measured[known, j])[0, 1]
    median = np.median(np.abs(correlation), axis=0).T  # measured x traits
    ax = panel('d')
    image = ax.imshow(median, cmap='magma', vmin=0, aspect='auto')
    for (j, t), value in np.ndenumerate(median):
        ax.text(t, j, f'{value:.2f}', ha='center', va='center', fontsize=small,
                color='black' if value > 0.6 * median.max() else 'white')
    ax.set_xticks(range(k), order)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel('synthetic trait')
    bar = fig.colorbar(image, cax=panel('colorbar'))
    bar.ax.set_xlabel('$|r|$')

    for name, x in LETTERS.items():
        left, width = PANELS[name]
        letter(panels[name], name, (x - left) / width)
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=300)

    print('b: test heritability per trait over the folds (25th, 50th, 75th percentile; mean)')
    for method in ordered(methods):
        values = test[methods.index(method)]
        print(f'  {LABELS[method]:16s} median', np.round(np.median(values, axis=0), 3))
        print(f'  {"":16s} 25th  ', np.round(np.percentile(values, 25, axis=0), 3))
        print(f'  {"":16s} 75th  ', np.round(np.percentile(values, 75, axis=0), 3))
        print(f'  {"":16s} mean  ', np.round(values.mean(axis=0), 3))
    print(f'  NDVI median test heritability over the folds {ndvi:.3f}')
    print('d: median |r| over folds (rows: measured traits; columns: synthetic traits)')
    for label, row in zip(labels, median, strict=True):
        print(f'  {label:24s}', np.round(row, 2))
    j, t = np.unravel_index(np.argmax(median), median.shape)
    print(f'  highest: trait {t + 1} with {labels[j]}, {median[j, t]:.2f}')


if __name__ == '__main__':
    main()
