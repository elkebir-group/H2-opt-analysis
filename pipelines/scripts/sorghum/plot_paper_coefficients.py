"""Supplementary sorghum figure of the wavelength coefficients of H2-opt and PCH (sorCoefficients).

Reads the coefficients of pch_coefficients.py, one .npz per fold (--coefficients, first file:
the fold shown in panels a and b, --fold). Panels: a, the coefficients of all traits of linear
H2-opt in that fold over the full spectrum, in two columns; each trait has two rows, H2-opt on
top and below it the PCH trait with the highest |Pearson r| on the fold's training plants, with
the sign of r, each on its own y axis; b, trait 1 and its PCH trait between --zoom nm, each scaled
to a largest |coefficient| of 1 in that window; c, the total variation of the coefficients over
their sum of absolute values (sum |theta[i+1] - theta[i]| / sum |theta[i]|) of traits 1 to 10 of
each method: median over the folds and band from the 25th to the 75th percentile. In c, trait t
of PCH is PCH's own trait t. The figure is drawn at the text width of the paper (6.5 in) in the
style of the simulation figures. Writes the figure as .pdf and, beside it, .png, and prints the
pairing and the total variation.

    python pipelines/scripts/sorghum/plot_paper_coefficients.py --fold 0 \\
        --coefficients pch_coefficients/fold0.npz ... pch_coefficients/fold9.npz \\
        --zoom 400 800 --out figures/paper/sorCoefficients.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from h2opt_analysis import sorghum  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    COLORS,
    MATH_RC,
    SIM_FIG_RC,
    letter,
    median_bands,
    ordered_legend,
)
from h2opt_analysis.methods import LABELS  # noqa: E402

# Positions in inches from the bottom left of the figure.
FIG_SIZE = (6.5, 6.6)
COLUMNS = [(0.55, 2.55), (3.85, 2.55)]  # panel a: left and width of each column
ROW = 0.27  # height of one method row in panel a
PAIR_GAP = 0.08  # between the H2-opt and the PCH row of a trait
TRAIT_GAP = 0.14  # between two traits
A_TOP = FIG_SIZE[1] - 0.32  # top of panel a (the legend and the letter are above it)
BOTTOM_PANELS = {'b': (0.55, 2.55), 'c': (3.85, 2.55)}  # left and width
BOTTOM_PANEL_Y = (0.42, 1.45)  # bottom and height of panels b and c
LETTER_X = 0.12  # x of the panel letters a and b in inches
C_LETTER_X = 3.3  # x of the panel letter c in inches
RC = {**SIM_FIG_RC, **MATH_RC}


def axes_at(fig, left, bottom, width, height):
    """Axes at a position given in inches from the bottom left of the figure."""
    fig_width, fig_height = fig.get_size_inches()
    return fig.add_axes([left / fig_width, bottom / fig_height, width / fig_width,
                         height / fig_height])


def row_label(ax, text):
    """Small label at the top left inside ax, on a light background."""
    ax.text(0.01, 0.95, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=plt.rcParams['xtick.labelsize'],
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.6))


def coefficient_row(ax, wavelengths, theta, color):
    """Coefficients theta over the wavelengths in ax, on a symmetric y axis."""
    top = np.abs(theta).max()
    ax.plot(wavelengths, theta, lw=0.6, color=color)
    ax.axhline(0, color='0.7', lw=0.5)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(-1.1 * top, 1.1 * top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, symmetric=True))
    ax.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True,
                        help='fold of the first --coefficients file (panels a and b)')
    parser.add_argument('--coefficients', nargs='+', required=True,
                        help='pch_coefficients.py output, one per fold, --fold first')
    parser.add_argument('--zoom', type=float, nargs=2, required=True, help='nm range of b')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()

    X, genotypes, _ = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    wavelengths = sorghum.FIRST_WAVELENGTH + np.arange(X.shape[1])
    fold = np.load(args.coefficients[0])
    h2opt = fold['h2opt_coefficients'].T  # wavelengths x traits
    pch = fold['coefficients'].T
    k = h2opt.shape[1]
    # the PCH trait of each H2-opt trait: highest |r| on the training plants, with the sign of r
    r = np.corrcoef((X[is_train] @ h2opt).T,
                    ((X[is_train] - fold['mean']) @ pch).T)[:k, k:]
    pairs = np.abs(r).argmax(axis=1)
    signs = np.sign(r[np.arange(k), pairs])
    paired = pch[:, pairs] * signs

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=FIG_SIZE)
    per_column = (k + 1) // 2
    top_axes, column_axes = None, []
    for t in range(k):
        column, slot = divmod(t, per_column)
        left, width = COLUMNS[column]
        top = A_TOP - slot * (2 * ROW + PAIR_GAP + TRAIT_GAP)
        upper = axes_at(fig, left, top - ROW, width, ROW)
        lower = axes_at(fig, left, top - 2 * ROW - PAIR_GAP, width, ROW)
        coefficient_row(upper, wavelengths, h2opt[:, t], COLORS['h2opt'])
        coefficient_row(lower, wavelengths, paired[:, t], COLORS['pch'])
        row_label(upper, f'trait {t + 1}')
        row_label(lower, f'PCH trait {pairs[t] + 1}, |r| = {abs(r[t, pairs[t]]):.2f}')
        upper.tick_params(axis='x', length=0, labelbottom=False)
        if slot < per_column - 1 and t < k - 1:
            lower.tick_params(axis='x', length=0, labelbottom=False)
        else:
            lower.set_xlabel('wavelength (nm)')
        if t == 0:
            top_axes = upper
        if slot == 0:
            column_axes.append((left, top))
    for left, top in column_axes:
        bottom = top - per_column * (2 * ROW + PAIR_GAP) - (per_column - 1) * TRAIT_GAP
        fig.text((left - 0.42) / FIG_SIZE[0], (top + bottom) / 2 / FIG_SIZE[1], 'coefficient',
                 rotation=90, ha='center', va='center')
    handles = [plt.Line2D([], [], color=COLORS[s]) for s in ['h2opt', 'pch']]
    fig.legend(handles, [LABELS['h2opt'], LABELS['pch']], loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, (A_TOP + 0.04) / FIG_SIZE[1]), frameon=False,
               borderaxespad=0, handlelength=1.5)
    letter(top_axes, 'a', (LETTER_X - COLUMNS[0][0]) / COLUMNS[0][1])

    # b: trait 1 and its PCH trait in the zoom window, scaled
    left, width = BOTTOM_PANELS['b']
    ax = axes_at(fig, left, BOTTOM_PANEL_Y[0], width, BOTTOM_PANEL_Y[1])
    window = (wavelengths >= args.zoom[0]) & (wavelengths <= args.zoom[1])
    for slug, theta, lw in [('pch', paired[window, 0], 0.6), ('h2opt', h2opt[window, 0], 0.9)]:
        ax.plot(wavelengths[window], theta / np.abs(theta).max(), lw=lw, color=COLORS[slug],
                label=LABELS[slug])
    ax.axhline(0, color='0.7', lw=0.5)
    ordered_legend(ax, [LABELS['h2opt'], LABELS['pch']], frameon=False, loc='lower right',
                   bbox_to_anchor=(1, 1), ncol=2, borderaxespad=0.1,
                   fontsize=plt.rcParams['xtick.labelsize'], handlelength=1.2)
    ax.set_xlim(*args.zoom)
    ax.set_title('trait 1', loc='left')
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('coefficient (scaled)')
    ax.spines[['top', 'right']].set_visible(False)
    letter(ax, 'b', (LETTER_X - left) / width)

    # c: total variation ratio per trait over the folds
    ratios = {'h2opt': [], 'pch': []}
    for path in args.coefficients:
        with np.load(path) as f:
            ratios['h2opt'].append(f['h2opt_total_variation_ratio'])
            ratios['pch'].append(f['pch_total_variation_ratio'])
    slugs = ['h2opt', 'pch']
    values = np.array([ratios[s] for s in slugs])  # methods x folds x traits
    traits = np.arange(1, values.shape[2] + 1)
    left, width = BOTTOM_PANELS['c']
    ax = axes_at(fig, left, BOTTOM_PANEL_Y[0], width, BOTTOM_PANEL_Y[1])
    median_bands(ax, traits, values, slugs)
    ordered_legend(ax, [LABELS[s] for s in slugs], frameon=False, loc='center right',
                   fontsize=plt.rcParams['xtick.labelsize'], handlelength=1.2)
    ax.set_xticks(traits)
    ax.set_ylim(0, None)
    ax.set_xlabel('synthetic trait')
    ax.set_ylabel('total variation / sum |coefficient|')
    ax.spines[['top', 'right']].set_visible(False)
    letter(ax, 'c', (C_LETTER_X - left) / width)

    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=300)

    print(f'a: PCH trait paired with each H2-opt trait (highest |r| on the training plants of '
          f'fold {args.fold}; PCH sign flipped where r < 0)')
    for t in range(k):
        print(f'  H2-opt trait {t + 1:2d} -> PCH trait {pairs[t] + 1:2d}: '
              f'r {r[t, pairs[t]]:+.3f}, sign {signs[t]:+.0f}')
    if len(set(pairs)) < k:
        print('  note: some PCH traits are paired with more than one H2-opt trait')
    print(f'b: trait 1, {args.zoom[0]:g}-{args.zoom[1]:g} nm, each scaled to a largest '
          '|coefficient| of 1 in the window')
    print(f'c: total variation / sum |coefficient| over {len(args.coefficients)} folds '
          '(median, 25th and 75th percentile)')
    for s, v in zip(slugs, values, strict=True):
        print(f'  {LABELS[s]:8s} median', np.round(np.median(v, axis=0), 3))
        print(f'  {"":8s} 25th  ', np.round(np.percentile(v, 25, axis=0), 3))
        print(f'  {"":8s} 75th  ', np.round(np.percentile(v, 75, axis=0), 3))


if __name__ == '__main__':
    main()
