"""Sparsity of the H2-opt coefficients against PCH's on Arabidopsis (images/arabiSparsity), in the
style of the sorghum figure of smoothness (sorCoefficients).

a, b: per treatment, the coefficients of H2-opt trait 1 in the first fold and of the PCH trait
paired with it (highest |r| on the fold's training samples, with the sign of r), drawn as in the
coefficient figure (arabidopsis.coefficient_map). c: per treatment, the number of features that
carry 90% of the total magnitude of each trait's coefficients (arabidopsis.share_features), median
and 25th to 75th percentile over the folds, H2-opt and PCH. Coefficients come from coefficients.py
(H2-opt) and pch_coefficients.py (PCH), traits from train_h2opt.py and baseline.py. The figure is
drawn at the text width of the paper (6.5 in). Saves --out (.pdf) and a .png next to it, and
prints the pairing and the feature counts.

    python pipelines/scripts/arabidopsis/plot_paper_sparsity.py \\
        --treatment 0d --h2opt-coefficients 0d/h2opt/coefficients_fold{0..4}.npz \\
        --pch-coefficients 0d/pch_coefficients/fold{0..4}.npz \\
        --h2opt-traits 0d/h2opt/fold{0..4}.npz --pch-traits 0d/pch/fold{0..4}.npz \\
        --treatment 6d ... --out arabiSparsity.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from h2opt_analysis import arabidopsis  # noqa: E402
from h2opt_analysis.figures import (  # noqa: E402
    LETTER_SIZE,
    MATH_RC,
    SIM_FIG_RC,
    median_bands,
    ordered_legend,
)
from h2opt_analysis.methods import LABELS  # noqa: E402

FIG_SIZE = (6.5, 3.9)
MZ_SHOWN = (100, 1300)  # m/z range of the maps (all bins below 100 are empty)
MARKER = 8  # marker area of the largest coefficient
SHARE = 0.9
RC = {**SIM_FIG_RC, **MATH_RC, 'axes.titlesize': 'medium'}
PER_TREATMENT = ('h2opt_coefficients', 'pch_coefficients', 'h2opt_traits', 'pch_traits')


def pairing(h2opt, pch, rows):
    """Index and sign of the PCH trait paired with each H2-opt trait (highest |r| on rows), and
    the |r|."""
    k = h2opt.shape[1]
    r = np.corrcoef(h2opt[rows].T, pch[rows].T)[:k, k:]
    pairs = np.abs(r).argmax(axis=1)
    picked = r[np.arange(k), pairs]
    return pairs, np.sign(picked), np.abs(picked)


def heading(subfigure, text, letter):
    subfigure.suptitle(f'$\\bf{{{letter}}}$  {text}', x=0.0, ha='left', fontsize=LETTER_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', action='append', required=True)
    for name in PER_TREATMENT:
        parser.add_argument('--' + name.replace('_', '-'), nargs='+', action='append',
                            required=True, help='one file per fold, after each --treatment')
    parser.add_argument('--out', required=True, help='.pdf; a .png is written beside it')
    args = parser.parse_args()
    files = {name: getattr(args, name) for name in PER_TREATMENT}
    if any(len(v) != len(args.treatment) for v in files.values()) or any(
            len({len(v[i]) for v in files.values()}) != 1 for i in range(len(args.treatment))):
        parser.error('give one file per fold of each kind after each --treatment')

    plt.rcParams.update(RC)
    fig = plt.figure(figsize=FIG_SIZE, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.05, wspace=0.03)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])
    maps = top.subfigures(1, len(args.treatment))
    counts = bottom.subplots(1, len(args.treatment), sharey=True)
    heading(bottom, f'features that carry {SHARE:.0%} of the total |coefficient|',
            'abcdefgh'[len(args.treatment)])
    for i, treatment in enumerate(args.treatment):
        days = f'{treatment.removesuffix("d")} day'
        _, accessions, _ = arabidopsis.design(treatment)
        fold_of = arabidopsis.folds(accessions)
        n = {'h2opt': [], 'pch': []}
        for fold, paths in enumerate(zip(*(files[name][i] for name in PER_TREATMENT),
                                         strict=True)):
            h2opt_c, pch_c, h2opt_t, pch_t = paths
            h2opt = np.load(h2opt_c)['coefficients']
            pch = np.load(pch_c)['coefficients']
            pairs, signs, r = pairing(np.load(h2opt_t)['traits'], np.load(pch_t)['traits'],
                                      fold_of != fold)
            n['h2opt'].append(arabidopsis.share_features(h2opt, SHARE))
            n['pch'].append(arabidopsis.share_features(pch[pairs], SHARE))
            print(f'{days}, fold {fold + 1}: PCH trait of each H2-opt trait {pairs + 1}, |r| '
                  f'{np.round(r, 2)}')
            print(f'  features with {SHARE:.0%}: H2-opt {n["h2opt"][-1]}, PCH {n["pch"][-1]}')
            if fold == 0:
                axes = maps[i].subplots(1, 2, sharex=True, sharey=True)
                heading(maps[i], f'{days}, trait 1, first fold', 'abcdefgh'[i])
                for ax, coefficients, title in [
                        (axes[0], h2opt[0], 'H2-opt'),
                        (axes[1], signs[0] * pch[pairs[0]],
                         f'PCH trait {pairs[0] + 1}, |r| = {r[0]:.2f}')]:
                    arabidopsis.coefficient_map(ax, coefficients, MARKER)
                    ax.set_title(title)
                    ax.set_xlabel('retention time (min)')
                    ax.spines[['top', 'right']].set_visible(False)
                axes[0].set_ylabel('$m/z$ (Da)')
                axes[0].set_xlim(0, arabidopsis.N_RT * arabidopsis.RT_BIN)
                axes[0].set_ylim(*MZ_SHOWN)
                axes[0].set_xticks([0, 10, 20])
                axes[0].set_yticks([100, 500, 1000])
        values = np.stack([n['h2opt'], n['pch']])  # methods x folds x traits
        order = np.arange(1, values.shape[2] + 1)
        ax = counts[i]
        median_bands(ax, order, values, ['h2opt', 'pch'])
        ax.set_yscale('log')
        ax.set_xticks(order)
        ax.set_xlabel('synthetic trait')
        ax.set_title(days)
        ax.grid(axis='y', color='0.9', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for method, v in zip(('h2opt', 'pch'), values, strict=True):
            print(f'{days}: {LABELS[method]} median over the folds '
                  f'{np.median(v, axis=0).astype(int)}')
    counts[0].set_ylabel('features')
    counts[0].set_ylim(100, 1e5)
    ordered_legend(counts[-1], [LABELS['h2opt'], LABELS['pch']], frameon=False,
                   loc='center right', handlelength=1.2, handletextpad=0.4)
    handles = [Line2D([], [], marker='o', ls='', color=c, alpha=0.6)
               for _, c in arabidopsis.LEVEL_COLORS]
    top.legend(handles, [name for name, _ in arabidopsis.LEVEL_COLORS], loc='upper right',
               frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    out = Path(args.out)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=300)


if __name__ == '__main__':
    main()
