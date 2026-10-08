"""Accuracy of H2-opt against the noise level, over the simulation instances (noise_sweep.py).

For each instance and level: |r| of true trait t with synthetic trait t (the paper's metric,
t = 1..3), and the test heritability of synthetic traits 1-3. The H2-opt traits of the pipeline
(train_h2opt.py, at the level of choose_noise.py) are marked at the median chosen level. Writes an
.npz (levels; abs_cor and test_herit: instances x levels x 3; chosen_level and chosen_abs_cor: per
instance), a figure, and prints the median |r| per level.

    python pipelines/scripts/simulation/plot_noise_sweep.py --sweep w0.npz ... \\
        --truth t0.npz ... --spectra s0.npz ... --h2opt h0.npz ... --noise n0.npz ... \\
        --out sweep.npz --figure noise_sweep.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import simulation


def position(levels, level):
    """Position of level on an axis with the levels at 0, 1, ...: log-linear between levels."""
    levels = np.asarray(levels)
    upper = int(np.searchsorted(levels, level))
    if upper == 0 or levels[upper - 1] == 0:
        return float(upper)
    return upper - 1 + np.log(level / levels[upper - 1]) / np.log(levels[upper] / levels[upper - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', nargs='+', required=True, help='one file per instance')
    parser.add_argument('--truth', nargs='+', required=True, help='in the order of --sweep')
    parser.add_argument('--spectra', nargs='+', required=True, help='in the order of --sweep')
    parser.add_argument('--h2opt', nargs='+', required=True, help='in the order of --sweep')
    parser.add_argument('--noise', nargs='+', required=True, help='in the order of --sweep')
    parser.add_argument('--out', required=True)
    parser.add_argument('--figure', required=True)
    args = parser.parse_args()
    n_inst, k = len(args.sweep), simulation.N_TRAITS
    if not n_inst == len(args.truth) == len(args.spectra) == len(args.h2opt) == len(args.noise):
        parser.error('give one sweep, truth, spectra, H2-opt and noise file per instance')

    abs_cor = test_herit = None
    chosen_level, chosen_cor = np.zeros(n_inst), np.zeros((n_inst, k))
    for i in range(n_inst):
        with np.load(args.sweep[i]) as f:
            levels, traits = f['levels'], f['traits']
        with np.load(args.truth[i]) as f:
            truth = f['traits']
        with np.load(args.spectra[i]) as f:
            names, test = f['names'], f['test']
        with np.load(args.h2opt[i]) as f:
            chosen = f['traits']
        with np.load(args.noise[i]) as f:
            chosen_level[i] = float(f['chosen'])
        if abs_cor is None:
            abs_cor, test_herit = np.zeros((2, n_inst, len(levels), k))
        for a, Y in enumerate(traits):
            abs_cor[i, a] = np.abs(np.diag(simulation.correlations(truth, Y[:, :k])))
            test_herit[i, a] = anova_heritability(Y[test, :k], names[test], None)
        chosen_cor[i] = np.abs(np.diag(simulation.correlations(truth, chosen[:, :k])))

    np.savez(args.out, levels=levels, abs_cor=abs_cor, test_herit=test_herit,
             chosen_level=chosen_level, chosen_abs_cor=chosen_cor)
    print('median |r| of true trait t with synthetic trait t (columns: traits 1-3)')
    for a, lev in enumerate(levels):
        print(f'  level {lev:<6g}', np.median(abs_cor[:, a], axis=0).round(3))
    print('  chosen levels', ' '.join(f'{lev:g}' for lev in chosen_level), '(pipeline H2-opt):',
          np.median(chosen_cor, axis=0).round(3))

    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3), sharey=True)
    x = np.arange(len(levels))
    chosen_x = position(levels, np.median(chosen_level))
    for t, ax in enumerate(axes):
        for i in range(n_inst):
            ax.plot(x, abs_cor[i, :, t], '-', color='0.75', lw=0.8)
        ax.plot(x, np.median(abs_cor[:, :, t], axis=0), 'o-', color='tab:blue', ms=3,
                label='median over instances')
        ax.plot(chosen_x, np.median(chosen_cor[:, t]), '*', color='tab:red', ms=9,
                label='pipeline, chosen levels (median)')
        ax.set_xticks(x, [f'{lev:g}' for lev in levels], fontsize=8)
        ax.set_xlabel('noise level (noise SD / noise_scale)')
        ax.set_title(f'trait {t + 1}', fontsize=10)
    axes[0].set_ylabel('|r| with the true trait')
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(args.figure)
    fig.savefig(str(args.figure).removesuffix('.pdf') + '.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
