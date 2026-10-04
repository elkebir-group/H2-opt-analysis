"""Train and held-out heritability of sorghum H2-opt over a grid of noise levels.

Each --level gives a noise level (the standard deviation of the normal noise, as a fraction of the
root mean variance of the wavelengths) and the traits file of H2-opt trained at that level with the
plants of --test-folds held out (train_h2opt.py). Writes an .npz (levels; train and test:
levels x traits), a figure of the held-out heritability per trait against the noise level (level
0, no noise, as a dashed line), and
prints both tables.

    python pipelines/scripts/sorghum/noise_scan.py --test-folds 0 1 \\
        --level 0.01 sd0.01.npz --level 0.1 sd0.1.npz --out noise_scan.npz --figure noise_scan.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-folds', type=int, nargs='+', required=True)
    parser.add_argument('--level', nargs=2, action='append', required=True,
                        metavar=('LEVEL', 'FILE'), help='a noise level and its traits file')
    parser.add_argument('--out', required=True)
    parser.add_argument('--figure', required=True)
    args = parser.parse_args()

    _, genotypes, environment = sorghum.load()
    is_test = np.isin(sorghum.folds(genotypes), args.test_folds)
    levels = sorted((float(level), path) for level, path in args.level)
    train, test = [], []
    for _, path in levels:
        Y = sorghum.load_traits(path)
        train.append(anova_heritability(Y[~is_test], genotypes[~is_test], environment[~is_test]))
        test.append(anova_heritability(Y[is_test], genotypes[is_test], environment[is_test]))
    levels = np.array([level for level, _ in levels])
    train, test = np.array(train), np.array(test)
    np.savez(args.out, levels=levels, train=train, test=test)

    for name, table in (('train', train), ('held-out', test)):
        print(f'{name} heritability per trait')
        for level, row in zip(levels, table, strict=True):
            print(f'  {level:<8g}', np.round(row, 3))

    # Positive levels on a log axis; the level 0 (no noise), if scanned, as a dashed line.
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    positive = levels > 0
    for t in range(test.shape[1]):
        line, = ax.plot(levels[positive], test[positive, t], 'o-', label=f'trait {t + 1}')
        if not positive.all():
            ax.axhline(test[~positive, t][0], color=line.get_color(), ls='--', lw=1)
    if not positive.all():
        ax.plot([], [], 'k--', lw=1, label='no noise')
    ax.set_xscale('log')
    ax.set_xlabel('noise SD / feature SD')
    ax.set_ylabel('held-out heritability')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figure)
    fig.savefig(args.figure.removesuffix('.pdf') + '.png', dpi=200)


if __name__ == '__main__':
    main()
