"""Train and test heritability of H2-opt and the baselines on sorghum, over the paper's 10 genotype folds.

H2-opt traits are read from results/sorghum/h2opt/fold{k}.npz (train_h2opt.py), or with --paper from the
predictions behind the paper (data/plant/syntheticTraits). Baselines are fitted on the training folds.
Writes results/sorghum/heritability.npz (train/test: methods x folds x traits) and prints the mean test heritability.

    uv run python scripts/sorghum/evaluate_heritability.py [--paper] [--traits 10]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from h2opt import decorrelate, load_npz

from h2opt_analysis import sorghum
from h2opt_analysis.evaluate import fit_baseline, heritability
from h2opt_analysis.paths import data

BASELINES = ['PCA', 'genetic PCA', 'LDA', 'PCH', 'max features']
RESULTS = Path('results/sorghum')


def h2opt_traits(fold, paper):
    if paper:
        Y = load_npz(data('plant', 'syntheticTraits', f'linear_crossVal_reg4_{fold}_mod10.npz'))
    else:
        Y = np.load(RESULTS / 'h2opt' / f'fold{fold}.npz')['traits']
    return decorrelate(torch.tensor(Y).float()).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', action='store_true', help="use the paper's H2-opt predictions")
    parser.add_argument('--traits', type=int, default=10)
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    folds = sorghum.folds(genotypes)
    methods = ['H2-opt'] + BASELINES
    train = np.zeros((len(methods), sorghum.N_FOLDS, args.traits))
    test = np.zeros_like(train)

    for fold in range(sorghum.N_FOLDS):
        is_train, is_test = folds != fold, folds == fold
        for m, method in enumerate(methods):
            if method == 'H2-opt':
                Y = h2opt_traits(fold, args.paper)[:, :args.traits]
            else:
                Y = fit_baseline(method, X, genotypes, environment, is_train, args.traits)
            train[m, fold] = heritability(Y, genotypes, environment, is_train)
            test[m, fold] = heritability(Y, genotypes, environment, is_test)
        print(f'fold {fold}: test trait 1 ' + ', '.join(f'{m} {test[i, fold, 0]:.3f}' for i, m in enumerate(methods)), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / 'heritability.npz', methods=methods, train=train, test=test)
    print('\nmean test heritability, traits 1-5')
    for m, method in enumerate(methods):
        print(f'{method:13s}', np.round(test[m].mean(0)[:5], 3))


if __name__ == '__main__':
    main()
