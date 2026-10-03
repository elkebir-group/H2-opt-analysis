"""Train and test heritability of H2-opt and the baselines on sorghum, over the 10 genotype folds.

The H2-opt traits of fold k are the k-th --h2opt-traits file. --source says how they were made:
'paper' (the predictions behind the paper) and 'rmsprop' (train_h2opt.py) are decorrelated here;
'lbfgs' (train_h2opt_lbfgs.py) traits are already uncorrelated on the training plants. Baselines
are fitted on the training folds. Writes an .npz (methods; train and test: methods x folds x traits)
and prints the mean test heritability.

    python pipelines/scripts/sorghum/evaluate_heritability.py --source lbfgs \\
        --h2opt-traits fold0.npz ... fold9.npz --out heritability_lbfgs.npz [--traits 5]
"""

import argparse

import numpy as np
import torch
from h2opt import decorrelate

from h2opt_analysis import sorghum
from h2opt_analysis.evaluate import fit_baseline, heritability

BASELINES = ['PCA', 'genetic PCA', 'LDA', 'PCH', 'max features']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['paper', 'rmsprop', 'lbfgs'], required=True)
    parser.add_argument('--h2opt-traits', nargs=sorghum.N_FOLDS, required=True,
                        help='traits file of each fold, in fold order')
    parser.add_argument('--out', required=True)
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
                Y = sorghum.load_traits(args.h2opt_traits[fold])
                if args.source != 'lbfgs':
                    Y = decorrelate(torch.tensor(Y).float()).numpy()
                Y = Y[:, :args.traits]
            else:
                Y = fit_baseline(method, X, genotypes, environment, is_train, args.traits)
            train[m, fold] = heritability(Y, genotypes, environment, is_train)
            test[m, fold] = heritability(Y, genotypes, environment, is_test)
        scores = ', '.join(f'{m} {test[i, fold, 0]:.3f}' for i, m in enumerate(methods))
        print(f'fold {fold}: test trait 1 {scores}', flush=True)

    np.savez(args.out, methods=methods, train=train, test=test)
    print('\nmean test heritability, traits 1-5')
    for m, method in enumerate(methods):
        print(f'{method:13s}', np.round(test[m].mean(0)[:5], 3))


if __name__ == '__main__':
    main()
