"""Train and test heritability of H2-opt and the baselines on sorghum, per genotype fold.

The H2-opt traits of fold --folds[k] are the k-th --h2opt-traits file, used as saved (each source
saves its traits uncorrelated: on the training plants for rmsprop and lbfgs, on all plants for
the paper's predictions). --source says how they were made. Baselines are fitted on the training
folds. Writes an .npz (methods: slugs of h2opt_analysis.methods; folds; train and test: methods x
folds x traits) and prints the mean test heritability of each trait.

    python pipelines/scripts/sorghum/evaluate_heritability.py --source lbfgs --folds 0 ... 9 \\
        --h2opt-traits fold0.npz ... fold9.npz --out heritability_lbfgs.npz --traits 5
"""

import argparse

import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import sorghum
from h2opt_analysis.methods import BASELINES, LABELS, fit_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['paper', 'rmsprop', 'lbfgs'], required=True)
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--h2opt-traits', nargs='+', required=True,
                        help='traits file of each fold, in the order of --folds')
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()
    if len(args.h2opt_traits) != len(args.folds):
        parser.error('give one --h2opt-traits file per fold')

    X, genotypes, environment = sorghum.load()
    fold_of_plant = sorghum.folds(genotypes)
    methods = [f'h2opt_{args.source}'] + list(BASELINES)
    train = np.zeros((len(methods), len(args.folds), args.traits))
    test = np.zeros_like(train)

    for k, fold in enumerate(args.folds):
        is_train, is_test = fold_of_plant != fold, fold_of_plant == fold
        for m, method in enumerate(methods):
            if m == 0:
                Y = sorghum.load_traits(args.h2opt_traits[k])[:, :args.traits]
            else:
                Y = fit_baseline(method, X, genotypes, environment, is_train, args.traits)
            for out, rows in ((train, is_train), (test, is_test)):
                out[m, k] = anova_heritability(Y[rows], genotypes[rows], environment[rows])
        scores = ', '.join(f'{LABELS[m]} {test[i, k, 0]:.3f}' for i, m in enumerate(methods))
        print(f'fold {fold}: test trait 1 {scores}', flush=True)

    np.savez(args.out, methods=methods, folds=args.folds, train=train, test=test)
    print('\nmean test heritability per trait')
    for m, method in enumerate(methods):
        print(f'{LABELS[method]:16s}', np.round(test[m].mean(0), 3))


if __name__ == '__main__':
    main()
