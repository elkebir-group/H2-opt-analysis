"""Train and test heritability of H2-opt and the baselines on sorghum, per genotype fold.

Each --method gives a method slug (h2opt_analysis.methods.LABELS) and its traits file for each fold,
in the order of --folds: the H2-opt traits (train_h2opt.py) and the baselines' traits (baseline.py),
all used as saved (uncorrelated on the training plants). The first --traits traits are evaluated.
Writes an .npz (methods; folds; train and test: methods x folds x traits) and prints the mean test
heritability of each trait.

    python pipelines/scripts/sorghum/evaluate_heritability.py --folds 0 ... 9 \\
        --method h2opt h2opt/fold0.npz ... --method pch pch/fold0.npz ... \\
        --out heritability_h2opt.npz --traits 10
"""

import argparse

import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import sorghum
from h2opt_analysis.methods import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--method', nargs='+', action='append', required=True,
                        metavar='SLUG_OR_FILE', help='a method slug and its traits file per fold')
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()
    if any(len(m) != 1 + len(args.folds) for m in args.method):
        parser.error('give one traits file per fold for each method')
    if any(m[0] not in LABELS for m in args.method):
        parser.error(f'method slugs must be in {list(LABELS)}')

    _, genotypes, environment = sorghum.load()
    fold_of_plant = sorghum.folds(genotypes)
    methods = [m[0] for m in args.method]
    train = np.zeros((len(methods), len(args.folds), args.traits))
    test = np.zeros_like(train)

    for k, fold in enumerate(args.folds):
        is_train, is_test = fold_of_plant != fold, fold_of_plant == fold
        for m, method in enumerate(args.method):
            Y = sorghum.load_traits(method[1 + k])[:, :args.traits]
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
