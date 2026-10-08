"""Henderson3 heritability of the Miscanthus traits of every method, date and fold.

For each method, the traits of every date and fold (hybrid H2-opt: one file per date, folds x plants
x traits; baselines: one file per fold, dates x plants x traits) are scored by Henderson3 (families,
plots within families, rep and field column fixed) on the fold's training and test plants. The
values are broad-sense on the plant scale; the paper's narrow-sense heritability of half-sib
families is 4 x the value (capped at 1). Writes an .npz (methods; dates; train and test: methods x
dates x folds x traits) and prints the mean narrow-sense test heritability of trait 1.

    python pipelines/scripts/miscanthus/evaluate_heritability.py --population central \\
        --by-date h2opt_hybrid date0.npz ... date13.npz --by-fold pch fold0.npz ... fold4.npz \\
        --traits 1 --out heritability.npz
"""

import argparse

import numpy as np
from h2opt import heritability

from h2opt_analysis import miscanthus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', choices=miscanthus.POPULATIONS, required=True)
    parser.add_argument('--by-date', nargs='+', action='append', default=[],
                        metavar=('METHOD', 'FILE'), help='a method and one file per date')
    parser.add_argument('--by-fold', nargs='+', action='append', default=[],
                        metavar=('METHOD', 'FILE'), help='a method and one file per fold')
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    design = miscanthus.design(args.population)
    n_dates, n_folds = len(miscanthus.DATES), miscanthus.N_FOLDS
    traits = {}  # method -> dates x folds x plants x traits
    for method, *files in args.by_date:
        assert len(files) == n_dates, f'{method}: give one file per date'
        traits[method] = np.stack([np.load(f)['traits'] for f in files])
    for method, *files in args.by_fold:
        assert len(files) == n_folds, f'{method}: give one file per fold'
        traits[method] = np.stack([np.load(f)['traits'] for f in files], axis=1)
    methods = list(traits)
    train, test = np.zeros((2, len(methods), n_dates, n_folds, args.traits))
    for m, method in enumerate(methods):
        for d in range(n_dates):
            for f in range(n_folds):
                Y = traits[method][d, f][:, :args.traits]
                for out, rows in ((train, design.fold != f), (test, design.fold == f)):
                    out[m, d, f] = heritability(Y[rows], design.families[rows],
                                                design.environment[rows], design.plots[rows],
                                                'henderson3')
    np.savez(args.out, methods=methods, dates=miscanthus.DATES, train=train, test=test)
    print('mean narrow-sense test heritability of trait 1 (4 x Henderson3, capped at 1), per date')
    print(f'{"date":8s}' + ''.join(f'{m:>14s}' for m in methods))
    for d, date in enumerate(miscanthus.DATES):
        row = np.minimum(4 * test[:, d, :, 0], 1).mean(axis=1)
        print(f'{date:8s}' + ''.join(f'{v:14.3f}' for v in row))


if __name__ == '__main__':
    main()
