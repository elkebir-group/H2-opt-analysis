"""Train and test heritability of H2-opt and the baselines on one Arabidopsis treatment, per
accession fold.

Each --method gives a method slug (h2opt_analysis.methods.LABELS) and its traits file for each fold,
in the order of --folds (train_h2opt.py, baseline.py), all saved uncorrelated on the training
samples. The first --traits traits are evaluated by ANOVA heritability among accessions, with the
year as environment. Writes an .npz (methods; folds; train and test: methods x folds x traits) and
prints the mean test heritability of each trait.

    python pipelines/scripts/arabidopsis/evaluate_heritability.py --treatment 6d --folds 0 ... 4 \\
        --method h2opt h2opt/fold0.npz ... --method pch pch/fold0.npz ... \\
        --out heritability.npz --traits 10
"""

import argparse

import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import arabidopsis
from h2opt_analysis.methods import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
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

    _, accessions, environment = arabidopsis.design(args.treatment)
    fold_of = arabidopsis.folds(accessions)
    methods = [m[0] for m in args.method]
    train = np.zeros((len(methods), len(args.folds), args.traits))
    test = np.zeros_like(train)
    for k, fold in enumerate(args.folds):
        for m, method in enumerate(args.method):
            with np.load(method[1 + k]) as f:
                Y = f['traits'][:, :args.traits]
            for out, rows in ((train, fold_of != fold), (test, fold_of == fold)):
                out[m, k] = anova_heritability(Y[rows], accessions[rows], environment[rows])
        scores = ', '.join(f'{LABELS[m]} {test[i, k, 0]:.3f}' for i, m in enumerate(methods))
        print(f'fold {fold}: test trait 1 {scores}', flush=True)

    np.savez(args.out, methods=methods, folds=args.folds, train=train, test=test)
    print('\nmean test heritability per trait')
    for m, method in enumerate(methods):
        print(f'{LABELS[method]:16s}', np.round(test[m].mean(0), 3))


if __name__ == '__main__':
    main()
