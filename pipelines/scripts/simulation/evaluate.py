"""Accuracy of each method's synthetic traits on the simulation instances.

For each method and instance: the Pearson correlation of each true trait with each synthetic
trait (all plants); the correlation of each true trait with its least-squares fit on the first
three synthetic traits; and the ANOVA heritability of each synthetic trait on the training and
test plants. Also the heritability of the true traits. Writes an .npz and prints a summary:
|r| of true trait t with synthetic trait t, the best |r| over the first three
synthetic traits, the mean absolute deviation of the test heritability of synthetic traits 1-3
from the true traits, and the mean test heritability of synthetic traits 4-5.

    python pipelines/scripts/simulation/evaluate.py --measurements s0.npz ... s9.npz \\
        --method pca p0.npz ... p9.npz [--method ...] --out evaluation.npz
"""

import argparse

import numpy as np
from h2opt import anova_heritability

from h2opt_analysis import simulation
from h2opt_analysis.methods import LABELS


def span_correlations(truth, Y):
    """Correlation of each true trait with its least-squares fit on the columns of Y."""
    design = np.column_stack([np.ones(len(Y)), Y])
    fitted = design @ np.linalg.lstsq(design, truth, rcond=None)[0]
    return np.array([np.corrcoef(fitted[:, t], truth[:, t])[0, 1] for t in range(truth.shape[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', nargs='+', required=True,
                        help='one file per instance (names and test are read)')
    parser.add_argument('--truth', nargs='+', required=True,
                        help='in the order of --measurements')
    parser.add_argument('--method', nargs='+', action='append', required=True,
                        metavar='SLUG_OR_FILE',
                        help='a method slug and its traits file per instance')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    n_inst, n_true = len(args.measurements), simulation.N_TRAITS
    if len(args.truth) != n_inst or any(len(m) != 1 + n_inst for m in args.method):
        parser.error('give one truth and one traits file per method for each measurements file')

    methods = [m[0] for m in args.method]
    cor = herit = span = None
    herit_truth = np.zeros((n_inst, 2, n_true))
    for i in range(n_inst):
        with np.load(args.measurements[i]) as f:
            names, test = f['names'], f['test']
        with np.load(args.truth[i]) as f:
            truth = f['traits']
        environment = np.zeros((len(names), 0))
        lines = np.unique(names[test])

        def line_means(v, names=names, lines=lines):
            return np.array([v[names == line].mean(axis=0) for line in lines])

        herit_truth[i] = [anova_heritability(truth[rows], names[rows], environment[rows])
                          for rows in (~test, test)]
        for m, method in enumerate(args.method):
            with np.load(method[1 + i]) as f:
                Y = f['traits']
            if cor is None:
                k = Y.shape[1]
                cor = np.zeros((len(methods), n_inst, n_true, k))
                herit = np.zeros((len(methods), n_inst, 2, k))
                span = np.zeros((len(methods), n_inst, n_true))
                cor_lines = np.zeros((len(methods), n_inst, n_true, k))
            cor[m, i] = simulation.correlations(truth, Y)
            cor_lines[m, i] = simulation.correlations(line_means(truth), line_means(Y))
            span[m, i] = span_correlations(truth, Y[:, :n_true])
            herit[m, i] = [anova_heritability(Y[rows], names[rows], environment[rows])
                           for rows in (~test, test)]

    np.savez(args.out, methods=methods, cor=cor, cor_test_lines=cor_lines, span_cor=span,
             herit=herit, herit_truth=herit_truth)

    diagonal = np.abs(cor[:, :, np.arange(n_true), np.arange(n_true)])
    best = np.abs(cor[:, :, :, :n_true]).max(axis=3)
    test_herit = np.clip(herit[:, :, 1], 0, None)
    deviation = np.abs(test_herit[:, :, :n_true] - np.clip(herit_truth[None, :, 1], 0, None))
    print(f'true test heritability (mean over instances): {herit_truth[:, 1].mean(0).round(3)}')
    print(f'{"method":16s} {"median |r| t-t":>22s} {"median best |r|":>22s} '
          f'{"median span r":>22s} {"MAD h2 1-3":>11s} {"h2 4-5":>7s}')
    for m, method in enumerate(methods):
        print(f'{LABELS[method]:16s} {np.median(diagonal[m], 0).round(3)!s:>22s} '
              f'{np.median(best[m], 0).round(3)!s:>22s} {np.median(span[m], 0).round(3)!s:>22s} '
              f'{deviation[m].mean():11.4f} {test_herit[m, :, n_true:].mean():7.3f}')
    print('the same on the test-line means (mean over the plants of each test line):')
    diagonal = np.abs(cor_lines[:, :, np.arange(n_true), np.arange(n_true)])
    best = np.abs(cor_lines[:, :, :, :n_true]).max(axis=3)
    print(f'{"method":16s} {"median |r| t-t":>22s} {"median best |r|":>22s} '
          f'{"median test h2 1-3":>22s}')
    for m, method in enumerate(methods):
        print(f'{LABELS[method]:16s} {np.median(diagonal[m], 0).round(3)!s:>22s} '
              f'{np.median(best[m], 0).round(3)!s:>22s} '
              f'{np.median(herit[m, :, 1, :n_true], 0).round(3)!s:>22s}')


if __name__ == '__main__':
    main()
