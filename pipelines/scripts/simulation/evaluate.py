"""Accuracy of each method's synthetic traits on the simulation instances.

For each method and instance: the Pearson correlation of each true trait with each synthetic
trait (all plants); the correlation of each true trait with its least-squares fit on the first
three synthetic traits; and the ANOVA heritability of each synthetic trait on the training and
test plants. Also the heritability of the true traits. Writes an .npz and prints a summary:
the paper's metric (|r| of true trait t with synthetic trait t), the best |r| over the first three
synthetic traits, the mean absolute deviation of the test heritability of synthetic traits 1-3
from the true traits, and the mean test heritability of synthetic traits 4-5.

    python pipelines/scripts/simulation/evaluate.py --spectra s0.npz ... s9.npz \\
        --method pca p0.npz ... p9.npz [--method ...] --out evaluation.npz
"""

import argparse

import numpy as np

from h2opt_analysis import simulation
from h2opt_analysis.evaluate import heritability


def correlations(truth, Y):
    """Pearson correlation of each column of truth with each column of Y, (k_truth, k_Y)."""
    a = (truth - truth.mean(0)) / truth.std(0)
    b = (Y - Y.mean(0)) / Y.std(0)
    return a.T @ b / len(a)


def span_correlations(truth, Y):
    """Correlation of each true trait with its least-squares fit on the columns of Y."""
    design = np.column_stack([np.ones(len(Y)), Y])
    fitted = design @ np.linalg.lstsq(design, truth, rcond=None)[0]
    return np.array([np.corrcoef(fitted[:, t], truth[:, t])[0, 1] for t in range(truth.shape[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', nargs=simulation.N_INSTANCES, required=True)
    parser.add_argument('--truth', nargs=simulation.N_INSTANCES, required=True)
    parser.add_argument('--method', nargs=1 + simulation.N_INSTANCES, action='append',
                        required=True, metavar='SLUG_OR_FILE',
                        help='a method slug and its traits file per instance')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    methods = [m[0] for m in args.method]
    n_inst, n_true = simulation.N_INSTANCES, simulation.N_TRAITS
    cor = herit = span = None
    herit_truth = np.zeros((n_inst, 2, n_true))
    for i in range(n_inst):
        with np.load(args.spectra[i]) as f:
            names, test = f['names'], f['test']
        with np.load(args.truth[i]) as f:
            truth = f['traits']
        environment = np.zeros((len(names), 0))
        herit_truth[i] = [heritability(truth, names, environment, rows) for rows in (~test, test)]
        for m, method in enumerate(args.method):
            with np.load(method[1 + i]) as f:
                Y = f['traits']
            if cor is None:
                k = Y.shape[1]
                cor = np.zeros((len(methods), n_inst, n_true, k))
                herit = np.zeros((len(methods), n_inst, 2, k))
                span = np.zeros((len(methods), n_inst, n_true))
            cor[m, i] = correlations(truth, Y)
            span[m, i] = span_correlations(truth, Y[:, :n_true])
            herit[m, i] = [heritability(Y, names, environment, rows) for rows in (~test, test)]

    np.savez(args.out, methods=methods, cor=cor, span_cor=span, herit=herit,
             herit_truth=herit_truth)

    diagonal = np.abs(cor[:, :, np.arange(n_true), np.arange(n_true)])
    best = np.abs(cor[:, :, :, :n_true]).max(axis=3)
    test_herit = np.clip(herit[:, :, 1], 0, None)
    deviation = np.abs(test_herit[:, :, :n_true] - np.clip(herit_truth[None, :, 1], 0, None))
    print(f'true test heritability (mean over instances): {herit_truth[:, 1].mean(0).round(3)}')
    print(f'{"method":16s} {"median |r| t-t":>22s} {"median best |r|":>22s} '
          f'{"median span r":>22s} {"MAD h2 1-3":>11s} {"h2 4-5":>7s}')
    for m, method in enumerate(methods):
        print(f'{simulation.METHODS[method]:16s} {np.median(diagonal[m], 0).round(3)!s:>22s} '
              f'{np.median(best[m], 0).round(3)!s:>22s} {np.median(span[m], 0).round(3)!s:>22s} '
              f'{deviation[m].mean():11.4f} {test_herit[m, :, n_true:].mean():7.3f}')


if __name__ == '__main__':
    main()
