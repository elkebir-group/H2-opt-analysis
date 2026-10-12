"""Detection of causal SNPs from the simulation GWAS, in 100 kb genomic bins (paper Fig. 3d-e).

For trait t, method m and instance i, each 100 kb bin of each chromosome (up to the last tested
SNP) gets the highest -log10 p of its SNPs in the GWAS of synthetic trait t (0 if it has none),
and is a true bin if it holds a QTN of true trait t. The AUC of these scores is computed per
trait, method and instance, and the ROC curves are averaged over the instances on a common
false-positive-rate grid. Writes an .npz (methods, auc: traits x methods x instances, fpr,
tpr: traits x methods x grid, the mean curves, and tpr_instances: traits x methods x instances x
grid, the curve of each instance).

    python pipelines/scripts/simulation/roc.py --instances 0 ... 9 \\
        --method pca g0.npz ... g9.npz [--method ...] --out roc.npz [--bin 100000]
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from h2opt_analysis import simulation
from h2opt_analysis.methods import LABELS


def binned(chromosome, position, score, qtns, bin_size):
    """Best score and QTN indicator of each bin, over all chromosomes."""
    scores, truth = [], []
    for c in np.unique(chromosome):
        on = chromosome == c
        bins = position[on] // bin_size
        best = np.zeros(bins.max() + 1)
        np.maximum.at(best, bins, score[on])
        is_true = np.zeros(len(best), bool)
        causal = qtns[qtns[:, 0] == c, 1] // bin_size
        is_true[causal[causal < len(best)]] = True
        scores.append(best)
        truth.append(is_true)
    return np.concatenate(scores), np.concatenate(truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=int, nargs='+', required=True)
    parser.add_argument('--method', nargs='+', action='append', required=True,
                        metavar='SLUG_OR_FILE',
                        help='a method slug and its GWAS file per instance, in instance order')
    parser.add_argument('--out', required=True)
    parser.add_argument('--bin', type=int, default=100000, help='bin size (bp)')
    args = parser.parse_args()
    n_t, n_i = simulation.N_TRAITS, len(args.instances)
    if any(len(m) != 1 + n_i for m in args.method):
        parser.error('give one GWAS file per instance for each method')

    methods = [m[0] for m in args.method]
    grid = np.linspace(0, 1, 101)
    auc = np.zeros((n_t, len(methods), n_i))
    tpr = np.zeros((n_t, len(methods), len(grid)))
    tpr_instances = np.full((n_t, len(methods), n_i, len(grid)), np.nan)
    for i, instance in enumerate(args.instances):
        qtns = simulation.qtns(instance)
        for m, method in enumerate(args.method):
            with np.load(method[1 + i]) as f:
                chromosome, position, score = f['chr'], f['ps'], f['neg_log10_p']
            auc[len(score):, m, i] = np.nan  # a file of fewer traits (cross-fitted trait 1)
            tpr[len(score):, m] = np.nan
            for t in range(min(n_t, len(score))):
                values, truth = binned(chromosome, position, score[t], qtns[t], args.bin)
                auc[t, m, i] = roc_auc_score(truth, values)
                fpr_t, tpr_t, _ = roc_curve(truth, values)
                tpr_instances[t, m, i] = np.interp(grid, fpr_t, tpr_t)
                tpr[t, m] += tpr_instances[t, m, i] / n_i

    np.savez(args.out, methods=methods, auc=auc, fpr=grid, tpr=tpr,
             tpr_instances=tpr_instances)
    print('median AUC over instances (traits 1-3)')
    for m, method in enumerate(methods):
        print(f'{LABELS.get(method, method):16s} {np.nanmedian(auc[:, m], axis=1).round(3)}')


if __name__ == '__main__':
    main()
