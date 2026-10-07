"""Fit one baseline on the training samples of an Arabidopsis accession fold; traits for all
samples of the treatment.

Baselines: pca, genetic_pca, pch (ridge tuned on the training samples) and max_features
(h2opt_analysis.methods.BASELINES). Traits are centered and uncorrelated on the training samples.
Writes an .npz with array 'traits' and, for pch, its ridge tuning
(h2opt_analysis.methods.fit_baseline).

    python pipelines/scripts/arabidopsis/baseline.py --features features.npy --treatment 6d \\
        --fold 0 --method pch --out fold0.npz --traits 10
"""

import argparse

import numpy as np

from h2opt_analysis import arabidopsis
from h2opt_analysis.methods import BASELINES, fit_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method', choices=list(BASELINES), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()

    X, accessions, environment = arabidopsis.load(args.features, args.treatment)
    is_train = arabidopsis.folds(accessions) != args.fold
    traits, tuning = fit_baseline(args.method, X, accessions, environment, is_train, args.traits)
    np.savez(args.out, traits=traits, **tuning)


if __name__ == '__main__':
    main()
