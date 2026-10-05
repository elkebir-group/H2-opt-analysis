"""Fit one baseline on the training plants of a sorghum genotype fold; traits for all plants.

Baselines: pca, genetic_pca, lda, pch (ridge tuned on the training plants) and max_features
(h2opt_analysis.methods.BASELINES). Traits are centered and uncorrelated on the training plants.
Writes an .npz with array 'traits' and, for pch, its ridge tuning
(h2opt_analysis.methods.fit_baseline).

    python pipelines/scripts/sorghum/baseline.py --fold 0 --method pch --out fold0.npz --traits 10
"""

import argparse

import numpy as np

from h2opt_analysis import sorghum
from h2opt_analysis.methods import BASELINES, fit_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method', choices=list(BASELINES), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    traits, tuning = fit_baseline(args.method, X, genotypes, environment, is_train, args.traits)
    np.savez(args.out, traits=traits, **tuning)


if __name__ == '__main__':
    main()
