"""Fit one baseline on the training plants of a simulation instance; traits for all plants.

Baselines: pca, genetic_pca, pch and max_features (h2opt_analysis.methods.BASELINES). PCH's ridge
is chosen on the training plants by PCH.tune, the selection procedure and rule of H2-opt's noise
level (h2opt_analysis.methods.fit_baseline). Traits are centered and uncorrelated on the training
plants.
Writes an .npz with array 'traits' and, for pch, its ridge tuning
(h2opt_analysis.methods.fit_baseline).

    python pipelines/scripts/simulation/baseline.py --measurements spectra.npz --method pch \\
        --out pch.npz --traits 5
"""

import argparse

import numpy as np

from h2opt_analysis.methods import BASELINES, fit_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measurements', required=True,
                        help='.npz with X (plants first; images are flattened), names, test')
    parser.add_argument('--method', choices=list(BASELINES), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()

    with np.load(args.measurements) as f:
        X, names, test = f['X'], f['names'], f['test']
    X = X.reshape((len(X), -1))
    environment = np.zeros((len(names), 0))
    traits, tuning = fit_baseline(args.method, X, names, environment, ~test, args.traits)
    np.savez(args.out, traits=traits, **tuning)


if __name__ == '__main__':
    main()
