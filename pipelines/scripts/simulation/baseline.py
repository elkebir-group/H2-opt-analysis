"""Fit one baseline on the training plants of a simulation instance; traits for all plants.

Baselines: pca, genetic_pca, lda, pch (ridge tuned on inner validation lines) and max_features
(h2opt_analysis.methods.BASELINES). Traits are centered and uncorrelated on the training plants.
Writes an .npz with array 'traits'.

    python pipelines/scripts/simulation/baseline.py --spectra spectra.npz --method pch \\
        --out pch.npz --traits 5
"""

import argparse

import numpy as np

from h2opt_analysis.methods import BASELINES, fit_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--method', choices=list(BASELINES), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    environment = np.zeros((len(names), 0))
    traits = fit_baseline(args.method, X, names, environment, ~test, args.traits)
    np.savez(args.out, traits=traits)


if __name__ == '__main__':
    main()
