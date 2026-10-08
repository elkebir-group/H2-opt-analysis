"""Fit one baseline on the training plants of a simulation instance; traits for all plants.

Baselines: pca, genetic_pca, lda, pch and max_features (h2opt_analysis.methods.BASELINES). PCH
uses the ridge that matches the input noise of linear H2-opt: ridge = level^2 with level the
level chosen by choose_noise.py (for a linear trait, input noise
of that standard deviation adds level^2 tr(B) / p to the phenotypic variance, as PCH's ridge
does). Traits are centered and uncorrelated on the training plants.
Writes an .npz with array 'traits' and, for pch, its ridge tuning
(h2opt_analysis.methods.fit_baseline).

    python pipelines/scripts/simulation/baseline.py --spectra spectra.npz --method pch \\
        --out pch.npz --traits 5 --noise noise.npz
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
    parser.add_argument('--noise', help='choose_noise.py output; sets the ridge of PCH')
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    environment = np.zeros((len(names), 0))
    ridge = None
    if args.method == 'pch':
        if args.noise is None:
            parser.error('pch needs --noise')
        with np.load(args.noise) as f:
            ridge = float(f['chosen']) ** 2
    traits, tuning = fit_baseline(args.method, X, names, environment, ~test, args.traits, ridge)
    np.savez(args.out, traits=traits, **tuning)


if __name__ == '__main__':
    main()
