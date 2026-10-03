"""Train linear H2-opt to convergence with L-BFGS on one simulation instance.

The ridge (grid RIDGES) and the number of L-BFGS steps are chosen on inner validation lines of the
training lines, by the validation heritability of trait 1. Writes an .npz with the traits of all
plants (uncorrelated on the training plants), the chosen ridge and steps, and the validation
curves.

    python pipelines/scripts/simulation/train_h2opt_lbfgs.py --spectra spectra.npz \\
        --out h2opt_lbfgs.npz [--traits 5] [--steps 100] [--seed 0]
"""

import argparse

import numpy as np
from h2opt import LinearH2opt

from h2opt_analysis.evaluate import RIDGES, validation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--steps', type=int, default=100, help='maximum L-BFGS steps')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    is_train = ~test
    fit, val = validation_rows(names, is_train)

    ridge, steps, curves = LinearH2opt.tune(X[fit], names[fit], None, (X[val], names[val], None),
                                            RIDGES, n_iter=args.steps, seed=args.seed)
    model = LinearH2opt(args.traits, ridge=ridge, n_iter=steps, seed=args.seed)
    traits = model.fit(X[is_train], names[is_train]).transform(X)
    print(f'ridge {ridge:g}, {steps} steps, validation heritability {curves.max():.3f}')
    np.savez(args.out, traits=traits, ridge=ridge, steps=steps, ridges=RIDGES, curves=curves)


if __name__ == '__main__':
    main()
