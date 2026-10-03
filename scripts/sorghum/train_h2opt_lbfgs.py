"""Train linear H2-opt to convergence with L-BFGS on sorghum for one of the paper's 10 genotype folds.

The ridge (grid RIDGES) and the number of L-BFGS steps are chosen on inner validation genotypes of the training
folds by the validation heritability of trait 1; the chosen setting is used for all traits.
Writes results/sorghum/h2opt_lbfgs/fold{k}.npz (traits of all plants, chosen ridge and steps, validation curves).

    uv run python scripts/sorghum/train_h2opt_lbfgs.py --fold 0 [--traits 5] [--steps 100]
"""

import argparse
from pathlib import Path

import numpy as np
from h2opt import LinearH2opt

from h2opt_analysis import sorghum
from h2opt_analysis.evaluate import RIDGES, validation_rows

RESULTS = Path('results/sorghum/h2opt_lbfgs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--steps', type=int, default=100, help='maximum L-BFGS steps')
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    fit, val = validation_rows(genotypes, is_train)

    curves = np.zeros((len(RIDGES), args.steps))
    for i, ridge in enumerate(RIDGES):
        model = LinearH2opt(1, ridge=ridge, n_iter=args.steps, seed=args.fold)
        model.fit(X[fit], genotypes[fit], environment[fit], validation=(X[val], genotypes[val], environment[val]))
        curves[i] = model.validation_curves_[0]
        print(f'ridge {ridge:g}: best validation heritability {curves[i].max():.3f} at step {curves[i].argmax() + 1}', flush=True)
    best, step = np.unravel_index(np.argmax(curves), curves.shape)
    ridge, steps = RIDGES[best], step + 1

    model = LinearH2opt(args.traits, ridge=ridge, n_iter=steps, seed=10 * args.fold)
    traits = model.fit(X[is_train], genotypes[is_train], environment[is_train]).transform(X)
    print(f'fold {args.fold}: ridge {ridge:g}, {steps} steps', flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / f'fold{args.fold}.npz', traits=traits, ridge=ridge, steps=steps, ridges=RIDGES, curves=curves)


if __name__ == '__main__':
    main()
