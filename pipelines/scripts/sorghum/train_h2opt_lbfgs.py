"""Train linear H2-opt to convergence with L-BFGS on sorghum for one of the 10 genotype folds.

One ridge for all traits is chosen on the training folds by LinearH2opt.tune (5-fold
cross-validation over genotypes, mean held-out heritability of the traits at convergence). Writes an
.npz with the traits of all plants, the chosen ridge, its equivalent input-noise standard deviation
(noise_sd), and the cross-validation scores of every ridge.

    python pipelines/scripts/sorghum/train_h2opt_lbfgs.py --fold 0 --out fold0.npz \\
        [--traits 5] [--steps 200]
"""

import argparse

import numpy as np
from h2opt import LinearH2opt, selection

from h2opt_analysis import sorghum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, default=5)
    parser.add_argument('--steps', type=int, default=200, help='L-BFGS steps per trait')
    args = parser.parse_args()

    X, genotypes, environment = sorghum.load()
    is_train = sorghum.folds(genotypes) != args.fold
    ridge, scores, fold_scores = LinearH2opt.tune(X[is_train], genotypes[is_train],
                                                  environment[is_train], n_traits=args.traits,
                                                  n_iter=args.steps, seed=args.fold)

    model = LinearH2opt(args.traits, ridge=ridge, n_iter=args.steps, seed=10 * args.fold)
    model.fit(X[is_train], genotypes[is_train], environment[is_train])
    traits = model.transform(X)
    print(f'fold {args.fold}: ridge {ridge:g} (noise sd {model.noise_sd_:.3g}), mean validation '
          f'heritability {scores.max():.3f}; per ridge {np.round(scores, 3)}', flush=True)
    np.savez(args.out, traits=traits, ridge=ridge, noise_sd=model.noise_sd_,
             ridges=selection.RIDGES, scores=scores, fold_scores=fold_scores)


if __name__ == '__main__':
    main()
