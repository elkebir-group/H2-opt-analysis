"""Train linear H2-opt to convergence with L-BFGS on one simulation instance.

One ridge for all traits is chosen on the training lines by LinearH2opt.tune (5-fold
cross-validation over lines, mean held-out heritability of the traits at convergence). Writes an
.npz with the traits of all plants (uncorrelated on the training plants), the chosen ridge, its
equivalent input-noise standard deviation (noise_sd), and the cross-validation scores of every
ridge.

    python pipelines/scripts/simulation/train_h2opt_lbfgs.py --spectra spectra.npz \\
        --out h2opt_lbfgs.npz --traits 5 --steps 200 --seed 0 --device cpu
"""

import argparse

import numpy as np
from h2opt import LinearH2opt, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--traits', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True,
                        help='maximum L-BFGS steps per trait')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True, help='torch device, e.g. cpu or cuda')
    args = parser.parse_args()

    with np.load(args.spectra) as f:
        X, names, test = f['X'], f['names'], f['test']
    is_train = ~test
    ridge, scores, fold_scores = LinearH2opt.tune(X[is_train], names[is_train], None,
                                                  n_traits=args.traits, n_iter=args.steps,
                                                  seed=args.seed, device=args.device)
    model = LinearH2opt(args.traits, ridge=ridge, n_iter=args.steps, seed=args.seed,
                        device=args.device)
    traits = model.fit(X[is_train], names[is_train]).transform(X)
    print(f'ridge {ridge:g} (noise sd {model.noise_sd_:.3g}), mean validation heritability '
          f'{scores.max():.3f}; per ridge {np.round(scores, 3)}; steps per trait '
          f'{model.n_steps_}')
    np.savez(args.out, traits=traits, ridge=ridge, noise_sd=model.noise_sd_,
             ridges=selection.RIDGES, scores=scores, fold_scores=fold_scores)


if __name__ == '__main__':
    main()
