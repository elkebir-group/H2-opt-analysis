"""Coefficients of the linear H2-opt traits of one Arabidopsis treatment and fold, per feature.

The synthetic traits are the model outputs made uncorrelated on the training accessions
(h2opt.synthetic_traits), so the coefficient of trait j is a mix of the weights of models 1..j:
coefficients = W.T @ components_ of the h2opt.Decorrelation fitted on the training outputs. The
script checks that the features times these coefficients give the saved traits (up to a constant
per trait). Each trait's coefficients are divided by their largest magnitude and saved signed, as
(traits, MS level, m/z, retention time) in the feature order of h2opt_analysis.arabidopsis.

    python pipelines/scripts/arabidopsis/coefficients.py --features features.npy --treatment 0d \\
        --fold 0 --model h2opt/fold0.pt --traits h2opt/fold0.npz --out coefficients_fold0.npz
"""

import argparse

import numpy as np
import torch
from h2opt import Decorrelation

from h2opt_analysis import arabidopsis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model', required=True, help='TraitModels of LinearModel (.pt)')
    parser.add_argument('--traits', required=True, help='the traits saved with the model (.npz)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    X, accessions, _ = arabidopsis.load(args.features, args.treatment)
    train = arabidopsis.folds(accessions) != args.fold
    model = torch.load(args.model, weights_only=False, map_location='cpu')
    weight = np.stack([m.lin1.weight.detach().numpy()[0] for m in model.models]).astype(float)
    bias = np.array([m.lin1.bias.item() for m in model.models])
    decorrelation = Decorrelation().fit(X[train] @ weight.T + bias)
    coefficients = weight.T @ decorrelation.components_  # features x traits

    saved = np.load(args.traits)['traits']
    fitted = X @ coefficients
    error = np.abs((fitted - fitted.mean(axis=0)) - (saved - saved.mean(axis=0))).max()
    print(f'largest difference to the saved traits: {error:.2e} (largest trait value '
          f'{np.abs(saved).max():.2e})')
    if error > 1e-4 * np.abs(saved).max():
        raise SystemExit('the coefficients do not reproduce the saved traits')

    coefficients = (coefficients / np.abs(coefficients).max(axis=0)).T
    coefficients = coefficients.reshape((-1, 2, arabidopsis.N_MZ, arabidopsis.N_RT))
    np.savez(args.out, coefficients=coefficients.astype(np.float32))
    for t, c in enumerate(coefficients):
        print(f'trait {t + 1}: mean magnitude {np.abs(c).mean():.5f}, at least 2%: '
              f'{(np.abs(c[0]) >= 0.02).sum()} MS1, {(np.abs(c[1]) >= 0.02).sum()} MS/MS')


if __name__ == '__main__':
    main()
