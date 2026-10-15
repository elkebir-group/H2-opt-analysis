"""Wavelength coefficients of PCH and of linear H2-opt on sorghum, per genotype fold.

PCH is refitted on the training plants of each fold at the ridge saved by baseline.py, by the same
code (h2opt.baselines.PCH). Its traits are (X - mean) @ coefficients, with X the raw spectra of
h2opt_analysis.sorghum.load. The script checks that the refitted traits reproduce the saved PCH
traits on all plants.

H2-opt's coefficients are the weights of the fold model times the map of h2opt.Decorrelation fitted
on the training outputs, as the saved traits are (h2opt.synthetic_traits). H2-opt trains on the
same raw spectra, so both sets of coefficients are on the raw-wavelength scale. The script checks
that they reproduce the saved H2-opt traits (correlation at least 1 - 1e-5 per trait; the
saved traits are float32 model outputs).

The log gives the smoothness of each trait's coefficients theta: the total variation
sum_i |theta[i+1] - theta[i]| divided by sum_i |theta[i]| (scale-free), per fold and the median
over the folds.

    python pipelines/scripts/sorghum/pch_coefficients.py --folds 0 1 \\
        --pch pch/fold0.npz pch/fold1.npz --h2opt h2opt/fold0.pt h2opt/fold1.pt \\
        --h2opt-traits h2opt/fold0.npz h2opt/fold1.npz \\
        --out pch_coefficients/fold0.npz pch_coefficients/fold1.npz
"""

import argparse

import numpy as np
import torch
from h2opt import Decorrelation
from h2opt.baselines import PCH

from h2opt_analysis import sorghum


def total_variation_ratio(coefficients):
    """sum_i |theta[i+1] - theta[i]| / sum_i |theta[i]| of each column (trait) of coefficients."""
    return np.abs(np.diff(coefficients, axis=0)).sum(axis=0) / np.abs(coefficients).sum(axis=0)


def h2opt_coefficients(model_path, X, is_train):
    """(wavelengths, traits) coefficients of the traits of a linear TraitModels, and the traits."""
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    weights = np.stack([m.lin1.weight.detach().numpy()[0] for m in model.models], axis=1)
    biases = np.array([m.lin1.bias.item() for m in model.models])
    raw = X @ weights.astype(float) + biases
    decorrelation = Decorrelation().fit(raw[is_train])
    return weights @ decorrelation.components_, decorrelation.transform(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--pch', nargs='+', required=True, help='baseline.py output per fold')
    parser.add_argument('--h2opt', nargs='+', required=True, help='H2-opt model per fold (.pt)')
    parser.add_argument('--h2opt-traits', nargs='+', required=True,
                        help='the traits saved with each H2-opt model (.npz)')
    parser.add_argument('--out', nargs='+', required=True, help='one file per fold')
    args = parser.parse_args()
    if not len(args.folds) == len(args.pch) == len(args.h2opt) == len(args.h2opt_traits) \
            == len(args.out):
        parser.error('give one file of each kind per fold')

    X, genotypes, environment = sorghum.load()
    X = np.asarray(X, dtype=float)
    fold_of = sorghum.folds(genotypes)
    smoothness = {'PCH': [], 'H2-opt': []}
    for fold, pch_path, model_path, traits_path, out in zip(
            args.folds, args.pch, args.h2opt, args.h2opt_traits, args.out, strict=True):
        is_train = fold_of != fold
        with np.load(pch_path) as f:
            saved, ridge = f['traits'], float(f['ridge'])
        pch = PCH(saved.shape[1], ridge=ridge).fit(X[is_train], genotypes[is_train],
                                                   environment[is_train])
        refit = pch.transform(X)
        error = np.abs(refit - saved).max(axis=0)
        r = np.array([abs(np.corrcoef(a, b)[0, 1]) for a, b in zip(refit.T, saved.T, strict=True)])
        print(f'fold {fold}, ridge {ridge:g}: PCH refit vs saved traits, largest |difference| per '
              f'trait {np.array2string(error, precision=1)}, largest |trait| '
              f'{np.abs(saved).max():.3g}, min |r| {r.min():.10f}')
        if error.max() > 1e-4 * np.abs(saved).max():
            raise SystemExit('the refitted PCH does not reproduce the saved traits')

        coefficients, traits = h2opt_coefficients(model_path, X, is_train)
        with np.load(traits_path) as f:
            saved_h2opt = f['traits']
        # the saved traits are float32 model outputs: compare by correlation, not by value
        h2opt_r = np.array([abs(np.corrcoef(a, b)[0, 1])
                            for a, b in zip(traits.T, saved_h2opt.T, strict=True)])
        print(f'fold {fold}: H2-opt coefficients vs saved traits, largest |difference| '
              f'{np.abs(traits - saved_h2opt).max():.2e}, largest |trait| '
              f'{np.abs(saved_h2opt).max():.3g}, min |r| {h2opt_r.min():.10f}')
        if h2opt_r.min() < 1 - 1e-5:
            raise SystemExit('the H2-opt coefficients do not reproduce the saved traits')

        tv = {'PCH': total_variation_ratio(pch.components_),
              'H2-opt': total_variation_ratio(coefficients)}
        for method, values in tv.items():
            smoothness[method].append(values)
            print(f'fold {fold}: {method} total variation / sum |coefficient|, traits 1-'
                  f'{len(values)}: {np.array2string(values, precision=3, max_line_width=200)}')
        np.savez(out, coefficients=pch.components_.T, mean=pch.mean_, ridge=ridge,
                 h2opt_coefficients=coefficients.T, pch_total_variation_ratio=tv['PCH'],
                 h2opt_total_variation_ratio=tv['H2-opt'])

    print('median over the folds of total variation / sum |coefficient|:')
    for method, values in smoothness.items():
        median = np.median(values, axis=0)
        print(f'  {method}: ' + ' '.join(f'{v:.3f}' for v in median))


if __name__ == '__main__':
    main()
