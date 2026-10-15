"""Coefficients of the PCH traits of one Arabidopsis treatment, per accession fold.

PCH is refitted on the training samples of each fold at the ridge saved by baseline.py, by the
same code (h2opt.baselines.PCH). The sign of each trait is arbitrary (an eigenvector), so each
refitted trait takes the sign of the saved one; the script then checks that the refitted traits
reproduce the saved PCH traits on all samples. Each trait's coefficients are divided by their
largest magnitude and saved signed, as (traits, MS level, m/z, retention time), as coefficients.py
saves H2-opt's. The log
gives, per fold and trait, the number of features that carry 90% of the total magnitude
(arabidopsis.share_features).

    python pipelines/scripts/arabidopsis/pch_coefficients.py --features features.npy \\
        --treatment 0d --folds 0 1 --pch pch/fold0.npz pch/fold1.npz \\
        --out pch_coefficients/fold0.npz pch_coefficients/fold1.npz
"""

import argparse

import numpy as np
from h2opt.baselines import PCH

from h2opt_analysis import arabidopsis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--treatment', choices=arabidopsis.TREATMENTS, required=True)
    parser.add_argument('--folds', type=int, nargs='+', required=True)
    parser.add_argument('--pch', nargs='+', required=True, help='baseline.py output per fold')
    parser.add_argument('--out', nargs='+', required=True, help='one file per fold')
    args = parser.parse_args()
    if not len(args.folds) == len(args.pch) == len(args.out):
        parser.error('give one file of each kind per fold')

    X, accessions, environment = arabidopsis.load(args.features, args.treatment)
    X = np.asarray(X, dtype=float)
    fold_of = arabidopsis.folds(accessions)
    for fold, pch_path, out in zip(args.folds, args.pch, args.out, strict=True):
        is_train = fold_of != fold
        with np.load(pch_path) as f:
            saved, ridge = f['traits'], float(f['ridge'])
        pch = PCH(saved.shape[1], ridge=ridge).fit(X[is_train], accessions[is_train],
                                                   environment[is_train])
        refit = pch.transform(X)
        signs = np.sign((refit * saved).sum(axis=0))
        components = pch.components_ * signs
        error = np.abs(refit * signs - saved).max()
        print(f'fold {fold}, ridge {ridge:g}: PCH refit vs saved traits (signs flipped: '
              f'{(signs < 0).sum()}), largest |difference| '
              f'{error:.2e}, largest |trait| {np.abs(saved).max():.3g}')
        if error > 1e-4 * np.abs(saved).max():
            raise SystemExit('the refitted PCH does not reproduce the saved traits')
        coefficients = (components / np.abs(components).max(axis=0)).T
        coefficients = coefficients.reshape((-1, 2, arabidopsis.N_MZ, arabidopsis.N_RT))
        np.savez(out, coefficients=coefficients.astype(np.float32), ridge=ridge)
        print(f'fold {fold}: features with 90% of the total |coefficient|, traits 1-'
              f'{len(coefficients)}: {arabidopsis.share_features(coefficients)}')


if __name__ == '__main__':
    main()
